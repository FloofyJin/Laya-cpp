#include "hip_nn.hpp"

#include <cfloat>

namespace laya::hip_nn {

namespace {

__global__ void broadcast_bias_kernel(const float* __restrict__ bias, int rows, int out_dim,
                                      float* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * out_dim;
    if (idx < total) {
        out[idx] = bias[idx % out_dim];
    }
}

__global__ void layer_norm_kernel(const float* __restrict__ x, int n, const float* __restrict__ weight,
                                  const float* __restrict__ bias, float eps, float* __restrict__ out) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const float* xr = x + static_cast<size_t>(row) * n;
    float* outr = out + static_cast<size_t>(row) * n;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    float sum = 0.0f;
    for (int i = tid; i < n; i += block_size) {
        sum += xr[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    const float mean = sdata[0] / static_cast<float>(n);
    __syncthreads();

    float var_sum = 0.0f;
    for (int i = tid; i < n; i += block_size) {
        const float d = xr[i] - mean;
        var_sum += d * d;
    }
    sdata[tid] = var_sum;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    const float var = sdata[0] / static_cast<float>(n);
    const float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < n; i += block_size) {
        const float normed = (xr[i] - mean) * inv_std;
        outr[i] = bias != nullptr ? normed * weight[i] + bias[i] : normed * weight[i];
    }
}

__global__ void gelu_mul_kernel(const float* __restrict__ proj, int inter, float* __restrict__ out) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const float* p = proj + static_cast<size_t>(row) * 2 * inter;
    float* o = out + static_cast<size_t>(row) * inter;
    for (int c = tid; c < inter; c += block_size) {
        const float a = p[c];
        const float g = p[inter + c];
        const float gelu_a = 0.5f * a * (1.0f + erff(a * 0.70710678118654752440f));
        o[c] = gelu_a * g;
    }
}

__global__ void relu_kernel(float* __restrict__ x, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
    }
}

__global__ void add_kernel(float* __restrict__ a, const float* __restrict__ b, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += b[idx];
    }
}

__global__ void add_broadcast_row_kernel(float* __restrict__ x, int hidden, const float* __restrict__ row) {
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    float* xr = x + static_cast<size_t>(r) * hidden;
    for (int i = tid; i < hidden; i += block_size) {
        xr[i] += row[i];
    }
}

__global__ void embedding_lookup_kernel(const float* __restrict__ table, const int32_t* __restrict__ ids,
                                        int hidden, float* __restrict__ out) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const float* src = table + static_cast<size_t>(ids[row]) * hidden;
    float* dst = out + static_cast<size_t>(row) * hidden;
    for (int i = tid; i < hidden; i += block_size) {
        dst[i] = src[i];
    }
}

__global__ void rope_apply_kernel(float* __restrict__ qkv, int hidden, int half_dim,
                                  const float* __restrict__ cos_t, const float* __restrict__ sin_t) {
    const int p = blockIdx.x;
    const int h = blockIdx.y;
    const int j = threadIdx.x;
    if (j >= half_dim) {
        return;
    }
    const int head_dim = 2 * half_dim;
    float* row = qkv + static_cast<size_t>(p) * 3 * hidden;
    float* q = row + h * head_dim;
    float* k = row + hidden + h * head_dim;
    const float c = cos_t[p * half_dim + j];
    const float s = sin_t[p * half_dim + j];

    const float q0 = q[j];
    const float q1 = q[half_dim + j];
    q[j] = q0 * c - q1 * s;
    q[half_dim + j] = q1 * c + q0 * s;

    const float k0 = k[j];
    const float k1 = k[half_dim + j];
    k[j] = k0 * c - k1 * s;
    k[half_dim + j] = k1 * c + k0 * s;
}

__global__ void attention_kernel(const float* __restrict__ qkv, int seq_len, int hidden, int head_dim, bool sliding,
                                 int window, float scaling, float* __restrict__ context) {
    extern __shared__ float shared[];
    float* scores = shared;
    float* reduce_buf = shared + seq_len;

    const int h = blockIdx.y;
    const int i = blockIdx.x;
    const int d = threadIdx.x;

    const float* qi = qkv + static_cast<size_t>(i) * 3 * hidden + h * head_dim;

    int jlo = 0;
    int jhi = seq_len;
    if (sliding) {
        jlo = i >= window ? i - window : 0;
        jhi = i + window + 1 < seq_len ? i + window + 1 : seq_len;
    }

    for (int j = jlo; j < jhi; ++j) {
        const float* kj = qkv + static_cast<size_t>(j) * 3 * hidden + hidden + h * head_dim;
        reduce_buf[d] = qi[d] * kj[d];
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (d < s) {
                reduce_buf[d] += reduce_buf[d + s];
            }
            __syncthreads();
        }
        if (d == 0) {
            scores[j] = reduce_buf[0] * scaling;
        }
        __syncthreads();
    }

    __shared__ float sum_val;
    if (d == 0) {
        float m = -FLT_MAX;
        for (int j = jlo; j < jhi; ++j) {
            m = fmaxf(m, scores[j]);
        }
        float s = 0.0f;
        for (int j = jlo; j < jhi; ++j) {
            const float e = expf(scores[j] - m);
            scores[j] = e;
            s += e;
        }
        sum_val = s;
    }
    __syncthreads();

    float acc = 0.0f;
    for (int j = jlo; j < jhi; ++j) {
        const float* vj = qkv + static_cast<size_t>(j) * 3 * hidden + 2 * hidden + h * head_dim;
        acc += (scores[j] / sum_val) * vj[d];
    }
    context[static_cast<size_t>(i) * hidden + h * head_dim + d] = acc;
}

}

void linear(hipblasHandle_t handle, const float* d_x, size_t rows, size_t in_dim, const float* d_weight,
           const float* d_bias, size_t out_dim, float* d_out) {
    const size_t total = rows * out_dim;
    if (d_bias != nullptr) {
        const int threads = 256;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        broadcast_bias_kernel<<<blocks, threads>>>(d_bias, static_cast<int>(rows), static_cast<int>(out_dim), d_out);
    } else {
        (void)hipMemset(d_out, 0, total * sizeof(float));
    }
    const float alpha = 1.0f;
    const float beta = 1.0f;
    hipblasSgemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(out_dim), static_cast<int>(rows),
                static_cast<int>(in_dim), &alpha, d_weight, static_cast<int>(in_dim), d_x, static_cast<int>(in_dim),
                &beta, d_out, static_cast<int>(out_dim));
}

void layer_norm(const float* d_x, size_t rows, size_t n, const float* d_weight, const float* d_bias, float eps,
                float* d_out) {
    const int threads = 256;
    layer_norm_kernel<<<static_cast<unsigned>(rows), threads, threads * sizeof(float)>>>(
        d_x, static_cast<int>(n), d_weight, d_bias, eps, d_out);
}

void gelu_mul(const float* d_proj, size_t rows, size_t inter, float* d_out) {
    const int threads = 256;
    gelu_mul_kernel<<<static_cast<unsigned>(rows), threads>>>(d_proj, static_cast<int>(inter), d_out);
}

void relu_inplace(float* d_x, size_t n) {
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    relu_kernel<<<blocks, threads>>>(d_x, n);
}

void add_inplace(float* d_a, const float* d_b, size_t n) {
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    add_kernel<<<blocks, threads>>>(d_a, d_b, n);
}

void add_broadcast_row(float* d_x, size_t rows, size_t hidden, const float* d_row) {
    const int threads = 256;
    add_broadcast_row_kernel<<<static_cast<unsigned>(rows), threads>>>(d_x, static_cast<int>(hidden), d_row);
}

void embedding_lookup(const float* d_table, const int32_t* d_ids, size_t seq_len, size_t hidden, float* d_out) {
    const int threads = 256;
    embedding_lookup_kernel<<<static_cast<unsigned>(seq_len), threads>>>(d_table, d_ids, static_cast<int>(hidden),
                                                                        d_out);
}

void rope_apply_inplace(float* d_qkv, size_t seq_len, size_t hidden, size_t num_heads, size_t half_dim,
                        const float* d_cos, const float* d_sin) {
    const dim3 grid(static_cast<unsigned>(seq_len), static_cast<unsigned>(num_heads));
    rope_apply_kernel<<<grid, static_cast<unsigned>(half_dim)>>>(d_qkv, static_cast<int>(hidden),
                                                                 static_cast<int>(half_dim), d_cos, d_sin);
}

void attention(const float* d_qkv, size_t seq_len, size_t hidden, size_t num_heads, bool sliding, size_t window,
              float* d_context) {
    const size_t head_dim = hidden / num_heads;
    const float scaling = 1.0f / sqrtf(static_cast<float>(head_dim));
    const dim3 grid(static_cast<unsigned>(seq_len), static_cast<unsigned>(num_heads));
    const size_t shared_bytes = (seq_len + head_dim) * sizeof(float);
    attention_kernel<<<grid, static_cast<unsigned>(head_dim), shared_bytes>>>(
        d_qkv, static_cast<int>(seq_len), static_cast<int>(hidden), static_cast<int>(head_dim), sliding,
        static_cast<int>(window), scaling, d_context);
}

}
