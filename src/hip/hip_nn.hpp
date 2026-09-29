#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

namespace laya::hip_nn {

void linear(hipblasHandle_t handle, const float* d_x, size_t rows, size_t in_dim, const float* d_weight,
           const float* d_bias, size_t out_dim, float* d_out);

void layer_norm(const float* d_x, size_t rows, size_t n, const float* d_weight, const float* d_bias, float eps,
                float* d_out);

void gelu_mul(const float* d_proj, size_t rows, size_t inter, float* d_out);

void relu_inplace(float* d_x, size_t n);

void add_inplace(float* d_a, const float* d_b, size_t n);

void add_broadcast_row(float* d_x, size_t rows, size_t hidden, const float* d_row);

void embedding_lookup(const float* d_table, const int32_t* d_ids, size_t seq_len, size_t hidden, float* d_out);

void rope_apply_inplace(float* d_qkv, size_t seq_len, size_t hidden, size_t num_heads, size_t half_dim,
                        const float* d_cos, const float* d_sin);

void attention(const float* d_qkv, size_t seq_len, size_t hidden, size_t num_heads, bool sliding, size_t window,
              float* d_context);

}
