#include "laya/nn.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace laya::nn {

void layer_norm_row(const float* x, size_t n, const float* weight, const float* bias, float eps, float* out) {
    float mean = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        mean += x[i];
    }
    mean /= static_cast<float>(n);

    float var = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float d = x[i] - mean;
        var += d * d;
    }
    var /= static_cast<float>(n);

    const float inv_std = 1.0f / std::sqrt(var + eps);
    for (size_t i = 0; i < n; ++i) {
        const float normed = (x[i] - mean) * inv_std;
        out[i] = bias != nullptr ? normed * weight[i] + bias[i] : normed * weight[i];
    }
}

void linear(const float* x, size_t rows, size_t in_dim, const float* weight, const float* bias, size_t out_dim,
           float* out) {
    for (size_t r = 0; r < rows; ++r) {
        const float* xr = x + r * in_dim;
        float* outr = out + r * out_dim;
        for (size_t o = 0; o < out_dim; ++o) {
            const float* w = weight + o * in_dim;
            float acc = bias != nullptr ? bias[o] : 0.0f;
            for (size_t i = 0; i < in_dim; ++i) {
                acc += xr[i] * w[i];
            }
            outr[o] = acc;
        }
    }
}

float gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
}

float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

void multi_head_attention(const float* x, size_t seq_len, size_t hidden, size_t num_heads,
                          const float* in_proj_weight, const float* in_proj_bias, const float* out_proj_weight,
                          const float* out_proj_bias, float* out) {
    const size_t head_dim = hidden / num_heads;
    std::vector<float> qkv(seq_len * 3 * hidden);
    linear(x, seq_len, hidden, in_proj_weight, in_proj_bias, 3 * hidden, qkv.data());

    const float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> context(seq_len * hidden);
    std::vector<float> scores(seq_len);

    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < seq_len; ++i) {
            const float* qi = qkv.data() + i * 3 * hidden + h * head_dim;
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < seq_len; ++j) {
                const float* kj = qkv.data() + j * 3 * hidden + hidden + h * head_dim;
                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    dot += qi[d] * kj[d];
                }
                dot *= scaling;
                scores[j] = dot;
                if (dot > max_score) {
                    max_score = dot;
                }
            }
            float sum = 0.0f;
            for (size_t j = 0; j < seq_len; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                sum += scores[j];
            }
            float* out_row = context.data() + i * hidden + h * head_dim;
            for (size_t d = 0; d < head_dim; ++d) {
                out_row[d] = 0.0f;
            }
            for (size_t j = 0; j < seq_len; ++j) {
                const float w = scores[j] / sum;
                const float* vj = qkv.data() + j * 3 * hidden + 2 * hidden + h * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    out_row[d] += w * vj[d];
                }
            }
        }
    }

    linear(context.data(), seq_len, hidden, out_proj_weight, out_proj_bias, hidden, out);
}

}
