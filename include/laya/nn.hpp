#pragma once

#include <cstddef>

namespace laya::nn {

void layer_norm_row(const float* __restrict x, size_t n, const float* __restrict weight,
                    const float* __restrict bias, float eps, float* __restrict out);
void linear(const float* __restrict x, size_t rows, size_t in_dim, const float* __restrict weight,
           const float* __restrict bias, size_t out_dim, float* __restrict out);
float gelu(float x);
float relu(float x);

void multi_head_attention(const float* __restrict x, size_t seq_len, size_t hidden, size_t num_heads,
                          const float* __restrict in_proj_weight, const float* __restrict in_proj_bias,
                          const float* __restrict out_proj_weight, const float* __restrict out_proj_bias,
                          float* __restrict out);

}
