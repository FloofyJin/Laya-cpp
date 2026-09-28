#pragma once

#include <cstddef>

namespace laya::nn {

void layer_norm_row(const float* x, size_t n, const float* weight, const float* bias, float eps, float* out);
void linear(const float* x, size_t rows, size_t in_dim, const float* weight, const float* bias, size_t out_dim,
           float* out);
float gelu(float x);
float relu(float x);

void multi_head_attention(const float* x, size_t seq_len, size_t hidden, size_t num_heads,
                          const float* in_proj_weight, const float* in_proj_bias, const float* out_proj_weight,
                          const float* out_proj_bias, float* out);

}
