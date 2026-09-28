#pragma once

#include <cstddef>

namespace laya::nn {

void layer_norm_row(const float* x, size_t n, const float* weight, const float* bias, float eps, float* out);
void linear(const float* x, size_t rows, size_t in_dim, const float* weight, const float* bias, size_t out_dim,
           float* out);
float gelu(float x);

}
