#include "laya/nn.hpp"

#include <cmath>

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

}
