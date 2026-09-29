#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <hipblas/hipblas.h>

#include "laya/modernbert.hpp"
#include "hip_weights.hpp"

namespace laya {

struct HipTensor {
    float* data = nullptr;
    size_t seq_len = 0;
    size_t hidden = 0;
};

HipTensor hip_modernbert_encoder(hipblasHandle_t handle, const HipWeights& weights, const ModernBertConfig& cfg,
                                 const std::vector<int32_t>& input_ids);

HipTensor hip_add_type_emb_and_head(hipblasHandle_t handle, const HipWeights& weights, const ModernBertConfig& cfg,
                                    HipTensor encoder_out, int32_t qtype);

void hip_free_tensor(HipTensor& t);

std::vector<float> hip_download(const HipTensor& t);

}
