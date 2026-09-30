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

struct HipScratchBuf {
    float* ptr = nullptr;
    size_t capacity = 0;
};

struct HipScratchIdBuf {
    int32_t* ptr = nullptr;
    size_t capacity = 0;
};

struct HipScratchRope {
    float* cos = nullptr;
    float* sin = nullptr;
    size_t capacity = 0;
};

struct HipEncoderScratch {
    HipEncoderScratch() = default;
    ~HipEncoderScratch();
    HipEncoderScratch(const HipEncoderScratch&) = delete;
    HipEncoderScratch& operator=(const HipEncoderScratch&) = delete;

    HipScratchIdBuf ids;
    HipScratchBuf embed, hidden, out;
    HipScratchRope rope_full, rope_local;
    HipScratchBuf attn_normed, attn_out, mlp_normed, mlp_out;
    HipScratchBuf qkv, context;
    HipScratchBuf proj, gated;
    HipScratchBuf head_normed1, head_attn_out, head_normed2, head_ff1, head_ff2;
};

HipTensor hip_modernbert_encoder(hipblasHandle_t handle, const HipWeights& weights, const ModernBertConfig& cfg,
                                 const std::vector<int32_t>& input_ids, HipEncoderScratch& scratch);

HipTensor hip_add_type_emb_and_head(hipblasHandle_t handle, const HipWeights& weights, const ModernBertConfig& cfg,
                                    HipTensor encoder_out, int32_t qtype, HipEncoderScratch& scratch);

std::vector<float> hip_download(const HipTensor& t);

}
