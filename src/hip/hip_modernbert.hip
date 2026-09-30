#include "hip_modernbert.hpp"

#include <algorithm>
#include <stdexcept>

#include <hip/hip_runtime.h>

#include "hip_nn.hpp"

namespace laya {

namespace {

float* scratch_ensure(HipScratchBuf& buf, size_t needed) {
    if (needed > buf.capacity) {
        if (buf.ptr != nullptr) {
            (void)hipFree(buf.ptr);
            buf.ptr = nullptr;
        }
        if (hipMalloc(&buf.ptr, needed * sizeof(float)) != hipSuccess) {
            buf.capacity = 0;
            throw std::runtime_error("hip_modernbert: scratch hipMalloc failed");
        }
        buf.capacity = needed;
    }
    return buf.ptr;
}

int32_t* scratch_ensure_ids(HipScratchIdBuf& buf, size_t needed) {
    if (needed > buf.capacity) {
        if (buf.ptr != nullptr) {
            (void)hipFree(buf.ptr);
            buf.ptr = nullptr;
        }
        if (hipMalloc(&buf.ptr, needed * sizeof(int32_t)) != hipSuccess) {
            buf.capacity = 0;
            throw std::runtime_error("hip_modernbert: scratch hipMalloc failed for ids");
        }
        buf.capacity = needed;
    }
    return buf.ptr;
}

void scratch_ensure_rope(HipScratchRope& rope, size_t seq_len, size_t half_dim, double theta) {
    if (seq_len <= rope.capacity) {
        return;
    }
    const RopeTable table = modernbert_rope_table(seq_len, half_dim, theta);
    if (rope.cos != nullptr) {
        (void)hipFree(rope.cos);
        rope.cos = nullptr;
    }
    if (rope.sin != nullptr) {
        (void)hipFree(rope.sin);
        rope.sin = nullptr;
    }
    rope.capacity = 0;
    const size_t bytes = seq_len * half_dim * sizeof(float);
    if (hipMalloc(&rope.cos, bytes) != hipSuccess || hipMalloc(&rope.sin, bytes) != hipSuccess) {
        throw std::runtime_error("hip_modernbert: scratch hipMalloc failed for rope table");
    }
    if (hipMemcpy(rope.cos, table.cos.data(), bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(rope.sin, table.sin.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
        throw std::runtime_error("hip_modernbert: scratch hipMemcpy failed for rope table");
    }
    rope.capacity = seq_len;
}

void attention_block(hipblasHandle_t handle, const HipWeights& weights, const std::string& prefix,
                     const float* d_input, float* d_out, size_t seq_len, size_t hidden, size_t num_heads,
                     bool sliding, size_t window, const float* d_cos, const float* d_sin, bool has_bias,
                     HipEncoderScratch& scratch) {
    const size_t head_dim = hidden / num_heads;
    const size_t half_dim = head_dim / 2;

    float* d_qkv = scratch_ensure(scratch.qkv, seq_len * 3 * hidden);
    const float* wqkv_b = has_bias ? weights.get(prefix + "in_proj_bias") : nullptr;
    const std::string wqkv_name = has_bias ? prefix + "in_proj_weight" : prefix + "Wqkv.weight";
    hip_nn::linear(handle, d_input, seq_len, hidden, weights.get(wqkv_name), wqkv_b, 3 * hidden, d_qkv);

    if (d_cos != nullptr) {
        hip_nn::rope_apply_inplace(d_qkv, seq_len, hidden, num_heads, half_dim, d_cos, d_sin);
    }

    float* d_context = scratch_ensure(scratch.context, seq_len * hidden);
    hip_nn::attention(d_qkv, seq_len, hidden, num_heads, sliding, window, d_context);

    const float* wo_b = has_bias ? weights.get(prefix + "out_proj.bias") : nullptr;
    const std::string wo_name = has_bias ? prefix + "out_proj.weight" : prefix + "Wo.weight";
    hip_nn::linear(handle, d_context, seq_len, hidden, weights.get(wo_name), wo_b, hidden, d_out);
}

void mlp_block(hipblasHandle_t handle, const HipWeights& weights, const std::string& wi_name,
              const std::string& wo_name, const float* d_input, float* d_out, size_t seq_len, size_t hidden,
              size_t inter, HipEncoderScratch& scratch) {
    float* d_proj = scratch_ensure(scratch.proj, seq_len * 2 * inter);
    hip_nn::linear(handle, d_input, seq_len, hidden, weights.get(wi_name), nullptr, 2 * inter, d_proj);

    float* d_gated = scratch_ensure(scratch.gated, seq_len * inter);
    hip_nn::gelu_mul(d_proj, seq_len, inter, d_gated);

    hip_nn::linear(handle, d_gated, seq_len, inter, weights.get(wo_name), nullptr, hidden, d_out);
}

}

HipEncoderScratch::~HipEncoderScratch() {
    if (ids.ptr != nullptr) (void)hipFree(ids.ptr);
    if (embed.ptr != nullptr) (void)hipFree(embed.ptr);
    if (hidden.ptr != nullptr) (void)hipFree(hidden.ptr);
    if (out.ptr != nullptr) (void)hipFree(out.ptr);
    if (rope_full.cos != nullptr) (void)hipFree(rope_full.cos);
    if (rope_full.sin != nullptr) (void)hipFree(rope_full.sin);
    if (rope_local.cos != nullptr) (void)hipFree(rope_local.cos);
    if (rope_local.sin != nullptr) (void)hipFree(rope_local.sin);
    if (attn_normed.ptr != nullptr) (void)hipFree(attn_normed.ptr);
    if (attn_out.ptr != nullptr) (void)hipFree(attn_out.ptr);
    if (mlp_normed.ptr != nullptr) (void)hipFree(mlp_normed.ptr);
    if (mlp_out.ptr != nullptr) (void)hipFree(mlp_out.ptr);
    if (qkv.ptr != nullptr) (void)hipFree(qkv.ptr);
    if (context.ptr != nullptr) (void)hipFree(context.ptr);
    if (proj.ptr != nullptr) (void)hipFree(proj.ptr);
    if (gated.ptr != nullptr) (void)hipFree(gated.ptr);
    if (head_normed1.ptr != nullptr) (void)hipFree(head_normed1.ptr);
    if (head_attn_out.ptr != nullptr) (void)hipFree(head_attn_out.ptr);
    if (head_normed2.ptr != nullptr) (void)hipFree(head_normed2.ptr);
    if (head_ff1.ptr != nullptr) (void)hipFree(head_ff1.ptr);
    if (head_ff2.ptr != nullptr) (void)hipFree(head_ff2.ptr);
}

HipTensor hip_modernbert_encoder(hipblasHandle_t handle, const HipWeights& weights, const ModernBertConfig& cfg,
                                 const std::vector<int32_t>& input_ids, HipEncoderScratch& scratch) {
    const size_t seq_len = input_ids.size();
    const size_t hidden = static_cast<size_t>(cfg.hidden_size);
    const size_t num_heads = static_cast<size_t>(cfg.num_attention_heads);
    const size_t head_dim = hidden / num_heads;
    const size_t half_dim = head_dim / 2;
    const size_t window = static_cast<size_t>(cfg.local_attention) / 2;

    int32_t* d_ids = scratch_ensure_ids(scratch.ids, seq_len);
    if (hipMemcpy(d_ids, input_ids.data(), seq_len * sizeof(int32_t), hipMemcpyHostToDevice) != hipSuccess) {
        throw std::runtime_error("hip_modernbert: hipMemcpy failed for ids");
    }

    float* d_embed = scratch_ensure(scratch.embed, seq_len * hidden);
    hip_nn::embedding_lookup(weights.get("encoder.embeddings.tok_embeddings.weight"), d_ids, seq_len, hidden,
                             d_embed);

    float* d_hidden = scratch_ensure(scratch.hidden, seq_len * hidden);
    hip_nn::layer_norm(d_embed, seq_len, hidden, weights.get("encoder.embeddings.norm.weight"), nullptr, cfg.norm_eps,
                       d_hidden);

    scratch_ensure_rope(scratch.rope_full, seq_len, half_dim, cfg.global_rope_theta);
    scratch_ensure_rope(scratch.rope_local, seq_len, half_dim, cfg.local_rope_theta);
    const float* d_cos_full = scratch.rope_full.cos;
    const float* d_sin_full = scratch.rope_full.sin;
    const float* d_cos_local = scratch.rope_local.cos;
    const float* d_sin_local = scratch.rope_local.sin;

    for (int layer_idx = 0; layer_idx < cfg.num_hidden_layers; ++layer_idx) {
        const bool sliding = cfg.layer_types.at(static_cast<size_t>(layer_idx)) == "sliding_attention";
        const std::string prefix = "encoder.layers." + std::to_string(layer_idx) + ".";

        const float* d_attn_input = d_hidden;
        if (layer_idx != 0) {
            float* d_attn_normed = scratch_ensure(scratch.attn_normed, seq_len * hidden);
            hip_nn::layer_norm(d_hidden, seq_len, hidden, weights.get(prefix + "attn_norm.weight"), nullptr,
                               cfg.norm_eps, d_attn_normed);
            d_attn_input = d_attn_normed;
        }

        float* d_attn_out = scratch_ensure(scratch.attn_out, seq_len * hidden);
        attention_block(handle, weights, prefix + "attn.", d_attn_input, d_attn_out, seq_len, hidden, num_heads,
                        sliding, window, sliding ? d_cos_local : d_cos_full, sliding ? d_sin_local : d_sin_full,
                        false, scratch);

        hip_nn::add_inplace(d_hidden, d_attn_out, seq_len * hidden);

        float* d_mlp_normed = scratch_ensure(scratch.mlp_normed, seq_len * hidden);
        hip_nn::layer_norm(d_hidden, seq_len, hidden, weights.get(prefix + "mlp_norm.weight"), nullptr, cfg.norm_eps,
                           d_mlp_normed);

        float* d_mlp_out = scratch_ensure(scratch.mlp_out, seq_len * hidden);
        mlp_block(handle, weights, prefix + "mlp.Wi.weight", prefix + "mlp.Wo.weight", d_mlp_normed, d_mlp_out,
                 seq_len, hidden, static_cast<size_t>(cfg.intermediate_size), scratch);

        hip_nn::add_inplace(d_hidden, d_mlp_out, seq_len * hidden);
    }

    float* d_out = scratch_ensure(scratch.out, seq_len * hidden);
    hip_nn::layer_norm(d_hidden, seq_len, hidden, weights.get("encoder.final_norm.weight"), nullptr, cfg.norm_eps,
                       d_out);

    return HipTensor{d_out, seq_len, hidden};
}

HipTensor hip_add_type_emb_and_head(hipblasHandle_t handle, const HipWeights& weights, const ModernBertConfig& cfg,
                                    HipTensor t, int32_t qtype, HipEncoderScratch& scratch) {
    (void)cfg;
    const size_t hidden = t.hidden;
    const size_t seq_len = t.seq_len;
    const size_t num_heads = std::max<size_t>(1, hidden / 64);

    const float* d_type_row = weights.get("type_emb.weight") + static_cast<size_t>(qtype) * hidden;
    hip_nn::add_broadcast_row(t.data, seq_len, hidden, d_type_row);

    for (int layer = 0; layer < 2; ++layer) {
        const std::string prefix = "head.layers." + std::to_string(layer) + ".";

        float* d_normed1 = scratch_ensure(scratch.head_normed1, seq_len * hidden);
        hip_nn::layer_norm(t.data, seq_len, hidden, weights.get(prefix + "norm1.weight"),
                           weights.get(prefix + "norm1.bias"), 1e-5f, d_normed1);

        float* d_attn_out = scratch_ensure(scratch.head_attn_out, seq_len * hidden);
        attention_block(handle, weights, prefix + "self_attn.", d_normed1, d_attn_out, seq_len, hidden, num_heads,
                        false, 0, nullptr, nullptr, true, scratch);

        hip_nn::add_inplace(t.data, d_attn_out, seq_len * hidden);

        float* d_normed2 = scratch_ensure(scratch.head_normed2, seq_len * hidden);
        hip_nn::layer_norm(t.data, seq_len, hidden, weights.get(prefix + "norm2.weight"),
                           weights.get(prefix + "norm2.bias"), 1e-5f, d_normed2);

        const size_t ff_dim = 4 * hidden;
        float* d_ff1 = scratch_ensure(scratch.head_ff1, seq_len * ff_dim);
        hip_nn::linear(handle, d_normed2, seq_len, hidden, weights.get(prefix + "linear1.weight"),
                       weights.get(prefix + "linear1.bias"), ff_dim, d_ff1);
        hip_nn::relu_inplace(d_ff1, seq_len * ff_dim);

        float* d_ff2 = scratch_ensure(scratch.head_ff2, seq_len * hidden);
        hip_nn::linear(handle, d_ff1, seq_len, ff_dim, weights.get(prefix + "linear2.weight"),
                       weights.get(prefix + "linear2.bias"), hidden, d_ff2);

        hip_nn::add_inplace(t.data, d_ff2, seq_len * hidden);
    }

    return t;
}

std::vector<float> hip_download(const HipTensor& t) {
    std::vector<float> out(t.seq_len * t.hidden);
    if (hipMemcpy(out.data(), t.data, out.size() * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) {
        throw std::runtime_error("hip_modernbert: hipMemcpy (download) failed");
    }
    return out;
}

}
