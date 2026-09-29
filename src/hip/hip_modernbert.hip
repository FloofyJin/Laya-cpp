#include "hip_modernbert.hpp"

#include <algorithm>
#include <stdexcept>

#include <hip/hip_runtime.h>

#include "hip_nn.hpp"

namespace laya {

namespace {

float* upload(const std::vector<float>& host) {
    float* d_ptr = nullptr;
    if (hipMalloc(&d_ptr, host.size() * sizeof(float)) != hipSuccess) {
        throw std::runtime_error("hip_modernbert: hipMalloc failed");
    }
    if (hipMemcpy(d_ptr, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) {
        (void)hipFree(d_ptr);
        throw std::runtime_error("hip_modernbert: hipMemcpy failed");
    }
    return d_ptr;
}

float* device_alloc(size_t count) {
    float* d_ptr = nullptr;
    if (hipMalloc(&d_ptr, count * sizeof(float)) != hipSuccess) {
        throw std::runtime_error("hip_modernbert: hipMalloc failed");
    }
    return d_ptr;
}

void attention_block(hipblasHandle_t handle, const HipWeights& weights, const std::string& prefix,
                     const float* d_input, float* d_out, size_t seq_len, size_t hidden, size_t num_heads,
                     bool sliding, size_t window, const float* d_cos, const float* d_sin, bool has_bias) {
    const size_t head_dim = hidden / num_heads;
    const size_t half_dim = head_dim / 2;

    float* d_qkv = device_alloc(seq_len * 3 * hidden);
    const float* wqkv_b = has_bias ? weights.get(prefix + "in_proj_bias") : nullptr;
    const std::string wqkv_name = has_bias ? prefix + "in_proj_weight" : prefix + "Wqkv.weight";
    hip_nn::linear(handle, d_input, seq_len, hidden, weights.get(wqkv_name), wqkv_b, 3 * hidden, d_qkv);

    if (d_cos != nullptr) {
        hip_nn::rope_apply_inplace(d_qkv, seq_len, hidden, num_heads, half_dim, d_cos, d_sin);
    }

    float* d_context = device_alloc(seq_len * hidden);
    hip_nn::attention(d_qkv, seq_len, hidden, num_heads, sliding, window, d_context);
    (void)hipFree(d_qkv);

    const float* wo_b = has_bias ? weights.get(prefix + "out_proj.bias") : nullptr;
    const std::string wo_name = has_bias ? prefix + "out_proj.weight" : prefix + "Wo.weight";
    hip_nn::linear(handle, d_context, seq_len, hidden, weights.get(wo_name), wo_b, hidden, d_out);
    (void)hipFree(d_context);
}

void mlp_block(hipblasHandle_t handle, const HipWeights& weights, const std::string& wi_name,
              const std::string& wo_name, const float* d_input, float* d_out, size_t seq_len, size_t hidden,
              size_t inter) {
    float* d_proj = device_alloc(seq_len * 2 * inter);
    hip_nn::linear(handle, d_input, seq_len, hidden, weights.get(wi_name), nullptr, 2 * inter, d_proj);

    float* d_gated = device_alloc(seq_len * inter);
    hip_nn::gelu_mul(d_proj, seq_len, inter, d_gated);
    (void)hipFree(d_proj);

    hip_nn::linear(handle, d_gated, seq_len, inter, weights.get(wo_name), nullptr, hidden, d_out);
    (void)hipFree(d_gated);
}

}

HipTensor hip_modernbert_encoder(hipblasHandle_t handle, const HipWeights& weights, const ModernBertConfig& cfg,
                                 const std::vector<int32_t>& input_ids) {
    const size_t seq_len = input_ids.size();
    const size_t hidden = static_cast<size_t>(cfg.hidden_size);
    const size_t num_heads = static_cast<size_t>(cfg.num_attention_heads);
    const size_t head_dim = hidden / num_heads;
    const size_t half_dim = head_dim / 2;
    const size_t window = static_cast<size_t>(cfg.local_attention) / 2;

    int32_t* d_ids = nullptr;
    if (hipMalloc(&d_ids, seq_len * sizeof(int32_t)) != hipSuccess) {
        throw std::runtime_error("hip_modernbert: hipMalloc failed for ids");
    }
    if (hipMemcpy(d_ids, input_ids.data(), seq_len * sizeof(int32_t), hipMemcpyHostToDevice) != hipSuccess) {
        (void)hipFree(d_ids);
        throw std::runtime_error("hip_modernbert: hipMemcpy failed for ids");
    }

    float* d_embed = device_alloc(seq_len * hidden);
    hip_nn::embedding_lookup(weights.get("encoder.embeddings.tok_embeddings.weight"), d_ids, seq_len, hidden,
                             d_embed);
    (void)hipFree(d_ids);

    float* d_hidden = device_alloc(seq_len * hidden);
    hip_nn::layer_norm(d_embed, seq_len, hidden, weights.get("encoder.embeddings.norm.weight"), nullptr, cfg.norm_eps,
                       d_hidden);
    (void)hipFree(d_embed);

    const RopeTable rope_full = modernbert_rope_table(seq_len, half_dim, cfg.global_rope_theta);
    const RopeTable rope_local = modernbert_rope_table(seq_len, half_dim, cfg.local_rope_theta);
    float* d_cos_full = upload(rope_full.cos);
    float* d_sin_full = upload(rope_full.sin);
    float* d_cos_local = upload(rope_local.cos);
    float* d_sin_local = upload(rope_local.sin);

    for (int layer_idx = 0; layer_idx < cfg.num_hidden_layers; ++layer_idx) {
        const bool sliding = cfg.layer_types.at(static_cast<size_t>(layer_idx)) == "sliding_attention";
        const std::string prefix = "encoder.layers." + std::to_string(layer_idx) + ".";

        const float* d_attn_input = d_hidden;
        float* d_attn_normed = nullptr;
        if (layer_idx != 0) {
            d_attn_normed = device_alloc(seq_len * hidden);
            hip_nn::layer_norm(d_hidden, seq_len, hidden, weights.get(prefix + "attn_norm.weight"), nullptr,
                               cfg.norm_eps, d_attn_normed);
            d_attn_input = d_attn_normed;
        }

        float* d_attn_out = device_alloc(seq_len * hidden);
        attention_block(handle, weights, prefix + "attn.", d_attn_input, d_attn_out, seq_len, hidden, num_heads,
                        sliding, window, sliding ? d_cos_local : d_cos_full, sliding ? d_sin_local : d_sin_full,
                        false);
        if (d_attn_normed != nullptr) {
            (void)hipFree(d_attn_normed);
        }

        hip_nn::add_inplace(d_hidden, d_attn_out, seq_len * hidden);
        (void)hipFree(d_attn_out);

        float* d_mlp_normed = device_alloc(seq_len * hidden);
        hip_nn::layer_norm(d_hidden, seq_len, hidden, weights.get(prefix + "mlp_norm.weight"), nullptr, cfg.norm_eps,
                           d_mlp_normed);

        float* d_mlp_out = device_alloc(seq_len * hidden);
        mlp_block(handle, weights, prefix + "mlp.Wi.weight", prefix + "mlp.Wo.weight", d_mlp_normed, d_mlp_out,
                 seq_len, hidden, static_cast<size_t>(cfg.intermediate_size));
        (void)hipFree(d_mlp_normed);

        hip_nn::add_inplace(d_hidden, d_mlp_out, seq_len * hidden);
        (void)hipFree(d_mlp_out);
    }

    (void)hipFree(d_cos_full);
    (void)hipFree(d_sin_full);
    (void)hipFree(d_cos_local);
    (void)hipFree(d_sin_local);

    float* d_out = device_alloc(seq_len * hidden);
    hip_nn::layer_norm(d_hidden, seq_len, hidden, weights.get("encoder.final_norm.weight"), nullptr, cfg.norm_eps,
                       d_out);
    (void)hipFree(d_hidden);

    return HipTensor{d_out, seq_len, hidden};
}

HipTensor hip_add_type_emb_and_head(hipblasHandle_t handle, const HipWeights& weights, const ModernBertConfig& cfg,
                                    HipTensor t, int32_t qtype) {
    (void)cfg;
    const size_t hidden = t.hidden;
    const size_t seq_len = t.seq_len;
    const size_t num_heads = std::max<size_t>(1, hidden / 64);

    const float* d_type_row = weights.get("type_emb.weight") + static_cast<size_t>(qtype) * hidden;
    hip_nn::add_broadcast_row(t.data, seq_len, hidden, d_type_row);

    for (int layer = 0; layer < 2; ++layer) {
        const std::string prefix = "head.layers." + std::to_string(layer) + ".";

        float* d_normed1 = device_alloc(seq_len * hidden);
        hip_nn::layer_norm(t.data, seq_len, hidden, weights.get(prefix + "norm1.weight"),
                           weights.get(prefix + "norm1.bias"), 1e-5f, d_normed1);

        float* d_attn_out = device_alloc(seq_len * hidden);
        attention_block(handle, weights, prefix + "self_attn.", d_normed1, d_attn_out, seq_len, hidden, num_heads,
                        false, 0, nullptr, nullptr, true);
        (void)hipFree(d_normed1);

        hip_nn::add_inplace(t.data, d_attn_out, seq_len * hidden);
        (void)hipFree(d_attn_out);

        float* d_normed2 = device_alloc(seq_len * hidden);
        hip_nn::layer_norm(t.data, seq_len, hidden, weights.get(prefix + "norm2.weight"),
                           weights.get(prefix + "norm2.bias"), 1e-5f, d_normed2);

        const size_t ff_dim = 4 * hidden;
        float* d_ff1 = device_alloc(seq_len * ff_dim);
        hip_nn::linear(handle, d_normed2, seq_len, hidden, weights.get(prefix + "linear1.weight"),
                       weights.get(prefix + "linear1.bias"), ff_dim, d_ff1);
        (void)hipFree(d_normed2);
        hip_nn::relu_inplace(d_ff1, seq_len * ff_dim);

        float* d_ff2 = device_alloc(seq_len * hidden);
        hip_nn::linear(handle, d_ff1, seq_len, ff_dim, weights.get(prefix + "linear2.weight"),
                       weights.get(prefix + "linear2.bias"), hidden, d_ff2);
        (void)hipFree(d_ff1);

        hip_nn::add_inplace(t.data, d_ff2, seq_len * hidden);
        (void)hipFree(d_ff2);
    }

    return t;
}

void hip_free_tensor(HipTensor& t) {
    if (t.data != nullptr) {
        (void)hipFree(t.data);
        t.data = nullptr;
    }
}

std::vector<float> hip_download(const HipTensor& t) {
    std::vector<float> out(t.seq_len * t.hidden);
    if (hipMemcpy(out.data(), t.data, out.size() * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) {
        throw std::runtime_error("hip_modernbert: hipMemcpy (download) failed");
    }
    return out;
}

}
