#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "laya/safetensors.hpp"

namespace laya {

struct ModernBertConfig {
    int32_t vocab_size = 0;
    int32_t hidden_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t num_attention_heads = 0;
    int32_t intermediate_size = 0;
    int32_t pad_token_id = 0;
    int32_t local_attention = 0;
    float norm_eps = 1e-5f;
    bool norm_bias = false;
    bool attention_bias = false;
    bool mlp_bias = false;
    std::string hidden_activation = "gelu";
    std::vector<std::string> layer_types;
    double global_rope_theta = 160000.0;
    double local_rope_theta = 10000.0;

    static ModernBertConfig from_file(const std::string& path);
};

std::vector<float> modernbert_embeddings(const SafetensorsFile& weights, const ModernBertConfig& cfg,
                                         const std::vector<int32_t>& input_ids);

struct RopeTable {
    size_t seq_len = 0;
    size_t half_dim = 0;
    std::vector<float> cos;
    std::vector<float> sin;
};

RopeTable modernbert_rope_table(size_t seq_len, size_t half_dim, double theta);

std::vector<float> modernbert_layer(const SafetensorsFile& weights, const ModernBertConfig& cfg, int layer_idx,
                                    const std::vector<float>& hidden_in, size_t seq_len, const RopeTable& rope);

std::vector<float> modernbert_encoder(const SafetensorsFile& weights, const ModernBertConfig& cfg,
                                      const std::vector<int32_t>& input_ids);

}
