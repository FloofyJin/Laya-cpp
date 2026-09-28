#include "laya/modernbert.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "laya/nn.hpp"

namespace laya {

namespace {

void apply_rope_pair(float& x0, float& x_half, float cos_v, float sin_v) {
    const float r0 = x0 * cos_v - x_half * sin_v;
    const float r_half = x_half * cos_v + x0 * sin_v;
    x0 = r0;
    x_half = r_half;
}

std::vector<float> modernbert_attention_forward(const SafetensorsFile& weights, const ModernBertConfig& cfg,
                                                 int layer_idx, const std::vector<float>& normed, size_t seq_len,
                                                 const RopeTable& rope, bool sliding) {
    const size_t hidden = static_cast<size_t>(cfg.hidden_size);
    const size_t num_heads = static_cast<size_t>(cfg.num_attention_heads);
    const size_t head_dim = hidden / num_heads;
    const size_t half_dim = head_dim / 2;
    const size_t window = static_cast<size_t>(cfg.local_attention) / 2;

    const std::string prefix = "encoder.layers." + std::to_string(layer_idx) + ".attn.";
    const std::vector<float> wqkv = weights.as_f32(prefix + "Wqkv.weight");
    const std::vector<float> wo = weights.as_f32(prefix + "Wo.weight");

    std::vector<float> qkv(seq_len * 3 * hidden);
    nn::linear(normed.data(), seq_len, hidden, wqkv.data(), nullptr, 3 * hidden, qkv.data());

    for (size_t p = 0; p < seq_len; ++p) {
        float* row = qkv.data() + p * 3 * hidden;
        for (size_t h = 0; h < num_heads; ++h) {
            float* q = row + h * head_dim;
            float* k = row + hidden + h * head_dim;
            for (size_t j = 0; j < half_dim; ++j) {
                const float c = rope.cos[p * half_dim + j];
                const float s = rope.sin[p * half_dim + j];
                apply_rope_pair(q[j], q[half_dim + j], c, s);
                apply_rope_pair(k[j], k[half_dim + j], c, s);
            }
        }
    }

    const float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> context(seq_len * hidden);
    std::vector<float> scores(seq_len);

    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < seq_len; ++i) {
            const float* qi = qkv.data() + i * 3 * hidden + h * head_dim;
            const size_t jlo = sliding ? (i >= window ? i - window : 0) : 0;
            const size_t jhi = sliding ? std::min(seq_len, i + window + 1) : seq_len;

            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t j = jlo; j < jhi; ++j) {
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
            for (size_t j = jlo; j < jhi; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                sum += scores[j];
            }
            float* out_row = context.data() + i * hidden + h * head_dim;
            for (size_t d = 0; d < head_dim; ++d) {
                out_row[d] = 0.0f;
            }
            for (size_t j = jlo; j < jhi; ++j) {
                const float w = scores[j] / sum;
                const float* vj = qkv.data() + j * 3 * hidden + 2 * hidden + h * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    out_row[d] += w * vj[d];
                }
            }
        }
    }

    std::vector<float> out(seq_len * hidden);
    nn::linear(context.data(), seq_len, hidden, wo.data(), nullptr, hidden, out.data());
    return out;
}

std::vector<float> modernbert_mlp_forward(const SafetensorsFile& weights, const ModernBertConfig& cfg,
                                          int layer_idx, const std::vector<float>& normed, size_t seq_len) {
    const size_t hidden = static_cast<size_t>(cfg.hidden_size);
    const size_t inter = static_cast<size_t>(cfg.intermediate_size);
    const std::string prefix = "encoder.layers." + std::to_string(layer_idx) + ".mlp.";
    const std::vector<float> wi = weights.as_f32(prefix + "Wi.weight");
    const std::vector<float> wo = weights.as_f32(prefix + "Wo.weight");

    std::vector<float> proj(seq_len * 2 * inter);
    nn::linear(normed.data(), seq_len, hidden, wi.data(), nullptr, 2 * inter, proj.data());

    std::vector<float> gated(seq_len * inter);
    for (size_t p = 0; p < seq_len; ++p) {
        const float* row = proj.data() + p * 2 * inter;
        float* g = gated.data() + p * inter;
        for (size_t i = 0; i < inter; ++i) {
            g[i] = nn::gelu(row[i]) * row[inter + i];
        }
    }

    std::vector<float> out(seq_len * hidden);
    nn::linear(gated.data(), seq_len, inter, wo.data(), nullptr, hidden, out.data());
    return out;
}

}

ModernBertConfig ModernBertConfig::from_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    const nlohmann::json j = nlohmann::json::parse(ss.str());

    ModernBertConfig cfg;
    cfg.vocab_size = j.at("vocab_size").get<int32_t>();
    cfg.hidden_size = j.at("hidden_size").get<int32_t>();
    cfg.num_hidden_layers = j.at("num_hidden_layers").get<int32_t>();
    cfg.num_attention_heads = j.at("num_attention_heads").get<int32_t>();
    cfg.intermediate_size = j.at("intermediate_size").get<int32_t>();
    cfg.pad_token_id = j.at("pad_token_id").get<int32_t>();
    cfg.local_attention = j.at("local_attention").get<int32_t>();
    cfg.norm_eps = j.at("norm_eps").get<float>();
    cfg.norm_bias = j.at("norm_bias").get<bool>();
    cfg.attention_bias = j.at("attention_bias").get<bool>();
    cfg.mlp_bias = j.at("mlp_bias").get<bool>();
    cfg.hidden_activation = j.at("hidden_activation").get<std::string>();
    for (const auto& t : j.at("layer_types")) {
        cfg.layer_types.push_back(t.get<std::string>());
    }
    const auto& rope = j.at("rope_parameters");
    cfg.global_rope_theta = rope.at("full_attention").at("rope_theta").get<double>();
    cfg.local_rope_theta = rope.at("sliding_attention").at("rope_theta").get<double>();
    return cfg;
}

std::vector<float> modernbert_embeddings(const SafetensorsFile& weights, const ModernBertConfig& cfg,
                                         const std::vector<int32_t>& input_ids) {
    const size_t hidden = static_cast<size_t>(cfg.hidden_size);
    const size_t seq = input_ids.size();
    const std::vector<float> norm_weight = weights.as_f32("encoder.embeddings.norm.weight");

    std::vector<float> out(seq * hidden);
    for (size_t i = 0; i < seq; ++i) {
        const size_t token = static_cast<size_t>(input_ids[i]);
        const std::vector<float> row =
            weights.as_f32_slice("encoder.embeddings.tok_embeddings.weight", token * hidden, hidden);
        nn::layer_norm_row(row.data(), hidden, norm_weight.data(), nullptr, cfg.norm_eps, out.data() + i * hidden);
    }
    return out;
}

RopeTable modernbert_rope_table(size_t seq_len, size_t half_dim, double theta) {
    RopeTable t;
    t.seq_len = seq_len;
    t.half_dim = half_dim;
    t.cos.resize(seq_len * half_dim);
    t.sin.resize(seq_len * half_dim);

    std::vector<double> inv_freq(half_dim);
    for (size_t k = 0; k < half_dim; ++k) {
        const double exponent = (2.0 * static_cast<double>(k)) / static_cast<double>(2 * half_dim);
        inv_freq[k] = 1.0 / std::pow(theta, exponent);
    }
    for (size_t p = 0; p < seq_len; ++p) {
        for (size_t k = 0; k < half_dim; ++k) {
            const double angle = static_cast<double>(p) * inv_freq[k];
            t.cos[p * half_dim + k] = static_cast<float>(std::cos(angle));
            t.sin[p * half_dim + k] = static_cast<float>(std::sin(angle));
        }
    }
    return t;
}

std::vector<float> modernbert_layer(const SafetensorsFile& weights, const ModernBertConfig& cfg, int layer_idx,
                                    const std::vector<float>& hidden_in, size_t seq_len, const RopeTable& rope) {
    const size_t hidden = static_cast<size_t>(cfg.hidden_size);
    const bool sliding = cfg.layer_types.at(static_cast<size_t>(layer_idx)) == "sliding_attention";

    std::vector<float> normed(seq_len * hidden);
    if (layer_idx == 0) {
        normed = hidden_in;
    } else {
        const std::vector<float> w =
            weights.as_f32("encoder.layers." + std::to_string(layer_idx) + ".attn_norm.weight");
        for (size_t p = 0; p < seq_len; ++p) {
            nn::layer_norm_row(hidden_in.data() + p * hidden, hidden, w.data(), nullptr, cfg.norm_eps,
                               normed.data() + p * hidden);
        }
    }

    const std::vector<float> attn_out =
        modernbert_attention_forward(weights, cfg, layer_idx, normed, seq_len, rope, sliding);

    std::vector<float> hidden1(seq_len * hidden);
    for (size_t i = 0; i < hidden1.size(); ++i) {
        hidden1[i] = hidden_in[i] + attn_out[i];
    }

    const std::vector<float> mlp_norm_w =
        weights.as_f32("encoder.layers." + std::to_string(layer_idx) + ".mlp_norm.weight");
    std::vector<float> normed2(seq_len * hidden);
    for (size_t p = 0; p < seq_len; ++p) {
        nn::layer_norm_row(hidden1.data() + p * hidden, hidden, mlp_norm_w.data(), nullptr, cfg.norm_eps,
                           normed2.data() + p * hidden);
    }

    const std::vector<float> mlp_out = modernbert_mlp_forward(weights, cfg, layer_idx, normed2, seq_len);

    std::vector<float> hidden2(seq_len * hidden);
    for (size_t i = 0; i < hidden2.size(); ++i) {
        hidden2[i] = hidden1[i] + mlp_out[i];
    }
    return hidden2;
}

std::vector<float> modernbert_encoder(const SafetensorsFile& weights, const ModernBertConfig& cfg,
                                      const std::vector<int32_t>& input_ids) {
    const size_t seq_len = input_ids.size();
    const size_t head_dim = static_cast<size_t>(cfg.hidden_size) / static_cast<size_t>(cfg.num_attention_heads);
    const size_t half_dim = head_dim / 2;

    const RopeTable rope_full = modernbert_rope_table(seq_len, half_dim, cfg.global_rope_theta);
    const RopeTable rope_local = modernbert_rope_table(seq_len, half_dim, cfg.local_rope_theta);

    std::vector<float> hidden = modernbert_embeddings(weights, cfg, input_ids);
    for (int layer_idx = 0; layer_idx < cfg.num_hidden_layers; ++layer_idx) {
        const bool sliding = cfg.layer_types.at(static_cast<size_t>(layer_idx)) == "sliding_attention";
        hidden = modernbert_layer(weights, cfg, layer_idx, hidden, seq_len, sliding ? rope_local : rope_full);
    }

    const std::vector<float> final_norm_w = weights.as_f32("encoder.final_norm.weight");
    std::vector<float> out(seq_len * static_cast<size_t>(cfg.hidden_size));
    for (size_t p = 0; p < seq_len; ++p) {
        nn::layer_norm_row(hidden.data() + p * cfg.hidden_size, static_cast<size_t>(cfg.hidden_size),
                           final_norm_w.data(), nullptr, cfg.norm_eps, out.data() + p * cfg.hidden_size);
    }
    return out;
}

}
