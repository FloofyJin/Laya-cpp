#include "laya/decision.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "laya/nn.hpp"

namespace laya {

namespace {

double clamp_temperature(double t) {
    if (!std::isfinite(t)) {
        return 1.0;
    }
    return std::min(5.0, std::max(0.5, t));
}

double clamp_temperature(const nlohmann::json& value) {
    if (!value.is_number()) {
        return 1.0;
    }
    return clamp_temperature(value.get<double>());
}

double round4(double x) {
    return std::round(x * 10000.0) / 10000.0;
}

double clamp01(double x) {
    return std::min(1.0, std::max(0.0, x));
}

std::vector<float> head_layer_forward(const SafetensorsFile& weights, int layer_idx,
                                      const std::vector<float>& hidden_in, size_t seq_len, size_t hidden) {
    const std::string prefix = "head.layers." + std::to_string(layer_idx) + ".";
    const std::vector<float> norm1_w = weights.as_f32(prefix + "norm1.weight");
    const std::vector<float> norm1_b = weights.as_f32(prefix + "norm1.bias");
    const std::vector<float> norm2_w = weights.as_f32(prefix + "norm2.weight");
    const std::vector<float> norm2_b = weights.as_f32(prefix + "norm2.bias");
    const std::vector<float> in_proj_w = weights.as_f32(prefix + "self_attn.in_proj_weight");
    const std::vector<float> in_proj_b = weights.as_f32(prefix + "self_attn.in_proj_bias");
    const std::vector<float> out_proj_w = weights.as_f32(prefix + "self_attn.out_proj.weight");
    const std::vector<float> out_proj_b = weights.as_f32(prefix + "self_attn.out_proj.bias");
    const std::vector<float> lin1_w = weights.as_f32(prefix + "linear1.weight");
    const std::vector<float> lin1_b = weights.as_f32(prefix + "linear1.bias");
    const std::vector<float> lin2_w = weights.as_f32(prefix + "linear2.weight");
    const std::vector<float> lin2_b = weights.as_f32(prefix + "linear2.bias");

    const size_t num_heads = std::max<size_t>(1, hidden / 64);
    const size_t ff_dim = lin1_b.size();

    std::vector<float> normed1(seq_len * hidden);
    for (size_t p = 0; p < seq_len; ++p) {
        nn::layer_norm_row(hidden_in.data() + p * hidden, hidden, norm1_w.data(), norm1_b.data(), 1e-5f,
                           normed1.data() + p * hidden);
    }

    std::vector<float> attn_out(seq_len * hidden);
    nn::multi_head_attention(normed1.data(), seq_len, hidden, num_heads, in_proj_w.data(), in_proj_b.data(),
                             out_proj_w.data(), out_proj_b.data(), attn_out.data());

    std::vector<float> hidden1(seq_len * hidden);
    for (size_t i = 0; i < hidden1.size(); ++i) {
        hidden1[i] = hidden_in[i] + attn_out[i];
    }

    std::vector<float> normed2(seq_len * hidden);
    for (size_t p = 0; p < seq_len; ++p) {
        nn::layer_norm_row(hidden1.data() + p * hidden, hidden, norm2_w.data(), norm2_b.data(), 1e-5f,
                           normed2.data() + p * hidden);
    }

    std::vector<float> ff1(seq_len * ff_dim);
    nn::linear(normed2.data(), seq_len, hidden, lin1_w.data(), lin1_b.data(), ff_dim, ff1.data());
    for (float& v : ff1) {
        v = nn::relu(v);
    }

    std::vector<float> ff2(seq_len * hidden);
    nn::linear(ff1.data(), seq_len, ff_dim, lin2_w.data(), lin2_b.data(), hidden, ff2.data());

    std::vector<float> hidden2(seq_len * hidden);
    for (size_t i = 0; i < hidden2.size(); ++i) {
        hidden2[i] = hidden1[i] + ff2[i];
    }
    return hidden2;
}

double entropy_confidence(const std::vector<double>& p, size_t k) {
    if (k < 2) {
        return 1.0;
    }
    double ent = 0.0;
    for (double v : p) {
        ent -= v * std::log(std::max(v, 1e-12));
    }
    return clamp01(1.0 - ent / std::log(static_cast<double>(k)));
}

}

TemperatureConfig TemperatureConfig::from_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    const nlohmann::json j = nlohmann::json::parse(py::nonfinite_to_null(ss.str()));

    TemperatureConfig cfg;
    if (j.contains("temperature")) {
        const auto& t = j.at("temperature");
        for (size_t i = 0; i < 3 && i < t.size(); ++i) {
            cfg.temperature[i] = clamp_temperature(t.at(i));
        }
    }
    if (j.contains("temperature_by_options")) {
        for (auto it = j.at("temperature_by_options").begin(); it != j.at("temperature_by_options").end(); ++it) {
            cfg.temperature_by_options[it.key()] = clamp_temperature(it.value());
        }
    }
    return cfg;
}

double TemperatureConfig::resolve(const std::string& type_name, int32_t qtype, int32_t option_count) const {
    const std::string size = option_count <= 2   ? "2"
                             : option_count <= 5  ? "3-5"
                             : option_count <= 10 ? "6-10"
                                                  : "11+";
    const std::string bucket = type_name + ":" + size;
    const auto it = temperature_by_options.find(bucket);
    if (it != temperature_by_options.end()) {
        return it->second;
    }
    return temperature[qtype];
}

py::Json predict_one(const SafetensorsFile& weights, const ModernBertConfig& enc_cfg,
                     const TemperatureConfig& temps, const Question& q, const SequenceItem& item) {
    const size_t seq_len = item.ids.size();
    const size_t hidden = static_cast<size_t>(enc_cfg.hidden_size);

    std::vector<float> h = modernbert_encoder(weights, enc_cfg, item.ids);

    const std::vector<float> type_emb =
        weights.as_f32_slice("type_emb.weight", static_cast<size_t>(item.qtype) * hidden, hidden);
    for (size_t p = 0; p < seq_len; ++p) {
        for (size_t d = 0; d < hidden; ++d) {
            h[p * hidden + d] += type_emb[d];
        }
    }

    for (int layer = 0; layer < 2; ++layer) {
        h = head_layer_forward(weights, layer, h, seq_len, hidden);
    }

    return decode_from_hidden(weights, temps, q, item, h, hidden);
}

py::Json decode_from_hidden(const SafetensorsFile& weights, const TemperatureConfig& temps, const Question& q,
                            const SequenceItem& item, const std::vector<float>& hidden_buf, size_t hidden) {
    const std::vector<float>& h = hidden_buf;
    const size_t k = item.markers.size();

    const std::vector<float> scorer_norm_w = weights.as_f32("scorer.0.weight");
    const std::vector<float> scorer_norm_b = weights.as_f32("scorer.0.bias");
    const std::vector<float> scorer_lin1_w = weights.as_f32("scorer.1.weight");
    const std::vector<float> scorer_lin1_b = weights.as_f32("scorer.1.bias");
    const std::vector<float> scorer_lin2_w = weights.as_f32("scorer.3.weight");
    const std::vector<float> scorer_lin2_b = weights.as_f32("scorer.3.bias");

    std::vector<float> logits(k);
    for (size_t m = 0; m < k; ++m) {
        const float* marker_vec = h.data() + static_cast<size_t>(item.markers[m]) * hidden;
        std::vector<float> normed(hidden);
        nn::layer_norm_row(marker_vec, hidden, scorer_norm_w.data(), scorer_norm_b.data(), 1e-5f, normed.data());
        std::vector<float> mid(hidden);
        nn::linear(normed.data(), 1, hidden, scorer_lin1_w.data(), scorer_lin1_b.data(), hidden, mid.data());
        for (float& v : mid) {
            v = nn::gelu(v);
        }
        float score = 0.0f;
        nn::linear(mid.data(), 1, hidden, scorer_lin2_w.data(), scorer_lin2_b.data(), 1, &score);
        logits[m] = score;
    }

    std::vector<float> p_raw(k);
    {
        const float max_l = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        for (size_t i = 0; i < k; ++i) {
            p_raw[i] = std::exp(logits[i] - max_l);
            sum += p_raw[i];
        }
        for (float& v : p_raw) {
            v /= sum;
        }
    }

    float top1 = 0.0f;
    float top2 = 0.0f;
    for (float v : p_raw) {
        if (v > top1) {
            top2 = top1;
            top1 = v;
        } else if (v > top2) {
            top2 = v;
        }
    }
    const float k_feat = static_cast<float>(std::max<size_t>(k, 2));
    float ent = 0.0f;
    for (float v : p_raw) {
        ent -= v * std::log(std::max(v, 1e-9f));
    }
    ent /= std::log(k_feat);

    std::vector<float> act_in(hidden + 4);
    std::copy(h.begin(), h.begin() + static_cast<std::ptrdiff_t>(hidden), act_in.begin());
    act_in[hidden + 0] = top1;
    act_in[hidden + 1] = top1 - top2;
    act_in[hidden + 2] = ent;
    act_in[hidden + 3] = k_feat / 255.0f;

    const std::vector<float> act0_w = weights.as_f32("act_head.0.weight");
    const std::vector<float> act0_b = weights.as_f32("act_head.0.bias");
    const std::vector<float> act2_w = weights.as_f32("act_head.2.weight");
    const std::vector<float> act2_b = weights.as_f32("act_head.2.bias");
    const size_t act_hidden = act0_b.size();
    const size_t n_act = act2_b.size();

    std::vector<float> act_mid(act_hidden);
    nn::linear(act_in.data(), 1, hidden + 4, act0_w.data(), act0_b.data(), act_hidden, act_mid.data());
    for (float& v : act_mid) {
        v = nn::gelu(v);
    }
    std::vector<float> act_logits(n_act);
    nn::linear(act_mid.data(), 1, act_hidden, act2_w.data(), act2_b.data(), n_act, act_logits.data());

    std::vector<float> act_probs(n_act);
    {
        const float max_l = *std::max_element(act_logits.begin(), act_logits.end());
        float sum = 0.0f;
        for (size_t i = 0; i < n_act; ++i) {
            act_probs[i] = std::exp(act_logits[i] - max_l);
            sum += act_probs[i];
        }
        for (float& v : act_probs) {
            v /= sum;
        }
    }
    const double act_probability = round4(act_probs[0]);

    const double t_scale = temps.resolve(q.type_name, item.qtype, static_cast<int32_t>(k));
    std::vector<double> p(k);
    {
        std::vector<double> z(k);
        for (size_t i = 0; i < k; ++i) {
            z[i] = static_cast<double>(logits[i]) / t_scale;
        }
        const double max_z = *std::max_element(z.begin(), z.end());
        double sum = 0.0;
        for (size_t i = 0; i < k; ++i) {
            p[i] = std::exp(z[i] - max_z);
            sum += p[i];
        }
        for (double& v : p) {
            v /= sum;
        }
    }

    const double answer_confidence = round4(clamp01(*std::max_element(p.begin(), p.end())));

    py::Json action;
    action["act_probability"] = act_probability;

    py::Json out;
    if (q.type == QType::Choice) {
        size_t best = 0;
        for (size_t i = 1; i < k; ++i) {
            if (p[i] > p[best]) {
                best = i;
            }
        }
        out["type"] = "choice";
        out["choice"] = q.options[best].first;
        py::Json probs = py::Json::object();
        for (size_t i = 0; i < k; ++i) {
            probs[py::str(q.options[i].first)] = round4(p[i]);
        }
        out["probabilities"] = probs;
        out["confidence"] = round4(entropy_confidence(p, k));
        out["answer_confidence"] = answer_confidence;
        out["action"] = action;
    } else if (q.type == QType::Score) {
        double exp_score = 0.0;
        for (size_t i = 0; i < k; ++i) {
            exp_score += static_cast<double>(i) * p[i];
        }
        out["type"] = "score";
        out["score"] = round4(exp_score);
        py::Json legend = py::Json::object();
        py::Json probs = py::Json::object();
        for (size_t i = 0; i < k; ++i) {
            legend[std::to_string(i)] = q.options[i].second;
            probs[std::to_string(i)] = round4(p[i]);
        }
        out["legend"] = legend;
        out["probabilities"] = probs;
        out["confidence"] = round4(entropy_confidence(p, k));
        out["answer_confidence"] = answer_confidence;
        out["action"] = action;
    } else {
        out["type"] = "noul";
        out["noul"] = round4(p[1]);
        out["confidence"] = round4(std::max(p[1], 1.0 - p[1]));
        out["answer_confidence"] = answer_confidence;
        out["action"] = action;
    }
    return out;
}

}
