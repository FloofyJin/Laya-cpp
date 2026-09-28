#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "laya/modernbert.hpp"
#include "laya/pyjson.hpp"
#include "laya/safetensors.hpp"

namespace {

using laya::py::Json;

std::string slurp(const std::string& path) {
    if (path == "-") {
        std::ostringstream ss;
        ss << std::cin.rdbuf();
        return ss.str();
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

nlohmann::json run_embeddings(const laya::SafetensorsFile& weights, const laya::ModernBertConfig& cfg,
                              const Json& req) {
    std::vector<int32_t> ids;
    for (const auto& v : req.at("input_ids")) {
        ids.push_back(v.get<int32_t>());
    }
    const std::vector<float> out = laya::modernbert_embeddings(weights, cfg, ids);
    nlohmann::json values = nlohmann::json::array();
    for (float v : out) {
        values.push_back(v);
    }
    return {{"shape", nlohmann::json::array({ids.size(), static_cast<size_t>(cfg.hidden_size)})},
            {"values", values}};
}

nlohmann::json run_layer(const laya::SafetensorsFile& weights, const laya::ModernBertConfig& cfg, const Json& req) {
    std::vector<int32_t> ids;
    for (const auto& v : req.at("input_ids")) {
        ids.push_back(v.get<int32_t>());
    }
    const int upto = req.at("layer").get<int>();
    const size_t seq_len = ids.size();
    const size_t head_dim = static_cast<size_t>(cfg.hidden_size) / static_cast<size_t>(cfg.num_attention_heads);
    const size_t half_dim = head_dim / 2;
    const laya::RopeTable rope_full = laya::modernbert_rope_table(seq_len, half_dim, cfg.global_rope_theta);
    const laya::RopeTable rope_local = laya::modernbert_rope_table(seq_len, half_dim, cfg.local_rope_theta);

    std::vector<float> hidden = laya::modernbert_embeddings(weights, cfg, ids);
    for (int i = 0; i <= upto; ++i) {
        const bool sliding = cfg.layer_types.at(static_cast<size_t>(i)) == "sliding_attention";
        hidden = laya::modernbert_layer(weights, cfg, i, hidden, seq_len, sliding ? rope_local : rope_full);
    }

    nlohmann::json values = nlohmann::json::array();
    for (float v : hidden) {
        values.push_back(v);
    }
    return {{"shape", nlohmann::json::array({seq_len, static_cast<size_t>(cfg.hidden_size)})}, {"values", values}};
}

nlohmann::json run_full(const laya::SafetensorsFile& weights, const laya::ModernBertConfig& cfg, const Json& req) {
    std::vector<int32_t> ids;
    for (const auto& v : req.at("input_ids")) {
        ids.push_back(v.get<int32_t>());
    }
    const std::vector<float> out = laya::modernbert_encoder(weights, cfg, ids);
    nlohmann::json values = nlohmann::json::array();
    for (float v : out) {
        values.push_back(v);
    }
    return {{"shape", nlohmann::json::array({ids.size(), static_cast<size_t>(cfg.hidden_size)})}, {"values", values}};
}

int usage() {
    std::cerr << "usage: laya-encoder-probe --model DIR embeddings FILE.jsonl\n"
                 "       laya-encoder-probe --model DIR layer FILE.jsonl\n"
                 "       laya-encoder-probe --model DIR full FILE.jsonl\n";
    return 2;
}

}

int main(int argc, char** argv) {
    if (argc != 5 || std::string(argv[1]) != "--model") {
        return usage();
    }
    const std::string dir = argv[2];
    const std::string mode = argv[3];
    const std::string arg = argv[4];
    try {
        laya::SafetensorsFile weights = laya::SafetensorsFile::open(dir + "/model.safetensors");
        laya::ModernBertConfig cfg = laya::ModernBertConfig::from_file(dir + "/encoder/config.json");

        std::istringstream lines(slurp(arg));
        std::string line;
        while (std::getline(lines, line)) {
            if (line.empty()) {
                continue;
            }
            nlohmann::json out;
            try {
                const Json v = Json::parse(line);
                if (mode == "embeddings") {
                    out = run_embeddings(weights, cfg, v);
                } else if (mode == "layer") {
                    out = run_layer(weights, cfg, v);
                } else if (mode == "full") {
                    out = run_full(weights, cfg, v);
                } else {
                    throw std::runtime_error("unknown mode " + mode);
                }
            } catch (const std::exception& e) {
                out = {{"error", e.what()}};
            }
            std::cout << out.dump() << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "laya-encoder-probe: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
