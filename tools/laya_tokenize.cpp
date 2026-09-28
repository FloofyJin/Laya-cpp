#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "laya/pyjson.hpp"
#include "laya/sequence.hpp"
#include "laya/tokenizer.hpp"

namespace {

using laya::py::Json;

struct Model {
    laya::Tokenizer tok;
    laya::AgentConfig cfg;
};

bool exists(const std::string& path) {
    std::ifstream in(path);
    return static_cast<bool>(in);
}

Model load_model(const std::string& dir) {
    const std::string tok_dir = exists(dir + "/tokenizer/tokenizer.json") ? dir + "/tokenizer" : dir;
    Model m{laya::Tokenizer::from_directory(tok_dir), laya::AgentConfig{}};
    if (exists(dir + "/rl_agent_config.json")) {
        m.cfg = laya::AgentConfig::from_file(dir + "/rl_agent_config.json");
    }
    return m;
}

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

nlohmann::json items_to_json(const std::vector<laya::SequenceItem>& items) {
    nlohmann::json out = nlohmann::json::array();
    for (const auto& it : items) {
        nlohmann::json stats = {{"options", it.stats.options}, {"options_distinct", it.stats.options_distinct}};
        stats["tokens_per_option"] = it.stats.tokens_per_option ? nlohmann::json(*it.stats.tokens_per_option) : nlohmann::json();
        out.push_back({{"qid", it.qid}, {"ids", it.ids}, {"markers", it.markers}, {"qtype", it.qtype}, {"options", stats}});
    }
    return out;
}

nlohmann::json run_request(const Model& m, const Json& req) {
    const int32_t max_len = req.contains("max_len") && !req["max_len"].is_null() ? req["max_len"].get<int32_t>() : m.cfg.max_len;
    const int32_t head_max_len =
        req.contains("head_max_len") && !req["head_max_len"].is_null() ? req["head_max_len"].get<int32_t>() : m.cfg.head_max_len;
    const Json state = req.contains("state") ? req["state"] : Json();
    const Json questions = req.contains("questions") ? req["questions"] : Json();
    return {{"items", items_to_json(laya::encode_state(m.tok, state, questions, max_len, head_max_len))}};
}

int usage() {
    std::cerr << "usage: laya-tokenize --model DIR text TEXT\n"
                 "       laya-tokenize --model DIR texts FILE.jsonl\n"
                 "       laya-tokenize --model DIR request FILE.json|-\n"
                 "       laya-tokenize --model DIR requests FILE.jsonl\n";
    return 2;
}

}

int main(int argc, char** argv) {
    if (argc != 5 || std::string(argv[1]) != "--model") {
        return usage();
    }
    const std::string mode = argv[3];
    const std::string arg = argv[4];
    try {
        const Model m = load_model(argv[2]);
        if (mode == "text") {
            std::cout << nlohmann::json(m.tok.encode(arg)).dump() << "\n";
        } else if (mode == "texts" || mode == "requests") {
            std::istringstream lines(slurp(arg));
            std::string line;
            while (std::getline(lines, line)) {
                if (line.empty()) {
                    continue;
                }
                nlohmann::json out;
                try {
                    const Json v = Json::parse(line);
                    if (mode == "texts") {
                        out = m.tok.encode(v.get<std::string>());
                    } else {
                        out = run_request(m, v);
                    }
                } catch (const std::exception& e) {
                    out = {{"error", e.what()}};
                }
                std::cout << out.dump() << "\n";
            }
        } else if (mode == "request") {
            std::cout << run_request(m, Json::parse(slurp(arg))).dump() << "\n";
        } else {
            return usage();
        }
    } catch (const std::exception& e) {
        std::cerr << "laya-tokenize: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
