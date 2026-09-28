#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "laya/decision.hpp"
#include "laya/modernbert.hpp"
#include "laya/pyjson.hpp"
#include "laya/safetensors.hpp"
#include "laya/sequence.hpp"
#include "laya/tokenizer.hpp"

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

struct Model {
    laya::Tokenizer tok;
    laya::AgentConfig cfg;
    laya::SafetensorsFile weights;
    laya::ModernBertConfig enc_cfg;
    laya::TemperatureConfig temps;
};

bool exists(const std::string& path) {
    std::ifstream in(path);
    return static_cast<bool>(in);
}

Model load_model(const std::string& dir) {
    const std::string tok_dir = exists(dir + "/tokenizer/tokenizer.json") ? dir + "/tokenizer" : dir;
    Model m{laya::Tokenizer::from_directory(tok_dir), laya::AgentConfig{},
            laya::SafetensorsFile::open(dir + "/model.safetensors"),
            laya::ModernBertConfig::from_file(dir + "/encoder/config.json"),
            laya::TemperatureConfig::from_file(dir + "/rl_agent_config.json")};
    if (exists(dir + "/rl_agent_config.json")) {
        m.cfg = laya::AgentConfig::from_file(dir + "/rl_agent_config.json");
    }
    return m;
}

Json run_request(const Model& m, const Json& req) {
    const int32_t max_len =
        req.contains("max_len") && !req["max_len"].is_null() ? req["max_len"].get<int32_t>() : m.cfg.max_len;
    const int32_t head_max_len = req.contains("head_max_len") && !req["head_max_len"].is_null()
                                     ? req["head_max_len"].get<int32_t>()
                                     : m.cfg.head_max_len;
    const Json state = req.contains("state") ? req["state"] : Json();
    const Json questions = req.contains("questions") ? req["questions"] : Json();

    std::unordered_map<std::string, laya::Question> internal;
    for (auto it = questions.begin(); it != questions.end(); ++it) {
        laya::check_question(it.key(), it.value());
        internal.emplace(it.key(), laya::to_internal(it.value()));
    }

    const std::vector<laya::SequenceItem> items = laya::encode_state(m.tok, state, questions, max_len, head_max_len);

    Json answers = Json::object();
    int64_t input_tokens = 0;
    for (const auto& item : items) {
        const laya::Question& q = internal.at(item.qid);
        answers[item.qid] = laya::predict_one(m.weights, m.enc_cfg, m.temps, q, item);
        input_tokens += static_cast<int64_t>(item.ids.size());
    }

    Json usage;
    usage["input_tokens"] = input_tokens;
    usage["output_tokens"] = 0;

    Json out;
    out["model"] = "laya-rl-agent";
    out["answers"] = answers;
    out["usage"] = usage;
    return out;
}

int usage_msg() {
    std::cerr << "usage: laya-predict --model DIR request FILE.json|-\n"
                 "       laya-predict --model DIR requests FILE.jsonl\n";
    return 2;
}

}

int main(int argc, char** argv) {
    if (argc != 5 || std::string(argv[1]) != "--model") {
        return usage_msg();
    }
    const std::string mode = argv[3];
    const std::string arg = argv[4];
    try {
        const Model m = load_model(argv[2]);
        if (mode == "request") {
            std::cout << run_request(m, Json::parse(slurp(arg))).dump() << "\n";
        } else if (mode == "requests") {
            std::istringstream lines(slurp(arg));
            std::string line;
            while (std::getline(lines, line)) {
                if (line.empty()) {
                    continue;
                }
                Json out;
                try {
                    out = run_request(m, Json::parse(line));
                } catch (const std::exception& e) {
                    out = Json{{"error", e.what()}};
                }
                std::cout << out.dump() << "\n";
            }
        } else {
            return usage_msg();
        }
    } catch (const std::exception& e) {
        std::cerr << "laya-predict: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
