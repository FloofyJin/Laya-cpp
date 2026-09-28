#include "laya/agent.hpp"

#include <fstream>
#include <unordered_map>

namespace laya {

namespace {

bool file_exists(const std::string& path) {
    std::ifstream in(path);
    return static_cast<bool>(in);
}

}

Agent::Agent(const std::string& model_dir)
    : tok_(Tokenizer::from_directory(file_exists(model_dir + "/tokenizer/tokenizer.json") ? model_dir + "/tokenizer"
                                                                                           : model_dir)),
      weights_(SafetensorsFile::open(model_dir + "/model.safetensors")),
      enc_cfg_(ModernBertConfig::from_file(model_dir + "/encoder/config.json")),
      temps_(TemperatureConfig::from_file(model_dir + "/rl_agent_config.json")) {
    if (file_exists(model_dir + "/rl_agent_config.json")) {
        cfg_ = AgentConfig::from_file(model_dir + "/rl_agent_config.json");
    }
}

py::Json Agent::predict(const py::Json& request) const {
    const int32_t max_len =
        request.contains("max_len") && !request["max_len"].is_null() ? request["max_len"].get<int32_t>() : cfg_.max_len;
    const int32_t head_max_len = request.contains("head_max_len") && !request["head_max_len"].is_null()
                                     ? request["head_max_len"].get<int32_t>()
                                     : cfg_.head_max_len;
    const py::Json state = request.contains("state") ? request["state"] : py::Json();
    const py::Json questions = request.contains("questions") ? request["questions"] : py::Json();

    std::unordered_map<std::string, Question> internal;
    for (auto it = questions.begin(); it != questions.end(); ++it) {
        check_question(it.key(), it.value());
        internal.emplace(it.key(), to_internal(it.value()));
    }

    const std::vector<SequenceItem> items = encode_state(tok_, state, questions, max_len, head_max_len);

    py::Json answers = py::Json::object();
    int64_t input_tokens = 0;
    for (const auto& item : items) {
        const Question& q = internal.at(item.qid);
        answers[item.qid] = predict_one(weights_, enc_cfg_, temps_, q, item);
        input_tokens += static_cast<int64_t>(item.ids.size());
    }

    py::Json usage;
    usage["input_tokens"] = input_tokens;
    usage["output_tokens"] = 0;

    py::Json out;
    out["model"] = "laya-rl-agent";
    out["answers"] = answers;
    out["usage"] = usage;
    return out;
}

}
