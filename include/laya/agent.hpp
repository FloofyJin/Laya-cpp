#pragma once

#include <string>

#include "laya/decision.hpp"
#include "laya/modernbert.hpp"
#include "laya/pyjson.hpp"
#include "laya/safetensors.hpp"
#include "laya/sequence.hpp"
#include "laya/tokenizer.hpp"

namespace laya {

class Agent {
public:
    explicit Agent(const std::string& model_dir);

    py::Json predict(const py::Json& request) const;

private:
    Tokenizer tok_;
    AgentConfig cfg_;
    SafetensorsFile weights_;
    ModernBertConfig enc_cfg_;
    TemperatureConfig temps_;
};

}
