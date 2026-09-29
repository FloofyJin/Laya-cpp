#pragma once

#include <memory>
#include <string>

#include "laya/chip_type.hpp"
#include "laya/decision.hpp"
#include "laya/modernbert.hpp"
#include "laya/pyjson.hpp"
#include "laya/safetensors.hpp"
#include "laya/sequence.hpp"
#include "laya/tokenizer.hpp"

namespace laya {

class Agent {
public:
    explicit Agent(const std::string& model_dir, Chip chip = Chip::Cpu);

    py::Json predict(const py::Json& request) const;

private:
    Tokenizer tok_;
    AgentConfig cfg_;
    SafetensorsFile weights_;
    ModernBertConfig enc_cfg_;
    TemperatureConfig temps_;
    std::unique_ptr<ChipType> chip_;
};

}
