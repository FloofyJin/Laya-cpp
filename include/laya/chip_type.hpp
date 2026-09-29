#pragma once

#include <memory>
#include <string>

#include "laya/decision.hpp"
#include "laya/modernbert.hpp"
#include "laya/pyjson.hpp"
#include "laya/safetensors.hpp"
#include "laya/sequence.hpp"

namespace laya {

enum class Chip { Cpu, Hip, Cuda };

Chip parse_chip(const std::string& name);
std::string chip_name(Chip chip);

class ChipType {
public:
    virtual ~ChipType() = default;
    virtual py::Json compute(const TemperatureConfig& temps, const Question& q, const SequenceItem& item) const = 0;
};

std::unique_ptr<ChipType> make_chip_type(Chip chip, const SafetensorsFile& weights, const ModernBertConfig& enc_cfg);

}
