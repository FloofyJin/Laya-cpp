#pragma once

#include "laya/chip_type.hpp"

namespace laya {

class CpuChip : public ChipType {
public:
    CpuChip(const SafetensorsFile& weights, const ModernBertConfig& enc_cfg);

    py::Json compute(const TemperatureConfig& temps, const Question& q, const SequenceItem& item) const override;

private:
    const SafetensorsFile& weights_;
    const ModernBertConfig& enc_cfg_;
};

}
