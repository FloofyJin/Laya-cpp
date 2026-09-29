#pragma once

#include <memory>

#include "laya/chip_type.hpp"

namespace laya {

class HipChip : public ChipType {
public:
    HipChip(const SafetensorsFile& weights, const ModernBertConfig& enc_cfg);
    ~HipChip() override;

    py::Json compute(const TemperatureConfig& temps, const Question& q, const SequenceItem& item) const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
