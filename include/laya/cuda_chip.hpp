#pragma once

#include "laya/chip_type.hpp"

namespace laya {

class CudaChip : public ChipType {
public:
    CudaChip(const SafetensorsFile& weights, const ModernBertConfig& enc_cfg);

    py::Json compute(const TemperatureConfig& temps, const Question& q, const SequenceItem& item) const override;
};

}
