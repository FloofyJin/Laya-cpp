#include "laya/cpu_chip.hpp"

namespace laya {

CpuChip::CpuChip(const SafetensorsFile& weights, const ModernBertConfig& enc_cfg)
    : weights_(weights), enc_cfg_(enc_cfg) {}

py::Json CpuChip::compute(const TemperatureConfig& temps, const Question& q, const SequenceItem& item) const {
    return predict_one(weights_, enc_cfg_, temps, q, item);
}

}
