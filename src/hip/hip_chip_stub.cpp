#include "laya/hip_chip.hpp"

#include <stdexcept>

namespace laya {

struct HipChip::Impl {};

HipChip::HipChip(const SafetensorsFile&, const ModernBertConfig&) {
    throw std::runtime_error("chip 'hip': not built (rebuild with -DLAYA_HIP=ON on a machine with ROCm installed)");
}

HipChip::~HipChip() = default;

py::Json HipChip::compute(const TemperatureConfig&, const Question&, const SequenceItem&) const {
    throw std::runtime_error("chip 'hip': not built (rebuild with -DLAYA_HIP=ON on a machine with ROCm installed)");
}

}
