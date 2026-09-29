#include "laya/cuda_chip.hpp"

#include <stdexcept>

namespace laya {

CudaChip::CudaChip(const SafetensorsFile&, const ModernBertConfig&) {}

py::Json CudaChip::compute(const TemperatureConfig&, const Question&, const SequenceItem&) const {
    throw std::runtime_error("chip 'cuda': not implemented yet");
}

}
