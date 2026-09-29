#include "laya/chip_type.hpp"

#include <stdexcept>

#include "laya/cpu_chip.hpp"
#include "laya/cuda_chip.hpp"
#include "laya/hip_chip.hpp"

namespace laya {

Chip parse_chip(const std::string& name) {
    if (name == "cpu") {
        return Chip::Cpu;
    }
    if (name == "hip") {
        return Chip::Hip;
    }
    if (name == "cuda") {
        return Chip::Cuda;
    }
    throw std::runtime_error("unknown chip '" + name + "'; expected cpu, hip, or cuda");
}

std::string chip_name(Chip chip) {
    switch (chip) {
        case Chip::Cpu:
            return "cpu";
        case Chip::Hip:
            return "hip";
        case Chip::Cuda:
            return "cuda";
    }
    throw std::runtime_error("unknown chip enum value");
}

std::unique_ptr<ChipType> make_chip_type(Chip chip, const SafetensorsFile& weights, const ModernBertConfig& enc_cfg) {
    switch (chip) {
        case Chip::Cpu:
            return std::make_unique<CpuChip>(weights, enc_cfg);
        case Chip::Hip:
            return std::make_unique<HipChip>(weights, enc_cfg);
        case Chip::Cuda:
            return std::make_unique<CudaChip>(weights, enc_cfg);
    }
    throw std::runtime_error("unknown chip enum value");
}

}
