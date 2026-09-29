#include "hip_weights.hpp"

#include <stdexcept>
#include <vector>

#include <hip/hip_runtime.h>

namespace laya {

HipWeights::HipWeights(const SafetensorsFile& weights) {
    for (const std::string& name : weights.names()) {
        if (name == "temperature") {
            continue;
        }
        const std::vector<float> host = weights.as_f32(name);
        float* d_ptr = nullptr;
        if (hipMalloc(&d_ptr, host.size() * sizeof(float)) != hipSuccess) {
            throw std::runtime_error("HipWeights: hipMalloc failed for '" + name + "'");
        }
        if (hipMemcpy(d_ptr, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) {
            (void)hipFree(d_ptr);
            throw std::runtime_error("HipWeights: hipMemcpy failed for '" + name + "'");
        }
        device_ptrs_[name] = d_ptr;
    }
}

HipWeights::~HipWeights() {
    for (auto& [name, ptr] : device_ptrs_) {
        (void)hipFree(ptr);
    }
}

const float* HipWeights::get(const std::string& name) const {
    const auto it = device_ptrs_.find(name);
    if (it == device_ptrs_.end()) {
        throw std::runtime_error("HipWeights: missing tensor '" + name + "'");
    }
    return it->second;
}

}
