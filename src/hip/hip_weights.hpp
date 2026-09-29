#pragma once

#include <string>
#include <unordered_map>

#include "laya/safetensors.hpp"

namespace laya {

class HipWeights {
public:
    explicit HipWeights(const SafetensorsFile& weights);
    ~HipWeights();
    HipWeights(const HipWeights&) = delete;
    HipWeights& operator=(const HipWeights&) = delete;

    const float* get(const std::string& name) const;

private:
    std::unordered_map<std::string, float*> device_ptrs_;
};

}
