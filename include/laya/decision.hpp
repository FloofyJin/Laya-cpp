#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include "laya/modernbert.hpp"
#include "laya/pyjson.hpp"
#include "laya/safetensors.hpp"
#include "laya/sequence.hpp"

namespace laya {

struct TemperatureConfig {
    double temperature[3] = {1.0, 1.0, 1.0};
    std::unordered_map<std::string, double> temperature_by_options;

    static TemperatureConfig from_file(const std::string& path);
    double resolve(const std::string& type_name, int32_t qtype, int32_t option_count) const;
};

py::Json predict_one(const SafetensorsFile& weights, const ModernBertConfig& enc_cfg,
                     const TemperatureConfig& temps, const Question& q, const SequenceItem& item);

}
