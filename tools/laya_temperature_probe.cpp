#include <iostream>
#include <map>
#include <string>

#include "laya/decision.hpp"
#include "laya/pyjson.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: laya-temperature-probe CONFIG.json\n";
        return 2;
    }
    laya::py::Json out;
    try {
        const laya::TemperatureConfig cfg = laya::TemperatureConfig::from_file(argv[1]);
        out["temperature"] = {cfg.temperature[0], cfg.temperature[1], cfg.temperature[2]};
        const std::map<std::string, double> sorted(cfg.temperature_by_options.begin(),
                                                   cfg.temperature_by_options.end());
        out["temperature_by_options"] = laya::py::Json::object();
        for (const auto& [key, value] : sorted) {
            out["temperature_by_options"][key] = value;
        }
    } catch (const std::exception& e) {
        out = laya::py::Json{{"error", e.what()}};
    }
    std::cout << out.dump() << "\n";
    return 0;
}
