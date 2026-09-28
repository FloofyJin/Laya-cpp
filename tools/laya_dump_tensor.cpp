#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "laya/pyjson.hpp"
#include "laya/safetensors.hpp"

namespace {

using laya::py::Json;

std::string slurp(const std::string& path) {
    if (path == "-") {
        std::ostringstream ss;
        ss << std::cin.rdbuf();
        return ss.str();
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string dtype_name(laya::DType dt) {
    switch (dt) {
        case laya::DType::F32: return "F32";
        case laya::DType::F16: return "F16";
        case laya::DType::BF16: return "BF16";
        case laya::DType::I64: return "I64";
        case laya::DType::I32: return "I32";
        case laya::DType::I8: return "I8";
        case laya::DType::U8: return "U8";
        case laya::DType::Bool: return "BOOL";
    }
    throw std::runtime_error("unknown dtype");
}

nlohmann::json names_json(const laya::SafetensorsFile& f) {
    nlohmann::json out = nlohmann::json::array();
    for (const auto& name : f.names()) {
        const laya::TensorInfo& t = f.info(name);
        out.push_back({{"name", name}, {"dtype", dtype_name(t.dtype)}, {"shape", t.shape}});
    }
    return {{"names", out}};
}

nlohmann::json values_json(const laya::SafetensorsFile& f, const Json& req) {
    const std::string name = req.at("name").get<std::string>();
    const laya::TensorInfo& t = f.info(name);
    const size_t numel = t.numel();
    const size_t start = req.contains("start") && !req["start"].is_null() ? req["start"].get<size_t>() : 0;
    if (start > numel) {
        throw std::runtime_error("start " + std::to_string(start) + " exceeds tensor size " +
                                  std::to_string(numel));
    }
    const size_t remaining = numel - start;
    const size_t count = req.contains("count") && !req["count"].is_null()
                              ? std::min(req["count"].get<size_t>(), remaining)
                              : remaining;

    const std::vector<float> values = f.as_f32_slice(name, start, count);
    nlohmann::json arr = nlohmann::json::array();
    for (float v : values) {
        arr.push_back(v);
    }
    return {{"name", name}, {"dtype", dtype_name(t.dtype)}, {"shape", t.shape},
            {"start", start}, {"count", count}, {"values", arr}};
}

int usage() {
    std::cerr << "usage: laya-dump-tensor --model DIR names FILE.jsonl\n"
                 "       laya-dump-tensor --model DIR values FILE.jsonl\n";
    return 2;
}

}

int main(int argc, char** argv) {
    if (argc != 5 || std::string(argv[1]) != "--model") {
        return usage();
    }
    const std::string dir = argv[2];
    const std::string mode = argv[3];
    const std::string arg = argv[4];
    try {
        laya::SafetensorsFile f = laya::SafetensorsFile::open(dir + "/model.safetensors");

        std::istringstream lines(slurp(arg));
        std::string line;
        while (std::getline(lines, line)) {
            if (line.empty()) {
                continue;
            }
            nlohmann::json out;
            try {
                const Json v = Json::parse(line);
                if (mode == "names") {
                    out = names_json(f);
                } else if (mode == "values") {
                    out = values_json(f, v);
                } else {
                    throw std::runtime_error("unknown mode " + mode);
                }
            } catch (const std::exception& e) {
                out = {{"error", e.what()}};
            }
            std::cout << out.dump() << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "laya-dump-tensor: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
