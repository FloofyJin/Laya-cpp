#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace laya {

enum class DType { F32, F16, BF16, I64, I32, I8, U8, Bool };

struct TensorInfo {
    std::string name;
    DType dtype = DType::F32;
    std::vector<int64_t> shape;
    size_t begin = 0;
    size_t end = 0;

    size_t numel() const;
    size_t nbytes() const { return end - begin; }
};

class SafetensorsError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class SafetensorsFile {
public:
    static SafetensorsFile open(const std::string& path);

    SafetensorsFile(SafetensorsFile&& other) noexcept;
    SafetensorsFile& operator=(SafetensorsFile&& other) noexcept;
    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;
    ~SafetensorsFile();

    const std::vector<std::string>& names() const { return names_; }
    bool contains(const std::string& name) const;
    const TensorInfo& info(const std::string& name) const;

    const uint8_t* raw(const std::string& name) const;
    std::vector<float> as_f32(const std::string& name) const;
    std::vector<float> as_f32_slice(const std::string& name, size_t start, size_t count) const;

private:
    SafetensorsFile() = default;

    int fd_ = -1;
    uint8_t* map_ = nullptr;
    size_t map_size_ = 0;
    size_t data_start_ = 0;

    std::vector<std::string> names_;
    std::unordered_map<std::string, TensorInfo> tensors_;
};

float f16_to_f32(uint16_t h);
float bf16_to_f32(uint16_t b);

}
