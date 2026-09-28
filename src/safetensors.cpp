#include "laya/safetensors.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

namespace laya {

namespace {

DType dtype_from_string(const std::string& s) {
    if (s == "F32") return DType::F32;
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "I64") return DType::I64;
    if (s == "I32") return DType::I32;
    if (s == "I8") return DType::I8;
    if (s == "U8") return DType::U8;
    if (s == "BOOL") return DType::Bool;
    throw SafetensorsError("safetensors: unsupported dtype '" + s + "'");
}

size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::BF16: return 2;
        case DType::I64: return 8;
        case DType::I32: return 4;
        case DType::I8: return 1;
        case DType::U8: return 1;
        case DType::Bool: return 1;
    }
    throw SafetensorsError("safetensors: unknown dtype size");
}

}

size_t TensorInfo::numel() const {
    size_t n = 1;
    for (int64_t d : shape) {
        n *= static_cast<size_t>(d);
    }
    return n;
}

float f16_to_f32(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            uint32_t e = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x3FFu;
            bits = sign | (e << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float bf16_to_f32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

SafetensorsFile SafetensorsFile::open(const std::string& path) {
    SafetensorsFile f;
    f.fd_ = ::open(path.c_str(), O_RDONLY);
    if (f.fd_ < 0) {
        throw SafetensorsError("safetensors: cannot open '" + path + "': " + std::strerror(errno));
    }
    struct stat st{};
    if (fstat(f.fd_, &st) != 0) {
        ::close(f.fd_);
        throw SafetensorsError("safetensors: fstat failed for '" + path + "'");
    }
    f.map_size_ = static_cast<size_t>(st.st_size);
    if (f.map_size_ < 8) {
        ::close(f.fd_);
        throw SafetensorsError("safetensors: '" + path + "' is too small to be a valid safetensors file");
    }
    void* m = mmap(nullptr, f.map_size_, PROT_READ, MAP_PRIVATE, f.fd_, 0);
    if (m == MAP_FAILED) {
        ::close(f.fd_);
        throw SafetensorsError("safetensors: mmap failed for '" + path + "'");
    }
    f.map_ = static_cast<uint8_t*>(m);

    uint64_t header_len = 0;
    std::memcpy(&header_len, f.map_, sizeof(header_len));
    if (header_len == 0 || 8 + header_len > f.map_size_) {
        munmap(f.map_, f.map_size_);
        ::close(f.fd_);
        throw SafetensorsError("safetensors: '" + path + "' has an invalid header length");
    }
    f.data_start_ = 8 + static_cast<size_t>(header_len);

    std::string header_str(reinterpret_cast<const char*>(f.map_ + 8), static_cast<size_t>(header_len));
    nlohmann::json header;
    try {
        header = nlohmann::json::parse(header_str);
    } catch (const std::exception& e) {
        munmap(f.map_, f.map_size_);
        ::close(f.fd_);
        throw SafetensorsError(std::string("safetensors: header is not valid JSON: ") + e.what());
    }

    for (auto it = header.begin(); it != header.end(); ++it) {
        if (it.key() == "__metadata__") {
            continue;
        }
        const nlohmann::json& v = it.value();
        TensorInfo info;
        info.name = it.key();
        info.dtype = dtype_from_string(v.at("dtype").get<std::string>());
        for (const auto& d : v.at("shape")) {
            info.shape.push_back(d.get<int64_t>());
        }
        const auto& offs = v.at("data_offsets");
        info.begin = offs.at(0).get<size_t>();
        info.end = offs.at(1).get<size_t>();

        if (info.end < info.begin || f.data_start_ + info.end > f.map_size_) {
            munmap(f.map_, f.map_size_);
            ::close(f.fd_);
            throw SafetensorsError("safetensors: tensor '" + info.name + "' data_offsets out of range");
        }
        const size_t expected = info.numel() * dtype_size(info.dtype);
        if (info.end - info.begin != expected) {
            munmap(f.map_, f.map_size_);
            ::close(f.fd_);
            throw SafetensorsError("safetensors: tensor '" + info.name + "' size mismatch (offsets imply " +
                                    std::to_string(info.end - info.begin) + " bytes, shape+dtype imply " +
                                    std::to_string(expected) + " bytes)");
        }

        f.names_.push_back(info.name);
        f.tensors_.emplace(info.name, std::move(info));
    }
    std::sort(f.names_.begin(), f.names_.end());
    return f;
}

SafetensorsFile::SafetensorsFile(SafetensorsFile&& other) noexcept
    : fd_(other.fd_), map_(other.map_), map_size_(other.map_size_), data_start_(other.data_start_),
      names_(std::move(other.names_)), tensors_(std::move(other.tensors_)) {
    other.fd_ = -1;
    other.map_ = nullptr;
    other.map_size_ = 0;
}

SafetensorsFile& SafetensorsFile::operator=(SafetensorsFile&& other) noexcept {
    if (this != &other) {
        if (map_) {
            munmap(map_, map_size_);
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
        fd_ = other.fd_;
        map_ = other.map_;
        map_size_ = other.map_size_;
        data_start_ = other.data_start_;
        names_ = std::move(other.names_);
        tensors_ = std::move(other.tensors_);
        other.fd_ = -1;
        other.map_ = nullptr;
        other.map_size_ = 0;
    }
    return *this;
}

SafetensorsFile::~SafetensorsFile() {
    if (map_) {
        munmap(map_, map_size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

bool SafetensorsFile::contains(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

const TensorInfo& SafetensorsFile::info(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw SafetensorsError("safetensors: no such tensor '" + name + "'");
    }
    return it->second;
}

const uint8_t* SafetensorsFile::raw(const std::string& name) const {
    const TensorInfo& t = info(name);
    return map_ + data_start_ + t.begin;
}

std::vector<float> SafetensorsFile::as_f32_slice(const std::string& name, size_t start, size_t count) const {
    const TensorInfo& t = info(name);
    const size_t n = t.numel();
    if (start > n || count > n - start) {
        throw SafetensorsError("safetensors: slice [" + std::to_string(start) + ", " +
                                std::to_string(start + count) + ") out of range for tensor '" + name +
                                "' with " + std::to_string(n) + " elements");
    }
    const uint8_t* p = map_ + data_start_ + t.begin;
    std::vector<float> out(count);
    switch (t.dtype) {
        case DType::F32: {
            std::memcpy(out.data(), p + start * sizeof(float), count * sizeof(float));
            break;
        }
        case DType::F16: {
            std::vector<uint16_t> raw16(count);
            std::memcpy(raw16.data(), p + start * sizeof(uint16_t), count * sizeof(uint16_t));
            for (size_t i = 0; i < count; ++i) {
                out[i] = f16_to_f32(raw16[i]);
            }
            break;
        }
        case DType::BF16: {
            std::vector<uint16_t> raw16(count);
            std::memcpy(raw16.data(), p + start * sizeof(uint16_t), count * sizeof(uint16_t));
            for (size_t i = 0; i < count; ++i) {
                out[i] = bf16_to_f32(raw16[i]);
            }
            break;
        }
        default:
            throw SafetensorsError("safetensors: as_f32 unsupported for tensor '" + name +
                                    "' (integer/bool dtype)");
    }
    return out;
}

std::vector<float> SafetensorsFile::as_f32(const std::string& name) const {
    return as_f32_slice(name, 0, info(name).numel());
}

}
