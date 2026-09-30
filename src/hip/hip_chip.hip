#include "laya/hip_chip.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <stdexcept>
#include <string>

#include "hip_modernbert.hpp"
#include "hip_weights.hpp"

namespace laya {

namespace {

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("chip 'hip': ") + what + ": " + hipGetErrorString(err));
    }
}

void check_hipblas(hipblasStatus_t status, const char* what) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("chip 'hip': ") + what + " failed");
    }
}

}

struct HipChip::Impl {
    hipblasHandle_t handle = nullptr;
    const SafetensorsFile& host_weights;
    HipWeights weights;
    ModernBertConfig enc_cfg;
    HipEncoderScratch scratch;

    Impl(const SafetensorsFile& sf, const ModernBertConfig& cfg) : host_weights(sf), weights(sf), enc_cfg(cfg) {}
};

HipChip::HipChip(const SafetensorsFile& weights, const ModernBertConfig& enc_cfg) {
    int count = 0;
    check_hip(hipGetDeviceCount(&count), "hipGetDeviceCount");
    if (count == 0) {
        throw std::runtime_error("chip 'hip': no ROCm device found");
    }
    check_hip(hipSetDevice(0), "hipSetDevice");

    impl_ = std::make_unique<Impl>(weights, enc_cfg);
    check_hipblas(hipblasCreate(&impl_->handle), "hipblasCreate");
}

HipChip::~HipChip() {
    if (impl_ && impl_->handle != nullptr) {
        hipblasDestroy(impl_->handle);
    }
}

py::Json HipChip::compute(const TemperatureConfig& temps, const Question& q, const SequenceItem& item) const {
    HipTensor t = hip_modernbert_encoder(impl_->handle, impl_->weights, impl_->enc_cfg, item.ids, impl_->scratch);
    t = hip_add_type_emb_and_head(impl_->handle, impl_->weights, impl_->enc_cfg, t, item.qtype, impl_->scratch);

    const std::vector<float> host_hidden = hip_download(t);

    return decode_from_hidden(impl_->host_weights, temps, q, item, host_hidden,
                              static_cast<size_t>(impl_->enc_cfg.hidden_size));
}

}
