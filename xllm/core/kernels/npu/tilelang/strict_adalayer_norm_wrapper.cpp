/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_STRICT_ADALAYER_NORM_REGISTRY_INC
#error "XLLM_TL_STRICT_ADALAYER_NORM_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kHiddenSize = 3072;

#include XLLM_TL_STRICT_ADALAYER_NORM_REGISTRY_INC

bool fits_int32(int64_t value) {
  return value > 0 &&
         value <= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
}

StrictAdalayerNormSpecialization build_runtime_specialization(
    const torch::Tensor& input) {
  return make_strict_adalayer_norm_specialization(
      StrictAdalayerNormHiddenSize{static_cast<int32_t>(input.size(2))},
      StrictAdalayerNormDType{to_tilelang_dtype(input.scalar_type())});
}

bool have_matching_modulation_contract(const torch::Tensor& modulation,
                                       const torch::Tensor& input) {
  return modulation.defined() && modulation.device() == input.device() &&
         modulation.scalar_type() == input.scalar_type() &&
         modulation.dim() == 2 && modulation.size(0) == input.size(0) &&
         modulation.size(1) == input.size(2) && modulation.stride(1) == 1 &&
         modulation.stride(0) >= modulation.size(1);
}

}  // namespace

bool can_strict_adalayer_norm(const torch::Tensor& input,
                              const torch::Tensor& scale,
                              const torch::Tensor& shift) {
  if (!input.defined() ||
      input.device().type() != c10::DeviceType::PrivateUse1 ||
      input.scalar_type() != torch::kBFloat16 || input.dim() != 3 ||
      input.size(0) <= 0 || input.size(1) <= 0 ||
      input.size(2) != kHiddenSize || !input.is_contiguous() ||
      !have_matching_modulation_contract(scale, input) ||
      !have_matching_modulation_contract(shift, input) ||
      scale.strides() != shift.strides()) {
    return false;
  }

  const int64_t row_count = input.size(0) * input.size(1);
  const std::array<int64_t, 3> runtime_values = {
      row_count, input.size(1), scale.stride(0)};
  for (int64_t value : runtime_values) {
    if (!fits_int32(value)) {
      return false;
    }
  }

  return find_strict_adalayer_norm_kernel_entry(
             build_runtime_specialization(input)) != nullptr;
}

torch::Tensor strict_adalayer_norm(const torch::Tensor& input,
                                   const torch::Tensor& scale,
                                   const torch::Tensor& shift,
                                   double eps) {
  CHECK(can_strict_adalayer_norm(input, scale, shift))
      << "TileLang strict AdaLayerNorm: unsupported tensor contract";

  torch::Tensor output = torch::empty_like(input);
  const StrictAdalayerNormSpecialization specialization =
      build_runtime_specialization(input);
  const auto* entry =
      find_strict_adalayer_norm_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang strict AdaLayerNorm: no compiled variant. Available "
         "variants: "
      << available_strict_adalayer_norm_variant_keys();

  const int64_t modulation_capacity =
      (scale.size(0) - 1) * scale.stride(0) + scale.size(1);
  const int64_t buffer_capacity =
      std::max(input.numel(), modulation_capacity);
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(input.device().index()).stream();
  entry->fn(
      reinterpret_cast<uint8_t*>(const_cast<void*>(input.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(scale.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(shift.data_ptr())),
      reinterpret_cast<uint8_t*>(output.data_ptr()),
      static_cast<int32_t>(input.size(0) * input.size(1)),
      static_cast<int32_t>(input.size(1)),
      static_cast<int32_t>(scale.stride(0)),
      static_cast<float>(eps),
      buffer_capacity,
      stream);
  return output;
}

}  // namespace xllm::kernel::npu::tilelang
