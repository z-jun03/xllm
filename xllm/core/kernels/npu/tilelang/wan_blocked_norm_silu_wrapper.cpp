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

#include <ATen/ops/linalg_vector_norm.h>
#include <acl/acl_base.h>
#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_WAN_BLOCKED_NORM_SILU_REGISTRY_INC
#error "XLLM_TL_WAN_BLOCKED_NORM_SILU_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kChannels = 96;
constexpr int64_t kTemporal = 4;
constexpr int64_t kPlaneElements = 512 * 512;

#include XLLM_TL_WAN_BLOCKED_NORM_SILU_REGISTRY_INC

WanBlockedNormSiluSpecialization build_runtime_specialization(
    const torch::Tensor& input) {
  return make_wan_blocked_norm_silu_specialization(
      WanBlockedNormSiluChannels{static_cast<int32_t>(input.size(1))},
      WanBlockedNormSiluTemporal{static_cast<int32_t>(input.size(2))},
      WanBlockedNormSiluPlaneElements{
          static_cast<int32_t>(input.size(3) * input.size(4))},
      WanBlockedNormSiluDType{to_tilelang_dtype(input.scalar_type())});
}

bool has_supported_input_contract(const torch::Tensor& input,
                                  const torch::Tensor& gamma) {
  return input.defined() && gamma.defined() &&
         input.device().type() == c10::DeviceType::PrivateUse1 &&
         input.device() == gamma.device() &&
         input.scalar_type() == torch::kBFloat16 &&
         gamma.scalar_type() == input.scalar_type() && input.dim() == 5 &&
         input.size(0) == 1 && input.size(1) == kChannels &&
         input.size(2) == kTemporal &&
         input.size(3) * input.size(4) == kPlaneElements &&
         input.is_contiguous() && gamma.numel() == kChannels &&
         gamma.is_contiguous() &&
         at_npu::native::get_npu_format(input) == ACL_FORMAT_NDC1HWC0;
}

}  // namespace

bool can_wan_blocked_norm_silu(const torch::Tensor& input,
                               const torch::Tensor& gamma) {
  if (!has_supported_input_contract(input, gamma)) {
    return false;
  }
  return find_wan_blocked_norm_silu_kernel_entry(
             build_runtime_specialization(input)) != nullptr;
}

torch::Tensor wan_blocked_norm_silu(const torch::Tensor& input,
                                    const torch::Tensor& gamma) {
  CHECK(can_wan_blocked_norm_silu(input, gamma))
      << "TileLang wan_blocked_norm_silu: unsupported tensor contract; input="
      << input.sizes() << ", gamma=" << gamma.sizes();

  const torch::Tensor norm = torch::linalg_vector_norm(
      input, 2.0, {1}, true, torch::kFloat32);
  CHECK(norm.is_contiguous());
  CHECK_EQ(at_npu::native::get_npu_format(norm), ACL_FORMAT_NCDHW);

  torch::Tensor output = at_npu::native::empty_with_format(
      input.sizes(), input.options(), ACL_FORMAT_NDC1HWC0);
  const WanBlockedNormSiluSpecialization specialization =
      build_runtime_specialization(input);
  const auto* entry =
      find_wan_blocked_norm_silu_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang wan_blocked_norm_silu: no compiled variant. Available "
         "variants: "
      << available_wan_blocked_norm_silu_variant_keys();

  aclrtStream stream =
      c10_npu::getCurrentNPUStream(input.device().index()).stream();
  entry->fn(
      reinterpret_cast<uint8_t*>(const_cast<void*>(input.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(norm.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(gamma.data_ptr())),
      reinterpret_cast<uint8_t*>(output.data_ptr()),
      stream);
  return output;
}

}  // namespace xllm::kernel::npu::tilelang
