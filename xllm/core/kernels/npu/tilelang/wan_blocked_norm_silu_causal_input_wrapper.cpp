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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_WAN_BLOCKED_NORM_SILU_CAUSAL_INPUT_REGISTRY_INC
#error "XLLM_TL_WAN_BLOCKED_NORM_SILU_CAUSAL_INPUT_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kMinimumTemporal = 1;
constexpr int64_t kCacheTemporal = 2;
constexpr int64_t kMinimumPlaneElements = 16;

// Normalization floor; must match NORM_EPS in the TileLang kernel.
constexpr double kNormalizationEps = 1e-12;
// Tile size of the first (main) kernel launch; must match
// CAUSAL_SPATIAL_TILES[0] in the TileLang kernel module.
constexpr int64_t kSpatialTileMain = 4096;
// Test/debug knob; "fused" forces the TileLang kernel, "eager" forces the
// reference chain, any other value falls back to automatic dispatch.
constexpr char kWanBlockedNormSiluDispatchEnv[] =
    "XLLM_TL_WAN_BLOCKED_NORM_SILU_DISPATCH";
// For tiny, channel-heavy planes the spatial tiling fragments into small
// tail tiles and the fused kernel's per-(channel, temporal) task-launch
// fixed overhead dominates, so such shapes dispatch to the eager reference
// chain. Calibrated on Ascend 910_9382 (48 vector cores, Sep 2026), where
// "work per task" is plane_elements / tiles-per-(channel-temporal):
//   >=1024 with a single tile (32x32): 640ch fused 1.70x       keep
//   1408 (65x65): 640ch cache1 1.09x / cache2 0.95x             cache2 falls back
//   1035 (45x46): 640ch 0.79x, 384ch 1.03x                      640ch falls back
//   544 (33x33): 640ch ~1.0x, 512ch 1.10x, 1024ch 0.81x         640ch+ falls back
constexpr int32_t kFallbackWorkPerTask = 1100;
constexpr int32_t kFallbackChannels = 640;
constexpr int32_t kFallbackCachedWorkPerTask = 1500;
constexpr int32_t kFallbackCachedChannels = 512;

#include XLLM_TL_WAN_BLOCKED_NORM_SILU_CAUSAL_INPUT_REGISTRY_INC

int64_t get_npu_format(const torch::Tensor& tensor) {
  return at_npu::native::get_npu_format(tensor);
}

int32_t select_spatial_tile(int64_t plane_elements) {
  if (plane_elements >= 4096) {
    return 4096;
  }
  if (plane_elements >= 2048) {
    return 2048;
  }
  if (plane_elements >= 1024) {
    return 1024;
  }
  if (plane_elements >= 512) {
    return 512;
  }
  if (plane_elements >= 256) {
    return 256;
  }
  if (plane_elements >= 128) {
    return 128;
  }
  if (plane_elements >= 64) {
    return 64;
  }
  if (plane_elements >= 32) {
    return 32;
  }
  return 16;
}

enum class WanBlockedNormSiluDispatchMode { AUTO, FORCE_FUSED, FORCE_EAGER };

WanBlockedNormSiluDispatchMode wan_blocked_norm_silu_dispatch_mode() {
  const char* raw = std::getenv(kWanBlockedNormSiluDispatchEnv);
  if (raw == nullptr) {
    return WanBlockedNormSiluDispatchMode::AUTO;
  }
  const std::string value(raw);
  if (value == "fused") {
    return WanBlockedNormSiluDispatchMode::FORCE_FUSED;
  }
  if (value == "eager") {
    return WanBlockedNormSiluDispatchMode::FORCE_EAGER;
  }
  LOG(WARNING) << "Unknown value \"" << value << "\" for "
               << kWanBlockedNormSiluDispatchEnv
               << "; falling back to automatic dispatch.";
  return WanBlockedNormSiluDispatchMode::AUTO;
}

// Returns true when the fused TileLang kernel is expected to lose to the
// eager reference chain for this shape (see the calibration comment on the
// kFallback* constants above).
bool should_use_eager_fallback(int64_t channels,
                               int64_t cache_temporal,
                               int64_t plane_elements) {
  const int64_t main_tiles = plane_elements / kSpatialTileMain;
  const int64_t tail_elements = plane_elements % kSpatialTileMain;
  int64_t tiles = main_tiles;
  if (tail_elements > 0) {
    const int32_t spatial_tile = select_spatial_tile(tail_elements);
    tiles += (tail_elements + spatial_tile - 1) / spatial_tile;
  }
  if (tiles <= 1) {
    // One clean tile per (channel, temporal): the kernel always amortizes
    // the launches regardless of channel width.
    return false;
  }
  const int64_t work_per_task = plane_elements / tiles;
  if (work_per_task < kFallbackWorkPerTask &&
      channels >= kFallbackChannels) {
    return true;
  }
  if (cache_temporal == kCacheTemporal &&
      work_per_task < kFallbackCachedWorkPerTask &&
      channels >= kFallbackCachedChannels) {
    return true;
  }
  return false;
}

// Reference chain that the fused kernel replaces; kept bitwise-identical to
// the authoritative expectation used by the wrapper tests.
std::pair<torch::Tensor, torch::Tensor> compute_causal_input_eager(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& feature_cache) {
  const int64_t cache_temporal =
      feature_cache.defined() && feature_cache.numel() > 0
          ? feature_cache.size(2)
          : 0;
  torch::Tensor activated = torch::nn::functional::normalize(
      input.to(torch::kFloat32),
      torch::nn::functional::NormalizeFuncOptions()
          .dim(1)
          .eps(kNormalizationEps))
      .to(input.scalar_type());
  activated = activated * std::sqrt(static_cast<double>(input.size(1)));
  activated = torch::silu(activated * gamma);
  torch::Tensor combined =
      feature_cache.defined() && feature_cache.numel() > 0
          ? torch::cat({feature_cache, activated}, 2)
          : activated;
  torch::Tensor conv_input = torch::nn::functional::pad(
      combined,
      torch::nn::functional::PadFuncOptions(
          {0, 0, 0, 0, kCacheTemporal - cache_temporal, 0}));
  torch::Tensor next_cache =
      combined
          .slice(2, std::max<int64_t>(combined.size(2) - kCacheTemporal, 0))
          .clone();
  return {conv_input, next_cache};
}

WanBlockedNormSiluCausalInputSpecialization build_runtime_specialization(
    const torch::Tensor& input,
    int64_t cache_temporal,
    int32_t spatial_tile) {
  return make_wan_blocked_norm_silu_causal_input_specialization(
      WanBlockedNormSiluCausalInputCacheTemporal{
          static_cast<int32_t>(cache_temporal)},
      WanBlockedNormSiluCausalInputSpatialTile{spatial_tile},
      WanBlockedNormSiluCausalInputDType{
          to_tilelang_dtype(input.scalar_type())});
}

bool has_supported_cache_contract(const torch::Tensor& input,
                                  const torch::Tensor& feature_cache) {
  if (!feature_cache.defined() || feature_cache.numel() == 0) {
    return true;
  }
  return feature_cache.device() == input.device() &&
         feature_cache.scalar_type() == input.scalar_type() &&
         feature_cache.dim() == 5 && feature_cache.size(0) == 1 &&
         feature_cache.size(1) == input.size(1) && feature_cache.size(2) > 0 &&
         feature_cache.size(2) <= kCacheTemporal &&
         feature_cache.size(3) == input.size(3) &&
         feature_cache.size(4) == input.size(4) &&
         feature_cache.is_contiguous() &&
         get_npu_format(feature_cache) == ACL_FORMAT_NCDHW;
}
bool has_supported_input_contract(const torch::Tensor& input,
                                  const torch::Tensor& gamma) {
  if (!input.defined() || !gamma.defined() ||
      input.device().type() != c10::DeviceType::PrivateUse1 ||
      input.device() != gamma.device() ||
      input.scalar_type() != torch::kBFloat16 ||
      gamma.scalar_type() != input.scalar_type() || input.dim() != 5 ||
      input.size(0) != 1 || input.size(2) < kMinimumTemporal ||
      !input.is_contiguous() || gamma.numel() != input.size(1) ||
      !gamma.is_contiguous() || get_npu_format(input) != ACL_FORMAT_NCDHW) {
    return false;
  }
  const int64_t plane_elements = input.size(3) * input.size(4);
  return plane_elements >= kMinimumPlaneElements &&
         plane_elements <= std::numeric_limits<int32_t>::max() / input.size(1) /
                               kCacheTemporal &&
         input.size(2) <= std::numeric_limits<int32_t>::max() / input.size(1) /
                                  plane_elements -
                              kCacheTemporal;
}

}  // namespace

bool can_wan_blocked_norm_silu_causal_input(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& feature_cache) {
  if (!has_supported_input_contract(input, gamma) ||
      !has_supported_cache_contract(input, feature_cache)) {
    return false;
  }
  const int64_t cache_temporal =
      feature_cache.defined() && feature_cache.numel() > 0
          ? feature_cache.size(2)
          : 0;
  const int64_t channels = input.size(1);
  const float scale = std::sqrt(static_cast<float>(channels));
  const int64_t plane_elements = input.size(3) * input.size(4);
  const int64_t tail_elements = plane_elements % 4096;
  const int32_t tail_tile =
      select_spatial_tile(tail_elements > 0 ? tail_elements : plane_elements);
  return find_wan_blocked_norm_silu_causal_input_kernel_entry(
             build_runtime_specialization(input, cache_temporal, 4096)) !=
             nullptr &&
         find_wan_blocked_norm_silu_causal_input_kernel_entry(
             build_runtime_specialization(input, cache_temporal, tail_tile)) !=
             nullptr;
}

std::pair<torch::Tensor, torch::Tensor> wan_blocked_norm_silu_causal_input(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& feature_cache) {
  CHECK(can_wan_blocked_norm_silu_causal_input(input, gamma, feature_cache))
      << "TileLang wan_blocked_norm_silu_causal_input: unsupported contract";
  const int64_t cache_temporal =
      feature_cache.defined() && feature_cache.numel() > 0
          ? feature_cache.size(2)
          : 0;
  const int64_t channels = input.size(1);
  const int64_t temporal = input.size(2);
  const int64_t next_cache_temporal =
      std::min(kCacheTemporal, temporal + cache_temporal);
  const int64_t plane_elements = input.size(3) * input.size(4);
  const WanBlockedNormSiluDispatchMode mode =
      wan_blocked_norm_silu_dispatch_mode();
  const bool use_eager =
      mode == WanBlockedNormSiluDispatchMode::FORCE_EAGER ||
      (mode == WanBlockedNormSiluDispatchMode::AUTO &&
       should_use_eager_fallback(channels, cache_temporal, plane_elements));
  if (use_eager) {
    return compute_causal_input_eager(input, gamma, feature_cache);
  }
  const torch::Tensor norm =
      torch::linalg_vector_norm(input, 2.0, {1}, true, torch::kFloat32);
  torch::Tensor conv_input = torch::empty(
      {1, channels, temporal + kCacheTemporal, input.size(3), input.size(4)},
      input.options());
  torch::Tensor next_cache = torch::empty(
      {1, channels, next_cache_temporal, input.size(3), input.size(4)},
      input.options());
  void* cache_pointer = feature_cache.defined()
                            ? const_cast<void*>(feature_cache.data_ptr())
                            : const_cast<void*>(input.data_ptr());
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(input.device().index()).stream();
  const auto launch = [&](int32_t spatial_tile,
                          int32_t spatial_begin,
                          int32_t spatial_elements) {
    const float scale = std::sqrt(static_cast<float>(channels));
    const WanBlockedNormSiluCausalInputSpecialization specialization =
        build_runtime_specialization(input, cache_temporal, spatial_tile);
    const auto* entry =
        find_wan_blocked_norm_silu_causal_input_kernel_entry(specialization);
    CHECK(entry != nullptr);
    entry->fn(reinterpret_cast<uint8_t*>(const_cast<void*>(input.data_ptr())),
              reinterpret_cast<uint8_t*>(const_cast<void*>(norm.data_ptr())),
              reinterpret_cast<uint8_t*>(const_cast<void*>(gamma.data_ptr())),
              reinterpret_cast<uint8_t*>(cache_pointer),
              reinterpret_cast<uint8_t*>(conv_input.data_ptr()),
              reinterpret_cast<uint8_t*>(next_cache.data_ptr()),
              static_cast<int32_t>(channels),
              static_cast<int32_t>(plane_elements),
              static_cast<int32_t>(temporal),
              spatial_begin,
              spatial_elements,
              scale,
              input.numel(),
              norm.numel(),
              gamma.numel(),
              feature_cache.defined() ? feature_cache.numel() : 0,
              conv_input.numel(),
              next_cache.numel(),
              stream);
  };
  const int64_t main_elements = plane_elements / 4096 * 4096;
  if (main_elements > 0) {
    launch(/*spatial_tile=*/4096,
           /*spatial_begin=*/0,
           /*spatial_elements=*/static_cast<int32_t>(main_elements));
  }
  const int64_t tail_elements = plane_elements - main_elements;
  if (tail_elements > 0) {
    const int32_t spatial_tile = select_spatial_tile(tail_elements);
    const int64_t spatial_elements =
        std::max<int64_t>(tail_elements, spatial_tile);
    launch(spatial_tile,
           /*spatial_begin=*/
           static_cast<int32_t>(plane_elements - spatial_elements),
           /*spatial_elements=*/static_cast<int32_t>(spatial_elements));
  }
  return {conv_input, next_cache};
}

}  // namespace xllm::kernel::npu::tilelang
