/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "dit_cache_impl.h"

#include "dit_non_cache.h"
#include "fbcache.h"
#include "fbcache_taylorseer.h"
#include "framework/parallel_state/parallel_state.h"
#include "residual_cache.h"
#include "taylorseer.h"

namespace xllm {

torch::Tensor DitCacheImpl::get_tensor_or_empty(const TensorMap& m,
                                                const std::string& k) {
  auto it = m.find(k);
  if (it != m.end()) return it->second;
  return torch::Tensor();
}

bool DitCacheImpl::is_similar(const torch::Tensor& lhs,
                              const torch::Tensor& rhs,
                              float threshold) {
  if (!lhs.defined() || !rhs.defined()) return false;
  if (lhs.sizes() != rhs.sizes()) return false;

  if (threshold <= 0.0f) {
    return torch::allclose(lhs, rhs);
  }

  torch::Device dev = lhs.device();
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
  // auto sum_abs_diff = (lhs - rhs).abs().sum().to(torch::kFloat32);
  // auto sum_abs_lhs = lhs.abs().sum().to(torch::kFloat32);
  // auto count = torch::tensor({static_cast<float>(lhs.numel())}, opts);
  auto local_diff = (lhs - rhs).abs().sum().to(torch::kFloat64);
  auto local_lhs = lhs.abs().sum().to(torch::kFloat64);
  auto local_count = torch::tensor(static_cast<double>(lhs.numel()), opts);

  if (runtime_ctx_.sp_enabled && runtime_ctx_.sp_world_size > 1 &&
      runtime_ctx_.sp_group != nullptr) {
    auto* sp_group = static_cast<ProcessGroup*>(runtime_ctx_.sp_group);
    CHECK(sp_group != nullptr)
        << "sp_group is null in DitCacheImpl::is_similar";
    LOG(INFO) << "before value and shape：" << "sum_abs_diff " << local_diff
              << "sum_abs_lhs " << local_lhs << "count" << local_count;
    local_diff = xllm::parallel_state::reduce(local_diff, sp_group);
    local_lhs = xllm::parallel_state::reduce(local_lhs, sp_group);
    local_count = xllm::parallel_state::reduce(local_count, sp_group);
    LOG(INFO) << "after value and shape：" << "sum_abs_diff " << local_diff
              << "sum_abs_lhs " << local_lhs << "count" << local_count;
  }
  auto mean_diff = local_diff / local_count;
  auto mean_lhs = local_lhs / local_count;
  auto rel = mean_diff / (mean_lhs + 1e-6);

  LOG(INFO) << "rel.item<double>(): " << rel.item<double>();

  return rel.item<double>() < threshold;

  // auto mean_diff = sum_abs_diff / count;
  // auto mean_lhs = sum_abs_lhs / count;

  // // auto diff = (lhs - rhs).abs();
  // // auto mean_diff = diff.mean();
  // // auto mean_lhs = lhs.abs().mean();

  // auto rel = mean_diff / (mean_lhs + 1e-6);
  // // auto* sp_group = static_cast<ProcessGroup*>(runtime_ctx_.sp_group);
  // // if (!sp_group) {
  //   return (rel < threshold).item<bool>();
  // }
  // torch::Device dev = lhs.device();
  // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
  // auto local_tensor = torch::tensor(rel.item<double>(),
  // options).unsqueeze(0);  // 本地张量，值为 rel
  // // auto gathered_tensor = torch::empty({runtime_ctx_.sp_world_size},
  // options); auto gathered_tensor =
  // xllm::parallel_state::gather(local_tensor,sp_group);
  // // auto gathered_cpu =
  // gathered_tensor.to(torch::kCPU).contiguous().view({-1}); auto gathered_flat
  // = gathered_tensor.contiguous().view({-1});  // 展平为 [world_size]
  // // auto global_mean = gathered_flat.mean().item<float>();  // 计算所有 rank
  // rel 的平均值 auto global_mean =
  // static_cast<float>(gathered_flat.mean().item<double>());

  // // 6. 基于全局平均值判断，所有 rank 返回一致结果
  // return global_mean < threshold;

  // return (rel < threshold).item<bool>();
}

std::unique_ptr<DitCacheImpl> create_dit_cache(const DiTCacheConfig& cfg) {
  switch (cfg.selected_policy) {
    case PolicyType::FBCache:
      return std::make_unique<FBCache>();
    case PolicyType::TaylorSeer:
      return std::make_unique<TaylorSeer>();
    case PolicyType::FBCacheTaylorSeer:
      return std::make_unique<FBCacheTaylorSeer>();
    case PolicyType::ResidualCache:
      return std::make_unique<ResidualCache>();
    default:
      return std::make_unique<DiTNonCache>();
  }
}
}  // namespace xllm
