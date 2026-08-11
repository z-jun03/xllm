/* Copyright 2025-2026 The xLLM Authors.

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

#include "regione.h"

#include <torch/nn/functional/pooling.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "core/common/global_flags.h"

#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#endif

namespace xllm {
namespace {
constexpr int64_t kVelocityCacheReuseIntervalSteps = 3;

double factorial(int64_t value) {
  return std::tgamma(static_cast<double>(value) + 1.0);
}

bool tensor_cache_ready(const std::vector<torch::Tensor>& cache,
                        int64_t block_id) {
  return block_id >= 0 && block_id < static_cast<int64_t>(cache.size()) &&
         cache[block_id].defined();
}

bool regione_env_enabled(const char* name, bool default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr) return default_value;
  return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "False") != 0;
}

torch::Tensor regione_allocate_cpu_cache(const torch::Tensor& tensor,
                                         bool use_pinned_memory) {
  auto cpu_options = tensor.options().device(torch::kCPU);
#if defined(USE_NPU)
  cpu_options = cpu_options.pinned_memory(use_pinned_memory);
#else
  static_cast<void>(use_pinned_memory);
#endif
  return torch::empty(tensor.sizes(), cpu_options);
}
}  // namespace

void RegionECache::init(const DiTCacheConfig& cfg) {
  config_ = cfg;
  regione_enabled_ = cfg.selected_policy == PolicyType::RegionE;
  regione_velocity_cache_ = torch::Tensor();
  regione_partial_step_count_ = 0;
  regione_current_block_ = -1;
  regione_current_use_cfg_ = false;
  regione_current_step_ = 0;
  regione_infer_steps_ = 0;
  regione_num_blocks_ = 0;
  regione_partial_mode_ = false;
  regione_use_pinned_cpu_cache_ =
      regione_env_enabled("REGIONE_USE_PINNED_CPU_CACHE", true);
  regione_use_async_offload_ =
      regione_env_enabled("REGIONE_USE_ASYNC_OFFLOAD", true);
  regione_use_async_prefetch_ =
      regione_env_enabled("REGIONE_USE_ASYNC_PREFETCH", true);
  if (regione_enabled_) {
    LOG(INFO) << "RegionE KV pipeline: pinned_cpu="
              << regione_use_pinned_cpu_cache_
              << ", async_offload=" << regione_use_async_offload_
              << ", async_prefetch=" << regione_use_async_prefetch_;
  }
  regione_target_seq_len_ = 0;
  regione_grid_h_ = 0;
  regione_grid_w_ = 0;
  regione_image_seq_len_ = 0;
  regione_sp_rank_ = 0;
  regione_sp_size_ = 1;
  regione_local_start_ = 0;
  regione_local_end_ = 0;
  regione_condition_latents_ = torch::Tensor();
  regione_edited_ids_ = torch::Tensor();
  regione_unedited_ids_ = torch::Tensor();
  regione_local_edited_global_ids_ = torch::Tensor();
  regione_local_edited_cache_ids_ = torch::Tensor();
  regione_local_image_global_ids_ = torch::Tensor();
  regione_cached_kv_ids_ = torch::Tensor();
  regione_cached_kv_full_seq_len_ = 0;
  regione_kv_cache_compact_.clear();
  regione_cond_kv_cache_compact_.clear();
  regione_clear_all_prefetch_slots();
#if defined(USE_NPU)
  regione_prefetch_stream_.reset();
  regione_offload_stream_.reset();
  regione_cache_ready_events_.clear();
  regione_cond_cache_ready_events_.clear();
#endif
}

void RegionECache::set_num_blocks(int64_t num_blocks) {
  regione_num_blocks_ = num_blocks;
  ensure_regione_kv_size(num_blocks);
}

bool RegionECache::regione_is_refresh_step(int64_t step) const {
  for (const auto refresh_step : config_.regione.refresh_steps) {
    const auto refresh_index =
        refresh_step > 0 ? refresh_step - 1 : refresh_step;
    if (step == refresh_index) return true;
  }
  return false;
}

bool RegionECache::regione_is_tail_step(int64_t step) const {
  return config_.regione.tail_steps > 0 && regione_infer_steps_ > 0 &&
         step >= regione_infer_steps_ - config_.regione.tail_steps;
}

bool RegionECache::regione_should_run_full_step(int64_t step) const {
  if (!regione_enabled_) return true;
  if (!regione_has_regions()) return true;
  if (step < config_.regione.warmup_steps) return true;
  if (regione_is_tail_step(step)) return true;
  return regione_is_refresh_step(step);
}

bool RegionECache::regione_should_direct_unedited(int64_t step) const {
  if (!regione_enabled_ || regione_infer_steps_ <= 0 ||
      regione_is_tail_step(step)) {
    return false;
  }
  const bool is_partition_step = step == config_.regione.warmup_steps - 1;
  if (!is_partition_step && !regione_is_refresh_step(step)) {
    return false;
  }
  return regione_next_direct_step(step) > step + 1;
}

int64_t RegionECache::regione_next_direct_step(int64_t step) const {
  int64_t tail_start = regione_infer_steps_;
  if (config_.regione.tail_steps > 0 && regione_infer_steps_ > 0) {
    tail_start =
        std::max<int64_t>(0, regione_infer_steps_ - config_.regione.tail_steps);
  }
  int64_t next_step = tail_start;
  for (const auto refresh_step : config_.regione.refresh_steps) {
    const auto refresh_index =
        refresh_step > 0 ? refresh_step - 1 : refresh_step;
    if (refresh_index > step && refresh_index < next_step)
      next_step = refresh_index;
  }
  if (next_step <= step) next_step = step + 1;
  if (regione_infer_steps_ > 0)
    next_step = std::min<int64_t>(next_step, regione_infer_steps_);
  return next_step;
}

void RegionECache::regione_prepare_inference(
    const torch::Tensor& latents,
    const torch::Tensor& condition_latents,
    int64_t grid_h,
    int64_t grid_w,
    int64_t sp_rank,
    int64_t sp_size) {
  if (!regione_enabled_) return;
  regione_target_seq_len_ =
      latents.defined() && latents.dim() > 1 ? latents.size(1) : 0;
  const auto condition_seq_len =
      condition_latents.defined() && condition_latents.dim() > 1
          ? condition_latents.size(1)
          : 0;
  regione_image_seq_len_ = regione_target_seq_len_ + condition_seq_len;
  regione_grid_h_ = grid_h;
  regione_grid_w_ = grid_w;
  regione_sp_rank_ = sp_rank;
  regione_sp_size_ = std::max<int64_t>(1, sp_size);
  const auto shard = regione_sp_size_ > 0
                         ? regione_image_seq_len_ / regione_sp_size_
                         : regione_image_seq_len_;
  regione_local_start_ = regione_sp_rank_ * shard;
  regione_local_end_ = regione_sp_rank_ == regione_sp_size_ - 1
                           ? regione_image_seq_len_
                           : regione_local_start_ + shard;
  if (regione_image_seq_len_ > 0 && regione_local_end_ > regione_local_start_) {
    regione_local_image_global_ids_ =
        torch::arange(regione_local_start_,
                      regione_local_end_,
                      latents.options().dtype(torch::kLong));
  } else {
    regione_local_image_global_ids_ = torch::Tensor();
  }
  regione_condition_latents_ = condition_latents;
  CHECK_EQ(regione_sp_size_, 1) << "RegionE currently requires --sp_size=1";
  CHECK_EQ(latents.dim(), 3) << "RegionE expects [batch, tokens, channels]";
  CHECK(condition_latents.defined())
      << "RegionE requires a reference image latent";
  CHECK_EQ(condition_latents.dim(), 3)
      << "RegionE expects [batch, tokens, channels] condition latents";
  CHECK_GT(regione_infer_steps_, config_.regione.warmup_steps)
      << "RegionE warmup steps must be smaller than inference steps";
  CHECK_GE(config_.regione.tail_steps, 0)
      << "RegionE tail steps must not be negative";
  CHECK_LT(config_.regione.tail_steps, regione_infer_steps_)
      << "RegionE tail steps must be smaller than inference steps";
  CHECK_GE(config_.regione.region_threshold, -1.0f)
      << "RegionE region threshold must be within [-1, 1]";
  CHECK_LE(config_.regione.region_threshold, 1.0f)
      << "RegionE region threshold must be within [-1, 1]";
  CHECK_GE(config_.regione.velocity_cache_threshold, 0.0f)
      << "RegionE velocity cache threshold must not be negative";
  CHECK_GE(config_.regione.velocity_cache_n_derivatives, 0)
      << "RegionE velocity cache derivative order must not be negative";
  for (const int64_t refresh_step : config_.regione.refresh_steps) {
    CHECK_GT(refresh_step, config_.regione.warmup_steps + 1)
        << "RegionE refresh steps must follow stabilization";
    CHECK_LE(refresh_step,
             regione_infer_steps_ - config_.regione.tail_steps - 1)
        << "RegionE refresh steps must precede smoothing";
  }
  for (size_t index = 1; index < config_.regione.refresh_steps.size();
       ++index) {
    CHECK_GT(config_.regione.refresh_steps[index] -
                 config_.regione.refresh_steps[index - 1],
             1)
        << "RegionE refresh steps must not be adjacent";
  }
  regione_edited_ids_ = torch::Tensor();
  regione_unedited_ids_ = torch::Tensor();
  regione_velocity_cache_ = torch::Tensor();
  regione_partial_step_count_ = 0;
  regione_partial_mode_ = false;
  regione_local_edited_global_ids_ = torch::Tensor();
  regione_local_edited_cache_ids_ = torch::Tensor();
  regione_cached_kv_ids_ = torch::Tensor();
  regione_cached_kv_full_seq_len_ = 0;
  regione_reset_edited_velocity_taylor();
  for (auto& cache : regione_k_cache_cpu_) cache = torch::Tensor();
  for (auto& cache : regione_v_cache_cpu_) cache = torch::Tensor();
  for (auto& cache : regione_cond_k_cache_cpu_) cache = torch::Tensor();
  for (auto& cache : regione_cond_v_cache_cpu_) cache = torch::Tensor();
  std::fill(regione_kv_cache_compact_.begin(),
            regione_kv_cache_compact_.end(),
            false);
  std::fill(regione_cond_kv_cache_compact_.begin(),
            regione_cond_kv_cache_compact_.end(),
            false);
#if defined(USE_NPU)
  std::fill(regione_cache_ready_events_.begin(),
            regione_cache_ready_events_.end(),
            nullptr);
  std::fill(regione_cond_cache_ready_events_.begin(),
            regione_cond_cache_ready_events_.end(),
            nullptr);
#endif
  regione_clear_all_prefetch_slots();
}

torch::Tensor RegionECache::regione_normalize_ids(
    const torch::Tensor& ids,
    const torch::Device& device) const {
  if (!ids.defined()) return torch::Tensor();
  auto out = ids;
  if (out.dim() > 1) out = out.reshape({-1});
  return out.to(device, torch::kLong, /*non_blocking=*/false, /*copy=*/false);
}

torch::Tensor RegionECache::regione_gather_ids(const torch::Tensor& tensor,
                                               const torch::Tensor& ids,
                                               int64_t dim) const {
  if (!tensor.defined() || !ids.defined()) return tensor;
  return tensor.index_select(dim, regione_normalize_ids(ids, tensor.device()));
}

torch::Tensor RegionECache::regione_scatter_ids(const torch::Tensor& values,
                                                const torch::Tensor& ids,
                                                const torch::Tensor& base,
                                                int64_t dim) const {
  if (!values.defined() || !ids.defined() || !base.defined()) return base;
  auto out = base.clone();
  out.index_copy_(dim, regione_normalize_ids(ids, base.device()), values);
  return out;
}

torch::Tensor RegionECache::regione_active_edited_ids() const {
  if (regione_is_partial_sp_mode() &&
      regione_local_edited_global_ids_.defined()) {
    return regione_local_edited_global_ids_;
  }
  return regione_edited_ids_;
}

torch::Tensor RegionECache::regione_kv_update_ids() const {
  if (regione_is_partial_sp_mode() &&
      regione_local_edited_cache_ids_.defined()) {
    return regione_local_edited_cache_ids_;
  }
  return regione_edited_ids_;
}

torch::Tensor RegionECache::regione_cached_kv_ids(int64_t full_seq_len,
                                                  const torch::Device& device) {
  CHECK_GE(full_seq_len, regione_target_seq_len_)
      << "RegionE full KV sequence is shorter than target sequence";
  if (regione_cached_kv_ids_.defined() &&
      regione_cached_kv_full_seq_len_ == full_seq_len &&
      regione_cached_kv_ids_.device() == device) {
    return regione_cached_kv_ids_;
  }

  torch::Tensor unedited_ids =
      regione_normalize_ids(regione_unedited_ids_, device);
  torch::Tensor condition_ids =
      torch::arange(regione_target_seq_len_,
                    full_seq_len,
                    torch::TensorOptions().device(device).dtype(torch::kLong));
  regione_cached_kv_ids_ = torch::cat({unedited_ids, condition_ids}, 0);
  regione_cached_kv_full_seq_len_ = full_seq_len;
  return regione_cached_kv_ids_;
}

torch::Tensor RegionECache::regione_gather_edited(
    const torch::Tensor& tensor) const {
  return regione_gather_ids(tensor, regione_active_edited_ids(), 1);
}

torch::Tensor RegionECache::regione_gather_unedited(
    const torch::Tensor& tensor) const {
  return regione_gather_ids(tensor, regione_unedited_ids_, 1);
}

torch::Tensor RegionECache::regione_scatter_edited(
    const torch::Tensor& edited,
    const torch::Tensor& base) const {
  return regione_scatter_ids(edited, regione_active_edited_ids(), base, 1);
}

torch::Tensor RegionECache::regione_scatter_unedited(
    const torch::Tensor& unedited,
    const torch::Tensor& base) const {
  return regione_scatter_ids(unedited, regione_unedited_ids_, base, 1);
}

torch::Tensor RegionECache::regione_gather_query_rope(
    const torch::Tensor& image_rope) const {
  auto ids = regione_active_edited_ids();
  if (!image_rope.defined() || !ids.defined()) return image_rope;
  if (image_rope.size(0) == ids.numel()) return image_rope;
  return regione_gather_ids(image_rope, ids, 0);
}

torch::Tensor RegionECache::regione_gather_key_rope(
    const torch::Tensor& image_rope,
    int64_t key_len) const {
  if (!image_rope.defined() || !regione_is_partial_sp_mode()) return image_rope;
  if (image_rope.size(0) == key_len) return image_rope;
  if (!regione_local_image_global_ids_.defined() ||
      regione_local_image_global_ids_.numel() != key_len) {
    return image_rope;
  }
  return regione_gather_ids(image_rope, regione_local_image_global_ids_, 0);
}

torch::Tensor RegionECache::regione_local_update_mask(
    const torch::Tensor& base) const {
  if (!base.defined()) return torch::Tensor();
  auto mask = torch::zeros({base.size(0), base.size(1), 1}, base.options());
  auto ids = regione_active_edited_ids();
  if (ids.defined() && ids.numel() > 0) {
    auto ones = torch::ones({base.size(0), ids.numel(), 1}, base.options());
    mask.index_copy_(1, regione_normalize_ids(ids, base.device()), ones);
  }
  return mask;
}

void RegionECache::regione_select_regions(const torch::Tensor& sample,
                                          const torch::Tensor& model_output,
                                          const torch::Tensor& sigmas,
                                          int64_t step) {
  if (!regione_enabled_ || regione_has_regions()) return;
  if (!sample.defined() || !model_output.defined() ||
      !regione_condition_latents_.defined()) {
    if (FLAGS_dit_debug_print) {
      LOG(INFO) << "RegionE ARP skipped at step " << step + 1
                << ": missing sample, velocity, or condition latents.";
    }
    return;
  }
  if (sample.dim() != 3 || sample.size(0) != 1) {
    if (FLAGS_dit_debug_print) {
      LOG(INFO) << "RegionE ARP skipped at step " << step + 1
                << ": sample shape must be [1, tokens, channels], got "
                << sample.sizes();
    }
    return;
  }
  auto condition = regione_condition_latents_;
  if (condition.dim() != 3 || condition.size(0) != sample.size(0) ||
      condition.size(1) != sample.size(1)) {
    if (FLAGS_dit_debug_print) {
      LOG(INFO) << "RegionE ARP skipped at step " << step + 1
                << ": condition shape " << condition.sizes()
                << " does not align with sample shape " << sample.sizes();
    }
    return;
  }
  condition = condition.to(sample.dtype());
  auto sigma = sigmas.index({step}).to(sample.device()).to(sample.dtype());
  auto sigma_final = sigmas.index({-1}).to(sample.device()).to(sample.dtype());
  auto estimate = sample + (sigma_final - sigma) * model_output;
  auto estimate_norm =
      estimate /
      torch::sqrt(torch::sum(estimate * estimate, -1, true)).clamp_min(1e-6);
  auto condition_norm =
      condition /
      torch::sqrt(torch::sum(condition * condition, -1, true)).clamp_min(1e-6);
  auto similarity = torch::sum(estimate_norm * condition_norm, -1);
  auto selected_mask = similarity <= config_.regione.region_threshold;
  if (regione_grid_h_ > 0 && regione_grid_w_ > 0 &&
      regione_grid_h_ * regione_grid_w_ == sample.size(1)) {
    auto mask2d = selected_mask.to(torch::kFloat)
                      .view({1, 1, regione_grid_h_, regione_grid_w_});
    auto vertical_pool_opts =
        torch::nn::functional::MaxPool2dFuncOptions({3, 1}).stride(1).padding(
            {1, 0});
    auto horizontal_pool_opts =
        torch::nn::functional::MaxPool2dFuncOptions({1, 3}).stride(1).padding(
            {0, 1});
    auto eroded_vertical =
        -torch::nn::functional::max_pool2d(-mask2d, vertical_pool_opts);
    auto eroded_horizontal =
        -torch::nn::functional::max_pool2d(-mask2d, horizontal_pool_opts);
    auto eroded = eroded_vertical * eroded_horizontal;
    auto dilation_pool_opts =
        torch::nn::functional::MaxPool2dFuncOptions({5, 5}).stride(1).padding(
            2);
    auto dilated =
        torch::nn::functional::max_pool2d(eroded, dilation_pool_opts);
    selected_mask = dilated.view({1, -1}) > 0.5;
  }
  auto selected_mask_cpu =
      selected_mask[0].to(torch::kCPU, torch::kBool).contiguous();
  auto edited_cpu = torch::nonzero(selected_mask_cpu).reshape({-1});
  if (edited_cpu.numel() == 0) {
    edited_cpu = std::get<1>(similarity[0].min(0, false))
                     .reshape({1})
                     .to(torch::kCPU, torch::kLong);
  }
  auto unedited_cpu =
      torch::nonzero(torch::logical_not(selected_mask_cpu)).reshape({-1});
  auto edited = edited_cpu.to(sample.device(), torch::kLong);
  auto unedited = unedited_cpu.to(sample.device(), torch::kLong);
  regione_edited_ids_ = edited;
  regione_unedited_ids_ = unedited;
  regione_update_local_ids();
  const char* debug_mask_path = std::getenv("REGIONE_DEBUG_MASK_PATH");
  if (debug_mask_path != nullptr && debug_mask_path[0] != '\0') {
    auto debug_mask = selected_mask.to(torch::kCPU, torch::kUInt8).contiguous();
    if (regione_grid_h_ > 0 && regione_grid_w_ > 0 &&
        regione_grid_h_ * regione_grid_w_ == sample.size(1)) {
      debug_mask = debug_mask.view({regione_grid_h_, regione_grid_w_});
    }
    torch::save(debug_mask, debug_mask_path);
  }
  if (FLAGS_dit_debug_print) {
    auto similarity_cpu =
        similarity.detach().to(torch::kCPU, torch::kFloat).contiguous();
    const int64_t similarity_count = similarity_cpu.numel();
    std::vector<float> similarity_values(similarity_count);
    std::copy_n(similarity_cpu.data_ptr<float>(),
                similarity_count,
                similarity_values.begin());
    std::sort(similarity_values.begin(), similarity_values.end());
    const size_t median_index = static_cast<size_t>(similarity_count / 2);
    const size_t upper_quartile_index =
        static_cast<size_t>(similarity_count * 3 / 4);
    const float edited_ratio =
        static_cast<float>(edited.numel()) / static_cast<float>(sample.size(1));
    LOG(INFO) << "RegionE ARP selected " << edited.numel() << "/"
              << sample.size(1) << " edited tokens (ratio=" << edited_ratio
              << ") at step " << step + 1
              << ", similarity: min=" << similarity_values.front()
              << ", median=" << similarity_values[median_index]
              << ", p75=" << similarity_values[upper_quartile_index]
              << ", max=" << similarity_values.back();
  }
}

void RegionECache::regione_update_local_ids() {
  regione_local_edited_global_ids_ = torch::Tensor();
  regione_local_edited_cache_ids_ = torch::Tensor();
  if (!regione_edited_ids_.defined() || regione_sp_size_ <= 1) return;
  auto ids =
      regione_normalize_ids(regione_edited_ids_, regione_edited_ids_.device());
  auto mask = (ids >= regione_local_start_) & (ids < regione_local_end_);
  auto local_global = ids.index({mask});
  regione_local_edited_global_ids_ = local_global;
  regione_local_edited_cache_ids_ = local_global - regione_local_start_;
}

void RegionECache::regione_update_velocity_cache(const torch::Tensor& value) {
  if (!regione_enabled_ || !value.defined()) return;
  if (regione_partial_mode_ && regione_velocity_cache_.defined() &&
      value.dim() == regione_velocity_cache_.dim() && value.dim() >= 2 &&
      value.size(0) == regione_velocity_cache_.size(0) &&
      value.size(1) == regione_active_edited_ids().numel()) {
    regione_velocity_cache_ =
        regione_scatter_edited(value, regione_velocity_cache_);
  } else {
    regione_velocity_cache_ = value;
  }
}

void RegionECache::regione_reset_edited_velocity_taylor() {
  const int64_t derivative_count =
      config_.regione.velocity_cache_n_derivatives + 1;
  regione_edited_velocity_derivatives_.assign(derivative_count,
                                              torch::Tensor());
  regione_prev_edited_velocity_derivatives_.assign(derivative_count,
                                                   torch::Tensor());
  regione_edited_velocity_derivatives_valid_.assign(derivative_count, false);
  regione_prev_edited_velocity_derivatives_valid_.assign(derivative_count,
                                                         false);
  regione_edited_velocity_last_exact_step_ = -1;
}

void RegionECache::regione_update_edited_velocity_taylor(
    const torch::Tensor& value,
    int64_t step) {
  if (!value.defined() || !regione_has_regions()) return;
  const int64_t derivative_count =
      config_.regione.velocity_cache_n_derivatives + 1;
  if (derivative_count <= 0) return;

  torch::Tensor edited_value = value;
  if (edited_value.dim() >= 2 &&
      edited_value.size(1) != regione_active_edited_ids().numel()) {
    edited_value = regione_gather_edited(edited_value);
  }
  if (!edited_value.defined() || edited_value.dim() < 2 ||
      edited_value.size(1) != regione_active_edited_ids().numel()) {
    return;
  }

  regione_prev_edited_velocity_derivatives_ =
      regione_edited_velocity_derivatives_;
  regione_prev_edited_velocity_derivatives_valid_ =
      regione_edited_velocity_derivatives_valid_;

  std::vector<torch::Tensor> derivatives(derivative_count);
  std::vector<bool> derivatives_valid(derivative_count, false);
  derivatives[0] = edited_value;
  derivatives_valid[0] = true;

  const int64_t elapsed_steps =
      regione_edited_velocity_last_exact_step_ >= 0
          ? step - regione_edited_velocity_last_exact_step_
          : 0;
  if (elapsed_steps > 0) {
    for (int64_t index = 0;
         index < config_.regione.velocity_cache_n_derivatives;
         ++index) {
      if (index >=
              static_cast<int64_t>(
                  regione_prev_edited_velocity_derivatives_valid_.size()) ||
          !regione_prev_edited_velocity_derivatives_valid_[index] ||
          !regione_prev_edited_velocity_derivatives_[index].defined()) {
        break;
      }
      derivatives[index + 1] =
          (derivatives[index] -
           regione_prev_edited_velocity_derivatives_[index]) /
          static_cast<double>(elapsed_steps);
      derivatives_valid[index + 1] = true;
    }
  }

  regione_edited_velocity_derivatives_ = std::move(derivatives);
  regione_edited_velocity_derivatives_valid_ = std::move(derivatives_valid);
  regione_edited_velocity_last_exact_step_ = step;
}

torch::Tensor RegionECache::regione_approximate_edited_velocity(
    int64_t step) const {
  if (regione_edited_velocity_last_exact_step_ < 0 ||
      regione_edited_velocity_derivatives_.empty() ||
      !regione_edited_velocity_derivatives_[0].defined()) {
    return torch::Tensor();
  }
  const int64_t elapsed_steps = step - regione_edited_velocity_last_exact_step_;
  if (elapsed_steps < 0) return torch::Tensor();

  torch::Tensor output =
      torch::zeros_like(regione_edited_velocity_derivatives_[0]);
  for (int64_t index = 0;
       index <
       static_cast<int64_t>(regione_edited_velocity_derivatives_.size());
       ++index) {
    if (index >= static_cast<int64_t>(
                     regione_edited_velocity_derivatives_valid_.size()) ||
        !regione_edited_velocity_derivatives_valid_[index]) {
      break;
    }
    const double coefficient =
        std::pow(static_cast<double>(elapsed_steps), index) / factorial(index);
    output += regione_edited_velocity_derivatives_[index] * coefficient;
  }
  return output;
}

torch::Tensor RegionECache::regione_velocity_cache() const {
  return regione_velocity_cache_;
}

std::optional<float> RegionECache::regione_velocity_decay(
    int64_t step,
    const torch::Tensor& timestep,
    const torch::Tensor& previous_timestep) {
  static_cast<void>(timestep);
  static_cast<void>(previous_timestep);
  if (!config_.regione.enable_velocity_cache ||
      !regione_velocity_cache_.defined() ||
      regione_edited_velocity_last_exact_step_ < 0 ||
      regione_edited_velocity_derivatives_.empty() ||
      !regione_edited_velocity_derivatives_[0].defined() ||
      regione_partial_step_count_ % kVelocityCacheReuseIntervalSteps == 0) {
    return std::nullopt;
  }
  return 1.0f;
}

RegionEStepPlan RegionECache::begin_step(
    int64_t step,
    const torch::Tensor& timestep,
    const torch::Tensor& previous_timestep) {
  RegionEStepPlan plan;
  plan.enabled = regione_enabled_;
  if (!regione_enabled_) {
    return plan;
  }

  regione_set_current_step(step);
  plan.full_step = regione_should_run_full_step(step);
  plan.partial_step = regione_has_regions() && !plan.full_step;
  regione_set_partial_mode(plan.partial_step);
  if (plan.full_step) {
    regione_partial_step_count_ = 0;
  }
  if (plan.partial_step) {
    const auto decay =
        regione_velocity_decay(step, timestep, previous_timestep);
    if (decay.has_value()) {
      plan.use_velocity_cache = true;
      plan.velocity_decay = *decay;
    }
    ++regione_partial_step_count_;
  }
  plan.run_partition = !regione_has_regions() &&
                       step == regione_warmup_steps() - 1 &&
                       !plan.use_velocity_cache;
  plan.direct_unedited = regione_should_direct_unedited(step);
  if (FLAGS_dit_debug_print) {
    LOG(INFO) << "RegionE step " << step + 1 << ": full=" << plan.full_step
              << ", partial=" << plan.partial_step
              << ", velocity_cache=" << plan.use_velocity_cache
              << ", direct_unedited=" << plan.direct_unedited;
  }
  return plan;
}

RegionEStepInput RegionECache::prepare_step_input(
    const torch::Tensor& latents,
    const torch::Tensor& condition_latents,
    const std::vector<std::vector<int64_t>>& main_shape,
    const RegionEStepPlan& plan) const {
  RegionEStepInput input;
  input.step_latents =
      plan.partial_step ? regione_gather_edited(latents) : latents;
  input.latent_model_input = input.step_latents;
  if (!plan.partial_step && condition_latents.defined()) {
    input.latent_model_input = torch::cat({latents, condition_latents}, 1);
  }
  input.main_shape = main_shape;
  if (plan.partial_step) {
    input.main_shape = {{1, input.step_latents.size(1), 1}};
  }
  input.use_cached_velocity = plan.use_velocity_cache;
  if (plan.use_velocity_cache) {
    input.cached_velocity =
        regione_approximate_edited_velocity(regione_current_step_) *
        plan.velocity_decay;
  }
  return input;
}

void RegionECache::observe_velocity(const torch::Tensor& latents,
                                    const torch::Tensor& noise_pred,
                                    const torch::Tensor& sigmas,
                                    int64_t step,
                                    const RegionEStepPlan& plan) {
  if (!regione_enabled_ || plan.use_velocity_cache) return;
  if (plan.run_partition) {
    regione_select_regions(latents, noise_pred, sigmas, step);
    regione_reset_edited_velocity_taylor();
  }
  regione_update_velocity_cache(noise_pred);
  regione_update_edited_velocity_taylor(noise_pred, step);
}

torch::Tensor RegionECache::apply_direct_unedited(
    const torch::Tensor& prev_latents,
    const torch::Tensor& latents,
    const torch::Tensor& noise_pred,
    const torch::Tensor& sigmas,
    int64_t step) const {
  auto sigma = sigmas.index({step}).to(latents.device()).to(latents.dtype());
  auto next_direct_step = regione_next_direct_step(step);
  auto sigma_direct =
      sigmas.index({next_direct_step}).to(latents.device()).to(latents.dtype());
  auto unedited_direct =
      regione_gather_unedited(latents) +
      (sigma_direct - sigma) * regione_gather_unedited(noise_pred);
  return regione_scatter_unedited(unedited_direct, prev_latents);
}

void RegionECache::regione_prefetch_img_kv(int64_t block_id,
                                           bool use_cfg,
                                           const torch::Tensor& reference) {
  if (!reference.defined()) return;
  regione_prefetch_img_kv(
      block_id, use_cfg, reference.device(), reference.scalar_type());
}

void RegionECache::regione_set_current_block(int64_t block_id,
                                             bool use_cfg,
                                             const torch::Tensor& reference) {
  regione_current_block_ = block_id;
  regione_current_use_cfg_ = use_cfg;
  static_cast<void>(reference);
}

void RegionECache::regione_finish_current_block(int64_t block_id,
                                                bool use_cfg) {
  for (auto& slot : regione_prefetch_slots(use_cfg)) {
    if (slot.block_id != block_id || !slot.in_use) continue;
#if defined(USE_NPU)
    if (slot.key.defined() && slot.key.device().is_privateuseone()) {
      Stream current_stream(
          c10_npu::getCurrentNPUStream(slot.key.device().index()));
      slot.reusable_event = current_stream.record_event();
      CHECK(slot.reusable_event != nullptr)
          << "RegionE failed to record KV slot reusable event";
    }
#endif
    slot.block_id = -1;
    slot.in_use = false;
    return;
  }
}

void RegionECache::regione_set_current_step(int64_t step) {
  regione_current_step_ = step;
}

void RegionECache::regione_set_partial_mode(bool partial_mode) {
  regione_partial_mode_ = regione_enabled_ && partial_mode;
}

bool RegionECache::regione_is_partial_sp_mode() const {
  return regione_enabled_ && regione_partial_mode_ && regione_sp_size_ > 1;
}

bool RegionECache::regione_should_store_kv() const {
  if (!regione_enabled_ || regione_partial_mode_) {
    return false;
  }
  if (regione_current_step_ == config_.regione.warmup_steps - 1) {
    return true;
  }
  return regione_has_regions() &&
         regione_is_refresh_step(regione_current_step_);
}

bool RegionECache::regione_should_patch_kv() const {
  return regione_enabled_ && regione_partial_mode_;
}

void RegionECache::ensure_regione_kv_size(int64_t num_blocks) {
  if (num_blocks <= 0) return;
  regione_k_cache_cpu_.resize(num_blocks);
  regione_v_cache_cpu_.resize(num_blocks);
  regione_cond_k_cache_cpu_.resize(num_blocks);
  regione_cond_v_cache_cpu_.resize(num_blocks);
  regione_kv_cache_compact_.resize(num_blocks, false);
  regione_cond_kv_cache_compact_.resize(num_blocks, false);
#if defined(USE_NPU)
  regione_cache_ready_events_.resize(num_blocks);
  regione_cond_cache_ready_events_.resize(num_blocks);
#endif
}

std::vector<bool>& RegionECache::regione_kv_cache_compact_flags(bool use_cfg) {
  return use_cfg ? regione_cond_kv_cache_compact_ : regione_kv_cache_compact_;
}

#if defined(USE_NPU)
std::vector<StreamEventPtr>& RegionECache::regione_cache_ready_events(
    bool use_cfg) {
  return use_cfg ? regione_cond_cache_ready_events_
                 : regione_cache_ready_events_;
}
#endif

std::vector<RegionECache::RegionEPrefetchedKV>&
RegionECache::regione_prefetch_slots(bool use_cfg) {
  auto& slots =
      use_cfg ? regione_cond_prefetch_slots_ : regione_prefetch_slots_;
  if (slots.empty()) slots.resize(2);
  return slots;
}

void RegionECache::regione_clear_prefetch_slot(RegionEPrefetchedKV& slot) {
  slot.block_id = -1;
  slot.in_use = false;
  slot.key = torch::Tensor();
  slot.value = torch::Tensor();
#if defined(USE_NPU)
  slot.ready_event.reset();
  slot.reusable_event.reset();
#endif
}

void RegionECache::regione_clear_prefetch_block(bool use_cfg,
                                                int64_t block_id) {
  for (auto& slot : regione_prefetch_slots(use_cfg)) {
    if (slot.block_id == block_id && !slot.in_use) {
      regione_clear_prefetch_slot(slot);
    }
  }
}

void RegionECache::regione_clear_all_prefetch_slots() {
  for (auto& slot : regione_prefetch_slots_) {
    regione_clear_prefetch_slot(slot);
  }
  for (auto& slot : regione_cond_prefetch_slots_) {
    regione_clear_prefetch_slot(slot);
  }
}

void RegionECache::regione_prefetch_img_kv(int64_t block_id,
                                           bool use_cfg,
                                           const torch::Device& device,
                                           c10::ScalarType dtype) {
  if (!regione_enabled_ || !regione_partial_mode_ ||
      !regione_use_async_prefetch_ || block_id < 0 ||
      block_id >= regione_num_blocks_) {
    return;
  }
  auto& k_cache = use_cfg ? regione_cond_k_cache_cpu_ : regione_k_cache_cpu_;
  auto& v_cache = use_cfg ? regione_cond_v_cache_cpu_ : regione_v_cache_cpu_;
  if (!tensor_cache_ready(k_cache, block_id) ||
      !tensor_cache_ready(v_cache, block_id)) {
    return;
  }

  auto& slots = regione_prefetch_slots(use_cfg);
  for (const auto& slot : slots) {
    if (slot.block_id == block_id && slot.key.defined() &&
        slot.value.defined() && slot.key.device() == device &&
        slot.key.scalar_type() == dtype) {
      return;
    }
  }

  RegionEPrefetchedKV& target =
      slots[static_cast<size_t>(block_id) % slots.size()];
  if (target.block_id >= 0 || target.in_use) {
    if (FLAGS_dit_debug_print) {
      LOG(INFO) << "RegionE KV prefetch skipped for block " << block_id
                << ": target slot is occupied by block " << target.block_id;
    }
    return;
  }

#if defined(USE_NPU)
  if (device.is_privateuseone()) {
    if (!regione_prefetch_stream_.has_value()) {
      regione_prefetch_stream_.emplace(
          c10_npu::getStreamFromPool(/*isHighPriority=*/false, device.index()));
    }
    Stream& stream = regione_prefetch_stream_.value();
    {
      c10::StreamGuard stream_guard = stream.set_stream_guard();
      CHECK(stream.wait_event(target.reusable_event))
          << "RegionE failed to wait for reusable KV slot";
      auto& cache_ready_events = regione_cache_ready_events(use_cfg);
      CHECK(stream.wait_event(cache_ready_events[block_id]))
          << "RegionE failed to wait for CPU KV cache";
      const torch::TensorOptions options =
          torch::TensorOptions().device(device).dtype(dtype);
      if (!target.key.defined() ||
          target.key.sizes() != k_cache[block_id].sizes() ||
          target.key.scalar_type() != dtype || target.key.device() != device) {
        target.key = torch::empty(k_cache[block_id].sizes(), options);
      }
      if (!target.value.defined() ||
          target.value.sizes() != v_cache[block_id].sizes() ||
          target.value.scalar_type() != dtype ||
          target.value.device() != device) {
        target.value = torch::empty(v_cache[block_id].sizes(), options);
      }
      target.key.copy_(k_cache[block_id], /*non_blocking=*/true);
      target.value.copy_(v_cache[block_id], /*non_blocking=*/true);
      target.ready_event = stream.record_event();
      CHECK(target.ready_event != nullptr)
          << "RegionE failed to record KV prefetch event";
    }
    target.block_id = block_id;
    return;
  }
#endif

  target.key = k_cache[block_id].to(
      device, dtype, /*non_blocking=*/false, /*copy=*/true);
  target.value = v_cache[block_id].to(
      device, dtype, /*non_blocking=*/false, /*copy=*/true);
  target.block_id = block_id;
}

bool RegionECache::regione_take_prefetched_img_kv(int64_t block_id,
                                                  bool use_cfg,
                                                  const torch::Device& device,
                                                  c10::ScalarType dtype,
                                                  torch::Tensor* key,
                                                  torch::Tensor* value) {
  for (auto& slot : regione_prefetch_slots(use_cfg)) {
    if (slot.block_id != block_id || !slot.key.defined() ||
        !slot.value.defined() || slot.key.device() != device ||
        slot.key.scalar_type() != dtype) {
      continue;
    }
#if defined(USE_NPU)
    if (slot.ready_event != nullptr && device.is_privateuseone()) {
      Stream current_stream(c10_npu::getCurrentNPUStream(device.index()));
      CHECK(current_stream.wait_event(slot.ready_event))
          << "RegionE failed to wait for prefetched KV";
      if (FLAGS_dit_debug_print) {
        LOG(INFO) << "RegionE KV prefetch hit for block " << block_id;
      }
    }
#endif
    *key = slot.key;
    *value = slot.value;
    slot.in_use = true;
    return true;
  }
  return false;
}

void RegionECache::regione_store_img_kv(int64_t block_id,
                                        bool use_cfg,
                                        const torch::Tensor& key,
                                        const torch::Tensor& value) {
  if (!regione_enabled_ || block_id < 0) return;
  ensure_regione_kv_size(std::max<int64_t>(regione_num_blocks_, block_id + 1));
  const bool compact =
      regione_has_regions() && regione_is_refresh_step(regione_current_step_);
  torch::Tensor cache_key = key;
  torch::Tensor cache_value = value;
  if (compact) {
    const torch::Tensor cached_ids =
        regione_cached_kv_ids(key.size(1), key.device());
    cache_key = key.index_select(/*dim=*/1, cached_ids);
    cache_value = value.index_select(/*dim=*/1, cached_ids);
  }
  regione_preallocate_img_kv(block_id, use_cfg, cache_key, cache_value);
  auto& k_cache = use_cfg ? regione_cond_k_cache_cpu_ : regione_k_cache_cpu_;
  auto& v_cache = use_cfg ? regione_cond_v_cache_cpu_ : regione_v_cache_cpu_;
  auto& compact_flags = regione_kv_cache_compact_flags(use_cfg);
  compact_flags[block_id] = compact;
#if defined(USE_NPU)
  if (regione_use_async_offload_ && cache_key.device().is_privateuseone() &&
      cache_value.device().is_privateuseone()) {
    if (!regione_offload_stream_.has_value()) {
      regione_offload_stream_.emplace(c10_npu::getStreamFromPool(
          /*isHighPriority=*/false, cache_key.device().index()));
    }
    Stream current_stream(
        c10_npu::getCurrentNPUStream(cache_key.device().index()));
    StreamEventPtr kv_ready_event = current_stream.record_event();
    CHECK(kv_ready_event != nullptr)
        << "RegionE failed to record produced KV event";

    Stream& offload_stream = regione_offload_stream_.value();
    {
      c10::StreamGuard stream_guard = offload_stream.set_stream_guard();
      CHECK(offload_stream.wait_event(kv_ready_event))
          << "RegionE failed to wait for produced KV";
      k_cache[block_id].copy_(cache_key.detach(), /*non_blocking=*/true);
      v_cache[block_id].copy_(cache_value.detach(), /*non_blocking=*/true);
      c10_npu::NPUCachingAllocator::recordStream(cache_key.storage().data_ptr(),
                                                 *offload_stream.get_stream());
      c10_npu::NPUCachingAllocator::recordStream(
          cache_value.storage().data_ptr(), *offload_stream.get_stream());
      auto& cache_ready_events = regione_cache_ready_events(use_cfg);
      cache_ready_events[block_id] = offload_stream.record_event();
      CHECK(cache_ready_events[block_id] != nullptr)
          << "RegionE failed to record CPU KV cache event";
    }
  } else {
    k_cache[block_id].copy_(cache_key.detach().contiguous(),
                            /*non_blocking=*/false);
    v_cache[block_id].copy_(cache_value.detach().contiguous(),
                            /*non_blocking=*/false);
  }
#else
  k_cache[block_id].copy_(cache_key.detach().contiguous(),
                          /*non_blocking=*/false);
  v_cache[block_id].copy_(cache_value.detach().contiguous(),
                          /*non_blocking=*/false);
#endif
  regione_clear_prefetch_block(use_cfg, block_id);
}

void RegionECache::regione_preallocate_img_kv(int64_t block_id,
                                              bool use_cfg,
                                              const torch::Tensor& key,
                                              const torch::Tensor& value) {
  if (!regione_enabled_ || block_id < 0) return;
  ensure_regione_kv_size(std::max<int64_t>(regione_num_blocks_, block_id + 1));
  auto& k_cache = use_cfg ? regione_cond_k_cache_cpu_ : regione_k_cache_cpu_;
  auto& v_cache = use_cfg ? regione_cond_v_cache_cpu_ : regione_v_cache_cpu_;
  if (!k_cache[block_id].defined() ||
      k_cache[block_id].sizes() != key.sizes() ||
      k_cache[block_id].scalar_type() != key.scalar_type()) {
    k_cache[block_id] =
        regione_allocate_cpu_cache(key, regione_use_pinned_cpu_cache_);
  }
  if (!v_cache[block_id].defined() ||
      v_cache[block_id].sizes() != value.sizes() ||
      v_cache[block_id].scalar_type() != value.scalar_type()) {
    v_cache[block_id] =
        regione_allocate_cpu_cache(value, regione_use_pinned_cpu_cache_);
  }
#if defined(USE_NPU)
  if (key.device().is_privateuseone() && value.device().is_privateuseone()) {
    auto& slots = regione_prefetch_slots(use_cfg);
    RegionEPrefetchedKV& slot =
        slots[static_cast<size_t>(block_id) % slots.size()];
    const torch::TensorOptions key_options =
        key.options().device(key.device()).dtype(key.scalar_type());
    const torch::TensorOptions value_options =
        value.options().device(value.device()).dtype(value.scalar_type());
    if (!slot.key.defined() || slot.key.sizes() != key.sizes() ||
        slot.key.scalar_type() != key.scalar_type() ||
        slot.key.device() != key.device()) {
      slot.key = torch::empty(key.sizes(), key_options);
    }
    if (!slot.value.defined() || slot.value.sizes() != value.sizes() ||
        slot.value.scalar_type() != value.scalar_type() ||
        slot.value.device() != value.device()) {
      slot.value = torch::empty(value.sizes(), value_options);
    }
  }
#endif
}

std::pair<torch::Tensor, torch::Tensor> RegionECache::process_image_kv(
    const torch::Tensor& key,
    const torch::Tensor& value) {
  if (!regione_enabled_) return {key, value};
  if (!regione_partial_mode_ &&
      regione_current_step_ < config_.regione.warmup_steps - 1) {
    regione_preallocate_img_kv(
        regione_current_block_, regione_current_use_cfg_, key, value);
  }
  if (regione_should_store_kv()) {
    regione_store_img_kv(
        regione_current_block_, regione_current_use_cfg_, key, value);
    return {key, value};
  }
  if (regione_should_patch_kv()) {
    return regione_patch_img_kv(
        regione_current_block_, regione_current_use_cfg_, key, value);
  }
  return {key, value};
}

std::pair<torch::Tensor, torch::Tensor> RegionECache::adjust_image_rope(
    const torch::Tensor& image_rope,
    int64_t key_len) const {
  auto query_rope = regione_partial_mode_
                        ? regione_gather_query_rope(image_rope)
                        : image_rope;
  if (regione_partial_mode_) {
    return {query_rope, query_rope};
  }
  auto key_rope = regione_gather_key_rope(image_rope, key_len);
  return {query_rope, key_rope};
}

std::pair<torch::Tensor, torch::Tensor> RegionECache::regione_patch_img_kv(
    int64_t block_id,
    bool use_cfg,
    const torch::Tensor& key,
    const torch::Tensor& value) {
  if (!regione_enabled_ || block_id < 0) return {key, value};
  auto& k_cache = use_cfg ? regione_cond_k_cache_cpu_ : regione_k_cache_cpu_;
  auto& v_cache = use_cfg ? regione_cond_v_cache_cpu_ : regione_v_cache_cpu_;
  auto& compact_flags = regione_kv_cache_compact_flags(use_cfg);
  if (!tensor_cache_ready(k_cache, block_id) ||
      !tensor_cache_ready(v_cache, block_id)) {
    regione_store_img_kv(block_id, use_cfg, key, value);
    return {key, value};
  }
  torch::Tensor full_key;
  torch::Tensor full_value;
  const auto took_prefetched = regione_take_prefetched_img_kv(block_id,
                                                              use_cfg,
                                                              key.device(),
                                                              key.scalar_type(),
                                                              &full_key,
                                                              &full_value);
  if (!took_prefetched) {
    if (FLAGS_dit_debug_print) {
      LOG(INFO) << "RegionE KV prefetch miss for block " << block_id;
    }
    full_key = k_cache[block_id].to(key.device(),
                                    key.scalar_type(),
                                    /*non_blocking=*/false,
                                    /*copy=*/true);
    full_value = v_cache[block_id].to(value.device(),
                                      value.scalar_type(),
                                      /*non_blocking=*/false,
                                      /*copy=*/true);
  }
  if (compact_flags[block_id]) {
    CHECK_EQ(full_key.size(1) + key.size(1), regione_cached_kv_full_seq_len_)
        << "RegionE compact KV does not cover the full image sequence";
    full_key = torch::cat({full_key, key}, /*dim=*/1);
    full_value = torch::cat({full_value, value}, /*dim=*/1);
    return {full_key, full_value};
  }
  if (full_key.sizes() == key.sizes()) {
    regione_store_img_kv(block_id, use_cfg, key, value);
    return {key, value};
  }
  auto update_ids = regione_kv_update_ids();
  if (update_ids.defined() && key.dim() >= 2 &&
      full_key.size(0) == key.size(0) && key.size(1) == update_ids.numel()) {
    const torch::Tensor normalized_ids =
        regione_normalize_ids(update_ids, full_key.device());
    full_key.index_copy_(/*dim=*/1, normalized_ids, key);
    full_value.index_copy_(/*dim=*/1, normalized_ids, value);
  }
  return {full_key, full_value};
}

}  // namespace xllm
