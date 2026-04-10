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

#pragma once

#include <torch/torch.h>

#include <deque>
#include <string>
#include <vector>

#include "dit_cache_impl.h"

namespace xllm {

using TensorMap = std::unordered_map<std::string, torch::Tensor>;

class DiCache : public DitCacheImpl {
 public:
  DiCache() = default;
  ~DiCache() = default;

  DiCache(const DiCache&) = delete;
  DiCache& operator=(const DiCache&) = delete;
  DiCache(DiCache&&) = default;
  DiCache& operator=(DiCache&&) = default;

  void init(const DiTCacheConfig& cfg) override;

  bool on_before_step(const CacheStepIn& stepin) override;
  CacheStepOut on_after_step(const CacheStepIn& stepin) override;

  bool on_before_block(const CacheBlockIn& blockin) override;
  CacheBlockOut on_after_block(const CacheBlockIn& blockin) override;

 private:
  // Reset all internal cache states
  void reset_cache();

  // Compute relative L1 distance between two tensors
  double relative_l1_distance(const torch::Tensor& A, const torch::Tensor& B);

  // Decide whether to skip remaining blocks and prepare extrapolated result
  torch::Tensor decide_and_prepare_skip(
      const torch::Tensor& original_hidden_states);

 private:
  // Configuration parameters
  int probe_depth_ = 1;
  double rel_l1_thresh_ = 0.4;
  double ret_ratio_ = 0.2;
  std::string error_choice_ = "delta_y";
  int num_steps_ = 28;

  // Runtime state
  int current_step_ = 0;
  double accumulated_error_ = 0.0;
  bool force_full_calc_ = false;
  bool resume_from_probe_ = false;
  bool skip_remaining_blocks_ = false;

  // Tensors for caching and extrapolation
  torch::Tensor approximate_final_hidden_;
  torch::Tensor previous_input_;
  torch::Tensor previous_probe_states_;  // 上一个step的探针结果，
  torch::Tensor previous_residual_;
  torch::Tensor previous_probe_residual_;

  std::deque<torch::Tensor> residual_window_;
  std::deque<torch::Tensor> probe_residual_window_;

  // Temporary data shared across blocks within one step
  torch::Tensor original_hidden_states_;
  torch::Tensor probe_hidden_states_;  // 当前step
  torch::Tensor probe_encoder_states_;
  int current_block_id_ = 0;
  bool use_cache_ = false;
};

}  // namespace xllm