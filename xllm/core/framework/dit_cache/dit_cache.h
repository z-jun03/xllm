#pragma once
#include <torch/torch.h>

#include <cmath>

#include "dit_cache_config.h"
#include "dit_cache_data.h"

namespace xllm {

class DiTCache {
 protected:
  int num_inference_steps = 25;
  int warmup_steps = 2;
  int executed_steps = 0;

 public:
  virtual ~DiTCache() = default;

  virtual void init(DiTCacheConfig cfg) = 0;

  bool is_in_warmup() { return executed_steps <= warmup_steps; }

  virtual bool on_before_block(CacheBlockIn blockin) { return false; }

  virtual CacheBlockOut on_after_block(CacheBlockIn blockin) {
    TensorMap out;
    if (contains_key(blockin.tensors, "hidden_states")) {
      out["hidden_states"] =
          get_tensor_or_empty(blockin.tensors, "hidden_states");
    }
    if (contains_key(blockin.tensors, "encoder_hidden_states")) {
      out["encoder_hidden_states"] =
          get_tensor_or_empty(blockin.tensors, "encoder_hidden_states");
    }
    return CacheBlockOut(out);
  }

  virtual bool on_before_step(CacheStepIn stepin) { return false; }

  virtual CacheStepOut on_after_step(CacheStepIn stepin) {
    TensorMap out;
    if (contains_key(stepin.tensors, "hidden_states")) {
      out["hidden_states"] =
          get_tensor_or_empty(stepin.tensors, "hidden_states");
    }
    return CacheStepOut(out);
  }

  bool is_similar(const torch::Tensor& t1,
                  const torch::Tensor& t2,
                  float threshold) {
    if (!t1.defined() || !t2.defined()) return false;
    if (threshold <= 0.0f) return torch::allclose(t1, t2);

    if (t1.sizes() != t2.sizes()) return false;

    auto diff = (t1 - t2).abs();
    auto mean_diff_tensor = diff.mean();
    auto mean_t1_tensor = t1.abs().mean();

    double mean_diff_val = mean_diff_tensor.cpu().item<double>();
    double mean_t1_val = mean_t1_tensor.cpu().item<double>();

    if (mean_t1_val == 0.0) {
      const double eps = 1e-6;
      return mean_diff_val < eps;
    }

    double rel = mean_diff_val / mean_t1_val;
    return rel < static_cast<double>(threshold);
  }
};

std::unique_ptr<DiTCache> create_dit_cache(const DiTCacheConfig& cfg);

}  // namespace xllm
