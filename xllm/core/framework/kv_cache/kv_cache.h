#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace xllm {
class KVCache final {
 public:
  KVCache() = default;
  KVCache(torch::Tensor key_cache, torch::Tensor value_cache);
  ~KVCache() = default;

  // TODO: pass in kv_shape and options instead
  torch::Tensor get_k_cache() const;
  torch::Tensor get_v_cache() const;

  bool empty() const {
    return !key_cache_.defined() || !value_cache_.defined();
  }

 private:
  torch::Tensor key_cache_;
  torch::Tensor value_cache_;
};

}  // namespace xllm
