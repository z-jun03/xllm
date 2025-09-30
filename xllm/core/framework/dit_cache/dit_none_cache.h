#pragma once
#include "dit_cache_impl.h"

namespace xllm {

class NoOpCache : public DitCacheImpl {
 public:
  void init(const DiTCacheConfig& cfg) override {};

  bool on_before_block(const CacheBlockIn& /*blockin*/) override {
    return false;
  }

  CacheBlockOut on_after_block(const CacheBlockIn& blockin) override {
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

  bool on_before_step(const CacheStepIn& stepin) override { return false; }

  virtual CacheStepOut on_after_step(const CacheStepIn& stepin) override {
    TensorMap out;
    if (contains_key(stepin.tensors, "hidden_states")) {
      out["hidden_states"] =
          get_tensor_or_empty(stepin.tensors, "hidden_states");
    }
    return CacheStepOut(out);
  }
};

}  // namespace xllm