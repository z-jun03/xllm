#pragma once

namespace xllm {

enum class MethodType {
  NONE = 0,
  FBCACHE,
  TAYLORSEER,
  FBCACHE_WITH_TAYLORSEER
};

struct CacheConfig {
  struct Method {
    int num_inference_steps = 25;
    int warmup_steps = 0;
    int max_cache_step = 0;
    int executed_steps = 0;
  } method;

  struct FBCacheOptions : public Method {
    float residual_diff_threshold = 0;
  } fbcache;

  struct TaylorSeerOptions : public Method {
    int n_derivatives = 2;
    int skip_interval_steps = 3;
  } taylorseer;

  struct FBCacheWithTaylorSeerOptions : public Method {
    float residual_diff_threshold = 0;
    int n_derivatives = 3;
  } fbcachewithtaylor;

  MethodType selected_method = MethodType::NONE;
};

}  // namespace xllm
