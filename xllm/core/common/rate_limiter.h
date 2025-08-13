#pragma once

#include <atomic>

namespace xllm {

class RateLimiter final {
 public:
  RateLimiter() = default;

  ~RateLimiter() = default;

  bool is_limited();

  void decrease_one_request();

 private:
  std::atomic<int32_t> num_concurrent_requests_{0};
};

}  // namespace xllm