#include "rate_limiter.h"

#include <gflags/gflags.h>

#include "common/global_flags.h"
#include "common/metrics.h"

namespace xllm {

bool RateLimiter::is_limited() {
  if (FLAGS_max_concurrent_requests > 0) {
    int32_t num_requests =
        num_concurrent_requests_.load(std::memory_order_relaxed);
    if (num_requests >= FLAGS_max_concurrent_requests) {
      COUNTER_INC(server_request_total_limit);
      return true;
    }
  }
  num_concurrent_requests_.fetch_add(1, std::memory_order_relaxed);
  GAUGE_SET(num_concurrent_requests,
            num_concurrent_requests_.load(std::memory_order_relaxed));

  return false;
}

void RateLimiter::decrease_one_request() {
  num_concurrent_requests_.fetch_sub(1, std::memory_order_relaxed);
  GAUGE_SET(num_concurrent_requests,
            num_concurrent_requests_.load(std::memory_order_relaxed));
}

}  // namespace xllm
