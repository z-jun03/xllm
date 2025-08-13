#pragma once

#include <folly/Function.h>

#include <functional>
#include <future>
#include <vector>

#include "common/macros.h"
#include "common/options.h"
#include "common/rate_limiter.h"
#include "common/types.h"
#include "framework/request/request_params.h"
#include "runtime/engine.h"

namespace xllm {

class Master {
 public:
  explicit Master(const Options& options, EngineType type);
  virtual ~Master() = default;
  virtual void run() = 0;
  virtual const Options& options() const { return options_; }

  virtual void get_cache_info(std::vector<uint64_t>& cluster_ids,
                              std::vector<std::string>& addrs,
                              std::vector<int64_t>& k_cache_ids,
                              std::vector<int64_t>& v_cache_ids) {}

  virtual bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                            const std::vector<std::string>& addrs,
                            const std::vector<std::string>& device_ips,
                            const std::vector<uint16_t>& ports,
                            const int32_t dp_size) {
    return false;
  }

  virtual bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                              const std::vector<std::string>& addrs,
                              const std::vector<std::string>& device_ips,
                              const std::vector<uint16_t>& ports,
                              const int32_t dp_size) {
    return false;
  }

  RateLimiter* get_rate_limiter() { return &rate_limiter_; }

 protected:
  Options options_;
  std::unique_ptr<Engine> engine_;

  RateLimiter rate_limiter_;
};

std::unique_ptr<Master> create_master(const std::string& backend,
                                      const Options& options);

}  // namespace xllm
