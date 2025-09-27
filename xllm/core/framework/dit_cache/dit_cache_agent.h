#pragma once
#include <glog/logging.h>

#include "dit_cache.h"

namespace xllm {

class DiTCacheAgent {
 protected:
  std::unique_ptr<DiTCache> active_cache;
  DiTCacheAgent() = default;

 public:
  DiTCacheAgent(const DiTCacheAgent&) = delete;
  DiTCacheAgent& operator=(const DiTCacheAgent&) = delete;
  DiTCacheAgent(DiTCacheAgent&&) = delete;
  DiTCacheAgent& operator=(DiTCacheAgent&&) = delete;

  ~DiTCacheAgent() = default;

  static DiTCacheAgent& getinstance() {
    static DiTCacheAgent ditcacheagent;
    return ditcacheagent;
  }

  bool init(const DiTCacheConfig& cfg) {
    active_cache = create_dit_cache(cfg);
    if (!active_cache) {
      return false;
    }
    active_cache->init(cfg);
    return true;
  }

  bool on_before_block(CacheBlockIn blockin) {
    return active_cache->on_before_block(blockin);
  }

  CacheBlockOut on_after_block(CacheBlockIn blockin) {
    return active_cache->on_after_block(blockin);
  }

  bool on_before_step(CacheStepIn stepin) {
    return active_cache->on_before_step(stepin);
  }

  CacheStepOut on_after_step(CacheStepIn stepin) {
    return active_cache->on_after_step(stepin);
  }
};

}  // namespace xllm
