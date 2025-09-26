#pragma once
#include <glog/logging.h>

#include "fbcache.h"
#include "fbcachewithtaylor.h"
#include "taylorseer.h"

namespace xllm {

class CacheAgent {
 private:
  std::unique_ptr<FBCache> fbcache;
  std::unique_ptr<TaylorSeer> taylorseer;
  std::unique_ptr<FBCacheWithTaylorSeer> fbcachewithtaylorseer;
  CacheBase* active_cache = nullptr;

  CacheAgent() = default;

 public:
  CacheAgent(const CacheAgent&) = delete;
  CacheAgent& operator=(const CacheAgent&) = delete;
  CacheAgent(CacheAgent&&) = delete;
  CacheAgent& operator=(CacheAgent&&) = delete;

  ~CacheAgent() = default;

  static CacheAgent& get_instance() {
    static CacheAgent cacheagent;
    return cacheagent;
  }

  static CacheAgent& getinstance() { return get_instance(); }

  void init(const CacheConfig& cfg) {
    switch (cfg.selected_method) {
      case MethodType::FBCACHE:
        fbcache = std::make_unique<FBCache>();
        active_cache = fbcache.get();
        LOG(INFO) << "using FBCACHE";
        break;

      case MethodType::TAYLORSEER:
        taylorseer = std::make_unique<TaylorSeer>();
        active_cache = taylorseer.get();
        LOG(INFO) << "using TAYLORSEER";
        break;

      case MethodType::FBCACHE_WITH_TAYLORSEER:
        fbcachewithtaylorseer = std::make_unique<FBCacheWithTaylorSeer>();
        active_cache = fbcachewithtaylorseer.get();
        LOG(INFO) << "using FBCACHE_WITH_TAYLORSEER";
        break;

      case MethodType::NONE:
      default:
        LOG(INFO) << "no cache selected";
        active_cache = nullptr;
        break;
    }

    if (active_cache) {
      try {
        active_cache->init(cfg);
      } catch (const std::exception& e) {
        LOG(ERROR) << "cache init failed: " << e.what();
        active_cache = nullptr;
      }
    }
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
