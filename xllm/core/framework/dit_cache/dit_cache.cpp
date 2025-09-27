#include "dit_cache.h"

#include "fbcache.h"
#include "fbcachewithtaylor.h"
#include "taylorseer.h"

namespace xllm {
std::unique_ptr<DiTCache> create_dit_cache(const DiTCacheConfig& cfg) {
  switch (cfg.selected_policy) {
    case PolicyType::FBCACHE:
      LOG(INFO) << "PolicyType: FBCACHE";
      return std::make_unique<FBCache>();
    case PolicyType::TAYLORSEER:
      LOG(INFO) << "PolicyType: TAYLORSEER";
      return std::make_unique<TaylorSeer>();
    case PolicyType::FBCACHE_WITH_TAYLORSEER:
      LOG(INFO) << "PolicyType: FBCACHE_WITH_TAYLORSEER";
      return std::make_unique<FBCacheWithTaylorSeer>();
    default:
      LOG(INFO) << "no cache selected";
      return nullptr;
  }
}
};  // namespace xllm
