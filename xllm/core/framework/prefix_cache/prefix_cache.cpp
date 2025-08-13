#include "prefix_cache.h"

#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>

#include "prefix_cache_hash_murmur3.h"
#include "prefix_cache_hash_sha256.h"

namespace xllm {

std::unique_ptr<PrefixCache> CreatePrefixCachePolicy(
    int32_t block_size,
    const std::string& policy,
    const bool& enbale_service_routing) {
  std::vector<absl::string_view> subs = absl::StrSplit(policy, ':');
  CHECK(subs.size() > 0) << " Prefix cache, input param invalid."
                         << " policy:" << policy;

  if ("sha256_hash" == subs[0]) {
    return std::make_unique<PrefixCacheHashSha256>(block_size);
  } else if ("murmur_hash3" == subs[0]) {
    return std::make_unique<PrefixCacheHashMurmur3>(block_size,
                                                    enbale_service_routing);
  } else {
    return nullptr;
  }
}

}  // namespace xllm
