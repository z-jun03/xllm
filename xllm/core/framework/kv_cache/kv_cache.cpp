#include "kv_cache.h"

namespace xllm {

KVCache::KVCache(torch::Tensor key_cache, torch::Tensor value_cache)
    : key_cache_(std::move(key_cache)), value_cache_(std::move(value_cache)) {}

torch::Tensor KVCache::get_k_cache() const { return key_cache_; }
torch::Tensor KVCache::get_v_cache() const { return value_cache_; }

}  // namespace xllm
