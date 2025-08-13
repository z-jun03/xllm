#pragma once

#include <openssl/sha.h>

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "util/slice.h"

namespace xllm {

constexpr uint32_t MURMUR_HASH3_VALUE_LEN = 16;
constexpr uint32_t SHA256_HASH_VALUE_LEN = 32;
constexpr uint32_t HASH_VALUE_MAX_LEN = 32;

struct Murmur3Key {
  uint8_t data[MURMUR_HASH3_VALUE_LEN];

  Murmur3Key() {}
  Murmur3Key(const uint8_t* const input_data) {
    memcpy(data, input_data, MURMUR_HASH3_VALUE_LEN);
  }

  bool operator==(const Murmur3Key& other) {
    return strncmp(reinterpret_cast<const char*>(data),
                   reinterpret_cast<const char*>(other.data),
                   MURMUR_HASH3_VALUE_LEN);
  }
  std::string debug_string() {
    std::string rt;
    for (int i = 0; i < MURMUR_HASH3_VALUE_LEN; i++) {
      rt += std::to_string(int64_t(data[i])) + " ";
    }
    return rt;
  }
};

struct Sha256Key {
  uint8_t data[SHA256_HASH_VALUE_LEN];

  Sha256Key() {}

  Sha256Key(const uint8_t* const input_data) {
    memcpy(data, input_data, SHA256_HASH_VALUE_LEN);
  }

  bool operator==(const Sha256Key& other) {
    return strncmp(reinterpret_cast<const char*>(data),
                   reinterpret_cast<const char*>(other.data),
                   SHA256_HASH_VALUE_LEN);
  }
};

template <class FixedStringKey>
struct FixedStringKeyHash {
  size_t operator()(const FixedStringKey& key) const {
    return std::hash<std::string_view>()(std::string_view(
        reinterpret_cast<const char*>(key.data), sizeof(key.data)));
  }
};

template <class FixedStringKey>
struct FixedStringKeyEqual {
  bool operator()(const FixedStringKey& left,
                  const FixedStringKey& right) const {
    return strncmp(reinterpret_cast<const char*>(left.data),
                   reinterpret_cast<const char*>(right.data),
                   sizeof(left.data)) == 0;
  }
};

// sha256 hash seed for first block cache
const uint8_t* sha256_hash_seed();

void sha256(const uint8_t* pre_hash_value,
            const Slice<int32_t>& token_ids,
            uint8_t* hash_value);

void murmur_hash3(const uint8_t* pre_hash_value,
                  const Slice<int32_t>& token_ids,
                  uint8_t* hash_value);

void print_hex_array(uint8_t* array, uint32_t len);

}  // namespace xllm
