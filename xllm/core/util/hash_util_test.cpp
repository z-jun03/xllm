
#include "hash_util.h"

#include <absl/random/random.h>
#include <gtest/gtest.h>
#include <string.h>

#include <iostream>

namespace xllm {

TEST(HashUtilTest, Basic) {
  uint8_t default_hash[SHA256_HASH_VALUE_LEN];
  memset(default_hash, 0, SHA256_HASH_VALUE_LEN);

  EXPECT_NE(strncmp(reinterpret_cast<const char*>(default_hash),
                    reinterpret_cast<const char*>(sha256_hash_seed()),
                    SHA256_DIGEST_LENGTH),
            0);
}

TEST(HashUtilTest, Sha256) {
  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[SHA256_HASH_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2, 3, 4, 5};
    uint8_t hash_value_2[SHA256_HASH_VALUE_LEN];

    sha256(sha256_hash_seed(), tokens_1, hash_value_1);
    sha256(sha256_hash_seed(), tokens_2, hash_value_2);

    EXPECT_EQ(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      SHA256_HASH_VALUE_LEN),
              0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[SHA256_HASH_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2, 3, 5, 4};
    uint8_t hash_value_2[SHA256_HASH_VALUE_LEN];

    sha256(sha256_hash_seed(), tokens_1, hash_value_1);
    sha256(sha256_hash_seed(), tokens_2, hash_value_2);

    EXPECT_NE(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      SHA256_HASH_VALUE_LEN),
              0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[SHA256_HASH_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {2, 1, 3, 5, 4};
    uint8_t hash_value_2[SHA256_HASH_VALUE_LEN];

    sha256(sha256_hash_seed(), tokens_1, hash_value_1);
    sha256(sha256_hash_seed(), tokens_2, hash_value_2);

    EXPECT_NE(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      SHA256_HASH_VALUE_LEN),
              0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[SHA256_HASH_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {2, 1, 3, 5, 4};
    uint8_t hash_value_2[SHA256_HASH_VALUE_LEN];

    sha256(sha256_hash_seed(), tokens_1, hash_value_1);
    sha256(sha256_hash_seed(), tokens_2, hash_value_2);

    EXPECT_NE(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      SHA256_HASH_VALUE_LEN),
              0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[SHA256_HASH_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2, 3, 4};
    uint8_t hash_value_2[SHA256_HASH_VALUE_LEN];

    sha256(sha256_hash_seed(), tokens_1, hash_value_1);
    sha256(sha256_hash_seed(), tokens_2, hash_value_2);

    EXPECT_NE(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      SHA256_HASH_VALUE_LEN),
              0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[SHA256_HASH_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2};
    uint8_t hash_value_2[SHA256_HASH_VALUE_LEN];

    sha256(sha256_hash_seed(), tokens_1, hash_value_1);
    sha256(sha256_hash_seed(), tokens_2, hash_value_2);

    EXPECT_NE(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      SHA256_HASH_VALUE_LEN),
              0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[SHA256_HASH_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    uint8_t hash_value_2[SHA256_HASH_VALUE_LEN];

    sha256(sha256_hash_seed(), tokens_1, hash_value_1);
    sha256(sha256_hash_seed(), tokens_2, hash_value_2);

    EXPECT_NE(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      SHA256_HASH_VALUE_LEN),
              0);
  }
}

TEST(HashUtilTest, MurmurHash3) {
  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[MURMUR_HASH3_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2, 3, 4, 5};
    uint8_t hash_value_2[MURMUR_HASH3_VALUE_LEN];

    murmur_hash3(nullptr, tokens_1, hash_value_1);
    murmur_hash3(nullptr, tokens_2, hash_value_2);

    EXPECT_EQ(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      MURMUR_HASH3_VALUE_LEN),
              0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[MURMUR_HASH3_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2, 3, 5, 4};
    uint8_t hash_value_2[MURMUR_HASH3_VALUE_LEN];

    murmur_hash3(nullptr, tokens_1, hash_value_1);
    murmur_hash3(nullptr, tokens_2, hash_value_2);

    EXPECT_NE(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      MURMUR_HASH3_VALUE_LEN),
              0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[MURMUR_HASH3_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {2, 1, 3, 5, 4};
    uint8_t hash_value_2[MURMUR_HASH3_VALUE_LEN];

    murmur_hash3(nullptr, tokens_1, hash_value_1);
    murmur_hash3(nullptr, tokens_2, hash_value_2);

    EXPECT_NE(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      MURMUR_HASH3_VALUE_LEN),
              0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[MURMUR_HASH3_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {2, 1, 3, 5, 4};
    uint8_t hash_value_2[MURMUR_HASH3_VALUE_LEN];

    murmur_hash3(nullptr, tokens_1, hash_value_1);
    murmur_hash3(nullptr, tokens_2, hash_value_2);

    EXPECT_NE(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      MURMUR_HASH3_VALUE_LEN),
              0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[MURMUR_HASH3_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2, 3, 4};
    uint8_t hash_value_2[MURMUR_HASH3_VALUE_LEN];

    murmur_hash3(nullptr, tokens_1, hash_value_1);
    murmur_hash3(nullptr, tokens_2, hash_value_2);

    EXPECT_NE(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      MURMUR_HASH3_VALUE_LEN),
              0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[MURMUR_HASH3_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2};
    uint8_t hash_value_2[MURMUR_HASH3_VALUE_LEN];

    murmur_hash3(nullptr, tokens_1, hash_value_1);
    murmur_hash3(nullptr, tokens_2, hash_value_2);

    EXPECT_NE(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      MURMUR_HASH3_VALUE_LEN),
              0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[MURMUR_HASH3_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    uint8_t hash_value_2[MURMUR_HASH3_VALUE_LEN];

    murmur_hash3(nullptr, tokens_1, hash_value_1);
    murmur_hash3(nullptr, tokens_2, hash_value_2);

    EXPECT_NE(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      MURMUR_HASH3_VALUE_LEN),
              0);
  }
}

}  // namespace xllm
