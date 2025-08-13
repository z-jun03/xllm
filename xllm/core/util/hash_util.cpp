
#include "hash_util.h"

#include <MurmurHash3.h>
#include <gflags/gflags.h>

#include <boost/archive/binary_oarchive.hpp>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

DEFINE_string(sha256_hash_seed,
              "xLLM hash seed",
              "default Hash seed string for sha256 algo.");

DEFINE_uint32(murmur_hash3_seed, 1024, "default Murmur Hash seed");

namespace xllm {

const uint8_t* sha256_hash_seed() {
  static uint8_t default_hash_value[HASH_VALUE_MAX_LEN];
  static std::once_flag flag;

  std::call_once(flag, []() {
    std::stringstream ss;
    boost::archive::binary_oarchive boa(ss);
    boa << FLAGS_sha256_hash_seed;

    std::string serialized_str = ss.str();

    SHA256(reinterpret_cast<unsigned char*>(serialized_str.data()),
           serialized_str.length(),
           default_hash_value);
  });

  return default_hash_value;
}

void sha256(const uint8_t* pre_hash_value,
            const Slice<int32_t>& token_ids,
            uint8_t* hash_value) {
  std::stringstream ss;
  ss.rdbuf()->pubsetbuf(nullptr, 0);
  ss.str("");
  ss.str().reserve(8 * 1024);

  std::string sv(reinterpret_cast<const char*>(pre_hash_value),
                 SHA256_DIGEST_LENGTH);

  boost::archive::binary_oarchive boa(ss);
  boa << sv;

  for (std::size_t i = 0; i < token_ids.size(); ++i) {
    boa << token_ids[i];
  }

  std::string serialized_str = ss.str();
  SHA256(reinterpret_cast<unsigned char*>(serialized_str.data()),
         serialized_str.length(),
         hash_value);
}

void murmur_hash3(const uint8_t* pre_hash_value,
                  const Slice<int32_t>& token_ids,
                  uint8_t* hash_value) {
  if (pre_hash_value == nullptr) {
    MurmurHash3_x64_128(reinterpret_cast<const void*>(token_ids.data()),
                        sizeof(int32_t) * token_ids.size(),
                        FLAGS_murmur_hash3_seed,
                        hash_value);
  } else {
    uint8_t key[1024];

    int32_t data_len =
        sizeof(int32_t) * token_ids.size() + MURMUR_HASH3_VALUE_LEN;
    assert(sizeof(key) > data_len);

    memcpy(key, pre_hash_value, MURMUR_HASH3_VALUE_LEN);
    memcpy(key + MURMUR_HASH3_VALUE_LEN,
           reinterpret_cast<const void*>(token_ids.data()),
           sizeof(int32_t) * token_ids.size());

    // print_hex_array(key, data_len);
    MurmurHash3_x64_128(reinterpret_cast<const void*>(key),
                        data_len,
                        FLAGS_murmur_hash3_seed,
                        hash_value);
  }
}

void print_hex_array(uint8_t* array, uint32_t len) {
  for (size_t i = 0; i < len; ++i) {
    unsigned char uc = static_cast<unsigned char>(array[i]);
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(uc);

    if (i % 16 == 15) {
      std::cout << std::endl;
    }

    else {
      std::cout << " ";
    }
  }
  std::cout << std::dec << std::endl;
}

}  // namespace xllm