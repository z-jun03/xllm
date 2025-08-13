#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

#include "slice.h"

namespace xllm {
namespace util {

std::pair<int, int> find_ones_indices(std::vector<int>& q_seq_lens);

template <typename T>
void pad_2d_vector(std::vector<std::vector<T>>& vec, T pad_value) {
  size_t max_col_size = 0;
  for (const auto& row : vec) {
    max_col_size = std::max(max_col_size, row.size());
  }

  for (auto& row : vec) {
    row.resize(max_col_size, pad_value);
  }
}

torch::ScalarType parse_dtype(const std::string& dtype_str,
                              const torch::Device& device);

std::optional<std::vector<uint32_t>> parse_batch_sizes(
    const std::string& batch_sizes_str);

template <typename T>
T sum(const std::vector<T>& vec) {
  if (vec.empty()) throw std::runtime_error("vector is empty.");
  return std::accumulate(vec.begin(), vec.end(), T{});
}

template <typename T>
const T& min(const std::vector<T>& vec) {
  if (vec.empty()) throw std::runtime_error("vector is empty.");
  return *std::min_element(vec.begin(), vec.end());
}

template <typename T>
const T& max(const std::vector<T>& vec) {
  if (vec.empty()) throw std::runtime_error("vector is empty.");
  return *std::max_element(vec.begin(), vec.end());
}

bool match_suffix(const Slice<int32_t>& data, const Slice<int32_t>& suffix);

}  // namespace util
}  // namespace xllm
