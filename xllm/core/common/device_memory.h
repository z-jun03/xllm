#pragma once
#include <torch/torch.h>

namespace xllm {

class DeviceMemory {
 public:
  DeviceMemory() = default;
  ~DeviceMemory() = default;

  // return total device memory.
  static int64_t total_memory(const torch::Device& device);

  // return available device memory.
  static int64_t available_memory(const torch::Device& device);
};

}  // namespace xllm
