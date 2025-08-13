#pragma once

#include <torch/torch.h>

#include "atb/atb_infer.h"

namespace xllm {

class AtbBuffer {
 public:
  explicit AtbBuffer(uint64_t bufferSize, at::Device device);
  ~AtbBuffer();
  void* get_buffer(uint64_t bufferSize);

 private:
  torch::Tensor create_attensor(uint64_t bufferSize) const;
  at::Tensor create_attensor_from_tensor_desc(
      const atb::TensorDesc& tensorDesc) const;

 private:
  void* buffer_ = nullptr;
  uint64_t buffer_size_ = 0;
  torch::Tensor at_tensor_;
  at::Device device_;
};

}  // namespace xllm
