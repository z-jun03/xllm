#pragma once
#include <glog/logging.h>
#include <torch/torch.h>

#include <stdexcept>

namespace xllm_ops_utils {

inline void check_tensor_shapes_equal(const torch::Tensor& a,
                                      const torch::Tensor& b,
                                      const std::string& func_name = "") {
  if (a.sizes() != b.sizes()) {
    LOG(ERROR) << func_name << ": tensor shapes do not match. "
               << "a shape: " << a.sizes() << ", b shape: " << b.sizes();
    throw std::runtime_error(func_name + ": tensor shapes do not match");
  }
}

inline void check_tensor(const torch::Tensor& t,
                         const std::string& name,
                         const std::string& func_name = "") {
  if (!t.defined()) {
    LOG(ERROR) << func_name << ": " << name << " is not defined";
    throw std::runtime_error(func_name + ": " + name + " is not defined");
  }
  if (t.numel() == 0) {
    LOG(ERROR) << func_name << ": " << name << " is empty";
    throw std::runtime_error(func_name + ": " + name + " is empty");
  }
  if (t.data_ptr() == nullptr) {
    LOG(ERROR) << func_name << ": " << name << " data pointer is null";
    throw std::runtime_error(func_name + ": " + name + " data pointer is null");
  }
}
}  // namespace xllm_ops_utils