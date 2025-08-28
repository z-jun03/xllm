/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

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