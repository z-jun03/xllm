/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include "parallel_state.h"

namespace xllm {

torch::Tensor gather_from_model_parallel_region(
    torch::Tensor input,
    const ParallelArgs& parallel_args);

torch::Tensor reduce_from_model_parallel_region(
    torch::Tensor input,
    const ParallelArgs& parallel_args);

torch::Tensor scatter_to_model_parallel_region(
    torch::Tensor input,
    const ParallelArgs& parallel_args);

torch::Tensor gather_from_data_parallel_region(
    torch::Tensor input,
    const ParallelArgs& parallel_args);

}  // namespace xllm
