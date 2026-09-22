/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

class StrictAdaLayerNormWrapperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() {
    torch_npu::finalize_npu();
    aclrtResetDevice(0);
    aclFinalize();
  }
};

torch::Tensor strict_bf16_reference(const torch::Tensor& input,
                                    const torch::Tensor& scale,
                                    const torch::Tensor& shift,
                                    double eps) {
  torch::Tensor normalized = torch::layer_norm(
      input, {input.size(-1)}, /*weight=*/{}, /*bias=*/{}, eps);
  torch::Tensor factor = 1 + scale;
  torch::Tensor product = normalized * factor.unsqueeze(1);
  return product + shift.unsqueeze(1);
}

TEST_F(StrictAdaLayerNormWrapperTest,
       PreservesBfloat16ModulationRoundingWithStridedInputs) {
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kSequenceLength = 128;
  constexpr int64_t kHiddenSize = 3072;
  constexpr double kEps = 1e-6;
  const torch::Device device("npu:0");
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);

  torch::manual_seed(20260915);
  torch::Tensor input =
      torch::randn({kBatchSize, kSequenceLength, kHiddenSize}, options);
  torch::Tensor modulation =
      torch::randn({kBatchSize, 3 * kHiddenSize}, options) * 0.1;
  std::vector<torch::Tensor> chunks = modulation.chunk(3, -1);
  const torch::Tensor& shift = chunks[0];
  const torch::Tensor& scale = chunks[1];

  ASSERT_FALSE(scale.is_contiguous());
  ASSERT_TRUE(can_strict_adalayer_norm(input, scale, shift));

  torch::Tensor actual = strict_adalayer_norm(input, scale, shift, kEps);
  torch::Tensor expected = strict_bf16_reference(input, scale, shift, kEps);
  torch::Tensor difference =
      (actual.to(torch::kFloat32) - expected.to(torch::kFloat32)).abs();
  torch::Tensor zero_scale = torch::zeros_like(scale);
  torch::Tensor zero_shift = torch::zeros_like(shift);
  torch::Tensor actual_normalized =
      strict_adalayer_norm(input, zero_scale, zero_shift, kEps);
  torch::Tensor expected_normalized = torch::layer_norm(
      input, {kHiddenSize}, /*weight=*/{}, /*bias=*/{}, kEps);
  torch::Tensor normalized_difference =
      (actual_normalized.to(torch::kFloat32) -
       expected_normalized.to(torch::kFloat32))
          .abs();
  torch::Tensor expected_from_actual_normalized =
      actual_normalized * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1);
  const int64_t mismatch_count =
      (actual != expected).sum().item<int64_t>();

  std::cout << "strict_to_unfused: mae="
            << difference.mean().item<double>()
            << ", max=" << difference.max().item<double>()
            << ", mismatches=" << mismatch_count
            << ", normalized_mae="
            << normalized_difference.mean().item<double>()
            << ", normalized_max="
            << normalized_difference.max().item<double>() << std::endl;
  EXPECT_TRUE(torch::equal(actual, expected_from_actual_normalized));
  EXPECT_LT(difference.mean().item<double>(), 1e-6);
  EXPECT_LE(difference.max().item<double>(), 0.015625);
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
