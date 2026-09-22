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

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

struct WanCausalConv3dInputTestParam {
  int64_t height;
  int64_t width;
  int64_t cache_temporal;
};

class WanCausalConv3dInputWrapperTest
    : public testing::TestWithParam<WanCausalConv3dInputTestParam> {};

TEST_P(WanCausalConv3dInputWrapperTest, MatchesCatPadAndCacheReference) {
  const WanCausalConv3dInputTestParam param = GetParam();
  const torch::Device device("npu:0");
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kChannels = 2;
  constexpr int64_t kHiddenTemporal = 4;

  torch::manual_seed(20260921 + param.cache_temporal + param.height);
  torch::Tensor hidden_states = torch::randn(
      {kBatchSize, kChannels, kHiddenTemporal, param.height, param.width},
      options);
  torch::Tensor feature_cache = torch::randn(
      {kBatchSize, kChannels, param.cache_temporal, param.height, param.width},
      options);

  ASSERT_TRUE(can_wan_causal_conv3d_input(hidden_states, feature_cache));
  auto [conv_input, next_cache] =
      wan_causal_conv3d_input(hidden_states, feature_cache);

  torch::Tensor concatenated = torch::cat({feature_cache, hidden_states}, 2);
  const int64_t remaining_padding = 2 - param.cache_temporal;
  torch::Tensor conv_input_reference =
      torch::nn::functional::pad(concatenated,
                                 torch::nn::functional::PadFuncOptions(
                                     {0, 0, 0, 0, remaining_padding, 0}));
  torch::Tensor next_cache_reference = hidden_states.slice(
      /*dim=*/2, kHiddenTemporal - 2, kHiddenTemporal);

  EXPECT_TRUE(torch::equal(conv_input, conv_input_reference));
  EXPECT_TRUE(torch::equal(next_cache, next_cache_reference));
}

INSTANTIATE_TEST_SUITE_P(
    DynamicSpatialShape,
    WanCausalConv3dInputWrapperTest,
    testing::Values(WanCausalConv3dInputTestParam{3, 7, 1},
                    WanCausalConv3dInputTestParam{128, 128, 1},
                    WanCausalConv3dInputTestParam{257, 513, 2},
                    WanCausalConv3dInputTestParam{320, 320, 2},
                    WanCausalConv3dInputTestParam{513, 769, 1},
                    WanCausalConv3dInputTestParam{704, 1280, 1},
                    WanCausalConv3dInputTestParam{768, 768, 2}));

TEST(WanCausalConv3dInputRegressionTest,
     RemainsExactAcrossRepeatedLargePlaneCalls) {
  const torch::Device device("npu:0");
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::Tensor hidden_states = torch::randn({1, 2, 4, 1024, 1024}, options);
  torch::Tensor feature_cache = torch::randn({1, 2, 1, 1024, 1024}, options);
  torch::Tensor conv_input_reference = torch::nn::functional::pad(
      torch::cat({feature_cache, hidden_states}, 2),
      torch::nn::functional::PadFuncOptions({0, 0, 0, 0, 1, 0}));
  torch::Tensor next_cache_reference = hidden_states.slice(2, 2, 4).clone();

  for (int32_t iteration = 0; iteration < 3; ++iteration) {
    auto [conv_input, next_cache] =
        wan_causal_conv3d_input(hidden_states, feature_cache);
    EXPECT_TRUE(torch::equal(conv_input, conv_input_reference));
    EXPECT_TRUE(torch::equal(next_cache, next_cache_reference));
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
