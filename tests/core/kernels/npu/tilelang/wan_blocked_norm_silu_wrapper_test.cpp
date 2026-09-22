#include <acl/acl.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kChannels = 96;
constexpr int64_t kTemporal = 4;
constexpr int64_t kHeight = 512;
constexpr int64_t kWidth = 512;

torch::Tensor reference_norm_silu(const torch::Tensor& input,
                                  const torch::Tensor& gamma) {
  torch::Tensor normed =
      torch::nn::functional::normalize(
          input.to(torch::kFloat32),
          torch::nn::functional::NormalizeFuncOptions().dim(1).eps(1e-12))
          .to(input.scalar_type());
  torch::Tensor scaled = normed * std::sqrt(static_cast<double>(input.size(1)));
  return torch::silu(scaled * gamma);
}

void fill_epsilon_boundary_input(torch::Tensor& input) {
  const std::vector<double> values = {0.0,
                                      1e-30,
                                      -1e-30,
                                      1e-14,
                                      -1e-14,
                                      5e-14,
                                      -5e-14,
                                      1e-13,
                                      -1e-13,
                                      1e-12,
                                      -1e-12,
                                      1.0,
                                      -1.0};
  const int64_t pattern_size = static_cast<int64_t>(values.size());
  for (int64_t column_index = 0;
       column_index < std::min(input.size(4), pattern_size);
       ++column_index) {
    input.slice(4, column_index, input.size(4), pattern_size)
        .fill_(values[column_index]);
  }
}

void expect_causal_input_matches_reference(const torch::Tensor& input,
                                           const torch::Tensor& gamma,
                                           const torch::Tensor& feature_cache) {
  ASSERT_TRUE(
      can_wan_blocked_norm_silu_causal_input(input, gamma, feature_cache));
  auto [conv_input, next_cache] =
      wan_blocked_norm_silu_causal_input(input, gamma, feature_cache);
  torch::Tensor activated = reference_norm_silu(input, gamma);
  const bool has_cache = feature_cache.defined() && feature_cache.numel() > 0;
  const int64_t cache_temporal = has_cache ? feature_cache.size(2) : 0;
  torch::Tensor combined =
      has_cache ? torch::cat({feature_cache, activated}, 2) : activated;
  torch::Tensor conv_input_reference =
      torch::nn::functional::pad(combined,
                                 torch::nn::functional::PadFuncOptions(
                                     {0, 0, 0, 0, 2 - cache_temporal, 0}));
  torch::Tensor next_cache_reference =
      combined.slice(2, std::max<int64_t>(combined.size(2) - 2, 0)).clone();

  EXPECT_TRUE(torch::isfinite(conv_input).all().item<bool>());
  EXPECT_TRUE(torch::isfinite(next_cache).all().item<bool>());
  EXPECT_TRUE(torch::equal(conv_input, conv_input_reference));
  EXPECT_TRUE(torch::equal(next_cache, next_cache_reference));
}

double measure_npu_event_ms(const std::function<void()>& fn,
                            int32_t device_id,
                            int32_t warmup_iters = 5,
                            int32_t measure_iters = 30) {
  CHECK_GT(measure_iters, 0);
  CHECK_GE(warmup_iters, 0);

  const aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  for (int32_t warmup_index = 0; warmup_index < warmup_iters; ++warmup_index) {
    fn();
  }
  CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);

  aclrtEvent start_event = nullptr;
  aclrtEvent end_event = nullptr;
  CHECK_EQ(aclrtCreateEvent(&start_event), ACL_SUCCESS);
  CHECK_EQ(aclrtCreateEvent(&end_event), ACL_SUCCESS);
  CHECK_EQ(aclrtRecordEvent(start_event, stream), ACL_SUCCESS);
  for (int32_t measure_index = 0; measure_index < measure_iters;
       ++measure_index) {
    fn();
  }
  CHECK_EQ(aclrtRecordEvent(end_event, stream), ACL_SUCCESS);
  CHECK_EQ(aclrtSynchronizeEvent(end_event), ACL_SUCCESS);

  float elapsed_ms = 0.0F;
  CHECK_EQ(aclrtEventElapsedTime(&elapsed_ms, start_event, end_event),
           ACL_SUCCESS);
  CHECK_EQ(aclrtDestroyEvent(start_event), ACL_SUCCESS);
  CHECK_EQ(aclrtDestroyEvent(end_event), ACL_SUCCESS);
  return static_cast<double>(elapsed_ms) / static_cast<double>(measure_iters);
}

struct WanBlockedNormSiluCausalInputTestParam {
  int64_t channels;
  int64_t temporal;
  int64_t height;
  int64_t width;
  int64_t cache_temporal;
};

class WanBlockedNormSiluCausalInputDynamicShapeTest
    : public testing::TestWithParam<WanBlockedNormSiluCausalInputTestParam> {};

std::vector<WanBlockedNormSiluCausalInputTestParam> dynamic_temporal_cases() {
  const std::vector<int64_t> temporal_sizes = {
      1, 2, 3, 4, 5, 8, 17, 33, 65, 129};
  std::vector<WanBlockedNormSiluCausalInputTestParam> cases;
  cases.reserve(temporal_sizes.size() * 2 * 4);
  for (int64_t temporal : temporal_sizes) {
    for (int64_t channels : {96, 192}) {
      for (int64_t cache_temporal : {-1, 0, 1, 2}) {
        cases.emplace_back(WanBlockedNormSiluCausalInputTestParam{
            channels, temporal, 17, 19, cache_temporal});
      }
    }
  }
  return cases;
}

TEST(WanBlockedNormSiluWrapperTest, MatchesOfficialBf16PathExactly) {
  const torch::Device device("npu:0");
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::manual_seed(20260920);
  torch::Tensor input =
      torch::randn({1, kChannels, kTemporal, kHeight, kWidth}, options);
  torch::Tensor gamma = torch::randn({kChannels, 1, 1, 1}, options);
  torch::Tensor blocked_input =
      at_npu::native::npu_format_cast(input, ACL_FORMAT_NDC1HWC0);

  ASSERT_TRUE(can_wan_blocked_norm_silu(blocked_input, gamma));
  torch::Tensor output = wan_blocked_norm_silu(blocked_input, gamma);
  torch::Tensor reference = reference_norm_silu(input, gamma);

  EXPECT_EQ(at_npu::native::get_npu_format(output), ACL_FORMAT_NDC1HWC0);
  EXPECT_TRUE(torch::equal(output, reference));
}

TEST(WanBlockedNormSiluWrapperTest, RepeatedCallsRemainBitwiseStable) {
  const torch::Device device("npu:0");
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::manual_seed(20260921);
  torch::Tensor input =
      torch::randn({1, kChannels, kTemporal, kHeight, kWidth}, options);
  torch::Tensor gamma = torch::randn({kChannels, 1, 1, 1}, options);
  torch::Tensor blocked_input =
      at_npu::native::npu_format_cast(input, ACL_FORMAT_NDC1HWC0);
  torch::Tensor expected = wan_blocked_norm_silu(blocked_input, gamma);

  for (int32_t iteration = 0; iteration < 3; ++iteration) {
    torch::Tensor actual = wan_blocked_norm_silu(blocked_input, gamma);
    EXPECT_TRUE(torch::equal(actual, expected));
  }
}

TEST(WanBlockedNormSiluWrapperTest, PreservesDownstreamConv3dOutput) {
  const torch::Device device("npu:0");
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::manual_seed(20260922);
  torch::Tensor input =
      torch::randn({1, kChannels, kTemporal, kHeight, kWidth}, options);
  torch::Tensor gamma = torch::randn({kChannels, 1, 1, 1}, options);
  torch::Tensor weight =
      torch::randn({kChannels, kChannels, 3, 3, 3}, options) * 0.01;
  torch::Tensor bias = torch::randn({kChannels}, options) * 0.01;
  torch::Tensor blocked_input =
      at_npu::native::npu_format_cast(input, ACL_FORMAT_NDC1HWC0);

  torch::Tensor output = wan_blocked_norm_silu(blocked_input, gamma);
  torch::Tensor actual = torch::nn::functional::conv3d(
      output,
      weight,
      torch::nn::functional::Conv3dFuncOptions().bias(bias).padding(1));
  torch::Tensor expected = torch::nn::functional::conv3d(
      reference_norm_silu(input, gamma),
      weight,
      torch::nn::functional::Conv3dFuncOptions().bias(bias).padding(1));

  EXPECT_TRUE(torch::equal(actual, expected));
}

TEST(WanBlockedNormSiluWrapperTest, MatchesEpsilonBoundaryExactly) {
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device("npu:0");
  torch::Tensor input =
      torch::zeros({1, kChannels, kTemporal, kHeight, kWidth}, options);
  torch::Tensor gamma =
      torch::linspace(-1.5, 1.5, kChannels, options).view({kChannels, 1, 1, 1});
  for (bool use_boundary_pattern : {false, true}) {
    SCOPED_TRACE(use_boundary_pattern);
    if (use_boundary_pattern) {
      fill_epsilon_boundary_input(input);
    }
    torch::Tensor blocked_input =
        at_npu::native::npu_format_cast(input, ACL_FORMAT_NDC1HWC0);
    ASSERT_TRUE(can_wan_blocked_norm_silu(blocked_input, gamma));
    torch::Tensor actual = wan_blocked_norm_silu(blocked_input, gamma);
    EXPECT_TRUE(torch::isfinite(actual).all().item<bool>());
    EXPECT_TRUE(torch::equal(actual, reference_norm_silu(input, gamma)));
  }
}

TEST(WanBlockedNormSiluCausalInputWrapperTest,
     MatchesCachePadAndDownstreamConv3dExactly) {
  const torch::Device device("npu:0");
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::manual_seed(20260923);
  torch::Tensor input =
      torch::randn({1, kChannels, kTemporal, kHeight, kWidth}, options);
  torch::Tensor gamma = torch::randn({kChannels, 1, 1, 1}, options);
  torch::Tensor activated = reference_norm_silu(input, gamma);
  torch::Tensor weight =
      torch::randn({kChannels, kChannels, 3, 3, 3}, options) * 0.01;
  torch::Tensor bias = torch::randn({kChannels}, options) * 0.01;

  for (int64_t cache_temporal = 0; cache_temporal <= 2; ++cache_temporal) {
    torch::Tensor feature_cache;
    torch::Tensor cache_reference;
    if (cache_temporal > 0) {
      cache_reference = torch::randn(
          {1, kChannels, cache_temporal, kHeight, kWidth}, options);
      feature_cache = cache_reference;
    }
    ASSERT_TRUE(
        can_wan_blocked_norm_silu_causal_input(input, gamma, feature_cache));
    auto [conv_input, next_cache] =
        wan_blocked_norm_silu_causal_input(input, gamma, feature_cache);

    torch::Tensor prefix =
        cache_temporal > 0
            ? cache_reference
            : torch::empty({1, kChannels, 0, kHeight, kWidth}, options);
    torch::Tensor conv_input_reference =
        torch::nn::functional::pad(torch::cat({prefix, activated}, 2),
                                   torch::nn::functional::PadFuncOptions(
                                       {0, 0, 0, 0, 2 - cache_temporal, 0}));
    torch::Tensor next_cache_reference = activated.slice(2, 2, 4).clone();
    EXPECT_TRUE(torch::equal(conv_input, conv_input_reference));
    EXPECT_TRUE(torch::equal(next_cache, next_cache_reference));

    torch::Tensor actual = torch::nn::functional::conv3d(
        conv_input,
        weight,
        torch::nn::functional::Conv3dFuncOptions().bias(bias).padding(
            {0, 1, 1}));
    torch::Tensor expected = torch::nn::functional::conv3d(
        conv_input_reference,
        weight,
        torch::nn::functional::Conv3dFuncOptions().bias(bias).padding(
            {0, 1, 1}));
    EXPECT_TRUE(torch::equal(actual, expected));
  }
}

TEST(WanBlockedNormSiluCausalInputWrapperTest,
     BenchmarksOfficialAndFusedPaths) {
  struct BenchmarkCase {
    std::string name;
    int64_t channels;
    int64_t temporal;
    int64_t height;
    int64_t width;
    int64_t cache_temporal;
  };
  const std::vector<BenchmarkCase> benchmark_cases = {
      {"aligned_t4_512x512", 96, 4, 512, 512, 1},
      {"tail_t4_65x65", 96, 4, 65, 65, 1},
      {"long_t17_256x256", 96, 17, 256, 256, 1},
      {"long_t33_256x256", 96, 33, 256, 256, 2},
  };
  const torch::Device device("npu:0");
  const int32_t device_id = device.index();
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);

  for (const BenchmarkCase& benchmark_case : benchmark_cases) {
    torch::manual_seed(20260921 + benchmark_case.temporal +
                       benchmark_case.height + benchmark_case.width);
    torch::Tensor input = torch::randn({1,
                                        benchmark_case.channels,
                                        benchmark_case.temporal,
                                        benchmark_case.height,
                                        benchmark_case.width},
                                       options);
    torch::Tensor gamma =
        torch::randn({benchmark_case.channels, 1, 1, 1}, options);
    torch::Tensor feature_cache = torch::randn({1,
                                                benchmark_case.channels,
                                                benchmark_case.cache_temporal,
                                                benchmark_case.height,
                                                benchmark_case.width},
                                               options);

    auto [fused_conv_input, fused_next_cache] =
        wan_blocked_norm_silu_causal_input(input, gamma, feature_cache);
    torch::Tensor activated = reference_norm_silu(input, gamma);
    torch::Tensor reference_conv_input = torch::nn::functional::pad(
        torch::cat({feature_cache, activated}, 2),
        torch::nn::functional::PadFuncOptions(
            {0, 0, 0, 0, 2 - benchmark_case.cache_temporal, 0}));
    torch::Tensor reference_next_cache =
        activated.slice(2, benchmark_case.temporal - 2, benchmark_case.temporal)
            .clone();
    ASSERT_TRUE(torch::equal(fused_conv_input, reference_conv_input));
    ASSERT_TRUE(torch::equal(fused_next_cache, reference_next_cache));

    const double official_ms = measure_npu_event_ms(
        [&]() {
          torch::Tensor official_activated = reference_norm_silu(input, gamma);
          [[maybe_unused]] torch::Tensor official_conv_input =
              torch::nn::functional::pad(
                  torch::cat({feature_cache, official_activated}, 2),
                  torch::nn::functional::PadFuncOptions(
                      {0, 0, 0, 0, 2 - benchmark_case.cache_temporal, 0}));
          [[maybe_unused]] torch::Tensor official_next_cache =
              official_activated
                  .slice(
                      2, benchmark_case.temporal - 2, benchmark_case.temporal)
                  .clone();
        },
        device_id);
    const double fused_ms = measure_npu_event_ms(
        [&]() {
          [[maybe_unused]] auto fused_outputs =
              wan_blocked_norm_silu_causal_input(input, gamma, feature_cache);
        },
        device_id);
    const double speedup = fused_ms > 0.0 ? official_ms / fused_ms : 0.0;
    std::cout << "[wan_blocked_norm_silu_wrapper_test] case="
              << benchmark_case.name << ", official_ms=" << official_ms
              << ", fused_ms=" << fused_ms << ", speedup=" << speedup << "x"
              << std::endl;
  }
}

TEST_P(WanBlockedNormSiluCausalInputDynamicShapeTest, MatchesReferenceExactly) {
  const WanBlockedNormSiluCausalInputTestParam param = GetParam();
  const torch::Device device("npu:0");
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::manual_seed(20260921 + param.channels + param.height + param.width);
  torch::Tensor input = torch::randn(
      {1, param.channels, param.temporal, param.height, param.width}, options);
  torch::Tensor gamma = torch::randn({param.channels, 1, 1, 1}, options);
  torch::Tensor feature_cache;
  if (param.cache_temporal >= 0) {
    feature_cache = torch::randn(
        {1, param.channels, param.cache_temporal, param.height, param.width},
        options);
  }

  expect_causal_input_matches_reference(input, gamma, feature_cache);
}

TEST_P(WanBlockedNormSiluCausalInputDynamicShapeTest,
       MatchesEpsilonBoundaryExactly) {
  const WanBlockedNormSiluCausalInputTestParam param = GetParam();
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device("npu:0");
  torch::manual_seed(20260921 + param.temporal + param.channels);
  torch::Tensor input = torch::zeros(
      {1, param.channels, param.temporal, param.height, param.width}, options);
  torch::Tensor gamma = torch::linspace(-1.5, 1.5, param.channels, options)
                            .view({param.channels, 1, 1, 1});
  torch::Tensor feature_cache;
  if (param.cache_temporal >= 0) {
    feature_cache = torch::randn(
        {1, param.channels, param.cache_temporal, param.height, param.width},
        options);
  }
  expect_causal_input_matches_reference(input, gamma, feature_cache);
  fill_epsilon_boundary_input(input);
  expect_causal_input_matches_reference(input, gamma, feature_cache);
}

TEST(WanBlockedNormSiluCausalInputWrapperTest, RejectsEmptyTemporalInput) {
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device("npu:0");
  torch::Tensor input = torch::empty({1, 96, 0, 17, 19}, options);
  torch::Tensor gamma = torch::ones({96, 1, 1, 1}, options);
  EXPECT_FALSE(can_wan_blocked_norm_silu_causal_input(input, gamma, {}));
}

TEST(WanBlockedNormSiluCausalInputWrapperTest,
     PreservesVariableLengthCacheChain) {
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device("npu:0");
  torch::manual_seed(20260921);
  torch::Tensor gamma = torch::randn({96, 1, 1, 1}, options);
  torch::Tensor feature_cache;
  torch::Tensor reference_cache;
  for (int64_t temporal : {1, 1, 17, 1, 33, 2, 4}) {
    SCOPED_TRACE(temporal);
    torch::Tensor input = torch::randn({1, 96, temporal, 65, 65}, options);
    ASSERT_TRUE(
        can_wan_blocked_norm_silu_causal_input(input, gamma, feature_cache));
    auto [conv_input, next_cache] =
        wan_blocked_norm_silu_causal_input(input, gamma, feature_cache);
    torch::Tensor activated = reference_norm_silu(input, gamma);
    const int64_t cache_temporal =
        reference_cache.defined() ? reference_cache.size(2) : 0;
    torch::Tensor combined = reference_cache.defined()
                                 ? torch::cat({reference_cache, activated}, 2)
                                 : activated;
    torch::Tensor expected_input =
        torch::nn::functional::pad(combined,
                                   torch::nn::functional::PadFuncOptions(
                                       {0, 0, 0, 0, 2 - cache_temporal, 0}));
    reference_cache =
        combined.slice(2, std::max<int64_t>(combined.size(2) - 2, 0)).clone();
    EXPECT_TRUE(torch::equal(conv_input, expected_input));
    EXPECT_TRUE(torch::equal(next_cache, reference_cache));
    feature_cache = next_cache;
  }
}

INSTANTIATE_TEST_SUITE_P(DynamicTemporalShape,
                         WanBlockedNormSiluCausalInputDynamicShapeTest,
                         testing::ValuesIn(dynamic_temporal_cases()));

INSTANTIATE_TEST_SUITE_P(
    DynamicSpatialShape,
    WanBlockedNormSiluCausalInputDynamicShapeTest,
    testing::Values(WanBlockedNormSiluCausalInputTestParam{96, 4, 65, 65, 1},
                    WanBlockedNormSiluCausalInputTestParam{192, 4, 65, 67, 2},
                    WanBlockedNormSiluCausalInputTestParam{96, 4, 129, 131, 1},
                    WanBlockedNormSiluCausalInputTestParam{96, 4, 3, 7, 0},
                    WanBlockedNormSiluCausalInputTestParam{192, 4, 5, 7, 1},
                    WanBlockedNormSiluCausalInputTestParam{96, 4, 9, 9, 2},
                    WanBlockedNormSiluCausalInputTestParam{192, 4, 17, 19, 2},
                    WanBlockedNormSiluCausalInputTestParam{96, 4, 33, 33, 1},
                    WanBlockedNormSiluCausalInputTestParam{192, 4, 45, 46, 0},
                    WanBlockedNormSiluCausalInputTestParam{96, 17, 17, 19, 2},
                    WanBlockedNormSiluCausalInputTestParam{192, 33, 9, 9, 1}));

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
