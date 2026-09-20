/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <torch_npu/torch_npu.h>

#include <array>
#include <tuple>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kHeadDim = 128;
constexpr int64_t kNumHeads = 24;

class QwenImageQkvEpilogueWrapperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() {
    torch_npu::finalize_npu();
    aclrtResetDevice(0);
    aclFinalize();
  }
};

torch::Tensor make_freqs(int64_t num_tokens,
                         const torch::Device& device,
                         int64_t seed) {
  torch::manual_seed(seed);
  torch::Tensor angles =
      torch::randn({num_tokens, kHeadDim / 2},
                   torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor components =
      torch::stack({torch::cos(angles), torch::sin(angles)}, -1);
  return torch::view_as_complex(components).to(device).contiguous();
}

torch::Tensor make_identity_freqs(int64_t num_tokens,
                                  const torch::Device& device) {
  torch::Tensor components =
      torch::zeros({num_tokens, kHeadDim / 2, 2},
                   torch::TensorOptions().dtype(torch::kFloat32));
  components.select(-1, 0).fill_(1.0);
  return torch::view_as_complex(components).to(device).contiguous();
}

std::pair<torch::Tensor, torch::Tensor> make_preexpanded_rotary(
    const torch::Tensor& img_freqs,
    const torch::Tensor& txt_freqs) {
  torch::Tensor joint_freqs = torch::cat({txt_freqs, img_freqs}, 0);
  const int64_t sequence_length = joint_freqs.size(0);
  torch::Tensor cos = torch::real(joint_freqs)
                          .unsqueeze(0)
                          .unsqueeze(2)
                          .unsqueeze(-1)
                          .expand({-1, -1, -1, -1, 2})
                          .reshape({1, sequence_length, 1, kHeadDim});
  torch::Tensor sin = torch::imag(joint_freqs);
  torch::Tensor signed_sin =
      torch::stack({-sin, sin}, -1).reshape({1, sequence_length, 1, kHeadDim});
  return std::make_pair(cos, signed_sin);
}

torch::Tensor rmsnorm_rope_reference(const torch::Tensor& input,
                                     const torch::Tensor& weight,
                                     const torch::Tensor& freqs,
                                     double eps) {
  torch::Tensor output = input.to(torch::kFloat32);
  output = output * torch::rsqrt(output.square().mean(-1, true) +
                                 static_cast<float>(eps));
  output = output * weight.to(torch::kFloat32);
  output = output.to(input.dtype()).to(torch::kFloat32);

  torch::Tensor pairs = output.unflatten(-1, {-1, 2});
  torch::Tensor rotated =
      torch::stack({-pairs.select(-1, 1), pairs.select(-1, 0)}, -1).flatten(-2);
  torch::Tensor cos =
      torch::real(freqs).repeat_interleave(2, -1).unsqueeze(0).unsqueeze(2);
  torch::Tensor sin =
      torch::imag(freqs).repeat_interleave(2, -1).unsqueeze(0).unsqueeze(2);
  return (output * cos + rotated * sin).to(input.dtype());
}

TEST_F(QwenImageQkvEpilogueWrapperTest,
       MatchesReferenceForTokenNarrowedInputs) {
  const torch::Device device("npu:0");
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kImgTokens = 17;
  constexpr int64_t kTxtTokens = 7;
  constexpr int64_t kPadding = 1;
  constexpr double kEps = 1e-6;

  torch::manual_seed(20260914);
  auto make_qkv = [&](int64_t num_tokens) {
    std::array<torch::Tensor, 3> tensors;
    for (torch::Tensor& tensor : tensors) {
      tensor =
          torch::randn({kBatchSize, num_tokens + kPadding, kNumHeads, kHeadDim},
                       options)
              .narrow(1, 0, num_tokens);
      EXPECT_FALSE(tensor.is_contiguous());
    }
    return tensors;
  };

  std::array<torch::Tensor, 3> img_qkv = make_qkv(kImgTokens);
  std::array<torch::Tensor, 3> txt_qkv = make_qkv(kTxtTokens);
  std::array<torch::Tensor, 4> weights;
  for (torch::Tensor& weight : weights) {
    weight = torch::randn(
        {kHeadDim},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
  }
  torch::Tensor img_freqs = make_freqs(kImgTokens, device, 1);
  torch::Tensor txt_freqs = make_freqs(kTxtTokens, device, 2);
  auto [rotary_cos, rotary_sin] = make_preexpanded_rotary(img_freqs, txt_freqs);

  ASSERT_TRUE(can_qwen_image_qkv_epilogue(img_qkv[0],
                                          img_qkv[1],
                                          img_qkv[2],
                                          txt_qkv[0],
                                          txt_qkv[1],
                                          txt_qkv[2],
                                          weights[0],
                                          weights[1],
                                          weights[2],
                                          weights[3],
                                          rotary_cos,
                                          rotary_sin));

  auto [joint_q, joint_k, joint_v] = qwen_image_qkv_epilogue(img_qkv[0],
                                                             img_qkv[1],
                                                             img_qkv[2],
                                                             txt_qkv[0],
                                                             txt_qkv[1],
                                                             txt_qkv[2],
                                                             weights[0],
                                                             weights[1],
                                                             weights[2],
                                                             weights[3],
                                                             rotary_cos,
                                                             rotary_sin,
                                                             kEps,
                                                             kEps);

  torch::Tensor img_q_ref =
      rmsnorm_rope_reference(img_qkv[0], weights[0], img_freqs, kEps);
  torch::Tensor img_k_ref =
      rmsnorm_rope_reference(img_qkv[1], weights[1], img_freqs, kEps);
  torch::Tensor txt_q_ref =
      rmsnorm_rope_reference(txt_qkv[0], weights[2], txt_freqs, kEps);
  torch::Tensor txt_k_ref =
      rmsnorm_rope_reference(txt_qkv[1], weights[3], txt_freqs, kEps);
  torch::Tensor joint_q_ref = torch::cat({txt_q_ref, img_q_ref}, 1);
  torch::Tensor joint_k_ref = torch::cat({txt_k_ref, img_k_ref}, 1);
  torch::Tensor joint_v_ref = torch::cat({txt_qkv[2], img_qkv[2]}, 1);

  EXPECT_TRUE(torch::equal(joint_q, joint_q_ref));
  EXPECT_TRUE(torch::equal(joint_k, joint_k_ref));
  EXPECT_TRUE(torch::equal(joint_v, joint_v_ref));
}

TEST_F(QwenImageQkvEpilogueWrapperTest,
       MatchesReferenceForPackedProjectionViews) {
  const torch::Device device("npu:0");
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kImgTokens = 19;
  constexpr int64_t kTxtTokens = 5;
  constexpr double kEps = 1e-6;

  torch::manual_seed(20260915);
  auto make_packed_qkv = [&](int64_t num_tokens) {
    torch::Tensor packed = torch::randn(
        {kBatchSize, num_tokens, 3 * kNumHeads * kHeadDim}, options);
    std::vector<torch::Tensor> chunks = packed.chunk(/*chunks=*/3, /*dim=*/-1);
    std::array<torch::Tensor, 3> tensors;
    for (int64_t index = 0; index < 3; ++index) {
      tensors[index] = chunks[index].unflatten(-1, {kNumHeads, kHeadDim});
      EXPECT_FALSE(tensors[index].is_contiguous());
    }
    return tensors;
  };

  std::array<torch::Tensor, 3> img_qkv = make_packed_qkv(kImgTokens);
  std::array<torch::Tensor, 3> txt_qkv = make_packed_qkv(kTxtTokens);
  std::array<torch::Tensor, 4> weights;
  for (torch::Tensor& weight : weights) {
    weight = torch::randn(
        {kHeadDim},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
  }
  torch::Tensor img_freqs = make_freqs(kImgTokens, device, 4);
  torch::Tensor txt_freqs = make_freqs(kTxtTokens, device, 5);

  ASSERT_TRUE(can_qwen_image_qkv_epilogue(img_qkv[0],
                                          img_qkv[1],
                                          img_qkv[2],
                                          txt_qkv[0],
                                          txt_qkv[1],
                                          txt_qkv[2],
                                          weights[0],
                                          weights[1],
                                          weights[2],
                                          weights[3],
                                          img_freqs,
                                          txt_freqs));

  auto [joint_q, joint_k, joint_v] = qwen_image_qkv_epilogue(img_qkv[0],
                                                             img_qkv[1],
                                                             img_qkv[2],
                                                             txt_qkv[0],
                                                             txt_qkv[1],
                                                             txt_qkv[2],
                                                             weights[0],
                                                             weights[1],
                                                             weights[2],
                                                             weights[3],
                                                             img_freqs,
                                                             txt_freqs,
                                                             kEps,
                                                             kEps);

  torch::Tensor joint_q_ref = torch::cat(
      {rmsnorm_rope_reference(txt_qkv[0], weights[2], txt_freqs, kEps),
       rmsnorm_rope_reference(img_qkv[0], weights[0], img_freqs, kEps)},
      1);
  torch::Tensor joint_k_ref = torch::cat(
      {rmsnorm_rope_reference(txt_qkv[1], weights[3], txt_freqs, kEps),
       rmsnorm_rope_reference(img_qkv[1], weights[1], img_freqs, kEps)},
      1);
  torch::Tensor joint_v_ref = torch::cat({txt_qkv[2], img_qkv[2]}, 1);

  EXPECT_TRUE(torch::equal(joint_q, joint_q_ref));
  EXPECT_TRUE(torch::equal(joint_k, joint_k_ref));
  EXPECT_TRUE(torch::equal(joint_v, joint_v_ref));
}

TEST_F(QwenImageQkvEpilogueWrapperTest,
       PreservesOfficialRmsNormBitsForProductionShape) {
  const torch::Device device("npu:0");
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kImgTokens = 4096;
  constexpr int64_t kTxtTokens = 728;
  constexpr double kEps = 1e-6;

  torch::manual_seed(20260917);
  auto make_packed_qkv = [&](int64_t num_tokens) {
    torch::Tensor packed = torch::randn(
        {kBatchSize, num_tokens, 3 * kNumHeads * kHeadDim}, options);
    std::vector<torch::Tensor> chunks = packed.chunk(/*chunks=*/3, /*dim=*/-1);
    std::array<torch::Tensor, 3> tensors;
    for (int64_t index = 0; index < 3; ++index) {
      tensors[index] = chunks[index].unflatten(-1, {kNumHeads, kHeadDim});
    }
    return tensors;
  };

  std::array<torch::Tensor, 3> img_qkv = make_packed_qkv(kImgTokens);
  std::array<torch::Tensor, 3> txt_qkv = make_packed_qkv(kTxtTokens);
  std::array<torch::Tensor, 4> weights;
  for (torch::Tensor& weight : weights) {
    weight = torch::randn(
        {kHeadDim},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
  }
  torch::Tensor img_freqs = make_identity_freqs(kImgTokens, device);
  torch::Tensor txt_freqs = make_identity_freqs(kTxtTokens, device);

  auto [joint_q, joint_k, joint_v] = qwen_image_qkv_epilogue(img_qkv[0],
                                                             img_qkv[1],
                                                             img_qkv[2],
                                                             txt_qkv[0],
                                                             txt_qkv[1],
                                                             txt_qkv[2],
                                                             weights[0],
                                                             weights[1],
                                                             weights[2],
                                                             weights[3],
                                                             img_freqs,
                                                             txt_freqs,
                                                             kEps,
                                                             kEps);

  torch::Tensor img_q_ref = std::get<0>(
      at_npu::native::custom_ops::npu_rms_norm(img_qkv[0], weights[0], kEps));
  torch::Tensor img_k_ref = std::get<0>(
      at_npu::native::custom_ops::npu_rms_norm(img_qkv[1], weights[1], kEps));
  torch::Tensor txt_q_ref = std::get<0>(
      at_npu::native::custom_ops::npu_rms_norm(txt_qkv[0], weights[2], kEps));
  torch::Tensor txt_k_ref = std::get<0>(
      at_npu::native::custom_ops::npu_rms_norm(txt_qkv[1], weights[3], kEps));

  EXPECT_TRUE(torch::equal(joint_q, torch::cat({txt_q_ref, img_q_ref}, 1)));
  EXPECT_TRUE(torch::equal(joint_k, torch::cat({txt_k_ref, img_k_ref}, 1)));
  EXPECT_TRUE(torch::equal(joint_v, torch::cat({txt_qkv[2], img_qkv[2]}, 1)));
}

TEST_F(QwenImageQkvEpilogueWrapperTest, RejectsOversizedWorkload) {
  const torch::Device device("npu:0");
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  constexpr int64_t kImgTokens = 22000;
  constexpr int64_t kTxtTokens = 1;

  torch::Tensor img_qkv =
      torch::empty({1, kImgTokens, kNumHeads, kHeadDim}, options);
  torch::Tensor txt_qkv =
      torch::empty({1, kTxtTokens, kNumHeads, kHeadDim}, options);
  torch::Tensor weight = torch::empty(
      {kHeadDim}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  torch::Tensor img_freqs = make_freqs(kImgTokens, device, 6);
  torch::Tensor txt_freqs = make_freqs(kTxtTokens, device, 7);

  EXPECT_FALSE(can_qwen_image_qkv_epilogue(img_qkv,
                                           img_qkv,
                                           img_qkv,
                                           txt_qkv,
                                           txt_qkv,
                                           txt_qkv,
                                           weight,
                                           weight,
                                           weight,
                                           weight,
                                           img_freqs,
                                           txt_freqs));
}

TEST_F(QwenImageQkvEpilogueWrapperTest, RejectsUnsupportedHeadDim) {
  const torch::TensorOptions options = torch::TensorOptions()
                                           .dtype(torch::kBFloat16)
                                           .device(torch::Device("npu:0"));
  torch::Tensor qkv = torch::empty({1, 1, 1, 64}, options);
  torch::Tensor weight = torch::empty({64},
                                      torch::TensorOptions()
                                          .dtype(torch::kFloat32)
                                          .device(torch::Device("npu:0")));
  torch::Tensor freqs =
      make_freqs(1, torch::Device("npu:0"), 3).slice(1, 0, 32);
  EXPECT_FALSE(can_qwen_image_qkv_epilogue(qkv,
                                           qkv,
                                           qkv,
                                           qkv,
                                           qkv,
                                           qkv,
                                           weight,
                                           weight,
                                           weight,
                                           weight,
                                           freqs,
                                           freqs));
}

TEST_F(QwenImageQkvEpilogueWrapperTest, RejectsUnsupportedHeadCount) {
  const torch::Device device("npu:0");
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  constexpr int64_t kUnsupportedNumHeads = 12;
  torch::Tensor qkv =
      torch::empty({1, 1, kUnsupportedNumHeads, kHeadDim}, options);
  torch::Tensor weight = torch::empty(
      {kHeadDim}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  torch::Tensor freqs = make_freqs(1, device, 8);

  EXPECT_FALSE(can_qwen_image_qkv_epilogue(qkv,
                                           qkv,
                                           qkv,
                                           qkv,
                                           qkv,
                                           qkv,
                                           weight,
                                           weight,
                                           weight,
                                           weight,
                                           freqs,
                                           freqs));
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
