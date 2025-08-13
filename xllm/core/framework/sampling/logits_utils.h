#pragma once
#include <torch/torch.h>

#include <memory>
#include <vector>

#include "kernels/npu/xllm_ops/top_k_top_p.h"

namespace xllm {

void apply_frequency_presence_penalties(
    torch::Tensor& logits,
    const torch::Tensor& unique_token_ids,
    const torch::Tensor& unique_token_counts,
    const torch::Tensor& frequency_penalties,
    const torch::Tensor& presence_penalties);

void apply_repetition_penalties(torch::Tensor& logits,
                                const torch::Tensor& unique_token_ids,
                                const torch::Tensor& penalties);

void apply_temperatures(torch::Tensor& logits,
                        const torch::Tensor& temperatures);

void apply_top_k_top_p(torch::Tensor& logits,
                       const torch::Tensor& top_k,
                       const torch::Tensor& top_p);

}  // namespace xllm
