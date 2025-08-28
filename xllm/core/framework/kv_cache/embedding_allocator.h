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

#include <cstdint>
#include <vector>

namespace xllm {

class EmbeddingAllocator final {
 public:
  EmbeddingAllocator(int32_t total_embeddings,
                     int32_t embedding_dim,
                     torch::ScalarType dtype);

  ~EmbeddingAllocator();

  // disable copy, move and assign
  EmbeddingAllocator(const EmbeddingAllocator&) = delete;
  EmbeddingAllocator(EmbeddingAllocator&&) = delete;
  EmbeddingAllocator& operator=(const EmbeddingAllocator&) = delete;
  EmbeddingAllocator& operator=(EmbeddingAllocator&&) = delete;

  int32_t allocate();
  void free(int32_t embedding_id);

  void write(int32_t embedding_id, const torch::Tensor& embeddings);
  void write(const std::vector<int32_t>& embedding_ids,
             const torch::Tensor& embeddings);
  void write_validate(const std::vector<int32_t>& embedding_ids,
                      torch::Tensor&& next_tokens,
                      const torch::Tensor& embeddings);

  torch::Tensor read(int32_t embedding_id);
  torch::Tensor read(const std::vector<int32_t>& embedding_ids);

  // get number of free embeddings
  size_t num_free_embeddings() const { return num_free_embeddings_; }

  // get number of total embeddings
  size_t num_total_embeddings() const { return free_embeddings_.size(); }

  void* get_embeddings_cache_ptr() {
    return embeddings_cache_tensor_.data_ptr();
  }

 private:
  // free embedding count
  size_t num_free_embeddings_ = 0;

  // free embedding list
  std::vector<int32_t> free_embeddings_;

  // embedding cache
  std::vector<torch::Tensor> embeddings_cache_;
  torch::Tensor embeddings_cache_tensor_;
};

}  // namespace xllm
