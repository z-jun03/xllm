#include "npu_executor_impl.h"

#include <glog/logging.h>

#include "common/metrics.h"

namespace xllm {

NpuExecutorImpl::NpuExecutorImpl(CausalLM* model,
                                 const ModelArgs& args,
                                 const torch::Device& device,
                                 const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options) {}

ForwardInput NpuExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(options_.num_decoding_tokens(), 0, args_);
}

// tokens: [num_tokens]
// positions: [num_tokens] token pos in the sequence
// returns: [num_tokens, hidden_size]
torch::Tensor NpuExecutorImpl::run(const torch::Tensor& tokens,
                                   const torch::Tensor& positions,
                                   std::vector<KVCache>& kv_caches,
                                   const ModelInputParams& params) {
  return model_->forward(tokens, positions, kv_caches, params);
}

}  // namespace xllm
