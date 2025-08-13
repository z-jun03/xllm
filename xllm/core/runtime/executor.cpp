#include "executor.h"

#include <c10/core/TensorOptions.h>
#include <glog/logging.h>

#include "common/metrics.h"
#include "runtime/npu_executor_impl.h"
#include "runtime/options.h"

namespace xllm {

Executor::Executor(CausalLM* model,
                   const ModelArgs& args,
                   const torch::Device& device,
                   const runtime::Options& options) {
  impl_ = std::make_unique<NpuExecutorImpl>(model, args, device, options);
}

ForwardInput Executor::prepare_inputs(Batch& batch) {
  return impl_->prepare_inputs(batch);
}

torch::Tensor Executor::forward(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& params) {
  COUNTER_INC(num_model_execution_total_eager);
  return impl_->run(tokens, positions, kv_caches, params);
}

}  // namespace xllm
