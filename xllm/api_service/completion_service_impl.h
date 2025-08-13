#pragma once

#include <absl/container/flat_hash_set.h>

#include "api_service_impl.h"
#include "completion.pb.h"
#include "stream_call.h"

namespace xllm {

using CompletionCall =
    StreamCall<proto::CompletionRequest, proto::CompletionResponse>;

// a class to handle completion requests
class CompletionServiceImpl final : public APIServiceImpl<CompletionCall> {
 public:
  CompletionServiceImpl(LLMMaster* master,
                        const std::vector<std::string>& models);

  // brpc call_data needs to use shared_ptr
  void process_async_impl(std::shared_ptr<CompletionCall> call);

 private:
  DISALLOW_COPY_AND_ASSIGN(CompletionServiceImpl);
};

}  // namespace xllm
