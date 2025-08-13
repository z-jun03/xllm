#pragma once
#include <cstdint>

#include "common/macros.h"
#include "common/types.h"
#include "util/threadpool.h"

namespace xllm {

class BlockManager;
class Request;
class Sequence;
class Tokenizer;
class AsyncResponseProcessor final {
 public:
  AsyncResponseProcessor(const Tokenizer* tokenizer,
                         const std::optional<InstanceRole>& role,
                         bool enable_schedule_overlap,
                         bool enable_decode_response_to_service);
  virtual ~AsyncResponseProcessor() = default;

  void process_completed_request(std::shared_ptr<Request> request);

  void process_failed_request(std::shared_ptr<Request> request, Status status);

  // in disagg pd mode, decode send requests' responses to prefill
  void process_completed_requests(
      std::vector<std::shared_ptr<Request>>& requests);

  void process_stream_request(std::shared_ptr<Request> request);

  // in disagg pd mode, decode send requests' responses to prefill
  void process_stream_requests(
      const std::vector<std::shared_ptr<Request>>& requests);

  // wait for all responses in queue to be handled
  void wait_completion();

 private:
  DISALLOW_COPY_AND_ASSIGN(AsyncResponseProcessor);

  Tokenizer* get_tls_tokenizer();
  void batch_process_completed_requests(
      std::vector<std::shared_ptr<Request>>& requests);

 private:
  // the threadpool to handle responses
  ThreadPool response_threadpool_;

  // the threadpool to handle rpc
  ThreadPool rpc_threadpool_;

  // tokenizer instance to decode token ids
  std::unique_ptr<Tokenizer> tokenizer_;

  InstanceRole role_ = InstanceRole::DEFAULT;

  bool enable_schedule_overlap_ = false;

  // for decode instance in disagg pd mode,
  // `True` means decode instance will response batch request_outputs to
  // prefill or xllm service, this will decrease rpc cost.
  // User can set the flag with env `ENABLE_PD_DECODE_BATCH_RESPONSE`
  bool enable_batch_response_ = false;

  // when service receive all user requests
  // and dispatch request to prefill instances.
  // decode can response to prefill or to service directly.
  // 1.
  // service <--- prefill <--- decode
  // 2.
  // service      prefill      decode
  //   ^                         |
  //   |                         |
  //   ---------------------------
  bool enable_decode_response_to_service_ = false;
};

}  // namespace xllm
