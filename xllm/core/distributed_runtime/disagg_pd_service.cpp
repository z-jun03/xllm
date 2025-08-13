#include "disagg_pd_service.h"

#include <brpc/closure_guard.h>
#include <glog/logging.h>

namespace xllm {

DisaggPrefillService::DisaggPrefillService(DisaggPrefillScheduler* scheduler) {
  disagg_prefill_service_impl_ =
      std::make_unique<DisaggPrefillServiceImpl>(scheduler);
}

void DisaggPrefillService::Generation(
    ::google::protobuf::RpcController* controller,
    const proto::DisaggStreamGeneration* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  // receive generations from Decode
  brpc::ClosureGuard done_guard(done);
  disagg_prefill_service_impl_->prefill_recv_generation(request, response);
}

void DisaggPrefillService::Generations(
    ::google::protobuf::RpcController* controller,
    const proto::DisaggStreamGenerations* requests,
    proto::StatusSet* responses,
    ::google::protobuf::Closure* done) {
  // receive generations from Decode
  brpc::ClosureGuard done_guard(done);
  disagg_prefill_service_impl_->prefill_recv_generations(requests, responses);
}

DisaggDecodeService::DisaggDecodeService(DisaggDecodeScheduler* scheduler,
                                         Engine* engine) {
  disagg_decode_service_impl_ =
      std::make_unique<DisaggDecodeServiceImpl>(scheduler, engine);
}

void DisaggDecodeService::AddNewRequests(
    ::google::protobuf::RpcController* controller,
    const proto::DisaggRequests* request,
    proto::DisaggResponses* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  // try to allocate blocks for new requests
  disagg_decode_service_impl_->decode_recv_new_requests(request, response);
}

// TODO: support embedding later, now we only support tokens
void DisaggDecodeService::FirstGeneration(
    ::google::protobuf::RpcController* controller,
    const proto::DisaggGenerations* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  // Receive first token from Prefill, schedule the request to running queue
  brpc::ClosureGuard done_guard(done);
  disagg_decode_service_impl_->decode_recv_first_generation(request, response);
}

}  // namespace xllm
