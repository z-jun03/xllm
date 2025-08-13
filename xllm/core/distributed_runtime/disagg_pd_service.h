#pragma once

#include "common/macros.h"
#include "disagg_pd.pb.h"
#include "disagg_pd_service_impl.h"

namespace xllm {

class DisaggPrefillService : public proto::DisaggPDService {
 public:
  explicit DisaggPrefillService(DisaggPrefillScheduler* scheduler);
  virtual ~DisaggPrefillService() = default;

  // for prefill recv decode response
  void Generation(::google::protobuf::RpcController* controller,
                  const proto::DisaggStreamGeneration* request,
                  proto::Status* response,
                  ::google::protobuf::Closure* done) override;

  // for prefill recv decode response
  void Generations(::google::protobuf::RpcController* controller,
                   const proto::DisaggStreamGenerations* requests,
                   proto::StatusSet* responses,
                   ::google::protobuf::Closure* done) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(DisaggPrefillService);
  std::unique_ptr<DisaggPrefillServiceImpl> disagg_prefill_service_impl_;
};

class DisaggDecodeService : public proto::DisaggPDService {
 public:
  explicit DisaggDecodeService(DisaggDecodeScheduler* scheduler,
                               Engine* engine);
  virtual ~DisaggDecodeService() = default;

  // for decode recv prefill request
  void AddNewRequests(::google::protobuf::RpcController* controller,
                      const proto::DisaggRequests* request,
                      proto::DisaggResponses* response,
                      ::google::protobuf::Closure* done) override;

  // for decode recv first token from prefill
  void FirstGeneration(::google::protobuf::RpcController* controller,
                       const proto::DisaggGenerations* request,
                       proto::Status* response,
                       ::google::protobuf::Closure* done) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(DisaggDecodeService);
  std::unique_ptr<DisaggDecodeServiceImpl> disagg_decode_service_impl_;
};

}  // namespace xllm
