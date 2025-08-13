#pragma once

#include "disagg_pd.pb.h"

namespace xllm {

class Engine;
class Request;
class DisaggPrefillScheduler;
class DisaggDecodeScheduler;

// a class to handle disagg_pd requests
class DisaggPDServiceImplInterface {
 public:
  DisaggPDServiceImplInterface() = default;
  explicit DisaggPDServiceImplInterface(bool is_decode) {
    is_decode_ = is_decode;
  }
  virtual ~DisaggPDServiceImplInterface() = default;

  virtual void decode_recv_new_requests(const proto::DisaggRequests* request,
                                        proto::DisaggResponses* response) {}

  virtual void decode_recv_first_generation(
      const proto::DisaggGenerations* request,
      proto::Status* response) {}

  virtual bool prefill_recv_generation(
      const proto::DisaggStreamGeneration* request,
      proto::Status* response) {
    return true;
  }

  virtual void prefill_recv_generations(
      const proto::DisaggStreamGenerations* requests,
      proto::StatusSet* responses) {}

  bool is_decode() const { return is_decode_; }

 private:
  bool is_decode_ = false;
};

class DisaggPrefillServiceImpl final : public DisaggPDServiceImplInterface {
 public:
  explicit DisaggPrefillServiceImpl(DisaggPrefillScheduler* scheduler);
  ~DisaggPrefillServiceImpl() = default;

  bool prefill_recv_generation(const proto::DisaggStreamGeneration* request,
                               proto::Status* response) override;

  void prefill_recv_generations(const proto::DisaggStreamGenerations* requests,
                                proto::StatusSet* responses) override;

 private:
  DisaggPrefillScheduler* scheduler_;  // not owned
};

class DisaggDecodeServiceImpl final : public DisaggPDServiceImplInterface {
 public:
  DisaggDecodeServiceImpl(DisaggDecodeScheduler* scheduler, Engine* engine);
  ~DisaggDecodeServiceImpl() = default;

  void decode_recv_new_requests(const proto::DisaggRequests* request,
                                proto::DisaggResponses* response) override;

  void decode_recv_first_generation(const proto::DisaggGenerations* request,
                                    proto::Status* response) override;

 private:
  std::shared_ptr<Request> generate_request(const proto::DisaggRequest& req);

 private:
  DisaggDecodeScheduler* scheduler_;  // not owned
  Engine* engine_;                    // not owned
};

}  // namespace xllm
