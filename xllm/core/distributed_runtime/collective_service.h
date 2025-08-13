#pragma once

#include <absl/container/flat_hash_set.h>
#include <hccl/hccl.h>

#include <unordered_map>

#include "common/macros.h"
#include "worker.pb.h"

namespace xllm {

class CollectiveService : public proto::Collective {
 public:
  CollectiveService(int dp_group_num, int total_num, int device_idx);
  virtual ~CollectiveService() = default;

  void Sync(::google::protobuf::RpcController* controller,
            const proto::AddressInfo* request,
            proto::CommUniqueIdList* response,
            ::google::protobuf::Closure* done) override;

  // wait all worker connected
  std::unordered_map<int32_t, std::string> wait();

 private:
  DISALLOW_COPY_AND_ASSIGN(CollectiveService);

  void to_proto_list(const std::vector<HcclRootInfo>& src,
                     proto::CommUniqueIdList* dst);
  void from_proto_list(const proto::CommUniqueIdList& src,
                       std::vector<HcclRootInfo>* dst);

 private:
  int total_num_ = 0;
  std::vector<HcclRootInfo> root_infos_;
  std::mutex mutex_;
  std::unordered_map<int32_t, std::string> addrs_map_;
};

}  // namespace xllm
