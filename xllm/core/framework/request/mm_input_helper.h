#pragma once

#include <string>
#include <vector>

#include "core/framework/chat_template/jinja_chat_template.h"
#include "mm_data.h"
#include "multimodal.pb.h"

namespace xllm {

struct MMInputItem {
  void clear() {
    type_ = MMType::NONE;
    raw_data_.clear();
  }

  MMType type_ = MMType::NONE;

  std::string raw_data_;  // binary

  torch::Tensor decode_data_;  // image: rgb, [c,h,w], uint8
};

using MMInputItemVec = std::vector<MMInputItem>;
using MMChatMessageVec =
    ::google::protobuf::RepeatedPtrField<proto::MMChatMessage>;
using MMInputDataVec = ::google::protobuf::RepeatedPtrField<proto::MMInputData>;

struct MMInput {
  bool empty() const { return items_.empty(); }

  std::vector<torch::Tensor> get_decode_data(MMType type) const {
    std::vector<torch::Tensor> vec;

    for (const auto& item : items_) {
      if (item.type_ == type) {
        vec.emplace_back(item.decode_data_);
      }
    }
    return std::move(vec);
  }

  MMInputItemVec items_;
};

class MMHandlerSet;
class MMInputHelper {
 public:
  MMInputHelper();
  ~MMInputHelper();

  bool trans(const MMChatMessageVec& vec,
             std::vector<Message>& messages,
             MMInputItemVec& inputs);

 private:
  bool trans(const MMInputDataVec& vec,
             Message::MMContentVec& mmc,
             MMInputItemVec& input);

  std::unique_ptr<MMHandlerSet> mm_handlers_;
};

}  // namespace xllm
