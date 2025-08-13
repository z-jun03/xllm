#pragma once
#include <c10/core/DeviceType.h>
#include <folly/system/MemoryMapping.h>
#include <torch/torch.h>

#include <memory>
#include <string_view>
#include <unordered_map>

namespace xllm {

class StateDict {
 public:
  StateDict(std::unordered_map<std::string, torch::Tensor> dict,
            const std::string& prefix = "");
  virtual ~StateDict() = default;

  // get the tensor with the given name. return nullptr if not found.
  virtual torch::Tensor get_tensor(const std::string& tensor_name) const;

  // get the sharded tensor with the given name for the given rank.
  virtual torch::Tensor get_sharded_tensor(const std::string& tensor_name,
                                           int64_t dim,
                                           int rank,
                                           int world_size) const;

  // get all the tensors whose name starts with prefix.
  // the returned tensor name will be the suffix of the original name.
  virtual StateDict get_dict_with_prefix(const std::string& prefix) const;

  // get all tensors whose name starts with prefix and apply the transform
  // for each tensor.
  using TensorTransform =
      std::function<torch::Tensor(const std::string&, const torch::Tensor&)>;
  virtual StateDict get_dict_with_prefix(const std::string& prefix,
                                         TensorTransform transform_func) const;

  size_t size() const { return dict_.size(); }

  std::string_view prefix() const { return prefix_; }

  auto begin() const { return dict_.begin(); }
  auto end() const { return dict_.end(); }

 protected:
  std::unordered_map<std::string, torch::Tensor> dict_;

  TensorTransform transform_func_ = nullptr;

  std::string prefix_;
};

class StateDictFromSafeTensor : public StateDict {
 public:
  StateDictFromSafeTensor(std::unique_ptr<folly::MemoryMapping> mem_map,
                          std::unordered_map<std::string, torch::Tensor> dict);

  static std::unique_ptr<StateDict> load(const std::string& weights_file);

 private:
  // memory mapping for safetensors
  std::unique_ptr<folly::MemoryMapping> mem_map_;
};

}  // namespace xllm
