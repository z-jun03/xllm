#pragma once

#include <tuple>
#include <unordered_map>
#include <vector>

#include "image_processor.h"

namespace xllm {

class Qwen2VLImageProcessor : public ImageProcessor {
 public:
  Qwen2VLImageProcessor(const ModelArgs&);
  ~Qwen2VLImageProcessor() override = default;

  bool process(const MMInput& mm_inputs, MMData& mm_datas) override;

 private:
  bool process_images(std::vector<torch::Tensor> images, MMData& mm_datas);
  bool process_image(torch::Tensor image,
                     std::vector<torch::Tensor>& pixel_values,
                     std::vector<int64_t>& grids);

 private:
  bool do_convert_rgb_ = true;
  bool do_normalize_ = true;

  bool do_rescale_ = true;
  bool do_resize_ = true;

  std::vector<double> image_mean_;
  std::vector<double> image_std_;

  int max_pixels_ = 12845056;
  int min_pixels_ = 3136;

  int merge_size_ = 2;
  int patch_size_ = 14;

  int resample_ = 3;
  double rescale_factor_ = 0.00392156862745098;

  std::unordered_map<std::string, int> size_;
  int temporal_patch_size_ = 2;
};

}  // namespace xllm
