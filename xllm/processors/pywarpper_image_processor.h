#pragma once

#include <vector>

#include "core/framework/request/mm_input_helper.h"
#include "image_processor.h"

namespace xllm {

struct MMData;

class PyWarpperImageProcessor : public ImageProcessor {
 public:
  PyWarpperImageProcessor(const ModelArgs&);
  ~PyWarpperImageProcessor() override = default;

  bool process(const MMInput& mm_inputs, MMData& mm_datas) override;
};

}  // namespace xllm
