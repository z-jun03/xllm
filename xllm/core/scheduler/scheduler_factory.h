#pragma once

#include "runtime/xservice_client.h"
#include "scheduler/continuous_scheduler.h"

namespace xllm {

std::unique_ptr<ContinuousScheduler> create_continuous_scheduler(
    Engine* engine,
    ContinuousScheduler::Options options);

}  // namespace xllm
