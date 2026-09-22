# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tilelang

from xllm.python.kernels_npu.tilelang import online_softmax_state_update as kernel_impl
from xllm.python.kernels_npu.tilelang import utils as tilelang_utils
from xllm.python.kernels_npu.tilelang.online_softmax_state_update import (
    DEFAULT_HEAD_DIM,
    build_online_softmax_state_update_kernel,
    detect_vec_core_num,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class OnlineSoftmaxStateUpdateKernel(TilelangKernel):
    DISPATCH_SCHEMA = [DispatchField("head_dim", "int32"), DispatchField("dtype", "dtype")]
    SPECIALIZATIONS = [
        {"variant_key": "hd128_bf16", "head_dim": DEFAULT_HEAD_DIM, "dtype": "bf16"}
    ]

    @staticmethod
    def generate_source(head_dim: int, dtype: str) -> str:
        if head_dim != DEFAULT_HEAD_DIM or dtype != "bf16":
            raise ValueError(f"unsupported specialization: head_dim={head_dim}, dtype={dtype}")
        tilelang.disable_cache()
        tilelang_kernel = build_online_softmax_state_update_kernel(
            vec_core_num=detect_vec_core_num()
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3, config=tilelang_utils.DEFAULT_ASCEND_PASS_CONFIGS
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source
