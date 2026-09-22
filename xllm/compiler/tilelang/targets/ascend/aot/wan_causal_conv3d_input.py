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

from xllm.python.kernels_npu.tilelang import utils as tilelang_utils
from xllm.python.kernels_npu.tilelang import (
    wan_causal_conv3d_input as kernel_impl,
)
from xllm.python.kernels_npu.tilelang.utils import DEFAULT_ASCEND_PASS_CONFIGS
from xllm.python.kernels_npu.tilelang.wan_causal_conv3d_input import (
    SUPPORTED_DTYPES,
    SUPPORTED_TEMPORAL_VARIANTS,
    build_wan_causal_conv3d_input_kernel,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class WanCausalConv3dInputKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("dtype", "dtype"),
        DispatchField("hidden_temporal", "int32"),
        DispatchField("cache_temporal", "int32"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": f"{dtype}_t{hidden_temporal}_c{cache_temporal}",
            "dtype": dtype,
            "hidden_temporal": hidden_temporal,
            "cache_temporal": cache_temporal,
        }
        for dtype in SUPPORTED_DTYPES
        for hidden_temporal, cache_temporal in SUPPORTED_TEMPORAL_VARIANTS
    ]

    @staticmethod
    def generate_source(
        dtype: str,
        hidden_temporal: int,
        cache_temporal: int,
    ) -> str:
        tilelang.disable_cache()
        tilelang_kernel = build_wan_causal_conv3d_input_kernel(
            dtype=dtype,
            hidden_temporal=hidden_temporal,
            cache_temporal=cache_temporal,
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3,
            config=DEFAULT_ASCEND_PASS_CONFIGS,
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source
