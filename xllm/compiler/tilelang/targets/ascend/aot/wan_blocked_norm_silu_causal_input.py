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
from xllm.python.kernels_npu.tilelang import wan_blocked_norm_silu as kernel_impl
from xllm.python.kernels_npu.tilelang.wan_blocked_norm_silu import (
    CAUSAL_SPATIAL_TILES,
    DEFAULT_DTYPE,
    build_wan_blocked_norm_silu_causal_input_kernel,
    detect_vec_core_num,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class WanBlockedNormSiluCausalInputKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("cache_temporal", "int32"),
        DispatchField("spatial_tile", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": (f"cache{cache_temporal}_tile{spatial_tile}_bf16"),
            "cache_temporal": cache_temporal,
            "spatial_tile": spatial_tile,
            "dtype": DEFAULT_DTYPE,
        }
        for cache_temporal in (0, 1, 2)
        for spatial_tile in CAUSAL_SPATIAL_TILES
    ]

    @staticmethod
    def generate_source(
        cache_temporal: int,
        spatial_tile: int,
        dtype: str,
    ) -> str:
        tilelang.disable_cache()
        tilelang_kernel = build_wan_blocked_norm_silu_causal_input_kernel(
            cache_temporal=cache_temporal,
            spatial_tile=spatial_tile,
            dtype=dtype,
            vec_core_num=detect_vec_core_num(),
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3,
            config=tilelang_utils.DEFAULT_ASCEND_PASS_CONFIGS,
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source
