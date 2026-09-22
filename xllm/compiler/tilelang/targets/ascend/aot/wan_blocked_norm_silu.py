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

import argparse
from pathlib import Path

import tilelang

from xllm.python.kernels_npu.tilelang import utils as tilelang_utils
from xllm.python.kernels_npu.tilelang import wan_blocked_norm_silu as kernel_impl
from xllm.python.kernels_npu.tilelang.wan_blocked_norm_silu import (
    DEFAULT_CHANNELS,
    DEFAULT_DTYPE,
    DEFAULT_PLANE_ELEMENTS,
    DEFAULT_TEMPORAL,
    build_wan_blocked_norm_silu_kernel,
    detect_vec_core_num,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class WanBlockedNormSiluKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("channels", "int32"),
        DispatchField("temporal", "int32"),
        DispatchField("plane_elements", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": "c96_t4_p262144_bf16",
            "channels": DEFAULT_CHANNELS,
            "temporal": DEFAULT_TEMPORAL,
            "plane_elements": DEFAULT_PLANE_ELEMENTS,
            "dtype": DEFAULT_DTYPE,
        }
    ]

    @staticmethod
    def generate_source(
        channels: int,
        temporal: int,
        plane_elements: int,
        dtype: str,
    ) -> str:
        tilelang.disable_cache()
        tilelang_kernel = build_wan_blocked_norm_silu_kernel(
            channels=channels,
            temporal=temporal,
            plane_elements=plane_elements,
            dtype=dtype,
            vec_core_num=detect_vec_core_num(),
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3,
            config=tilelang_utils.DEFAULT_ASCEND_PASS_CONFIGS,
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TileLang AscendC source for Wan blocked norm SiLU.")
    parser.add_argument("--output", required=True, help="Output AscendC .cpp file")
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--temporal", type=int, default=DEFAULT_TEMPORAL)
    parser.add_argument("--plane-elements", type=int, default=DEFAULT_PLANE_ELEMENTS)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        WanBlockedNormSiluKernel.generate_source(
            args.channels,
            args.temporal,
            args.plane_elements,
            args.dtype,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
