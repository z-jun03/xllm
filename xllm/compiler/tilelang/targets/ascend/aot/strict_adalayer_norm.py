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

from xllm.python.kernels_npu.tilelang import strict_adalayer_norm as kernel_impl
from xllm.python.kernels_npu.tilelang import utils as tilelang_utils
from xllm.python.kernels_npu.tilelang.strict_adalayer_norm import (
    DEFAULT_DTYPE,
    DEFAULT_HIDDEN_SIZE,
    build_strict_adalayer_norm_kernel,
    detect_vec_core_num,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class StrictAdalayerNormKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("hidden_size", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": "hs3072_bf16",
            "hidden_size": DEFAULT_HIDDEN_SIZE,
            "dtype": DEFAULT_DTYPE,
        }
    ]

    @staticmethod
    def generate_source(hidden_size: int, dtype: str) -> str:
        tilelang.disable_cache()
        tilelang_kernel = build_strict_adalayer_norm_kernel(
            hidden_size=hidden_size,
            dtype=dtype,
            vec_core_num=detect_vec_core_num(),
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3,
            config=tilelang_utils.DEFAULT_ASCEND_PASS_CONFIGS,
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TileLang AscendC source for strict AdaLayerNorm."
    )
    parser.add_argument("--output", required=True, help="Output AscendC .cpp file")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        StrictAdalayerNormKernel.generate_source(args.hidden_size, args.dtype),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
