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

from xllm.python.kernels_npu.tilelang import qwen_image_qkv_epilogue as kernel_impl
from xllm.python.kernels_npu.tilelang import utils as tilelang_utils
from xllm.python.kernels_npu.tilelang.qwen_image_qkv_epilogue import (
    DEFAULT_DTYPE,
    DEFAULT_HEAD_DIM,
    DEFAULT_NUM_HEADS,
    _run_ref_check,
    build_qwen_image_qkv_epilogue_kernel,
    detect_vec_core_num,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class QwenImageQkvEpilogueKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("head_dim", "int32"),
        DispatchField("num_heads", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": "hd128_nh24_bf16",
            "head_dim": DEFAULT_HEAD_DIM,
            "num_heads": DEFAULT_NUM_HEADS,
            "dtype": DEFAULT_DTYPE,
        }
    ]

    @staticmethod
    def generate_source(head_dim: int, num_heads: int, dtype: str) -> str:
        tilelang.disable_cache()
        tilelang_kernel = build_qwen_image_qkv_epilogue_kernel(
            head_dim=head_dim,
            num_heads=num_heads,
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
    parser = argparse.ArgumentParser(description="Generate TileLang AscendC source for Qwen-Image QKV epilogue.")
    parser.add_argument("--output", required=True, help="Output AscendC .cpp file")
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--skip-ref-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        QwenImageQkvEpilogueKernel.generate_source(args.head_dim, args.num_heads, args.dtype),
        encoding="utf-8",
    )
    if not args.skip_ref_check:
        _run_ref_check()


if __name__ == "__main__":
    main()
