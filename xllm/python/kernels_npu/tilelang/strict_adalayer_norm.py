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

import tilelang.language as T

from .utils import detect_vec_core_num

SYMBOL_BUFFER_CAPACITY = T.symbolic("buffer_capacity")
DEFAULT_DTYPE = "bf16"
DEFAULT_HIDDEN_SIZE = 3072
VEC_NUM = 2


def build_strict_adalayer_norm_kernel(
    *, hidden_size: int, dtype: str, vec_core_num: int
):
    if hidden_size != DEFAULT_HIDDEN_SIZE:
        raise ValueError(
            "strict_adalayer_norm only supports "
            f"hidden_size={DEFAULT_HIDDEN_SIZE}, got {hidden_size}"
        )
    if dtype != DEFAULT_DTYPE:
        raise ValueError(
            f"strict_adalayer_norm only supports {DEFAULT_DTYPE}, got {dtype}"
        )
    if vec_core_num <= 0 or vec_core_num % VEC_NUM != 0:
        raise ValueError(
            f"vec_core_num({vec_core_num}) must be positive and divisible by {VEC_NUM}"
        )

    input_dtype = "bfloat16"
    acc_dtype = "float32"
    block_num = vec_core_num // VEC_NUM

    @T.prim_func
    def strict_adalayer_norm_kernel(
        x: T.Tensor((1, SYMBOL_BUFFER_CAPACITY), input_dtype),
        scale: T.Tensor((1, SYMBOL_BUFFER_CAPACITY), input_dtype),
        shift: T.Tensor((1, SYMBOL_BUFFER_CAPACITY), input_dtype),
        output: T.Tensor((1, SYMBOL_BUFFER_CAPACITY), input_dtype),
        row_count: T.int32,
        sequence_length: T.int32,
        modulation_stride: T.int32,
        eps: T.float32,
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            rows_per_task = (row_count + vec_core_num - 1) // vec_core_num
            row_start = task_id * rows_per_task
            rows_left = T.if_then_else(
                row_count > row_start,
                row_count - row_start,
                0,
            )
            task_row_count = T.if_then_else(
                rows_left < rows_per_task,
                rows_left,
                rows_per_task,
            )

            with T.Scope("V"):
                half_ub = T.alloc_shared((1, hidden_size), input_dtype)
                modulation_half_ub = T.alloc_shared(
                    (2, hidden_size), input_dtype
                )
                value_ub = T.alloc_shared((1, hidden_size), acc_dtype)
                work_ub = T.alloc_shared((1, hidden_size), acc_dtype)
                statistic_ub = T.alloc_shared((1,), acc_dtype)

                for local_row in T.serial(task_row_count):
                    row = row_start + local_row
                    row_offset = row * hidden_size
                    batch = row // sequence_length
                    modulation_offset = batch * modulation_stride

                    T.copy(x[0, row_offset], half_ub[0, :])
                    T.tile.cast(
                        value_ub,
                        half_ub,
                        "CAST_NONE",
                        hidden_size,
                    )

                    T.reduce_sum(value_ub, statistic_ub, dim=-1)
                    T.tile.mul(
                        statistic_ub,
                        statistic_ub,
                        1.0 / hidden_size,
                    )
                    T.tile.broadcast(work_ub, statistic_ub)
                    T.tile.sub(value_ub, value_ub, work_ub)

                    T.tile.mul(work_ub, value_ub, value_ub)
                    T.reduce_sum(work_ub, statistic_ub, dim=-1)
                    T.tile.mul(
                        statistic_ub,
                        statistic_ub,
                        1.0 / hidden_size,
                    )
                    T.tile.add(statistic_ub, statistic_ub, eps)
                    T.tile.sqrt(statistic_ub, statistic_ub)
                    T.tile.broadcast(work_ub, statistic_ub)
                    T.tile.div(value_ub, value_ub, work_ub)
                    T.tile.cast(
                        half_ub,
                        value_ub,
                        "CAST_RINT",
                        hidden_size,
                    )

                    T.copy(
                        scale[0, modulation_offset],
                        modulation_half_ub[0, :],
                    )
                    T.copy(
                        shift[0, modulation_offset],
                        modulation_half_ub[1, :],
                    )
                    T.tile.cast(
                        work_ub,
                        modulation_half_ub[0, :],
                        "CAST_NONE",
                        hidden_size,
                    )
                    T.tile.add(work_ub, work_ub, 1.0)
                    T.tile.cast(
                        modulation_half_ub[0, :],
                        work_ub,
                        "CAST_RINT",
                        hidden_size,
                    )

                    T.tile.cast(
                        value_ub,
                        half_ub,
                        "CAST_NONE",
                        hidden_size,
                    )
                    T.tile.cast(
                        work_ub,
                        modulation_half_ub[0, :],
                        "CAST_NONE",
                        hidden_size,
                    )
                    T.tile.mul(value_ub, value_ub, work_ub)
                    T.tile.cast(
                        half_ub,
                        value_ub,
                        "CAST_RINT",
                        hidden_size,
                    )

                    T.tile.cast(
                        value_ub,
                        half_ub,
                        "CAST_NONE",
                        hidden_size,
                    )
                    T.tile.cast(
                        work_ub,
                        modulation_half_ub[1, :],
                        "CAST_NONE",
                        hidden_size,
                    )
                    T.tile.add(value_ub, value_ub, work_ub)
                    T.tile.cast(
                        half_ub,
                        value_ub,
                        "CAST_RINT",
                        hidden_size,
                    )
                    T.copy(half_ub[0, :], output[0, row_offset])

    return strict_adalayer_norm_kernel


__all__ = [
    "DEFAULT_DTYPE",
    "DEFAULT_HIDDEN_SIZE",
    "build_strict_adalayer_norm_kernel",
    "detect_vec_core_num",
]
