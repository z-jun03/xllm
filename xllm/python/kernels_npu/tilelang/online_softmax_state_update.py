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
DEFAULT_HEAD_DIM = 128
FUSION_ATTENTION_STAT_WIDTH = 8
ROWS_PER_VECTOR_TASK = 48
VEC_NUM = 2


def build_online_softmax_state_update_kernel(*, vec_core_num: int):
    if vec_core_num <= 0 or vec_core_num % VEC_NUM != 0:
        raise ValueError(
            f"vec_core_num must be positive and divisible by {VEC_NUM}, got {vec_core_num}"
        )

    block_num = vec_core_num // VEC_NUM

    @T.prim_func
    def online_softmax_state_update(
        state_max: T.Tensor((1, SYMBOL_BUFFER_CAPACITY), "float32"),
        state_normalizer: T.Tensor((1, SYMBOL_BUFFER_CAPACITY), "float32"),
        state_weighted_value: T.Tensor(
            (1, SYMBOL_BUFFER_CAPACITY), "float32"
        ),
        tile_max_stats: T.Tensor((1, SYMBOL_BUFFER_CAPACITY), "float32"),
        tile_normalizer_stats: T.Tensor(
            (1, SYMBOL_BUFFER_CAPACITY), "float32"
        ),
        tile_output: T.Tensor((1, SYMBOL_BUFFER_CAPACITY), "bfloat16"),
        batch_size: T.int32,
        query_tokens: T.int32,
        num_heads: T.int32,
        tile_stat_stride: T.int32,
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            total_rows = batch_size * num_heads * query_tokens
            groups_per_task = (
                (total_rows + ROWS_PER_VECTOR_TASK - 1)
                // ROWS_PER_VECTOR_TASK
                + vec_core_num
                - 1
            ) // vec_core_num
            group_start = task_id * groups_per_task
            groups_left = T.if_then_else(
                group_start
                < (total_rows + ROWS_PER_VECTOR_TASK - 1)
                // ROWS_PER_VECTOR_TASK,
                (total_rows + ROWS_PER_VECTOR_TASK - 1)
                // ROWS_PER_VECTOR_TASK
                - group_start,
                0,
            )
            group_count = T.if_then_else(
                groups_left < groups_per_task, groups_left, groups_per_task
            )
            with T.Scope("V"):
                old_max_scalar = T.alloc_shared([1], "float32")
                old_normalizer_scalar = T.alloc_shared([1], "float32")
                tile_max_scalar = T.alloc_shared([1], "float32")
                tile_normalizer_scalar = T.alloc_shared([1], "float32")
                old_max = T.alloc_shared([1, DEFAULT_HEAD_DIM], "float32")
                old_normalizer = T.alloc_shared(
                    [1, DEFAULT_HEAD_DIM], "float32"
                )
                tile_max_value = T.alloc_shared(
                    [1, DEFAULT_HEAD_DIM], "float32"
                )
                tile_normalizer_value = T.alloc_shared(
                    [1, DEFAULT_HEAD_DIM], "float32"
                )
                next_max = T.alloc_shared([1, DEFAULT_HEAD_DIM], "float32")
                previous_scale = T.alloc_shared(
                    [1, DEFAULT_HEAD_DIM], "float32"
                )
                tile_scale = T.alloc_shared([1, DEFAULT_HEAD_DIM], "float32")
                next_normalizer = T.alloc_shared(
                    [1, DEFAULT_HEAD_DIM], "float32"
                )
                old_weighted_value = T.alloc_shared(
                    [1, DEFAULT_HEAD_DIM], "float32"
                )
                tile_output_bf16 = T.alloc_shared(
                    [1, DEFAULT_HEAD_DIM], "bfloat16"
                )
                tile_weighted_value = T.alloc_shared(
                    [1, DEFAULT_HEAD_DIM], "float32"
                )
                max_mask = T.alloc_ub([1, DEFAULT_HEAD_DIM], "uint32")

                for local_group in T.serial(group_count):
                    row_start = (group_start + local_group) * ROWS_PER_VECTOR_TASK
                    rows_left = T.if_then_else(
                        row_start < total_rows, total_rows - row_start, 0
                    )
                    valid_rows = T.if_then_else(
                        rows_left < ROWS_PER_VECTOR_TASK,
                        rows_left,
                        ROWS_PER_VECTOR_TASK,
                    )
                    for local_row in T.serial(valid_rows):
                        row = row_start + local_row
                        batch_index = row // (num_heads * query_tokens)
                        row_in_batch = row % (num_heads * query_tokens)
                        head_index = row_in_batch // query_tokens
                        query_index = row_in_batch % query_tokens
                        tile_output_offset = (
                            (batch_index * query_tokens + query_index)
                            * num_heads
                            * DEFAULT_HEAD_DIM
                            + head_index * DEFAULT_HEAD_DIM
                        )
                        tile_stat_offset = row * tile_stat_stride

                        T.copy(
                            state_max[0, row : row + 1], old_max_scalar[0:1]
                        )
                        T.copy(
                            state_normalizer[0, row : row + 1],
                            old_normalizer_scalar[0:1],
                        )
                        T.copy(
                            tile_max_stats[
                                0, tile_stat_offset : tile_stat_offset + 1
                            ],
                            tile_max_scalar[0:1],
                        )
                        T.copy(
                            tile_normalizer_stats[
                                0, tile_stat_offset : tile_stat_offset + 1
                            ],
                            tile_normalizer_scalar[0:1],
                        )
                        T.tile.broadcast(old_max, old_max_scalar)
                        T.tile.broadcast(old_normalizer, old_normalizer_scalar)
                        T.tile.broadcast(tile_max_value, tile_max_scalar)
                        T.tile.broadcast(
                            tile_normalizer_value, tile_normalizer_scalar
                        )
                        T.tile.compare(max_mask, old_max, tile_max_value, "GE")
                        T.tile.select(
                            next_max,
                            max_mask,
                            old_max,
                            tile_max_value,
                            "VSEL_CMPMASK_SPR",
                        )
                        T.tile.sub(previous_scale, old_max, next_max)
                        T.tile.exp(previous_scale, previous_scale)
                        T.tile.sub(tile_scale, tile_max_value, next_max)
                        T.tile.exp(tile_scale, tile_scale)
                        T.tile.mul(
                            old_normalizer, old_normalizer, previous_scale
                        )
                        T.tile.mul(
                            tile_normalizer_value,
                            tile_normalizer_value,
                            tile_scale,
                        )
                        T.tile.add(
                            next_normalizer,
                            old_normalizer,
                            tile_normalizer_value,
                        )
                        T.copy(
                            state_weighted_value[
                                0,
                                row
                                * DEFAULT_HEAD_DIM : (row + 1)
                                * DEFAULT_HEAD_DIM,
                            ],
                            old_weighted_value[0, :],
                        )
                        T.copy(
                            tile_output[
                                0,
                                tile_output_offset : tile_output_offset
                                + DEFAULT_HEAD_DIM,
                            ],
                            tile_output_bf16[0, :],
                        )
                        T.tile.cast(
                            tile_weighted_value,
                            tile_output_bf16,
                            "CAST_NONE",
                            DEFAULT_HEAD_DIM,
                        )
                        T.tile.mul(
                            old_weighted_value,
                            old_weighted_value,
                            previous_scale,
                        )
                        T.tile.mul(
                            tile_weighted_value,
                            tile_weighted_value,
                            tile_normalizer_value,
                        )
                        T.tile.add(
                            old_weighted_value,
                            old_weighted_value,
                            tile_weighted_value,
                        )
                        T.copy(next_max[0, 0:1], state_max[0, row : row + 1])
                        T.copy(
                            next_normalizer[0, 0:1],
                            state_normalizer[0, row : row + 1],
                        )
                        T.copy(
                            old_weighted_value[0, :],
                            state_weighted_value[
                                0,
                                row
                                * DEFAULT_HEAD_DIM : (row + 1)
                                * DEFAULT_HEAD_DIM,
                            ],
                        )

    return online_softmax_state_update


def online_softmax_state_update_kernel_jit():
    return build_online_softmax_state_update_kernel(vec_core_num=detect_vec_core_num())
