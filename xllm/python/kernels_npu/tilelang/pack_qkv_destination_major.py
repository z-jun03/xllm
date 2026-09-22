#!/usr/bin/env python3

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

from .utils import DEFAULT_ASCEND_PASS_CONFIGS, detect_vec_core_num

SYMBOL_BUFFER_CAPACITY = T.symbolic("buffer_capacity")
SUPPORTED_DTYPES = ("bf16",)
TENSOR_DTYPES = {"bf16": "bfloat16"}
VEC_NUM = 2


def build_pack_qkv_destination_major_kernel(*, dtype: str, vec_core_num: int):
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"pack_qkv_destination_major only supports {SUPPORTED_DTYPES}, got {dtype}"
        )
    if vec_core_num <= 0 or vec_core_num % VEC_NUM != 0:
        raise ValueError(f"vec_core_num({vec_core_num}) must be positive and divisible by {VEC_NUM}")

    tensor_dtype = TENSOR_DTYPES[dtype]
    block_num = vec_core_num // VEC_NUM

    @T.prim_func
    def pack_qkv_destination_major(
        query: T.Tensor((SYMBOL_BUFFER_CAPACITY,), tensor_dtype),
        key: T.Tensor((SYMBOL_BUFFER_CAPACITY,), tensor_dtype),
        value: T.Tensor((SYMBOL_BUFFER_CAPACITY,), tensor_dtype),
        packed_output: T.Tensor((SYMBOL_BUFFER_CAPACITY,), tensor_dtype),
        batch_size: T.int32,
        shard_sequence_length: T.int32,
        global_head_num: T.int32,
        local_head_num: T.int32,
        head_size: T.int32,
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, _):
            total_elements = (
                batch_size
                * shard_sequence_length
                * global_head_num
                * head_size
                * 3
            )
            elements_per_task = (total_elements + block_num - 1) // block_num
            element_start = cid * elements_per_task
            elements_left = T.if_then_else(
                total_elements > element_start,
                total_elements - element_start,
                0,
            )
            element_count = T.if_then_else(
                elements_left < elements_per_task,
                elements_left,
                elements_per_task,
            )

            for local_element in T.serial(element_count):
                packed_offset = element_start + local_element
                qkv_offset = packed_offset % (3 * head_size)
                element_offset = packed_offset // (3 * head_size)
                local_head = element_offset % local_head_num
                element_offset = element_offset // local_head_num
                batch = element_offset % batch_size
                element_offset = element_offset // batch_size
                sequence = element_offset % shard_sequence_length
                destination_rank = element_offset // shard_sequence_length
                source_head = destination_rank * local_head_num + local_head
                source_offset = (
                    ((batch * shard_sequence_length + sequence) * global_head_num + source_head)
                    * head_size
                    + qkv_offset % head_size
                )

                if qkv_offset < head_size:
                    packed_output[packed_offset] = query[source_offset]
                elif qkv_offset < 2 * head_size:
                    packed_output[packed_offset] = key[source_offset]
                else:
                    packed_output[packed_offset] = value[source_offset]

    return pack_qkv_destination_major
