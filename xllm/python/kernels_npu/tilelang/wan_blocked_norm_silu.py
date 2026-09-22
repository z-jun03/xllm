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

import math

import tilelang.language as T
from tilelang import tvm

from .utils import detect_vec_core_num

DEFAULT_DTYPE = "bf16"
NORM_EPS = 1e-12
DEFAULT_CHANNELS = 96
DEFAULT_TEMPORAL = 4
DEFAULT_PLANE_ELEMENTS = 512 * 512
SPATIAL_TILE = 512
CAUSAL_SPATIAL_TILES = (4096, 2048, 1024, 512, 256, 128, 64, 32, 16)
VEC_NUM = 2
SYMBOL_CAUSAL_SOURCE_CAPACITY = T.symbolic("causal_source_capacity")
SYMBOL_CAUSAL_NORM_CAPACITY = T.symbolic("causal_norm_capacity")
SYMBOL_CAUSAL_GAMMA_CAPACITY = T.symbolic("causal_gamma_capacity")
SYMBOL_CAUSAL_CACHE_CAPACITY = T.symbolic("causal_cache_capacity")
SYMBOL_CAUSAL_CONV_INPUT_CAPACITY = T.symbolic("causal_conv_input_capacity")
SYMBOL_CAUSAL_NEXT_CACHE_CAPACITY = T.symbolic("causal_next_cache_capacity")


def build_wan_blocked_norm_silu_kernel(
    *,
    channels: int,
    temporal: int,
    plane_elements: int,
    dtype: str,
    vec_core_num: int,
):
    if dtype != DEFAULT_DTYPE:
        raise ValueError(f"wan_blocked_norm_silu only supports {DEFAULT_DTYPE}, got {dtype}")
    if channels <= 0 or channels % 16 != 0:
        raise ValueError(f"channels({channels}) must be positive and divisible by 16")
    if temporal <= 0:
        raise ValueError(f"temporal({temporal}) must be positive")
    if plane_elements <= 0 or plane_elements % SPATIAL_TILE != 0:
        raise ValueError(f"plane_elements({plane_elements}) must be divisible by {SPATIAL_TILE}")
    if vec_core_num <= 0 or vec_core_num % VEC_NUM != 0:
        raise ValueError(f"vec_core_num({vec_core_num}) must be positive and divisible by {VEC_NUM}")

    tensor_dtype = "bfloat16"
    acc_dtype = "float32"
    channel_blocks = channels // 16
    spatial_tiles = plane_elements // SPATIAL_TILE
    total_tasks = temporal * spatial_tiles
    tile_elements = SPATIAL_TILE * 16
    block_num = vec_core_num // VEC_NUM
    scale = math.sqrt(channels)

    @T.prim_func
    def wan_blocked_norm_silu(
        blocked_source: T.Tensor((channels * temporal * plane_elements,), tensor_dtype),
        norm: T.Tensor((temporal * plane_elements,), acc_dtype),
        gamma: T.Tensor((channels,), tensor_dtype),
        blocked_output: T.Tensor((channels * temporal * plane_elements,), tensor_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (block_index, vector_index):
            worker_index = block_index * VEC_NUM + vector_index
            tasks_per_worker = (total_tasks + vec_core_num - 1) // vec_core_num
            task_start = worker_index * tasks_per_worker
            tasks_left = T.if_then_else(
                total_tasks > task_start,
                total_tasks - task_start,
                0,
            )
            task_count = T.if_then_else(
                tasks_left < tasks_per_worker,
                tasks_left,
                tasks_per_worker,
            )

            with T.Scope("V"):
                input_half_ub = T.alloc_ub((SPATIAL_TILE, 16), tensor_dtype)
                output_half_ub = T.alloc_ub((SPATIAL_TILE, 16), tensor_dtype)
                input_fp32_ub = T.alloc_ub((SPATIAL_TILE, 16), acc_dtype)
                work_fp32_ub = T.alloc_ub((SPATIAL_TILE, 16), acc_dtype)
                norm_fp32_ub = T.alloc_ub((1, SPATIAL_TILE), acc_dtype)
                norm_2d_fp32_ub = T.alloc_ub((SPATIAL_TILE, 16), acc_dtype)
                gamma_half_ub = T.alloc_ub((1, 16), tensor_dtype)
                gamma_2d_half_ub = T.alloc_ub((SPATIAL_TILE, 16), tensor_dtype)

                for local_task in T.serial(task_count):
                    task_index = task_start + local_task
                    temporal_index = task_index // spatial_tiles
                    tile_index = task_index % spatial_tiles
                    spatial_offset = tile_index * SPATIAL_TILE
                    norm_offset = temporal_index * plane_elements + spatial_offset
                    T.copy(
                        norm[norm_offset : norm_offset + SPATIAL_TILE],
                        norm_fp32_ub[0, :],
                    )
                    T.tile.max(norm_fp32_ub, norm_fp32_ub, NORM_EPS)
                    for spatial_index in T.serial(SPATIAL_TILE):
                        T.tile.fill(
                            norm_2d_fp32_ub[spatial_index, :],
                            norm_fp32_ub[0, spatial_index],
                        )

                    for channel_block in T.serial(channel_blocks):
                        input_offset = (
                            (temporal_index * channel_blocks + channel_block) * plane_elements + spatial_offset
                        ) * 16
                        T.copy(
                            blocked_source[input_offset : input_offset + tile_elements],
                            input_half_ub,
                        )
                        T.tile.cast(
                            input_fp32_ub,
                            input_half_ub,
                            "CAST_NONE",
                            tile_elements,
                        )
                        T.tile.div(
                            input_fp32_ub,
                            input_fp32_ub,
                            norm_2d_fp32_ub,
                        )
                        T.tile.cast(
                            output_half_ub,
                            input_fp32_ub,
                            "CAST_RINT",
                            tile_elements,
                        )
                        T.tile.cast(
                            input_fp32_ub,
                            output_half_ub,
                            "CAST_NONE",
                            tile_elements,
                        )
                        T.tile.mul(input_fp32_ub, input_fp32_ub, scale)
                        T.tile.cast(
                            output_half_ub,
                            input_fp32_ub,
                            "CAST_RINT",
                            tile_elements,
                        )
                        T.copy(
                            gamma[channel_block * 16 : (channel_block + 1) * 16],
                            gamma_half_ub,
                        )
                        T.tile.broadcast(gamma_2d_half_ub, gamma_half_ub)
                        T.tile.cast(
                            input_fp32_ub,
                            output_half_ub,
                            "CAST_NONE",
                            tile_elements,
                        )
                        T.tile.cast(
                            work_fp32_ub,
                            gamma_2d_half_ub,
                            "CAST_NONE",
                            tile_elements,
                        )
                        T.tile.mul(
                            input_fp32_ub,
                            input_fp32_ub,
                            work_fp32_ub,
                        )
                        T.tile.cast(
                            output_half_ub,
                            input_fp32_ub,
                            "CAST_RINT",
                            tile_elements,
                        )
                        T.tile.cast(
                            input_fp32_ub,
                            output_half_ub,
                            "CAST_NONE",
                            tile_elements,
                        )
                        T.tile.silu(work_fp32_ub, input_fp32_ub)
                        T.tile.cast(
                            output_half_ub,
                            work_fp32_ub,
                            "CAST_RINT",
                            tile_elements,
                        )
                        T.copy(
                            output_half_ub,
                            blocked_output[input_offset : input_offset + tile_elements],
                        )

    return wan_blocked_norm_silu


def build_default_wan_blocked_norm_silu_kernel():
    return build_wan_blocked_norm_silu_kernel(
        channels=DEFAULT_CHANNELS,
        temporal=DEFAULT_TEMPORAL,
        plane_elements=DEFAULT_PLANE_ELEMENTS,
        dtype=DEFAULT_DTYPE,
        vec_core_num=detect_vec_core_num(),
    )


def build_wan_blocked_norm_silu_causal_input_kernel(
    *,
    cache_temporal: int,
    spatial_tile: int,
    dtype: str,
    vec_core_num: int,
) -> tvm.tir.PrimFunc:
    if cache_temporal < 0 or cache_temporal > 2:
        raise ValueError(f"cache_temporal({cache_temporal}) must be in [0, 2]")
    if dtype != DEFAULT_DTYPE:
        raise ValueError(f"wan_blocked_norm_silu_causal_input only supports {DEFAULT_DTYPE}, got {dtype}")
    if spatial_tile not in CAUSAL_SPATIAL_TILES:
        raise ValueError(f"spatial_tile({spatial_tile}) must be one of {CAUSAL_SPATIAL_TILES}")
    if vec_core_num <= 0 or vec_core_num % VEC_NUM != 0:
        raise ValueError(f"vec_core_num({vec_core_num}) must be positive and divisible by {VEC_NUM}")

    tensor_dtype = "bfloat16"
    acc_dtype = "float32"
    block_num = vec_core_num // VEC_NUM

    @T.prim_func
    def wan_blocked_norm_silu_causal_input(
        source: T.Tensor((SYMBOL_CAUSAL_SOURCE_CAPACITY,), tensor_dtype),
        norm: T.Tensor((SYMBOL_CAUSAL_NORM_CAPACITY,), acc_dtype),
        gamma: T.Tensor((SYMBOL_CAUSAL_GAMMA_CAPACITY,), tensor_dtype),
        feature_cache: T.Tensor((SYMBOL_CAUSAL_CACHE_CAPACITY,), tensor_dtype),
        conv_input: T.Tensor((SYMBOL_CAUSAL_CONV_INPUT_CAPACITY,), tensor_dtype),
        next_cache: T.Tensor((SYMBOL_CAUSAL_NEXT_CACHE_CAPACITY,), tensor_dtype),
        channels: T.int32,
        plane_elements: T.int32,
        temporal: T.int32,
        spatial_begin: T.int32,
        spatial_elements: T.int32,
        scale: T.float32,
    ):
        with T.Kernel(block_num, is_npu=True) as (block_index, vector_index):
            worker_index = block_index * VEC_NUM + vector_index
            current_tasks = channels * temporal
            next_cache_temporal = T.min(temporal + cache_temporal, 2)
            spatial_tiles = (spatial_elements + spatial_tile - 1) // spatial_tile
            prefix_tasks = channels * 2
            prefix_tasks_per_worker = (prefix_tasks + vec_core_num - 1) // vec_core_num
            prefix_start = worker_index * prefix_tasks_per_worker
            prefix_left = T.if_then_else(
                prefix_tasks > prefix_start,
                prefix_tasks - prefix_start,
                0,
            )
            prefix_count = T.if_then_else(
                prefix_left < prefix_tasks_per_worker,
                prefix_left,
                prefix_tasks_per_worker,
            )
            current_tasks_per_worker = (current_tasks + vec_core_num - 1) // vec_core_num
            current_start = worker_index * current_tasks_per_worker
            current_left = T.if_then_else(
                current_tasks > current_start,
                current_tasks - current_start,
                0,
            )
            current_count = T.if_then_else(
                current_left < current_tasks_per_worker,
                current_left,
                current_tasks_per_worker,
            )
            with T.Scope("V"):
                input_half_ub = T.alloc_ub((spatial_tile,), tensor_dtype)
                output_half_ub = T.alloc_ub((spatial_tile,), tensor_dtype)
                input_fp32_ub = T.alloc_ub((spatial_tile,), acc_dtype)
                work_fp32_ub = T.alloc_ub((spatial_tile,), acc_dtype)
                norm_fp32_ub = T.alloc_ub((spatial_tile,), acc_dtype)
                gamma_half_ub = T.alloc_ub((1,), tensor_dtype)
                gamma_fp32_ub = T.alloc_ub((1,), acc_dtype)
                gamma_vector_fp32_ub = T.alloc_ub((spatial_tile,), acc_dtype)

                for local_task in T.serial(prefix_count):
                    task_index = prefix_start + local_task
                    channel_index = task_index // 2
                    temporal_index = task_index % 2
                    for tile_index in T.serial(spatial_tiles):
                        spatial_offset = T.if_then_else(
                            (tile_index + 1) * spatial_tile > spatial_elements,
                            spatial_begin + spatial_elements - spatial_tile,
                            spatial_begin + tile_index * spatial_tile,
                        )
                        output_offset = (
                            channel_index * (temporal + 2) + temporal_index
                        ) * plane_elements + spatial_offset
                        with T.If(temporal_index < 2 - cache_temporal):
                            with T.Then():
                                T.tile.fill(output_half_ub, 0.0)
                            with T.Else():
                                cache_index = temporal_index - (2 - cache_temporal)
                                cache_offset = (
                                    channel_index * cache_temporal + cache_index
                                ) * plane_elements + spatial_offset
                                T.copy(
                                    feature_cache[cache_offset : cache_offset + spatial_tile],
                                    output_half_ub,
                                )
                        T.copy(
                            output_half_ub,
                            conv_input[output_offset : output_offset + spatial_tile],
                        )
                        if cache_temporal > 0:
                            with T.If((temporal == 1) & (temporal_index == 1)), T.Then():
                                cache_offset = channel_index * 2 * plane_elements + spatial_offset
                                T.copy(
                                    output_half_ub,
                                    next_cache[cache_offset : cache_offset + spatial_tile],
                                )

                for local_task in T.serial(current_count):
                    task_index = current_start + local_task
                    channel_index = task_index % channels
                    temporal_index = task_index // channels
                    T.copy(gamma[channel_index : channel_index + 1], gamma_half_ub)
                    T.tile.cast(gamma_fp32_ub, gamma_half_ub, "CAST_NONE", 1)
                    T.tile.broadcast(gamma_vector_fp32_ub, gamma_fp32_ub)
                    for tile_index in T.serial(spatial_tiles):
                        spatial_offset = T.if_then_else(
                            (tile_index + 1) * spatial_tile > spatial_elements,
                            spatial_begin + spatial_elements - spatial_tile,
                            spatial_begin + tile_index * spatial_tile,
                        )
                        norm_offset = temporal_index * plane_elements + spatial_offset
                        source_offset = (channel_index * temporal + temporal_index) * plane_elements + spatial_offset
                        output_offset = (
                            channel_index * (temporal + 2) + temporal_index + 2
                        ) * plane_elements + spatial_offset
                        T.copy(
                            norm[norm_offset : norm_offset + spatial_tile],
                            norm_fp32_ub,
                        )
                        T.tile.max(norm_fp32_ub, norm_fp32_ub, NORM_EPS)
                        T.copy(
                            source[source_offset : source_offset + spatial_tile],
                            input_half_ub,
                        )
                        T.tile.cast(
                            input_fp32_ub,
                            input_half_ub,
                            "CAST_NONE",
                            spatial_tile,
                        )
                        T.tile.div(input_fp32_ub, input_fp32_ub, norm_fp32_ub)
                        T.tile.cast(
                            output_half_ub,
                            input_fp32_ub,
                            "CAST_RINT",
                            spatial_tile,
                        )
                        T.tile.cast(
                            input_fp32_ub,
                            output_half_ub,
                            "CAST_NONE",
                            spatial_tile,
                        )
                        T.tile.mul(input_fp32_ub, input_fp32_ub, scale)
                        T.tile.cast(
                            output_half_ub,
                            input_fp32_ub,
                            "CAST_RINT",
                            spatial_tile,
                        )
                        T.tile.cast(
                            input_fp32_ub,
                            output_half_ub,
                            "CAST_NONE",
                            spatial_tile,
                        )
                        T.tile.mul(
                            input_fp32_ub,
                            input_fp32_ub,
                            gamma_vector_fp32_ub,
                        )
                        T.tile.cast(
                            output_half_ub,
                            input_fp32_ub,
                            "CAST_RINT",
                            spatial_tile,
                        )
                        T.tile.cast(
                            input_fp32_ub,
                            output_half_ub,
                            "CAST_NONE",
                            spatial_tile,
                        )
                        T.tile.silu(work_fp32_ub, input_fp32_ub)
                        T.tile.cast(
                            output_half_ub,
                            work_fp32_ub,
                            "CAST_RINT",
                            spatial_tile,
                        )
                        T.copy(
                            output_half_ub,
                            conv_input[output_offset : output_offset + spatial_tile],
                        )
                        with T.If(temporal_index >= temporal - 2), T.Then():
                            cache_offset = (
                                channel_index * next_cache_temporal + temporal_index - temporal + next_cache_temporal
                            ) * plane_elements + spatial_offset
                            T.copy(
                                output_half_ub,
                                next_cache[cache_offset : cache_offset + spatial_tile],
                            )

    return wan_blocked_norm_silu_causal_input
