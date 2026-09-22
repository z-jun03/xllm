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

SYMBOL_CONV_INPUT_CAPACITY = T.symbolic("conv_input_capacity")
SYMBOL_NEXT_CACHE_CAPACITY = T.symbolic("next_cache_capacity")
SYMBOL_HIDDEN_STATES_CAPACITY = T.symbolic("hidden_states_capacity")
SYMBOL_FEATURE_CACHE_CAPACITY = T.symbolic("feature_cache_capacity")
SUPPORTED_DTYPES = ("bf16",)
TENSOR_DTYPES = {"bf16": "bfloat16"}
VEC_NUM = 2
COPY_ELEMENTS = 8192
TAIL_DMA_ELEMENTS = (4096, 2048, 1024, 512, 256, 128, 64, 32, 16)
MIN_DMA_ELEMENTS = TAIL_DMA_ELEMENTS[-1]
FINAL_DMA_UB_OFFSET = COPY_ELEMENTS - MIN_DMA_ELEMENTS
SUPPORTED_TEMPORAL_VARIANTS = ((4, 1), (4, 2))


def build_wan_causal_conv3d_input_kernel(
    *,
    dtype: str,
    hidden_temporal: int,
    cache_temporal: int,
):
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"wan_causal_conv3d_input only supports {SUPPORTED_DTYPES}, got {dtype}")
    if (hidden_temporal, cache_temporal) not in SUPPORTED_TEMPORAL_VARIANTS:
        raise ValueError(
            "Unsupported wan_causal_conv3d_input specialization: "
            f"hidden_temporal={hidden_temporal}, cache_temporal={cache_temporal}"
        )

    tensor_dtype = TENSOR_DTYPES[dtype]
    zero_temporal = 2 - cache_temporal
    output_temporal = 2 + hidden_temporal

    @T.macro
    def copy_gm_to_ub_dma(
        source,
        source_offset,
        target,
        copy_elements,
        dma_elements,
    ):
        dma_offset = (copy_elements // (2 * dma_elements)) * (2 * dma_elements)
        with T.If((copy_elements // dma_elements) % 2 == 1):
            with T.Then():
                T.copy(
                    source[
                        source_offset + dma_offset : source_offset
                        + dma_offset
                        + dma_elements
                    ],
                    target[dma_offset : dma_offset + dma_elements],
                )

    @T.macro
    def copy_tail_gm_to_ub(source, source_offset, target, copy_elements):
        copy_gm_to_ub_dma(source, source_offset, target, copy_elements, 4096)
        copy_gm_to_ub_dma(source, source_offset, target, copy_elements, 2048)
        copy_gm_to_ub_dma(source, source_offset, target, copy_elements, 1024)
        copy_gm_to_ub_dma(source, source_offset, target, copy_elements, 512)
        copy_gm_to_ub_dma(source, source_offset, target, copy_elements, 256)
        copy_gm_to_ub_dma(source, source_offset, target, copy_elements, 128)
        copy_gm_to_ub_dma(source, source_offset, target, copy_elements, 64)
        copy_gm_to_ub_dma(source, source_offset, target, copy_elements, 32)
        copy_gm_to_ub_dma(source, source_offset, target, copy_elements, 16)

        with T.If(copy_elements % MIN_DMA_ELEMENTS != 0):
            with T.Then():
                T.copy(
                    source[
                        source_offset
                        + copy_elements
                        - MIN_DMA_ELEMENTS : source_offset
                        + copy_elements
                    ],
                    target[
                        FINAL_DMA_UB_OFFSET : FINAL_DMA_UB_OFFSET
                        + MIN_DMA_ELEMENTS
                    ],
                )

    @T.macro
    def copy_ub_to_gm_dma(
        source,
        target,
        target_offset,
        copy_elements,
        dma_elements,
    ):
        dma_offset = (copy_elements // (2 * dma_elements)) * (2 * dma_elements)
        with T.If((copy_elements // dma_elements) % 2 == 1):
            with T.Then():
                T.copy(
                    source[dma_offset : dma_offset + dma_elements],
                    target[
                        target_offset + dma_offset : target_offset
                        + dma_offset
                        + dma_elements
                    ],
                )

    @T.macro
    def copy_tail_ub_to_gm(source, target, target_offset, copy_elements):
        copy_ub_to_gm_dma(source, target, target_offset, copy_elements, 4096)
        copy_ub_to_gm_dma(source, target, target_offset, copy_elements, 2048)
        copy_ub_to_gm_dma(source, target, target_offset, copy_elements, 1024)
        copy_ub_to_gm_dma(source, target, target_offset, copy_elements, 512)
        copy_ub_to_gm_dma(source, target, target_offset, copy_elements, 256)
        copy_ub_to_gm_dma(source, target, target_offset, copy_elements, 128)
        copy_ub_to_gm_dma(source, target, target_offset, copy_elements, 64)
        copy_ub_to_gm_dma(source, target, target_offset, copy_elements, 32)
        copy_ub_to_gm_dma(source, target, target_offset, copy_elements, 16)

        with T.If(copy_elements % MIN_DMA_ELEMENTS != 0):
            with T.Then():
                T.copy(
                    source[
                        FINAL_DMA_UB_OFFSET : FINAL_DMA_UB_OFFSET
                        + MIN_DMA_ELEMENTS
                    ],
                    target[
                        target_offset
                        + copy_elements
                        - MIN_DMA_ELEMENTS : target_offset
                        + copy_elements
                    ],
                )

    @T.prim_func
    def wan_causal_conv3d_input(
        hidden_states: T.Tensor((SYMBOL_HIDDEN_STATES_CAPACITY,), tensor_dtype),
        feature_cache: T.Tensor((SYMBOL_FEATURE_CACHE_CAPACITY,), tensor_dtype),
        conv_input: T.Tensor((SYMBOL_CONV_INPUT_CAPACITY,), tensor_dtype),
        next_cache: T.Tensor((SYMBOL_NEXT_CACHE_CAPACITY,), tensor_dtype),
        batch_size: T.int32,
        channels: T.int32,
        plane_elements: T.int32,
    ):
        with T.Kernel(batch_size * channels, is_npu=True) as (batch_channel, vid):
            full_chunks = plane_elements // COPY_ELEMENTS
            chunks_per_vector = (full_chunks + VEC_NUM - 1) // VEC_NUM
            chunk_start = vid * chunks_per_vector
            chunks_left = T.if_then_else(
                full_chunks > chunk_start,
                full_chunks - chunk_start,
                0,
            )
            chunk_count = T.if_then_else(
                chunks_left < chunks_per_vector,
                chunks_left,
                chunks_per_vector,
            )
            tail_elements = plane_elements - full_chunks * COPY_ELEMENTS
            tail_offset = full_chunks * COPY_ELEMENTS
            with T.Scope("V"):
                data_ub = T.alloc_ub((COPY_ELEMENTS,), tensor_dtype)
                for temporal_index in T.serial(zero_temporal):
                    for local_chunk in T.serial(chunk_count):
                        chunk_index = chunk_start + local_chunk
                        chunk_offset = chunk_index * COPY_ELEMENTS
                        output_offset = (
                            batch_channel * output_temporal + temporal_index
                        ) * plane_elements + chunk_offset
                        T.tile.fill(data_ub, 0.0)
                        T.copy(data_ub, conv_input[output_offset])

                    with T.If(vid == 0):
                        with T.Then():
                            with T.If(tail_elements > 0):
                                with T.Then():
                                    output_offset = (
                                        batch_channel * output_temporal + temporal_index
                                    ) * plane_elements + tail_offset
                                    T.tile.fill(data_ub, 0.0)
                                    copy_tail_ub_to_gm(
                                        data_ub,
                                        conv_input,
                                        output_offset,
                                        tail_elements,
                                    )

                for cache_index in T.serial(cache_temporal):
                    temporal_index = zero_temporal + cache_index
                    for local_chunk in T.serial(chunk_count):
                        chunk_index = chunk_start + local_chunk
                        chunk_offset = chunk_index * COPY_ELEMENTS
                        cache_offset = (batch_channel * cache_temporal + cache_index) * plane_elements + chunk_offset
                        output_offset = (
                            batch_channel * output_temporal + temporal_index
                        ) * plane_elements + chunk_offset
                        T.copy(feature_cache[cache_offset], data_ub)
                        T.copy(data_ub, conv_input[output_offset])

                    with T.If(vid == 0):
                        with T.Then():
                            with T.If(tail_elements > 0):
                                with T.Then():
                                    cache_offset = (
                                        batch_channel * cache_temporal + cache_index
                                    ) * plane_elements + tail_offset
                                    output_offset = (
                                        batch_channel * output_temporal + temporal_index
                                    ) * plane_elements + tail_offset
                                    copy_tail_gm_to_ub(
                                        feature_cache,
                                        cache_offset,
                                        data_ub,
                                        tail_elements,
                                    )
                                    copy_tail_ub_to_gm(
                                        data_ub,
                                        conv_input,
                                        output_offset,
                                        tail_elements,
                                    )

                for hidden_index in T.serial(hidden_temporal - 2):
                    temporal_index = 2 + hidden_index
                    for local_chunk in T.serial(chunk_count):
                        chunk_index = chunk_start + local_chunk
                        chunk_offset = chunk_index * COPY_ELEMENTS
                        hidden_offset = (batch_channel * hidden_temporal + hidden_index) * plane_elements + chunk_offset
                        output_offset = (
                            batch_channel * output_temporal + temporal_index
                        ) * plane_elements + chunk_offset
                        T.copy(hidden_states[hidden_offset], data_ub)
                        T.copy(data_ub, conv_input[output_offset])

                    with T.If(vid == 0):
                        with T.Then():
                            with T.If(tail_elements > 0):
                                with T.Then():
                                    hidden_offset = (
                                        batch_channel * hidden_temporal + hidden_index
                                    ) * plane_elements + tail_offset
                                    output_offset = (
                                        batch_channel * output_temporal + temporal_index
                                    ) * plane_elements + tail_offset
                                    copy_tail_gm_to_ub(
                                        hidden_states,
                                        hidden_offset,
                                        data_ub,
                                        tail_elements,
                                    )
                                    copy_tail_ub_to_gm(
                                        data_ub,
                                        conv_input,
                                        output_offset,
                                        tail_elements,
                                    )

                for next_cache_index in T.serial(2):
                    hidden_index = hidden_temporal - 2 + next_cache_index
                    temporal_index = 2 + hidden_index
                    for local_chunk in T.serial(chunk_count):
                        chunk_index = chunk_start + local_chunk
                        chunk_offset = chunk_index * COPY_ELEMENTS
                        hidden_offset = (batch_channel * hidden_temporal + hidden_index) * plane_elements + chunk_offset
                        next_cache_offset = (batch_channel * 2 + next_cache_index) * plane_elements + chunk_offset
                        output_offset = (
                            batch_channel * output_temporal + temporal_index
                        ) * plane_elements + chunk_offset
                        T.copy(hidden_states[hidden_offset], data_ub)
                        T.copy(data_ub, conv_input[output_offset])
                        T.copy(data_ub, next_cache[next_cache_offset])

                    with T.If(vid == 0):
                        with T.Then():
                            with T.If(tail_elements > 0):
                                with T.Then():
                                    hidden_offset = (
                                        batch_channel * hidden_temporal + hidden_index
                                    ) * plane_elements + tail_offset
                                    next_cache_offset = (
                                        batch_channel * 2 + next_cache_index
                                    ) * plane_elements + tail_offset
                                    output_offset = (
                                        batch_channel * output_temporal + temporal_index
                                    ) * plane_elements + tail_offset
                                    copy_tail_gm_to_ub(
                                        hidden_states,
                                        hidden_offset,
                                        data_ub,
                                        tail_elements,
                                    )
                                    copy_tail_ub_to_gm(
                                        data_ub,
                                        conv_input,
                                        output_offset,
                                        tail_elements,
                                    )
                                    copy_tail_ub_to_gm(
                                        data_ub,
                                        next_cache,
                                        next_cache_offset,
                                        tail_elements,
                                    )

    return wan_causal_conv3d_input
