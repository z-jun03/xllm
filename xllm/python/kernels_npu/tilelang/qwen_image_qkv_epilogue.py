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
import tilelang.language as T

from scripts.logger import logger

from .utils import DEFAULT_ASCEND_PASS_CONFIGS, detect_vec_core_num

SYMBOL_BUFFER_CAPACITY = T.symbolic("buffer_capacity")
DEFAULT_DTYPE = "bf16"
DEFAULT_HEAD_DIM = 128
DEFAULT_NUM_HEADS = 24
VEC_NUM = 2
ROPE_HEADS_PER_GROUP = 12


def build_qwen_image_qkv_epilogue_kernel(*, head_dim: int, num_heads: int, dtype: str, vec_core_num: int):
    if dtype != DEFAULT_DTYPE:
        raise ValueError(f"qwen_image_qkv_epilogue only supports {DEFAULT_DTYPE}, got {dtype}")
    if head_dim != DEFAULT_HEAD_DIM:
        raise ValueError(f"qwen_image_qkv_epilogue only supports head_dim={DEFAULT_HEAD_DIM}, got {head_dim}")
    if num_heads != DEFAULT_NUM_HEADS:
        raise ValueError(f"qwen_image_qkv_epilogue only supports num_heads={DEFAULT_NUM_HEADS}, got {num_heads}")
    if vec_core_num <= 0 or vec_core_num % VEC_NUM != 0:
        raise ValueError(f"vec_core_num({vec_core_num}) must be positive and divisible by {VEC_NUM}")

    input_dtype = "bfloat16"
    acc_dtype = "float32"
    mask_dtype = "uint32"
    block_num = vec_core_num // VEC_NUM
    head_width = num_heads * head_dim
    rope_group_width = ROPE_HEADS_PER_GROUP * head_dim

    @T.macro
    def load_rotation(
        rotary_cos,
        rotary_sin,
        freq_offset,
        cos_ub,
        sin_ub,
    ):
        T.copy(rotary_cos[0, freq_offset], cos_ub[0, :])
        T.copy(rotary_sin[0, freq_offset], sin_ub[0, :])

    @T.macro
    def process_qk(
        input_tensor,
        output_tensor,
        input_offset,
        output_offset,
        group_half_ub,
        group_fp32_ub,
        cos_heads_ub,
        sin_heads_ub,
        rotated_heads_ub,
        rotate_gather_ub,
    ):
        for group_index in T.serial(num_heads // ROPE_HEADS_PER_GROUP):
            group_offset = group_index * rope_group_width
            T.copy(
                input_tensor[0, input_offset + group_offset],
                group_half_ub,
            )
            T.tile.cast(
                group_fp32_ub,
                group_half_ub,
                "CAST_NONE",
                rope_group_width,
            )
            T.tile.gather(
                rotated_heads_ub,
                group_fp32_ub,
                rotate_gather_ub,
                0,
            )
            T.tile.mul(group_fp32_ub, group_fp32_ub, cos_heads_ub)
            T.tile.mul(rotated_heads_ub, rotated_heads_ub, sin_heads_ub)
            T.tile.add(group_fp32_ub, group_fp32_ub, rotated_heads_ub)
            T.tile.cast(
                group_half_ub,
                group_fp32_ub,
                "CAST_RINT",
                rope_group_width,
            )
            T.copy(
                group_half_ub,
                output_tensor[0, output_offset + group_offset],
            )

    @T.prim_func
    def qwen_image_qkv_epilogue_batch24_kernel(
        img_q: T.Tensor((1, SYMBOL_BUFFER_CAPACITY), input_dtype),
        img_k: T.Tensor((1, SYMBOL_BUFFER_CAPACITY), input_dtype),
        img_v: T.Tensor((1, SYMBOL_BUFFER_CAPACITY), input_dtype),
        txt_q: T.Tensor((1, SYMBOL_BUFFER_CAPACITY), input_dtype),
        txt_k: T.Tensor((1, SYMBOL_BUFFER_CAPACITY), input_dtype),
        txt_v: T.Tensor((1, SYMBOL_BUFFER_CAPACITY), input_dtype),
        rotary_cos: T.Tensor((1, SYMBOL_BUFFER_CAPACITY), acc_dtype),
        rotary_sin: T.Tensor((1, SYMBOL_BUFFER_CAPACITY), acc_dtype),
        rotate_gather_offsets: T.Tensor((ROPE_HEADS_PER_GROUP * head_dim,), mask_dtype),
        joint_q: T.Tensor((1, SYMBOL_BUFFER_CAPACITY), input_dtype),
        joint_k: T.Tensor((1, SYMBOL_BUFFER_CAPACITY), input_dtype),
        joint_v: T.Tensor((1, SYMBOL_BUFFER_CAPACITY), input_dtype),
        batch_size: T.int32,
        img_tokens: T.int32,
        txt_tokens: T.int32,
        runtime_num_heads: T.int32,
        img_batch_stride: T.int32,
        img_token_stride: T.int32,
        img_v_batch_stride: T.int32,
        img_v_token_stride: T.int32,
        txt_batch_stride: T.int32,
        txt_token_stride: T.int32,
        txt_v_batch_stride: T.int32,
        txt_v_token_stride: T.int32,
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            joint_tokens = img_tokens + txt_tokens
            txt_token_works = batch_size * txt_tokens
            txt_tokens_per_task = (txt_token_works + vec_core_num - 1) // vec_core_num
            txt_token_start = task_id * txt_tokens_per_task
            txt_tokens_left = T.if_then_else(
                txt_token_works > txt_token_start,
                txt_token_works - txt_token_start,
                0,
            )
            txt_token_count = T.if_then_else(
                txt_tokens_left < txt_tokens_per_task,
                txt_tokens_left,
                txt_tokens_per_task,
            )
            img_token_works = batch_size * img_tokens
            img_tokens_per_task = (img_token_works + vec_core_num - 1) // vec_core_num
            img_token_start = task_id * img_tokens_per_task
            img_tokens_left = T.if_then_else(
                img_token_works > img_token_start,
                img_token_works - img_token_start,
                0,
            )
            img_token_count = T.if_then_else(
                img_tokens_left < img_tokens_per_task,
                img_tokens_left,
                img_tokens_per_task,
            )

            with T.Scope("V"):
                rotate_gather_offsets_ub = T.alloc_ub((ROPE_HEADS_PER_GROUP * head_dim,), mask_dtype)
                T.copy(rotate_gather_offsets, rotate_gather_offsets_ub)
                T.barrier_all()

                q_group_half_ub = T.alloc_shared((rope_group_width,), input_dtype)
                k_group_half_ub = T.alloc_shared((rope_group_width,), input_dtype)
                v_io_half_ub = T.alloc_shared((head_width,), input_dtype)
                q_group_fp32_ub = T.alloc_shared((ROPE_HEADS_PER_GROUP, head_dim), acc_dtype)
                k_group_fp32_ub = T.alloc_shared((ROPE_HEADS_PER_GROUP, head_dim), acc_dtype)
                cos_ub = T.alloc_shared((1, head_dim), acc_dtype)
                sin_ub = T.alloc_shared((1, head_dim), acc_dtype)
                cos_heads_ub = T.alloc_shared((ROPE_HEADS_PER_GROUP, head_dim), acc_dtype)
                sin_heads_ub = T.alloc_shared((ROPE_HEADS_PER_GROUP, head_dim), acc_dtype)
                rotated_heads_ub = T.alloc_shared((ROPE_HEADS_PER_GROUP, head_dim), acc_dtype)
                for local_token in T.serial(txt_token_count):
                    token_work = txt_token_start + local_token
                    source_token = token_work % txt_tokens
                    batch = token_work // txt_tokens
                    load_rotation(
                        rotary_cos,
                        rotary_sin,
                        source_token * head_dim,
                        cos_ub,
                        sin_ub,
                    )
                    T.tile.broadcast(cos_heads_ub, cos_ub)
                    T.tile.broadcast(sin_heads_ub, sin_ub)
                    output_offset = (batch * joint_tokens + source_token) * head_width
                    input_offset = batch * txt_batch_stride + source_token * txt_token_stride
                    process_qk(
                        txt_q,
                        joint_q,
                        input_offset,
                        output_offset,
                        q_group_half_ub,
                        q_group_fp32_ub,
                        cos_heads_ub,
                        sin_heads_ub,
                        rotated_heads_ub,
                        rotate_gather_offsets_ub,
                    )
                    process_qk(
                        txt_k,
                        joint_k,
                        input_offset,
                        output_offset,
                        k_group_half_ub,
                        k_group_fp32_ub,
                        cos_heads_ub,
                        sin_heads_ub,
                        rotated_heads_ub,
                        rotate_gather_offsets_ub,
                    )
                    v_input_offset = batch * txt_v_batch_stride + source_token * txt_v_token_stride
                    T.copy(txt_v[0, v_input_offset], v_io_half_ub)
                    T.copy(v_io_half_ub, joint_v[0, output_offset])

                for local_token in T.serial(img_token_count):
                    token_work = img_token_start + local_token
                    source_token = token_work % img_tokens
                    batch = token_work // img_tokens
                    load_rotation(
                        rotary_cos,
                        rotary_sin,
                        (txt_tokens + source_token) * head_dim,
                        cos_ub,
                        sin_ub,
                    )
                    T.tile.broadcast(cos_heads_ub, cos_ub)
                    T.tile.broadcast(sin_heads_ub, sin_ub)
                    output_offset = (batch * joint_tokens + txt_tokens + source_token) * head_width
                    input_offset = batch * img_batch_stride + source_token * img_token_stride
                    process_qk(
                        img_q,
                        joint_q,
                        input_offset,
                        output_offset,
                        q_group_half_ub,
                        q_group_fp32_ub,
                        cos_heads_ub,
                        sin_heads_ub,
                        rotated_heads_ub,
                        rotate_gather_offsets_ub,
                    )
                    process_qk(
                        img_k,
                        joint_k,
                        input_offset,
                        output_offset,
                        k_group_half_ub,
                        k_group_fp32_ub,
                        cos_heads_ub,
                        sin_heads_ub,
                        rotated_heads_ub,
                        rotate_gather_offsets_ub,
                    )
                    v_input_offset = batch * img_v_batch_stride + source_token * img_v_token_stride
                    T.copy(img_v[0, v_input_offset], v_io_half_ub)
                    T.copy(v_io_half_ub, joint_v[0, output_offset])

    return qwen_image_qkv_epilogue_batch24_kernel


@tilelang.jit(pass_configs=DEFAULT_ASCEND_PASS_CONFIGS)
def qwen_image_qkv_epilogue_batch24_kernel_jit(
    head_dim: int,
    num_heads: int,
    dtype: str,
    vec_core_num: int,
):
    return build_qwen_image_qkv_epilogue_kernel(
        head_dim=head_dim,
        num_heads=num_heads,
        dtype=dtype,
        vec_core_num=vec_core_num,
    )


def _torch_reference(
    img_q,
    img_k,
    img_v,
    txt_q,
    txt_k,
    txt_v,
    img_q_weight,
    img_k_weight,
    txt_q_weight,
    txt_k_weight,
    img_freqs,
    txt_freqs,
    img_eps,
    txt_eps,
):
    import torch
    import torch_npu

    def rmsnorm_rope(tensor, weight, freqs, eps):
        output, _ = torch_npu.npu_rms_norm(tensor, weight, eps)
        cos = freqs.real.repeat_interleave(2, dim=-1)[None, :, None, :]
        sin = freqs.imag.repeat_interleave(2, dim=-1)[None, :, None, :]
        return torch_npu.npu_rotary_mul(output.float(), cos, sin, "interleave").to(tensor.dtype)

    img_q = rmsnorm_rope(img_q, img_q_weight, img_freqs, img_eps)
    img_k = rmsnorm_rope(img_k, img_k_weight, img_freqs, img_eps)
    txt_q = rmsnorm_rope(txt_q, txt_q_weight, txt_freqs, txt_eps)
    txt_k = rmsnorm_rope(txt_k, txt_k_weight, txt_freqs, txt_eps)
    return tuple(torch.cat(tensors, dim=1) for tensors in ((txt_q, img_q), (txt_k, img_k), (txt_v, img_v)))


def _run_ref_check() -> None:
    import torch
    import torch_npu

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        logger.warning("Skip Qwen-Image QKV epilogue reference check: NPU is not available")
        return

    torch.manual_seed(42)
    device = torch.device("npu")
    options = {"device": device, "dtype": torch.bfloat16}
    batch_size = 2
    img_tokens = 17
    txt_tokens = 7
    num_heads = DEFAULT_NUM_HEADS
    head_dim = DEFAULT_HEAD_DIM
    img_qkv = [torch.randn((batch_size, img_tokens, num_heads, head_dim), **options) for _ in range(3)]
    txt_qkv = [torch.randn((batch_size, txt_tokens, num_heads, head_dim), **options) for _ in range(3)]
    weights = [torch.randn((head_dim,), device=device, dtype=torch.float32) for _ in range(4)]
    img_angles = torch.randn((img_tokens, head_dim // 2), device=device, dtype=torch.float32)
    txt_angles = torch.randn((txt_tokens, head_dim // 2), device=device, dtype=torch.float32)
    img_freqs = torch.polar(torch.ones_like(img_angles), img_angles).contiguous()
    txt_freqs = torch.polar(torch.ones_like(txt_angles), txt_angles).contiguous()
    outputs = [torch.empty((batch_size, txt_tokens + img_tokens, num_heads, head_dim), **options) for _ in range(3)]
    buffer_capacity = outputs[0].numel()

    def padded_flat_buffer(tensor):
        flat = tensor.reshape(-1)
        buffer = torch.empty((1, buffer_capacity), device=tensor.device, dtype=tensor.dtype)
        buffer[0, : flat.numel()].copy_(flat)
        return buffer

    tilelang.disable_cache()
    kernel = qwen_image_qkv_epilogue_batch24_kernel_jit(
        head_dim=head_dim,
        num_heads=num_heads,
        dtype=DEFAULT_DTYPE,
        vec_core_num=detect_vec_core_num(),
    )
    normalized_img_qkv = [
        torch_npu.npu_rms_norm(img_qkv[0], weights[0], 1e-6)[0],
        torch_npu.npu_rms_norm(img_qkv[1], weights[1], 1e-6)[0],
        img_qkv[2],
    ]
    normalized_txt_qkv = [
        torch_npu.npu_rms_norm(txt_qkv[0], weights[2], 1e-6)[0],
        torch_npu.npu_rms_norm(txt_qkv[1], weights[3], 1e-6)[0],
        txt_qkv[2],
    ]
    joint_freqs = torch.cat((txt_freqs, img_freqs), 0)
    rotary_cos = torch.real(joint_freqs).repeat_interleave(2, -1)
    rotary_sin = torch.imag(joint_freqs).repeat_interleave(2, -1)
    rotary_sin[:, 0::2] = -rotary_sin[:, 0::2]
    rotate_gather_offsets = torch.empty((ROPE_HEADS_PER_GROUP * head_dim,), dtype=torch.int32)
    for index in range(ROPE_HEADS_PER_GROUP * head_dim):
        head_offset = (index // head_dim) * head_dim
        head_index = index % head_dim
        pair_start = (head_index // 2) * 2
        rotate_gather_offsets[index] = 4 * (head_offset + pair_start + 1 - head_index % 2)
    kernel(
        *(padded_flat_buffer(tensor) for tensor in normalized_img_qkv),
        *(padded_flat_buffer(tensor) for tensor in normalized_txt_qkv),
        padded_flat_buffer(rotary_cos),
        padded_flat_buffer(rotary_sin),
        rotate_gather_offsets.to(device).view(torch.uint32),
        *(tensor.view(1, -1) for tensor in outputs),
        batch_size,
        img_tokens,
        txt_tokens,
        num_heads,
        normalized_img_qkv[0].stride(0),
        normalized_img_qkv[0].stride(1),
        normalized_img_qkv[2].stride(0),
        normalized_img_qkv[2].stride(1),
        normalized_txt_qkv[0].stride(0),
        normalized_txt_qkv[0].stride(1),
        normalized_txt_qkv[2].stride(0),
        normalized_txt_qkv[2].stride(1),
    )
    torch.npu.synchronize()
    references = _torch_reference(
        *img_qkv,
        *txt_qkv,
        *weights,
        img_freqs,
        txt_freqs,
        1e-6,
        1e-6,
    )
    for output, reference in zip(outputs, references):
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
    logger.info("Qwen-Image QKV epilogue output matches torch reference")
