import os
import torch
from torch import Tensor

import triton
import triton.language as tl


KEYS = ['N_CTX', 'HEAD_DIM']

DTYPES = {
    'tf32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


def get_fast_config():
    configs = []
    for BM in [32, 64]:
        for BN in [32, 64]:
            if BM <= BN:
                continue
            for w in [2, 4]:
                for s in [2, 4]:
                    configs.append(
                        triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_warps=w, num_stages=s)
                    )
    return configs


def get_default_config():
    return [triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=1, num_stages=1)]


AUTOTUNE_SETTING = os.environ.get("FLASH_AUTOTUNE", "default")
USE_CUDA_GRAPH = os.environ.get("FLASH_CUDA_GRAPH", "0") == "1"
get_autotune_config = get_fast_config if AUTOTUNE_SETTING == "fast" else get_default_config


@triton.autotune(configs=get_autotune_config(), key=KEYS, use_cuda_graph=USE_CUDA_GRAPH)
@triton.jit
def qkv_flash_kernel(
    Q, K, V, Out,
    qk_combined_scale,
    pv_dequant_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    EVEN_CTX: tl.constexpr = (N_CTX % BLOCK_M == 0) and (N_CTX % BLOCK_N == 0)

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn), offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset, shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn), offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    NEG_INF = float('-inf')
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + NEG_INF
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if EVEN_CTX:
        q = tl.load(Q_block_ptr)
    else:
        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if EVEN_CTX:
            k = tl.load(K_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        # INT8 QK MMA -> int32, fused dequant + softmax base-2 scaling.
        qk_i32 = tl.dot(q, k, out_dtype=tl.int32)
        qk = qk_i32.to(tl.float32) * qk_combined_scale
        if not EVEN_CTX:
            padding = (offs_m < N_CTX)[:, None] & ((start_n + offs_n) < N_CTX)[None, :]
            qk = tl.where(padding, qk, NEG_INF)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(qk - m_ij[:, None])  # in [0, 1]
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # Re-quantize P to int8: p in [0, 1] -> round(p * 127) in [0, 127].
        # Trick `+ 0.5 then floor-cast` works since p is non-negative.
        p_i8 = (p * 127.0 + 0.5).to(tl.int8)

        if EVEN_CTX:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        # INT8 PV MMA -> int32, dequant and add into fp32 accumulator.
        pv_i32 = tl.dot(p_i8, v, out_dtype=tl.int32)
        acc += pv_i32.to(tl.float32) * pv_dequant_scale

        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    acc = acc / l_i[:, None]
    if EVEN_CTX:
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))
    else:
        tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0,))


def qkv_flash_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dtype: str = 'fp16',
    scale: float | None = None,
    return_mode: str = 'output',
):
    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {list(DTYPES)}; got {dtype!r}")
    out_dtype = DTYPES[dtype]

    if scale is None:
        scale = 1.0 / (HEAD_DIM_Q ** 0.5)
    LOG2E = 1.44269504

    # Per-tensor symmetric int8 quantization of Q, K, V.
    q_scale = (query.abs().amax() / torch.iinfo(torch.int8).max).item()
    k_scale = (key.abs().amax() / torch.iinfo(torch.int8).max).item()
    v_scale = (value.abs().amax() / torch.iinfo(torch.int8).max).item()
    q_i8 = torch.clamp(torch.round(query / q_scale), -128, 127).to(torch.int8)
    k_i8 = torch.clamp(torch.round(key / k_scale), -128, 127).to(torch.int8)
    v_i8 = torch.clamp(torch.round(value / v_scale), -128, 127).to(torch.int8)

    Z, H, N, _ = q_i8.shape
    o = torch.empty(Z, H, N, HEAD_DIM_V, device=query.device, dtype=out_dtype)

    qk_combined_scale = q_scale * k_scale * scale * LOG2E
    pv_dequant_scale = v_scale / 127.0

    grid = lambda args: (triton.cdiv(N, args["BLOCK_M"]), Z * H, 1)

    def kernel():
        qkv_flash_kernel[grid](
            q_i8, k_i8, v_i8, o,
            qk_combined_scale, pv_dequant_scale,
            *q_i8.stride(), *k_i8.stride(), *v_i8.stride(), *o.stride(),
            Z, H, N_CTX=N, HEAD_DIM=HEAD_DIM_K,
        )

    if return_mode == 'latency':
        return triton.testing.do_bench(kernel, warmup=10, rep=100, return_mode='median')
    kernel()
    return o


class qkv_flash:
    kernel = [qkv_flash_kernel]
    forward = qkv_flash_forward
