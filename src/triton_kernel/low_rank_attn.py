# Customized FlashAttention that work on low-rank features
# based on https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/low_rank_attn_triton.py
import math
import os 
import sys 

import torch
from torch import Tensor 
from torch.nn.functional import scaled_dot_product_attention
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=w, num_stages=s)
        for m in [64, 128] 
        for n in [64, 128]
        for w in [4, 8] 
        for s in [1, 2]
    ],
    key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'IS_CAUSAL', 'BLOCK_HEADDIM']
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q1,
    K1,
    V1,
    W_M,
    W_V2,
    B1,
    B2,
    B3,
    Out,
    Lse,
    softmax_scale,
    stride_qb,
    stride_qm,
    stride_kb,
    stride_kn,
    stride_vb,
    stride_vk,
    stride_Wmh,
    stride_Wmd,
    stride_Wvh,
    stride_Wvd,
    stride_b1h,
    stride_b1q,
    stride_b2h,
    stride_b2k,
    stride_b3h,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RANK_Q: tl.constexpr,
    RANK_K: tl.constexpr,
    RANK_V: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    # off_hb = off_b * nheads + off_h
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_rank_q = tl.arange(0, RANK_Q)
    offs_rank_k = tl.arange(0, RANK_K)
    offs_rank_v = tl.arange(0, RANK_V)
    offs_one = tl.arange(0, 1)
    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)
    q1_ptrs = (
        Q1 + off_b * stride_qb + (offs_m[:, None] * stride_qm + offs_rank_q[None, :])
    )
    k1_ptrs = (
        K1 + off_b * stride_kb + (offs_n[:, None] * stride_kn + offs_rank_k[None, :])
    )
    v1_ptrs = (
        V1 + off_b * stride_vb + (offs_n[:, None] * stride_vk + offs_rank_v[None, :])
    )
    W_M_ptrs = (
        W_M + off_h * stride_Wmh + (offs_rank_q[:, None] * stride_Wmd + offs_rank_k[None, :])
    )
    W_V2_ptrs = (
        W_V2 + off_h * stride_Wvh + (offs_rank_v[:, None] * stride_Wvd + offs_d[None, :])
    )
    b1_ptrs = (
        B1 + off_h * stride_b1h + offs_rank_q
    )
    b2_ptrs = (
        B2 + off_h * stride_b2h + offs_rank_k
    )
    b3_ptrs = B3 + off_h * stride_b3h
    # initialize pointer to m and l
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q1 = tl.load(q1_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    w_m = tl.load(W_M_ptrs)
    w_v2 = tl.load(W_V2_ptrs)
    q_wm = tl.dot(q1, w_m)
    q_wm = q_wm.to(q1.dtype)
    b1 = tl.load(b1_ptrs)
    b2 = tl.load(b2_ptrs)
    b3 = tl.load(b3_ptrs)
    bias = tl.sum(q1 * b1, 1) + b3
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k1 = tl.load(
            k1_ptrs + start_n * stride_kn,
            mask=(start_n + offs_n)[:, None] < seqlen_k,
            other=0.0,
        )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        k_t = tl.trans(k1)
        qk += tl.dot(q_wm, k_t)
        qk += tl.sum(k1 * b2, 1)
        qk += bias[:, None]
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))

        qk = qk * softmax_scale
        m_ij = tl.maximum(tl.max(qk, 1), lse_i)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator --
        acc_o = acc_o * acc_o_scale[:, None]
        # update acc_o
        v1 = tl.load(
            v1_ptrs + start_n * stride_vk,
            mask=(start_n + offs_n)[:, None] < seqlen_k,
            other=0.0,
        )
        p = p.to(v1.dtype)
        tmp = tl.dot(p, v1)
        tmp = tmp.to(w_v2.dtype)
        acc_o += tl.dot(tmp, w_v2)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def _low_rank_attn_forward(q1, k1, v1, W_m, W_v2, b1, b2, b3, causal=False, softmax_scale=None):
    # shape constraints
    # batch, seqlen_q, nheads, d = q.shape
    # _, seqlen_k, _, _ = k.shape
    batch, seqlen_q, rank_q = q1.shape
    _, seqlen_k, rank_k = k1.shape
    _, _, rank_v = v1.shape
    nheads, _, d = W_v2.shape
    rank_v_pow2 = 1 << (rank_v - 1).bit_length()
    # assert k.shape == (batch, seqlen_k, nheads, d)
    # assert v.shape == (batch, seqlen_k, nheads, d)
    assert k1.shape == (batch, seqlen_k, rank_k)
    assert v1.shape == (batch, seqlen_k, rank_v)
    assert W_m.shape == (nheads, rank_q, rank_k)
    assert W_v2.shape == (nheads, rank_v, d)
    assert b1.shape == (nheads, rank_q, 1)
    assert b2.shape == (nheads, 1, rank_k)
    assert b3.shape == (nheads, 1, 1)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q1.dtype == k1.dtype == v1.dtype, "All tensors must have the same type"
    assert q1.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q1.is_cuda and k1.is_cuda and v1.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q1.device, dtype=torch.float32)
    o = torch.empty([batch, seqlen_q, nheads, d], device=q1.device, dtype=q1.dtype)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 64
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_kernel[grid](
        q1,
        k1,
        v1,
        W_m,
        W_v2,
        b1,
        b2,
        b3,
        o,
        lse,
        softmax_scale,
        q1.stride(0), # batch
        q1.stride(1), # seqlen
        k1.stride(0), # batch
        k1.stride(1), # seqlen
        v1.stride(0), # batch
        v1.stride(1), # seqlen
        W_m.stride(0), # head
        W_m.stride(1), # dim (q)
        W_v2.stride(0), # head
        W_v2.stride(1), # dim
        b1.stride(0), # head
        b1.stride(1), # dim
        b2.stride(0), # head
        b2.stride(1), # dim
        b3.stride(0), # head
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        causal,
        BLOCK_HEADDIM,
        RANK_Q=rank_q,
        RANK_K=rank_k,
        RANK_V=rank_v,
        # BLOCK_M=BLOCK,
        # BLOCK_N=BLOCK,
        # num_warps=num_warps,
        # num_stages=2,
    )
    return o, lse, softmax_scale  # softmax_scale could have been updated


def low_rank_attn_triton(q1, k1, v1, W_m, W_v2, b_1, b_2, b_3, causal=False, softmax_scale=None):
    """
    q1: (batch_size, seqlen_q, rank_q)
    k1: (batch_size, seqlen_k, rank_k)
    v1: (batch_size, seqlen_k, rank_v)
    W_m: (nheads, rank_q, rank_k)
    W_v2: (nheads, headdim, rank_v)
    bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
        For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
        ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
    """
    # Make sure that the last dimension is contiguous
    assert q1.stride(-1) == 1
    assert k1.stride(-1) == 1
    assert v1.stride(-1) == 1
    assert b_1.stride(-1) == 1
    assert b_2.stride(-1) == 1
    assert b_3.stride(-1) == 1
    o, lse, softmax_scale = _low_rank_attn_forward(
        q1, k1, v1, W_m, W_v2, b_1, b_2, b_3, causal=causal, softmax_scale=softmax_scale
    )
    return o


def ref_attn(
    x, W_q1, W_q2, W_k1, W_k2, W_v1, W_v2, b_q, b_k, b_v
):
    q = torch.nn.functional.linear(
        torch.nn.functional.linear(x, W_q1),
        W_q2, 
        bias=b_q,
    )
    k = torch.nn.functional.linear(
        torch.nn.functional.linear(x, W_k1),
        W_k2,
        bias=b_k,
    )
    v = torch.nn.functional.linear(
        torch.nn.functional.linear(x, W_v1),
        W_v2,
        bias=b_v,
    )
    q = q.view(*q.shape[:2], 20, -1).permute(0, 2, 1, 3)
    k = k.view(*k.shape[:2], 20, -1).permute(0, 2, 1, 3)
    v = v.view(*v.shape[:2], 20, -1).permute(0, 2, 1, 3)
    a = scaled_dot_product_attention(q, k, v, is_causal=False)
    wv = a.permute(0, 2, 1, 3).flatten(start_dim=2)
    return wv 

def optimized(
    x, W_q1, W_k1, W_m, W_v1, W_v2, b_1, b_2, b_3, b_v
):
    q1 = torch.nn.functional.linear(x, W_q1)
    k1 = torch.nn.functional.linear(x, W_k1)
    v1 = torch.nn.functional.linear(x, W_v1)

    a = low_rank_attn_triton(q1, k1, v1, W_m, W_v2, b_1, b_2, b_3)
    wv = a.flatten(start_dim=2) + b_v
    return wv 

if __name__ == "__main__":
    B, H, M, D = 1, 20, 1500, 64
    rank = 32 
    # scale_val = (1280 // 20) ** (-0.25)

    x = torch.randn(B, M, H * D, dtype=torch.float16, device='cuda')

    W_q1 = torch.randn(rank, H * D, dtype=torch.float16, device='cuda')
    W_q2 = torch.randn(H * D, rank, dtype=torch.float16, device='cuda')
    W_k1 = torch.randn(rank, H * D, dtype=torch.float16, device='cuda')
    W_k2 = torch.randn(H * D, rank, dtype=torch.float16, device='cuda')
    W_v1 = torch.randn(rank, H * D, dtype=torch.float16, device='cuda')
    W_v2 = torch.randn(H * D, rank, dtype=torch.float16, device='cuda')
    
    b_q  = torch.randn(H * D, dtype=torch.float16, device='cuda')
    b_k  = torch.randn(H * D, dtype=torch.float16, device='cuda')
    b_v  = torch.randn(H * D, dtype=torch.float16, device='cuda')

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from compile import CUDAGraphed
    import time 

    ref_attn = CUDAGraphed(ref_attn)
    optimized = CUDAGraphed(optimized)

    ref = ref_attn(x, W_q1, W_q2, W_k1, W_k2, W_v1, W_v2, b_q, b_k, b_v)
    print(ref.shape, ref)

    for _ in range(20):
        start = time.time()
        ref_attn(x, W_q1, W_q2, W_k1, W_k2, W_v1, W_v2, b_q, b_k, b_v)
        torch.cuda.synchronize()
        print(f"Ref: {(time.time() - start) * 1000:.3f}ms")

    W_q2 = W_q2.view(20, 64, -1).permute(0, 2, 1)
    W_k2 = W_k2.view(20, 64, -1)
    W_m = W_q2 @ W_k2
    W_m = W_m.contiguous()
    assert W_m.shape == (20, rank, rank)

    b_q = b_q.view(20, 1, 64)
    b_k = b_k.view(20, 1, 64).permute(0, 2, 1)
    b_1 = W_q2 @ b_k
    b_2 = b_q @ W_k2
    b_3 = b_q @ b_k
    assert b_1.shape == (20, rank, 1)
    assert b_2.shape == (20, 1, rank)
    assert b_3.shape == (20, 1, 1)
    
    W_v2 = W_v2.view(20, 64, -1).permute(0, 2, 1).contiguous()
    
    out = optimized(x, W_q1, W_k1, W_m, W_v1, W_v2, b_1, b_2, b_3, b_v)
    print(out.shape, out)

    for _ in range(20):
        start = time.time()
        optimized(x, W_q1, W_k1, W_m, W_v1, W_v2, b_1, b_2, b_3, b_v)
        torch.cuda.synchronize()
        print(f"Opt: {(time.time() - start) * 1000:.3f}ms")
