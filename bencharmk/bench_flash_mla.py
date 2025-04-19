# MLA Triton kernel is from: https://github.com/monellz/vllm/commit/feebaa7c063be6bfb590a876741aeef1c5f58cf8#diff-7b2e1c9032522f7266051b9887246a65753871dfb3625a258fee40109fe6e87a
import argparse
import math
import random

import flashinfer
import torch
import triton
import triton.language as tl

# pip install flashinfer-python
from flash_mla import flash_mla_with_kvcache, get_mla_metadata


def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse


@torch.inference_mode()
def run_torch_mla(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")
    blocked_v = blocked_k[..., :dv]

    def ref_mla():
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                h_q, h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out, lse

    out_torch, lse_torch = ref_mla()
    t = triton.testing.do_bench(ref_mla)
    return out_torch, lse_torch, t

@torch.inference_mode()
def run_flash_mla(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")
    blocked_v = blocked_k[..., :dv]

    tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

    def flash_mla():
        return flash_mla_with_kvcache(
            q, blocked_k, block_table, cache_seqlens, dv,
            tile_scheduler_metadata, num_splits, causal=causal,
        )

    out_flash, lse_flash = flash_mla()
    t = triton.testing.do_bench(flash_mla)
    return out_flash, lse_flash, t


@torch.inference_mode()
def run_flash_infer(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")

    assert d > dv, "mla with rope dim should be larger than no rope dim"
    q_nope, q_pe = q[..., :dv].contiguous(), q[..., dv:].contiguous()
    blocked_k_nope, blocked_k_pe = blocked_k[..., :dv].contiguous(), blocked_k[..., dv:].contiguous()
    
    
    kv_indptr = [0]
    kv_indices = []
    for i in range(b):
        seq_len = cache_seqlens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_table[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
    for seq_len in cache_seqlens[1:]:
        kv_indptr.append((seq_len + block_size - 1) // block_size + kv_indptr[-1])
        
    q_indptr = torch.arange(0, b + 1).int() * s_q
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)

    mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.int8),
        backend="fa3"
    )
