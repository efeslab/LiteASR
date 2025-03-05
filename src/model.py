import base64
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False

from compile import CUDAGraphed
from triton_kernel import low_rank_attn_triton, triton_index_select_single_row, fill_kv_cache_triton

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class LinearLowRank(nn.Module):
    def __init__(self, in_features: int, out_features: int, low_rank_features: int):
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty(low_rank_features, in_features, dtype=torch.float16, device="cuda"))
        self.weight2 = nn.Parameter(torch.empty(out_features, low_rank_features, dtype=torch.float16, device="cuda"))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16, device="cuda"))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            F.linear(x, self.weight1, None), self.weight2, self.bias
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


@contextmanager
def disable_sdpa():
    prev_state = MultiHeadAttention.use_sdpa
    try:
        MultiHeadAttention.use_sdpa = False
        yield
    finally:
        MultiHeadAttention.use_sdpa = prev_state


class MultiHeadAttention(nn.Module):
    use_sdpa = True

    def __init__(
        self, 
        n_state: int, 
        n_head: int, 
        low_rank_config: Dict[str, int] = None,
    ):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_state // n_head
        if low_rank_config is not None:
            self.query = LinearLowRank(n_state, n_state, low_rank_config["q_proj"]) \
                if "q_proj" in low_rank_config else Linear(n_state, n_state)
            self.key = LinearLowRank(n_state, n_state, low_rank_config["k_proj"]) \
                if "k_proj" in low_rank_config else Linear(n_state, n_state, bias=False)
            self.value = LinearLowRank(n_state, n_state, low_rank_config["v_proj"]) \
                if "v_proj" in low_rank_config else Linear(n_state, n_state)
            self.out = LinearLowRank(n_state, n_state, low_rank_config["out_proj"]) \
                if "out_proj" in low_rank_config else Linear(n_state, n_state)
        else:
            self.query = Linear(n_state, n_state)
            self.key = Linear(n_state, n_state, bias=False)
            self.value = Linear(n_state, n_state)
            self.out = Linear(n_state, n_state)

        self.calibration_data = {
            "query": [], 
            "key": [],
            "value": [],
            "out": [],
        }
        self.low_rank_config = low_rank_config
        self.enable_custom_kernel = False
    
    def prepare_custom_kernel(self):
        if self.low_rank_config["q_proj"] <= 32 and self.low_rank_config["k_proj"] <= 32 and self.low_rank_config["v_proj"] <= 32:
            # Experimental: use custom kernel for low-rank self-attention at encoder
            # Currently, rank=48 is not allowed due to power-of-2 issue in Triton kernel, but which makes the usage of kernel
            # only for very limited cases.
            self.enable_custom_kernel = True
            self.W_q2 = self.query.weight2.view(self.n_head, self.head_dim, -1).permute(0, 2, 1)
            self.W_k2 = self.key.weight2.view(self.n_head, self.head_dim, -1)
            self.W_v2 = self.value.weight2.view(self.n_head, self.head_dim, -1).permute(0, 2, 1)
            self.W_m = self.W_q2 @ self.W_k2

            self.b_q = self.query.bias.view(self.n_head, 1, self.head_dim)
            self.b_k = self.key.bias.view(self.n_head, 1, self.head_dim).permute(0, 2, 1)
            self.b_v = self.value.bias
            self.b_1 = self.W_q2 @ self.b_k
            self.b_2 = self.b_q @ self.W_k2
            self.b_3 = self.b_q @ self.b_k

        else:
            self.enable_custom_kernel = False

    
    def _low_rank_attn(self, x: Tensor):
        q1 = F.linear(x, self.query.weight1)
        k1 = F.linear(x, self.key.weight1)
        v1 = F.linear(x, self.value.weight1)

        a = low_rank_attn_triton(q1, k1, v1, self.W_m, self.W_v2, self.b_1, self.b_2, self.b_3)
        wv = a.flatten(start_dim=2) + self.b_v
        return wv 

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        offset: Optional[Tensor] = None,
        is_encoder: bool = False,
        is_prefilling: bool = False, # True only at prefilling stage
        is_calibrating: bool = False, # True only at calibration stage
    ):
        if kv_cache is None:
            # for encoder self-attention 
            if self.enable_custom_kernel:
                wv = self._low_rank_attn(x)
                qk = None
            else:
                q = self.query(x)
                k = self.key(x)
                v = self.value(x)
                wv, qk = self.qkv_attention(q, k, v, mask=None)
        
        elif xa is None:
            # for decoder self-attention
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)
            if is_prefilling:
                kv_cache["self-key"][:, offset : offset + x.shape[1]].copy_(k)
                kv_cache["self-value"][:, offset : offset + x.shape[1]].copy_(v)
                sliced_mask = mask[offset : offset + x.shape[1], :]
            else:
                fill_kv_cache_triton(kv_cache["self-key"], kv_cache["self-value"], k, v, offset)
                sliced_mask = torch.zeros(q.shape[1], mask.shape[1], device=q.device, dtype=q.dtype)
                triton_index_select_single_row(mask, sliced_mask, offset)

            wv, qk = self.qkv_attention(q, kv_cache["self-key"], kv_cache["self-value"], mask=sliced_mask)
        
        else:
            # for decoder cross-attention
            q = self.query(x)
            if is_prefilling:
                k = self.key(xa)
                v = self.value(xa)
                kv_cache["cross-key"].copy_(k)
                kv_cache["cross-value"].copy_(v)

            wv, qk = self.qkv_attention(q, kv_cache["cross-key"], kv_cache["cross-value"], mask=None)

        if is_encoder and is_calibrating:
            self.calibration_data["query"].append(q[0].detach().cpu())
            self.calibration_data["key"].append(k[0].detach().cpu())
            self.calibration_data["value"].append(v[0].detach().cpu())

        out = self.out(wv)
        if is_encoder and is_calibrating:
            self.calibration_data["out"].append(out[0].detach().cpu())
        
        return out, qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            a = scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
            )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            raise NotImplementedError("Use SDPA implementation")
            # qk = (q * scale) @ (k * scale).transpose(-1, -2)
            # if mask is not None:
            #     qk = qk + mask[:n_ctx, :n_ctx]
            # qk = qk.float()

            # w = F.softmax(qk, dim=-1).to(q.dtype)
            # out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            # qk = qk.detach()

        return out, qk


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self, 
        n_state: int, 
        n_head: int, 
        cross_attention: bool = False,
        low_rank_config: Dict[str, int] = None,
    ):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head, low_rank_config=low_rank_config)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        if low_rank_config is not None:
            self.mlp = nn.Sequential(
                LinearLowRank(n_state, n_mlp, low_rank_config["fc1"]) if "fc1" in low_rank_config else Linear(n_state, n_mlp),
                nn.GELU(), 
                LinearLowRank(n_mlp, n_state, low_rank_config["fc2"]) if "fc2" in low_rank_config else Linear(n_mlp, n_state),
            )
        else:
            self.mlp = nn.Sequential(
                Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
            )
        self.mlp_ln = LayerNorm(n_state)

        self.calibration_data = {
            "mlp1": [],
            "mlp2": [],
        }

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        offset: Optional[Tensor] = None,
        is_encoder: bool = False,
        is_prefilling: bool = False, # True only at prefilling stage
        is_calibrating: bool = False, # True only at calibration stage
    ):   
        x = x + self.attn(
            self.attn_ln(x), 
            mask=mask, 
            kv_cache=kv_cache, 
            offset=offset,
            is_encoder=is_encoder, 
            is_prefilling=is_prefilling,
            is_calibrating=is_calibrating,
        )[0]

        if self.cross_attn:
            x = x + self.cross_attn(
                self.cross_attn_ln(x), 
                xa, 
                kv_cache=kv_cache,
                is_prefilling=is_prefilling,
            )[0]
        
        hidden_feature = self.mlp[0](self.mlp_ln(x))
        if is_encoder and is_calibrating:
            self.calibration_data["mlp1"].append(hidden_feature[0].detach().cpu())
        
        hidden_feature = self.mlp[2](self.mlp[1](hidden_feature))
        if is_encoder and is_calibrating:
            self.calibration_data["mlp2"].append(hidden_feature[0].detach().cpu())
        
        x = x + hidden_feature
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, 
        n_mels: int, 
        n_ctx: int, 
        n_state: int, 
        n_head: int, 
        n_layer: int,
        low_rank_config: List[Dict[str, int]],
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList([
            ResidualAttentionBlock(n_state, n_head, low_rank_config=low_rank_config[i]) 
            for i in range(n_layer)
        ])
        self.ln_post = LayerNorm(n_state)

        self.audio_feature = None
        self.is_calibrating = False 
    
    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for i_layer, block in enumerate(self.blocks):
            x = block(x, is_encoder=True, is_calibrating=self.is_calibrating)

        x = self.ln_post(x)

        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, offset: Tensor, kv_cache: Optional[list[dict]] = None, is_prefilling: bool = False):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        if is_prefilling:
            x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        else:
            x = self.token_embedding(x)
            y = torch.zeros_like(x)
            triton_index_select_single_row(self.positional_embedding, y, offset)
            x = x + y
        x = x.to(xa.dtype)

        for i, block in enumerate(self.blocks):
            x = block(
                x, 
                xa, 
                mask=self.mask, 
                kv_cache=kv_cache[i],
                offset=offset,
                is_encoder=False,
                is_prefilling=is_prefilling,
            )

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits


class Whisper(nn.Module):
    """
    Inference Whisper models compressed with LiteASR.
    Supporting CUDA Graph-compatibel for faster inference.
    """
    def __init__(
        self, 
        dims: ModelDimensions, 
        low_rank_config: List[Dict[str, int]],
        bs: int = 1, 
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            low_rank_config=low_rank_config,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )

        self.bs = bs
        self.dtype = dtype
        self.device = device

        # allocate static KV cache
        self.kv_cache = [
            {
                "cross-key": torch.zeros(bs, self.dims.n_audio_ctx, self.dims.n_text_state,).to(device).to(self.dtype), 
                "cross-value": torch.zeros(bs, self.dims.n_audio_ctx, self.dims.n_text_state,).to(device).to(self.dtype),
                "self-key": torch.zeros(bs, self.dims.n_text_ctx, self.dims.n_text_state,).to(device).to(self.dtype),
                "self-value": torch.zeros(bs, self.dims.n_text_ctx, self.dims.n_text_state,).to(device).to(self.dtype),
            } for _ in range(self.dims.n_text_layer)
        ]
        self.offset = torch.zeros(bs, dtype=torch.long).to(device)

    def reinit_kv_cache(self):
        for cache in self.kv_cache:
            cache["cross-key"].zero_() 
            cache["cross-value"].zero_() 
            cache["self-key"].zero_() 
            cache["self-value"].zero_() 
        
        self.offset.zero_()
    
    def init_cuda_graph(self):
        with torch.no_grad():
            self._init_encoder()
            self._init_decoder_generation()
            self.reinit_kv_cache()

            print("Initialization complete.")
    
    def _init_encoder(self):
        self.encoder(torch.randn(1, self.dims.n_mels, 3000).to(self.device).to(self.dtype))
        self.encoder.forward = CUDAGraphed(self.encoder.forward)
        for _ in range(5):
            y = self.encoder.forward(torch.randn(1, self.dims.n_mels, 3000).to(self.device).to(self.dtype))
    
    def _init_decoder_generation(self):
        self.decoder.forward_generation = CUDAGraphed(self.decoder.forward)
        for i in range(5):
            self.decoder.forward_generation(
                torch.randint(0, self.dims.n_vocab, (self.bs, 1)).to(self.device).to(torch.long),
                torch.randn(1, self.dims.n_audio_ctx, self.dims.n_audio_state).to(self.device).to(self.dtype),
                self.offset,
                self.kv_cache,
                False,
            )
    
    def prepare_custom_kernel(self):
        for block in self.encoder.blocks:
            block.attn.prepare_custom_kernel()

    def forward(
        self, mel: torch.Tensor, decoder_input_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        self.reinit_kv_cache()

        # single batch inference with CUDA Graph
        ret_tokens = []
        
        encoded = self.encoder.forward(mel)
        
        logits = self.decoder.forward(decoder_input_ids, encoded, self.offset, self.kv_cache, True)
        self.offset.add_(logits.shape[1])
        next_token = torch.argmax(logits, dim=-1)[:, -1:]
        
        for _ in range(self.dims.n_text_ctx):
            # greedy search until EOS token 
            ret_tokens.append(next_token[0, 0].item())
            if next_token[0, -1] == 50257:
                break
            
            logits = self.decoder.forward_generation(next_token, encoded, self.offset, self.kv_cache, False)
            next_token = torch.argmax(logits, dim=-1)[:, -1:]
            self.offset.add_(1)
        
        return ret_tokens
