import torch
import torch.utils.checkpoint
from torch import nn
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.modeling_whisper import (
    WhisperEncoderLayer,
    WhisperEncoder,
    WhisperModel,
    WhisperForConditionalGeneration,
)

from .configuration_lite_whisper import LiteWhisperConfig


class LinearLowRank(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        low_rank_features: int,
    ):
        super().__init__()

        self.weight1 = nn.Parameter(torch.randn(in_features, low_rank_features))
        self.weight2 = nn.Parameter(torch.randn(low_rank_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.weight1) @ self.weight2 + self.bias


class LiteWhisperEncoderLayer(WhisperEncoderLayer):
    def __init__(self, config: WhisperConfig, low_rank_config: dict[str, int]):
        super().__init__(config)

        if "k_proj" in low_rank_config:
            self.self_attn.k_proj = LinearLowRank(self.embed_dim, self.embed_dim, low_rank_config["k_proj"])
        
        if "v_proj" in low_rank_config:
            self.self_attn.v_proj = LinearLowRank(self.embed_dim, self.embed_dim, low_rank_config["v_proj"])
        
        if "q_proj" in low_rank_config:
            self.self_attn.q_proj = LinearLowRank(self.embed_dim, self.embed_dim, low_rank_config["q_proj"])
        
        if "out_proj" in low_rank_config:
            self.self_attn.out_proj = LinearLowRank(self.embed_dim, self.embed_dim, low_rank_config["out_proj"])

        if "fc1" in low_rank_config:
            self.fc1 = LinearLowRank(self.embed_dim, config.encoder_ffn_dim, low_rank_config["fc1"])
        
        if "fc2" in low_rank_config:
            self.fc2 = LinearLowRank(config.encoder_ffn_dim, self.embed_dim, low_rank_config["fc2"])


class LiteWhisperEncoder(WhisperEncoder):
    def __init__(self, config: WhisperConfig, low_rank_config: list[dict[str, int]]):
        super().__init__(config)

        self.layers = nn.ModuleList([
            LiteWhisperEncoderLayer(config, low_rank_config[i]) 
            for i in range(config.encoder_layers)
        ])


class LiteWhisperModel(WhisperModel):
    def __init__(self, config: WhisperConfig, low_rank_config: list[dict[str, int]]):
        super().__init__(config)

        self.encoder = LiteWhisperEncoder(config, low_rank_config)


class LiteWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    config_class = LiteWhisperConfig

    def __init__(self, config: LiteWhisperConfig):
        low_rank_config = getattr(config, "low_rank_config", None)

        super().__init__(config)
        self.model = LiteWhisperModel(config, low_rank_config)
