# Copyright © 2023 Apple Inc.

import json
from pathlib import Path

from huggingface_hub import snapshot_download
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
import torch 

from . import whisper
from .convert import hf_to_pt

def load_model(
    path_or_hf_repo: str,
    dtype: mx.Dtype = mx.float32,
) -> whisper.Whisper:
    def remap(key, value):
        key = key.replace("mlp.0", "mlp1")
        key = key.replace("mlp.2", "mlp2")
        if "conv" in key and value.ndim == 3:
            value = value.swapaxes(1, 2)
        if isinstance(value, torch.Tensor):
            value = mx.array(value.detach())
        return key, value.astype(dtype)
    
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

    with open(str(model_path / "config.json"), "r") as f:
        config = json.loads(f.read())
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)

    low_rank_config = config.pop("low_rank_config", None)
    # model_args = whisper.ModelDimensions(**config)

    wf = model_path / "model.safetensors"
    if wf.exists():
        weights = mx.load(str(wf))
    else:
        split_pattern = list(model_path.glob("model-*-of-*.safetensors"))
        weights = {}
        for wf in split_pattern:
            weights.update(mx.load(str(wf)))

    weights, config = hf_to_pt(weights, config)
    weights.pop("encoder.positional_embedding", None)
    weights = dict(remap(k, v) for k, v in weights.items())

    model_args = whisper.ModelDimensions(**config)

    model = whisper.Whisper(model_args, dtype, low_rank_config=low_rank_config)

    if quantization is not None:
        class_predicate = (
            lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
            and f"{p}.scales" in weights
        )
        nn.quantize(model, **quantization, class_predicate=class_predicate)

    # weights = tree_unflatten(list(weights.items()))
    # model.update(weights)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model
