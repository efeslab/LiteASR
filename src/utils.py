import torch
import subprocess
import numpy as np
from transformers import AutoConfig

from lite_whisper.configuration_lite_whisper import LiteWhisperConfig
from lite_whisper.modeling_lite_whisper import LiteWhisperForConditionalGeneration


def ffmpeg_read(filename: str, sampling_rate: int, start: float = 0.0) -> np.array:
    """
    Helper function to read an audio file through ffmpeg.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    start = str(start)
    format_for_conversion = "f32le"
    ffmpeg_command = [
        "ffmpeg",
        "-ss",
        start,
        "-i",
        filename,
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    try:
        with subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE) as ffmpeg_process:
            output_stream = ffmpeg_process.communicate()
    except FileNotFoundError as error:
        raise ValueError("ffmpeg was not found but is required to load audio files from filename") from error
    out_bytes = output_stream[0]
    audio = np.frombuffer(out_bytes, np.float32)
    if audio.shape[0] == 0:
        raise ValueError(
            "Soundfile is either not in the correct format or is malformed. Ensure that the soundfile has "
            "a valid audio file extension (e.g. wav, flac or mp3) and is not corrupted. If reading from a remote "
            "URL, ensure that the URL is the full address to **download** the audio file."
        )
    return audio


def upload_to_hf(model_path, repo_name, org_name, base_model="openai/whisper-large-v3"):
    a = torch.load(model_path)

    def remap(k):
        k = k.replace("positional_embedding", "embed_positions.weight")
        k = k.replace("decoder.token_embedding", "decoder.embed_tokens")
        k = k.replace("encoder.ln_post", "encoder.layer_norm")
        k = k.replace("decoder.ln", "decoder.layer_norm")
        k = k.replace(".mlp.2", ".fc2")
        k = k.replace(".mlp.0", ".fc1")
        k = k.replace(".out", ".out_proj")
        k = k.replace(".value", ".v_proj")
        k = k.replace(".key", ".k_proj")
        k = k.replace(".query", ".q_proj")
        k = k.replace(".mlp_ln", ".final_layer_norm")
        k = k.replace(".cross_attn_ln", ".encoder_attn_layer_norm")
        k = k.replace(".cross_attn.", ".encoder_attn.")
        k = k.replace(".attn_ln", ".attn_layer_norm")
        k = k.replace(".attn", ".self_attn")
        k = k.replace(".blocks", ".layers")
        k = "model." + k
        return k

    b = {remap(k): v for k, v in a.items()}
    b["proj_out.weight"] = b["model.decoder.embed_tokens.weight"]

    LiteWhisperConfig.register_for_auto_class()
    LiteWhisperForConditionalGeneration.register_for_auto_class("AutoModel")

    linear_layers = [
        "self_attn.q_proj", 
        "self_attn.k_proj", 
        "self_attn.v_proj", 
        "self_attn.out_proj", 
        "fc1", 
        "fc2",
    ]

    orig_whisper_config = AutoConfig.from_pretrained(base_model)
    lite_whisper_config = LiteWhisperConfig(
        low_rank_config=[
            {
                layer.split(".")[-1]: b[f"model.encoder.layers.{i}.{layer}.weight1"].shape[1]
                for layer in linear_layers
                if f"model.encoder.layers.{i}.{layer}.weight1" in b
            } for i in range(orig_whisper_config.encoder_layers)
        ],
        **orig_whisper_config.to_dict(),
    )
    model = LiteWhisperForConditionalGeneration(lite_whisper_config)
    model.load_state_dict(b)
    model.config._name_or_path = f"{org_name}/{repo_name}"
    model.push_to_hub(f"{org_name}/{repo_name}")

