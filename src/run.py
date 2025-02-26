# Running inference for the compressed Whisper models.
# This code is for demonstration purposes only, and longer audio, custom sampling algorithms, 
# or batch inference would require modification of the code.
import argparse
from collections import OrderedDict
import re

import torch 
from transformers import AutoModel, WhisperFeatureExtractor, WhisperTokenizerFast

from model import Whisper, ModelDimensions
from utils import ffmpeg_read


def convert_from_hf_whisper(hf_model, device=torch.device("cuda"), use_custom_kernel=False):
    print("Converting from HuggingFace model to Whisper model...")
    reverse_translation = OrderedDict({
        r"^encoder\.layers\.(\d+)\.self_attn.k_proj\.(\w+)$": r"encoder.blocks.\1.attn.key.\2",
        r"^encoder\.layers\.(\d+)\.self_attn.out_proj\.(\w+)$": r"encoder.blocks.\1.attn.out.\2",
        r"^encoder\.layers\.(\d+)\.self_attn.q_proj\.(\w+)$": r"encoder.blocks.\1.attn.query.\2",
        r"^encoder\.layers\.(\d+)\.self_attn.v_proj\.(\w+)$": r"encoder.blocks.\1.attn.value.\2",
        r"^encoder\.layers\.(\d+)\.self_attn_layer_norm\.(\w+)$": r"encoder.blocks.\1.attn_ln.\2",
        r"^encoder\.layers\.(\d+)\.fc1\.(\w+)$": r"encoder.blocks.\1.mlp.0.\2",
        r"^encoder\.layers\.(\d+)\.fc2\.(\w+)$": r"encoder.blocks.\1.mlp.2.\2",
        r"^encoder\.layers\.(\d+)\.final_layer_norm\.(\w+)$": r"encoder.blocks.\1.mlp_ln.\2",
        r"^encoder\.embed_positions\.weight$": r"encoder.positional_embedding",
        r"^encoder\.layer_norm\.(\w+)$": r"encoder.ln_post.\1",
        r"^encoder\.(\w+)\.(\w+)": r"encoder.\1.\2",
    \
        r"^decoder\.embed_positions\.weight$": r"decoder.positional_embedding",
        r"^decoder\.embed_tokens\.weight$": r"decoder.token_embedding.weight",
        r"^decoder\.layer_norm\.(\w+)$": r"decoder.ln.\1",
    \
        r"^decoder\.layers\.(\d+)\.encoder_attn\.k_proj.(\w+)$": r"decoder.blocks.\1.cross_attn.key.\2",
        r"^decoder\.layers\.(\d+)\.encoder_attn\.out_proj.(\w+)$": r"decoder.blocks.\1.cross_attn.out.\2",
        r"^decoder\.layers\.(\d+)\.encoder_attn\.q_proj.(\w+)$": r"decoder.blocks.\1.cross_attn.query.\2",
        r"^decoder\.layers\.(\d+)\.encoder_attn\.v_proj.(\w+)$": r"decoder.blocks.\1.cross_attn.value.\2",
        r"^decoder\.layers\.(\d+)\.encoder_attn_layer_norm\.(\w+)$": r"decoder.blocks.\1.cross_attn_ln.\2",
    \
        r"^decoder\.layers\.(\d+)\.self_attn\.k_proj\.(\w+)$": r"decoder.blocks.\1.attn.key.\2",
        r"^decoder\.layers\.(\d+)\.self_attn\.out_proj\.(\w+)$": r"decoder.blocks.\1.attn.out.\2",
        r"^decoder\.layers\.(\d+)\.self_attn\.q_proj\.(\w+)$": r"decoder.blocks.\1.attn.query.\2",
        r"^decoder\.layers\.(\d+)\.self_attn\.v_proj\.(\w+)$": r"decoder.blocks.\1.attn.value.\2",
        r"^decoder\.layers\.(\d+)\.self_attn_layer_norm\.(\w+)$": r"decoder.blocks.\1.attn_ln.\2",
        r"^decoder\.layers\.(\d+)\.fc1\.(\w+)$": r"decoder.blocks.\1.mlp.0.\2",
        r"^decoder\.layers\.(\d+)\.fc2\.(\w+)$": r"decoder.blocks.\1.mlp.2.\2",
        r"^decoder\.layers\.(\d+)\.final_layer_norm\.(\w+)$": r"decoder.blocks.\1.mlp_ln.\2",
    })

    model_dims = ModelDimensions(
        n_mels=hf_model.config.num_mel_bins, # 128
        n_audio_ctx=hf_model.config.max_source_positions, # 1500
        n_audio_state=hf_model.config.d_model, # 1280
        n_audio_head=hf_model.config.encoder_attention_heads, # 20
        n_audio_layer=hf_model.config.encoder_layers, # 32
        n_vocab=hf_model.config.vocab_size, # 51866
        n_text_ctx=hf_model.config.max_target_positions, # 448
        n_text_state=hf_model.config.d_model, # 1280
        n_text_head=hf_model.config.decoder_attention_heads, # 20
        n_text_layer=hf_model.config.decoder_layers, # 32 or 4
    )

    low_rank_config = hf_model.config.low_rank_config

    new_state_dict = {}
    new_model = Whisper(
        model_dims, 
        low_rank_config=low_rank_config,
        bs=1, 
        device=device,
    )# .to(torch.float16)

    for key, value in hf_model.state_dict().items():
        for pattern, replacement in reverse_translation.items():
            if re.match(pattern, key):
                new_key = re.sub(pattern, replacement, key)
                # transpose the value if the name ends with *.weight1 and *.weight2
                if key.endswith("weight1") or key.endswith("weight2"):
                    value = value.T.contiguous()
                new_state_dict[new_key] = value
                break
    
    torch.cuda.synchronize()
    new_model.load_state_dict(new_state_dict, strict=True)
    new_model.decoder.lm_head = new_model.decoder.token_embedding.weight
    if use_custom_kernel:
        new_model.prepare_custom_kernel()

    del hf_model
    print("Conversion complete.")
    return new_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--model", type=str, default="efficient-speech/lite-whisper-large-v3-turbo")
    parser.add_argument("--base-model", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--audio-path", type=str, default="audio.wav")
    parser.add_argument("--use-custom-kernel", action="store_true")
    args = parser.parse_args()

    device = args.device
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype)
    model = model.model

    # Convert the model
    model = convert_from_hf_whisper(model, device=device, use_custom_kernel=args.use_custom_kernel)
    model = model.to(device).to(dtype)
    model.is_calibrating = False

    # capturing CUDA Graph 
    model.init_cuda_graph()

    feature_extractor: WhisperFeatureExtractor = WhisperFeatureExtractor.from_pretrained(args.base_model)
    tokenizer: WhisperTokenizerFast = WhisperTokenizerFast.from_pretrained(args.base_model)

    prefix_special_tokens = {}
    langs = ["en"]
    for lang in langs:
        tokenizer.set_prefix_tokens(language=lang, task="transcribe", predict_timestamps=False)
        prefix_special_tokens[lang] = tokenizer.prefix_tokens

    audio = ffmpeg_read(args.audio_path, sampling_rate=16_000)

    with torch.no_grad():
        input_features = feature_extractor(audio, sampling_rate=16_000, return_tensors="pt")["input_features"].to(device)
        tokens = torch.tensor(prefix_special_tokens["en"]).unsqueeze(0).to(device)

        ret_tokens = model.forward(input_features.to(torch.float16), tokens)
        
        print(tokenizer.decode(ret_tokens))
        model.reinit_kv_cache()
