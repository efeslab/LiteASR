# Running inference for the compressed Whisper models using MLX.
# This code is for demonstration purposes only.

import argparse 

import mlx_whisper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="efficient-speech/lite-whisper-large-v3-turbo")
    parser.add_argument("--audio-path", type=str, default="audio.wav")
    parser.add_argument("--language", type=str, default="en")
    args = parser.parse_args()

    result = mlx_whisper.transcribe(
        args.audio_path, 
        path_or_hf_repo=args.model,
        language=args.language,
        temperature=0.0,
        fp16=True,
    )

    print(result)
