import argparse
import numpy as np
import torch
import evaluate
from normalizer import data_utils
import tqdm
import whisper
from datasets import load_dataset, concatenate_datasets
import numpy as np

from run import convert_from_hf_whisper

# random seed for np 
np.random.seed(42)

def compute_snr(clean, noisy):
    """Compute SNR in dB between clean and noisy signals (1D numpy arrays)."""
    clean = np.asarray(clean)
    noisy = np.asarray(noisy)
    noise = noisy - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def add_gaussian_noise(audio, noise_std):
    """Add Gaussian white noise to audio with specified std deviation."""
    noise = np.random.normal(0, noise_std, size=audio.shape)
    return audio + noise

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision('high')


class LinearLowRank(torch.nn.Module):
    def __init__(self, weight1: torch.Tensor, weight2: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.weight1 = torch.nn.Parameter(weight1.half().cuda())
        self.weight2 = torch.nn.Parameter(weight2.half().cuda())
        self.bias = torch.nn.Parameter(bias.half().cuda())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.weight1) @ self.weight2 + self.bias


def transcribe(
        model,
        path, 
        temperature=0.0,
    ):
    result = model.transcribe(
        path,
        task="transcribe",
        language="en",
        temperature=temperature,
    )
    return result["text"]

def apply_low_rank(model, dataset, rank_threshold):
    model.encoder.is_calibrating = True
    base_dim = model.dims.n_audio_state # 1280 for large-v3
    
    for i in tqdm.tqdm(range(len(dataset))):
        audio = dataset[i]["audio"]["array"].astype(np.float32)
        transcribe(model, audio)
    
    for i_layer in tqdm.tqdm(range(len(model.encoder.blocks))):
        for i, layer in enumerate([
            model.encoder.blocks[i_layer].attn.query,
            model.encoder.blocks[i_layer].attn.key,
            model.encoder.blocks[i_layer].attn.value,
            model.encoder.blocks[i_layer].attn.out,
            model.encoder.blocks[i_layer].mlp[0],
            model.encoder.blocks[i_layer].mlp[2],
        ]):
            concat_dim = 0
            if i == 0:
                features = torch.concat(
                    model.encoder.blocks[i_layer].attn.calibration_data['query'], 
                    dim=concat_dim,
                )
            elif i == 1:
                features = torch.concat(
                    model.encoder.blocks[i_layer].attn.calibration_data['key'], 
                    dim=concat_dim,
                )
            elif i == 2:
                features = torch.concat(
                    model.encoder.blocks[i_layer].attn.calibration_data['value'], 
                    dim=concat_dim,
                )
            elif i == 3:
                features = torch.concat(
                    model.encoder.blocks[i_layer].attn.calibration_data['out'], 
                    dim=concat_dim,
                )
            elif i == 4:
                features = torch.concat(
                    model.encoder.blocks[i_layer].calibration_data['mlp1'], 
                    dim=concat_dim,
                )
            elif i == 5:
                features = torch.concat(
                    model.encoder.blocks[i_layer].calibration_data['mlp2'], 
                    dim=concat_dim,
                )

            if ":" in rank_threshold:
                # apply different thresholds for self-attention and MLP layers
                thresh = float(rank_threshold.split(":")[0]) if i <= 3 else float(rank_threshold.split(":")[1])
            else:
                thresh = float(rank_threshold)

            features = features.cuda().float()
            Y_mean = features.mean(dim=0, keepdim=False).cuda()
            
            # center the data
            features_centered = features - Y_mean

            # compute SVD (X_centered = U * S * Vt)
            U, S, Vt = torch.linalg.svd(features_centered, full_matrices=False)

            # determine the k to be the smallest multiple of 16 that satisfies accuracy constraint
            S_F = S ** 2

            k = -1
            for j in range(0, len(S_F), 16):
                if S_F[:j].sum() / S_F.sum() > thresh:
                    k = j
                    break
            print(f"Layer {i_layer}, Component {i}, k = {k} / {len(S_F)}")
            
            # if k is too big, there is no benefit in using low rank approximation
            if i <= 3:
                if k > 0.5 * base_dim:
                    continue
            else:
                if k > 0.8 * base_dim:
                    continue

            V = Vt.T 
            V_k = V[:, :k] # corresponding to top k principal components

            V_k = V_k.cuda()
            W = layer.weight.T
            w1 = W @ V_k 
            w2 = V_k.T 
            if layer.bias is None:
                bias = Y_mean - Y_mean @ V_k @ V_k.T
            else:
                bias = Y_mean + (layer.bias.half() - Y_mean) @ V_k @ V_k.T
            
            new_layer = LinearLowRank(w1, w2, bias)

            if i == 0:
                model.encoder.blocks[i_layer].attn.query = new_layer
            elif i == 1:
                model.encoder.blocks[i_layer].attn.key = new_layer
            elif i == 2:
                model.encoder.blocks[i_layer].attn.value = new_layer
            elif i == 3:
                model.encoder.blocks[i_layer].attn.out = new_layer
            elif i == 4:
                model.encoder.blocks[i_layer].mlp[0] = new_layer
            elif i == 5:
                model.encoder.blocks[i_layer].mlp[2] = new_layer
            
            features = features.cpu()

        torch.cuda.empty_cache()
    
    model.encoder.is_calibrating = False
    return model 


def main(args):
    model = whisper.load_model(args.base_model)

    # Load pre-compressed model weights from Hugging Face model path if provided
    if getattr(args, "hf_model_path", None):
        from transformers import AutoModel
        new_model = AutoModel.from_pretrained(
            args.hf_model_path,
            trust_remote_code=True, 
            torch_dtype=torch.float16,
        )
        new_model = new_model.model 
        new_model = convert_from_hf_whisper(
            new_model,
            device="cuda",
            use_custom_kernel=False,
        )
        new_model = new_model.to(torch.float16).cuda()
        new_model.is_calibrating = False
        new_model.transcribe = model.transcribe
        model = new_model

    eval_dataset_list = []
    calibrate_dataset = []
    benchs = [
        # "voxpopuli", 
        # "ami", 
        # "earnings22", 
        # "gigaspeech", 
        "librispeech:test.clean", 
        # "librispeech:test.other", 
        # "spgispeech", 
        # "tedlium",
    ]

    # prepare calibration and test datasets from ESB benchmarks
    for bench in benchs:
        split = "test"
        if ":" in bench:
            bench, split = bench.split(":")
        print(bench, split)
        dataset = load_dataset(
            "hf-audio/esb-datasets-test-only-sorted",
            bench,
            split=split,
            streaming=False,
            token=True,
        )
        dataset = dataset.shuffle(seed=42)
        dataset = data_utils.prepare_data(dataset)
        if args.max_eval_samples is None:
            calibrate_dataset.append(dataset.select(range(args.num_calibration_samples)))
            eval_dataset_list.append(dataset.select(range(
                args.num_calibration_samples,
                len(dataset),
            )))

        else:
            eval_dataset_list.append(dataset.select(range(args.max_eval_samples)))
            calibrate_dataset.append(dataset.select(range(
                args.max_eval_samples, 
                args.max_eval_samples + args.num_calibration_samples,
            )))
    
    calibrate_dataset = concatenate_datasets(calibrate_dataset).shuffle(seed=42).select(range(args.num_calibration_samples))

    # Add noise and compute SNR for calibration dataset
    if getattr(args, "noise_std", 0.0) > 0:
        def add_noise_and_snr(batch):
            clean = batch["audio"]["array"].astype(np.float32)
            noisy = add_gaussian_noise(clean, args.noise_std)
            snr = compute_snr(clean, noisy)
            return {"audio": {"array": noisy, "sampling_rate": batch["audio"]["sampling_rate"]}, "snr": snr}
        calibrate_dataset = calibrate_dataset.map(add_noise_and_snr)
        print("Added Gaussian noise to calibration dataset with std:", args.noise_std)
        for i in range(min(5, len(calibrate_dataset))):
            print(f"Calibration sample {i} SNR: {calibrate_dataset[i]['snr']:.2f} dB")
    else:
        # If no noise, set SNR to inf for all
        calibrate_dataset = calibrate_dataset.map(lambda batch: {**batch, "snr": float('inf')})

    if args.low_rank:
        model = apply_low_rank(model, calibrate_dataset, args.rank_threshold)

        if args.save_weight:
            torch.save(model.state_dict(), f"lite-{args.base_model}_{args.rank_threshold}.pth")
    
    print(f"Number of parameters in encoder: {sum(p.numel() for p in model.encoder.parameters())}")

    if not args.do_eval:
        return

    # accuracy benchmark
    total_sum_wer = 0
    for i_bench, dataset in enumerate(eval_dataset_list):
        # Add noise and compute SNR for eval dataset
        if getattr(args, "noise_std", 0.0) > 0:
            def add_noise_and_snr_eval(batch):
                clean = batch["audio"]["array"].astype(np.float32)
                noisy = add_gaussian_noise(clean, args.noise_std)
                snr = compute_snr(clean, noisy)
                return {"audio": {"array": noisy, "sampling_rate": batch["audio"]["sampling_rate"]}, "snr": snr}
            dataset = dataset.map(add_noise_and_snr_eval)
            print(f"Added Gaussian noise to eval dataset {benchs[i_bench]} with std:", args.noise_std)
        else:
            dataset = dataset.map(lambda batch: {**batch, "snr": float('inf')})

        sum_wer = 0
        sum_snr = 0
        wer_metric = evaluate.load("wer")
        for i in tqdm.tqdm(range(len(dataset))):
            audio = dataset[i]["audio"]["array"].astype(np.float32)
            snr = dataset[i]["snr"]
            print(f"SNR for sample {i} in {benchs[i_bench]}: {snr:.2f} dB (noise_std={args.noise_std})")

            pred = transcribe(model, audio)
            wer = wer_metric.compute(
                references=[dataset[i]["norm_text"]], predictions=[data_utils.normalizer(pred)]
            )
            wer = round(100 * wer, 2)
            sum_wer += wer
            sum_snr += snr
            # print("WER:", wer, "%")
        
        with open('output.txt', 'a') as f:
            f.write(f"Number of parameters: {sum(p.numel() for p in model.encoder.parameters())}\n")
            f.write(str(args) + '\n')
            f.write(f"{benchs[i_bench]}\n")
            f.write(f"Average WER: {sum_wer / len(dataset)}\n")
            f.write(f"Average SNR: {sum_snr / len(dataset)}\n")
            f.write("=====================================\n")
        total_sum_wer += sum_wer / len(dataset)

    total_avg_wer = total_sum_wer / len(eval_dataset_list)
    model_name = ('lite-' if args.low_rank else '') + 'whisper-' + args.base_model + ('-' + args.rank_threshold if args.low_rank else '')
    # with open('output.txt', 'a') as f:
    #     f.write(f'Final model evaluation results:\n')
    #     f.write(f'max_eval_samples:{args.max_eval_samples} | {model_name} | total_avg_wer: {total_avg_wer} | encoder_params: {sum(p.numel() for p in model.encoder.parameters())} | decoder_params: {sum(p.numel() for p in model.decoder.parameters())}\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        type=str,
        default="large-v3",
        help="Base model name. *E.g.* `'large-v3'` for the Large v3 model, or `'turbo'` for the Turbo model."
    )
    parser.add_argument(
        "--hf_model_path",
        type=str,
        default=None,
        help="Path, directory, or Hugging Face repo id to pre-compressed model weights (supports .safetensors, sharded safetensors, or .bin). If provided, loads these weights after initializing the base model."
    )
    parser.add_argument(
        "--low_rank",
        action="store_true",
        help="Whether to apply low-rank approximation for the activations",
    )
    parser.add_argument(
        "--rank_threshold",
        type=str,
        default="0.99:0.999",
        help="The threshold for the rank approximation. If two values are provided, the first one is for self-attention layers and the second one is for MLP layers",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=1000,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=100,
        help="Number of samples to be used for calibration",
    )
    parser.add_argument(
        "--save_weight",
        action="store_true",
        help="Whether to save the compressed weights",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.0,
        help="Standard deviation of Gaussian white noise to add to audio. 0.0 means no noise.",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to evaluate the model",
    )
    args = parser.parse_args()

    main(args)

