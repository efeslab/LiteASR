import argparse
import numpy as np 
import torch
import evaluate
from normalizer import data_utils
import tqdm
import whisper
from datasets import load_dataset, concatenate_datasets

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
        padding=None, 
        temperature=0.0,
        token_list=None, 
    ):
    result = model.transcribe(
        path,
        task="transcribe",
        language="en",
        temperature=temperature,
        padding=padding,
        token_list=token_list,
    )
    return result["text"]

def apply_low_rank(model, dataset, rank_threshold):
    model.encoder.is_calibrating = True
    base_dim = model.dims.n_audio_state # 1280 for large-v3
    
    for i in tqdm.tqdm(range(len(dataset))):
        model.encoder.register_audio_feature(None)
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
            
            new_layer = whisper.model.LinearLowRank(
                w1, w2, bias
            )

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

    eval_dataset_list = []
    calibrate_dataset = []
    benchs = [
        "voxpopuli", 
        "ami", 
        "earnings22", 
        "gigaspeech", 
        "librispeech:test.clean", 
        "librispeech:test.other", 
        "spgispeech", 
        "tedlium",
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

    if args.low_rank:
        model = apply_low_rank(model, calibrate_dataset, args.rank_threshold)

        if args.save_weight:
            torch.save(model.state_dict(), f"lite-{args.base_model}_{args.rank_threshold}.pth")
    
    print(f"Number of parameters in encoder: {sum(p.numel() for p in model.encoder.parameters())}")

    if not args.do_eval:
        return

    # accuracy benchmark
    for i_bench, dataset in enumerate(eval_dataset_list):
        sum_wer = 0
        wer_metric = evaluate.load("wer")
        for i in range(len(dataset)):
            model.encoder.register_audio_feature(None)
            audio = dataset[i]["audio"]["array"].astype(np.float32)

            pred = transcribe(model, audio)
            wer = wer_metric.compute(
                references=[dataset[i]["norm_text"]], predictions=[data_utils.normalizer(pred)]
            )
            wer = round(100 * wer, 2)
            sum_wer += wer
            # print("WER:", wer, "%")
        
        with open('output.txt', 'a') as f:
            f.write(f"Number of parameters: {sum(p.numel() for p in model.encoder.parameters())}\n")
            f.write(str(args) + '\n')
            f.write(f"{benchs[i_bench]}\n")
            f.write(f"Average WER: {sum_wer / len(dataset)}\n")
            f.write("=====================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        type=str,
        default="large-v3",
        help="Base model name. *E.g.* `'large-v3'` for the Large v3 model, or `'turbo'` for the Turbo model."
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
        "--do_eval",
        action="store_true",
        help="Whether to evaluate the model",
    )
    args = parser.parse_args()

    main(args)
