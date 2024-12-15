import argparse
import gc
from functools import partial
from typing import Literal
import os.path

import torch
import tqdm
from datasets import load_dataset
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from awq.quantize.pre_quant_lester import apply_awq, run_awq
from awq.quantize.fake_quant_lester import quantize_opt_model

ActQuantType = Literal["per_token", "per_tensor", "none", "per_channel"]


def evaluate(model, tokenizer, nsamples: int = 40):
    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")

    testenc = testenc.input_ids.to(model.device)
    model = model.eval()

    nlls = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * 2048
        nlls.append(neg_log_likelihood)

    return torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))


def get_model_size(model: nn.Module, data_width=16, group_size=-1):

    if group_size != -1:
        data_width += (16 + 4) / group_size

    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width


Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


def main(args):

    ### parsing configs
    model_path = args.model_path

    original_model_n_bits = 16
    torch_dtype = torch.float16 if original_model_n_bits == 16 else torch.float32
    q_config = {
        "zero_point": True,  # by default True
        "q_group_size": args.q_group_size,  # whether to use group quantization
        "w_n_bits": args.w_n_bits,
        "a_n_bits": args.a_n_bits,
        "act_quant": args.act_quant,
    }

    ### constructing model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map="auto"
    )

    if args.run_awq or (len(args.awq_path) >= 1):
        if args.load_awq_result:
            print("loading AWQ results...")
            awq_results = torch.load(args.awq_path)
        else:
            awq_results = run_awq(
                model,
                tokenizer,
                w_bit=q_config["w_n_bits"],
                q_config=q_config,
                n_samples=128,
                seqlen=512,
            )

        # apply the AWQ results
        # re-construct the model for safety, because run_awq will in-place modify the model
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map="auto"
        )
        apply_awq(model, awq_results)

        if (len(args.awq_path) > 1) and args.save_awq_result:
            torch.save(awq_results, args.awq_path)

    model = quantize_opt_model(
        model,
        w_n_bits=q_config["w_n_bits"],
        a_n_bits=q_config["a_n_bits"],
        act_quant=(
            q_config["act_quant"]
            if len(args.act_quant_override) <= 1
            else args.act_quant_override
        ),
        group_size=q_config["q_group_size"],
    )

    torch.cuda.empty_cache()
    model.cuda()

    # Evaluate the model
    model_perplexity = evaluate(
        model,
        tokenizer,
        nsamples=args.n_evalution_samples,
    )
    model_size = get_model_size(
        model,
        data_width=q_config["w_n_bits"],
        group_size=q_config["q_group_size"],
    )
    print(f"\nmodel perplexity: {model_perplexity:.2f}")
    print(f"model size: {model_size/MiB:.2f} MiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_kwrd", type=str, choices=["opt-1.3b", "opt-2.7b", "opt-6.7b"]
    )
    parser.add_argument(
        "--run_awq", action="store_true", help="whether to run AWQ or not"
    )
    parser.add_argument(
        "--load_awq_result",
        action="store_true",
        help="whether to load AWQ result or not",
    )
    parser.add_argument(
        "--save_awq_result",
        action="store_true",
        help="whether to save AWQ result or not",
    )
    parser.add_argument(
        "--awq_path",
        type=str,
        default="",
        help="path to save/load AWQ results",
    )
    parser.add_argument(
        "--act_quant_override",
        type=str,
        default="",
        choices=["per_token", "per_tensor", "none", "per_channel"],
        help="override the activation quantization type",
    )
    parser.add_argument("--w_n_bits", type=int, default=4)
    parser.add_argument("--a_n_bits", type=int, default=4)
    parser.add_argument("--q_group_size", type=int, default=128)
    parser.add_argument(
        "--act_quant",
        type=str,
        default="per_channel",
        choices=["per_token", "per_tensor", "none", "per_channel"],
    )

    parser.add_argument("--n_evalution_samples", type=int, default=40)
    # parser.add_argument

    args = parser.parse_args()
    args.model_path = f"facebook/{args.model_kwrd}"

    #### print args as key value pairs
    print("\n\n")
    print("Arguments:")
    for key, value in vars(args).items():
        print(" " * 4, key, ":", value)
    print("\n\n")

    main(args)
