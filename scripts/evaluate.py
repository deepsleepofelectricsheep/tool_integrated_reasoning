"""Evaluate a trained model on the GSM8k or MATH eval split.

Works for both CoT-trained and TIR-trained models — the only difference is
which answer-extraction function is used (determined by --answer_format).

Examples
--------
# Evaluate a CoT-trained model:
python scripts/evaluate.py --model checkpoints/cot_gsm8k/final --dataset gsm8k
python scripts/evaluate.py --model checkpoints/cot_math/final  --dataset math

# Quick smoke test on 50 examples:
python scripts/evaluate.py --model checkpoints/cot_gsm8k/final --dataset gsm8k --limit 50

# TIR-trained model (interactive eval with code execution):
python scripts/evaluate.py --model checkpoints/tir_gsm8k/final --dataset gsm8k --answer_format boxed --prompt_format tir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wandb

from src.training.dataset import COT_SYSTEM_PROMPT, TIR_SYSTEM_PROMPT
from src.utils.math_grading import (
    extract_boxed_answer,
    extract_gsm8k_answer,
    grade_answer,
)

# ---------------------------------------------------------------------------
# Dataset configs (eval split)
# ---------------------------------------------------------------------------

EVAL_CONFIGS = {
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "split":   "test",
        "problem_key":  "question",
        "solution_key": "answer",
    },
    "math": {
        "hf_path": "lighteval/MATH",
        "hf_name": "all",
        "split":   "test",
        "problem_key":  "problem",
        "solution_key": "solution",
    },
}


def get_ground_truth(sample: dict, dataset: str) -> str:
    if dataset == "gsm8k":
        return extract_gsm8k_answer(sample["answer"]) or ""
    return extract_boxed_answer(sample["solution"]) or ""


def extract_predicted(output: str, answer_format: str, dataset: str) -> str | None:
    """Pull the model's answer out of its generated text."""
    if answer_format == "boxed":
        return extract_boxed_answer(output)
    # "native": use the dataset's own format
    if dataset == "gsm8k":
        # Model was trained on the #### format; fall back to \boxed{} if absent
        ans = extract_gsm8k_answer(output)
        return ans if ans is not None else extract_boxed_answer(output)
    return extract_boxed_answer(output)


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------

def build_prompts(samples: list[dict], dataset: str, tokenizer, prompt_format: str = "cot") -> list[str]:
    cfg = EVAL_CONFIGS[dataset]
    system_prompt = TIR_SYSTEM_PROMPT if prompt_format == "tir" else COT_SYSTEM_PROMPT
    prompts = []
    for s in samples:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": s[cfg["problem_key"]]},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    return prompts


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    device: torch.device,
) -> list[str]:
    """Generate responses for a batch of prompts, return decoded strings."""
    # Left-pad so all sequences align for generation
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (strip the prompt)
    results = []
    for ids in out_ids:
        prompt_len = enc["input_ids"].shape[1]
        new_ids = ids[prompt_len:]
        results.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True, help="Path to checkpoint or HF model ID")
    p.add_argument("--dataset", choices=["gsm8k", "math"], required=True)
    p.add_argument("--answer_format", choices=["native", "boxed"], default="native",
                   help="'native' uses #### for gsm8k; 'boxed' always uses \\boxed{}")
    p.add_argument("--prompt_format", choices=["cot", "tir"], default="cot",
                   help="'cot': batched generation; 'tir': interactive loop with code execution")
    p.add_argument("--batch_size",    type=int, default=8)
    p.add_argument("--max_new_tokens",type=int, default=1024)
    p.add_argument("--max_rounds",    type=int, default=3,
                   help="Max tool-use rounds per problem (TIR eval only)")
    p.add_argument("--code_timeout",  type=int, default=0,
                   help="Code execution timeout in seconds; 0 = auto (5s gsm8k, 10s math)")
    p.add_argument("--limit",         type=int, default=None,
                   help="Evaluate on at most this many examples (for quick tests)")
    p.add_argument("--output_dir",    default="results",
                   help="Directory for the per-example JSONL output")
    p.add_argument("--wandb_project",  type=str, default=None)
    p.add_argument("--wandb_group",    type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"          # required for batched generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    cfg  = EVAL_CONFIGS[args.dataset]
    ds   = load_dataset(cfg["hf_path"], cfg["hf_name"], split=cfg["split"])
    samples = list(ds)
    if args.limit:
        samples = samples[: args.limit]
    print(f"Evaluating on {len(samples)} examples from {args.dataset} ({cfg['split']})")

    # ------------------------------------------------------------------
    # Evaluate in batches
    # ------------------------------------------------------------------
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_tag = Path(args.model).name
    results_file = out_path / f"{model_tag}_{args.dataset}.jsonl"

    correct = total = 0
    records: list[dict] = []

    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            group=args.wandb_group,
            name=args.wandb_run_name,
        )

    if args.prompt_format == "tir":
        # Interactive TIR evaluation: generate one example at a time with real
        # code execution. This matches the training distribution (the model was
        # trained on trajectories that always had real executor output injected).
        from src.data_generation.generator import generate_tir_trajectory
        code_timeout = args.code_timeout or (5 if args.dataset == "gsm8k" else 10)
        print(f"TIR interactive eval (max_rounds={args.max_rounds}, code_timeout={code_timeout}s)")

        for sample in tqdm(samples, desc="Evaluating (TIR)"):
            output    = generate_tir_trajectory(
                model, tokenizer,
                problem=sample[cfg["problem_key"]],
                max_rounds=args.max_rounds,
                max_new_tokens=args.max_new_tokens,
                code_timeout=code_timeout,
            )
            gt        = get_ground_truth(sample, args.dataset)
            predicted = extract_predicted(output, args.answer_format, args.dataset)
            is_correct = grade_answer(predicted, gt, args.dataset)
            correct   += is_correct
            total     += 1
            records.append({
                "problem":          sample[cfg["problem_key"]],
                "ground_truth":     gt,
                "model_output":     output,
                "predicted_answer": predicted,
                "correct":          is_correct,
            })
    else:
        # Batched CoT evaluation
        prompts_all = build_prompts(samples, args.dataset, tokenizer, args.prompt_format)
        for start in tqdm(range(0, len(samples), args.batch_size), desc="Evaluating (CoT)"):
            batch_samples = samples[start : start + args.batch_size]
            batch_prompts = prompts_all[start : start + args.batch_size]
            batch_outputs = generate_batch(
                model, tokenizer, batch_prompts, args.max_new_tokens, device
            )
            for sample, output in zip(batch_samples, batch_outputs):
                gt        = get_ground_truth(sample, args.dataset)
                predicted = extract_predicted(output, args.answer_format, args.dataset)
                is_correct = grade_answer(predicted, gt, args.dataset)
                correct   += is_correct
                total     += 1
                records.append({
                    "problem":          sample[cfg["problem_key"]],
                    "ground_truth":     gt,
                    "model_output":     output,
                    "predicted_answer": predicted,
                    "correct":          is_correct,
                })

    # ------------------------------------------------------------------
    # Save & report
    # ------------------------------------------------------------------
    with open(results_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    accuracy = correct / total if total else 0.0
    print(f"\n{'='*50}")
    print(f"Dataset : {args.dataset} ({cfg['split']})")
    print(f"Model   : {args.model}")
    print(f"Accuracy: {correct}/{total} ({accuracy:.1%})")
    print(f"Results : {results_file}")
    print(f"{'='*50}")

    if use_wandb:
        wandb.log({"eval/accuracy": accuracy, "eval/correct": correct, "eval/total": total})
        wandb.finish()


if __name__ == "__main__":
    main()
