"""Output Space Shaping: generate additional TIR trajectories via nucleus sampling.

Takes a fine-tuned TIR model, generates multiple candidate trajectories per
training problem using nucleus sampling, keeps only the correct ones, and
merges them with the original TIR dataset.

Usage:
    python scripts/generate_oss.py \
        --model checkpoints/tir_gsm8k/final \
        --dataset gsm8k \
        --n 1000 \
        --num_samples 5 \
        --base_tir_data data/tir_gsm8k_1000.jsonl \
        --output_path data/tir_gsm8k_oss.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_generation.generator import generate_tir_trajectory
from src.utils.math_grading import extract_boxed_answer, extract_gsm8k_answer, grade_answer

DATASETS = {
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "split": "train",
        "problem_key": "question",
    },
    "math": {
        "hf_path": "lighteval/MATH",
        "hf_name": "all",
        "split": "train",
        "problem_key": "problem",
    },
}


def get_ground_truth(sample: dict, dataset: str) -> str:
    if dataset == "gsm8k":
        return extract_gsm8k_answer(sample["answer"]) or ""
    return extract_boxed_answer(sample["solution"]) or ""


def load_samples(dataset: str, n: int, seed: int) -> tuple[list[dict], list[int]]:
    cfg = DATASETS[dataset]
    ds = load_dataset(cfg["hf_path"], cfg["hf_name"], split=cfg["split"])
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    return [ds[i] for i in indices], indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OSS trajectories and merge with base TIR data."
    )
    parser.add_argument("--model", required=True, help="Path to fine-tuned TIR checkpoint")
    parser.add_argument("--dataset", choices=["gsm8k", "math"], required=True)
    parser.add_argument("--n", type=int, default=1000, help="Number of training problems")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Candidate trajectories per problem")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--code_timeout", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_tir_data", required=True,
                        help="Path to original TIR JSONL to merge with")
    parser.add_argument("--output_path", required=True,
                        help="Where to write the merged JSONL")
    args = parser.parse_args()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    oss_raw_path = out_path.with_suffix(".oss_only.jsonl")

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")

    samples, indices = load_samples(args.dataset, args.n, args.seed)
    print(f"Loaded {len(samples)} problems from {args.dataset}")
    print(f"Generating {args.num_samples} candidates per problem "
          f"(temperature={args.temperature}, top_p={args.top_p})")

    code_timeout = args.code_timeout or (5 if args.dataset == "gsm8k" else 10)
    problem_key = DATASETS[args.dataset]["problem_key"]

    total_candidates = 0
    total_correct = 0

    with open(oss_raw_path, "w") as fout:
        for sample, orig_idx in tqdm(zip(samples, indices), total=len(samples)):
            problem = sample[problem_key]
            ground_truth = get_ground_truth(sample, args.dataset)

            for sample_i in range(args.num_samples):
                trajectory = generate_tir_trajectory(
                    model,
                    tokenizer,
                    problem,
                    max_rounds=args.max_rounds,
                    max_new_tokens=args.max_new_tokens,
                    code_timeout=code_timeout,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                predicted = extract_boxed_answer(trajectory)
                is_correct = grade_answer(predicted, ground_truth, args.dataset)
                total_candidates += 1
                total_correct += is_correct

                record = {
                    "dataset": args.dataset,
                    "original_index": orig_idx,
                    "problem": problem,
                    "ground_truth": ground_truth,
                    "trajectory": trajectory,
                    "predicted_answer": predicted,
                    "correct": is_correct,
                    "source": "oss",
                    "sample_index": sample_i,
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()

    print(f"\nOSS generation complete:")
    print(f"  Total candidates: {total_candidates}")
    print(f"  Correct: {total_correct} ({total_correct/total_candidates:.1%})")

    # Merge: original base data + new correct OSS trajectories
    base_records = []
    with open(args.base_tir_data) as f:
        for line in f:
            line = line.strip()
            if line:
                base_records.append(json.loads(line))

    oss_correct = []
    with open(oss_raw_path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r["correct"]:
                    oss_correct.append(r)

    with open(out_path, "w") as fout:
        for r in base_records:
            fout.write(json.dumps(r) + "\n")
        for r in oss_correct:
            fout.write(json.dumps(r) + "\n")

    base_correct = sum(1 for r in base_records if r.get("correct"))
    print(f"\nMerged dataset: {out_path}")
    print(f"  Base records: {len(base_records)} ({base_correct} correct)")
    print(f"  New OSS correct: {len(oss_correct)}")
    print(f"  Total records: {len(base_records) + len(oss_correct)}")
    print(f"  Total correct (for training): {base_correct + len(oss_correct)}")


if __name__ == "__main__":
    main()
