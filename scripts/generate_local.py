"""Generate TIR training data locally on a GPU.

Usage:
    python scripts/generate_local.py --dataset gsm8k --n 1000
    python scripts/generate_local.py --dataset math --n 1000 --resume
"""

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

MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct"

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
    # MATH: extract from \boxed{} inside the solution field
    return extract_boxed_answer(sample["solution"]) or ""


def load_samples(dataset: str, n: int, seed: int) -> tuple[list[dict], list[int]]:
    cfg = DATASETS[dataset]
    ds = load_dataset(cfg["hf_path"], cfg["hf_name"], split=cfg["split"])
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    return [ds[i] for i in indices], indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gsm8k", "math"], required=True)
    parser.add_argument("--n", type=int, default=1000, help="Number of examples to generate")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--code_timeout", type=int, default=None,
                        help="Seconds per code execution (default: 5 for gsm8k, 10 for math)")
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--do_sample", action="store_true", help="Enable nucleus sampling")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--resume", action="store_true", help="Skip already-completed examples")
    args = parser.parse_args()

    out_path = Path(args.output_dir) / f"tir_{args.dataset}_{args.n}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")

    # Load dataset samples
    samples, indices = load_samples(args.dataset, args.n, args.seed)
    print(f"Loaded {len(samples)} samples from {args.dataset}")

    # Resume: collect already-finished original indices
    done: set[int] = set()
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                done.add(json.loads(line)["original_index"])
        print(f"Resuming — {len(done)} examples already done, skipping")

    code_timeout = args.code_timeout or (5 if args.dataset == "gsm8k" else 10)
    print(f"Code execution timeout: {code_timeout}s")

    correct = total = 0
    problem_key = DATASETS[args.dataset]["problem_key"]

    with open(out_path, "a" if args.resume else "w") as fout:
        for sample, orig_idx in tqdm(zip(samples, indices), total=len(samples)):
            if orig_idx in done:
                continue

            problem = sample[problem_key]
            ground_truth = get_ground_truth(sample, args.dataset)

            trajectory = generate_tir_trajectory(
                model,
                tokenizer,
                problem,
                max_rounds=args.max_rounds,
                max_new_tokens=args.max_new_tokens,
                code_timeout=code_timeout,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            predicted = extract_boxed_answer(trajectory)
            is_correct = grade_answer(predicted, ground_truth, args.dataset)
            total += 1
            correct += is_correct

            record = {
                "dataset": args.dataset,
                "original_index": orig_idx,
                "problem": problem,
                "ground_truth": ground_truth,
                "trajectory": trajectory,
                "predicted_answer": predicted,
                "correct": is_correct,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

    print(f"\nDone. {correct}/{total} correct ({correct/total:.1%})")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
