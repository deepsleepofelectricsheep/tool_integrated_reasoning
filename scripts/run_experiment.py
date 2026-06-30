"""Controlled CoT vs TIR comparison experiment.

Trains both a CoT and a TIR model from the same Qwen2.5-0.5B checkpoint using
the same N training examples and identical hyperparameters, then evaluates both
on the dataset's eval split and prints a comparison table.

All output (subprocess + summary) is written to a logfile in --output_dir.
After evaluation, 5 correct and 5 incorrect trajectories are sampled from
each model and saved under output_dir/trajectories/.

All runs are logged to W&B under the same group so training curves and eval
accuracy are visible side-by-side on a single dashboard.

Examples
--------
# Full experiment on GSM8k with 500 examples:
python scripts/run_experiment.py \\
    --dataset gsm8k \\
    --n 500 \\
    --tir_data_path data/tir_gsm8k_1000.jsonl \\
    --output_dir checkpoints/exp_gsm8k_n500 \\
    --wandb_project tir-experiments

# Quick smoke test (50 training examples, limit eval to 100 examples):
python scripts/run_experiment.py \\
    --dataset gsm8k \\
    --n 50 \\
    --tir_data_path data/tir_gsm8k_1000.jsonl \\
    --output_dir checkpoints/exp_gsm8k_smoke \\
    --eval_limit 100 \\
    --wandb_project tir-experiments
"""

from __future__ import annotations

import argparse
import io
import json
import random
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Tee: writes to both terminal and logfile simultaneously
# ---------------------------------------------------------------------------

class _Tee(io.TextIOBase):
    """Wraps two text streams, writing to both on every write."""

    def __init__(self, primary: io.TextIOBase, secondary: io.TextIOBase) -> None:
        self._primary   = primary
        self._secondary = secondary

    def write(self, s: str) -> int:
        self._primary.write(s)
        self._secondary.write(s)
        self._secondary.flush()
        return len(s)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()


# ---------------------------------------------------------------------------
# Subprocess runner (tees output to both terminal and logfile)
# ---------------------------------------------------------------------------

def run(cmd: list[str]) -> None:
    """Stream a subprocess to stdout; raise on non-zero exit."""
    print(f"\n$ {' '.join(cmd)}\n{'─'*60}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


# ---------------------------------------------------------------------------
# Post-evaluation helpers
# ---------------------------------------------------------------------------

def read_results(results_jsonl: Path) -> list[dict]:
    records = []
    with open(results_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_accuracy(records: list[dict]) -> tuple[int, int]:
    correct = sum(int(r["correct"]) for r in records)
    return correct, len(records)


def save_trajectories(records: list[dict], out_dir: Path, prefix: str, seed: int, k: int = 5) -> None:
    """Sample k correct and k incorrect examples and save to out_dir."""
    rng       = random.Random(seed)
    correct   = [r for r in records if     r["correct"]]
    incorrect = [r for r in records if not r["correct"]]
    rng.shuffle(correct)
    rng.shuffle(incorrect)

    out_dir.mkdir(parents=True, exist_ok=True)
    for label, subset in [("correct", correct[:k]), ("incorrect", incorrect[:k])]:
        path = out_dir / f"{prefix}_{label}.jsonl"
        with open(path, "w") as f:
            for r in subset:
                f.write(json.dumps(r) + "\n")
        print(f"  {label:>9} trajectories ({len(subset)}) → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a controlled CoT vs TIR comparison experiment."
    )
    # Experiment identity
    p.add_argument("--dataset",       choices=["gsm8k", "math"], required=True)
    p.add_argument("--n",             type=int, required=True,
                   help="Number of training examples for both CoT and TIR")
    p.add_argument("--tir_data_path", required=True,
                   help="Path to TIR JSONL file (correct=True rows will be used)")
    p.add_argument("--output_dir",    required=True,
                   help="Root directory for checkpoints, results, and logfile")
    p.add_argument("--seed",          type=int, default=42)

    # Shared training hyperparameters
    p.add_argument("--epochs",      type=int,   default=1)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--warmup_frac", type=float, default=0.03)
    p.add_argument("--grad_accum",  type=int,   default=4)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--max_length",  type=int,   default=2048)
    p.add_argument("--log_every",   type=int,   default=10)
    p.add_argument("--save_every",  type=int,   default=None)

    # Evaluation
    p.add_argument("--eval_batch_size",  type=int, default=8)
    p.add_argument("--eval_max_tokens",  type=int, default=1024)
    p.add_argument("--eval_limit",       type=int, default=None,
                   help="Cap eval at N examples (useful for smoke tests)")

    # W&B
    p.add_argument("--wandb_project", type=str, default=None,
                   help="W&B project name; omit to disable W&B logging")

    # Control flow
    p.add_argument("--skip_cot_train", action="store_true",
                   help="Skip CoT training (use existing checkpoint)")
    p.add_argument("--skip_tir_train", action="store_true",
                   help="Skip TIR training (use existing checkpoint)")
    p.add_argument("--skip_eval",      action="store_true",
                   help="Skip evaluation")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    python = sys.executable
    root   = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    log_path = root / "experiment.log"

    with open(log_path, "w") as log_file:
        # Tee all print() output to the logfile
        sys.stdout = _Tee(sys.__stdout__, log_file)

        try:
            _run_experiment(args, python, root, log_file)
        finally:
            sys.stdout = sys.__stdout__

    print(f"\nFull log saved → {log_path}")


def _run_experiment(args: argparse.Namespace, python: str, root: Path, log_file) -> None:
    cot_ckpt    = root / "cot" / "final"
    tir_ckpt    = root / "tir" / "final"
    cot_results = root / "results" / "cot"
    tir_results = root / "results" / "tir"
    traj_dir    = root / "trajectories"

    group = f"{args.dataset}_n{args.n}_ep{args.epochs}_lr{args.lr:.0e}"

    print(f"Experiment: {group}")
    print(f"Output dir: {root}")
    print(f"Log file  : {root / 'experiment.log'}")

    shared_train = [
        "--epochs",      str(args.epochs),
        "--batch_size",  str(args.batch_size),
        "--lr",          str(args.lr),
        "--warmup_frac", str(args.warmup_frac),
        "--grad_accum",  str(args.grad_accum),
        "--grad_clip",   str(args.grad_clip),
        "--max_length",  str(args.max_length),
        "--log_every",   str(args.log_every),
        "--seed",        str(args.seed),
        "--n",           str(args.n),
    ]
    if args.save_every:
        shared_train += ["--save_every", str(args.save_every)]
    if args.wandb_project:
        shared_train += ["--wandb_project", args.wandb_project, "--wandb_group", group]

    shared_eval = [
        "--dataset",        args.dataset,
        "--batch_size",     str(args.eval_batch_size),
        "--max_new_tokens", str(args.eval_max_tokens),
    ]
    if args.eval_limit:
        shared_eval += ["--limit", str(args.eval_limit)]
    if args.wandb_project:
        shared_eval += ["--wandb_project", args.wandb_project, "--wandb_group", group]

    # ------------------------------------------------------------------
    # 1. Train CoT
    # ------------------------------------------------------------------
    if not args.skip_cot_train:
        print(f"\n{'='*60}")
        print(f"STEP 1 / 4 — Train CoT  (n={args.n}, epochs={args.epochs})")
        print(f"{'='*60}")
        run([
            python, "scripts/train_cot.py",
            "--dataset",        args.dataset,
            "--output_dir",     str(root / "cot"),
            "--wandb_run_name", "cot_train",
        ] + shared_train)
    else:
        print(f"\nSkipping CoT training — using checkpoint at {cot_ckpt}")

    # ------------------------------------------------------------------
    # 2. Train TIR
    # ------------------------------------------------------------------
    if not args.skip_tir_train:
        print(f"\n{'='*60}")
        print(f"STEP 2 / 4 — Train TIR  (n={args.n}, epochs={args.epochs})")
        print(f"{'='*60}")
        run([
            python, "scripts/train_tir.py",
            "--data_path",      args.tir_data_path,
            "--output_dir",     str(root / "tir"),
            "--wandb_run_name", "tir_train",
        ] + shared_train)
    else:
        print(f"\nSkipping TIR training — using checkpoint at {tir_ckpt}")

    if args.skip_eval:
        return

    # ------------------------------------------------------------------
    # 3. Evaluate CoT
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 3 / 4 — Evaluate CoT")
    print(f"{'='*60}")
    run([
        python, "scripts/evaluate.py",
        "--model",          str(cot_ckpt),
        "--answer_format",  "native",
        "--prompt_format",  "cot",
        "--output_dir",     str(cot_results),
        "--wandb_run_name", "cot_eval",
    ] + shared_eval)

    # ------------------------------------------------------------------
    # 4. Evaluate TIR
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 4 / 4 — Evaluate TIR")
    print(f"{'='*60}")
    run([
        python, "scripts/evaluate.py",
        "--model",          str(tir_ckpt),
        "--answer_format",  "boxed",
        "--prompt_format",  "tir",
        "--output_dir",     str(tir_results),
        "--wandb_run_name", "tir_eval",
    ] + shared_eval)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    cot_file = cot_results / f"final_{args.dataset}.jsonl"
    tir_file = tir_results / f"final_{args.dataset}.jsonl"

    cot_records = read_results(cot_file)
    tir_records = read_results(tir_file)

    cot_correct, cot_total = read_accuracy(cot_records)
    tir_correct, tir_total = read_accuracy(tir_records)
    cot_acc = cot_correct / cot_total if cot_total else 0.0
    tir_acc = tir_correct / tir_total if tir_total else 0.0
    delta   = tir_acc - cot_acc

    print(f"\n{'='*60}")
    print(f"RESULTS — {args.dataset.upper()}  (n_train={args.n}, epochs={args.epochs})")
    print(f"{'='*60}")
    print(f"{'Model':<8}  {'Correct':>8}  {'Total':>8}  {'Accuracy':>10}")
    print(f"{'─'*44}")
    print(f"{'CoT':<8}  {cot_correct:>8}  {cot_total:>8}  {cot_acc:>9.1%}")
    print(f"{'TIR':<8}  {tir_correct:>8}  {tir_total:>8}  {tir_acc:>9.1%}")
    print(f"{'─'*44}")
    sign = "+" if delta >= 0 else ""
    print(f"{'Δ (TIR−CoT)':<20}  {sign}{delta:.1%}")
    print(f"{'='*60}")
    if args.wandb_project:
        print(f"\nW&B group '{group}' in project '{args.wandb_project}'")

    # ------------------------------------------------------------------
    # Trajectory samples
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"TRAJECTORY SAMPLES (5 correct + 5 incorrect per model)")
    print(f"{'='*60}")
    print("CoT:")
    save_trajectories(cot_records, traj_dir, prefix="cot", seed=args.seed)
    print("TIR:")
    save_trajectories(tir_records, traj_dir, prefix="tir", seed=args.seed)


if __name__ == "__main__":
    main()
