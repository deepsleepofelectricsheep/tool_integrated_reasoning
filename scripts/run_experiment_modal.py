"""Run the full CoT vs TIR (vs TIR+OSS) experiment on Modal Labs.

Mirrors run_experiment.py but executes training and evaluation on Modal GPUs.
When --oss is enabled, OSS data generation is parallelized across multiple
containers via Modal's .map(), then training and evaluation run on a single GPU.

Small result files (experiment.log, results JSONL, trajectories) are downloaded
to the local output_dir. Model checkpoints remain in the 'tir-experiments'
Modal Volume and can be retrieved later with:
    modal volume get tir-experiments <experiment_name>/cot/final ./local_path

Usage
-----
# 2-way comparison (CoT vs TIR):
modal run scripts/run_experiment_modal.py \\
    --dataset gsm8k --n 500 \\
    --tir-data-path data/tir_gsm8k_1000.jsonl \\
    --output-dir results/exp_gsm8k_n500

# 3-way comparison (CoT vs TIR vs TIR+OSS):
modal run scripts/run_experiment_modal.py \\
    --dataset gsm8k --n 1000 \\
    --tir-data-path data/tir_gsm8k_1000.jsonl \\
    --output-dir results/exp_gsm8k_n1000_oss \\
    --oss --oss-num-samples 5

Note: Modal converts Python parameter underscores to hyphens in the CLI,
so --tir_data_path becomes --tir-data-path, --output_dir becomes --output-dir, etc.

W&B note: to enable W&B logging, create a Modal secret named "wandb" containing
WANDB_API_KEY, uncomment the `secrets` lines in @app.function, and pass
--wandb-project.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Image + volumes
# ---------------------------------------------------------------------------

APP_NAME        = "tir-experiment"
HF_CACHE        = "/vol/hf_cache"
EXPERIMENTS_DIR = "/vol/experiments"

_root       = Path(__file__).resolve().parent.parent
_src_dir    = _root / "src"
_scripts_dir = Path(__file__).resolve().parent

hf_vol          = modal.Volume.from_name("tir-hf-cache",    create_if_missing=True)
experiments_vol = modal.Volume.from_name("tir-experiments", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "datasets>=2.20.0",
        "sympy>=1.13.0",
        "tqdm",
        "huggingface_hub>=0.24.0",
        "wandb>=0.17.0",
    )
    .add_local_dir(_src_dir,     remote_path="/root/src")
    .add_local_dir(_scripts_dir, remote_path="/root/scripts")
)

app = modal.App(APP_NAME, image=image)


# ---------------------------------------------------------------------------
# Remote function: training + evaluation (single GPU)
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",
    volumes={
        HF_CACHE:        hf_vol,
        EXPERIMENTS_DIR: experiments_vol,
    },
    timeout=43200,   # 12 hours
    # secrets=[modal.Secret.from_name("wandb")],  # uncomment for W&B logging
)
def run_experiment_fn(
    experiment_name: str,
    dataset: str,
    tir_data: str,           # JSONL file contents (passed from local machine)
    cli_args: list[str],     # forwarded verbatim to run_experiment.py
) -> dict[str, str]:         # {relative_path: file_content} for small result files
    import os
    import subprocess

    os.environ["HF_HOME"] = HF_CACHE
    os.chdir("/root")   # scripts/train_cot.py etc. are at /root/scripts/

    # Write TIR data to the experiments volume so run_experiment.py can read it
    exp_dir      = Path(EXPERIMENTS_DIR) / experiment_name
    tir_data_path = exp_dir / "tir_data.jsonl"
    exp_dir.mkdir(parents=True, exist_ok=True)
    tir_data_path.write_text(tir_data)
    experiments_vol.commit()

    cmd = [
        sys.executable, "scripts/run_experiment.py",
        "--tir_data_path", str(tir_data_path),
        "--output_dir",    str(exp_dir),
    ] + cli_args

    subprocess.run(cmd, check=True)
    experiments_vol.commit()

    # Return small result files to the local machine.
    # Checkpoints (cot/final/, tir/final/) are large — they stay in the Volume.
    _RESULT_PATHS = [
        "experiment.log",
        f"results/cot/final_{dataset}.jsonl",
        f"results/tir/final_{dataset}.jsonl",
        f"results/tir_oss/final_{dataset}.jsonl",
        "trajectories/cot_correct.jsonl",
        "trajectories/cot_incorrect.jsonl",
        "trajectories/tir_correct.jsonl",
        "trajectories/tir_incorrect.jsonl",
        "trajectories/tir_oss_correct.jsonl",
        "trajectories/tir_oss_incorrect.jsonl",
        "tir_oss_data.jsonl",
    ]
    results: dict[str, str] = {}
    for rel in _RESULT_PATHS:
        path = exp_dir / rel
        if path.exists():
            results[rel] = path.read_text()

    return results


# ---------------------------------------------------------------------------
# Remote function: OSS data generation (parallelized across containers)
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",
    volumes={
        HF_CACHE:        hf_vol,
        EXPERIMENTS_DIR: experiments_vol,
    },
    timeout=7200,   # 2 hours per batch
)
def oss_generate_batch(
    problems: list[dict],
    model_path: str = "",
    num_samples: int = 5,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_rounds: int = 3,
    max_new_tokens: int = 1024,
    code_timeout: int = 5,
) -> list[dict]:
    """Generate multiple nucleus-sampled TIR trajectories for a batch of problems."""
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from src.data_generation.generator import generate_tir_trajectory
    from src.utils.math_grading import extract_boxed_answer, grade_answer

    os.environ["HF_HOME"] = HF_CACHE
    experiments_vol.reload()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    results = []
    for item in problems:
        for sample_i in range(num_samples):
            trajectory = generate_tir_trajectory(
                model, tokenizer, item["problem"],
                max_rounds=max_rounds,
                max_new_tokens=max_new_tokens,
                code_timeout=code_timeout,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
            predicted = extract_boxed_answer(trajectory)
            is_correct = grade_answer(predicted, item["ground_truth"], item["dataset"])
            results.append({
                **item,
                "trajectory": trajectory,
                "predicted_answer": predicted,
                "correct": is_correct,
                "source": "oss",
                "sample_index": sample_i,
            })
    return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

# Add project root to path so local `src` imports work when running the entrypoint
sys.path.insert(0, str(_root))

# Pass-through args are forwarded to run_experiment.py unchanged.
# --tir_data_path and --output_dir are handled here (not forwarded).

@app.local_entrypoint()
def main(
    dataset:          str   = "gsm8k",
    n:                int   = 500,
    tir_data_path:    str   = "data/tir_gsm8k_1000.jsonl",
    output_dir:       str   = "results/exp",
    seed:             int   = 42,
    # Training
    epochs:           int   = 1,
    batch_size:       int   = 2,
    lr:               float = 2e-5,
    warmup_frac:      float = 0.03,
    grad_accum:       int   = 4,
    grad_clip:        float = 1.0,
    max_length:       int   = 2048,
    log_every:        int   = 10,
    # Evaluation
    eval_batch_size:  int   = 8,
    eval_max_tokens:  int   = 1024,
    eval_limit:       int   = 0,    # 0 = no limit
    # Output Space Shaping
    oss:              bool  = False,
    oss_num_samples:  int   = 5,
    oss_temperature:  float = 0.8,
    oss_top_p:        float = 0.95,
    oss_batch_size:   int   = 50,
    # Control flow
    skip_cot_train:   bool  = False,
    skip_tir_train:   bool  = False,
    skip_oss_gen:     bool  = False,
    skip_oss_train:   bool  = False,
    skip_eval:        bool  = False,
    # W&B (requires the "wandb" Modal secret — see file header)
    wandb_project:    str   = "",
) -> None:
    from src.utils.math_grading import extract_boxed_answer, extract_gsm8k_answer

    # ------------------------------------------------------------------
    # Read TIR data locally and pass to the container
    # ------------------------------------------------------------------
    tir_path = Path(tir_data_path)
    if not tir_path.exists():
        raise FileNotFoundError(f"TIR data file not found: {tir_path}")
    tir_data = tir_path.read_text()

    # ------------------------------------------------------------------
    # Build the experiment name and shared CLI args
    # ------------------------------------------------------------------
    experiment_name = f"{dataset}_n{n}_ep{epochs}_lr{lr:.0e}"

    base_cli_args: list[str] = [
        "--dataset",      dataset,
        "--n",            str(n),
        "--seed",         str(seed),
        "--epochs",       str(epochs),
        "--batch_size",   str(batch_size),
        "--lr",           str(lr),
        "--warmup_frac",  str(warmup_frac),
        "--grad_accum",   str(grad_accum),
        "--grad_clip",    str(grad_clip),
        "--max_length",   str(max_length),
        "--log_every",    str(log_every),
        "--eval_batch_size", str(eval_batch_size),
        "--eval_max_tokens", str(eval_max_tokens),
    ]
    if eval_limit:
        base_cli_args += ["--eval_limit", str(eval_limit)]
    if wandb_project:
        base_cli_args += ["--wandb_project", wandb_project]

    # ------------------------------------------------------------------
    # Non-OSS path: single remote call (original behavior)
    # ------------------------------------------------------------------
    if not oss:
        cli_args = list(base_cli_args)
        if skip_cot_train:
            cli_args.append("--skip_cot_train")
        if skip_tir_train:
            cli_args.append("--skip_tir_train")
        if skip_eval:
            cli_args.append("--skip_eval")

        print(f"Launching experiment '{experiment_name}' on Modal (gpu=A10G)")
        print(f"TIR data: {tir_data_path} ({len(tir_data.splitlines())} lines)")

        result_files = run_experiment_fn.remote(
            experiment_name=experiment_name,
            dataset=dataset,
            tir_data=tir_data,
            cli_args=cli_args,
        )
        _download_results(result_files, output_dir, experiment_name)
        return

    # ------------------------------------------------------------------
    # OSS path: 3 phases
    # ------------------------------------------------------------------

    # --- Phase 1: Train CoT + TIR (skip eval and OSS) ---
    if not (skip_cot_train and skip_tir_train):
        phase1_args = list(base_cli_args) + ["--skip_eval"]
        if skip_cot_train:
            phase1_args.append("--skip_cot_train")
        if skip_tir_train:
            phase1_args.append("--skip_tir_train")

        print(f"Phase 1/3 — Training CoT + TIR on Modal (gpu=A10G)")
        print(f"TIR data: {tir_data_path} ({len(tir_data.splitlines())} lines)")

        run_experiment_fn.remote(
            experiment_name=experiment_name,
            dataset=dataset,
            tir_data=tir_data,
            cli_args=phase1_args,
        )
        print("Phase 1/3 complete — CoT and TIR models trained.")
    else:
        print("Phase 1/3 — Skipping training (using existing checkpoints)")

    # --- Phase 2: Parallel OSS data generation ---
    if not skip_oss_gen:
        print(f"\nPhase 2/3 — Generating OSS data in parallel on Modal")
        print(f"  {n} problems × {oss_num_samples} samples = "
              f"{n * oss_num_samples} trajectory generations")
        print(f"  Batch size: {oss_batch_size} problems → "
              f"{(n + oss_batch_size - 1) // oss_batch_size} parallel containers")
        print(f"  Sampling: temperature={oss_temperature}, top_p={oss_top_p}")

        model_path = f"{EXPERIMENTS_DIR}/{experiment_name}/tir/final"
        code_timeout = 5 if dataset == "gsm8k" else 10

        # Load problems from the dataset (same sampling as training)
        from datasets import load_dataset

        DATASETS = {
            "gsm8k": {"hf_path": "openai/gsm8k", "hf_name": "main",
                       "split": "train", "problem_key": "question"},
            "math":  {"hf_path": "lighteval/MATH", "hf_name": "all",
                       "split": "train", "problem_key": "problem"},
        }
        cfg = DATASETS[dataset]
        ds = load_dataset(cfg["hf_path"], cfg["hf_name"], split=cfg["split"])
        rng = random.Random(seed)
        indices = rng.sample(range(len(ds)), min(n, len(ds)))

        problems: list[dict] = []
        for idx in indices:
            sample = ds[idx]
            gt = (
                extract_gsm8k_answer(sample["answer"])
                if dataset == "gsm8k"
                else extract_boxed_answer(sample["solution"])
            ) or ""
            problems.append({
                "dataset":        dataset,
                "original_index": idx,
                "problem":        sample[cfg["problem_key"]],
                "ground_truth":   gt,
            })

        batches = [
            problems[i : i + oss_batch_size]
            for i in range(0, len(problems), oss_batch_size)
        ]

        all_oss_records: list[dict] = []
        for batch_results in oss_generate_batch.map(
            batches,
            kwargs={
                "model_path":     model_path,
                "num_samples":    oss_num_samples,
                "temperature":    oss_temperature,
                "top_p":          oss_top_p,
                "code_timeout":   code_timeout,
            },
            order_outputs=False,
        ):
            all_oss_records.extend(batch_results)

        total_candidates = len(all_oss_records)
        oss_correct = [r for r in all_oss_records if r["correct"]]
        print(f"\n  OSS generation complete:")
        print(f"    Total candidates: {total_candidates}")
        print(f"    Correct: {len(oss_correct)} "
              f"({len(oss_correct)/total_candidates:.1%})")

        # Merge: original TIR data + new correct OSS trajectories
        base_records = [json.loads(line) for line in tir_data.splitlines() if line.strip()]
        merged_lines = []
        for r in base_records:
            merged_lines.append(json.dumps(r))
        for r in oss_correct:
            merged_lines.append(json.dumps(r))
        merged_data = "\n".join(merged_lines) + "\n"

        base_correct = sum(1 for r in base_records if r.get("correct"))
        print(f"    Base correct: {base_correct}")
        print(f"    Merged total records: {len(base_records) + len(oss_correct)}")
        print(f"    Merged correct (for training): "
              f"{base_correct + len(oss_correct)}")

        # Write merged data to the experiments volume
        print(f"\n  Writing merged OSS data to volume...")
        _write_oss_data_fn.remote(experiment_name, merged_data)
        print("Phase 2/3 complete — OSS data generated and merged.")
    else:
        print("\nPhase 2/3 — Skipping OSS generation (using existing data)")

    # --- Phase 3: Train TIR+OSS + Evaluate all ---
    phase3_args = list(base_cli_args) + [
        "--oss",
        "--skip_cot_train",
        "--skip_tir_train",
        "--skip_oss_gen",
    ]
    if skip_oss_train:
        phase3_args.append("--skip_oss_train")
    if skip_eval:
        phase3_args.append("--skip_eval")

    print(f"\nPhase 3/3 — Training TIR+OSS + evaluating all models on Modal")

    result_files = run_experiment_fn.remote(
        experiment_name=experiment_name,
        dataset=dataset,
        tir_data=tir_data,
        cli_args=phase3_args,
    )
    _download_results(result_files, output_dir, experiment_name)


# ---------------------------------------------------------------------------
# Helper: write OSS data to the experiments volume
# ---------------------------------------------------------------------------

@app.function(
    volumes={EXPERIMENTS_DIR: experiments_vol},
    timeout=300,
)
def _write_oss_data_fn(experiment_name: str, merged_data: str) -> None:
    exp_dir = Path(EXPERIMENTS_DIR) / experiment_name
    oss_path = exp_dir / "tir_oss_data.jsonl"
    oss_path.write_text(merged_data)
    experiments_vol.commit()


# ---------------------------------------------------------------------------
# Helper: download results to local disk
# ---------------------------------------------------------------------------

def _download_results(
    result_files: dict[str, str], output_dir: str, experiment_name: str
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for rel_path, content in result_files.items():
        dest = out / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        print(f"  Downloaded → {dest}")

    # Print the comparison table from the log (last section after final ===)
    log_content = result_files.get("experiment.log", "")
    if log_content:
        lines = log_content.splitlines()
        last_results = max(
            (i for i, l in enumerate(lines) if "RESULTS" in l and "===" in lines[i - 1]),
            default=None,
        )
        if last_results is not None:
            print("\n" + "\n".join(lines[last_results - 1:]))

    print(f"\nCheckpoints remain in Modal Volume 'tir-experiments' under '{experiment_name}/'")
    print(f"Retrieve with: modal volume get tir-experiments {experiment_name}/cot/final <local_path>")
