"""Generate TIR training data on Modal Labs.

Modal auto-scales GPU containers based on the number of batches — each batch
runs in its own container (or is queued into an existing one). The model is
loaded once per container lifecycle via @modal.enter().

Usage:
    # Run with default settings (1000 examples, gsm8k):
    modal run scripts/generate_modal.py

    # Specify dataset and size:
    modal run scripts/generate_modal.py --dataset math --n 1000

    # Smaller batches → more parallelism (more containers spun up):
    modal run scripts/generate_modal.py --dataset gsm8k --n 1000 --batch_size 20
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Image + volume
# ---------------------------------------------------------------------------

APP_NAME = "tir-data-generation"
MODEL_ID  = "Qwen/Qwen2.5-Math-7B-Instruct"
HF_CACHE  = "/vol/hf_cache"

# Persists downloaded model weights across runs so we don't re-download 14 GB
hf_vol = modal.Volume.from_name("tir-hf-cache", create_if_missing=True)

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
    )
    # Mount the local src/ package into the container
    .add_local_dir(
        Path(__file__).resolve().parent.parent / "src",
        remote_path="/root/src",
    )
)

app = modal.App(APP_NAME, image=image)


# ---------------------------------------------------------------------------
# GPU worker class
# ---------------------------------------------------------------------------

@app.cls(
    gpu="A10G",
    volumes={HF_CACHE: hf_vol},
    timeout=7200,
)
class TIRWorker:
    @modal.enter()
    def load_model(self) -> None:
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ["HF_HOME"] = HF_CACHE
        print(f"Loading {MODEL_ID} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        hf_vol.commit()  # flush any newly downloaded files to the volume
        print("Model ready.")

    @modal.method()
    def generate_batch(
        self,
        problems: list[dict],
        max_rounds: int = 3,
        max_new_tokens: int = 1024,
        code_timeout: int = 10,
    ) -> list[dict]:
        # src/ is mounted at /root/src; add /root so `from src.x import y` works
        if "/root" not in sys.path:
            sys.path.insert(0, "/root")

        from src.data_generation.generator import generate_tir_trajectory
        from src.utils.math_grading import extract_boxed_answer, grade_answer

        results = []
        for item in problems:
            trajectory = generate_tir_trajectory(
                self.model,
                self.tokenizer,
                item["problem"],
                max_rounds=max_rounds,
                max_new_tokens=max_new_tokens,
                code_timeout=code_timeout,
            )
            predicted  = extract_boxed_answer(trajectory)
            is_correct = grade_answer(predicted, item["ground_truth"], item["dataset"])
            results.append(
                {
                    **item,
                    "trajectory":       trajectory,
                    "predicted_answer": predicted,
                    "correct":          is_correct,
                }
            )
        return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

DATASETS = {
    "gsm8k": {"hf_path": "openai/gsm8k",  "hf_name": "main", "split": "train", "problem_key": "question"},
    "math":  {"hf_path": "lighteval/MATH", "hf_name": "all",  "split": "train", "problem_key": "problem"},
}

# Add project root to path so local `src` imports work when running the entrypoint
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@app.local_entrypoint()
def main(
    dataset:        str = "gsm8k",
    n:              int = 1000,
    seed:           int = 42,
    output_dir:     str = "data",
    batch_size:     int = 50,
    max_rounds:     int = 3,
    max_new_tokens: int = 1024,
    code_timeout:   int = 0,   # 0 = auto: 5s for gsm8k, 10s for math
) -> None:
    from datasets import load_dataset

    from src.utils.math_grading import extract_boxed_answer, extract_gsm8k_answer

    code_timeout = code_timeout or (5 if dataset == "gsm8k" else 10)
    cfg = DATASETS[dataset]

    ds      = load_dataset(cfg["hf_path"], cfg["hf_name"], split=cfg["split"])
    rng     = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))

    problems: list[dict] = []
    for idx in indices:
        sample = ds[idx]
        gt = (
            extract_gsm8k_answer(sample["answer"])
            if dataset == "gsm8k"
            else extract_boxed_answer(sample["solution"])
        ) or ""
        problems.append(
            {
                "dataset":        dataset,
                "original_index": idx,
                "problem":        sample[cfg["problem_key"]],
                "ground_truth":   gt,
            }
        )

    batches = [problems[i : i + batch_size] for i in range(0, len(problems), batch_size)]
    print(f"{len(problems)} problems → {len(batches)} batches "
          f"(batch_size={batch_size}, code_timeout={code_timeout}s)")

    out_path = Path(output_dir) / f"tir_{dataset}_{n}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    worker = TIRWorker()
    total = correct = 0

    with open(out_path, "w") as fout:
        for batch_results in worker.generate_batch.map(
            batches,
            kwargs={
                "max_rounds":     max_rounds,
                "max_new_tokens": max_new_tokens,
                "code_timeout":   code_timeout,
            },
            order_outputs=False,
        ):
            for record in batch_results:
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                total   += 1
                correct += record["correct"]

    print(f"\nSaved {total} trajectories → {out_path}")
    if total:
        print(f"Accuracy on generated data: {correct}/{total} ({correct/total:.1%})")
