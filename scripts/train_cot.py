"""CoT SFT training for Qwen2.5-0.5B on GSM8k or MATH.

Examples
--------
# Overfit 8 examples to validate the pipeline:
python scripts/train_cot.py --dataset gsm8k --overfit

# Full training run:
python scripts/train_cot.py --dataset gsm8k --epochs 3 --output_dir checkpoints/cot_gsm8k
python scripts/train_cot.py --dataset math  --epochs 3 --output_dir checkpoints/cot_math
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.dataset import CoTDataset, PaddedCollator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-0.5B"

DATASET_CONFIGS = {
    "gsm8k": {"hf_path": "openai/gsm8k",  "hf_name": "main", "split": "train"},
    "math":  {"hf_path": "lighteval/MATH", "hf_name": "all",  "split": "train"},
}

OVERFIT_N      = 8      # examples used for overfit validation
OVERFIT_STEPS  = 300    # steps to train during overfit
OVERFIT_LR     = 5e-4   # higher LR to converge quickly
OVERFIT_TARGET = 0.05   # loss threshold for "PASSED"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_samples(dataset: str, seed: int, n: int | None = None) -> list[dict]:
    cfg = DATASET_CONFIGS[dataset]
    ds = load_dataset(cfg["hf_path"], cfg["hf_name"], split=cfg["split"])
    samples = list(ds)
    if n is not None:
        rng = random.Random(seed)
        samples = rng.sample(samples, min(n, len(samples)))
    return samples


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of non-masked output tokens predicted correctly."""
    # HuggingFace already shifts inside the model, so logits and labels
    # are both length-n; we replicate the shift here for the accuracy metric.
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    if not mask.any():
        return float("nan")
    preds = shift_logits.argmax(-1)
    return (preds[mask] == shift_labels[mask]).float().mean().item()


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(
    model,
    dataloader: DataLoader,
    *,
    max_steps: int,
    lr: float,
    warmup_frac: float,
    grad_accum: int,
    grad_clip: float,
    log_every: int,
    save_every: int | None,
    output_dir: Path | None,
    device: torch.device,
    overfit: bool,
    use_wandb: bool = False,
) -> list[tuple[int, float, float]]:
    """Run the training loop; returns [(step, loss, acc), ...]."""

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup_steps = max(1, int(warmup_frac * max_steps))
    scheduler = build_scheduler(optimizer, warmup_steps, max_steps)

    log: list[tuple[int, float, float]] = []
    global_step = 0
    optimizer.zero_grad()

    model.train()
    epoch = 0
    while global_step < max_steps:
        epoch += 1
        for batch in dataloader:
            if global_step >= max_steps:
                break

            input_ids      = batch["input_ids"].to(device)
            labels         = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / grad_accum

            loss.backward()

            # Only step optimizer after accumulating grad_accum micro-batches
            if (global_step + 1) % grad_accum == 0 or overfit:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            raw_loss = (loss * grad_accum).item()
            acc = token_accuracy(outputs.logits.detach(), labels)

            if global_step % log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"step {global_step:>5d} | loss {raw_loss:.4f} | "
                    f"acc {acc:.4f} | lr {lr_now:.2e}"
                )
                if use_wandb:
                    wandb.log({"train/loss": raw_loss, "train/acc": acc, "train/lr": lr_now}, step=global_step)

            log.append((global_step, raw_loss, acc))
            global_step += 1

            if save_every and output_dir and global_step % save_every == 0:
                ckpt = output_dir / f"step_{global_step}"
                model.save_pretrained(ckpt)
                print(f"  → saved checkpoint: {ckpt}")

    return log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    choices=["gsm8k", "math"], required=True)
    p.add_argument("--model",      default=MODEL_ID)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--seed",       type=int,   default=42)

    # Overfit validation mode
    p.add_argument("--overfit",       action="store_true",
                   help="Overfit 8 examples to validate the pipeline")
    p.add_argument("--overfit_steps", type=int, default=OVERFIT_STEPS)

    # Hyperparameters (ignored in --overfit mode)
    p.add_argument("--epochs",      type=int,   default=1)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--warmup_frac", type=float, default=0.03)
    p.add_argument("--grad_accum",  type=int,   default=1)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--max_length",  type=int,   default=2048)
    p.add_argument("--n",              type=int,   default=None,
                   help="Subsample N training examples (default: use all)")
    p.add_argument("--log_every",      type=int,   default=10)
    p.add_argument("--save_every",     type=int,   default=None)
    p.add_argument("--wandb_project",  type=str,   default=None,
                   help="W&B project name; omit to disable W&B logging")
    p.add_argument("--wandb_group",    type=str,   default=None)
    p.add_argument("--wandb_run_name", type=str,   default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            group=args.wandb_group,
            name=args.wandb_run_name,
        )

    # ------------------------------------------------------------------
    # Model & tokenizer
    # ------------------------------------------------------------------
    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    if args.overfit:
        print(f"\n{'='*60}")
        print(f"OVERFIT MODE — {OVERFIT_N} examples, {args.overfit_steps} steps")
        print(f"{'='*60}\n")
        samples = load_samples(args.dataset, args.seed, n=OVERFIT_N)
        batch_size  = OVERFIT_N   # fit all examples in one batch
        lr          = OVERFIT_LR
        max_steps   = args.overfit_steps
        log_every   = 10
        save_every  = None
        output_dir  = None
        grad_accum  = 1
    else:
        samples    = load_samples(args.dataset, args.seed, n=args.n)
        batch_size = args.batch_size
        lr         = args.lr
        log_every  = args.log_every
        save_every = args.save_every
        output_dir = Path(args.output_dir) if args.output_dir else None
        grad_accum = args.grad_accum
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        steps_per_epoch = math.ceil(len(samples) / (batch_size * grad_accum))
        max_steps = steps_per_epoch * args.epochs

    print(f"Dataset: {args.dataset} | Samples: {len(samples)} | "
          f"Batch: {batch_size} | Steps: {max_steps} | LR: {lr:.2e}")

    dataset = CoTDataset(samples, tokenizer, args.dataset, max_length=args.max_length)
    collator = PaddedCollator(tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not args.overfit,  # keep order fixed when overfitting
        collate_fn=collator,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    log = run_training(
        model, dataloader,
        max_steps=max_steps,
        lr=lr,
        warmup_frac=args.warmup_frac if not args.overfit else 0.0,
        grad_accum=grad_accum,
        grad_clip=args.grad_clip,
        log_every=log_every,
        save_every=save_every,
        output_dir=output_dir,
        device=device,
        overfit=args.overfit,
        use_wandb=use_wandb,
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    final_loss = log[-1][1]
    final_acc  = log[-1][2]
    print(f"\nFinal step: loss={final_loss:.4f}, acc={final_acc:.4f}")

    if use_wandb:
        wandb.finish()

    if args.overfit:
        if final_loss < OVERFIT_TARGET:
            print(f"\n✓ PASSED — loss {final_loss:.4f} < {OVERFIT_TARGET} target")
        else:
            print(f"\n✗ FAILED — loss {final_loss:.4f} did not reach {OVERFIT_TARGET} target")
            sys.exit(1)
    else:
        if output_dir:
            model.save_pretrained(output_dir / "final")
            tokenizer.save_pretrained(output_dir / "final")
            print(f"Saved final model → {output_dir / 'final'}")


if __name__ == "__main__":
    main()
