"""TIR SFT training for Qwen2.5-0.5B on generated trajectory data.

Examples
--------
# Overfit 8 examples to validate the pipeline:
python scripts/train_tir.py --data_path data/tir_gsm8k_1000.jsonl --overfit

# Full training run:
python scripts/train_tir.py --data_path data/tir_gsm8k_1000.jsonl --epochs 1 --output_dir checkpoints/tir_gsm8k
python scripts/train_tir.py --data_path data/tir_math_1000.jsonl  --epochs 1 --output_dir checkpoints/tir_math
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.dataset import PaddedCollator, TIRDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-0.5B"

OVERFIT_N      = 4
OVERFIT_STEPS  = 300
OVERFIT_LR     = 5e-5
OVERFIT_TARGET = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
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
    while global_step < max_steps:
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
    p.add_argument("--data_path",  required=True,
                   help="Path to JSONL file with TIR trajectories (correct=True rows used)")
    p.add_argument("--model",      default=MODEL_ID)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--seed",       type=int,   default=42)

    # Overfit validation mode
    p.add_argument("--overfit",       action="store_true",
                   help="Overfit 8 examples to validate the pipeline")
    p.add_argument("--overfit_steps", type=int, default=OVERFIT_STEPS)

    # Hyperparameters (ignored in --overfit mode)
    p.add_argument("--epochs",      type=int,   default=1)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--warmup_frac", type=float, default=0.03)
    p.add_argument("--grad_accum",  type=int,   default=4)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--max_length",  type=int,   default=2048)
    p.add_argument("--n",             type=int,   default=None,
                   help="Subsample N correct trajectories (default: use all)")
    p.add_argument("--log_every",     type=int,   default=10)
    p.add_argument("--save_every",    type=int,   default=None)
    p.add_argument("--wandb_project", type=str,   default=None,
                   help="W&B project name; omit to disable W&B logging")
    p.add_argument("--wandb_group",    type=str,  default=None)
    p.add_argument("--wandb_run_name", type=str,  default=None)
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
    full_dataset = TIRDataset.from_jsonl(args.data_path, tokenizer, max_length=args.max_length)
    print(f"Loaded {len(full_dataset)} correct trajectories from {args.data_path}")

    if args.n is not None and len(full_dataset) > args.n:
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(full_dataset)), args.n)
        full_dataset = TIRDataset(
            [full_dataset._samples[i] for i in indices], tokenizer, max_length=args.max_length
        )
        print(f"Subsampled to {len(full_dataset)} examples (--n {args.n})")

    if args.overfit:
        print(f"\n{'='*60}")
        print(f"OVERFIT MODE — {OVERFIT_N} examples, {args.overfit_steps} steps")
        print(f"{'='*60}\n")
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(full_dataset)), min(OVERFIT_N, len(full_dataset)))
        overfit_records = [full_dataset._samples[i] for i in indices]
        dataset    = TIRDataset(overfit_records, tokenizer, max_length=args.max_length)
        batch_size = OVERFIT_N
        lr         = OVERFIT_LR
        max_steps  = args.overfit_steps
        log_every  = 10
        save_every = None
        output_dir = None
        grad_accum = 1
    else:
        dataset    = full_dataset
        batch_size = args.batch_size
        lr         = args.lr
        log_every  = args.log_every
        save_every = args.save_every
        output_dir = Path(args.output_dir) if args.output_dir else None
        grad_accum = args.grad_accum
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        import math
        steps_per_epoch = math.ceil(len(dataset) / (batch_size * grad_accum))
        max_steps = steps_per_epoch * args.epochs

    print(f"Samples: {len(dataset)} | Batch: {batch_size} | Steps: {max_steps} | LR: {lr:.2e}")

    collator   = PaddedCollator(tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not args.overfit,
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
