"""SFT datasets for CoT and TIR training.

Each example is formatted as a chat with:
  - system: task instruction
  - user: the problem
  - assistant: the solution / trajectory (target for the loss)

Labels are -100 for the prompt tokens so the loss is only computed over
the assistant response.
"""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset

COT_SYSTEM_PROMPT = "Solve the following math problem step by step."

# Must match TIR_SYSTEM_PROMPT in src/data_generation/generator.py
TIR_SYSTEM_PROMPT = (
    "Please integrate natural language reasoning with programs to solve math problems "
    "using the following guidelines:\n"
    "- Analyze the question and write functions to solve the problem; "
    "the function should not take any arguments.\n"
    "- Present the final result in LaTeX using a '\\boxed{}' without any units.\n"
    "- Utilize the 'pi' symbol and 'Rational' from Sympy for $\\pi$ and fractions, "
    "and simplify all fractions and square roots without converting them to decimal values."
)

# Keys used to extract the problem and solution from each dataset
DATASET_KEYS = {
    "gsm8k": {"problem": "question", "solution": "answer"},
    "math":  {"problem": "problem",  "solution": "solution"},
}


class CoTDataset(Dataset):
    """Single-turn SFT dataset where the target is the CoT solution."""

    def __init__(
        self,
        samples: list[dict],
        tokenizer,
        dataset_name: str,
        max_length: int = 2048,
    ) -> None:
        keys = DATASET_KEYS[dataset_name]
        self._problem_key = keys["problem"]
        self._solution_key = keys["solution"]
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._samples = samples

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self._samples[idx]
        problem = sample[self._problem_key]
        solution = sample[self._solution_key]

        system_msg   = {"role": "system",    "content": COT_SYSTEM_PROMPT}
        user_msg     = {"role": "user",      "content": problem}
        assistant_msg = {"role": "assistant", "content": solution}

        # Full sequence and prompt-only sequence (to compute the split point)
        full_text = self._tokenizer.apply_chat_template(
            [system_msg, user_msg, assistant_msg],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = self._tokenizer.apply_chat_template(
            [system_msg, user_msg],
            tokenize=False,
            add_generation_prompt=True,
        )

        full_ids   = self._tokenizer(full_text,   add_special_tokens=False)["input_ids"]
        prompt_ids = self._tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        # Truncate to max_length before building labels
        full_ids = full_ids[: self._max_length]
        labels   = [-100] * min(prompt_len, len(full_ids)) + full_ids[prompt_len:]

        return {
            "input_ids":      full_ids,
            "labels":         labels,
            "attention_mask": [1] * len(full_ids),
        }


class TIRDataset(Dataset):
    """SFT dataset over JSONL-formatted TIR trajectories.

    Loads records from a JSONL file, keeps only those where correct=True,
    and formats each as system + user(problem) + assistant(trajectory).
    """

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int = 2048,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._samples = [r for r in records if r.get("correct")]

    @classmethod
    def from_jsonl(cls, path: str | Path, tokenizer, max_length: int = 2048) -> "TIRDataset":
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return cls(records, tokenizer, max_length)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        record = self._samples[idx]
        problem    = record["problem"]
        trajectory = record["trajectory"]

        system_msg    = {"role": "system",    "content": TIR_SYSTEM_PROMPT}
        user_msg      = {"role": "user",      "content": problem}
        assistant_msg = {"role": "assistant", "content": trajectory}

        full_text = self._tokenizer.apply_chat_template(
            [system_msg, user_msg, assistant_msg],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = self._tokenizer.apply_chat_template(
            [system_msg, user_msg],
            tokenize=False,
            add_generation_prompt=True,
        )

        full_ids   = self._tokenizer(full_text,   add_special_tokens=False)["input_ids"]
        prompt_ids = self._tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        full_ids = full_ids[: self._max_length]
        labels   = [-100] * min(prompt_len, len(full_ids)) + full_ids[prompt_len:]

        return {
            "input_ids":      full_ids,
            "labels":         labels,
            "attention_mask": [1] * len(full_ids),
        }


# ------------------------------------------------------------------
# Collator
# ------------------------------------------------------------------

import torch


class PaddedCollator:
    """Right-pad sequences in a batch to equal length."""

    def __init__(self, pad_token_id: int) -> None:
        self._pad_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)

        input_ids      = []
        labels         = []
        attention_mask = []

        for b in batch:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"]      + [self._pad_id] * pad)
            labels.append(   b["labels"]         + [-100]         * pad)
            attention_mask.append(b["attention_mask"] + [0]       * pad)

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
