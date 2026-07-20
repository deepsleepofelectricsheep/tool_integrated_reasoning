"""TIR trajectory generation using Qwen2.5-Math-7B-Instruct.

Format follows TORA Appendix E: rationale interleaved with ```python / ```output blocks.
Qwen2.5-Math-Instruct natively uses this same format in TIR mode.
"""

import re

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, StoppingCriteria, StoppingCriteriaList

from src.data_generation.executor import execute_code

# Matches TORA Appendix E and Qwen2.5-Math TIR system prompt
TIR_SYSTEM_PROMPT = (
    "Please integrate natural language reasoning with programs to solve math problems "
    "using the following guidelines:\n"
    "- Analyze the question and write functions to solve the problem; "
    "the function should not take any arguments.\n"
    "- Present the final result in LaTeX using a '\\boxed{}' without any units.\n"
    "- Utilize the 'pi' symbol and 'Rational' from Sympy for $\\pi$ and fractions, "
    "and simplify all fractions and square roots without converting them to decimal values."
)

_CODE_OUTPUT_MARKER = "```output"


class _StopAtOutputBlock(StoppingCriteria):
    """Halt generation the moment the model begins writing a ```output block."""

    def __init__(self, tokenizer: PreTrainedTokenizer, input_length: int) -> None:
        self._tokenizer = tokenizer
        self._input_length = input_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **_) -> bool:
        new_ids = input_ids[0][self._input_length :]
        if len(new_ids) < 3:
            return False
        # Decode only a trailing window — enough to catch the multi-token marker
        window = new_ids[-40:]
        decoded = self._tokenizer.decode(window, skip_special_tokens=False)
        return _CODE_OUTPUT_MARKER in decoded


def _extract_last_python_block(text: str) -> str | None:
    matches = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def generate_tir_trajectory(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    problem: str,
    max_rounds: int = 3,
    max_new_tokens: int = 1024,
    code_timeout: int = 30,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    """Generate an interactive TIR trajectory for a single math problem.

    Each round:
      1. Generate until \\boxed{} (done) or ```output (need real execution).
      2. If code execution is needed, run the last python block and inject the result.
      3. Repeat up to max_rounds.

    Returns the full trajectory string (rationale + code blocks + outputs + answer).
    """
    messages = [
        {"role": "system", "content": TIR_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    base_prompt: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    trajectory = ""

    for _ in range(max_rounds):
        full_context = base_prompt + trajectory
        input_ids = tokenizer(full_context, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        input_len = input_ids.shape[1]

        sampling_kwargs = {}
        if do_sample:
            sampling_kwargs = dict(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            sampling_kwargs = dict(do_sample=False)

        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                stopping_criteria=StoppingCriteriaList(
                    [_StopAtOutputBlock(tokenizer, input_len)]
                ),
                pad_token_id=tokenizer.eos_token_id,
                **sampling_kwargs,
            )

        new_text: str = tokenizer.decode(
            out_ids[0][input_len:], skip_special_tokens=True
        )

        # Case 1: model produced a final answer — we're done
        if "\\boxed{" in new_text:
            trajectory += new_text
            break

        # Case 2: model started an output block — run the code, inject the result
        if _CODE_OUTPUT_MARKER in new_text:
            # Keep everything up to and including the marker
            cut = new_text.rfind(_CODE_OUTPUT_MARKER)
            trajectory += new_text[: cut + len(_CODE_OUTPUT_MARKER)]

            code = _extract_last_python_block(trajectory)
            result = execute_code(code, timeout=code_timeout) if code else "Error: no python block found"
            trajectory += f"\n{result}\n```\n"

        else:
            # Model ran out of tokens without finishing — save what we have and stop
            trajectory += new_text
            break

    return trajectory
