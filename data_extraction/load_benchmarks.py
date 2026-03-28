"""
Load benchmark examples for vector extraction.

Supports InfiniteBench (4 tasks) and LongBench v2 (2 tasks).
Tokenizes and truncates to config max_length, preserving
the question/answer portion at the end.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

TASK_CONFIG = {
    # InfiniteBench tasks
    "math_calc": {
        "source": "infinitebench",
        "hf_name": "xinrongzhang2022/InfiniteBench",
        "task_field": "task",
        "task_value": "math_calc",
    },
    "code_run": {
        "source": "infinitebench",
        "hf_name": "xinrongzhang2022/InfiniteBench",
        "task_field": "task",
        "task_value": "code_run",
    },
    "longbook_sum_eng": {
        "source": "infinitebench",
        "hf_name": "xinrongzhang2022/InfiniteBench",
        "task_field": "task",
        "task_value": "longbook_sum_eng",
    },
    "passkey": {
        "source": "infinitebench",
        "hf_name": "xinrongzhang2022/InfiniteBench",
        "task_field": "task",
        "task_value": "passkey",
    },
    # LongBench v2 tasks
    "multi_doc_qa": {
        "source": "longbench_v2",
        "hf_name": "THUDM/LongBench-v2",
        "domain_filter": "Multi-Document QA",
    },
    "single_doc_qa": {
        "source": "longbench_v2",
        "hf_name": "THUDM/LongBench-v2",
        "domain_filter": "Single-Document QA",
    },
}

PROMPT_TEMPLATE = (
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def _load_infinitebench(task_value: str) -> List[Dict]:
    """Load one InfiniteBench task from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset(
        "xinrongzhang2022/InfiniteBench",
        split="train",
    )
    examples = []
    for row in ds:
        if row.get("task") != task_value:
            continue
        examples.append({
            "id": f"{task_value}_{len(examples)}",
            "task": task_value,
            "context": row.get("context", ""),
            "question": row.get("input", ""),
            "answer": row.get("answer", ""),
            "source": "infinitebench",
        })
    return examples


def _load_longbench_v2(domain_filter: str) -> List[Dict]:
    """Load one LongBench v2 domain from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset(
        "THUDM/LongBench-v2", split="train",
    )
    task_name = domain_filter.lower().replace(
        " ", "_"
    ).replace("-", "_")
    examples = []
    for row in ds:
        if row.get("domain") != domain_filter:
            continue
        examples.append({
            "id": f"{task_name}_{len(examples)}",
            "task": task_name,
            "context": row.get("context", ""),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "source": "longbench_v2",
            "domain": row.get("domain", ""),
            "sub_domain": row.get(
                "sub_domain", ""
            ),
        })
    return examples


def load_task(task_name: str) -> List[Dict]:
    """
    Load all examples for a task.

    Returns list of dicts with: id, task, context,
    question, answer, source.
    """
    if task_name not in TASK_CONFIG:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available: {list(TASK_CONFIG.keys())}"
        )
    cfg = TASK_CONFIG[task_name]
    if cfg["source"] == "infinitebench":
        return _load_infinitebench(
            cfg["task_value"]
        )
    else:
        return _load_longbench_v2(
            cfg["domain_filter"]
        )


def format_prompt(example: Dict) -> str:
    """Format example into the extraction prompt."""
    return PROMPT_TEMPLATE.format(
        context=example["context"],
        question=example["question"],
    )


def tokenize_and_truncate(
    tokenizer,
    prompt: str,
    max_length: int,
) -> List[int]:
    """
    Tokenize prompt, truncate to max_length.

    Keeps the END of the prompt (question + answer
    portion) when truncation is needed, since the
    question tokens are what we evaluate on.
    """
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_length:
        tokens = tokens[-max_length:]
    return tokens


def save_benchmark_examples(
    examples: List[Dict],
    task_name: str,
    out_dir: Path,
) -> None:
    """Save benchmark examples JSON for provenance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task_name}.json"
    # Strip large context field for storage
    saved = []
    for ex in examples:
        entry = {k: v for k, v in ex.items()}
        entry["context_chars"] = len(
            entry.get("context", "")
        )
        entry["context_preview"] = entry.get(
            "context", ""
        )[:200]
        del entry["context"]
        saved.append(entry)
    with open(path, "w") as f:
        json.dump(saved, f, indent=2)
