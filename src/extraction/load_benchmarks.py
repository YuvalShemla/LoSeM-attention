"""
Load benchmark examples for vector extraction.

Supports InfiniteBench and LongBench v2 tasks.
Task definitions come from extraction_config.yaml's
task_sources section — no hardcoded registry.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

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
        data_files=f"{task_value}.jsonl",
        split="train",
    )
    examples = []
    for row in ds:
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


def load_task(
    task_name: str,
    task_source: Dict,
) -> List[Dict]:
    """
    Load all examples for a task.

    task_source: dict from config's task_sources,
    with keys: benchmark, hf_name, filter_field,
    filter_value.

    Returns list of dicts with: id, task, context,
    question, answer, source.
    """
    benchmark = task_source["benchmark"]
    if benchmark == "infinitebench":
        return _load_infinitebench(
            task_source["filter_value"]
        )
    elif benchmark == "longbench_v2":
        return _load_longbench_v2(
            task_source["filter_value"]
        )
    else:
        raise ValueError(
            f"Unknown benchmark '{benchmark}' for "
            f"task '{task_name}'. Supported: "
            f"infinitebench, longbench_v2"
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
