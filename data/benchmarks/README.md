# Benchmark Examples

Stored copies of the benchmark examples used in
extraction, so experiments are self-contained and
reproducible without re-downloading from HuggingFace.

## Structure

```
benchmarks/
├── infinitebench/
│   ├── math_calc.json
│   ├── code_run.json
│   ├── longbook_sum_eng.json
│   └── passkey.json
└── longbench_v2/
    ├── multi_doc_qa.json
    └── single_doc_qa.json
```

## Format

Each JSON file contains an array of examples:
```json
[
  {
    "id": "math_calc_0",
    "task": "math_calc",
    "question": "What is the intermediate sum...",
    "answer": "42",
    "context_chars": 69171,
    "context_preview": "First 200 chars...",
    "source": "infinitebench"
  }
]
```

The full context is stored only during extraction
(it's too large for the JSON). The `context_chars`
field records the original length.

## Prompt Template

All examples use:
```
Context: {context}

Question: {question}

Answer:
```

## Tasks

| # | Task | Source | Typical Tokens |
|---|------|--------|---------------:|
| 1 | math_calc | InfiniteBench | 19K |
| 2 | code_run | InfiniteBench | 75K |
| 3 | longbook_sum_eng | InfiniteBench | 120K |
| 4 | passkey | InfiniteBench | 127K |
| 5 | multi_doc_qa | LongBench v2 | 61K |
| 6 | single_doc_qa | LongBench v2 | 85K |
