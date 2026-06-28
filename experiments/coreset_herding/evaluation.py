"""
End-to-end attention error measurement.

Measures actual attention output error (Eqn 1 in the plan),
not just the feature-space residual norm. This validates
that the herding residual is a meaningful proxy for
downstream attention quality.
"""

import numpy as np
from typing import Dict, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core import full_attention, relative_l2_error


def measure_attention_errors(
    K: np.ndarray,
    V: np.ndarray,
    Q: np.ndarray,
    selected_indices: List[int],
    coreset_sizes: List[int],
    rng: np.random.Generator,
    d: int = 128,
) -> Dict[str, Dict[int, float]]:
    """Measure attention error at multiple coreset sizes.

    Compares three methods:
      - Subset Herding: first S atoms from the herding sequence
      - Oracle TopK: S keys with highest logits (per query)
      - Uniform: S random keys

    Args:
        K: [n, d] keys.
        V: [n, d] values.
        Q: [n_q, d] test queries.
        selected_indices: ordered list of key indices chosen
            by subset herding.
        coreset_sizes: list of sizes to evaluate.
        rng: random generator.
        d: head dimension for attention scaling.

    Returns:
        Dict mapping method name to {size: mean_error}.
    """
    n = K.shape[0]
    n_q = min(len(Q), 20)
    queries = Q[rng.choice(len(Q), n_q, replace=False)]

    errors = {
        "Subset Herding": {s: [] for s in coreset_sizes},
        "Oracle TopK": {s: [] for s in coreset_sizes},
        "Uniform": {s: [] for s in coreset_sizes},
    }

    for q in queries:
        full_out, logits, _ = full_attention(q, K, V, d)

        for s in coreset_sizes:
            # Subset herding
            idx = np.array(
                list(set(selected_indices[:s]))
            )
            if len(idx) > 0:
                out, _, _ = full_attention(
                    q, K[idx], V[idx], d
                )
                errors["Subset Herding"][s].append(
                    relative_l2_error(out, full_out)
                )

            # Oracle TopK
            topk = np.argsort(logits)[-s:]
            out, _, _ = full_attention(
                q, K[topk], V[topk], d
            )
            errors["Oracle TopK"][s].append(
                relative_l2_error(out, full_out)
            )

            # Uniform
            uid = rng.choice(n, size=s, replace=False)
            out, _, _ = full_attention(
                q, K[uid], V[uid], d
            )
            errors["Uniform"][s].append(
                relative_l2_error(out, full_out)
            )

    return {
        method: {
            s: float(np.mean(vals))
            for s, vals in size_dict.items()
            if len(vals) > 0
        }
        for method, size_dict in errors.items()
    }
