"""WildCat2 kernels and AttentionAlgorithm (microsoft/wildcat port)."""

from .algorithm import WildCat2
from .compress_kv import compress_kv
from .weighted_attention import weighted_attention

__all__ = ["WildCat2", "compress_kv", "weighted_attention"]
