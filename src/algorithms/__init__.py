"""
Algorithm registry.

Baselines are distinguished by kind='baseline' and are
auto-included in every experiment. Algorithms must be
explicitly requested.
"""

from dataclasses import dataclass

from .baselines import (
    OracleTopK,
    OracleSampling,
    OracleGrouping,
)
from .meanq_grouping import MeanQGrouping
from .multiq_grouping import MultiQGrouping
from .kmeans_clustering import KMeansClustering


@dataclass
class MethodSpec:
    cls: type
    kind: str   # "baseline" or "algorithm"


METHOD_REGISTRY = {
    "oracle_topk": MethodSpec(
        OracleTopK, "baseline",
    ),
    "oracle_sampling": MethodSpec(
        OracleSampling, "baseline",
    ),
    "oracle_grouping": MethodSpec(
        OracleGrouping, "baseline",
    ),
    "meanq": MethodSpec(MeanQGrouping, "algorithm"),
    "multiq": MethodSpec(
        MultiQGrouping, "algorithm",
    ),
    "kmeans": MethodSpec(
        KMeansClustering, "algorithm",
    ),
}
