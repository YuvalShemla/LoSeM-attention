"""
Algorithm registry.

Idealized methods are distinguished by kind='idealized'
and are auto-included in every evaluation. Algorithms
must be explicitly requested.
"""

from dataclasses import dataclass

from .idealized_methods import (
    IdealTopK,
    IdealSampling,
    IdealEqualSplits,
    IdealEqualWeightSplits,
)
from .multiq_grouping import MultiQGrouping
from .kmeans_clustering import KMeansClustering
from .lsh_crosspoly_multiprobe import LSHCrossPolytope
from .lsh_crosspoly_clustered import LSHCrossPolytopeClustered
from .lsh_simhash_snis import LSHSimHashSNIS
from .lsh_crosspoly_snis import LSHCrossPolySNIS
from .kmeans_value_clustering import (
    KMeansValueClustering,
    KMeansKeyClustering,
)
from .value_cluster_is import ValueClusterIS


@dataclass
class MethodSpec:
    cls: type
    kind: str   # "idealized" or "algorithm"


METHOD_REGISTRY = {
    "ideal_topk": MethodSpec(
        IdealTopK, "idealized",
    ),
    "ideal_sampling": MethodSpec(
        IdealSampling, "idealized",
    ),
    "ideal_equal_splits": MethodSpec(
        IdealEqualSplits, "idealized",
    ),
    "ideal_equal_weight_splits": MethodSpec(
        IdealEqualWeightSplits, "idealized",
    ),
    "multiq": MethodSpec(
        MultiQGrouping, "algorithm",
    ),
    "kmeans": MethodSpec(
        KMeansClustering, "algorithm",
    ),
    "lsh_crosspoly": MethodSpec(
        LSHCrossPolytope, "algorithm",
    ),
    "lsh_crosspoly_clustered": MethodSpec(
        LSHCrossPolytopeClustered, "algorithm",
    ),
    "lsh_simhash_snis": MethodSpec(
        LSHSimHashSNIS, "algorithm",
    ),
    "lsh_crosspoly_snis": MethodSpec(
        LSHCrossPolySNIS, "algorithm",
    ),
    "kmeans_value": MethodSpec(
        KMeansValueClustering, "algorithm",
    ),
    "kmeans_key": MethodSpec(
        KMeansKeyClustering, "algorithm",
    ),
    "value_cluster_is": MethodSpec(
        ValueClusterIS, "algorithm",
    ),
}
