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
from .lsh_crosspoly import LSHCrossPolytope
from .lsh_crosspoly_clustered import LSHCrossPolytopeClustered
from .lsh_crosspoly_hybrid import LSHCrossPolytopeHybrid
from .ideal_splits_hybrid import IdealSplitsHybrid
from .kmeans_residual import KMeansResidualClustering
from .kmeans_residual_sparse import KMeansResidualSparse
from .kmeans_multiview import KMeansMultiViewClustering
from .kmeans_nearest_center import KMeansNearestCenter
from .kmeans_medoid_select import KMeansMedoidSelect
from .kmeans_query_weighted import KMeansQueryWeighted
from .kmeans_hybrid_topk import KMeansHybridTopK
from .uniform_sampling import UniformSampling
from .kmeans_ablation import KMeansAblation
from .lsh_simhash_snis import LSHSimHashSNIS
from .kmeans_kk import KMeansKK
from .exactz_sampled_keys import ExactZSampledKeys
from .hierarchical_kmeans import HierarchicalKMeans
from .ideal_ews_ablation import IdealEWSAblation


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
    "lsh_crosspoly_hybrid": MethodSpec(
        LSHCrossPolytopeHybrid, "algorithm",
    ),
    "ideal_splits_hybrid": MethodSpec(
        IdealSplitsHybrid, "algorithm",
    ),
    "kmeans_residual": MethodSpec(
        KMeansResidualClustering, "algorithm",
    ),
    "kmeans_residual_sparse": MethodSpec(
        KMeansResidualSparse, "algorithm",
    ),
    "kmeans_multiview": MethodSpec(
        KMeansMultiViewClustering, "algorithm",
    ),
    "kmeans_nearest_center": MethodSpec(
        KMeansNearestCenter, "algorithm",
    ),
    "kmeans_medoid_select": MethodSpec(
        KMeansMedoidSelect, "algorithm",
    ),
    "kmeans_query_weighted": MethodSpec(
        KMeansQueryWeighted, "algorithm",
    ),
    "kmeans_hybrid_topk": MethodSpec(
        KMeansHybridTopK, "algorithm",
    ),
    "uniform_sampling": MethodSpec(
        UniformSampling, "idealized",
    ),
    "kmeans_ablation": MethodSpec(
        KMeansAblation, "algorithm",
    ),
    "lsh_simhash_snis": MethodSpec(
        LSHSimHashSNIS, "algorithm",
    ),
    "kmeans_kk": MethodSpec(
        KMeansKK, "algorithm",
    ),
    "exactz_sampled_keys": MethodSpec(
        ExactZSampledKeys, "algorithm",
    ),
    "hierarchical_kmeans": MethodSpec(
        HierarchicalKMeans, "algorithm",
    ),
    "ews_ablation": MethodSpec(
        IdealEWSAblation, "idealized",
    ),
}
