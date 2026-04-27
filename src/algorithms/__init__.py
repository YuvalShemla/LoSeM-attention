"""
Algorithm registry.

Idealized methods are distinguished by kind='idealized'
and are auto-included in every evaluation. Algorithms
must be explicitly requested.
"""

from dataclasses import dataclass

from .idealized_methods import (
    IdealTopK,
    IdealSamplingSubset,
    IdealSamplingIS,
    VAttentionOracle,
    IdealEqualSplits,
    IdealEqualWeightSplits,
)
from .multiq_grouping import MultiQGrouping
from .kmeans_clustering import KMeansClustering
from .lsh_crosspoly_multiprobe import LSHCrossPolytope
from .lsh_cp_group import LSHCPGroup
from .lsh_crosspoly_clustered import LSHCrossPolytopeClustered
from .lsh_simhash_snis import LSHSimHashSNIS
from .lsh_crosspoly_snis import LSHCrossPolySNIS
from .kmeans_value_clustering import (
    KMeansValueClustering,
    KMeansKeyClustering,
)
from .value_cluster_is import ValueClusterIS
from .cp_value_cluster import CPValueCluster
from .kcluster_topk import KClusterTopK, OracleClusterPQTopK
from .tree_attention import TreeAttention
from .qclust_topk import QClustTopK
from .qclust_augmented import QClustAugTopK
from .twostage_cluster import KeyClustValSub, QClustValSub
from .topk_cluster_comparison import (
    TopKKeyClusters,
    TopKOracleClusters,
    TopKValueClusters,
)
from .pq_methods import (
    VAttentionPQ,
    IVFPQCluster,
)
from .value_cluster_methods import (
    VClusterMeanKey,
    VClusterSampled,
    VClusterTopK,
    KClusterSampled,
    VClusterLastKey,
    VClusterMeanLogit,
    KMeansKK,
)


@dataclass
class MethodSpec:
    cls: type
    kind: str   # "idealized" or "algorithm"


METHOD_REGISTRY = {
    "ideal_topk": MethodSpec(
        IdealTopK, "idealized",
    ),
    "ideal_sampling_subset": MethodSpec(
        IdealSamplingSubset, "idealized",
    ),
    "ideal_sampling_is": MethodSpec(
        IdealSamplingIS, "idealized",
    ),
    "vattention_oracle": MethodSpec(
        VAttentionOracle, "idealized",
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
    "lsh_cp_group": MethodSpec(
        LSHCPGroup, "algorithm",
    ),
    "lsh_crosspoly_multiprobe": MethodSpec(
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
    "cp_value_cluster": MethodSpec(
        CPValueCluster, "algorithm",
    ),
    "vcluster_meankey": MethodSpec(
        VClusterMeanKey, "algorithm",
    ),
    "vcluster_sampled": MethodSpec(
        VClusterSampled, "algorithm",
    ),
    "vcluster_topk": MethodSpec(
        VClusterTopK, "algorithm",
    ),
    "kcluster_sampled": MethodSpec(
        KClusterSampled, "algorithm",
    ),
    "vcluster_lastkey": MethodSpec(
        VClusterLastKey, "algorithm",
    ),
    "vcluster_meanlogit": MethodSpec(
        VClusterMeanLogit, "algorithm",
    ),
    "vattention_pq": MethodSpec(
        VAttentionPQ, "algorithm",
    ),
    "ivfpq_cluster": MethodSpec(
        IVFPQCluster, "algorithm",
    ),
    "kmeans_kk": MethodSpec(
        KMeansKK, "algorithm",
    ),
    "kcluster_topk": MethodSpec(
        KClusterTopK, "algorithm",
    ),
    "oracle_cluster_pq_topk": MethodSpec(
        OracleClusterPQTopK, "algorithm",
    ),
    "tree_attention": MethodSpec(
        TreeAttention, "algorithm",
    ),
    "qclust_topk": MethodSpec(
        QClustTopK, "algorithm",
    ),
    "qclust_aug_topk": MethodSpec(
        QClustAugTopK, "algorithm",
    ),
    "keyclust_valsub": MethodSpec(
        KeyClustValSub, "algorithm",
    ),
    "qclust_valsub": MethodSpec(
        QClustValSub, "algorithm",
    ),
    "topk_key_clusters": MethodSpec(
        TopKKeyClusters, "algorithm",
    ),
    "topk_oracle_clusters": MethodSpec(
        TopKOracleClusters, "algorithm",
    ),
    "topk_value_clusters": MethodSpec(
        TopKValueClusters, "algorithm",
    ),
}
