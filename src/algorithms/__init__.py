"""
Algorithm registry.

Idealized methods are distinguished by kind='idealized'
and are auto-included in every evaluation. Algorithms
must be explicitly requested.
"""

from dataclasses import dataclass
import warnings

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
from .cp_value_cluster import CPValueCluster
from .kcluster_topk import KClusterTopK, OracleClusterPQTopK
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
    FullAttentionPQ,
)
from .wildcat2 import WildCat2
from .wildcat3 import WildCat3
from .fcfw_l2 import FCFrankWolfeL2
from .tensor_fcfw_l2 import TensorFCFWL2
from .tensor_fcfw_lq import TensorFCFWLq
from .learned import LearnedCoreset
from .kvsculpt import KVSculpt
_OPTIONAL_MISSING = []

try:
    from .value_cluster_is import ValueClusterIS
except ImportError:
    ValueClusterIS = None
    _OPTIONAL_MISSING.append("value_cluster_is")
try:
    from .tree_attention import TreeAttention
except ImportError:
    TreeAttention = None
    _OPTIONAL_MISSING.append("tree_attention")
try:
    from .value_cluster_methods import (
        VClusterMeanKey,
        VClusterSampled,
        VClusterTopK,
        KClusterSampled,
        VClusterLastKey,
        VClusterMeanLogit,
        KMeansKK,
    )
except ImportError:
    VClusterMeanKey = None
    VClusterSampled = None
    VClusterTopK = None
    KClusterSampled = None
    VClusterLastKey = None
    VClusterMeanLogit = None
    KMeansKK = None
    _OPTIONAL_MISSING.extend([
        "vcluster_meankey",
        "vcluster_sampled",
        "vcluster_topk",
        "kcluster_sampled",
        "vcluster_lastkey",
        "vcluster_meanlogit",
        "kmeans_kk",
    ])


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
    "cp_value_cluster": MethodSpec(
        CPValueCluster, "algorithm",
    ),
    "vattention_pq": MethodSpec(
        VAttentionPQ, "algorithm",
    ),
    "ivfpq_cluster": MethodSpec(
        IVFPQCluster, "algorithm",
    ),
    "fullattention_pq": MethodSpec(
        FullAttentionPQ, "algorithm",
    ),
    "wildcat2": MethodSpec(
        WildCat2, "algorithm",
    ),
    "wildcat3": MethodSpec(
        WildCat3, "algorithm",
    ),
    "fc_frank_wolfe_l2": MethodSpec(
        FCFrankWolfeL2, "algorithm",
    ),
    "tensor_fcfw_l2": MethodSpec(
        TensorFCFWL2, "algorithm",
    ),
    "tensor_fcfw_lq": MethodSpec(
        TensorFCFWLq, "algorithm",
    ),
    "learned": MethodSpec(
        LearnedCoreset, "algorithm",
    ),
    "kvsculpt": MethodSpec(
        KVSculpt, "algorithm",
    ),
    "kcluster_topk": MethodSpec(
        KClusterTopK, "algorithm",
    ),
    "oracle_cluster_pq_topk": MethodSpec(
        OracleClusterPQTopK, "algorithm",
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

if ValueClusterIS is not None:
    METHOD_REGISTRY["value_cluster_is"] = MethodSpec(
        ValueClusterIS, "algorithm",
    )
if VClusterMeanKey is not None:
    METHOD_REGISTRY["vcluster_meankey"] = MethodSpec(
        VClusterMeanKey, "algorithm",
    )
if VClusterSampled is not None:
    METHOD_REGISTRY["vcluster_sampled"] = MethodSpec(
        VClusterSampled, "algorithm",
    )
if VClusterTopK is not None:
    METHOD_REGISTRY["vcluster_topk"] = MethodSpec(
        VClusterTopK, "algorithm",
    )
if KClusterSampled is not None:
    METHOD_REGISTRY["kcluster_sampled"] = MethodSpec(
        KClusterSampled, "algorithm",
    )
if VClusterLastKey is not None:
    METHOD_REGISTRY["vcluster_lastkey"] = MethodSpec(
        VClusterLastKey, "algorithm",
    )
if VClusterMeanLogit is not None:
    METHOD_REGISTRY["vcluster_meanlogit"] = MethodSpec(
        VClusterMeanLogit, "algorithm",
    )
if KMeansKK is not None:
    METHOD_REGISTRY["kmeans_kk"] = MethodSpec(
        KMeansKK, "algorithm",
    )
if TreeAttention is not None:
    METHOD_REGISTRY["tree_attention"] = MethodSpec(
        TreeAttention, "algorithm",
    )

if _OPTIONAL_MISSING:
    missing = ", ".join(sorted(_OPTIONAL_MISSING))
    warnings.warn(
        "Optional algorithm modules unavailable; "
        f"skipping registry entries: {missing}",
        RuntimeWarning,
        stacklevel=1,
    )
