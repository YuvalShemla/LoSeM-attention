"""
Algorithm registry — graceful imports for partial installations.

All imports are wrapped in try/except so the package works even when
only a subset of algorithm modules are available (e.g., on Colab with
only the core modules cloned).
"""

from dataclasses import dataclass
import warnings

_OPTIONAL_MISSING = []


# Core idealized methods
try:
    from .idealized_methods import (
        IdealTopK, IdealSamplingSubset, IdealSamplingIS,
        VAttentionOracle, IdealEqualSplits, IdealEqualWeightSplits,
    )
except ImportError:
    _OPTIONAL_MISSING.append("idealized_methods")

# Core algorithms used by the notebook
try:
    from .mq_beta_cluster import MQBetaCluster, MQBetaClusterOnly
except ImportError:
    _OPTIONAL_MISSING.append("mq_beta_cluster")

try:
    from .wildcat2 import WildCat2
except ImportError:
    _OPTIONAL_MISSING.append("wildcat2")

try:
    from .tensor_fcfw_lq import TensorFCFWLq
except ImportError:
    _OPTIONAL_MISSING.append("tensor_fcfw_lq")

try:
    from .learned import LearnedCoreset
except ImportError:
    _OPTIONAL_MISSING.append("learned")

try:
    from .kvsculpt import KVSculpt
except ImportError:
    _OPTIONAL_MISSING.append("kvsculpt")

# All other algorithms — optional, fail silently
_other_modules = [
    ("multiq_grouping", "MultiQGrouping"),
    ("kmeans_clustering", "KMeansClustering"),
    ("lsh_crosspoly_multiprobe", "LSHCrossPolytope"),
    ("lsh_cp_group", "LSHCPGroup"),
    ("lsh_crosspoly_clustered", "LSHCrossPolytopeClustered"),
    ("lsh_simhash_snis", "LSHSimHashSNIS"),
    ("lsh_crosspoly_snis", "LSHCrossPolySNIS"),
    ("kmeans_value_clustering", "KMeansValueClustering"),
    ("cp_value_cluster", "CPValueCluster"),
    ("kcluster_topk", "KClusterTopK"),
    ("mq_cluster_topk", "MQClusterTopK"),
    ("qclust_topk", "QClustTopK"),
    ("qclust_augmented", "QClustAugTopK"),
    ("twostage_cluster", "KeyClustValSub"),
    ("topk_cluster_comparison", "TopKKeyClusters"),
    ("topk_cluster_eval", "TopKClusterEval"),
    ("vcluster_pq_kde", "VClusterPQKDE"),
    ("kmeans_cluster_only", "KMeansClusterOnly"),
    ("pq_methods", "VAttentionPQ"),
    ("wildcat3", "WildCat3"),
    ("fcfw_l2", "FCFrankWolfeL2"),
    ("tensor_fcfw_l2", "TensorFCFWL2"),
    ("attention_matching", "AttentionMatchingTopK"),
    ("h2o_eviction", "H2OEviction"),
    ("snapkv_eviction", "SnapKVEviction"),
    ("cake_eviction", "CAKEEviction"),
    ("clusterkv", "ClusterKVEviction"),
    ("vattn_favor_residual", "VAttentionFavorResidual"),
    ("wildcat", "WildcatKVCompression"),
    ("fw_herding", "FWHerdingNystrom"),
    ("value_cluster_is", "ValueClusterIS"),
    ("value_cluster_methods", "VClusterMeanKey"),
    ("tree_attention", "TreeAttention"),
]

for _mod, _cls in _other_modules:
    try:
        _m = __import__(f"src.algorithms.{_mod}", fromlist=[_cls])
        globals()[_cls] = getattr(_m, _cls)
    except (ImportError, AttributeError):
        _OPTIONAL_MISSING.append(_mod)


@dataclass
class MethodSpec:
    cls: type
    kind: str


def _build_registry():
    registry = {}
    _entries = [
        ("ideal_topk", "IdealTopK", "idealized"),
        ("ideal_sampling_is", "IdealSamplingIS", "idealized"),
        ("vattention_oracle", "VAttentionOracle", "idealized"),
        ("mq_beta_cluster", "MQBetaCluster", "algorithm"),
        ("wildcat2", "WildCat2", "algorithm"),
        ("tensor_fcfw_lq", "TensorFCFWLq", "algorithm"),
        ("learned", "LearnedCoreset", "algorithm"),
        ("kvsculpt", "KVSculpt", "algorithm"),
    ]
    for key, cls_name, kind in _entries:
        cls = globals().get(cls_name)
        if cls is not None:
            registry[key] = MethodSpec(cls, kind)
    return registry


METHOD_REGISTRY = _build_registry()

if _OPTIONAL_MISSING:
    missing = ", ".join(sorted(set(_OPTIONAL_MISSING)))
    warnings.warn(
        f"Optional algorithm modules unavailable (OK for notebook): {missing}",
        RuntimeWarning, stacklevel=1,
    )
