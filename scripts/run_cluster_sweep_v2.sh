#!/bin/bash
# Sequential runs: all-clusters-included + topK, C=1024, 2048, 4096
set -e
cd /Users/yuvalshemla/Desktop/LoSeM-attention

for C in 1024 2048 4096; do
    echo ""
    echo "=========================================="
    echo "  Running C=${C} (all clusters + topK)"
    echo "=========================================="

    # Update config
    python3 -c "
import yaml
with open('src/evaluation/evaluation_config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['algorithm_configs']['topk_key_clusters']['n_clusters_sweep'] = [${C}]
cfg['algorithm_configs']['topk_oracle_clusters']['n_clusters_sweep'] = [${C}]
cfg['algorithm_configs']['topk_value_clusters']['n_clusters_sweep'] = [${C}]
with open('src/evaluation/evaluation_config.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
print(f'Config: n_clusters_sweep=[${C}]')
"

    python3 -m src.evaluation.run_evaluation \
        --algorithms topk_key_clusters topk_oracle_clusters topk_value_clusters \
        --tasks code_run \
        --name cluster_allincl_C${C}

    echo "=== C=${C} done ==="
done

# Restore to 1024
python3 -c "
import yaml
with open('src/evaluation/evaluation_config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['algorithm_configs']['topk_key_clusters']['n_clusters_sweep'] = [1024]
cfg['algorithm_configs']['topk_oracle_clusters']['n_clusters_sweep'] = [1024]
cfg['algorithm_configs']['topk_value_clusters']['n_clusters_sweep'] = [1024]
with open('src/evaluation/evaluation_config.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"

echo ""
echo "=== All runs complete ==="
