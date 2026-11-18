#!/usr/bin/env bash
set -euo pipefail
workspace='/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/04_Performance_v2_Audits_and_Transfers/performance-v2-audit'
script="$workspace/53_extract_performance_v2_diagnostics.py"
test -d "$workspace"
printf '%s  %s\n' '1b3ad98fbe8acd572f1427586c31f755d6457b49640ce8a95eebb8b74e09b01d' "$script" | sha256sum --check --strict
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /experiment-storage/chaidir/env/env-310-fawad/bin/python -B "$script" --repo "/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/03_ConsenDxFormer_and_Datasets/ConsenDxFormer" --output-parent "$workspace"
