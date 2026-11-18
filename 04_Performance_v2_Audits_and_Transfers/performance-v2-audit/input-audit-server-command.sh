set -euo pipefail
cd "/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/04_Performance_v2_Audits_and_Transfers/performance-v2-audit"
printf '%s  %s\n' '84090d157959a2b4c525484504c1fc14f398d36fb20b58f34a5a6097321d6bed' '55_audit_performance_v2_inputs.py' | sha256sum --check -
audit_output="input-audit-$(date -u +%Y%m%dT%H%M%S%NZ)"
printf 'Report directory: %s/%s\n' "$PWD" "$audit_output"
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /experiment-storage/chaidir/env/env-310-fawad/bin/python -B '55_audit_performance_v2_inputs.py' --repo "/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/03_ConsenDxFormer_and_Datasets/ConsenDxFormer" --audit-zip audit-20260927T100333767013Z-539541171cb6.zip --output "$audit_output"
