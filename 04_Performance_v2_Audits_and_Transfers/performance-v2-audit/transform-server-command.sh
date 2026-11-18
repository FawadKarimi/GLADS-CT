set -euo pipefail
cd "/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/04_Performance_v2_Audits_and_Transfers/performance-v2-audit"
printf '%s  %s\n' '3203c0d99072e2bc03574b67e1ad47a8ea61bf5dbef3ee8572594274897efab6' '54_probe_performance_v2_transforms.py' | sha256sum --check -
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /experiment-storage/chaidir/env/env-310-fawad/bin/python -B '54_probe_performance_v2_transforms.py' --output "transform-probe-$(date -u +%Y%m%dT%H%M%S%NZ)"
