"""Freeze source without changing any existing implementation/configuration."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

R = Path(__file__).resolve().parent
S = R.parent/'ConsenDxFormer-performance-v2'
paths = set()
for pattern in ('src/consendxformer/performance_v2/*.py', 'tests/test_performance_v2_*.py',
                'configs/performance_v2/*.json', 'docs/performance_v2_*.md', 'requirements-phase*.txt'):
    paths.update(p for p in S.glob(pattern) if p.name != 'native_acceptance_source_lock.json')
paths.update(S/name for name in ('src/consendxformer/__init__.py', 'src/consendxformer/phase4_model.py',
    'scripts/53_extract_performance_v2_diagnostics.py', 'scripts/54_probe_performance_v2_transforms.py',
    'scripts/55_audit_performance_v2_inputs.py', 'scripts/performance_v2_native_acceptance.py',
    'docs/r3_entropy_results_and_closure.md', 'docs/r4_claims_packaging_acceptance.md',
    'docs/stage_a_bounded_signoff.md', 'docs/stage_a_evidence_reconciliation.md'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
lock = {'schema': 'performance_v2_native_acceptance_source_lock_v1',
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=S).decode().strip(),
        'pilot_execution_authorized': False, 'self_excluded': True,
        'files': {p.relative_to(S).as_posix(): sha(p) for p in sorted(paths)}}
target = S/'configs/performance_v2/native_acceptance_source_lock.json'
target.write_text(json.dumps(lock, sort_keys=True, indent=2)+'\n', encoding='utf-8', newline='\n')
for p in sorted(paths | {target}):
    out = R/'candidate'/p.relative_to(S)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, out)
(R/'SOURCE-LOCK.sha256').write_text(sha(target)+'  candidate/configs/performance_v2/native_acceptance_source_lock.json\n', newline='\n')
print(json.dumps({'files': len(paths), 'source_lock_sha256': sha(target)}))
