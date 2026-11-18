"""Only invented metadata tests; never invoke the acceptance suite."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'candidate' if (HERE/'candidate').is_dir() else HERE.parent/'ConsenDxFormer-performance-v2'

def deny(event, args):
    if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.posix_spawnp'):
        raise RuntimeError('focused offline test network/subprocess denied: ' + event)

sys.addaudithook(deny)
spec = importlib.util.spec_from_file_location('focused_metadata_tests', SOURCE/'tests/test_performance_v2_acceptance_entry.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.MetadataBindingTests)
with (HERE/'FOCUSED-TESTS.log').open('w', encoding='utf-8') as log:
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
record = {'python': sys.version, 'executable': sys.executable,
          'executable_sha256': hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
          'platform': sys.platform, 'tests': result.testsRun, 'failures': len(result.failures),
          'errors': len(result.errors), 'skips': len(result.skipped), 'passed': result.wasSuccessful(),
          'scope': 'MetadataBindingTests only; invented records and temporary directories',
          'network_and_subprocess_denied': True, 'acceptance_suite_executed': False,
          'native_linux_validation': False, 'pilot_execution_authorized': False}
record['source_hashes'] = {name: hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in
    ('scripts/performance_v2_native_acceptance.py', 'tests/test_performance_v2_acceptance_entry.py')}
(HERE/'FOCUSED-TESTS.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record, indent=2))
raise SystemExit(0 if result.wasSuccessful() else 1)
