"""Explicit advisory/document retrieval only: no binaries, install or execution."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    inventory = json.loads(a.inventory.read_bytes())
    lock = json.loads(a.lock.read_bytes())
    receipt = {'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'sources': [], 'errors': [], 'inventory_sha256': hashlib.sha256(a.inventory.read_bytes()).hexdigest(), 'lock_sha256': hashlib.sha256(a.lock.read_bytes()).hexdigest()}
    targets = [('debian-tracker.json', 'https://security-tracker.debian.org/tracker/data/json')]
    packages = [(p['name'], p['version'].split('+')[0]) for p in lock['packages']]
    packages += [(n, d['version']) for n, d in inventory['embedded_python_distributions'].items()]
    for index, (name, version) in enumerate(packages):
        targets.append((f'pip-{index}.json', 'https://api.github.com/advisories?ecosystem=pip&affects=' + urllib.parse.quote(name + '@' + version) + '&per_page=100'))
    for name, version in [('actions/checkout', '4.2.2'), ('actions/upload-artifact', '4.6.2')]:
        targets.append((name.split('/')[1] + '.json', 'https://api.github.com/advisories?ecosystem=actions&affects=' + urllib.parse.quote(name + '@' + version) + '&per_page=100'))
    for suffix in ('18', '19', '20', '21'):
        targets.append(('python-310' + suffix + '.html', 'https://www.python.org/downloads/release/python-310' + suffix + '/'))
    for filename, url in targets:
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Performance-v2-read-only-security-review'})
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read(40 * 1024**2 + 1)
                if len(raw) > 40 * 1024**2:
                    raise ValueError('oversize advisory metadata')
                link = response.headers.get('Link', '')
            with (a.output / filename).open('xb') as f:
                f.write(raw)
            receipt['sources'].append({'file': filename, 'url': url, 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw), 'pagination_incomplete': 'rel="next"' in link})
        except Exception as exc:
            receipt['errors'].append({'file': filename, 'url': url, 'error': str(exc)})
    receipt['python_queries'] = packages
    with (a.output / 'receipt.json').open('x') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps({'sources': len(receipt['sources']), 'errors': receipt['errors']}, indent=2))

if __name__ == '__main__':
    main()
