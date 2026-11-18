"""Review-round metadata collector. Standard library only; never GET artifacts.

The source-embedded plan is the exact allowlist. Unknown dependency URLs need
another source review; discovered links never authorize network requests.
"""
import argparse
import datetime
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOTS = (
    ('torch', '2.13.0+cpu', 'https://download-r2.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl', 'b3c7f4b3f77d06f86a42aab7d0facef7b661ca3c22967bfcc396917d935dd335'),
    ('torchvision', '0.28.0+cpu', 'https://download-r2.pytorch.org/whl/cpu/torchvision-0.28.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl', 'a656cfc3611928c2004484b8d7fcb56616eb771b9a204afd56158fec58075d7b'),
)
ALLOWLIST = frozenset((method, url + suffix) for _, _, url, _ in ROOTS for method, suffix in [('HEAD', ''), ('GET', '.metadata')])
# No redirects have yet been reviewed: any 3xx is recorded and rejected.
REDIRECTS = ()
MAX_BODY = 1024 * 1024

def sha(raw): return hashlib.sha256(raw).hexdigest()
def save(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, sort_keys=True, indent=2); f.write('\n')

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs): return None

class Client:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.end = clock() + 90
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, method, url):
        if (method, url) not in ALLOWLIST: raise ValueError('unreviewed URL/method')
        remaining = self.end - self.clock()
        if remaining <= 0: raise TimeoutError('aggregate metadata deadline')
        request = urllib.request.Request(url, method=method, headers={'Accept-Encoding':'identity', 'User-Agent':'pv2-review-metadata-only'})
        try:
            response = self.opener.open(request, timeout=min(10, remaining))
        except urllib.error.HTTPError as exc:
            # Never read an error or redirect body (which may be artifact data).
            status, location = exc.code, exc.headers.get('Location')
            exc.close()
            raise ValueError(f'HTTP {status}; rejected Location={str(location)[:2048]!r}') from None
        with response:
            if self.clock() >= self.end: raise TimeoutError('backend overrun')
            if response.status != 200 or response.geturl() != url: raise ValueError('status/final URL mismatch')
            headers = {k: response.headers.get(k) for k in ('Content-Length','Content-Type','Content-Encoding','ETag','Last-Modified','Location')}
            if headers['Content-Encoding'] not in (None, 'identity'): raise ValueError('encoded response refused')
            raw = bytearray()
            if method == 'GET':
                while True:
                    if self.clock() >= self.end: raise TimeoutError('aggregate metadata deadline')
                    chunk = response.read(min(16384, MAX_BODY + 1 - len(raw)))
                    if self.clock() >= self.end: raise TimeoutError('backend overrun')
                    raw.extend(chunk)
                    if len(raw) > MAX_BODY: raise ValueError('metadata overflow')
                    if not chunk: break
            return headers, bytes(raw)

def parse_core(raw, name, version, expected_sha):
    if sha(raw) != expected_sha: raise ValueError('publisher sidecar hash mismatch')
    raw.decode('utf-8')
    m = BytesParser().parsebytes(raw)
    norm = lambda v: re.sub(r'[-_.]+', '-', v).lower()
    if len(m.get_all('Name', [])) != 1 or norm(m['Name']) != name or m.get_all('Version') != [version]:
        raise ValueError('core metadata identity mismatch')
    if m.get('Metadata-Version') not in ('2.1','2.2','2.3','2.4','2.5'): raise ValueError('unsupported core metadata version')
    return {'name':name,'version':version,'requires_python':m.get_all('Requires-Python', []), 'requires_dist':m.get_all('Requires-Dist', []), 'dynamic':m.get_all('Dynamic', []), 'metadata_sha256':sha(raw)}

def graph(nodes):
    """Recursive raw dependency evidence, not a home-grown PEP 440/508 resolver."""
    edges, seen = [], set()
    def visit(name):
        if name in seen: return
        seen.add(name)
        for req in nodes[name]['requires_dist']:
            match = re.match(r'^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)', req)
            child = re.sub(r'[-_.]+','-',match[1]).lower() if match else None
            edges.append({'parent':name,'raw_requires_dist':req,'candidate_name':child,
                          'marker_constraint_extras_validated':False,
                          'exact_node_present':child in nodes,
                          'status':'review_required; not an automatically active dependency'})
            if child in nodes: visit(child)
    for name in sorted(nodes): visit(name)
    return edges

def collect(output, client):
    nodes, sizes, failures, receipts = {}, {}, [], []
    for name, version, url, digest in ROOTS:
        for method, requested in [('HEAD',url), ('GET',url+'.metadata')]:
            record={'method':method,'url':requested,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            try:
                headers, raw = client.request(method, requested)
                record.update(headers=headers, received_body_bytes=len(raw), metadata_sha256=sha(raw) if method=='GET' else None)
                if method == 'HEAD':
                    value = headers.get('Content-Length')
                    if raw or not isinstance(value,str) or not re.fullmatch(r'[0-9]+',value) or not 0 < int(value) <= 2**31:
                        raise ValueError('missing/invalid/unbounded HEAD length; no GET fallback')
                    sizes[name]={'publisher_reported_size_bytes':int(value),'artifact_bytes_verified':False}
                else:
                    with (output/(name+'.metadata')).open('xb') as f: f.write(raw)
                    nodes[name]=parse_core(raw,name,version,digest)
                record['status']='metadata_verified_only'
            except Exception as exc:
                record.update(status='failed',error=str(exc)[:4096]); failures.append(record)
            save(output/(name+'-'+method.lower()+'.json'),record); receipts.append(record)
    report={'schema':'pv2_closure_metadata_round_v1','script_sha256':sha(Path(__file__).read_bytes()),
            'target':{'python':'3.12.15','implementation':'CPython','os':'Linux','machine':'x86_64','variant':'cpu','extras':[]},
            'sizes':sizes,'nodes':nodes,'edges':graph(nodes),'failures':failures,
            'dependency_closure_complete':False,'status':'partial_closure_for_review',
            'unresolved':['New exact dependency versions/artifact URLs and sidecar hashes require review before recursive requests.',
                          'PEP 440/508 constraints, markers, extras, Requires-Python and wheel tags not evaluated here.',
                          'No sidecar availability => unresolved; never inspect wheel bytes as fallback.',
                          'Installer roots pip/wheel/setuptools and packaging dependency must join the full closure.'],
            'execution_approved':False,'security_approved':False,'pilot_execution_authorized':False,
            'artifact_bytes_verified':False,'stage_b':'paused'}
    save(output/'report.json',report)
    return 1 if failures else 0

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--authorize-metadata-only',action='store_true'); p.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if not a.authorize_metadata_only: p.error('explicit metadata-only authorization required')
    if a.worker: return collect(a.output,Client())
    a.output.mkdir(parents=False,exist_ok=False)
    save(a.output/'started.json',{'script_sha256':sha(Path(__file__).read_bytes()),'execution_approved':False,'status':'incomplete','allowed_requests':sorted(ALLOWLIST),'redirects':REDIRECTS})
    code=1; error=None
    try:
        # Hard wall limit also covers blocking DNS/header reads; no shell, retries,
        # background daemons or descendants created by the worker.
        r=subprocess.run([sys.executable,'-I','-S','-B',str(Path(__file__).resolve()),'--worker','--authorize-metadata-only','--output',str(a.output.resolve())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=120)
        code=r.returncode
    except Exception as exc: error=repr(exc)[:4096]
    save(a.output/'finished.json',{'exit_code':code,'error':error,'report_present':(a.output/'report.json').is_file(),'execution_approved':False,'closure_complete':False})
    return code

if __name__=='__main__': raise SystemExit(main())
