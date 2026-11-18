import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import closure_metadata as m

class Tests(unittest.TestCase):
    def test_exact_allowlist(self):
        c=m.Client()
        for method,url in [('GET',m.ROOTS[0][2]),('GET','https://example.com/a.metadata'),('HEAD',m.ROOTS[0][2]+'?x=1')]:
            with self.assertRaises(ValueError): c.request(method,url)

    def test_head_never_reads(self):
        class Response:
            status=200; headers={'Content-Length':'42'}
            def geturl(self): return m.ROOTS[0][2]
            def __enter__(self): return self
            def __exit__(self,*a): pass
            def read(self,*a): raise AssertionError('HEAD body read')
        c=m.Client(); c.opener.open=lambda *a,**k:Response()
        self.assertEqual(c.request('HEAD',m.ROOTS[0][2])[1],b'')

    def test_redirect_rejected(self):
        c=m.Client()
        def fail(*a,**k): raise m.urllib.error.HTTPError(m.ROOTS[0][2],302,'redirect',{'Location':'https://other.invalid/a'},None)
        c.opener.open=fail
        with self.assertRaisesRegex(ValueError,'rejected Location'): c.request('HEAD',m.ROOTS[0][2])

    def test_identity_and_hash(self):
        b=b'Metadata-Version: 2.4\nName: torch\nVersion: 2.13.0+cpu\nRequires-Dist: foo>=1\n\n'
        self.assertEqual(m.parse_core(b,'torch','2.13.0+cpu',m.sha(b))['requires_dist'],['foo>=1'])
        for raw,name,digest in [(b+b'x','torch',m.sha(b)),(b,'torchvision',m.sha(b))]:
            with self.assertRaises(ValueError): m.parse_core(raw,name,'2.13.0+cpu',digest)

    def test_failure_retention_no_get_fallback(self):
        class Fake:
            calls=[]
            def request(self,method,url):
                self.calls.append((method,url))
                if method=='HEAD': return {'Content-Length':'1e6'},b''
                raise ValueError('invented unavailable sidecar')
        with tempfile.TemporaryDirectory() as d:
            f=Fake(); self.assertEqual(m.collect(Path(d),f),1)
            r=json.loads((Path(d)/'report.json').read_text())
            self.assertEqual(len(r['failures']),4); self.assertFalse(r['dependency_closure_complete'])
            self.assertTrue(all(method!='GET' or url.endswith('.metadata') for method,url in f.calls))
            with self.assertRaises(FileExistsError): m.collect(Path(d),f)

    def test_recursive_cycles_and_unknown(self):
        g=m.graph({'a':{'requires_dist':['b>=1; python_version>="3.12"']},'b':{'requires_dist':['a','missing[extra]>=2']}})
        self.assertEqual(len(g),3)
        self.assertFalse(g[-1]['exact_node_present'])
        self.assertTrue(all(not e['marker_constraint_extras_validated'] for e in g))

    def test_deadline(self):
        times=iter([0,91]); c=m.Client(clock=lambda:next(times))
        with self.assertRaises(TimeoutError): c.request('HEAD',m.ROOTS[0][2])

    def test_overflow(self):
        class Response:
            status=200; headers={}
            def geturl(self): return m.ROOTS[0][2]+'.metadata'
            def __enter__(self): return self
            def __exit__(self,*a): pass
            def read(self,n): return b'x'*n
        c=m.Client(); c.opener.open=lambda *a,**k:Response()
        with self.assertRaisesRegex(ValueError,'overflow'): c.request('GET',m.ROOTS[0][2]+'.metadata')

    def test_parent_timeout_preserves_report(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'result'
            with patch('sys.argv',['x','--output',str(out),'--authorize-metadata-only']), patch.object(m.subprocess,'run',side_effect=m.subprocess.TimeoutExpired('fake worker',120)):
                self.assertEqual(m.main(),1)
            r=json.loads((out/'finished.json').read_text())
            self.assertIn('TimeoutExpired',r['error']); self.assertFalse(r['closure_complete'])

if __name__=='__main__': unittest.main()
