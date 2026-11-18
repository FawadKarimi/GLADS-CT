import os,json,hashlib,zipfile,sys,stat
from pathlib import Path
os.umask(0o077)
d=Path('/home/chaidir/.performance-v2-r50-execution-next-20261011')
a=Path('/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/04_Performance_v2_Audits_and_Transfers/performance-v2-audit')
sys.path.insert(0,str(d));import bounded_supervisor as s
h=s.sha
for p in (d,s.ROOT,s.REPORTS):s.private(p,True)
for p in (d/'gpu-binding.ATTEMPT-003.ACTIVE.json',s.REPORTS/'r50-cuda-synthetic-003',d/'history-pre-binding-correction-003',d/'review-delivery-003'):
 assert not p.exists() and not p.is_symlink(),str(p)
history={}
for n,v in {'performance-v2-r50-gpu-synthetic-attempt-001-evidence-COMPLETE.zip':'280dfa5e1e6ad504d4a1beffb61058606d0fbdff45dbfe7583be41eae0b2688b','performance-v2-r50-gpu-synthetic-attempt-002-evidence.zip':'e2bb089ead51c4d4730d32193a6e2db0367e31c87f5d086149ab9c7e485d36e7'}.items():
 assert h((a/n).read_bytes())==v,n;history[str(a/n)]=v
for attempt in ('001','002'):
 r=s.REPORTS/('r50-cuda-synthetic-'+attempt)
 for p in r.iterdir():
  if p.is_file():history[str(p)]=h(s.read(p))
 for p in d.glob('GPU-ATTEMPT-'+attempt+'*'):
  if p.is_file():history[str(p)]=h(s.read(p))
old=s.read(d/'SOURCE-LOCK.json');assert h(old)=='e2520348a376defa5989815c227419292f13d8043247d74bcd33a2b1c851fdee'
oldlock=json.loads(old);assert len(oldlock)==20
for n,v in oldlock.items():assert h(s.read(d/n))==v,n
original=s.read(s.ROOT/'source-lock.json');assert h(original)==s.LOCK
for n,v in json.loads(original).items():assert h(s.read(s.ROOT/'candidate'/n))==v,n
zp=a/'performance-v2-probe-binding-REVIEW-20261011.zip'
assert not zp.is_symlink() and stat.S_ISREG(zp.stat().st_mode) and zp.stat().st_size<1024**2
assert h(zp.read_bytes())=='d020570b24013f3b6a764b481cfa4c4c8ad549fe4201012f5d06f40d9340ba7f'
with zipfile.ZipFile(zp) as z:
 assert z.testzip() is None
 names=z.namelist();assert len(names)==len(set(names))==37
 for info in z.infolist():
  p=Path(info.filename);assert not p.is_absolute() and all(n not in ('..','.') for n in p.parts) and chr(92) not in info.filename
  assert stat.S_IFMT(info.external_attr>>16) in (0,stat.S_IFREG) and info.file_size<=1024**2
 ledgerraw=z.read('SHA256-LEDGER.json');assert h(ledgerraw)=='538270137838f6ee0b4ccc86b3956d1152dcab14caf671cc387ad92ffdd954a4'
 ledger=json.loads(ledgerraw);assert set(ledger)==set(names)-{'SHA256-LEDGER.json'}
 for n,v in ledger.items():assert h(z.read(n))==v,n
 newraw=z.read('candidate/SOURCE-LOCK.json');assert h(newraw)=='2660b38aa0af3bdb1581a71068ab21137ecf4f9216e01b7f270de5b9d9881c8e'
 newlock=json.loads(newraw);assert len(newlock)==22
 assert set(newlock)-set(oldlock)=={'probe_binding.py','test_probe_binding.py'}
 assert {n for n in oldlock if oldlock[n]!=newlock[n]}=={'gpu_probe.py'}
 newnames={'probe_binding.py','test_probe_binding.py','PROBE-RUNTIME-EVIDENCE.json','gpu-binding.ATTEMPT-003.PROPOSED.json','BINDING-CORRECTION-003-LEDGER.json','GPU-ATTEMPT-003-HISTORICAL-BASE.json','GPU-ATTEMPT-003-STAGING.json'}
 for n in newnames:assert not (d/n).exists() and not (d/n).is_symlink(),n
 delivery=d/'review-delivery-003';delivery.mkdir(mode=0o700)
 for n in names:
  p=delivery/n;p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(z.read(n));f.flush();os.fsync(f.fileno())
 for n,v in ledger.items():assert h(s.read(delivery/n))==v,n
 hist=d/'history-pre-binding-correction-003';hist.mkdir(mode=0o700)
 for n in list(oldlock)+['SOURCE-LOCK.json','SHA256-LEDGER.json']:
  with (hist/n).open('xb') as f:f.write(s.read(d/n));f.flush();os.fsync(f.fileno())
 for n in ('gpu_probe.py','SOURCE-LOCK.json'):
  p=d/(n+'.binding-new')
  with p.open('xb') as f:f.write(z.read('candidate/'+n));f.flush();os.fsync(f.fileno())
  assert h(s.read(d/n))==(oldlock[n] if n in oldlock else h(old))
  os.replace(p,d/n)
 for n in ('probe_binding.py','test_probe_binding.py'):
  with (d/n).open('xb') as f:f.write(z.read('candidate/'+n));f.flush();os.fsync(f.fileno())
 for n in ('PROBE-RUNTIME-EVIDENCE.json','gpu-binding.ATTEMPT-003.PROPOSED.json'):
  with (d/n).open('xb') as f:f.write(z.read('records/'+n));f.flush();os.fsync(f.fileno())
 current=json.loads(s.read(hist/'SHA256-LEDGER.json'))
 for n in list(newlock)+['SOURCE-LOCK.json','PROBE-RUNTIME-EVIDENCE.json','gpu-binding.ATTEMPT-003.PROPOSED.json']:
  current[n]=h(s.read(d/n))
 replacement=(json.dumps(current,indent=2,sort_keys=True)+chr(10)).encode()
 tmp=d/'SHA256-LEDGER.json.binding-new'
 with tmp.open('xb') as f:f.write(replacement);f.flush();os.fsync(f.fileno())
 os.replace(tmp,d/'SHA256-LEDGER.json')
 for n,v in newlock.items():assert h(s.read(d/n))==v,n
for p,v in history.items():assert h(Path(p).read_bytes())==v,p
with (d/'GPU-ATTEMPT-003-HISTORICAL-BASE.json').open('xb') as f:f.write(json.dumps(history,indent=2).encode());f.flush();os.fsync(f.fileno())
receipt={'package_sha256':h(zp.read_bytes()),'addon_lock_sha256':h(newraw),'previous_addon_lock_sha256':h(old),'addon_bindings_matching':22,'candidate_bindings_matching':140,'historical_files_matching':len(history),'exclusive_review_delivery':str(delivery),'previous_addon_retained':str(hist),'probe_launched':False}
with (d/'GPU-ATTEMPT-003-STAGING.json').open('xb') as f:f.write(json.dumps(receipt,indent=2).encode());f.flush();os.fsync(f.fileno())
print(json.dumps(receipt,indent=2))
