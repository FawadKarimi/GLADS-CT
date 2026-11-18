"""Prepare fixed synthetic numeric reference cases; no research inputs."""
from fractions import Fraction as F
import csv
import hashlib
import json
from pathlib import Path
import sys
import mpmath as mp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from consendxformer.phase5_bounded_mean import HYPOTHESIS_IDS
from campaign import reserve_run, csv_write, json_bytes, cp_interval
from reference import legacy_p, holm_oracle


def prepare(name):
    out=reserve_run(name)
    hand=[(0,1,0,0),(0,0,1,0),(1,1,0,0),(3,0,0,0)]
    cases={'hand_4':hand,'repeated_32':hand*8,'all_zero_99':[(1,0,0,0)]*99,
           'cancel_99':[(0,1,1,0)]*99,'one_nonzero_99':[(0,1,0,0)]+[(1,0,0,0)]*98}
    for k in range(21):
        cases['counter_K%02d'%k]=[(0,4,0,0)]*k+[(3,0,1,0)]*(20-k)+[(2,0,0,0)]*24+[(1,0,0,0)]*55
    counts=[];expected=[]
    for name,cells in cases.items():
        scores=[F(b-c,a+b+c+d) for a,b,c,d in cells]
        s=sum(scores,F(0));v=sum((u*u for u in scores),F(0))
        p=legacy_p(s,v)
        expected.append(dict(case_id=name,Q='' if v==0 else str(s*s/v),p='' if p is None else mp.nstr(p,40)))
        for i,(a,b,c,d) in enumerate(cells): counts.append(dict(case_id=name,patient=i,a=a,b=b,c=c,d=d))
    csv_write(out/'legacy_counts.csv',counts,('case_id','patient','a','b','c','d'))
    csv_write(out/'legacy_expected.csv',expected,('case_id','Q','p'))
    rows=[]
    for case in ('zeros','ones','ties','mixed','one_undefined','all_undefined'):
        vals={key:F(i+1,2000) for i,key in enumerate(HYPOTHESIS_IDS)}
        if case=='zeros': vals={k:F(0) for k in HYPOTHESIS_IDS}
        if case=='ones': vals={k:F(1) for k in HYPOTHESIS_IDS}
        if case=='ties': vals={k:F(1,300) for k in HYPOTHESIS_IDS}
        if case=='one_undefined': vals[HYPOTHESIS_IDS[2]]=None
        if case=='all_undefined': vals={k:None for k in HYPOTHESIS_IDS}
        oracle=holm_oracle(vals)
        for key in HYPOTHESIS_IDS:
            rows.append(dict(case_id=case,id=key,raw='' if vals[key] is None else str(vals[key]),
                adjusted='' if oracle[key] is None else str(oracle[key]),
                reject=int(oracle[key] is not None and oracle[key]<=F(1,20))))
    csv_write(out/'holm_cases.csv',rows,('case_id','id','raw','adjusted','reject'))
    cps=[]
    for count in (0,1,2,10,100,1234,50000,99999,100000):
        lo,hi=cp_interval(count)
        cps.append(dict(rejections=count,B=100000,confidence=str(F(1)-F(1,234000)),lower=format(lo,'.17g'),upper=format(hi,'.17g')))
    csv_write(out/'binomial_cases.csv',cps,('rejections','B','confidence','lower','upper'))
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    (out/'fixture_manifest.json').write_bytes(json_bytes(dict(scope='synthetic_numeric_reference_only',sha256=manifest)))
    print(out)


if __name__=='__main__': prepare(sys.argv[1])
