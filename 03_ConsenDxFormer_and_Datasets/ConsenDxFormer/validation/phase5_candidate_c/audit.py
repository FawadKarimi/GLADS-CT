"""Independent read-back of the synthetic bundle; no production core imports."""
from __future__ import annotations
import csv
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import sys

import mpmath as mp
import numpy as np
from scipy.special import erfc
from scipy.stats import beta


def read_csv(path):
    with path.open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))


def _bin(n,k,p): return F(math.comb(n,k))*p**k*(1-p)**(n-k)


def _tri(n,k,l,p,q):
    return F(math.comb(n,k)*math.comb(n-k,l))*p**k*q**l*(1-p-q)**(n-k-l)


def reconstruct(setting,latent):
    """Rebuild the 99 individual scores, independent of generator aggregates."""
    kind=setting['kind'];scores=[]
    if kind=='asymmetric':
        d,m=setting['d'],setting['M'];k=latent[0]
        scores=[F(1)]*k+[-F(1,d)]*(m-k)
        mass=_bin(m,k,F(1,d+1))
    elif kind=='H1':
        mass=F(1)
        for n,d,k in zip((12,12,4),(2,4,8),latent):
            scores += [F(1)]*k+[-F(1,d)]*(n-k)
            mass *= _bin(n,k,F(1,d+1))
    elif kind=='H2':
        k,p,n=latent
        scores=[F(1) if k else -F(1,4)]+[F(1,4)]*p+[-F(1,4)]*n+[F(0)]*(27-p-n)
        mass=_bin(1,k,F(1,5))*_tri(27,p,n,F(1,200),F(1,200))
    elif kind=='zero_inflated':
        p,n=latent;rho=F(setting['rho'])
        scores=[F(1)]*p+[-F(1,4)]*n+[F(0)]*(20-p-n)
        mass=_tri(20,p,n,rho/5,4*rho/5)
    elif kind=='aggregate_null':
        p,n=latent;r=F(setting['r'])
        scores=[F(1)]*p+[F(0)]*(49-p)+[-F(1)]*n+[F(0)]*(49-n)
        mass=_bin(49,p,r)*_bin(49,n,r)
    elif kind=='deterministic_balanced':
        scores=[F(1),F(-1)]*49;mass=F(1)
    else: raise AssertionError('Unexpected exact kind')
    scores+=[F(0)]*(99-len(scores))
    if setting['mirror']: scores=[-u for u in scores]
    assert len(scores)==99
    return sum(scores,F(0)),sum((u*u for u in scores),F(0)),sum(u!=0 for u in scores),mass


def audit(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'artifact_manifest.json').read_text())
    actual={p.name for p in directory.iterdir() if p.is_file()}
    assert actual==set(manifest)|{'artifact_manifest.json'},'Unexpected/missing file'
    for name,expected in manifest.items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==expected,'Hash corruption: '+name
    status=json.loads((directory/'VALIDATION_STATUS.json').read_text())
    assert status['research_ready'] is False and status['model_forwards']==0
    assert status['stage_a_complete'] is False and status['full_supplemental_completion'] is False
    assert set(status['hypothesis_status'].values())=={'unresolved_not_computed'}
    register=json.loads((directory/'scenario_register.json').read_text())
    settings=register['frozen']+register['additional_exact']
    assert len(register['frozen'])==52 and len(register['additional_exact'])==6
    assert [s['index'] for s in settings]==list(range(58))
    indexed={s['id']:s for s in settings}
    tails=read_csv(directory/'tail_checks.csv')
    assert len(tails)==58*15*3
    expected_keys={(s['id'],r,j) for s in settings for r in ('candidate_c','historical_durkalski','historical_durkalski_20gate') for j in range(1,16)}
    assert {(r['scenario'],r['rule'],int(r['j'])) for r in tails}==expected_keys
    accumulated={};masses={};expectation={};support_rows=0;checked_p={}
    with mp.workdps(130):
        for row in read_csv(directory/'exact_support.csv'):
            scenario=row['scenario'];spec=indexed[scenario]
            s,v,nz,mass=reconstruct(spec,json.loads(row['latent']))
            assert (s,v,nz,mass)==(F(row['S']),F(row['second_moment']),int(row['nonzero']),F(row['mass']))
            support_rows+=1
            masses[scenario]=masses.get(scenario,F(0))+mass
            expectation[scenario]=expectation.get(scenario,F(0))+mass*s
            if s not in checked_p:
                mps=mp.mpf(s.numerator)/s.denominator
                checked_p[s]=min(mp.mpf(1),2*mp.exp(-mps**2/198))
            p=checked_p[s]
            assert mp.mpf(row['candidate_p_lower'])<=p<=mp.mpf(row['candidate_p_upper'])
            candidate=''.join(str(int(p<=mp.mpf(1)/(20*j))) for j in range(1,16))
            assert candidate==row['candidate_rejections']
            for rule,column in (('candidate_c','candidate_rejections'),('historical_durkalski','historical_rejections'),('historical_durkalski_20gate','historical_gated_rejections')):
                for j,decision in enumerate(row[column],1):
                    key=scenario,rule,j
                    accumulated[key]=accumulated.get(key,F(0))+mass*int(decision)
    assert all(x==1 for x in masses.values()) and all(x==0 for x in expectation.values())
    simulated={}
    for setting in settings[:6]:
        with np.load(directory/(setting['id']+'_ledger.npz'),allow_pickle=False) as z:
            assert set(z.files)=={'sum_x12','second_x144','nonzero'}
            s=z['sum_x12'];v=z['second_x144'];nz=z['nonzero']
            assert s.shape==v.shape==nz.shape==(100000,)
            assert all(a.dtype==np.dtype('int64') for a in (s,v,nz))
            assert np.all((0<=nz)&(nz<=99)) and np.all(v>=0)
            # Separate log-threshold arithmetic; guard marginal float ambiguity.
            exponents=s.astype(float)**2/(144*198)
            cand=np.column_stack([exponents>=math.log(40*j) for j in range(1,16)])
            oldp=np.ones(100000);mask=v>0
            oldp[mask]=erfc(np.abs(s[mask])/np.sqrt(2*v[mask]))
            old=np.column_stack([(oldp<=1/(20*j))&mask for j in range(1,16)])
            gated=old&(nz>=20)[:,None]
            for rule,decisions in zip(('candidate_c','historical_durkalski','historical_durkalski_20gate'),(cand,old,gated)):
                for j,count in enumerate(decisions.sum(axis=0),1): simulated[setting['id'],rule,j]=int(count)
    for row in tails:
        key=row['scenario'],row['rule'],int(row['j'])
        alpha=F(1,20*int(row['j']))
        assert F(row['alpha'])==alpha
        if row['calculation']=='exact':
            assert F(row['rejection_probability'])==accumulated[key]
        else:
            count=simulated[key];assert int(row['rejections'])==count and int(row['replicates'])==100000
            # Recalculate binomial endpoints from beta quantiles, independently of helper.
            low=0 if count==0 else beta.ppf(1/468000,count,100001-count)
            high=1 if count==100000 else beta.isf(1/468000,count+1,100000-count)
            assert abs(low-float(row['mc_lower']))<1e-15 and abs(high-float(row['mc_upper']))<1e-15
    return dict(status='PASS',scope='independent_synthetic_readback',exact_support_rows=support_rows,
        exact_mass_and_aggregate_mean_checks=len(masses),tail_cells=len(tails),synthetic_ledger_rows=600000,
        caveat='Historical R runtime and research source integration remain pending; no research validation.')


if __name__=='__main__':
    print(json.dumps(audit(sys.argv[1]),indent=2))
