"""Run only the frozen synthetic campaign. No research-data input option.

Full validation release remains blocked until R runtime checks and historical
source-loader integration are independently reviewed. This harness never emits
a research COMPLETED marker or marks any of the 15 study hypotheses tested.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import re
import shutil
import sys
import zipfile
from fractions import Fraction as F
from functools import lru_cache

import mpmath as mp
import numpy as np
import scipy
from scipy.special import erfc
from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from consendxformer.phase5_bounded_mean import bounded_tail, HYPOTHESIS_IDS
from generators import register, added_fixtures, exact_support, symmetric_batch, sizes
from reference import candidate_p, legacy_p

BUDGET=100000
ALPHAS=tuple(F(1,20*j) for j in range(1,16))
RULES=('candidate_c','historical_durkalski','historical_durkalski_20gate')
HASHES={
 'phase5_supplemental_evaluation_proposal.md':'4d3f7500071a5c352b2a918c0d26ffabc90b4e11af5982cdfdef7880e1054620',
 'phase5_inference_resolution_proposal.md':'30a15d75f3bb09a6c44680c652100354713a492e2f0a5bfd2df3eed76699a315',
 'phase5_inference_amendment_decision.md':'e3e6232cab56261a126122aa34ad9a6ef2ddde4c22036321349bc509f02dacda',
}


def digest(data): return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')


def checked_documents(directory):
    result={}
    for name,expected in HASHES.items():
        data=(directory/name).read_bytes()
        norm=data.replace(b'\r\n',b'\n').replace(b'\r',b'\n')
        if digest(norm)!=expected: raise ValueError('Governing document hash mismatch: '+name)
        result[name]=dict(raw_sha256=digest(data),lf_sha256=digest(norm),bytes=len(data))
    return result


def source_snapshot():
    paths=list((ROOT/'src').rglob('*.py'))+list((ROOT/'tests').rglob('*.py'))
    paths+=list((ROOT/'validation'/'phase5_candidate_c').glob('*.py'))
    paths+=list((ROOT/'validation'/'phase5_candidate_c').glob('*.R'))
    paths += [ROOT/'requirements-phase5-synthetic-validation.txt', ROOT/'docs'/'phase5_inference_implementation_authorization.md']
    return {str(p.relative_to(ROOT)).replace('\\','/'):digest(p.read_bytes()) for p in sorted(paths)}


def csv_write(path,rows,fields):
    with path.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def deterministic_npz(path,**arrays):
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,array in sorted(arrays.items()):
            out=io.BytesIO(); np.save(out,np.asarray(array),allow_pickle=False)
            item=zipfile.ZipInfo(name+'.npy',date_time=(1980,1,1,0,0,0))
            item.compress_type=zipfile.ZIP_DEFLATED
            item.external_attr=0o100644<<16
            z.writestr(item,out.getvalue())


@lru_cache(None)
def candidate_checked(s):
    tail=bounded_tail(s,99,precision=80)
    with mp.workdps(130):
        ref=candidate_p(s,99)
        lo,hi=mp.mpf(str(tail.p_lower)),mp.mpf(str(tail.p_upper))
        if not lo<=ref<=hi: raise AssertionError('Independent oracle outside production enclosure')
        answers=[]
        for alpha in ALPHAS:
            if F(tail.p_upper)<=alpha: decision=True
            elif F(tail.p_lower)>alpha: decision=False
            else: raise ArithmeticError('Numerical threshold ambiguity')
            if decision!=bool(ref<=mp.mpf(alpha.numerator)/alpha.denominator):
                raise AssertionError('Candidate decision disagrees with independent oracle')
            answers.append(decision)
    return tuple(answers),str(tail.p_lower),str(tail.p_upper)


@lru_cache(None)
def legacy_checked(s,second):
    if second==0: return (False,)*15,''
    q=s*s/second
    approx=math.erfc(math.sqrt(float(q)/2))
    with mp.workdps(100):
        ref=legacy_p(s,second)
        exact=tuple(bool(ref<=mp.mpf(a.numerator)/a.denominator) for a in ALPHAS)
        if tuple(approx<=float(a) for a in ALPHAS)!=exact:
            raise ArithmeticError('Legacy float/oracle decision mismatch')
        return exact,mp.nstr(ref,35)


def cp_interval(rejections):
    # 1-confidence = .01/(52*15*3); two tails, exactly 1/468000 each.
    tail=1/468000
    lo=0.0 if rejections==0 else float(beta.ppf(tail,rejections,BUDGET-rejections+1))
    hi=1.0 if rejections==BUDGET else float(beta.isf(tail,rejections+1,BUDGET-rejections))
    if not 0<=lo<=rejections/BUDGET<=hi<=1: raise ArithmeticError('Invalid binomial interval')
    return lo,hi


def exact_scenario(setting,writer):
    totals=[[F(0) for _ in ALPHAS] for _ in RULES]
    total_mass=F(0); expected_s=F(0); zero=F(0); withheld=F(0); states=0
    for s,second,nz,mass,latent in exact_support(setting):
        if mass<0 or second<0 or abs(s)>99 or not 0<=nz<=99:
            raise AssertionError('Malformed exact support')
        states+=1; total_mass+=mass; expected_s+=s*mass
        zero+=mass*(second==0); withheld+=mass*(nz<20 or second==0)
        cand,plo,phi=candidate_checked(s)
        legacy,legacy_value=legacy_checked(s,second)
        gate=legacy if nz>=20 else (False,)*15
        for ri,decisions in enumerate((cand,legacy,gate)):
            for j,reject in enumerate(decisions): totals[ri][j]+=mass*reject
        writer.writerow(dict(scenario=setting['id'],latent=json.dumps(latent,separators=(',',':')),
            S=str(s),second_moment=str(second),nonzero=nz,mass=str(mass),candidate_p_lower=plo,
            candidate_p_upper=phi,historical_p=legacy_value,
            candidate_rejections=''.join(str(int(x)) for x in cand),
            historical_rejections=''.join(str(int(x)) for x in legacy),
            historical_gated_rejections=''.join(str(int(x)) for x in gate)))
    if total_mass!=1 or expected_s!=0: raise AssertionError('Support mass or aggregate mean null failure')
    rows=[]
    for ri,rule in enumerate(RULES):
        for j,alpha in enumerate(ALPHAS):
            rate=totals[ri][j]
            if ri==0 and rate>alpha: raise AssertionError('Candidate exact null violation')
            rows.append(dict(scenario=setting['id'],kind=setting['kind'],calculation='exact',rule=rule,
                j=j+1,alpha=str(alpha),rejection_probability=str(rate),rate_float=format(float(rate),'.17g'),
                excess=str(rate-alpha),replicates='',rejections='',mc_lower='',mc_upper='',
                zero_second_moment=str(zero),withheld=str(withheld if ri==2 else zero if ri==1 else F(0)),
                status='above_nominal' if rate>alpha else 'not_above_nominal'))
    return rows,dict(states=states,total_mass=str(total_mass),expected_S=str(expected_s))


def simulated_scenario(setting,outdir):
    index=setting['index']; rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20260916,index])))
    ts=[];vs=[];nzs=[]; stream=hashlib.sha256()
    for start in range(0,BUDGET,1000):
        t,v,nz,ledger=symmetric_batch(rng,setting,min(1000,BUDGET-start))
        stream.update(np.ascontiguousarray(ledger,dtype='<f8').tobytes())
        ts.append(t);vs.append(v);nzs.append(nz)
    t=np.concatenate(ts); v=np.concatenate(vs); nz=np.concatenate(nzs)
    # Verify every realized candidate sum against the production+independent oracle lookup.
    unique=np.unique(np.abs(t))
    lookup={int(k):candidate_checked(F(int(k),12))[0] for k in unique}
    rejects_c=np.asarray([lookup[int(k)] for k in np.abs(t)],dtype=bool)
    nonzero=v>0
    p=np.ones(BUDGET,dtype=np.float64)
    p[nonzero]=erfc(np.sqrt((t[nonzero].astype(float)**2/v[nonzero])/2))
    thresholds=np.asarray([float(a) for a in ALPHAS])
    rejects_old=(p[:,None]<=thresholds[None,:])&nonzero[:,None]
    # Resolve all close threshold comparisons independently; do not relax cutoffs.
    near=np.where(np.min(np.abs(p[:,None]-thresholds[None,:]),axis=1)<1e-12)[0]
    for row in near:
        rejects_old[row,:]=legacy_checked(F(int(t[row]),12),F(int(v[row]),144))[0]
    # Deterministic independent high-precision spot check, not claimed R execution.
    oracle_rows=np.linspace(0,BUDGET-1,100,dtype=int)
    for row in oracle_rows:
        check=legacy_checked(F(int(t[row]),12),F(int(v[row]),144))[0]
        if tuple(bool(x) for x in rejects_old[row])!=check: raise AssertionError('Legacy oracle spot check')
    gated=rejects_old&(nz>=20)[:,None]
    deterministic_npz(outdir/(setting['id']+'_ledger.npz'),sum_x12=t,second_x144=v,nonzero=nz)
    rows=[]
    for rule,rej in zip(RULES,(rejects_c,rejects_old,gated)):
        for j,alpha in enumerate(ALPHAS):
            count=int(rej[:,j].sum()); lower,upper=cp_interval(count)
            status='supports_this_cell' if upper<=float(alpha) else 'failure' if lower>float(alpha) else 'inconclusive'
            if rule=='candidate_c' and status=='failure': raise AssertionError('Candidate simulated lower bound above nominal')
            withheld=int(np.sum(~nonzero | (nz<20))) if rule==RULES[2] else int(np.sum(~nonzero)) if rule==RULES[1] else 0
            rows.append(dict(scenario=setting['id'],kind=setting['kind'],calculation='simulation',rule=rule,
                j=j+1,alpha=str(alpha),rejection_probability='',rate_float=format(count/BUDGET,'.17g'),
                excess=str(F(count,BUDGET)-alpha),replicates=BUDGET,rejections=count,
                mc_lower=format(lower,'.17g'),mc_upper=format(upper,'.17g'),
                zero_second_moment=str(F(int(np.sum(~nonzero)),BUDGET)),withheld=str(F(withheld,BUDGET)),status=status))
    return rows,dict(replicates=BUDGET,uniform_stream_sha256=stream.hexdigest(),
        rng='Generator(PCG64(SeedSequence([20260916,%d])))'%index,
        candidate_unique_absolute_sums=len(unique),legacy_mp_spot_checks=len(oracle_rows),near_threshold_rows=len(near))


def reserve_run(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name): raise ValueError('Invalid run name')
    parent=ROOT/'validation'/'phase5_candidate_c'/'runs'
    if any(p.is_symlink() for p in (ROOT,ROOT/'validation',parent.parent,parent)):
        raise ValueError('Symlink output path rejected')
    parent.mkdir(parents=True,exist_ok=True)
    target=parent/name
    target.mkdir(exist_ok=False)
    return target


def run(name,documents):
    sources=source_snapshot(); docs=checked_documents(documents)
    out=reserve_run(name)
    try:
        settings=register(); added=added_fixtures()
        (out/'scenario_register.json').write_bytes(json_bytes(dict(frozen=settings,additional_exact=added,
            simulation_budget=BUDGET,mc_confidence=str(F(1)-F(1,234000)),
            note='52 frozen settings; added fixtures exact only, no simulated-cell expansion')))
        rows=[]; details={}
        fields=('scenario','latent','S','second_moment','nonzero','mass','candidate_p_lower','candidate_p_upper','historical_p','candidate_rejections','historical_rejections','historical_gated_rejections')
        with (out/'exact_support.csv').open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
            for setting in settings+added:
                if setting['calculation']=='exact': r,detail=exact_scenario(setting,writer)
                else: r,detail=simulated_scenario(setting,out)
                rows.extend(r);details[setting['id']]=detail
                print(setting['id'],setting['kind'],setting['calculation'],'checked',flush=True)
        if len(rows)!=58*15*3: raise AssertionError('Missing settings/rules/thresholds')
        csv_write(out/'tail_checks.csv',rows,tuple(rows[0]))
        # Explicit known counterexample fixture, exact rational evidence from the frozen plan.
        counter=next(x for x in settings if x['kind']=='asymmetric' and x['d']==4 and x['M']==20 and not x['mirror'])
        counter_rows=[x for x in rows if x['scenario']==counter['id']]
        for j,expected in ((1,F(7549007091249,95367431640625)),(15,F(1100957668913,95367431640625))):
            found=next(x for x in counter_rows if x['j']==j and x['rule']==RULES[1])
            if F(found['rejection_probability'])!=expected: raise AssertionError('Historical counterexample mismatch')
            for rule in (RULES[1],RULES[2]):
                value=next(x for x in counter_rows if x['j']==j and x['rule']==rule)
                if F(value['rejection_probability'])!=expected: raise AssertionError('Historical safeguard unexpectedly fixes counterexample')
        if any(F(x['rejection_probability'])!=0 for x in counter_rows if x['rule']==RULES[0]):
            raise AssertionError('Unexpected counterexample rejection under C')
        if source_snapshot()!=sources or checked_documents(documents)!=docs:
            raise AssertionError('Inputs/source changed during validation')
        bad_c=[x for x in rows if x['rule']==RULES[0] and x['status'] in ('failure','inconclusive','above_nominal')]
        status=dict(scope='synthetic_core_validation_only',date='2026-09-18',
            base_commit='ca6c018ffbedae12988074fb66052592212a3c2f',
            frozen_settings=52,additional_exact_fixtures=6,exact_settings=46,
            simulated_settings=6,simulated_replicates_total=600000,tail_cells=len(rows),
            candidate_cells=len([x for x in rows if x['rule']==RULES[0]]),
            candidate_adverse_or_inconclusive_cells=len(bad_c),
            candidate_finite_suite_status='PASS' if not bad_c else 'BLOCKED',
            historical_counterexample_reproduced=True,
            rscript_available=bool(shutil.which('Rscript')),r_runtime_reference_status='NOT_RUN',
            research_source_loader_integration='NOT_IMPLEMENTED_IN_THIS_CORE_TRANCHE',
            independent_code_review='PENDING',research_ready=False,research_data_read=False,
            model_forwards=0,formal_research_tests_computed=False,
            hypothesis_status={k:'unresolved_not_computed' for k in HYPOTHESIS_IDS},
            full_supplemental_completion=False,stage_a_complete=False,
            warnings=['Finite synthetic settings do not prove the broad-null theorem.',
                      'Independence is an accepted modeling assumption, not empirically established.',
                      'R runtime/reference, repository integration and independent review remain gates.'],
            environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,mpmath=mp.__version__,platform=platform.platform()),
            sources=sources,governing_documents=docs,scenario_details=details)
        (out/'VALIDATION_STATUS.json').write_bytes(json_bytes(status))
        manifest={str(p.relative_to(out)):digest(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()}
        (out/'artifact_manifest.json').write_bytes(json_bytes(manifest))
        print('SYNTHETIC_CORE_CAMPAIGN='+status['candidate_finite_suite_status'])
        print('RESEARCH_READY=False; R_RUNTIME_REFERENCE=NOT_RUN; SOURCE_INTEGRATION=PENDING')
        return out
    except BaseException as exc:
        (out/'FAILED.json').write_bytes(json_bytes(dict(error_type=type(exc).__name__,error=str(exc),research_ready=False)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-frozen-null-campaign',action='store_true',required=True)
    parser.add_argument('--documents-dir',type=Path,required=True)
    parser.add_argument('--output-name',required=True)
    args=parser.parse_args()
    run(args.output_name,args.documents_dir)
