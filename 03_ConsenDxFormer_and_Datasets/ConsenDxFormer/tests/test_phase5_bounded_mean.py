"""Isolated synthetic tests only; no original study data or model imports."""
import ast
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path
import sys

import mpmath as mp
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'validation'/'phase5_candidate_c'))
from consendxformer.phase5_bounded_mean import (
    PatientCounts as PC, NodulePrediction as NP, InputIntegrityError,
    NumericalFailure, HYPOTHESIS_IDS as IDS, bounded_patient_mean,
    bounded_tail, decide_tail, holm_exact, bounded_family, pair_correctness,
)
from reference import candidate_p, patient_oracle, holm_oracle
from generators import register, added_fixtures, sizes, exact_support, symmetric_batch


def hand():
    return [PC(str(i),*row) for i,row in enumerate(((0,1,0,0),(0,0,1,0),(1,1,0,0),(3,0,0,0)))]


def test_hand_arithmetic_and_independent_oracle():
    x=bounded_patient_mean(hand())
    s,d,p=patient_oracle([(r.a,r.b,r.c,r.d) for r in hand()])
    assert x.score_sum==s==F(1,2)
    assert x.patient_mean==d==F(1,8) and x.nodule_mean==F(1,7)
    assert x.nodule_count==7 and x.nonzero_patient_count==3
    assert x.tail.p_upper==p==1


@pytest.mark.parametrize('kind',['zero','cancel','positive','negative','one_nonzero'])
def test_all_patients_retained(kind):
    rows=[PC(f'p{i:03}',1,0,0,0) for i in range(99)]
    if kind=='cancel': rows=[PC(f'p{i:03}',0,1,1,0) for i in range(99)]
    if kind=='positive': rows=[PC(f'p{i:03}',0,1,0,0) for i in range(99)]
    if kind=='negative': rows=[PC(f'p{i:03}',0,0,1,0) for i in range(99)]
    if kind=='one_nonzero': rows[0]=PC('p000',0,1,0,0)
    x=bounded_patient_mean(rows)
    assert len(x.counts)==x.tail.patient_count==99
    if kind in ('zero','cancel'): assert x.tail.p_lower==x.tail.p_upper==1
    if kind=='cancel': assert x.cancellation_patient_count==99 and x.discordant_patient_count==99
    if kind=='one_nonzero': assert x.nonzero_patient_count==1 and x.patient_mean==F(1,99)


@pytest.mark.parametrize('multiple',[1,2,7,101])
def test_replication_and_permutation(multiple):
    base=bounded_patient_mean(hand())
    rows=[PC(r.patient_id,r.a*multiple,r.b*multiple,r.c*multiple,r.d*multiple) for r in reversed(hand())]
    other=bounded_patient_mean(rows)
    assert base.score_sum==other.score_sum and base.tail==other.tail
    swapped=bounded_patient_mean([PC(r.patient_id,r.a,r.c,r.b,r.d) for r in rows])
    assert swapped.score_sum==-other.score_sum and swapped.tail==other.tail


@pytest.mark.parametrize('bad',[True,1.0,-1,'1',None,np.int64(1)])
def test_invalid_count_cells(bad):
    with pytest.raises(InputIntegrityError): PC('p',0,bad,0,0)


@pytest.mark.parametrize('identity',['',' ',' p','p ',None,1])
def test_invalid_patient_identity(identity):
    with pytest.raises(InputIntegrityError): PC(identity,1,0,0,0)


def test_duplicate_and_empty():
    with pytest.raises(InputIntegrityError): PC('p',0,0,0,0)
    with pytest.raises(InputIntegrityError): bounded_patient_mean([])
    with pytest.raises(InputIntegrityError): bounded_patient_mean([PC('p',1,0,0,0)]*2)
    with pytest.raises(InputIntegrityError): bounded_patient_mean([{'patient_id':'p'}])


@pytest.mark.parametrize('s',[F(0),F(1),F(20),F(27),F(36),F(99),F(-99),F(199,3),F(7,13)])
def test_probability_enclosure(s):
    tail=bounded_tail(s,99)
    with mp.workdps(130):
        ref=candidate_p(s,99)
        assert mp.mpf(str(tail.p_lower))<=ref<=mp.mpf(str(tail.p_upper))
        assert mp.mpf(str(tail.log_lower))<=mp.log(ref)<=mp.mpf(str(tail.log_upper))
        assert tail.p_upper>0
    for j in range(1,16):
        decision,_=decide_tail(s,99,F(1,20*j))
        with mp.workdps(130): assert decision==bool(ref<=mp.mpf(1)/(20*j))


def test_decimal_context_cannot_change_results():
    expected=bounded_tail(F(91,3),99)
    with localcontext() as ctx:
        ctx.prec=8
        assert bounded_tail(F(91,3),99)==expected


def test_numerical_failure_does_not_relax_threshold():
    s=F(36)
    tail=bounded_tail(s,99,precision=8)
    alpha=(F(tail.p_lower)+F(tail.p_upper))/2
    with pytest.raises(NumericalFailure): decide_tail(s,99,alpha,precisions=(8,))
    result,refined=decide_tail(s,99,alpha)
    with mp.workdps(130): assert result==bool(candidate_p(s,99)<=mp.mpf(alpha.numerator)/alpha.denominator)
    assert refined.precision>=80


@pytest.mark.parametrize('case',['zeros','ones','ties','mixed','one_none','all_none'])
def test_holm_vs_rank_free_oracle(case):
    vals={key:F(i+1,2000) for i,key in enumerate(IDS)}
    if case=='zeros': vals={k:F(0) for k in IDS}
    if case=='ones': vals={k:F(1) for k in IDS}
    if case=='ties': vals={k:F(1,300) for k in IDS}
    if case=='one_none': vals[IDS[2]]=None
    if case=='all_none': vals={k:None for k in IDS}
    out=holm_exact(dict(reversed(list(vals.items()))))
    expected=holm_oracle(vals)
    assert all(out[k]['adjusted']==expected[k] for k in IDS)
    assert len(out)==15
    for k in IDS:
        if vals[k] is None: assert out[k]['raw'] is None and not out[k]['reject']
    if case=='ties': assert all(out[k]['adjusted']==F(1,20) and out[k]['reject'] for k in IDS)


@pytest.mark.parametrize('bad',[F(-1),F(2),float('nan'),float('inf'),0.05,Decimal('NaN')])
def test_invalid_pvalues(bad):
    v={k:F(1) for k in IDS}; v[IDS[0]]=bad
    with pytest.raises(InputIntegrityError): holm_exact(v)


def test_missing_family_and_wrong_cohort():
    with pytest.raises(InputIntegrityError): holm_exact({IDS[0]:F(1)})
    fam={k:hand() for k in IDS}
    fam[IDS[1]]=[PC('different',1,0,0,0)]
    with pytest.raises(InputIntegrityError): bounded_family(fam)


def test_candidate_family_identical_and_mixed():
    zeros=[PC(f'p{i}',1,0,0,0) for i in range(99)]
    ones=[PC(f'p{i}',0,1,0,0) for i in range(99)]
    for fam in ({k:zeros for k in IDS},{k:ones for k in IDS},{k:ones if i<3 else zeros for i,k in enumerate(IDS)}):
        out=bounded_family(fam)
        for k in IDS: assert out[k]['reject']==(fam[k] is ones)


def predictions():
    a=[NP('p1','n1','s1','test','benign','benign',42),NP('p1','n2','s1','test','indeterminate','malignant',42),NP('p2','n3','s2','test','malignant','malignant',42)]
    b=[replace(a[0],predicted_class='indeterminate'),replace(a[1],predicted_class='indeterminate'),a[2]]
    return a,b


def test_pairing_cancellation_and_permutation():
    a,b=predictions(); expected=[r.identity for r in a]
    counts=pair_correctness(a,list(reversed(b)),seed=42,expected_identities=expected)
    assert counts==(PC('p1',0,1,1,0),PC('p2',1,0,0,0))
    assert bounded_patient_mean(counts).cancellation_patient_count==1


@pytest.mark.parametrize('field,value',[('patient_id','another'),('nodule_id','another'),('series_instance_uid','another'),('true_class','malignant'),('seed',43)])
def test_pairing_corruption_fails(field,value):
    a,b=predictions(); expected=[r.identity for r in a]
    b[0]=replace(b[0],**{field:value})
    with pytest.raises(InputIntegrityError): pair_correctness(a,b,seed=42,expected_identities=expected)


@pytest.mark.parametrize('field,value',[('split','train'),('seed',99),('true_class','other'),('predicted_class',float('nan'))])
def test_bad_prediction_records(field,value):
    a,_=predictions()
    with pytest.raises(InputIntegrityError): replace(a[0],**{field:value})


def test_missing_duplicate_extra_pairing():
    a,b=predictions()
    for wrong in (b[:-1],b+[b[0]],[b[0],b[0],b[2]]):
        with pytest.raises(InputIntegrityError): pair_correctness(a,wrong,seed=42,expected_identities=[r.identity for r in a])


def test_frozen_register_and_cluster_sizes():
    reg=register(); assert [s['index'] for s in reg]==list(range(52))
    assert [sum(x['kind']==k for x in reg) for k in ('symmetric','asymmetric','H1','H2','zero_inflated')]==[6,36,2,2,6]
    for x in reg+added_fixtures(): assert len(sizes(x))==99 and sum(sizes(x))==x['N']


@pytest.mark.parametrize('index',list(range(6)))
def test_rng_traversal_and_batch_invariance(index):
    setting=register()[index]; ns=sizes(setting)
    make=lambda:np.random.Generator(np.random.PCG64(np.random.SeedSequence([20260916,index])))
    rng=make(); full=symmetric_batch(rng,setting,3)
    rng=make(); pieces=[symmetric_batch(rng,setting,1) for _ in range(3)]
    for col in range(4): np.testing.assert_array_equal(full[col],np.concatenate([p[col] for p in pieces]))
    # Independent literal scalar implementation of the exact specified draw order.
    rng=make(); scalar=[]
    for rep in range(3):
        scores=[]
        for n in ns:
            if setting['mode']=='independent':
                d=0
                for _ in range(n):
                    disc,sign,correct=rng.random(),rng.random(),rng.random()
                    if disc<float(F(setting['discordance'])): d+=1 if sign<.5 else -1
            else:
                disc,sign=rng.random(),rng.random()
                for _ in range(n): correct=rng.random()
                d=(n if sign<.5 else -n) if disc<float(F(setting['discordance'])) else 0
            scores.append(F(d,n))
        scalar.append((sum(scores,F(0))*12,sum((u*u for u in scores),F(0))*144,sum(u!=0 for u in scores)))
    for rep,vals in enumerate(scalar): assert tuple(int(full[c][rep]) for c in range(3))==vals


def test_source_imports_no_models_or_old_runners():
    source=(ROOT/'src'/'consendxformer'/'phase5_bounded_mean.py').read_text()
    tree=ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): roots=[n.name.split('.')[0] for n in node.names]
        elif isinstance(node,ast.ImportFrom): roots=[(node.module or '').split('.')[0]]
        else: continue
        assert set(roots)<={'__future__','dataclasses','decimal','fractions','typing'}
    assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('open','eval','exec') for node in ast.walk(tree))


@pytest.mark.parametrize('expected',[[],[()], [('p','n')], [('p','n','s','train','benign')], [('p','n','s','test','invalid')]])
def test_invalid_expected_pairing_contract(expected):
    a,b=predictions()
    with pytest.raises(InputIntegrityError): pair_correctness(a,b,seed=42,expected_identities=expected)


def test_governing_hash_corruption(tmp_path):
    import campaign
    import shutil
    # Repository governing documents replace the standalone review-only folder.
    for name in campaign.HASHES: shutil.copyfile(ROOT/'docs'/name,tmp_path/name)
    before=campaign.checked_documents(tmp_path)
    p=tmp_path/'phase5_inference_amendment_decision.md'
    p.write_bytes(p.read_bytes()+b'CHANGED')
    with pytest.raises(ValueError): campaign.checked_documents(tmp_path)
    assert before


def test_validation_output_reservation(tmp_path,monkeypatch):
    import campaign
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    out=campaign.reserve_run('test_run')
    sentinel=out/'sentinel.txt';sentinel.write_text('preserve')
    with pytest.raises(FileExistsError): campaign.reserve_run('test_run')
    assert sentinel.read_text()=='preserve'
    for name in ('../escape','/absolute','','nested/path','a\\b'):
        with pytest.raises(ValueError): campaign.reserve_run(name)


def test_symlink_output_rejected(tmp_path,monkeypatch):
    import campaign
    actual=tmp_path/'real';actual.mkdir()
    (tmp_path/'validation').symlink_to(actual,target_is_directory=True)
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    with pytest.raises(ValueError): campaign.reserve_run('test_run')


def test_deterministic_numpy_export(tmp_path):
    import campaign
    a=tmp_path/'a.npz';b=tmp_path/'b.npz'
    campaign.deterministic_npz(a,z=np.arange(99,dtype=np.int64),y=np.zeros(4))
    campaign.deterministic_npz(b,y=np.zeros(4),z=np.arange(99,dtype=np.int64))
    assert a.read_bytes()==b.read_bytes()
    with pytest.raises(FileExistsError): campaign.deterministic_npz(a,z=np.ones(5))
    with np.load(a,allow_pickle=False) as z: np.testing.assert_array_equal(z['z'],np.arange(99))


@pytest.mark.parametrize('bad_kind',['edit','delete','extra'])
def test_synthetic_bundle_corruption_detected(tmp_path,bad_kind):
    import hashlib
    import json
    from audit import audit
    # A minimal invented integrity fixture fails before numerical read-back.
    # Never require/copy campaign outputs into the repository or rerun a campaign.
    dest=tmp_path/'copy';dest.mkdir()
    member=dest/'tail_checks.csv';member.write_bytes(b'synthetic-only\n')
    (dest/'artifact_manifest.json').write_text(json.dumps({
        member.name:hashlib.sha256(member.read_bytes()).hexdigest()}))
    if bad_kind=='edit':
        p=dest/'tail_checks.csv';p.write_bytes(p.read_bytes()+b'changed\n')
    elif bad_kind=='delete': member.unlink()
    else: (dest/'EXTRA.txt').write_text('unexpected')
    with pytest.raises(AssertionError): audit(dest)
