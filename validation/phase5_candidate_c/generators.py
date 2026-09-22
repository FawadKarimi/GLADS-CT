"""The frozen 52-setting synthetic register plus six approved exact fixtures.

No fitting to research outcomes. Integer/rational masses include all outcomes.
"""
from fractions import Fraction as F
from math import comb
import numpy as np


def register():
    settings = []
    for discordance in (F(1, 10), F(3, 10), F(3, 5)):
        for mode in ('independent', 'shared'):
            settings.append(dict(kind='symmetric', discordance=str(discordance), mode=mode, P=99, N=246))
    for d, ms in ((2, (1, 2, 5, 10, 20, 40, 84)), (4, (1, 2, 5, 10, 20, 28)), (8, (1, 2, 5, 10, 12))):
        for m in ms:
            for mirror in (False, True):
                settings.append(dict(kind='asymmetric', d=d, M=m, mirror=mirror, P=99, N=183))
    for kind in ('H1', 'H2'):
        for mirror in (False, True):
            settings.append(dict(kind=kind, mirror=mirror, P=99, N=183))
    for rho in (F(1,20), F(1,4), F(1,2)):
        for mirror in (False, True):
            settings.append(dict(kind='zero_inflated', rho=str(rho), mirror=mirror, P=99, N=183))
    if len(settings) != 52:
        raise AssertionError('Frozen register count changed')
    for index, item in enumerate(settings):
        item['index'] = index
        item['id'] = 'S%02d' % index
        item['calculation'] = 'simulation' if item['kind'] == 'symmetric' else 'exact'
    return settings


def added_fixtures():
    out = []
    for r in (F(1,20), F(1,2)):
        for mirror in (False, True):
            out.append(dict(kind='aggregate_null', r=str(r), mirror=mirror, P=99, N=183))
    for mirror in (False, True):
        out.append(dict(kind='deterministic_balanced', mirror=mirror, P=99, N=183))
    for i, x in enumerate(out):
        x.update(id='A%02d' % i, index=52+i, calculation='exact')
    return out


def sizes(setting):
    kind = setting['kind']
    if kind == 'symmetric':
        result = [1 + p % 4 for p in range(99)]
    elif kind == 'asymmetric':
        d, m = setting['d'], setting['M']
        twos = 84 - m * (d-1)
        result = [d]*m + [2]*twos + [1]*(99-m-twos)
    elif kind == 'H1':
        result = [2]*12 + [4]*12 + [8]*4 + [2]*8 + [1]*63
    elif kind == 'H2':
        result = [4]*28 + [1]*71
    else:
        result = [4]*20 + [2]*24 + [1]*55
    assert len(result) == 99 and sum(result) == setting['N'] and min(result) > 0
    return result


def binomial_support(n, probability):
    for k in range(n+1):
        yield k, F(comb(n,k)) * probability**k * (1-probability)**(n-k)


def trinomial_support(n, pos, neg):
    zero = 1-pos-neg
    for kp in range(n+1):
        for kn in range(n-kp+1):
            kz = n-kp-kn
            mass = F(comb(n,kp)*comb(n-kp,kn)) * pos**kp * neg**kn * zero**kz
            yield kp,kn,mass


def exact_support(setting):
    """Yield (S, sum U^2, nonzero count, probability, latent counts)."""
    sizes(setting)
    sign = -1 if setting.get('mirror') else 1
    kind = setting['kind']
    if kind == 'asymmetric':
        d,m = setting['d'],setting['M']
        for k,mass in binomial_support(m,F(1,d+1)):
            yield sign*(F(k)-F(m-k,d)), F(k)+F(m-k,d*d), m, mass, (k,)
    elif kind == 'H1':
        for k2,p2 in binomial_support(12,F(1,3)):
            for k4,p4 in binomial_support(12,F(1,5)):
                for k8,p8 in binomial_support(4,F(1,9)):
                    s=sum((F(k)-F(n-k,d) for k,n,d in ((k2,12,2),(k4,12,4),(k8,4,8))),F(0))
                    v=sum((F(k)+F(n-k,d*d) for k,n,d in ((k2,12,2),(k4,12,4),(k8,4,8))),F(0))
                    yield sign*s,v,28,p2*p4*p8,(k2,k4,k8)
    elif kind == 'H2':
        for k0,p0 in binomial_support(1,F(1,5)):
            for kp,kn,mass in trinomial_support(27,F(1,200),F(1,200)):
                s=F(k0)-F(1-k0,4)+F(kp-kn,4)
                v=F(k0)+F(1-k0,16)+F(kp+kn,16)
                yield sign*s,v,1+kp+kn,p0*mass,(k0,kp,kn)
    elif kind == 'zero_inflated':
        rho=F(setting['rho'])
        for kp,kn,mass in trinomial_support(20,rho/5,rho*4/5):
            yield sign*(F(kp)-F(kn,4)), F(kp)+F(kn,16), kp+kn, mass, (kp,kn)
    elif kind == 'aggregate_null':
        r=F(setting['r'])
        # 10 + 12 + 27 pairs within the three size categories, one leftover zero.
        for kp,pp in binomial_support(49,r):
            for kn,pn in binomial_support(49,r):
                yield F(sign*(kp-kn)),F(kp+kn),kp+kn,pp*pn,(kp,kn)
    elif kind == 'deterministic_balanced':
        yield F(0),F(98),98,F(1),(49,49)
    else:
        raise ValueError('This setting is simulated, not exact')


def symmetric_batch(rng, setting, batch):
    """Draw order is replicate -> patient -> nodule, including unused draws.

    Independent: three uniforms per nodule (discordance, sign, correctness).
    Shared: two patient uniforms then one correctness uniform per nodule.
    Returns integer D by patient, plus full uniform ledger for stream hashing.
    """
    ns=np.asarray(sizes(setting),dtype=np.int64)
    discordance=float(F(setting['discordance']))
    starts=np.r_[0,np.cumsum(ns)[:-1]]
    if setting['mode']=='independent':
        ledger=rng.random((batch,int(ns.sum()),3))
        differences=(ledger[:,:,0]<discordance)*np.where(ledger[:,:,1]<0.5,1,-1)
        ds=np.add.reduceat(differences,starts,axis=1)
    else:
        widths=ns+2
        offsets=np.r_[0,np.cumsum(widths)[:-1]]
        ledger=rng.random((batch,int(widths.sum())))
        active=ledger[:,offsets]<discordance
        signs=np.where(ledger[:,offsets+1]<0.5,1,-1)
        ds=active*signs*ns
    # lcm(1,2,3,4)=12. Scores and second moments aggregate EXACTLY as integers.
    scaled_sum=np.sum(ds*(12//ns),axis=1,dtype=np.int64)
    scaled_second=np.sum(ds*ds*(144//(ns*ns)),axis=1,dtype=np.int64)
    nz=np.count_nonzero(ds,axis=1).astype(np.int64)
    return scaled_sum,scaled_second,nz,ledger
