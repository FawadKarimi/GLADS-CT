"""Exact legacy artifact only. No loading, export I/O, or authorization here."""
import collections, hashlib, json

ARTIFACT_SHA = '0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a'
ARTIFACT_BYTES = 102530333
MISSING = frozenset(['bn1.num_batches_tracked', 'layer1.0.bn1.num_batches_tracked', 'layer1.0.bn2.num_batches_tracked', 'layer1.0.bn3.num_batches_tracked', 'layer1.0.downsample.1.num_batches_tracked', 'layer1.1.bn1.num_batches_tracked', 'layer1.1.bn2.num_batches_tracked', 'layer1.1.bn3.num_batches_tracked', 'layer1.2.bn1.num_batches_tracked', 'layer1.2.bn2.num_batches_tracked', 'layer1.2.bn3.num_batches_tracked', 'layer2.0.bn1.num_batches_tracked', 'layer2.0.bn2.num_batches_tracked', 'layer2.0.bn3.num_batches_tracked', 'layer2.0.downsample.1.num_batches_tracked', 'layer2.1.bn1.num_batches_tracked', 'layer2.1.bn2.num_batches_tracked', 'layer2.1.bn3.num_batches_tracked', 'layer2.2.bn1.num_batches_tracked', 'layer2.2.bn2.num_batches_tracked', 'layer2.2.bn3.num_batches_tracked', 'layer2.3.bn1.num_batches_tracked', 'layer2.3.bn2.num_batches_tracked', 'layer2.3.bn3.num_batches_tracked', 'layer3.0.bn1.num_batches_tracked', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.bn3.num_batches_tracked', 'layer3.0.downsample.1.num_batches_tracked', 'layer3.1.bn1.num_batches_tracked', 'layer3.1.bn2.num_batches_tracked', 'layer3.1.bn3.num_batches_tracked', 'layer3.2.bn1.num_batches_tracked', 'layer3.2.bn2.num_batches_tracked', 'layer3.2.bn3.num_batches_tracked', 'layer3.3.bn1.num_batches_tracked', 'layer3.3.bn2.num_batches_tracked', 'layer3.3.bn3.num_batches_tracked', 'layer3.4.bn1.num_batches_tracked', 'layer3.4.bn2.num_batches_tracked', 'layer3.4.bn3.num_batches_tracked', 'layer3.5.bn1.num_batches_tracked', 'layer3.5.bn2.num_batches_tracked', 'layer3.5.bn3.num_batches_tracked', 'layer4.0.bn1.num_batches_tracked', 'layer4.0.bn2.num_batches_tracked', 'layer4.0.bn3.num_batches_tracked', 'layer4.0.downsample.1.num_batches_tracked', 'layer4.1.bn1.num_batches_tracked', 'layer4.1.bn2.num_batches_tracked', 'layer4.1.bn3.num_batches_tracked', 'layer4.2.bn1.num_batches_tracked', 'layer4.2.bn2.num_batches_tracked', 'layer4.2.bn3.num_batches_tracked'])
EXCLUDED = frozenset(('fc.weight', 'fc.bias'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require_recipe(spec):
    from consendxformer.performance_v2.imagenet_mapping import r50_spec
    leaves, frozen = r50_spec()
    normalized = {k: (tuple(v[0]), v[1], v[2]) for k,v in spec.items()}
    if normalized != frozen or len(frozen) != 320 or len(MISSING) != 53:
        raise ValueError('frozen 320-entry recipe required')
    counters = {k for k,v in frozen.items() if k.endswith('.num_batches_tracked')}
    bns = {k for k,v in leaves.items() if v[0] == 'BatchNorm2d'}
    if counters != MISSING or {k.removesuffix('.num_batches_tracked') for k in MISSING} != bns:
        raise ValueError('exact observed missing counters required')
    if any(frozen[k] != ((), 'torch.int64', 'buffer') for k in MISSING):
        raise ValueError('counter shape/dtype/role')
    if any(leaves[k][2:] != (1e-5, .1, True, True) for k in bns):
        raise ValueError('frozen fixed-momentum BatchNorm required')


def require_identity(identity):
    if (type(identity) is not dict or set(identity) != {'path','bytes','sha256'}
        or type(identity['path']) is not str or not identity['path']
        or type(identity['bytes']) is not int or identity['bytes'] != ARTIFACT_BYTES
        or identity['sha256'] != ARTIFACT_SHA):
        raise ValueError('exact raw artifact identity required')


def _attest_raw(state, spec):
    """Internal validator; public adaptation additionally gates the frozen recipe."""
    import torch
    from consendxformer.performance_v2.checkpoint import state_sha256
    if type(state) not in (dict, collections.OrderedDict):
        raise ValueError('raw container type')
    if len(state) != 267 or set(state) != set(spec)-MISSING:
        raise ValueError('exact 267 raw keys / 53 absent counters / no extras required')
    intervals=[]; inventory={}
    # Validate every supplied entry before creating any adapted tensor.
    for name in sorted(state):
        t=state[name];shape,dtype,role=spec[name]
        if (type(t) not in (torch.Tensor, torch.nn.Parameter)
            or t.device.type != 'cpu' or t.layout != torch.strided
            or tuple(t.shape) != tuple(shape) or str(t.dtype) != dtype
            or not t.is_contiguous() or t.storage_offset() != 0
            or t.numel() == 0 or t.is_conj() or t.is_neg()
            or t.untyped_storage().nbytes() != t.numel()*t.element_size()
            or not bool(torch.isfinite(t).all())
            or (role == 'buffer' and (type(t) is torch.nn.Parameter or t.requires_grad))):
            raise ValueError('invalid raw tensor: '+name)
        lo=t.untyped_storage().data_ptr();hi=lo+t.untyped_storage().nbytes()
        if any(lo<b and a<hi for a,b in intervals): raise ValueError('raw storage alias')
        intervals.append((lo,hi))
        if name.endswith('running_var') and bool((t<0).any()): raise ValueError('negative variance')
        inventory[name]={'shape':list(shape),'dtype':dtype,'role':role,
                         'type':'Parameter' if type(t) is torch.nn.Parameter else 'Tensor',
                         'requires_grad':bool(t.requires_grad),'value_sha256':state_sha256({name:t}),
                         'device':'cpu','layout':'strided','stride':list(t.stride()),'storage_offset':0,
                         'storage_bytes':t.numel()*t.element_size(),'contiguous':True,
                         'finite':True,'unaliased':True}
    return inventory


def adapt_legacy(state,spec,identity):
    import torch
    from consendxformer.performance_v2.checkpoint import state_sha256
    require_identity(identity);require_recipe(spec)
    supplied=_attest_raw(state,spec)
    raw_identity=digest({'artifact':identity,'container':type(state).__name__,'entries':supplied})
    adapted={};origins={}
    for name in sorted(state):
        adapted[name]=state[name].detach().clone(memory_format=torch.contiguous_format)
        if (type(adapted[name]) is not torch.Tensor or adapted[name].requires_grad
            or not torch.equal(adapted[name],state[name])
            or state_sha256({name:adapted[name]}) != supplied[name]['value_sha256']):
            raise ValueError('conversion changed supplied values')
        origins[name]='supplied'
    bookkeeping={}
    for name in sorted(MISSING):
        adapted[name]=torch.zeros((),dtype=torch.int64,device='cpu');origins[name]='initialized_bookkeeping'
        bookkeeping[name]={'shape':[],'dtype':'torch.int64','role':'buffer','initialization':'zero',
                           'pretrained_content':False,'value_sha256':state_sha256({name:adapted[name]})}
    if _attest_raw(state,spec)!=supplied: raise ValueError('raw supplied state drift during conversion')
    entries={k:{'shape':list(spec[k][0]),'dtype':spec[k][1],'origin':origins[k],
                'value_sha256':state_sha256({k:v})} for k,v in adapted.items()}
    attestation={'container':type(state).__name__,'raw_entries':267,'raw_supplied':supplied,
                 'raw_identity_sha256':raw_identity,'bookkeeping':bookkeeping,
                 'adapted_entries':320,'adapted_inventory':entries,
                 'adapted_identity_sha256':digest({'raw_identity_sha256':raw_identity,'entries':entries})}
    return adapted,attestation


def verify_receipt_metadata(receipt,spec):
    """No Torch import: shared parent/receiver structural traceability checks."""
    require_identity(receipt['artifact']);require_recipe(spec)
    expected=set(spec)-MISSING; mapping={k:'backbone.'+k for k in spec if k not in EXCLUDED}
    r=receipt
    if (r['schema']!='pv2_imagenet_legacy_export_v2' or r['raw_entries']!=267
        or r['adapted_entries']!=320 or r['backbone_entries']!=318
        or r['excluded']!=sorted(EXCLUDED) or r['mapping']!=mapping
        or r['container'] not in ('dict','OrderedDict')
        or set(r['raw_supplied'])!=expected or set(r['bookkeeping'])!=MISSING
        or set(r['adapted_inventory'])!=set(spec) or set(r['export_origins'])!=set(mapping)
        or r['production_execution_authorized'] is not False or r['training_authorized'] is not False):
        raise ValueError('receipt schema/count/scope drift')
    for k,v in r['raw_supplied'].items():
        shape,dtype,role=spec[k]
        strides=[];elements=1
        for dimension in reversed(shape):strides.insert(0,elements);elements*=dimension
        if (set(v)!={'shape','dtype','role','type','requires_grad','value_sha256','device','layout','stride',
                        'storage_offset','storage_bytes','contiguous','finite','unaliased'}
            or v['shape']!=list(shape) or v['dtype']!=dtype or v['role']!=role
            or v['device']!='cpu' or v['layout']!='strided' or v['stride']!=strides
            or v['storage_offset']!=0 or v['storage_bytes']!=elements*4
            or any(v[field] is not True for field in ('contiguous','finite','unaliased'))
            or v['type'] not in ('Tensor','Parameter') or type(v['requires_grad']) is not bool
            or (role=='buffer' and (v['type']!='Tensor' or v['requires_grad']))):
            raise ValueError('raw attestation layout/type drift')
    for k,v in r['bookkeeping'].items():
        if (set(v)!={'shape','dtype','role','initialization','pretrained_content','value_sha256'}
            or v['shape']!=[] or v['dtype']!='torch.int64' or v['role']!='buffer'
            or v['initialization']!='zero' or v['pretrained_content'] is not False):
            raise ValueError('bookkeeping claim drift')
    for k,v in r['adapted_inventory'].items():
        origin='initialized_bookkeeping' if k in MISSING else 'supplied'
        source=r['bookkeeping'][k] if k in MISSING else r['raw_supplied'][k]
        if v!={'shape':list(spec[k][0]),'dtype':spec[k][1],'origin':origin,'value_sha256':source['value_sha256']}:
            raise ValueError('adapted traceability drift')
        h=v['value_sha256']
        if type(h) is not str or len(h)!=64 or any(c not in '0123456789abcdef' for c in h):
            raise ValueError('tensor digest format')
        if k in mapping and r['export_origins'][k]!=origin:raise ValueError('export origin drift')
    if (r['raw_identity_sha256']!=digest({'artifact':r['artifact'],'container':r['container'],'entries':r['raw_supplied']})
        or r['adapted_identity_sha256']!=digest({'raw_identity_sha256':r['raw_identity_sha256'],'entries':r['adapted_inventory']})
        or r['export_identity_sha256']!=digest({'adapted_identity_sha256':r['adapted_identity_sha256'],
                    'mapping':mapping,'origins':r['export_origins'],'export_sha256':r['export_sha256']})):
        raise ValueError('state identity drift')
