"""Prepared RadImageNet-only non-pickle conversion; real execution inactive."""
import collections,hashlib,json
import torch
from consendxformer.performance_v2.checkpoint import encode_state,decode_state,state_sha256
ARTIFACT={'bytes':94345733,'sha256':'08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734'}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def mapping(spec):
 stages={'conv1':'0','bn1':'1','layer1':'4','layer2':'5','layer3':'6','layer4':'7'}
 canonical={k:v for k,v in spec.items() if k not in ('fc.weight','fc.bias')}
 return {'backbone.'+stages[k.split('.')[0]]+'.'+k.split('.',1)[1]:k for k in canonical}
def validate(t,name,spec,seen):
 shape,dtype,_=spec[name]
 if (type(t) is not torch.Tensor or t.device.type!='cpu' or t.layout!=torch.strided
     or list(t.shape)!=list(shape) or str(t.dtype)!=dtype or t.requires_grad
     or t.is_conj() or t.is_neg() or not t.is_contiguous() or t.storage_offset()!=0
     or t.untyped_storage().nbytes()!=t.numel()*t.element_size() or not bool(torch.isfinite(t).all())):
  raise ValueError('strict tensor layout/value gate: '+name)
 ptr=t.untyped_storage().data_ptr()
 if ptr in seen:raise ValueError('source/export storage alias')
 seen.add(ptr)
 if name.endswith(('running_var','num_batches_tracked')) and bool((t<0).any()):raise ValueError('negative BatchNorm buffer')
def value_hash(t):
 # Raw value digest independent of key renaming. Source spec admits float32/int64.
 return sha(t.reshape(-1).view(torch.uint8).numpy().tobytes())
def export_state(state,spec,plan_sha):
 m=mapping(spec)
 if type(state) not in (dict,collections.OrderedDict) or set(state)!=set(m):raise ValueError('exact observed raw keys only')
 seen=set();canonical={};attestations={}
 for raw,name in sorted(m.items()):
  t=state[raw];validate(t,name,spec,seen)
  # No Parameters, synthesized counters, classifier removal or value conversion.
  canonical[name]=t
  attestations[name]={'raw_name':raw,'origin':'supplied','shape':list(t.shape),'dtype':str(t.dtype),
   'raw_named_state_sha256':state_sha256({raw:t}),'canonical_named_state_sha256':state_sha256({name:t}),'value_sha256':value_hash(t)}
 raw=encode_state(canonical)
 receipt={'schema':'pv2_radimagenet_export_v1','artifact':ARTIFACT,'operation_plan_sha256':plan_sha,
  'raw_entries':len(state),'export_entries':len(canonical),'source_state_sha256':state_sha256(dict(state)),
  'canonical_state_sha256':state_sha256(canonical),'raw_to_canonical':m,
  'mapping':{name:'backbone.'+name for name in m.values()},'attestations':attestations,
  'initialized_bookkeeping':[],'excluded_raw_entries':[],'export_bytes':len(raw),'export_sha256':sha(raw),
  'training_authorized':False,'production_authorized':False}
 return raw,(json.dumps(receipt,sort_keys=True,separators=(',',':'))+'\n').encode()
def verify_export(raw,receipt_raw,*,trusted_receipt_sha,trusted_plan_sha,spec):
 if type(receipt_raw) is not bytes or len(receipt_raw)>1024**2 or sha(receipt_raw)!=trusted_receipt_sha:raise ValueError('receipt hash before parse')
 r=json.loads(receipt_raw);m=mapping(spec);names=set(m.values())
 expected_fields={'schema','artifact','operation_plan_sha256','raw_entries','export_entries','source_state_sha256','canonical_state_sha256','raw_to_canonical','mapping','attestations','initialized_bookkeeping','excluded_raw_entries','export_bytes','export_sha256','training_authorized','production_authorized'}
 if (set(r)!=expected_fields or r['schema']!='pv2_radimagenet_export_v1' or r['artifact']!=ARTIFACT
  or r['operation_plan_sha256']!=trusted_plan_sha or r['raw_to_canonical']!=m
  or r['mapping']!={n:'backbone.'+n for n in names} or r['raw_entries']!=len(m) or r['export_entries']!=len(m)
  or set(r['attestations'])!=names or r['initialized_bookkeeping']!=[] or r['excluded_raw_entries']!=[]
  or r['training_authorized'] is not False or r['production_authorized'] is not False):raise ValueError('exact artifact/mapping/origin/plan')
 if type(raw) is not bytes or len(raw)>512*1024**2 or len(raw)!=r['export_bytes'] or sha(raw)!=r['export_sha256']:raise ValueError('export hash before decoding')
 state=decode_state(raw,expected_sha256=r['export_sha256'])
 if type(state) is not dict or set(state)!=names:raise ValueError('decoded complete coverage')
 seen=set();reconstructed={}
 for raw_name,name in sorted(m.items()):
  t=state[name];validate(t,name,spec,seen);a=r['attestations'][name]
  expected={'raw_name':raw_name,'origin':'supplied','shape':list(t.shape),'dtype':str(t.dtype),
   'raw_named_state_sha256':state_sha256({raw_name:t}),'canonical_named_state_sha256':state_sha256({name:t}),'value_sha256':value_hash(t)}
  if a!=expected:raise ValueError('per-entry supplied value/origin drift')
  reconstructed[raw_name]=t
 if state_sha256(state)!=r['canonical_state_sha256'] or state_sha256(reconstructed)!=r['source_state_sha256']:raise ValueError('raw/canonical identity drift')
 return state,r
