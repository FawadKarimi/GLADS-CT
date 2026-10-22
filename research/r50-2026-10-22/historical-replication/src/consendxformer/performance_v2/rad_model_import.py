"""Inactive RadImageNet CPU import addon for the frozen native R50 candidate.

Requires separately pinned export/receipt/plan and native source admission.
No filenames, downloads, pickle or model forward are used by this interface.
"""
import torch
from .rad_export_codec import verify_export
from consendxformer.performance_v2.checkpoint import state_sha256
from consendxformer.performance_v2.imagenet_mapping import r50_spec
from consendxformer.performance_v2.streams import capture_cpu_rng
def import_verified_cpu(model,raw,receipt_bytes,*,trusted_receipt_sha,trusted_plan_sha,streams,resume=False):
 if resume:raise PermissionError('resume must use bounded checkpoint restoration; no pretrained reapplication')
 if getattr(model,'_pv2_pretrained_applied',False) or getattr(model,'_pv2_initialization_failed',False):raise ValueError('one fresh initialization only')
 _,spec=r50_spec()
 state,receipt=verify_export(raw,receipt_bytes,trusted_receipt_sha=trusted_receipt_sha,trusted_plan_sha=trusted_plan_sha,spec=spec)
 # Reuse the accepted frozen Hybrid/ResNet/Bottleneck/stride/BN/topology gate.
 from consendxformer.performance_v2.imagenet_initialization import verify_target
 from .supervision_admission import SUPERVISION_ARMS,validate_optional_head
 verify_target(model,admitted_objectives=('HARD',)+SUPERVISION_ARMS)
 validate_optional_head(model)
 target=model.state_dict();mapping=receipt['mapping']
 if len(mapping)!=318 or {k for k in target if k.startswith('backbone.')}!=set(mapping.values()):raise ValueError('complete318 target coverage')
 if len({t.untyped_storage().data_ptr() for t in target.values()})!=len(target):raise ValueError('target storage alias')
 for source,destination in mapping.items():
  t,v=target[destination],state[source]
  if t.device.type!='cpu' or t.shape!=v.shape or t.dtype!=v.dtype:raise ValueError('exact CPU target layout')
 new={k:v.detach().clone() for k,v in target.items() if not k.startswith('backbone.')}
 modes={k:m.training for k,m in model.named_modules()};flags={k:v.requires_grad for k,v in model.named_parameters()}
 before=capture_cpu_rng(streams)
 try:
  with torch.no_grad():
   for source,destination in mapping.items():target[destination].copy_(state[source])
  after=model.state_dict();rows=[]
  for source,destination in sorted(mapping.items()):
   value_hash=state_sha256({source:after[destination]})
   if not torch.equal(after[destination],state[source]) or value_hash!=receipt['attestations'][source]['canonical_named_state_sha256']:raise ValueError('post-copy per-entry value drift')
   rows.append({'source':source,'raw_source':receipt['attestations'][source]['raw_name'],'target':destination,'origin':'supplied','equal':True,'value_sha256':value_hash})
  if set(after)!=set(target) or any(not torch.equal(after[k],v) for k,v in new.items()) or modes!={k:m.training for k,m in model.named_modules()} or flags!={k:v.requires_grad for k,v in model.named_parameters()}:raise ValueError('non-backbone/mode/gradient drift')
  if state_sha256(capture_cpu_rng(streams))!=state_sha256(before):raise ValueError('RNG stream drift')
  model._pv2_pretrained_applied=True
  return {'verified_entries':rows,'verified_count':318,'supplied_count':318,'initialized_bookkeeping_count':0,'non_backbone_unchanged':True,'rng_unchanged':True,'combined_state_sha256':state_sha256(dict(after)),'training_authorized':False}
 except BaseException:
  model._pv2_initialization_failed=True
  raise
