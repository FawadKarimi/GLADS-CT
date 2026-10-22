"""Static size proof for one frozen HARD/R50/AdamW run; no Torch or file mutation."""
import json,math,copy
from dataclasses import dataclass
GiB=1024**3;MiB=1024**2
@dataclass(frozen=True)
class Tensor:
 dtype:str
 shape:tuple
@dataclass(frozen=True)
class Float:
 pass
WIDTH={'float32':4,'int64':8,'uint8':1}
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def encoded_length(x):
 if isinstance(x,Tensor):
  n=math.prod(x.shape)*WIDTH[x.dtype]
  return len(canonical(['tensor',x.dtype,list(x.shape),'']))+4*((n+2)//3)
 if isinstance(x,Float):return len('["scalar",]')+24
 if x is None or type(x) in (str,bool,int,float):return len(canonical(['scalar',x]))
 if type(x) in (tuple,list):
  return len(canonical(['tuple' if type(x) is tuple else 'list',[]]))+sum(encoded_length(v) for v in x)+max(0,len(x)-1)
 if type(x) is dict:
  # Each pair adds its brackets/comma; exact typed-JSON codec ordering.
  return len(canonical(['dict',[]]))+sum(encoded_length(k)+encoded_length(v)+3 for k,v in x.items())+max(0,len(x)-1)
 raise TypeError(type(x).__name__)
def worst_float_tree(x):
 if type(x) is float:return Float()
 if type(x) is dict:return {k:worst_float_tree(v) for k,v in x.items()}
 if type(x) is list:return [worst_float_tree(v) for v in x]
 if type(x) is tuple:return tuple(worst_float_tree(v) for v in x)
 return x
def estimate(spec):
 from .supervision_admission import admit_contract
 objective=admit_contract(spec['contract'])
 if spec['selection']['max_epochs']!=100 or spec['batch_size']!=16 or spec['drop_last'] is not False or spec['workers']!=0:raise ValueError('frozen recipe drift')
 params=spec['trainable_parameter_inventory'];buffers=spec['buffer_inventory']
 extra=2 if objective=='SEPARATE-010' else 0
 if len(params)!=211+extra or len(buffers)!=174 or any(x['dtype']!='torch.float32' for x in params):raise ValueError('unreviewed inventory')
 readers={x['name']:x for x in params if x['name'].startswith('reader_head.')}
 if objective=='SEPARATE-010':
  if set(readers)!={'reader_head.weight','reader_head.bias'} or readers['reader_head.weight']['shape']!=[3,128] or readers['reader_head.bias']['shape']!=[3] or readers['reader_head.weight']['optimizer_group']!='head_weights' or readers['reader_head.bias']['optimizer_group']!='head_bias_normalization':raise ValueError('exact auxiliary inventory')
 elif readers:raise ValueError('unexpected auxiliary inventory')
 model={x['name']:Tensor(x['dtype'].removeprefix('torch.'),tuple(x['shape'])) for x in params+buffers}
 if spec['model_layout']!={x['name']:{'shape':x['shape'],'dtype':x['dtype']} for x in params+buffers}:raise ValueError('inventory/layout drift')
 if sum(math.prod(x['shape'])*4 for x in params)!=100109072+(1548 if extra else 0) or sum(math.prod(x['shape'])*WIDTH[x['dtype'].removeprefix('torch.')] for x in buffers)!=218064:raise ValueError('fixed tensor-byte inventory drift')
 core=sum(4*((math.prod(v.shape)*WIDTH[v.dtype]+2)//3) for v in model.values())
 opt={};groups=[];next_id=0
 names=['encoder_weights','encoder_bias_normalization','head_weights','head_bias_normalization']
 for name in names:
  rows=[x for x in params if x['optimizer_group']==name];ids=list(range(next_id,next_id+len(rows)));next_id+=len(rows)
  for i,x in zip(ids,rows):
   opt[i]={'step':Tensor('float32',()),'exp_avg':Tensor('float32',tuple(x['shape'])),'exp_avg_sq':Tensor('float32',tuple(x['shape']))}
   core+=8+2*4*((math.prod(x['shape'])*4+2)//3)
  groups.append({'name':name,'params':ids,'lr':Float(),'weight_decay':Float(),'betas':(Float(),Float()),'eps':Float(),'amsgrad':False,'maximize':False,'foreach':False,'capturable':False,'differentiable':False,'fused':False,'decoupled_weight_decay':True})
 # Reviewed CUDA validator bounds the saved CUDA byte vector at65536. The
 # qualified exact Torch CPU generator state is much smaller;65536 is a
 # conservative fixed-runtime allowance, not a new acceptance of larger RNG.
 mt=(3,tuple([2**32-1]*624+[624]),Float())
 streams={'schema':'performance_v2_data_streams_v1','seed':spec['contract']['seed'],'last_epoch':100,'shuffle':mt,'augmentation':mt,'stream_seeds':{n:2**64-1 for n in ('shared','diagnostic_head','reader_head','shuffle','augmentation')}}
 rng={'schema':'pv2_single_cuda_rng_v1','torch_cpu':Tensor('uint8',(65536,)),'python':mt,'numpy':('MT19937',[2**32-1]*624,624,1,Float()),'data_streams':streams,'cuda':{'binding':spec['device_binding'],'state':Tensor('uint8',(65536,))}}
 selection={'history':[{'epoch':i,'macro_f1':Float(),'diagnostic_ce':Float()} for i in range(1,101)],'best_epoch':100,'last_epoch':100,'raw_best':Float(),'without_improvement':100,'stopped':False,'rules':spec['selection']}
 payload={'schema':'performance_v2_checkpoint_v1','spec':worst_float_tree(spec),'components':{'model':model,'optimizer':{'state':opt,'param_groups':groups},'scheduler':{'schema':'recipe_A_warmup_cosine_v1','last_epoch':100,'next_epoch':101,'multiplier':Float()},'rng':rng},'progress':{'epoch':100,'updates':5100,'elapsed_seconds':Float()},'selection':selection}
 encoded=encoded_length(payload)
 # All variable numeric fields were replaced by maximum-width finite64-bit
 # JSON tokens. Add16MiB beyond ALL calculated metadata for format/shape/RNG
 # allowance, then round UP to400MiB; keep the512MiB runtime file limit.
 metadata_allowance=16*MiB
 upper=core+metadata_allowance
 checkpoint_ceiling=400*MiB
 if encoded>upper or upper>checkpoint_ceiling:raise ValueError('checkpoint proof exceeded; no automatic threshold reduction')
 terms={'100_retained_full_checkpoints':100*checkpoint_ceiling,'one_extra_partial_checkpoint_write':checkpoint_ceiling,'all_epoch_plans_batches_metrics_manifests_budget_records':GiB,'stdout_stderr_at_unchanged512MiB_each':GiB,'two_metadata_only_archives_or_partial_archive_writes':2*GiB,'source_staging_delivery_and_misc_temporary_writes':GiB//4,'explicit_free_safety_headroom':4*GiB}
 from .retention import enabled
 sparse=enabled(spec)
 if sparse:
  terms={'six_complete_or_partial_checkpoint_file_caps':6*512*MiB,'all_epoch_plans_batches_metrics_manifests_budget_records':GiB,'stdout_stderr_at_unchanged512MiB_each':GiB,'two_metadata_only_archives_or_partial_archive_writes':2*GiB,'source_staging_delivery_and_misc_temporary_writes':GiB//4,'explicit_free_safety_headroom':4*GiB}
 required=sum(terms.values());threshold=12*GiB if sparse else 48*GiB
 if required>threshold:raise ValueError('storage schedule exceeds proposed gate')
 return {'core_base64_bytes':core,'modeled_worst_typed_checkpoint_bytes':encoded,'metadata_allowance_bytes':metadata_allowance,'checkpoint_conservative_ceiling_bytes':checkpoint_ceiling,'original_runtime_file_limit_bytes':512*MiB,'terms':terms,'derived_total_bytes':required,'derived_total_GiB':required/GiB,'old_threshold_bytes':60*GiB,'new_threshold_bytes':threshold,'new_threshold_GiB':threshold/GiB,'protocol_changed':False,'checkpoint_history_retained':not sparse,'checkpoint_digest_history_retained':True,'retention_amendment':sparse,'checkpoint_bodies_in_packages':False,'no_new_baseline_or_import_execution':True}
def admit(free_bytes,free_inodes,proof,*,metadata_only_packaging=True,quota_formats_disabled=True):
 if not metadata_only_packaging or not quota_formats_disabled:raise ValueError('packaging/quota scope not bound')
 if free_bytes<proof['new_threshold_bytes'] or free_inodes<2000:raise ValueError('insufficient storage/inodes')
 return True


def stage_estimate(specs):
 from .retention import enabled
 from .retention import REPLICATION_POLICY,REPLICATION_RUNS
 old=len(specs)==3 and [s['contract']['objective'] for s in specs]==['LEGACY','SHARED-010','SEPARATE-010'] and all(s['contract']['seed']==42 for s in specs)
 new=len(specs)==4 and [(s['contract']['seed'],s['contract']['objective']) for s in specs]==[(43,'HARD'),(43,'LEGACY'),(44,'HARD'),(44,'LEGACY')] and all(s.get('checkpoint_retention')==REPLICATION_POLICY for s in specs)
 if not (old or new) or not all(enabled(s) for s in specs):raise ValueError('exact retained stage or four-run replication only')
 proofs=[estimate(s) for s in specs]
 terms={'three_attempts_six_file_caps_each':3*6*512*MiB,'three_attempts_complete_epoch_and_budget_metadata':3*GiB,'three_attempts_stdout_stderr':3*GiB,'three_attempts_two_metadata_archives':6*GiB,'shared_source_and_delivery':GiB//4,'shared_free_safety_headroom':4*GiB}
 if new:
  terms={'four_attempts_six_file_caps_each':4*6*512*MiB,'four_attempts_complete_epoch_and_budget_metadata':4*GiB,'four_attempts_stdout_stderr':4*GiB,'four_attempts_two_metadata_archives':8*GiB,'shared_source_and_delivery':GiB//4,'shared_free_safety_headroom':4*GiB}
 total=sum(terms.values());threshold=(35 if new else 27)*GiB
 if total>threshold:raise ValueError('stage proof exceeds fixed gate')
 return {'terms':terms,'derived_total_bytes':total,'derived_total_GiB':total/GiB,'threshold_bytes':threshold,'threshold_GiB':35 if new else 27,'maximum_admitted_attempts':4 if new else 3,'no_automatic_resume_or_retry':True,'per_arm':proofs,'historical_artifacts_not_removed':True,'checkpoint_bodies_in_packages':False}
