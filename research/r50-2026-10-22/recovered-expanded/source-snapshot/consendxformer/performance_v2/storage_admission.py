"""Exact-account read-only quota and derived storage admission for Rad42 only."""
import os,ctypes,subprocess,json
from .storage_bound import estimate,admit
def check(spec,path):
 from .supervision_admission import admit_contract
 if spec['contract']['initialization']!='RADIMAGENET':raise ValueError('Rad storage scope')
 admit_contract(spec['contract'])
 source=subprocess.run(['findmnt','-n','-o','SOURCE','-T',str(path)],capture_output=True,check=True,timeout=5).stdout.strip()
 if not source.startswith(b'/dev/') or b'\n' in source:raise ValueError('reviewed local filesystem source')
 libc=ctypes.CDLL(None,use_errno=True);libc.quotactl.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_void_p];libc.quotactl.restype=ctypes.c_int
 quota={}
 for label,kind in (('user',0),('group',1),('project',2)):
  fmt=ctypes.c_uint32();ctypes.set_errno(0)
  result=libc.quotactl(ctypes.c_int((0x800004<<8)|kind),source,0,ctypes.byref(fmt));err=ctypes.get_errno()
  quota[label]={'return':result,'errno':err}
  if result!=-1 or err!=3:raise ValueError('quota state differs; exact limits must be reconciled')
 v=os.statvfs(path);proof=estimate(spec)
 admit(v.f_bavail*v.f_frsize,v.f_favail,proof,quota_formats_disabled=True,metadata_only_packaging=True)
 return {'available_bytes':v.f_bavail*v.f_frsize,'available_inodes':v.f_favail,'threshold_bytes':proof['new_threshold_bytes'],'derived_total_bytes':proof['derived_total_bytes'],'quota_format_observations':quota,'checkpoint_body_history_complete':proof['checkpoint_history_retained'],'epoch_metric_history_preserved':True,'selection_history_preserved':True,'checkpoint_hash_history_preserved':True,'checkpoint_retention_policy':'best_latest' if proof['retention_amendment'] else 'full_history'}


def check_stage(specs,path):
 from .storage_bound import stage_estimate
 proof=stage_estimate(specs)
 observed=check(specs[0],path)
 if observed['available_bytes']<proof['threshold_bytes'] or observed['available_inodes']<2000:raise ValueError('insufficient derived aggregate stage storage')
 return dict(observed,stage_storage_proof=proof)
