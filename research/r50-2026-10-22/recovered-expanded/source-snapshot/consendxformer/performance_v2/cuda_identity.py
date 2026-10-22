"""Review-only exact UUID bridge for the inspected Torch 2.3 runtime.

Uses its existing NVML visibility translation, never accepts a missing UUID.
No extra library installation, device selection, fallback or ACL exception.
Native agreement between this translation and CUDA execution remains a gate.
"""
import hashlib
import os
import re
from pathlib import Path
import torch
from consendxformer.performance_v2.training_runner import TrainingOperations

CUDA_SOURCE_SHA256 = '1183ef48a38265aec807bd4486ad75cbb0678374b02fdb5ef8b8a65abad4d788'


def exact_uuid(binding):
    if (binding['index'] != 0 or os.environ.get('CUDA_VISIBLE_DEVICES') != binding['uuid']
            or re.fullmatch(r'GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',binding['uuid']) is None):
        raise ValueError('one full UUID must map to logical cuda:0')
    if (torch.__version__ != '2.3.0+cu121'
            or hashlib.sha256(Path(torch.cuda.__file__).read_bytes()).hexdigest() != CUDA_SOURCE_SHA256):
        raise ValueError('unreviewed CUDA visibility implementation')
    uuids = torch.cuda._raw_device_uuid_nvml()
    ordinal = torch.cuda._get_nvml_device_index(torch.device('cuda:0'))
    if (type(uuids) is not list or not uuids or len(uuids)!=len(set(uuids))
            or type(ordinal) is not int or not 0<=ordinal<len(uuids)
            or uuids[ordinal] != binding['uuid'] or torch.cuda.device_count() != 1):
        raise ValueError('exact NVML/visibility UUID mismatch')
    return uuids[ordinal]


class UuidBoundOperations(TrainingOperations):
    """Same update/state implementation; mandatory additional device checks."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if any(b.device!=self.device or (b.is_floating_point() and b.dtype!=torch.float32)
               for b in self.model.buffers()):
            raise ValueError('all buffers must share the bound FP32 device')

    def check_device(self):
        if self.device.type=='cpu': return super().check_device()
        identity=exact_uuid(self.binding)
        props=torch.cuda.get_device_properties(self.device)
        observed={'uuid':identity,'name':props.name,'total_memory':props.total_memory,
                  'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}
        if any(observed[k]!=self.binding[k] for k in observed):
            raise ValueError('fresh actual GPU/runtime mismatch')

    def restore(self,*args,**kwargs):
        self.check_device()
        super().restore(*args,**kwargs)
        for parameter,state in self.optimizer.state.items():
            if (state['exp_avg'].device!=parameter.device or state['exp_avg_sq'].device!=parameter.device
                    or state['step'].device.type!='cpu'):
                raise ValueError('default AdamW resumed placement mismatch; discard target')
