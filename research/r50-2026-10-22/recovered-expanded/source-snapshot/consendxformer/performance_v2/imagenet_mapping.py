"""Strict, non-loading ImageNet R50 verification preparation.

Only accepts already supplied CPU tensors. No path, deserialize, download,
load_state_dict, forward, optimizer or approval API exists here.
"""
from .reuse import canonical, digest

WEIGHT_SHA = '0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a'
RECEIPT_SHA = 'd170a6fd6cc23b54a169e405b959d931f4012f86ffb8b8b93de6dcca5b92c046'
SOURCE_SHA = '7499a5059ad7b1a1f8f7543c04b4a13795140fad837c38530b48ff5d9610bf6e'


def r50_spec():
    """Independent leaf-module/state recipe for torchvision V1.5 [3,4,6,3]."""
    leaves, state = {}, {}

    def conv(name, cin, cout, kernel, stride=1, padding=0):
        leaves[name] = ('Conv2d', cin, cout, (kernel, kernel), (stride, stride),
                        (padding, padding), (1, 1), 1, 'zeros', False)
        state[name + '.weight'] = ((cout, cin, kernel, kernel), 'torch.float32', 'parameter')

    def bn(name, channels):
        leaves[name] = ('BatchNorm2d', channels, 1e-5, .1, True, True)
        for suffix in ('weight', 'bias', 'running_mean', 'running_var'):
            state[name + '.' + suffix] = ((channels,), 'torch.float32',
                                         'parameter' if suffix in ('weight', 'bias') else 'buffer')
        state[name + '.num_batches_tracked'] = ((), 'torch.int64', 'buffer')

    conv('conv1', 3, 64, 7, 2, 3); bn('bn1', 64)
    leaves['relu'] = ('ReLU', True)
    leaves['maxpool'] = ('MaxPool2d', 3, 2, 1, 1, False, False)
    cin = 64
    for stage, depth in enumerate((3, 4, 6, 3), 1):
        width = 64 * 2 ** (stage - 1)
        for block in range(depth):
            name = f'layer{stage}.{block}'
            stride = 2 if stage > 1 and block == 0 else 1
            conv(name + '.conv1', cin, width, 1); bn(name + '.bn1', width)
            conv(name + '.conv2', width, width, 3, stride, 1); bn(name + '.bn2', width)
            conv(name + '.conv3', width, 4 * width, 1); bn(name + '.bn3', 4 * width)
            leaves[name + '.relu'] = ('ReLU', True)
            if stride != 1 or cin != 4 * width:
                conv(name + '.downsample.0', cin, 4 * width, 1, stride)
                bn(name + '.downsample.1', 4 * width)
            cin = 4 * width
    leaves['avgpool'] = ('AdaptiveAvgPool2d', (1, 1))
    leaves['fc'] = ('Linear', 2048, 1000, True)
    state['fc.weight'] = ((1000, 2048), 'torch.float32', 'parameter')
    state['fc.bias'] = ((1000,), 'torch.float32', 'parameter')
    return leaves, state


def semantic_inventory(model):
    import torch.nn as nn
    result = {}
    for name, m in model.named_modules():
        if not name or list(m.children()):
            continue
        if type(m) is nn.Conv2d:
            row = ('Conv2d', m.in_channels, m.out_channels, m.kernel_size, m.stride,
                   m.padding, m.dilation, m.groups, m.padding_mode, m.bias is not None)
        elif type(m) is nn.BatchNorm2d:
            row = ('BatchNorm2d', m.num_features, m.eps, m.momentum, m.affine, m.track_running_stats)
        elif type(m) is nn.ReLU:
            row = ('ReLU', m.inplace)
        elif type(m) is nn.MaxPool2d:
            row = ('MaxPool2d', m.kernel_size, m.stride, m.padding, m.dilation, m.return_indices, m.ceil_mode)
        elif type(m) is nn.AdaptiveAvgPool2d:
            row = ('AdaptiveAvgPool2d', m.output_size)
        elif type(m) is nn.Linear:
            row = ('Linear', m.in_features, m.out_features, m.bias is not None)
        elif type(m) is nn.Identity:
            row = ('Identity',)
        else:
            raise ValueError('unknown R50 leaf: ' + name)
        result[name] = row
    return result


def verify_mapping(source_state, source_model, hybrid):
    """Verify supplied state without loading it; emit exact retained/excluded keys.

    Runtime source/forward implementation identity and safe restricted loading
    must be reviewed separately BEFORE real tensors may reach this function.
    A returned report explicitly does not certify a successful real-weight load.
    """
    import torch
    from torchvision.models.resnet import ResNet, Bottleneck
    from .model import MODEL_ID, R50Hybrid
    from .checkpoint import state_sha256
    leaves, spec = r50_spec()
    if type(source_model) is not ResNet or semantic_inventory(source_model) != leaves:
        raise ValueError('source semantic topology mismatch')
    retained_leaves = dict(leaves, fc=('Identity',), avgpool=('Identity',))
    if type(hybrid) is not R50Hybrid or hybrid.model_id != MODEL_ID or semantic_inventory(hybrid.backbone) != retained_leaves:
        raise ValueError('reviewed hybrid backbone required; no plain R50 control')
    for model in (source_model, hybrid.backbone):
        if type(model) is not ResNet or model.groups != 1 or model.base_width != 64 or model.dilation != 1:
            raise ValueError('source/target R50 container semantics mismatch')
        for stage, depth in enumerate((3, 4, 6, 3), 1):
            blocks = getattr(model, f'layer{stage}')
            if type(blocks) is not torch.nn.Sequential or len(blocks) != depth:
                raise ValueError('R50 stage container mismatch')
            for number, block in enumerate(blocks):
                if (type(block) is not Bottleneck or block.expansion != 4
                        or block.stride != (2 if stage > 1 and number == 0 else 1)):
                    raise ValueError('R50 bottleneck container mismatch')
        if any('forward' in module.__dict__ or module._forward_hooks or module._forward_pre_hooks
               for module in model.modules()):
            raise ValueError('unreviewed R50 forward override/hooks')
    parameters = dict(source_model.named_parameters(remove_duplicate=False))
    buffers = dict(source_model.named_buffers(remove_duplicate=False))
    if (set(parameters) & set(buffers) or set(parameters) | set(buffers) != set(spec)
            or len({id(x) for x in list(parameters.values()) + list(buffers.values())}) != len(spec)
            or any(('parameter' if k in parameters else 'buffer') != v[2] for k, v in spec.items())):
        raise ValueError('source parameter/buffer inventory mismatch or aliases')
    if type(source_state) not in (dict, __import__('collections').OrderedDict) or set(source_state) != set(spec):
        raise ValueError('exact full source keys including 1000-class head required')
    storages = set()
    for key, (shape, dtype, _) in spec.items():
        tensor = source_state[key]
        if (type(tensor) is not torch.Tensor or tensor.device.type != 'cpu'
                or tuple(tensor.shape) != shape or str(tensor.dtype) != dtype
                or tensor.layout != torch.strided or not tensor.is_contiguous()
                or tensor.storage_offset() != 0 or tensor.requires_grad
                or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()
                or not torch.isfinite(tensor).all()):
            raise ValueError('invalid full source tensor: ' + key)
        ptr = tensor.untyped_storage().data_ptr()
        if ptr in storages:
            raise ValueError('aliased source storage: ' + key)
        storages.add(ptr)
        if (key.endswith('running_var') or key.endswith('num_batches_tracked')) and (tensor < 0).any():
            raise ValueError('invalid normalization buffer: ' + key)
    mapping = {k: 'backbone.' + k for k in spec if k not in ('fc.weight', 'fc.bias')}
    target = hybrid.state_dict()
    if {k for k in target if k.startswith('backbone.')} != set(mapping.values()):
        raise ValueError('exact retained target coverage required')
    for key, mapped in mapping.items():
        if (target[mapped].device.type != 'cpu' or target[mapped].shape != source_state[key].shape
                or target[mapped].dtype != source_state[key].dtype):
            raise ValueError('target state shape/dtype mismatch: ' + mapped)
    new_keys = sorted(set(target) - set(mapping.values()))
    return {'schema': 'pv2_imagenet_mapping_review_v1', 'source_state_sha256': state_sha256(source_state),
            'source_semantics_sha256': digest(canonical(leaves)), 'mapping': mapping,
            'excluded_source_keys': ['fc.bias', 'fc.weight'], 'new_hybrid_keys': new_keys,
            'new_hybrid_state_sha256': state_sha256({k: target[k] for k in new_keys}),
            'expected_acquired_file_sha256': WEIGHT_SHA, 'acquisition_receipt_sha256': RECEIPT_SHA,
            'real_weight_loaded': False, 'load_performed': False,
            'production_execution_authorized': False}
