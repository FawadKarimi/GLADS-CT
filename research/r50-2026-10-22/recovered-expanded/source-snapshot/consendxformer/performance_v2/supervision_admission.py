"""Narrow prespecified RadImageNet seed42 objective release; not authorization."""
SUPERVISION_ARMS = ('LEGACY', 'SHARED-010', 'SEPARATE-010')


def admit_contract(contract):
    objective = contract.get('objective')
    initialization = contract.get('initialization')
    seed = contract.get('seed')
    if objective == 'HARD':
        if initialization not in ('RANDOM', 'IMAGENET', 'RADIMAGENET') or seed not in (42, 43, 44):
            raise PermissionError('existing HARD initialization/seed scope')
        # Exact Rad replication seeds are released for preparation; launch stays gated.
    elif (initialization != 'RADIMAGENET' or not
          ((seed == 42 and objective in SUPERVISION_ARMS) or
           (seed in (43,44) and objective == 'LEGACY'))):
        raise PermissionError('only three reviewed Rad seed42 supervision arms')
    return objective


def validate_optional_head(model):
    from torch import nn
    if model.objective == 'SEPARATE-010':
        if type(model.reader_head) is not nn.Linear or model.reader_head.in_features != 128 or model.reader_head.out_features != 3 or model.reader_head.bias is None:
            raise ValueError('exact separate128-to3 supporting head')
    elif model.reader_head is not None:
        raise ValueError('unexpected supporting head')


def common_non_backbone_state(model):
    return {k: v for k, v in model.state_dict().items()
            if not k.startswith(('backbone.', 'reader_head.'))}


def shared_permission_scope(initialization, objective):
    """Keep per-spec/report permission binding; do not invent an admin grant."""
    admit_contract({'initialization':initialization,'objective':objective,'seed':42})
    return 'single-'+initialization+'-'+objective+'-TRAIN-VAL'
