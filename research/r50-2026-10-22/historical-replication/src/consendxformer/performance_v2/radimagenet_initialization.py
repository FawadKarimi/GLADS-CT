"""Exact RadImageNet supplied-only fresh initialization; never used on resume."""
from pathlib import Path
from .rad_model_import import import_verified_cpu
from .training_runner import initial_receipt
from .checkpoint import state_sha256
ROOT=Path('/home/chaidir/.performance-v2-radimagenet-receiver-20261013/artifacts')
RECEIPT_SHA='3817d15712462a6a500b7efc8aa8720e201c029e5110e38eeddb523c82e69918'
PLAN_SHA='7809f8ff53150fde8e4273a21c0c4559d1ae6f665f34b22245b6398bc94b3f5c'
EXPECTED={'backbone_state_sha256': '2538142397e1e3c390a0b80b537f698afe4c8928d618853e6a1f6e16cdc52e82', 'kind': 'RADIMAGENET', 'new_modules_state_sha256': '30f6c4054a415bd97bb4d1742d442a64549ee71df4abc0923597abca78f45f88', 'artifact_sha256': '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734', 'transfer_sha256': 'f2976a659bb42c9885fb2421856459df55e7b5c9225c2a57a674a05acbc35643', 'loading_review_sha256': '21c9e45e01bcf9d2540fa7e6f14205f12b02281548efc9fe2e636e5cf28ba442'}
def initialize(model,*,streams,read,expected_initialization=None):
 from .reuse import digest
 if model.seed in (43,44):
  from .supervision_admission import admit_contract
  admit_contract({'initialization':'RADIMAGENET','objective':model.objective,'seed':model.seed})
  if (type(expected_initialization) is not dict or set(expected_initialization)!=set(EXPECTED)
      or any(expected_initialization[k]!=EXPECTED[k] for k in EXPECTED if k!='new_modules_state_sha256')
      or type(expected_initialization['new_modules_state_sha256']) is not str
      or len(expected_initialization['new_modules_state_sha256'])!=64
      or any(c not in '0123456789abcdef' for c in expected_initialization['new_modules_state_sha256'])):
   raise ValueError('native seed-specific initialization binding required before artifact read')
  receipt=read(ROOT/'export-receipt.json',cap=1024**2)
  if digest(receipt)!=RECEIPT_SHA:raise ValueError('Rad receipt before payload')
  raw=read(ROOT/'backbone.typed-json',cap=125682928)
  detail=import_verified_cpu(model,raw,receipt,trusted_receipt_sha=RECEIPT_SHA,trusted_plan_sha=PLAN_SHA,streams=streams)
  observed=initial_receipt(model)
  observed.update(kind='RADIMAGENET',artifact_sha256=EXPECTED['artifact_sha256'],transfer_sha256=EXPECTED['transfer_sha256'],loading_review_sha256=EXPECTED['loading_review_sha256'])
  if observed!=expected_initialization:raise ValueError('seed-specific initial identity drift')
  return observed,detail
 if model.objective!='HARD':
  from .supervision_admission import SUPERVISION_ARMS,validate_optional_head
  if model.objective not in SUPERVISION_ARMS or model.seed!=42:raise PermissionError('only reviewed Rad seed42 initializer')
  validate_optional_head(model)
  if (type(expected_initialization) is not dict or set(expected_initialization)!=set(EXPECTED)
      or any(expected_initialization[k]!=EXPECTED[k] for k in EXPECTED if k!='new_modules_state_sha256')
      or type(expected_initialization['new_modules_state_sha256']) is not str
      or len(expected_initialization['new_modules_state_sha256'])!=64
      or any(c not in '0123456789abcdef' for c in expected_initialization['new_modules_state_sha256'])):
   raise ValueError('independent per-arm initial binding required before artifact read')
 receipt=read(ROOT/'export-receipt.json',cap=1024**2)
 if digest(receipt)!=RECEIPT_SHA:raise ValueError('Rad receipt before payload')
 raw=read(ROOT/'backbone.typed-json',cap=125682928)
 detail=import_verified_cpu(model,raw,receipt,trusted_receipt_sha=RECEIPT_SHA,trusted_plan_sha=PLAN_SHA,streams=streams)
 observed=initial_receipt(model)
 observed.update(kind='RADIMAGENET',artifact_sha256=EXPECTED['artifact_sha256'],transfer_sha256=EXPECTED['transfer_sha256'],loading_review_sha256=EXPECTED['loading_review_sha256'])
 if model.objective=='HARD':
  if expected_initialization is not None and expected_initialization!=EXPECTED:raise ValueError('HARD initializer cannot be relabeled')
  if observed!=EXPECTED or detail['combined_state_sha256']!='1ef89268cb14b1fb8a135abb45b034f26de8d49f8c390bc0dc69121feb119777':raise ValueError('native seed42 initial identity drift')
 else:
  from .supervision_admission import SUPERVISION_ARMS,common_non_backbone_state,validate_optional_head
  if model.objective not in SUPERVISION_ARMS or model.seed!=42:raise PermissionError('only reviewed Rad seed42 initializer')
  validate_optional_head(model)
  if (type(expected_initialization) is not dict or set(expected_initialization)!=set(EXPECTED)
      or any(expected_initialization[k]!=EXPECTED[k] for k in EXPECTED if k!='new_modules_state_sha256')
      or state_sha256(common_non_backbone_state(model))!=EXPECTED['new_modules_state_sha256']
      or observed!=expected_initialization):raise ValueError('bound backbone/common/auxiliary initial identity drift')
  if model.objective!='SEPARATE-010' and detail['combined_state_sha256']!='1ef89268cb14b1fb8a135abb45b034f26de8d49f8c390bc0dc69121feb119777':raise ValueError('native common initial identity drift')
 return observed,detail
