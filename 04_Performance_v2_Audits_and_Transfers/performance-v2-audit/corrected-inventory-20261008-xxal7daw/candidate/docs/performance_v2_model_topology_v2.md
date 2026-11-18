# Performance-v2 R50 hybrid v2 — exact proposed topology

Recorded 2026-10-07. Model ID: `performance_v2_r50_hybrid_v2`.
This is a proposed architecture specification, not implemented model code or
approval to integrate weights. `topology_approved=false`,
`pretrained_integration_approved=false`, `pilot_execution_authorized=false`.
The [v2 configuration](../configs/performance_v2/pilot_protocol_v2.json) and
[current protocol](performance_v2_protocol.md) govern the proposed experiment.

The existing `src/consendxformer/phase4_model.py` has a custom three-stage CNN
with channels 32/64/128 and stride 8. Arbitrary ResNet50 tensors do not fit it.
The proposal replaces that encoder with a specifically mapped R50 feature
extractor while retaining the declared Transformer, spatial-attention and
coarse-to-fine fusion operations below. It requires a new implementation;
changing `cnn_channels` in the frozen constructor is insufficient because its
encoder and token-grid calculation still assume the old stride. Stage A code
and trained models stay unchanged. A plain R50+GAP control is neither this
hybrid nor a trained ConsenDxFormer result.

## Tensor graph

All arms consume the same float32 `[B,3,224,224]` replicated-grayscale interface.
RANDOM, IMAGENET and any later eligible RADIMAGENET arm use exactly this graph,
stage selection, projection dimensions, normalization and diagnostic head.
Only declared initial backbone values differ. Objective comparisons retain the
same shared graph; SEPARATE/CAPACITY alone add the explicitly specified reader
head. That extra head is part of the contrast, not a disguised backbone change.

| Node | Exact proposal / output excluding batch dimension |
|---|---|
| Backbone | ResNet50 V1.5 bottlenecks `[3,4,6,3]`, groups 1, base width 64, no dilated-stride replacement, zero-init-residual false |
| Stem | Conv 3→64, 7×7, stride 2, padding 3, no bias; BN; ReLU; max-pool 3×3, stride 2, padding 1, ceil-mode false |
| C1 | `layer2`, 512×28×28, stride 8; layer1 remains part of the backbone |
| C2 | `layer3`, 1024×14×14, stride 16 |
| C3 | `layer4`, 2048×7×7, stride 32; bottleneck downsampling is in the 3×3 convolution |
| Tokens | C3 1×1 Conv 2048→128 with bias, no norm/activation; row-major spatial flatten to 49×128; add learned position table `[1,49,128]`; no class token |
| Transformer | Two pre-norm residual blocks, dimension 128, four heads, MHA bias true, batch-first; LN eps 1e-5; MLP 128→512→128 with exact GELU. Attention, residual and MLP dropout all 0; no extra final norm |
| Native spatial attention | Reshape tokens to 128×7×7; 1×1 Conv 128→1 with bias; spatial softmax over all 49 cells; multiply features by that map, without a compensating scale factor |
| Deep fusion | Independent 1×1 Conv C3 2048→128 with bias; concatenate with attended features; 1×1 Conv 256→128 without bias, BN, ReLU; output 128×7×7 |
| C2 fusion | Bilinear upsample deep features to 14×14 (`align_corners=false`, `antialias=false`); lateral 1×1 Conv 1024→128 without bias, BN, ReLU. Gate = sigmoid(1×1 Conv 256→128 with bias on concatenation). Refine `deep + gate*lateral` with 3×3 Conv 128→128, padding 1, no bias, BN, ReLU |
| C1 fusion | Same operation at 28×28 with lateral input 512; output 128×28×28 |
| Pool | Adaptive spatial average to vector z of dimension 128 |
| Diagnostic output | New linear 128→3 with bias, no hidden layer/dropout; softmax class order benign/indeterminate/malignant |
| Reader output | In SEPARATE-010/CAPACITY-010 only: independently initialized linear 128→3 with bias, no hidden layer/dropout. Discard at inference |

All convolutions use dilation 1, groups 1, zero padding mode and stride 1 unless
explicitly stated otherwise. Backbone residual downsample projections follow the
pinned V1.5 source. Every BN uses eps 1e-5, momentum .1, affine parameters and
tracked statistics; all are trainable and update only during train. Validation
uses eval mode. LayerNorm is affine. ReLU is the ordinary non-leaky operation.
No ordinal head, mask input, reader-input channel, hidden attention loss, TTA,
ensemble or additional projection is present. `lambda_o=lambda_a=0` throughout
the core study; DSA/fusion are still present even though alignment loss is zero.
Restoring losses or adding modules requires a new design, not a core-budget retry.

This retains the existing fusion semantics as a design reference, not bytewise
equivalence to the old full model: encoder, spatial grids, linear heads and zero
dropout differ explicitly. It also supersedes v1's plain-CNN 2048-dimensional
head with a 128-dimensional hybrid head. No forward or tensor shape experiment
has been run to certify this proposal.

## Initialization and optimizer assignment

RANDOM backbone initialization follows the pinned V1.5 source: convolution
Kaiming normal fan-out/ReLU, BN weight one/bias zero, residual zero-init disabled.
An eligible pretrained backbone starts from its original approved source state
for every run, never from a trained screening checkpoint. The source 1000-class
head is not reused for diagnosis. All new shared modules use a separate
`shared_fusion_initialization` stream with the existing v1 SHA-256 derivation;
the original encoder/head/shuffle/augmentation stream names and derivation stay
unchanged. Module initialization order is token projection, position table,
Transformer blocks 0/1, attention projection, deep fusion, C2 fusion, C1 fusion.
Diagnostic and reader heads each use their original distinct stream.

New Conv/Linear weights and their biases use uniform bounds
`[-1/sqrt(fan_in), +1/sqrt(fan_in)]`; MHA's combined QKV matrix uses Xavier
uniform, its in/out projection biases are zero, and its output weight uses the
Linear bound. Norm weights are one and biases zero. Position initialization is
truncated normal, mean 0, std .02, bounds [-2,2]. Exact library implementation,
call order and resulting state hashes require later synthetic verification;
do not assert cross-version seeded-byte equivalence from these distributions.
Source classifier initialization/validation must not consume the streams used
for the new hybrid modules or heads. Adding the auxiliary head must not change
diagnostic initialization, shared modules, batch order or augmentation traces.

Recipe A and the corrected four AdamW groups are retained: encoder LR1e-4,
head LR1e-3, decay1e-4 for Conv/Linear weights, decay0 for biases and normalization
affine parameters. "Encoder" explicitly includes the backbone, Transformer,
attention, token/position and fusion modules; only diagnostic/reader classifiers
belong to head groups. MHA `in_proj_weight` is a linear-projection weight even
though it is not a child `nn.Linear`. The learned `position.position` table is
an explicit zero-decay exception in `encoder_bias_normalization`, at encoder LR.
This named extension is required by the hybrid; it does not change the four
rates/decays or permit a catch-all group. Every trainable parameter must occur
exactly once by identity; buffers are excluded; unknown parameters fail review.
Production grouping and exact initialization remain implementation blockers.

## Pretrained compatibility and security gate

The archived torchvision v0.18 source at commit
`6043bc250768b129e90a5321e318c1d51ee48a5c` is a semantic reference. Its saved
`resnet.py` SHA-256 is
`7499a5059ad7b1a1f8f7543c04b4a13795140fad837c38530b48ff5d9610bf6e`.
This does not approve the old installed runtime or select a current executable
library version. Later supported-runtime/dependency/security records remain
authoritative for their own review scope and are unchanged.

Before any separately authorized integration, require:

1. Source, release/terms, acquisition receipt, raw SHA-256, all-split contamination
   review, exact transforms and supported secure runtime/native inventory.
2. Semantic comparison of the actual installed constructor against this graph,
   including stride, stem/pool, BN, keys/shapes/dtypes/layout/storage and buffers.
3. Offline resource-bounded restricted CPU loading under the existing reviewed
   inspection controls; no unsafe pickle, JIT fallback, object allowlisting,
   network, model forward or guessed runtime repair. The retained no-forward
   inspector is an integration dependency, not authority to load weights now.
4. Validate the complete original ImageNet 2048→1000 source state strictly in
   its source topology, then map only stem/BN/layer1–4 under `backbone.*`.
   Enumerate the exclusion of `fc.weight`/`fc.bias` and every new hybrid key.
   Require complete exact coverage of retained backbone keys; export source
   and loaded-state hashes. No `strict=False` missing-key bypass or
   shape-compatible copy into the old custom CNN is permitted.
5. RadImageNet's actual member mapping, source topology, terms, bytes, possible
   channel-order differences and numerical interface remain unknown. Do not
   extrapolate the ImageNet mapping or silently convert/substitute a mirror.
   If a common interface is not defensible, block it or explicitly change the
   question to checkpoint-plus-preprocessing; it would not isolate initialization.

ImageNet V1 acquisition is already recorded, not repeated or undone. Loaded-state
compatibility, terms/overlap clearance, model construction and synthetic topology
tests are still pending. Stage A closure is preserved; Stage B and GPU pilots
remain paused.
