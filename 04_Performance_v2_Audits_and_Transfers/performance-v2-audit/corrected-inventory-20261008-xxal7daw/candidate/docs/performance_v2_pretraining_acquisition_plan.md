# Performance-v2 pretrained-weight acquisition and compatibility plan

## Later bounded acquisition record — 2026-09-27

The owner subsequently authorized byte-only official ImageNet V1 acquisition.
The [input acceptance and acquisition record](performance_v2_input_acceptance_and_acquisition.md)
records completed acquisition, immutable source pins, source/terms receipts,
remaining blockers and the proposed next CPU-only compatibility gate.
This supersedes acquisition-pending wording below only within that scope:
no weights have been deserialized or compatibility-validated; RadImageNet is
not acquired. Pilot execution remains unauthorized. The following is preserved
as the historical pre-acquisition plan, not a claim about current byte holdings.

## Historical documentation-only review

Recorded 2026-09-27. Documentation review only; no weight bytes were downloaded
or loaded. `pilot_execution_authorized=false`. The approved RANDOM/IMAGENET/
RADIMAGENET x A/B x seeds 42/43/44 matrix in
[the protocol](performance_v2_protocol.md) and
[the specification](../configs/performance_v2/pilot_protocol_v1.json) is unchanged.
Stage A evidence remains immutable and Stage B paused. This plan supplements
[the provenance register](../configs/performance_v2/pretraining_provenance_v1.json);
it does not resolve its missing artifact hashes, release terms or overlap fields.

## Official documentation inspected

The following are fresh read-only web observations. Release-tag references are
documentation/source identities, not verification of the installed wheel or an
immutable source SHA. Moving `main` references must be resolved to a commit and
saved with raw hashes before acquisition approval. No publisher-wide assumption
of compatible or unrestricted weights is made.

| Arm | Exact documentary source and candidate | Architecture / preprocessing | Remaining uncertainty |
|---|---|---|---|
| RANDOM | torchvision `v0.18.0`, `resnet50(weights=None)`; no pretrained file | Common ResNet50 V1.5 and new 2048-to-3 head; proposed v2 replicated-grayscale interface | Installed constructor/source identity, initialization and semantic topology checks pending; pretraining overlap not applicable |
| IMAGENET | torchvision `v0.18.0`, explicit `ResNet50_Weights.IMAGENET1K_V1`; official URL `https://download.pytorch.org/models/resnet50-0676ba61.pth` | V1.5; replace original 1000-class head with separately initialized 3-class head | Full raw SHA-256, exact acquired bytes/date, applicable artifact terms, installed-source match and overlap review pending |
| RADIMAGENET | Official BMEII-AI/RadImageNet repository links PyTorch Google Drive artifact ID `1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR`; example references `resnet50_torch.pt` | Example takes torchvision ResNet50's first nine children, including pooling; numerical interface differs from proposed common normalization | No immutable release/commit or actual member verified; artifact terms, topology, mapping, preprocessing and overlap unresolved; arm remains blocked |

The [tagged torchvision model source](https://github.com/pytorch/vision/blob/v0.18.0/torchvision/models/resnet.py)
binds V1 to that URL; `DEFAULT` selects V2 and must not replace V1. Its bottleneck
downsampling is in the 3x3 convolution; the stem has three input channels and
global adaptive average pooling feeds 2048 features. These source facts do not
prove equality to an installed binary or RadImageNet weights.

The [versioned V1 documentation](https://docs.pytorch.org/vision/0.18/models/generated/torchvision.models.resnet50.html)
specifies bilinear resize to 256, center crop 224, [0,1] scaling and ImageNet
mean `[0.485,0.456,0.406]` / std `[0.229,0.224,0.225]`.
The [tagged transform preset](https://github.com/pytorch/vision/blob/v0.18.0/torchvision/transforms/_presets.py)
defaults resize antialiasing to true. The approved pilot instead resizes the
unchanged ROI directly to 224 and replicates grayscale. This deliberate
development interface is not the stock pretrained inference recipe; the CPU
transform probe cannot establish transfer-learning suitability.

The [official RadImageNet repository](https://github.com/BMEII-AI/RadImageNet)
describes 1.35 million CT/MRI/ultrasound images, 11 anatomies and 165 pathology
labels, links dataset split CSVs and a requested-access dataset, and links the
[PyTorch bundle](https://drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR/view?usp=sharing).
Those landing-page statements are not a checkpoint-specific patient manifest.
The [official example](https://github.com/BMEII-AI/RadImageNet/blob/main/pytorch_example.ipynb)
reads with OpenCV and applies `(image-127.5)*2/255`, then OpenCV resize to 224 and
ToTensor. It does not document the pilot's ImageNet mean/std normalization.
OpenCV color decoding suggests BGR in that example; release-specific channel
semantics remain unverified. Replicated grayscale alone does not remove the
normalization discrepancy. The notebook was read as text, never executed.

## Terms and overlap evidence

The [torchvision v0.18.0 LICENSE](https://github.com/pytorch/vision/blob/v0.18.0/LICENSE)
contains the project's BSD-style copyright and redistribution conditions.
The [RadImageNet repository LICENSE](https://github.com/BMEII-AI/RadImageNet/blob/main/LICENSE)
is MIT. These code licenses are recorded separately from unresolved checkpoint
and underlying-data permissions. Preserve the relevant notices and obtain a
clear release-specific terms record before a weight arm is released; do not
infer that either repository license settles all artifact/data rights.

ImageNet-1K is the V1 dataset designation in the official source. Neither that
designation nor its natural-image purpose constitutes an identity comparison
against the frozen project cohort. For RadImageNet, the linked split CSVs may
support a later review, but were not downloaded or examined here. The current
evidence establishes neither checkpoint's absence of LIDC-IDRI patient/image
overlap. Unknown overlap remains unknown, not verified zero.

The later overlap dossier must identify the exact pretraining release and its
available institution/patient/study/image identifiers, document their linkage
limits, and compare available provenance against all frozen project splits
without consulting test predictions. This is a separate provenance review; the
TRAIN/VALIDATION input audit must not open TEST images. If source identifiers
cannot support the required exclusion, retain a blocked/unknown status for
review rather than asserting independence or silently weakening the gate.

## Proposed acquisition record and compatibility gate

These steps are proposed for later authorization; this task executes none of
them. No automatic downloader or model-loading command is supplied.

1. Fix the official source commit, enum/release, final resolved URL and terms
   document. Save source/terms text with retrieval timestamps and SHA-256s.
   For RadImageNet, obtain a release-specific mapping between the official
   bundle and intended ResNet50 member. A renamed mirror is not a substitute.
2. Acquire only the approved artifact into a separate protected, no-overwrite
   directory. Record original filename, byte count, raw SHA-256, source and
   redirect chain, acquisition time, and any publisher checksum/signature.
   For a bundle record its hash and every selected member hash. The `0676ba61`
   filename component is not a full SHA-256 and is not sufficient evidence.
3. Before later CPU loading, review the serialization format and safe loading
   route. Use the pinned runtime and a state-dict-only loading policy; unexpected
   executable objects, incomplete mappings or unsupported format stop review.
   Do not run an upstream example notebook or silently convert foreign weights.
4. Audit semantic topology as well as dimensions: input stem, bottleneck stride
   location, padding, groups/dilation, pooling, convolution and BatchNorm
   behavior/buffers. Export every original-to-target key and shape/dtype mapping.
   Require exact encoder coverage, including buffers; permit only explicitly
   identified original classifier removal. Hash the loaded encoder state and
   independently initialized diagnostic head. No permissive missing-key load.
5. Keep the proposed common transform frozen while reviewing compatibility.
   The RadImageNet example's different scaling requires evidence that the common
   interface is justified or an explicit owner-reviewed amendment to a
   checkpoint-plus-preprocessing comparison. Neither adjustment is adopted here.
6. Complete overlap/terms review, installed-source verification, synthetic
   topology/initialization tests and the separate CUDA/determinism gate before
   requesting pilot release. A passed CPU import/transform probe does not
   authorize acquisition, establish pretrained compatibility or launch pilots.

Failure retains the original evidence and blocked status; no replacement model,
weight source, recipe, arm-specific preprocessing or favorable retry is selected
from validation performance. All release decisions must remain auditable and
separate from the existing Stage A completion markers.

## Next concrete provenance action

Review this plan together with the pending TRAIN/VALIDATION input-audit script.
For weights, the exact outstanding evidence is: approved official bytes plus full
SHA-256s; immutable source/constructor pin; applicable artifact terms; a documented
all-split overlap assessment; strict topology/state mapping; and a resolution of
RadImageNet's common-input incompatibility. No weight download, load, inference,
test-image inspection or GPU pilot is authorized by this document.
