# Performance-v2 bounded input acceptance and byte acquisition

Recorded 2026-09-27 in the isolated `performance-v2` worktree at
`f8278c34d6145170345dce0a518276adf0bb4144`. This records the owner's authorized
archive read-back and official byte acquisition, not training approval.
Stage A remains preserved; Stage B paused; `pilot_execution_authorized=false`.
The [pilot matrix](../configs/performance_v2/pilot_protocol_v1.json) is unchanged.
The [earlier acquisition plan](performance_v2_pretraining_acquisition_plan.md),
[input-audit plan](performance_v2_input_audit_plan.md) and original reports remain
historical evidence. No audit, transform probe or research computation was rerun.

## 1. Accepted input evidence, within archived-report scope

| Identity | SHA-256 |
|---|---|
| `input-audit-20260927T121832477449883Z.zip` | `1dbea55d35db44743e9d5c78f4e40a73d13d6fc8355f250208e65835f09761e7` |
| `artifact_manifest.json` | `d1e68d9c2af5ed480f5b1433efc2db30596e8c9b2ea4e79b176848b9bbc32ff8` |
| Manifest canonical payload-map digest | `cd941f0cc263d20350caed2a752ced66a4c4db0d4cb53883480cda05943bafde` |
| `report.json` | `cb147a14c487aa3487fddc08407aa0e77256e3a5611b667c2c153c76cf3dccff` |
| `input_ledger.json` | `eba6307a6dd715eb84020f1342f6314539996c427a03bd076475e27ddcb7d6ea` |
| `source_hashes.json` | `5285f2bd63723611560c4e681627e8080c80836f45196013e9c0602005b8d2bd` |
| Canonical pre-decode cohort ledger | `74ade0365dc6b7ea5593b4736692956b863b4245e122829e2921b0937663b5b1` |
| Executed input-audit script | `84090d157959a2b4c525484504c1fc14f398d36fb20b58f34a5a6097321d6bed` |
| Earlier diagnostic ZIP | `e5663b68523b345a3b270108e6958e1f49498e3bec14ef6959e9f4d53394e36b` |

Fresh desktop checks verified all five ZIP entries without extraction, raw
payload hashes, canonical manifest digest and exact completion binding. The
cohort digest was independently reconstructed from the ten pre-decode fields
defined by script 55 (sorted by split/nodule; sorted-key compact UTF-8 JSON,
no NaN or trailing newline in the digest). The 950 nodule identities and paths
are unique; 807 training nodules belong to 437 patients and 143 validation
nodules to 84 patients, with no patient intersection. Identities, series,
labels and valid-reader counts match all 18 earlier diagnostic run copies
(17,100 repeated records, not 17,100 independent nodules).

The 972 source references comprise 950 selected image references and 22 other
references. Every image reference matches its ledger hash and expected path.
Five shared full-model seed-42 source-file hashes/sizes match the diagnostic
archive, including completion, manifest, input lock, supervision and identity;
canonical/rendered checksum references match its recorded input fingerprint.
The selected binding remains `phase4-final-s42-c1e650d77585ed9c`.

The server report records all 950 decoded arrays as finite float32 `[64,64]`,
bounded in `[0,1]`, with zero division-by-255 triggers, zero warnings and 950
pre-decode image hash checks. Desktop read-back verified consistency of every
record and split summary; it did not decode or rehash original image bytes.
Its `git_before`/`git_after` reports are equal and clean at server historical
HEAD `7908b8d69a8064240ac7733d683c28489b0077a2`, not the current desktop HEAD.
Source preservation and runtime observations are attributed to that supplied
server report, not independent live-server observation.

The report binds the previously accepted CPU transform report
`3e79f956d99d39e79cd81dd218397f66f8bc4faa71ce07ccb62220ed3a69c894`;
it explicitly did not rerun transforms. The original oblique partial-edge
limitation remains. Decoder source/function and NumPy/OpenCV module/binary
hashes remain in the report/source ledger; these are archived references, not
fresh local verification of university binaries. Full original result bundles
were not verified. Mixed metadata (including TEST identities) was hashed by
the server audit, but TEST images were not decoded; desktop review opened no
TEST images or predictions. Reported model forwards/GPU calls/training are zero.

**Bounded acceptance:** the supplied audit meets the TRAIN/VALIDATION input
audit's archived-report contract. It does not approve CUDA determinism,
pretrained compatibility, all original source bytes, test evaluation or pilots.
Original reports and the earlier diagnostic partial marker remain unchanged.

## 2. Official immutable source and preserved evidence

Retrieved 2026-09-27 UTC; exact per-request start/end times, resolved URLs,
redirect chains, headers, sizes and full hashes are in the external receipt.

| Official repository / reference | Resolved commit |
|---|---|
| `pytorch/vision` `v0.18.0` | `6043bc250768b129e90a5321e318c1d51ee48a5c` |
| `pytorch/pytorch` `v2.3.0` | `97ff6cfd9c86c5c09d7ce775ab64ec5c99230f5d` |
| `BMEII-AI/RadImageNet` observed `main` | `0ce16f7375db4236e646829d1eca61cdb4282133` |

The preserved sources include commit API responses, torchvision `resnet.py`,
`_api.py`, transform `_presets.py`, LICENSE and `models.rst`; PyTorch `hub.py`;
RadImageNet README, LICENSE and example notebook (text only); versioned ResNet50
HTML, ImageNet access-terms HTML and the official RadImageNet Drive landing page.
The [pinned model source](https://github.com/pytorch/vision/blob/6043bc250768b129e90a5321e318c1d51ee48a5c/torchvision/models/resnet.py)
binds the explicit V1 URL. Pins establish source references, not installed-wheel
identity. HTML pages are retrieval snapshots, not immutable upstream revisions.

External evidence root (outside both repositories):
`C:/Users/Lenovo y520/Documents/Playground/performance-v2-acquisition-20260927-authorized/`.
Its `acquisition-receipt.json` SHA-256 is
`d170a6fd6cc23b54a169e405b959d931f4012f86ffb8b8b93de6dcca5b92c046`.
The receipt enumerates all 15 source documents with raw SHA-256s and URLs;
each saved source also has an individual receipt. Retain this evidence directory
with the review package; no permanent public hosting is claimed.
An initial sandbox-network-denied attempt remains separately preserved in
`performance-v2-acquisition-20260927/`; no partial file was reused or overwritten.

## 3. ImageNet V1 acquired, not loaded or validated

- Enum: `ResNet50_Weights.IMAGENET1K_V1`, never `DEFAULT` or V2.
- Requested and resolved URL:
  `https://download.pytorch.org/models/resnet50-0676ba61.pth`.
- HTTP 200, no redirects observed; filename `resnet50-0676ba61.pth`.
- Stored under external `weights/`; size **102,530,333 bytes** matches the
  HTTP Content-Length; full raw SHA-256
  `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`.
- Acquired 2026-09-27 12:30:42.631567–12:30:48.675435 UTC using exclusive file
  creation. No torch import, deserialization, conversion or model call occurred.
- The publisher's eight-hex filename SHA-256 prefix `0676ba61` matches.
  Pinned `hub.py` documents prefix checking; pinned `_api.py` routes enum state
  retrieval through the URL loader. No independent full publisher digest or
  signature was found in these inspected sources. HTTP ETag is recorded but
  is not treated as a full SHA-256 or authentication signature.

The pinned source identifies ImageNet-1K and the V1 resize-256/center-crop-224
recipe; the common v2 direct ROI resize remains a documented deviation, not
verified equivalence. Repository BSD-style terms cover torchvision code.
The [pinned model documentation](https://github.com/pytorch/vision/blob/6043bc250768b129e90a5321e318c1d51ee48a5c/docs/source/models.rst)
warns that pretrained terms may derive from training data and require separate
use-case review. The saved [ImageNet access page](https://www.image-net.org/download.php)
sets noncommercial research/education conditions for database access; that page
does not alone settle all rights in this distributed checkpoint. Byte acquisition
is complete as explicitly authorized, but use/redistribution terms assessment is
not declared complete. No dataset access agreement was submitted or accepted.

## 4. RadImageNet acquisition remains blocked

The [pinned official README](https://github.com/BMEII-AI/RadImageNet/blob/0ce16f7375db4236e646829d1eca61cdb4282133/README.md)
links Drive file `1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR`. The saved landing page
identifies `RadImageNet_pytorch.zip`. This verifies the official reference, not
the archive's contents, immutable release hash or release-specific conditions.
No RadImageNet weight bytes were acquired. Needed before acquisition:

1. An explicit authoritative applicability statement for this weight bundle's
   acquisition/use/redistribution terms (repository MIT code license alone is
   not assumed to resolve weight/data terms).
2. A release-specific link between that bundle and intended `resnet50_torch.pt`
   member, preferably publisher digest/version evidence. The notebook names the
   file but does not authenticate a particular bundle member's bytes.

Further compatibility/release blockers remain: semantic topology and key mapping,
overlap evidence and preprocessing. The pinned example uses OpenCV image decode,
`(image-127.5)*2/255`, resize to 224 and tensor conversion; it does not establish
the common ImageNet mean/std interface. Possible BGR semantics and normalization
must be resolved explicitly. No arm-specific preprocessing is silently adopted.

## 5. Origin is not an overlap exclusion check

ImageNet-1K origin is documented by the enum. RadImageNet's official README
describes medical CT/MRI/ultrasound pretraining and linked/requested-access data.
Neither source provides a checkpoint-specific patient/image identity exclusion
against this project's frozen splits. No pretraining identity comparison has
been performed; unknown overlap remains unknown. Database modality or a dataset
name is not evidence of zero overlap. Needed: release-specific lineage and
available patient/study/image identifiers or a documented authoritative exclusion
statement, with explicit linkage limits. No TEST images/predictions are needed
or authorized by this byte-acquisition review.

## 6. Proposed next CPU-only topology/loading review — not executed

Request a separate bounded authorization after review of this receipt and terms.
Do not run upstream notebooks or automatically download missing artifacts.

1. Verify the full artifact hash and source/receipt hashes before any import or
   deserialize step. Use an isolated offline, resource-bounded process with
   no research data mounted, CUDA hidden before imports, no credentials/network
   access, and an explicitly reviewed runtime. Inventory Python/Torch/torchvision
   versions and installed constructor/source identities against the pinned
   references; a version-string match is insufficient. Review that runtime's
   serialization security status before approval; no installation is authorized.
2. Only after that approval, explicitly request a CPU-only restricted
   state-dict load (`weights_only=True`, `map_location='cpu'`); never permit
   executable object allowlisting, unrestricted pickle or fallback. Hash
   verification establishes byte identity, not serialization safety. If the
   approved runtime cannot safely support it, stop and seek a runtime decision.
3. Construct `resnet50(weights=None)` on CPU without a forward. Inventory stem
   3-channel 7x7 stride-2/padding-3, max-pool, bottleneck [3,4,6,3], stride in
   3x3, groups/base-width/dilation, all convolution/BatchNorm settings and buffers,
   adaptive pool and 2048-to-1000 source head. Require exact key, shape and dtype
   coverage including integer tracking buffers, finite tensors and no extras.
   Export the mapping and semantic topology, not a shape-only assertion.
4. Strictly load the complete source state before explicitly replacing only
   `fc.weight`/`fc.bias` with a separately seeded 2048-to-3 head. Preserve seeds
   42/43/44 and the approved head recipe; record hashes and separate RNG state.
   Require all trainable parameters assigned exactly once to the four approved
   encoder/head weight versus bias/normalization AdamW groups, without performing
   an optimizer step. No permissive missing-key loading or training.
5. Export an external no-overwrite inventory/receipt with artifact hash, source
   pins, mapping, runtime, topology, unloaded/loaded distinctions, failures and
   zero-forward count. A load success is not transform suitability, CUDA
   determinism, overlap clearance or pilot authorization. Synthetic forward
   checks, if later desired, require separately scoped approval.
6. Do not apply this ImageNet procedure speculatively to RadImageNet. Keep it
   blocked until its own bytes/terms are approved, then review its sequential
   backbone key map and semantics explicitly; no conversion or guessed mapping.

The next concrete action is owner review of the acquired V1 receipt, terms
limitations and this CPU-only proposal, plus obtaining the two precise
RadImageNet release/conditions records. No pilots or Stage B work are released.
