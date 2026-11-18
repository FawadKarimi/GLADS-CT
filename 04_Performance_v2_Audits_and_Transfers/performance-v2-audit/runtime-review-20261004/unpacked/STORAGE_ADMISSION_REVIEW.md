# Storage-admission lifecycle correction — 2026-09-29

Base archive SHA-256:
`a1f9f1568132414ab4cbbc6c97afcf9bcc0050d85f63cb90071b644c5f18ad8e`.

Preparation and mocked testing only. Execution/security/dependency approvals
remain false. Production pins, Stage A and historical reports are unchanged;
Stage B remains paused. This supersedes only the all-roles /var/tmp admission
rule described in the historical activation review.

## Policy and placement

The stable environment checks still run for every role. The separate dedicated
state filesystem check (<=1 GiB capacity, >=768 MiB free) is unchanged. The erroneous
unconditional /var/tmp >=12 GiB test is replaced by a role-aware storage check
called by validate_trust **before a native grant is returned**:

- Bootstrap: observe one statvfs snapshot and the containing filesystem/device;
  require >=12 GiB. Save that observation in the verified grant, then exclusively
  persist `storage-initial.json` after the single-use reservation and before any
  controller launch. Bind it to token, boot, scenario and authorization hash.
- Controller: require the matching saved initial observation, unchanged
  filesystem/device, >=12 GiB immediately before provisioning, and no unexplained
  pre-provision decrease greater than 16 MiB. Existing Coordinator/lifecycle reserve
  and fully-allocated backing checks are not changed.
- Watchdog/helper: use their existing early/full ownership capsule respectively.
  Verify exactly one recorded backing and mount for each of the data (8 GiB) and
  evidence (512 MiB) volumes. Validate those observations against original
  intent/observation pairs and the journal's hash ledger. A pending service-start
  intent is not mistaken for missing volume ownership. Read the ledger again to
  reject a changing snapshot; no retry or speculative exclusion is introduced.
  Reobserve root-owned nonsymlink run-directory/backing inode, device, size and
  allocated extent; exact mount record and loop-to-backing mapping through proc
  and sysfs. No Docker or privileged command is used for these checks.

The backing's original containing filesystem must match the initial filesystem.
Sparse, missing, duplicate, replaced, mismatched or unjournaled allocations fail.
The /var/tmp sticky shared ancestor is not treated like a protected source-document
directory; the fixed run directory and regular backing must themselves be root-owned
and not group/other writable, and symlinks are rejected throughout the path.

Let F0 be initial free bytes, A be the freshly verified total physical allocation,
E = 8 GiB + 512 MiB, and M = 16 MiB. Require E <= A <= E + M. Reobserve free space
after allocation validation and require:

`remaining_free >= max(12 GiB - E - M, F0 - A - M)`.

At minimum initial admission, normal allocation leaves 3.5 GiB; the remaining
floor is 3.5 GiB minus 16 MiB = **3,741,319,168 bytes**. The allowance is an explicit
conservative admission-policy budget for host filesystem/run-directory metadata,
not a measured native overhead or permission for additional data growth. It is
small relative to the remaining reserve; no data grows beyond preallocated files.
Runtime tmpfs consumes memory, and control journals/exports use the separate
bounded state filesystem. Any overhead/other consumption beyond the bound fails
closed and requires investigation, not automatic adjustment. A larger initial
disk does not conceal unrelated consumption: the F0-A-M condition still applies.
These readbacks do not reserve free space against competing processes; later
allocation/space checks remain authoritative and native races remain untested.

Each admitted child stores its check in `<role>-storage-admission.json`, including
verified allocated bytes, required remaining space, actual observation and binding.
These records become part of the existing final evidence manifest. Bootstrap
record/persistence failure prevents a valid child grant. No fallback removes the
storage test.

## Validation and unchanged limits

Focused tests cover minimum initial capacity followed by valid controller,
watchdog and helper admission after 8.5 GiB allocation; insufficient initial
capacity; metadata-budget boundary; unexpected consumption on both minimum and
large disks; inode/device/size/sparse/loop mismatch; absent/duplicate/unjournaled
records; journal corruption; authorization/filesystem drift; one statvfs snapshot;
and record ordering. All inputs and OS results are invented. Windows lacks
os.statvfs, so the test explicitly creates a mocked Linux-only API attribute;
no native filesystem inspection is implied.

See STORAGE_VALIDATION.json and storage-*-tests.log for exact results. The existing
activation tests' fake IO acquired a labeled storage-boundary method; production
has no validation bypass. Original resource, hierarchy, termination, deadline,
persistence, authorization and dependency guards remain in force. No native Linux
acceptance, security clearance or execution authorization is supplied. Existing
host/runtime inventory and subsequent dependency/weight-inspection prerequisites
remain unresolved and separate.
