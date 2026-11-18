# Observed account home and proposed directories

Read-only inspection through newly created Jupyter terminal /terminals/29.
Only account uid 1009 (chaidir), its home ancestors and immediate home-child
directory metadata were inspected. No child contents, unrelated users, research
images, credentials or weights were read. No permissions or directories changed.

passwd home, environment HOME and resolved home all equal `/home/chaidir`.
No symlinks occur at `/`, `/home` or `/home/chaidir`.

| Path | Mode | Owner UID:GID | ACL |
| --- | --- | --- | --- |
| / | 0755 | 0:0 root:root | user rwx, group r-x, other r-x |
| /home | 0755 | 0:0 root:root | user rwx, group r-x, other r-x |
| /home/chaidir | 0755 | 1009:1009 chaidir:chaidir | user rwx, group r-x, other r-x |
| /home/chaidir/pv2-metadata-review | 0775 | 1009:1009 chaidir:chaidir | user rwx, group rwx, other r-x |

The ACL outputs show no named or default ACL entries. Home and its ancestors
are not writable by group/other. The relevant existing pv2-metadata-review
directory is group-writable and is not approved as scratch/reports.

Mount: `/`, source `/dev/mapper/ubuntu--vg-ubuntu--lv`, ext4, `rw,relatime`.
At inspection: 459,333,928 total 1-KiB blocks; 338,844,080 used;
99,643,760 available; 78% capacity. Inodes: 29,237,248 total;
1,370,484 used; 27,866,764 free; 5% used. No durability guarantee is inferred.

Immediate directories with mode 0700 were application/configuration locations
(.cache, .config, .docker, .elinks, .gnupg, .mozilla, .nv, .pki, .ssh, snap).
Their contents were not inspected or repurposed. Other ordinary home directories
were 0755/0775. No dedicated private scratch/report pair was identified in this
bounded inspection. This is not an exhaustive recursive search of the home.

Safest inspected parent: `/home/chaidir`. Propose two new dedicated directories:

- `/home/chaidir/.performance-v2-scratch`
- `/home/chaidir/.performance-v2-reports`

Both names were absent from the immediate directory/symlink inventory. Before
any later authorized creation, check absence of files as well; create exclusively
as uid 1009 with umask 077 and mode 0700, then verify owner, ACLs, symlinks and
mount afresh. Do not chmod or repurpose pv2-metadata-review or the audit area.
Neither proposed directory was created. Mode 0700 does not isolate processes
sharing uid 1009; account-use assumptions remain an explicit review condition.

Stage A and frozen B1 preserved. Stage B and GPU pilots paused.
pilot_execution_authorized=false.
