# Interpretation of retained results

Original initialization, original supervision/replication, expanded replication and post-hoc ensembles answer different questions and must not be pooled into one benchmark.

- **Original 807-nodule cohort**: RadImageNet/HARD remains the seed-42 validation nomination. Historical TEST favors ImageNet in aggregate Macro-F1, but TEST cannot retroactively choose initialization. Original HARD/LEGACY mean validation Macro-F1 is about 70.00%/70.34% across three seeds, with mixed paired differences.
- **Expanded 1,297-nodule cohort**: HARD 68.99 ± 1.93% versus LEGACY 66.27 ± 1.40% validation Macro-F1 (mean ± sample standard deviation). All three paired LEGACY values are lower; every expanded HARD/LEGACY result is lower than its corresponding original result. Mixed targets raise mean malignant recall while lowering indeterminate recall/F1. Report all seeds and CE/class trade-offs; do not choose the best seed.
- **Exploratory ensembles**: D achieves 111/143 correct (77.62%) and Macro-F1 73.99% after searching the same validation labels. E and C trade benign accuracy for indeterminate F1. These do not establish independent-patient performance, calibrated probabilities or a clinical referral policy. Recalculation of saved confusion matrices does not prove logit replay or provenance.

CSVs in `results/expanded1297` are exact copies of retained reporting records. JSON extracts identify their parent records by full SHA-256; they are transparent aggregate extracts, not original complete evidence archives. The verifier checks saved matrices and class metrics; diagnostic/weighted CE cannot be recovered from a confusion matrix.

Cross-paper numerical comparisons are descriptive only because label rules, cohorts, splits, inputs and selection differ. No synthetic capacity test, slice-leaking benchmark or selective subset score is substituted for patient-separated performance in this release.
