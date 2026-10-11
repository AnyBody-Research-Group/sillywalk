# External datasets

Basic onboarding uses [synthetic data](tutorial.md) and never downloads a
dataset. This keeps the example independent of service availability, participant
data access and uncertain reuse permissions.

## Shared ANSUR adaptation

The authors state that data they share with this repository is free to use,
including their ANSUR adaptation. Their intended use is to fit an anthropometric
model and reconstruct complete profiles from a subset of measurements.
This records the author's permission statement. It does not assign a formal
license to the original ANSUR data or establish its source edition, population,
adaptation procedure or applicable consent and reuse conditions.

The tracked file
[`tests/test-data/ansur_2026-01-21-21207731455.parquet`](https://github.com/AnyBody-Research-Group/sillywalk/blob/main/tests/test-data/ansur_2026-01-21-21207731455.parquet)
contains 3,982 rows and 182 columns. Its schema includes `Height`, `Weight`,
AnyBody anthropometric dimensions, original-style measurement fields, and
subject/trial and revision metadata. These are inspected table dimensions,
not an independently verified count of participants or a data-quality
assessment.

Keep `SubjectID`, `TrialID`, `TrialType`, `Dataset`, `AMMR_git_hash` and
`ANSUR_git_hash` outside the fitted measurement matrix. Numeric identifiers
such as `SUBJECT_NUMBER` and `index`, acquisition fields and encoded categories
such as `GENDER` and `HANDNESS_WRITING` are not continuous anthropometric
measurements merely because they have numeric dtypes.

Choose an explicit set of documented continuous measurements rather than
fitting every numeric column. The file has both `Height` and `STATURE`, and
`Weight` and `WEIGHT`. Confirm their definitions and units before treating
them as equivalents. Revision-hash fields can help trace an adaptation but
do not replace a source citation or preprocessing script.

This adaptation supplies anthropometric fields, not a complete cyclic-motion
coefficient set. Its use does not require AnyBody unless the resulting
parameters are exported and loaded in a model. A conditioned PCA profile is
not guaranteed to remain within the observed measurement ranges or training
population. Assess held-out errors and physical plausibility before drawing
research conclusions.

The original source/version, data dictionary, adaptation script and explicit
reuse terms still need to be recorded. Until those are documented, the
[synthetic example](tutorial.md) remains the reproducible onboarding workflow
and no measured-data accuracy claim is made.

## Historical remote example

An earlier README linked this Parquet file:

```text
https://anybodydatasets.blob.core.windows.net/sillywalk/kso-running/kso-running-fourier-2025-12-28-0.parquet
```

The earlier displayed table had 114 rows and 1,317 columns. That table is not
evidence of 114 independent subjects. Rows could represent repeated trials,
cycles or participant summaries. The number of unique people is unverified.

The source study, participant identifiers, preprocessing and exact reuse terms
for this historical remote file have not been established in this
documentation. The author's general permission statement about shared data
does not establish which source/version this URL contains or replace its
provenance record. The MIT software license does not automatically cover
research datasets.

The URL is retained as a historical pointer, not a recommended dataset,
reproducibility guarantee or confirmed source of research evidence. The offline
example does not use it, and these instructions do not require accessing it.

## Before using a measured dataset

Confirm a stable source and version, source publication or acquisition protocol,
reuse license and any access or ethical restrictions. Determine what each row
represents and how repeated observations map to participants. Establish column
definitions, units, coordinate conventions, cycle normalization and the AnyBody/
AMMR versions used during processing.

For a paper or archived example, cite the dataset independently of the software.
Record a checksum and preprocessing script when permitted. Use subject-level
train/test separation if the evaluation concerns new participants.

Author questions about these missing facts are tracked in the local preparation
file
[joss-prep/documentation-questions.md](https://github.com/AnyBody-Research-Group/sillywalk/blob/main/joss-prep/documentation-questions.md).
Do not replace missing provenance with an inference from a filename or row
count.

The supplied 2021
[_Running in circles_ paper](https://doi.org/10.1016/j.jbiomech.2020.110187)
is methodological background. No verified link from that paper to the
historical Parquet file or the shared ANSUR adaptation has been established.
