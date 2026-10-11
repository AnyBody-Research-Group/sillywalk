---
title: "sillywalk: Constrained statistical prediction of anthropometry and cyclic motion for AnyBody"
tags:
  - Python
  - biomechanics
  - anthropometry
  - human motion
  - principal component analysis
authors:
  # Author-confirmed membership in the cited 2020 paper's order; final sign-off pending.
  - name: John Rasmussen
    orcid: 0000-0003-3257-5653
    affiliation: 1
  - name: Morten Enemark Lund
    orcid: 0000-0001-9920-4051
    affiliation: 2
  - name: Rasmus Plenge Waagepetersen
    orcid: 0000-0001-6911-0089
    affiliation: 3
affiliations:
  - name: "Department of Materials and Production, Aalborg University, Denmark"
    index: 1
  - name: "Nykredit, Denmark"
    index: 2
  - name: "Department of Mathematical Sciences, Aalborg University, Denmark"
    index: 3
date: 11 October 2026
bibliography: paper.bib
---

- **Author-review draft, not for submission:** final sign-off, research-use evidence, funding and conflict disclosures, and complete AI disclosure remain pending.
- The metadata date is the preparation date; outstanding questions are tracked in `joss-prep/paper-questions.md`.

# Summary

- `sillywalk` uses correlated reference data to estimate missing body measurements and cyclic motion, representing movement with Fourier coefficients.
- It exports predictions as AnyScript inputs for the AnyBody Modeling System [@Damsgaard2006]; statistical preparation does not require AnyBody, but simulation does.

# Statement of need

- The package addresses incomplete subject-specific data by predicting correlated anthropometry and motion together, avoiding independent guesses and repeated manual model editing.
- It combines established methods for correlated data and cyclic motion in a workflow that prepares simulation inputs [@Rasmussen2020; @Skejo2021].
- Users supply representative reference data; the package does not acquire measurements, process motion capture, or solve musculoskeletal dynamics.

# State of the field

- Existing tools cover PCA, simulator orchestration, and musculoskeletal simulation [@Jolliffe2016; @Pedregosa2011; @Lund2019; @Delp2007], but target different parts of the workflow.
- `sillywalk` contributes a domain-specific combination of correlated-feature prediction and AnyBody driver generation, not a new PCA method or simulation engine; authors still need to confirm the historical rationale and closer alternatives.

# Software design

- The design separates dataframe-based statistical prediction from AnyBody-specific signal conversion and templating.
- Prediction uses ordinary PCA and Gaussian assumptions for component scores; it supports partial measurements at prediction time, but does not impute incomplete training data.
- Equality constraints steer predictions through the retained PCA representation, which can limit achievable measurement combinations; inspect residuals because general post-solve feasibility is not checked.
- Fourier coefficients suit cyclic motion and AnyBody drivers, but users must preprocess and align cycles; arbitrary non-periodic motion is outside this representation.
- Custom transformations and saved models have compatibility and trust constraints; the workflow does not enforce physical feasibility or quantify predictive uncertainty.

# Research impact statement

- Tests and an offline synthetic example demonstrate the software workflow without human-subject data or AnyBody, but do not establish scientific validity or research adoption.
- **Author input required:** document an actual research use, including package version, role, and evidence; the cited methodological papers are not evidence of this package's adoption [@Rasmussen2020; @Skejo2021].

# AI usage disclosure

- Authors report that the original core predates generative AI; GitHub Copilot CLI 1.0.94-3, powered by GPT-6.1 Sol (`gpt-6.1-sol`), later assisted with manuscript and submission preparation, code and test review, references, examples, documentation, and build work.
- Repository sources, examples, tests, and publisher metadata were used to check implementation and reference claims; this does not replace human review.
- **Author input required:** disclose other AI use and its scope, complete human review and validation, and confirm who made the core design decisions.

# Acknowledgements

- **Author input required:** identify financial support, grants, non-author contributions, and sponsor involvement, or confirm there was none.

# Conflicts of interest

- Morten Enemark Lund was previously employed by AnyBody Technology and is currently employed by Nykredit.
- **Author input required:** disclose relevant financial relationships, Nykredit's role if any, and other potential conflicts; employment facts alone do not resolve these questions.

# References
