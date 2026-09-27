# WIFICSI_RX_ALIGNMENT

Multi-receiver WiFi CSI research workspace for studying **receiver-shared and receiver-specific structure** in synchronized wireless sensing.

> **Status:** repository preparation in progress.  
> The research code, sanitized configurations, reproducibility scripts, and releasable data assets will be added progressively after internal cleanup and privacy review.

## Overview

WiFi Channel State Information (CSI) collected by multiple receivers provides synchronized but heterogeneous views of the same physical event. This project studies how much sensing information is:

- shared across synchronized receivers,
- specific to individual receiver views,
- preserved or lost by receiver-invariant representations,
- distributed across frequency regions,
- and useful under acquisition or deployment shift.

The project emphasizes **paired multi-receiver analysis**: receiver observations are aligned by the same physical event rather than treated as unrelated samples.

The current research workflow contains two complementary levels:

1. **Mechanism analysis**  
   Receiver-identification probes, true-vs-shuffled pairing controls, shared/private representation analysis, and frequency-structure diagnostics.

2. **Frozen evaluation policies**  
   Task-specific and paired-shared branches are evaluated under fixed protocols, including strict sample-first splitting and cross-batch/domain tests.

## Repository Goals

This repository will serve as the long-term open-source home for the project. The planned release includes:

- privacy-safe source code for the frozen experimental pipelines;
- preprocessing and synchronized multi-receiver data handling;
- training and evaluation scripts;
- experiment configurations and seed definitions;
- paired-RX representation and frequency-analysis utilities;
- baseline implementations/adaptations used in the study;
- aggregation, statistics, and figure-generation scripts;
- reproducibility documentation;
- sanitized metadata and, where permitted, anonymized research data.

Raw internal identifiers, private filesystem paths, operator aliases, and other non-public acquisition metadata will not be released.

## Planned Repository Structure

The repository will be organized approximately as follows:

```text
WIFICSI_RX_ALIGNMENT/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   ├── training/
│   └── evaluation/
├── src/
│   └── rxalign/
│       ├── data/
│       ├── models/
│       ├── alignment/
│       ├── experts/
│       ├── metrics/
│       └── utils/
├── scripts/
│   ├── prepare_data/
│   ├── train/
│   ├── evaluate/
│   └── reproduce/
├── analysis/
│   ├── receiver_identity/
│   ├── pairing_controls/
│   ├── frequency_structure/
│   └── cross_domain/
├── figures/
│   ├── data/
│   └── scripts/
├── docs/
│   ├── reproducibility.md
│   ├── data_format.md
│   └── experiment_protocol.md
└── data/
    └── README.md
```

The final public structure may be simplified during release cleanup.

## Work Plan

Open-sourcing will be completed in stages.

### Stage 1 — Source cleanup

- separate publication-facing code from internal research utilities;
- remove private paths, local aliases, and acquisition-specific identifiers;
- retain exact frozen model and evaluation semantics;
- verify configuration and dependency consistency.

### Stage 2 — Reproducibility package

- provide environment/dependency instructions;
- provide deterministic configuration and seed definitions;
- provide data preprocessing and cache-generation scripts;
- provide one-command or compact reproduction entry points;
- document expected outputs and evaluation metrics.

### Stage 3 — Analysis and figure reproduction

- release scripts for receiver-ID analysis;
- release true-vs-shuffled pairing controls;
- release frequency-structure analysis;
- release cross-batch/domain evaluation;
- release data tables and scripts used to regenerate public figures.

### Stage 4 — Data release

Where privacy, consent, storage, and redistribution constraints allow, anonymized research data and metadata will be released or linked from this repository.

Before release, all data will undergo:

- identifier removal,
- path and filename sanitization,
- participant/operator anonymization,
- metadata review,
- integrity checks.

### Stage 5 — Stable release

A tagged release will freeze:

- code version,
- configuration files,
- dependency versions,
- reproducibility instructions,
- checksums for released assets.

## Reproducibility Principles

The repository follows several rules intended to keep evaluation scientifically traceable:

- **sample-first splitting:** windows from the same physical recording must not cross train/test boundaries;
- **true event pairing:** synchronized receiver views are paired only when they originate from the same physical event;
- **development and frozen evaluation are separated;**
- **frozen evaluation settings are not retuned from final evaluation results;**
- **operational failures are distinguished from valid negative scientific results;**
- **adapted external baselines are explicitly labeled as adaptations rather than verbatim reproductions.**

## Data and Code Availability

The project supports public release of code and anonymized research data, but the complete release is **not yet available**.

The repository will be updated as privacy review, code cleanup, reproducibility packaging, and data preparation are completed.

## Privacy

Public materials will use anonymized labels only. Internal participant/operator aliases, account names, device-management history, private storage paths, and other identifying acquisition metadata are excluded from the public release.

## Citation

Citation information will be added when the associated research artifact is publicly available.

## Contact

For questions about the repository, please use GitHub Issues after the public release is complete.
