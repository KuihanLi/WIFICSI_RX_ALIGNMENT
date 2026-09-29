# Provenance and freeze boundary

- Source identity: internal `00_WIFICSI_RX_ALIGNMENT_B2_MEAN_FROZEN_SOURCE.zip`.
- Public snapshot date: 2026-09-29.
- Purpose: expose the frozen B2_MEAN architecture/fusion/training semantics used by the manuscript's mechanism analysis.
- Scientific status: **frozen**. This publication step does not select new parameters or modify headline evidence.

## Exact semantics preserved

1. Shared `ViewEncoder` applied independently to each of three synchronized receiver views.
2. Encoder: Conv(2->24, k5/s2/p2) -> BN -> GELU -> Conv(24->48, k3/s2/p1) -> BN -> GELU -> Conv(48->64, k3/s2/p1) -> BN -> GELU -> adaptive average pool -> dropout(0.15) -> Linear(64->64) -> LayerNorm(64).
3. B2_MEAN has no adapter and no alignment/adversarial objective.
4. Receiver fusion is the equal mean of the three 64-D receiver latents.
5. Classifier is dropout(0.15) + linear head.
6. AdamW training: learning rate 8e-4, weight decay 1e-4, batch size 128, max 18 epochs, patience 3, AMP enabled where CUDA is available.
7. Window loss weights preserve physical-recording/sample-first semantics: equal mass across physical recordings within each class, then equal mass across windows of a recording.
8. Inner validation is split at physical-recording level and early stopping uses classification balanced accuracy.

## Deliberately omitted from the public snapshot

The internal frozen bundle also contains data-routing utilities tied to private acquisition metadata. Those files are not copied verbatim because they include non-public operator aliases and local/server paths. The public model/training code above is therefore a privacy-safe extraction of the frozen algorithmic semantics rather than a byte-for-byte publication of the internal bundle.
