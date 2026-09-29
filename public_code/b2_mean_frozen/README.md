# B2_MEAN frozen public snapshot

This directory is the first public code release from **WIFICSI_RX_ALIGNMENT**. It exposes the frozen **B2_MEAN** mechanism encoder used by the paper-facing receiver-identity analysis without publishing private dataset-routing code, filesystem paths, or internal operator aliases.

## What B2_MEAN is

For each synchronized event, three receiver views are processed by the **same** convolutional encoder. Each receiver produces a 64-dimensional latent vector. B2_MEAN performs an **equal arithmetic mean** over the three receiver latents and feeds the pooled vector to a dropout + linear task classifier.

There is no receiver-specific adapter, adversarial receiver loss, learned receiver gate, or alignment loss in B2_MEAN.

```text
3 x receiver CSI view
        |
 shared ViewEncoder
        |
3 x 64-D receiver latent z_r
        |
  equal mean over r
        |
 dropout + linear classifier
```

## Frozen input contract

The paper-facing configuration uses tensors shaped `[B, 3, 2, 128, 64]`:

- 3 synchronized receiver views;
- 2 channels: robust-normalized CSI amplitude and observation mask;
- 128 time bins;
- 64 frequency/subcarrier bins.

The internal raw-data parser is intentionally **not** part of this first public snapshot. Public users should construct tensors satisfying the documented contract.

## Files

- `b2_mean.py` - standalone frozen model semantics.
- `train_b2_mean.py` - minimal public training helpers, including the physical-recording-aware window weighting used by the frozen trainer.
- `config_b2_mean.json` - sanitized frozen hyperparameters and input contract.
- `PROVENANCE.md` - relationship to the internal frozen source and release boundary.
- `smoke_test.py` - shape/forward and weighting checks.
- `SHA256SUMS.txt` - checksums of this source snapshot.

## Quick smoke test

```bash
python smoke_test.py
```

Requires Python 3.10+ and PyTorch. NumPy is required by the training helper smoke check.

## Reproducibility boundary

This is an **algorithmic frozen-source snapshot**, not yet the complete end-to-end dataset reproduction package. The following are deliberately excluded until the broader public release is finalized:

- private filesystem paths and server configuration;
- raw acquisition/operator aliases;
- private dataset-routing and collection metadata;
- unreleased de-identified CSI recordings.

The complete paper evidence also includes analytic MCCA/shared-alignment and post-freeze audits that will be released separately. Publishing B2_MEAN does not change or retune the frozen paper results.

## Privacy

No internal participant/operator aliases or private filesystem paths are included in this directory.
