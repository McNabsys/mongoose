# Project Mongoose: Cloud Training Proposal

## Executive Summary

Project Mongoose replaces the legacy parametric T2D (time-to-distance) model with a deep learning approach that learns the nonlinear translocation physics directly from raw nanochannel waveforms. The model is built and tested locally (105 unit tests passing). Training on the full E. coli dataset requires a GPU cloud instance due to dataset size (900 GB raw TDB, ~5 GB preprocessed) and training compute.

**Estimated cost: $100-200 total for the full development cycle.**

## Problem

The current T2D model (`L = C * (t + offset)^alpha`) fits 3 parameters per channel. Empirical analysis shows trailing-end velocity is 3.5x faster than mid-molecule velocity, with a leading-edge acceleration burst (head dive) that the parametric model cannot capture. The model also cannot adapt to channel degradation, TVC effects, or concentration polarization during a run.

## Approach

A 1D Physics-Informed U-Net that:
- Predicts per-sample instantaneous velocity, integrated via cumulative sum to monotonic base-pair coordinates
- Uses shift-invariant inter-probe deltas anchored to the E. coli reference genome (no legacy T2D bias)
- Conditions on 6 measured physical observables (not channel IDs) for cross-instrument generalization
- Trains on 30 E. coli BssSI runs across 3 instruments and 5 dies

## What Is Built

| Component | Status |
|-----------|--------|
| TDB binary reader | Done, tested |
| Probes.bin parser | Done, tested |
| Probe assignment parser | Done, tested |
| Reference map parser | Done, tested |
| Transform file parser | Done, tested |
| Ground truth builder | Done, tested |
| 1D U-Net model (~15M params) | Done, tested |
| Loss functions (focal + sparse Huber + sparse L2) | Done, tested |
| Dataset + DataLoader with augmentations | Done, tested |
| Preprocessing pipeline (900 GB -> 5 GB) | Done, tested |
| Training loop (mixed precision, checkpointing) | Done, tested |
| Inference pipeline (velocity-adaptive NMS) | Done, tested |
| Evaluation vs legacy T2D | Done, tested |

Total: 105 unit tests, all passing.

## Cloud Requirements

### Compute

| Resource | Specification | Justification |
|----------|--------------|---------------|
| GPU | NVIDIA L4 (24 GB VRAM) | Fits batch size 8-16 with mixed precision. L4 is cost-efficient for 1D convolutions. |
| vCPUs | 4 | Data loading workers |
| RAM | 16 GB | Preprocessing + training overhead |
| Instance type | GCP `g2-standard-4` | L4 GPU, 4 vCPU, 16 GB RAM |

### Storage

| Resource | Size | Justification |
|----------|------|---------------|
| Boot disk (SSD) | 50 GB | OS + code + packages |
| Data disk (pd-balanced) | 1 TB | 900 GB TDB files + 5 GB preprocessed cache + headroom |

### Training Time Estimate

| Phase | Duration | Notes |
|-------|----------|-------|
| Data transfer (TDB files) | 2-4 hours | 900 GB from Nabsys file server to GCP disk |
| Preprocessing | 2-4 hours | One-time conversion to compact format |
| V1 training (50 epochs) | 60-100 hours | On spot instance with checkpointing |
| Evaluation + iteration | 10-20 hours | Hyperparameter tuning, V2 bootstrap |

Total active compute: ~80-130 hours.

### Cost Estimate

| Item | Unit Cost | Duration/Size | Total |
|------|-----------|---------------|-------|
| L4 GPU instance (spot) | $0.21/hr | 130 hours | $27 |
| VM base (spot) | $0.08/hr | 130 hours | $10 |
| Boot disk (SSD, 50 GB) | $0.17/GB/mo | 2 months | $17 |
| Data disk (balanced, 1 TB) | $0.10/GB/mo | 2 months | $200 |
| Network egress | ~$0.12/GB | ~10 GB results | $1 |
| **Total** | | | **~$255** |

The data disk dominates. If we preprocess locally and transfer only the 5 GB cache (instead of the full 900 GB TDB), the data disk drops to 50 GB ($10/mo) and total cost falls to **~$75**.

**Recommended approach: preprocess locally, transfer only the cache.**

| Scenario | Storage | Compute | Total |
|----------|---------|---------|-------|
| Transfer raw TDB (900 GB) | ~$200 | ~$37 | ~$255 |
| Transfer preprocessed cache only (5 GB) | ~$30 | ~$37 | ~$75 |

## Risk Mitigation

- **Spot instance preemption:** Training saves checkpoints every 5 epochs. Training resumes from last checkpoint automatically. Spot instances are ~70% cheaper than on-demand.
- **Data integrity:** Preprocessing validates molecule quality (clean, remapped, sufficient probes) before caching. 105 unit tests cover all IO parsers and data transformations.
- **Model failure:** If V1 does not outperform legacy T2D on the held-out die (D08), the evaluation pipeline provides detailed per-interval error analysis to diagnose the failure mode before investing in V2.

## Success Criteria

1. **ML metric:** Inter-probe interval MAE on held-out die D08 is lower than legacy T2D MAE against the E. coli reference genome.
2. **Business metric:** Full remapping pipeline with DL model output rescues molecules that legacy T2D failed to map, increasing usable mapping yield.

## Timeline

| Week | Activity |
|------|----------|
| 1 | Preprocess E. coli data locally. Set up GCP instance. Transfer cache. |
| 2-3 | V1 training + evaluation. Hyperparameter tuning. |
| 4 | V2 bootstrap (if V1 shows promise). Final evaluation. Report. |

## Approval Requested

- GCP project/billing authorization for estimated $75-255 spend
- Access to mount Nabsys file shares (higgs, quark) for preprocessing, or assistance copying E. coli TDB files to a staging location
