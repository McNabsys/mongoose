# V1 Ablation Protocol

Diagnostics to quantify how much the T2D U-Net depends on wfmproc's weakly
supervised duration labels.

## Motivation

The T2D U-Net trains on labels derived from wfmproc. Two concerns:

1. The model may be inheriting wfmproc's peak-detection limitations (false
   negatives, cluster merging).
2. `L_velocity` (weight `lambda_vel`) trains against wfmproc probe durations,
   the weakest-supervised part of training.

Two diagnostics expose these concerns:

- **Peak-count discrepancy** (`evaluate_peak_counts`): compare model peak
  count vs wfmproc-matched probe count per molecule. A positive mean means
  the model finds more peaks than wfmproc labeled.
- **`lambda_vel=0` ablation**: train an otherwise identical model with the
  velocity loss disabled; if MAE is unchanged, `L_velocity` is not pulling
  weight and the wfmproc-duration dependency can be dropped.

## Procedure

1. Train a baseline V1 model:

    ```bash
    python scripts/train.py --epochs 50 --batch-size 16
    ```

2. Train an ablation model with `lambda_vel=0.0` (edit `TrainConfig` or pass
   through your config override).

3. Evaluate both on the held-out die and emit JSON:

    ```bash
    python scripts/evaluate.py --checkpoint v1_baseline.pt \
        --probes-bin ... --assigns ... --reference-map ... \
        --output-json v1_baseline.json --run-id v1-baseline

    python scripts/evaluate.py --checkpoint v1_no_vel.pt \
        --probes-bin ... --assigns ... --reference-map ... \
        --output-json v1_no_vel.json --run-id v1-no-velocity
    ```

4. Compare:

    ```bash
    python scripts/compare_runs.py \
        --results v1_baseline.json v1_no_vel.json \
        --labels "baseline" "no_velocity"
    ```

## Interpretation

- **MAE equivalent across the two runs:** `L_velocity` is not contributing;
  the wfmproc duration dependency can be dropped.
- **Baseline MAE lower:** `L_velocity` is pulling weight and is worth keeping.
- **Peak discrepancy mean strongly positive:** model finds systematically
  more peaks than wfmproc labeled - it is not simply inheriting wfmproc's
  false-negative bias.
- **Peak discrepancy mean near zero:** model is matching wfmproc's peak count
  closely, which is suggestive (though not proof) of inherited behaviour.
