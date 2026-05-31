# PULS Structural Evaluator Report

## Ablation Flags (Provenance)
- F1 (SP-PIM): enabled
- F2 (Window cap): default 3
- F3 (A/B): enabled
- F5 (Channel-indep): enabled

## Idle Fraction
- gpu_instance_a: 0.0003
- pim_instance_a: 0.9961
- gpu_instance_b: 0.0000

## PIM Utilization: 1.0000
## Pipeline Efficiency: 0.5161

## Convergence
- converged: True
- oscillating: False
- in_band_fraction: 0.9827
- samples: 9931025

## Acceleration Decomposition (Direction Only — Impl-10 calibrated 값)
| Source | cycle_with | cycle_without | ratio | direction+ |
|---|---|---|---|---|
| F1_SP_PIM | 4.0000 | 4.0000 | 1.0000 | False |
| F2_DOUBLE_BUFFER | 4.0000 | 4.2670 | 1.0668 | True |
| F3_INSTANCE_AB | 4.2670 | 8.2670 | 1.9374 | True |
| F5_CHANNEL_INDEP | 0.2670 | 0.5340 | 2.0000 | True |