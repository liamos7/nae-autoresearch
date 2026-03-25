# NAE Autoresearch — Agent Instructions

## Context

You are an autonomous ML research agent working on **Normalized Autoencoder (NAE)** training for anomaly detection, applied to particle physics (LHC jet tagging) and image datasets. The NAE is an energy-based model that uses reconstruction error as its energy function, with Langevin Monte Carlo sampling to generate negative samples for contrastive training.

**The core problem**: NAE Phase 2 (energy-based) training is notoriously unstable. It often collapses — energies diverge, gradients explode, or the MCMC sampler fails. Your job is to find hyperparameter configurations and code changes that make training converge reliably while maximizing anomaly detection AUC.

## Files

- `train.py` — **YOU MODIFY THIS.** Contains the model, hyperparameters, Langevin sampler, loss function, and training loop. Everything is fair game.
- `evaluate.py` — **DO NOT MODIFY.** The evaluation harness that runs your experiment and scores it. Score = AUC × stability. Higher is better.
- `program.md` — These instructions (you don't modify this either).

## How To Run an Experiment

```bash
# 1. First, do a git checkout of a new branch for this experiment
git checkout -b experiment-$(date +%s)

# 2. Make your changes to train.py

# 3. Run the experiment through the evaluation harness
python evaluate.py --dataset CICADA --holdout-class 0 --pretrained-path <path_to_phase1_weights> --time-budget 600

# 4. Check the score
cat autoresearch_logs/history.json | python -m json.tool | tail -20

# 5. If score improved: commit and keep
git add train.py
git commit -m "Experiment: <brief description of change>. Score: <score>"

# 6. If score did not improve: revert
git checkout main -- train.py
```

## The Metric

**Score = best_val_auc × stability_multiplier**

- `best_val_auc`: ROC-AUC for distinguishing holdout class (anomalies) from inlier classes using reconstruction energy. Range [0, 1], higher is better.
- `stability_multiplier`: 1.0 if training completed without collapse, 0.0 otherwise.
- A collapsed run (NaN loss, diverging energies, or energy ratio explosion) always scores 0.0.

**Your goal**: Maximize this score. A score of 0.95+ is excellent. Baseline (hand-tuned) is typically 0.85-0.93.

## What to Explore

### High Priority — These are the most likely sources of instability

1. **LMC Step Sizes and Noise** (`Z_STEP_SIZE`, `Z_NOISE_STD`, `X_STEP_SIZE`, `X_NOISE_STD`)
   - The theoretical relationship is 2λ = σ², but in practice they're tuned separately.
   - Too large → chains diverge. Too small → chains don't mix and negative samples are too close to data.
   - Try different ratios and absolute scales.

2. **Number of MCMC Steps** (`Z_STEPS`, `X_STEPS`)
   - More steps = better negative samples but slower training and more gradient computation.
   - The original paper uses 20-60 steps in Z and 10-50 in X.
   - Try finding the sweet spot for your time budget.

3. **Energy Regularization** (`GAMMA`)
   - Regularizes squared energy of positive AND negative samples.
   - Too small → energies diverge. Too large → model can't learn contrast.
   - This is the most sensitive parameter. Try log-scale: 1e-3, 5e-3, 1e-2, 5e-2, 1e-1.

4. **Temperature** (`TEMPERATURE`, `TEMPERATURE_TRAINABLE`)
   - Controls the sharpness of the energy landscape.
   - Making it trainable can help the model self-regulate but adds instability.
   - Try fixed values: 0.1, 0.5, 1.0, 2.0, 5.0.

### Medium Priority

5. **Replay Buffer** (`BUFFER_PROB`, `BUFFER_SIZE`, `BUFFER_REINIT_PROB`)
   - Buffer stores negative latent samples between iterations for chain continuity.
   - Original paper: 95% from buffer, 5% fresh noise. Try other ratios.
   - Buffer seeded from encoded training data (not random noise) — try alternative seeding.

6. **Negative Weight** (`NEG_LAMBDA`)
   - Weight on the negative energy term in the contrastive loss.
   - Default 1.0. Try asymmetric weighting: 0.5, 0.8, 1.2, 1.5.

7. **Learning Rate and Schedule** (`LEARNING_RATE`, `WARMUP_FRACTION`, `USE_COSINE_DECAY`)
   - Energy-based training benefits from careful LR scheduling.
   - Try lower LRs (1e-5) with longer warmup. Try cyclical schedules.

8. **Gradient Clipping** (`GRAD_CLIP`, `LMC_GRAD_CLIP`)
   - Critical for preventing explosion. Try different thresholds.
   - The Du & Mordatch 2019 recommendation is 0.01 for LMC gradients.

### Lower Priority (but potentially impactful)

9. **Architecture Changes**
   - Add batch normalization, spectral normalization, or residual connections.
   - Try different activation functions (SiLU, GELU instead of LeakyReLU).
   - Try ResBlocks in the encoder/decoder.

10. **Loss Function Modifications**
    - Try regularizing only negative energy (not positive).
    - Try Huber loss instead of L2 for reconstruction.
    - Try spectral normalization on weights instead of L2 penalty.

11. **Latent Space** (`LATENT_DIM`, `SPHERICAL`)
    - The paper shows spherical latent space has advantages.
    - Try different latent dimensions: 3 (for visualization), 10, 20, 32, 64.

12. **Annealing Strategies**
    - Noise annealing in the data chain (`X_USE_ANNEALING`, `X_ANNEALING_DECAY`).
    - Temperature annealing over training epochs.
    - LMC step size annealing over training.

## Research Strategy

1. **Start with stability.** A stable run that achieves AUC=0.80 beats a collapsed run that might have achieved 0.95. First make it not collapse, then optimize.

2. **Change one thing at a time.** Make a single modification per experiment. This lets you attribute improvements/regressions to specific changes.

3. **Log scale for sensitive parameters.** When exploring GAMMA, LEARNING_RATE, step sizes — try powers of 10 or factors of 2-3.

4. **If a run collapses, diagnose why:**
   - NaN loss → gradient explosion → reduce LR or increase grad clip
   - Energy divergence (neg ≫ pos) → increase GAMMA or reduce NEG_LAMBDA
   - No energy separation (pos ≈ neg) → increase MCMC steps or step sizes
   - Slow convergence → increase LR or reduce warmup

5. **Keep the git log clean.** Every commit should describe what was changed and what score resulted. This is your lab notebook.

## Known Failure Modes (from the literature)

- **Sampler collapse**: Negative energies diverge to -∞. The sampler generates garbage. Detected by `MAX_ENERGY_RATIO`. Fix: increase `GAMMA`, decrease step sizes.
- **Mode collapse in replay buffer**: All buffer entries converge to same point. Fix: increase `BUFFER_REINIT_PROB` or add noise injection.
- **Correlated chains**: MCMC chains become correlated (cover only part of energy landscape). Fix: increase fresh noise probability (`1 - BUFFER_PROB`).
- **Spurious modes**: Model assigns low energy to regions far from data. Fix: more MCMC steps, better initialization.
- **Oscillating loss**: Loss swings between positive and negative. Normal early on, but should stabilize. Fix: reduce LR, increase warmup.

## Environment

- **Cluster**: Princeton Adroit (NVIDIA GPUs, Slurm scheduler)
- **Time budget**: 10 minutes per experiment (configurable with `--time-budget`)
- **Dataset**: CICADA (LHC calorimeter images, 18×14 pixels). Background = Zero Bias (label 0). Signals (labels 1-10): ggH→ττ, ggH→γγ, H→2LL→4b, SingleNeutrino, SUEP, tt̄, VBF H→2b, VBF H→ττ, Z'→ττ, ZZ.
- **Data root**: `/scratch/network/lo8603/thesis/fast-ad/data/h5_files/`
- **Phase 1 weights**: Pretrained AE at `--pretrained-path`

## One More Thing

Don't be afraid to make bold changes. The current hyperparameters are a starting point, not a known optimum. The original NAE papers note that "stabilizing the training during its different phases requires serious effort" — you're doing that effort automatically.

Good luck.