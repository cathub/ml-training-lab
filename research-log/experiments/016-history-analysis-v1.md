# 016 · Validation by user history length v1 · 2026-11-21

**VERIFIED** · Client date. Frozen-checkpoint validation inference and an SVD diagnostic ran. No training, new model selection, or test scoring ran. No language-model prompt, decoding, or adapter applied. All four existing tests passed.

## Protocol

Hypothesis: shorter user histories are associated with higher validation loss, with a larger penalty at high latent dimension. Recreated the unchanged seed-42 MovieLens 100K per-user split, using its 69,575 training observations to define history and 14,970 validation observations to score. All 943 users were covered. Validation bins were 0–20, 21–50, 51–100, 101–200, and 201+. The rule merges a bin with fewer than 30 users or 100 validation examples with an adjacent bin. All bins passed. Users are assigned using only their training rating counts. No test pairs or prior test scores were used.

32D is the original Phase 1 checkpoint at epoch 2. The frozen Phase 2A checkpoints are 2D at epoch 5, 8D at epoch 3, and 128D at epoch 2. BCE averages over validation ratings within each bin. A second column averages within users first, then gives each user equal weight. ROC-AUC pools logits and labels within a bin and is undefined if only one class is present. All five bins contained both classes. Cross-bin AUC reflects ranking across pooled user–movie ratings.

## Results

[Complete bin metrics](../evidence/016-history-analysis-v1/by-history.csv) · [Learning across history lengths](../evidence/016-history-analysis-v1/validation-by-history.png)

32D validation results (users, validation ratings, BCE, ROC-AUC):

- 0–20 · 199 users, 696 ratings, 0.61599 BCE, 0.70692 AUC
- 21–50 · 307 users, 2,190 ratings, 0.57713 BCE, 0.74551 AUC
- 51–100 · 192 users, 3,046 ratings, 0.57807 BCE, 0.75449 AUC
- 101–200 · 184 users, 5,572 ratings, 0.56217 BCE, 0.77825 AUC
- 201+ · 61 users, 3,466 ratings, 0.52502 BCE, 0.81830 AUC

For each dimension, the shortest bin had the highest BCE and the longest bin had the lowest. Equal user weighting preserved that pattern. The 128D penalty relative to 32D was about 0.01746 BCE in the shortest bin and 0.00424 in the longest. But the shortest bin also penalized shrinking from 32D to 2D (0.01517), and 8D and 32D were close there (0.00080 BCE). The observed relationship with dimension is non-monotonic.

This is an observational analysis of one split and one checkpoint per dimension. Users, movies, and class mix can differ across history bins. The shortest bin is about 60% positive and the longest is about 50% positive. The comparison across dimensions within each bin uses the same examples. No uncertainty intervals were estimated. A follow-up could downsample training histories for the same users while keeping their validation examples fixed, then repeat over seeds.

## What the rank calculation guarantees

Selected MovieLens user 19 by minimum training count, breaking ties by lowest ID. For each checkpoint, fixed movie embeddings and biases and formed the matrix Q from the 14 training movie vectors as rows. SVD used float64 arithmetic on stored weights and counted singular values greater than max(Q.shape) × eps(float64) × σmax. The threshold for 32D was 2.35e−14. Numerical ranks for 2D, 8D, 32D and 128D were 2, 8, 14 and 14, giving numerical nullities 0, 0, 18 and 114. [Singular values and diagnostic output](../evidence/016-history-analysis-v1/rank-diagnostic.json)

With those items and biases fixed, the vector of observed logits is Qp + b. Any perturbation v in the exact kernel of Q preserves every observed logit and therefore the observed BCE, with no embedding penalty in the objective. The exact nullity is d − rank(Q), so fewer than d equations guarantee at least d − n null directions for n observed items. The calculation concerns the user vector with biases fixed. Full column rank gives no null direction for this fixed linear map, without guaranteeing that the user's BCE has a finite unique minimizer.

A unit SVD null direction at 32D changed training logits by at most 1.94e−16 in float64 and changed at least one validation logit by 0.432 in absolute value. This illustrative perturbation was never applied to the checkpoint used for scoring. For unseen movie vectors orthogonal to the kernel, logits would be invariant too. The rank guarantee establishes directions that training logits leave unconstrained. It does not predict the sign or magnitude of a held-out loss difference, and the observed 128D penalty does not establish that those directions caused it.

## Reproduction and provenance

From `ml-training-lab`:

```sh
.venv/bin/python -m pytest -q
MPLCONFIGDIR=/tmp/ml-training-matplotlib .venv/bin/python -m ml_training_lab.history_analysis
```

Parent revision `52bc07ab4281132f7364cd59a22b3d640d82ee9d`, with working sources hashed. Python 3.11.14, PyTorch 2.8.0, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, matplotlib 3.10.7 and macOS 14.5 arm64. Verified the archive and checkpoint hashes against prior evidence and checked each rating-weighted aggregate BCE against its saved validation minimum. No execution failures or repairs.

[SHA-256 manifest](../evidence/016-history-analysis-v1/SHA256SUMS) covers the compact output snapshots and logs. [Local-only checkpoint, dataset, and source hashes](../evidence/016-history-analysis-v1/sources-and-local.sha256) record exact repository-relative paths. If weights are lost, rerun the recorded phase training on a fresh output path. Byte-identical serialization is runtime-dependent.
