# 015 · Phase 2A embedding dimension ablation v1 · 2026-11-21

**VERIFIED** · Client date. Four models trained on CPU (bias only, 2D, 8D, 128D). The verified Phase 1 32D history and checkpoint were reused. No test scoring occurred in Phase 2A. No prompt, decoding, or adapter applied. All four existing tests passed.

## Experimental controls

Hypothesis: lower latent dimension constrains fitted interactions, reducing the train–validation gap and potentially lowering validation BCE. The bias-only comparator learns one scalar per user and movie with no interaction term. All matrix factorization arms retain both bias terms.

The original archive and unchanged split function produce the exact Phase 1 per-user split, with 69,575 training and 14,970 validation observed pairs. The 15,455 test pairs are reserved. Ratings 4–5 are positive and ratings 1–3 are negative. Missing pairs are excluded. This is observed-rating prediction.

All runs use CPU, one PyTorch thread, deterministic algorithms, Adam at 0.01, zero weight decay, batches of 1024, BCEWithLogitsLoss, a 40-epoch cap, and five epochs of patience on strict validation BCE improvement. Seed 42 resets model initialization and the independent batch shuffle generator for each arm. Biases start at zero. Each latent coordinate is drawn from a zero-mean normal with standard deviation 0.05. The same seed and initialization rule do not force identical initial dot-product scales across dimensions.

## Measured results

[Combined comparison table](../evidence/015-capacity-ablation-v1/comparison.csv) · [Learning curves](../evidence/015-capacity-ablation-v1/learning-curves.png)

8D had the lowest validation BCE at epoch 3 (0.560823), followed by 32D (0.561498), 2D (0.562924), bias only (0.563944), and 128D (0.566558). The 8D improvement over 32D was 0.000675 BCE and the improvement over bias only was 0.003121. 32D had the highest ROC-AUC among the BCE-selected checkpoints (0.777326 vs. 0.776273 for 8D). Bias only reached 0.773181 and outscored 128D on both validation metrics at their respective BCE-selected checkpoints.

The gap widened monotonically with latent dimension, from 0.061889 at 2D to 0.305502 at 128D. This supports the capacity hypothesis for the gap and for shrinking the widest model. The best BCE at 8D and the weaker 2D and bias-only results show that reducing capacity beyond 8D did not improve this run. The gap is validation BCE minus a fresh, frozen-checkpoint BCE pass over the training set. The training curve instead averages batch loss as the weights change during an epoch.

Unexpected observations: the bias-only model came within 0.003121 BCE of the best factorization, and 128D had the weakest validation BCE despite the lowest selected-checkpoint training BCE. The preferred dimension depends on whether evaluation emphasizes BCE or ranking quality. All results are point estimates from one fixed split and initialization seed. Small differences may change with initialization or sampling. The fixed optimizer and per-coordinate initialization couple latent dimension to optimization dynamics as well as representational capacity.

Recommended next experiment: repeat the dimension comparison over several initialization seeds while holding the seed-42 split and batch order fixed. Compare paired validation BCE differences and gaps before selecting a configuration for a final test pass.

## Reproduction and evidence

From `ml-training-lab` with its Phase 1 environment and baseline artifacts:

```sh
.venv/bin/python -m pytest -q
MPLCONFIGDIR=/tmp/ml-training-matplotlib .venv/bin/python -m ml_training_lab.capacity_ablation
```

Parent revision `52bc07ab4281132f7364cd59a22b3d640d82ee9d`. The working-tree source hashes are preserved. The Phase 1 archive, checkpoint, config, history, plot and key source hashes were checked against the historical manifests before reuse. The dependency and platform snapshot matched Phase 1 (Python 3.11.14, PyTorch 2.8.0, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, matplotlib 3.10.7, macOS 14.5 arm64). No training failures, unfinished evaluations, or output repairs.

[Evidence manifest](../evidence/015-capacity-ablation-v1/SHA256SUMS) covers config, history, comparison, plots, test and run logs, and environment. [Local-only artifact and source hashes](../evidence/015-capacity-ablation-v1/sources-and-local.sha256) give the exact paths for the archive and five selected checkpoints. If weights are lost, retraining can regenerate models under the recorded settings but byte-identical serialization is not guaranteed. The original Phase 1 artifacts were preserved, including the historical 32D test outcome, which was excluded from selection.
