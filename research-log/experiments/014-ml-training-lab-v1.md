# 014 · MovieLens 100K matrix factorization v1 · 2026-11-21

**VERIFIED** · Client date. One CPU training run and one final held-out evaluation completed. No language-model inference, prompt, decoding, or adapter applied. All four tests passed.

## Hypothesis and protocol

A biased dot product can learn signal from observed binary ratings, and validation loss can locate the point where further fitting stops improving held-out prediction. Official GroupLens MovieLens 100K archive, 100,000 unique observed user–movie pairs across 943 users and 1,682 movies. Labels are 1 for ratings 4–5 and 0 for ratings 1–3. Missing pairs are excluded. NumPy seed 42 shuffles within each user, with floored cuts at 70% and 85%. The published ID ranges define embedding slots. No fitted preprocessing. Train updates only, validation selects on BCE, and test is scored once after loading the selected state.

PyTorch CPU with one thread and deterministic algorithms, seed 42 for initialization and the independent training shuffle generator. 32 dimensions, Gaussian embedding initialization (standard deviation 0.05), zero biases, Adam at 0.01, batch size 1024, BCEWithLogitsLoss, at most 40 epochs, patience five. ROC-AUC uses logits and is undefined for a single class. The separate diagnostic uses one fixed two-dimensional example and SGD at 0.1.

## Reproduction and evidence

From `ml-training-lab` with Python 3.11.14, create the virtual environment, install with `.venv/bin/python -m pip install -e '.[test]'`, and download the archive as in the project README. Commands executed:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ml_training_lab.diagnostics
MPLCONFIGDIR=/tmp/ml-training-matplotlib .venv/bin/python -m ml_training_lab.train --seed 42 --output outputs/seed42
```

Parent revision `52bc07ab4281132f7364cd59a22b3d640d82ee9d`, with uncommitted project sources hashed at execution. macOS 14.5 arm64, PyTorch 2.8.0, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, matplotlib 3.10.7. [Environment, logs, config, history, scores, and curve](../evidence/014-ml-training-lab-v1/) are preserved with a [SHA-256 manifest](../evidence/014-ml-training-lab-v1/SHA256SUMS). The archive at `ml-training-lab/data/ml-100k.zip` and selected weights at `ml-training-lab/outputs/seed42/best.pt` are local-only, with repository-relative paths and hashes in [sources-and-local.sha256](../evidence/014-ml-training-lab-v1/sources-and-local.sha256). Download the archive again and verify its digest if it is lost. Retraining can recreate a checkpoint but serialization and numeric results may vary by runtime.

The first sandboxed download failed DNS resolution and succeeded with network access. uv failed on its default cache and then panicked with a writable cache. Python venv and pip succeeded. These setup attempts preceded training. No evaluation was unfinished and no outputs were repaired.

## Observations and limits

Train had 69,575 ratings (38,470 positive), validation had 14,970 (8,299 positive), and test had 15,455 (8,606 positive). The run selected epoch 2, with validation BCE 0.561498 and ROC-AUC 0.777326. Early stopping fired at epoch 7, when training BCE was 0.176533 and validation BCE was 0.796001. The selected checkpoint produced test BCE 0.567707 and ROC-AUC 0.772106. The diagnostic matched both finite, nonzero analytical embedding gradients and the SGD updates.

These are point estimates from one seed and random per-user split. There is no interval or seed-variability estimate, and repeated ratings by the same users and movies are dependent. There were 23 test ratings for movies absent from training, whose movie parameters stayed at initialization. The protocol measures observed-rating prediction. Population ranking and temporal generalization were outside the experiment.

Next question for a later phase: how does embedding dimension change the epoch at which validation loss reaches its minimum?
