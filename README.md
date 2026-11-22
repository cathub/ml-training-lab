# ml-training-lab · Phase 1

Predict whether a **recorded** MovieLens rating is at least 4. This is observed-rating prediction. A full implicit-feedback recommendation benchmark would require a different task and evaluation protocol. Missing user–movie pairs are excluded.

## Reproduce on CPU

From this directory, using Python 3.11 (the baseline used 3.11.14):

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
mkdir -p data
curl -fL https://files.grouplens.org/datasets/movielens/ml-100k.zip -o data/ml-100k.zip
python -m pytest -q
python -m ml_training_lab.diagnostics
python -m ml_training_lab.train --seed 42 --output outputs/seed42
```

Use a fresh output directory for each run. The run saves `config.json`, epoch-level `history.json`, `best.pt`, `loss.png`, and final `metrics.json`. See `python -m ml_training_lab.train --help` for overrides.

## What the model learns

For user vector $p_u$ and movie vector $q_i$, the logit is $z=p_u^Tq_i+b_u+b_i$. Adam updates both embeddings and both biases using batches of observed ratings. BCEWithLogitsLoss takes the logit directly. The probability of a rating of at least 4 is $\sigma(z)$. The two-dimensional diagnostic uses a single SGD step and checks $\nabla_{p_u}L=(\sigma(z)-y)q_i$ and $\nabla_{q_i}L=(\sigma(z)-y)p_u$ against autograd.

Each user's rows are shuffled with seed 42, then cut at the floored 70% and 85% boundaries. Every observed pair occurs in exactly one split. Published IDs directly determine embedding indices and there is no fitted preprocessing. Movies absent from training use their initialized parameters. The split is random within user and does not simulate future ratings.

Default settings are 32 dimensions, Adam learning rate 0.01, batch size 1024, and up to 40 epochs. Early stopping waits five epochs for a strict improvement in validation BCE. The final test pass loads the minimum-validation-BCE checkpoint. ROC-AUC uses logits and is recorded as `null` when a split has only one class. Epoch training BCE averages losses measured as parameters change across batches.

## Seed 42 baseline

The official archive has 100,000 observed ratings from 943 users on 1,682 movies. The train, validation, and test splits contain 69,575, 14,970, and 15,455 ratings, respectively. Their positive rates are 55.29%, 55.44%, and 55.68%.

Epoch 2 had the lowest validation BCE (0.56150, ROC-AUC 0.77733). The run stopped at epoch 7. The selected checkpoint scored test BCE 0.56771 and ROC-AUC 0.77211. Training BCE continued to fall after epoch 2 while validation BCE rose.

## Phase 2A · Embedding dimension ablation

With the original Phase 1 run at `outputs/seed42`, run from this directory:

```sh
python -m pytest -q
python -m ml_training_lab.capacity_ablation
```

The command verifies the saved Phase 1 artifacts and sources against the repository's research evidence. Use `--phase1` if those run artifacts are stored elsewhere, and `--output` for a fresh destination. The defaults write all five runs, `comparison.csv`, and `learning-curves.png` to `outputs/capacity-ablation/`. The 32D history and checkpoint come from Phase 1. The ablation scores only train and validation. See the [research report](research-log/experiments/015-capacity-ablation-v1.md).

## Validation by user history length

With the seed-42 checkpoint available, run:

```sh
python -m ml_training_lab.history_analysis
```

The analysis uses the original 32D model and includes 2D, 8D, and 128D when their Phase 2A checkpoints are present. A new `outputs/history-analysis/` directory contains training counts by MovieLens user ID, validation bin metrics, a plot, and a numerical rank diagnostic. Use `--output` to choose a fresh directory. See the [research report](research-log/experiments/016-history-analysis-v1.md).
