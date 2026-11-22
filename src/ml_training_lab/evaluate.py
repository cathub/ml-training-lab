"""Score an evaluation split in batches without updating parameters."""

import torch
from sklearn.metrics import roc_auc_score


@torch.inference_mode()
def evaluate(model, loader, loss_fn):
    model.eval()
    loss_sum = 0.0
    scores, labels = [], []
    for users, movies, targets in loader:
        logits = model(users, movies)
        loss_sum += loss_fn(logits, targets).item() * targets.numel()
        scores.append(logits)
        labels.append(targets)

    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    # ROC-AUC only needs scores in rank order. Sigmoid preserves logit order.
    auc = float(roc_auc_score(labels, scores)) if labels.min() != labels.max() else None
    return {"bce": loss_sum / len(labels), "roc_auc": auc}
