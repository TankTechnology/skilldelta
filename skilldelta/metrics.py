"""Fixed-panel evaluation. Ratios are fractions unless a key ends in _pp."""
import numpy as np


def auroc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("labels and finite scores must be matching vectors")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[labels].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def classification(labels, scores, actions):
    labels = np.asarray(labels, bool)
    actions = np.asarray(actions, bool)
    positive, negative = labels.sum(), (~labels).sum()
    tpr = np.mean(actions[labels]) if positive else None
    tnr = np.mean(~actions[~labels]) if negative else None
    return {"auroc": auroc(labels, scores),
            "balanced_accuracy": float((tpr + tnr) / 2) if positive and negative else None,
            "direction_accuracy": float(np.mean(labels == actions)),
            "use_rate": float(np.mean(actions))}


def policy(y0, y1, tokens0, tokens1, actions, *, mean_router_tokens=0.0):
    y0, y1, tokens0, tokens1, actions = [np.asarray(x, dtype=float)
                                       for x in (y0, y1, tokens0, tokens1, actions)]
    chosen = y0 + actions * (y1 - y0)
    tokens = tokens0 + actions * (tokens1 - tokens0)
    return {"success": float(chosen.mean()),
            "always_skip_success": float(y0.mean()),
            "always_use_success": float(y1.mean()),
            "gain_vs_always_skip_pp": float(100 * (chosen - y0).mean()),
            "gap_vs_always_use_pp": float(100 * (chosen - y1).mean()),
            "use_rate": float(actions.mean()),
            "mean_execution_tokens": float(tokens.mean()),
            "mean_router_tokens": float(mean_router_tokens),
            "token_saving": float(1 - (tokens.mean() + mean_router_tokens) / tokens1.mean())}


def matched_rate_advantage(y0, y1, actions):
    """Exact expected random activation; no simulated random decisions."""
    y0, y1, actions = [np.asarray(x, dtype=float) for x in (y0, y1, actions)]
    random_success = float(y0.mean() + actions.mean() * (y1 - y0).mean())
    success = float(np.mean(y0 + actions * (y1 - y0)))
    return {"random_success": random_success,
            "selection_advantage_pp": 100 * (success - random_success)}
