"""
Refit acceptance gate: retention-conditioned coefficient of variation.

Larranaga's convergence criterion is the mean CV over active terms with a 5%
cutoff, used to terminate an active-learning loop. This computes the same
statistic per refit so it can gate individual model updates instead.

The retention conditioning is the part that isn't in the paper. Under library
dropout a member can show a zero coefficient for two totally different reasons:
the fit killed the term, or the member never got offered it. Averaging over all
members conflates those and washes out the collinearity signal we're trying to
detect. So we only average over members that were offered the term AND kept it.
"""

import numpy as np


def gate_cv(stacked, masks, median=None, restrict_to_support=True,
            active_frac=0.5, min_members=5, mean_floor=1e-8):
    """
    median : (n_targets, n_features), the first return of ensemble_parallel.
             when restrict_to_support is True, only terms that are nonzero in
             the median (i.e. terms that actually get DEPLOYED) are scored.

    the retention vote alone is not enough. at low N the members disagree
    violently on junk terms that the element-wise median then discards, so the
    CV goes huge while the deployed model is fine. scoring the median's own
    support puts the statistic on the same object the error metric measures.
    """
    stacked = np.asarray(stacked)
    masks = np.asarray(masks, dtype=bool)
    B, n_targets, n_features = stacked.shape
    if masks.shape != (B, n_features):
        raise ValueError(f"masks must be {(B, n_features)}, got {masks.shape}")

    if restrict_to_support:
        if median is None:
            raise ValueError("restrict_to_support=True requires median")
        support = np.asarray(median) != 0.0
    else:
        support = np.ones((n_targets, n_features), dtype=bool)

    rows = []
    for t in range(n_targets):
        for f in range(n_features):
            if not support[t, f]:
                continue
            offered = masks[:, f]
            n_offered = int(offered.sum())
            if n_offered == 0:
                continue

            kept = offered & (stacked[:, t, f] != 0.0)
            n_kept = int(kept.sum())
            if n_kept < min_members:
                continue

            retention = n_kept / n_offered
            if retention < active_frac:
                continue

            vals = stacked[kept, t, f]
            mu = float(vals.mean())
            sd = float(vals.std(ddof=1))
            cv = np.inf if abs(mu) < mean_floor else sd / abs(mu)

            rows.append({
                "target": t, "feature": f,
                "n_offered": n_offered, "n_kept": n_kept,
                "retention": retention, "mean": mu, "std": sd, "cv": cv,
            })
    if not rows:
        return {"mean_cv": np.inf, "median_cv": np.inf, "max_cv": np.inf,
                "n_active": 0, "terms": []}

    cvs = np.array([r["cv"] for r in rows])
    return {
        "mean_cv": float(cvs.mean()),
        "median_cv": float(np.median(cvs)),
        "max_cv": float(cvs.max()),
        "n_active": len(rows),
        "terms": rows,
    }

    if not rows:
        # no active terms at all. that is itself a reject, not a pass
        return {"mean_cv": np.inf, "median_cv": np.inf, "max_cv": np.inf,
                "n_active": 0, "terms": []}

    cvs = np.array([r["cv"] for r in rows])
    return {
        "mean_cv": float(cvs.mean()),
        "median_cv": float(np.median(cvs)),
        "max_cv": float(cvs.max()),
        "n_active": len(rows),
        "terms": rows,
    }


def accept_refit(stacked, masks, threshold, median=None, stat_key="median_cv", **kw):
    """
    threshold is NOT Larranaga's 5%. hers is tuned for a stopping rule on a
    converging sequence, this is a per-refit accept/reject. calibrate it.

    median is required unless you pass restrict_to_support=False through kw.
    """
    stat = gate_cv(stacked, masks, median=median, **kw)
    value = stat[stat_key]
    return bool(np.isfinite(value) and value <= threshold), stat