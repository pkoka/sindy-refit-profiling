"""


Parallel E-SINDy refit: bootstrap members are independent, so the ensemble
loop is embarrassingly parallel. Day 1 showed cost scales as B^0.98, i.e.
almost exactly linear in ensemble count, so this is the right lever.

reuses the Day 1 harness for library construction and buffer generation

usage:
    python refit_parallel.py --check      # correctness vs EnsembleOptimizer
    python refit_parallel.py --sweep      # n_jobs speedup sweep


"""
import argparse
import time

import numpy as np
from joblib import Parallel, delayed
from pysindy.optimizers import EnsembleOptimizer, STLSQ

from refit_profile import (ALPHA, N_STATE, THRESHOLD, build_theta,
                           make_buffer, time_refit)


def fit_one_member(theta, y, seed, n_drop=1):
    """
    
    fit a single ensemble member

    mirrors EnsembleOptimizer._reduce:
      1. bootstrap rows: n_subset = n_samples, replace=True
      2. drop n_drop library columns, chosen without replacement, sorted
      3. fit a FRESH STLSQ (never share one across workers)
      4. scatter coefficients back into a full-width zero array

    returns array of shape (n_targets, n_features)

    """

    rng = np.random.default_rng(seed)
    n_samples, n_features = theta.shape

    rows = rng.integers(0, n_samples, n_samples)
    theta_boot, y_boot = theta[rows], y[rows]

    keep_inds = np.sort(rng.choice(n_features, n_features - n_drop, replace=False))
    theta_sub = theta_boot[:, keep_inds]

    opt = STLSQ(threshold=THRESHOLD, alpha=ALPHA)
    opt.fit(theta_sub, y_boot)

    coef = np.zeros((y.shape[1], n_features))
    coef[:, keep_inds] = opt.coef_
    return coef


def ensemble_parallel(theta, y, n_models=20, n_jobs=1, base_seed=0):
    """
    
    run n_models bootstrap members under joblib and aggregate them

    each member gets distinct seed (base_seed + i) so bootstrap row draws and column dropouts differ across members

    returns:

    median : ndarray, shape (n_targets, n_features)
        element-wise median across members, this is the refit estimate
    stacked : ndarray, shape (n_models, n_targets, n_features)
        all member coefficients, retained for spread diagnostics
        e.g. coefficient of variation over active terms, for use as a refit-acceptance gate.


    """

    theta = np.asarray(theta).view(np.ndarray)
    y = np.asarray(y).view(np.ndarray)

    coefs = Parallel(n_jobs=n_jobs)(
        delayed(fit_one_member)(theta, y, base_seed + i) for i in range(n_models)
    )

    stacked = np.stack(coefs, axis=0)
    return np.median(stacked, axis=0), stacked


def check_correctness(n_samples=8000, degree=2, n_models=20, n_trials=10):
    """
    
    compare parallel implementation against EnsembleOptimizer

    EnsembleOptimizer uses global np.random, so exact equality not available

    compare instead:
      - recovered support (which coefficients are nonzero) should match
      - nonzero values should agree to within few percent
    
    print PASS/FAIL
    don't proceed to timing till this passes


    """

    theta, y = make_buffer(n_samples, degree)
    n_samples = theta.shape[0]

    #TODO 7 below
    ref_coefs, ref_members = [], []
    for _ in range(n_trials):
        ens = EnsembleOptimizer(
            STLSQ(threshold=THRESHOLD, alpha=ALPHA),
            bagging=True,
            n_subset=n_samples,        # bootstrap at full size
            n_models=n_models,
            library_ensemble=True,
            n_candidates_to_drop=1,
        )
        ens.fit(theta, y)
        ref_coefs.append(ens.coef_.copy())
        ref_members.append(np.array(ens.coef_list))       # (n_models, n_targets, n_features)

    # TODO 8 below
    par_coefs, par_members = [], []
    for t in range(n_trials):
        C, S = ensemble_parallel(theta, y, n_models=n_models,
                                 n_jobs=1, base_seed=1000 * t)
        par_coefs.append(C)
        par_members.append(S)


    # TODO 9 below
    ref = np.stack(ref_coefs)                 # (n_trials, n_targets, n_features)
    par = np.stack(par_coefs)

    # support: "stable" = nonzero in every trial; "any" = nonzero in at least one
    ref_stable, ref_any = np.all(ref != 0, axis=0), np.any(ref != 0, axis=0)
    par_stable, par_any = np.all(par != 0, axis=0), np.any(par != 0, axis=0)

    only_ref = ref_stable & ~par_any          # reference always finds it, i never do
    only_par = par_stable & ~ref_any          # i always find it, reference never does
    shared   = ref_stable & par_stable

    if shared.sum() == 0:
        print("FAIL: no stable terms in common")
        return

    ref_mean, par_mean = ref.mean(axis=0), par.mean(axis=0)
    rel_diff = np.abs(par_mean[shared] - ref_mean[shared]) / np.abs(ref_mean[shared])

    # member-level spread, averaged over trials — this is what n_subset controls
    ref_spread = np.mean([m.std(axis=0) for m in ref_members], axis=0)[shared]
    par_spread = np.mean([m.std(axis=0) for m in par_members], axis=0)[shared]
    ratio = par_spread / ref_spread


    # negative control: n_candidates_to_drop=3 on the reference drives median spread ratio to 0.63 -> FAIL, test is sensitive to variance-only defects, mean comparison alone isn't
    # --------------------------------
    # gate on median spread ratio only
    # per-term ratios have wide tails (p90 abt 2.0 even at n_trials=10) because library dropout contributes discrete per-term hits, this doesn't avg out
    # requires n_trials >= 10, at n_trials=3 the median itself is unstable (observed 0.63-1.50 across runs of identical code)
    ok = (only_ref.sum() == 0 and only_par.sum() == 0
          and rel_diff.max() < 0.05
          and 0.7 < np.median(ratio) < 1.4)

    print(f"stable support   ref {ref_stable.sum():3d}   par {par_stable.sum():3d}")
    print(f"disagreement     ref-only {only_ref.sum()}   par-only {only_par.sum()}")
    print(f"rel difference   median {np.median(rel_diff):.4f}  max {rel_diff.max():.4f}")
    print(f"spread ratio     median {np.median(ratio):.3f}  "
          f"p10 {np.percentile(ratio,10):.3f}  p90 {np.percentile(ratio,90):.3f}")
    print(f"-> {'PASS' if ok else 'FAIL'}")


def sweep_njobs(configs, n_jobs_grid, out="parallel_times.csv"):
    """Time each config across n_jobs. Report speedup vs n_jobs=1.

    Use two configs with different per-member work:
      small members: N=16000, degree=2, B=20   (the published anchor)
      large members: N=32000, degree=3, B=40   (worst case)
    Parallel efficiency should be noticeably better for the large one,
    since process overhead amortizes over longer fits.
    """
    # TODO 10: nested loop, time ensemble_parallel, compute speedup,
    #          write CSV with columns: N, degree, d, B, n_jobs, mode,
    #          refit_s, speedup
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()
    if args.check:
        check_correctness()
    if args.sweep:
        sweep_njobs(
            configs=[(16000, 2, 20), (32000, 3, 40)],
            n_jobs_grid=[1, 2, 3, 4, 5, 6, 8, 10, 12, 15],
        )
