"""
Gate calibration on a real SINDy problem: the Hopf normal form.

The synthetic in sweep_gate.py was a standard-normal design matrix with one
planted correlated column. That is not a SINDy library. Neither is
refit_profile.make_buffer, which evaluates a polynomial library on iid Gaussian
points, where x and x^2, and x^2 and y^2, are all uncorrelated by construction.
Both give a near-orthogonal library and neither exhibits the geometry that
causes the pathology.

Here the data comes from integrating

    xdot = mu*x - omega*y - A*x*(x^2 + y^2)
    ydot = omega*x + mu*y - A*y*(x^2 + y^2)

which has a stable limit cycle at radius sqrt(mu/A). The knob is where the
initial conditions live:

    concentration = 1.0  all trajectories start ON the limit cycle. every
                         sample lies on a 1-D curve in a 2-D state space, the
                         polynomial library loses rank, condition number blows
                         up. this is Larranaga's on-attractor case.
    concentration = 0.0  initial conditions spread across phase space, the
                         library is well conditioned.

Written to the same CSV schema as sweep_gate.py so analyze_calib.py reads it
unchanged. The `rho` column holds 1 - concentration, NOT a correlation, so that
rho=0 still means "the well-conditioned end" and the baseline-relative analysis
in analyze_calib keeps its meaning. `ic_spread` carries the same number under a
clearer name.

usage:
    python sweep_hopf.py --out gate_calib_hopf.csv
    python analyze_calib.py --csv gate_calib_hopf.csv
"""

import argparse
import csv

import numpy as np
from pysindy.feature_library import PolynomialLibrary
from scipy.integrate import solve_ivp

from gate import gate_cv
from refit_parallel import ensemble_parallel

MU = 0.5
OMEGA = 1.0
A = 1.0
R0 = np.sqrt(MU / A)        # limit cycle radius, about 0.707

DEGREE = 3                  # need cubics, the true model has x^3 and x*y^2
N_MODELS = 20
T_END = 2.0                 # about a third of a period. long enough to see
                            # dynamics, short enough that off-attractor starts
                            # have not yet collapsed onto the cycle


def rhs(t, s):
    x, y = s
    r2 = x * x + y * y
    return [MU * x - OMEGA * y - A * x * r2,
            OMEGA * x + MU * y - A * y * r2]


def true_coefs(feature_names):
    """
    map the analytic coefficients onto whatever column order pysindy produced.
    never hardcode indices here, PolynomialLibrary ordering is not obvious and
    silently mismatching it would corrupt every error number downstream.
    """
    xdot = {"x": MU, "y": -OMEGA, "x^3": -A, "x y^2": -A}
    ydot = {"x": OMEGA, "y": MU, "x^2 y": -A, "y^3": -A}

    xi = np.zeros((2, len(feature_names)))
    for j, name in enumerate(feature_names):
        key = name.strip()
        xi[0, j] = xdot.get(key, 0.0)
        xi[1, j] = ydot.get(key, 0.0)

    # sanity: every analytic term must have found a column, otherwise pysindy's
    # naming changed and the truth is silently wrong
    found = {n.strip() for n in feature_names}
    for k in set(xdot) | set(ydot):
        if k not in found:
            raise RuntimeError(f"true term {k!r} not in library names {sorted(found)}")
    return xi

def make_trajectories(concentration, n_samples, noise, seed, n_per=25):
    """
    n_traj scales with n_samples, n_per is held fixed. previously n_traj was
    pinned at 10, so raising N only added redundant points along the same ten
    curves and effective sample size never grew, which is why err at conc=0 was
    non-monotonic in N (0.277 -> 0.0007 -> 0.081 -> 0.034)
    """
    rng = np.random.default_rng(seed)
    spread = 1.0 - concentration
    n_traj = max(2, n_samples // n_per)

    states, derivs = [], []
    for _ in range(n_traj):
        phase = rng.uniform(0, 2 * np.pi)
        # at concentration=1 this is exactly R0, at 0 it ranges over roughly
        # 0.1*R0 to 2.5*R0
        r = R0 * (1.0 + spread * rng.uniform(-0.9, 1.5))
        s0 = [r * np.cos(phase), r * np.sin(phase)]

        t_eval = np.sort(rng.uniform(0.0, T_END, n_per))
        sol = solve_ivp(rhs, (0.0, T_END), s0, t_eval=t_eval,
                        rtol=1e-10, atol=1e-12)
        if not sol.success:
            continue
        s = sol.y.T                       # (n_per, 2)
        # exact derivatives from the ODE. finite differencing would add its own
        # error and confound the conditioning effect we are trying to isolate
        d = np.array([rhs(0.0, row) for row in s])
        states.append(s)
        derivs.append(d)

    x = np.vstack(states)
    xdot = np.vstack(derivs)
    xdot = xdot + noise * rng.standard_normal(xdot.shape)
    return x, xdot


def run_point(concentration, n_samples, noise, seed):
    x, xdot = make_trajectories(concentration, n_samples, noise, seed)

    lib = PolynomialLibrary(degree=DEGREE, include_bias=False)
    theta = lib.fit_transform(x)
    names = lib.get_feature_names(input_features=["x", "y"])
    truth = true_coefs(names)

    median, stacked, masks = ensemble_parallel(
        theta, xdot, n_models=N_MODELS, n_jobs=1, base_seed=1000 * seed
    )
    stat = gate_cv(stacked, masks, median=median)

    err = float(np.linalg.norm(median - truth) / np.linalg.norm(truth))
    spurious = int(np.sum((median != 0) & (truth == 0)))
    missed = int(np.sum((median == 0) & (truth != 0)))

    return dict(
        # rho holds 1 - concentration so rho=0 is still the well-conditioned end
        rho=round(1.0 - concentration, 6),
        ic_spread=round(1.0 - concentration, 6),
        concentration=concentration,
        n_features=theta.shape[1],
        n_samples=theta.shape[0],
        noise=noise,
        seed=seed,
        cond=float(np.linalg.cond(theta)),
        mean_cv=stat["mean_cv"],
        median_cv=stat["median_cv"],
        max_cv=stat["max_cv"],
        n_active=stat["n_active"],
        coef_err=err,
        spurious=spurious,
        missed=missed,
    )


def sweep(concentrations, sample_counts, noises, seeds, out):
    rows = []
    for N in sample_counts:
        for noise in noises:
            for c in concentrations:
                grp = []
                for s in seeds:
                    r = run_point(c, N, noise, s)
                    rows.append(r)
                    grp.append(r)
                finite = [g["mean_cv"] for g in grp if np.isfinite(g["mean_cv"])]
                mcv = np.median(finite) if finite else np.inf
                print(f"N={N:5d} noise={noise:<6.3g} conc={c:<5.2f} "
                      f"cond={np.median([g['cond'] for g in grp]):11.1f}  "
                      f"mean_cv={mcv:9.4f}  "
                      f"med_cv={np.median([g['median_cv'] for g in grp]):9.4f}  "
                      f"err={np.median([g['coef_err'] for g in grp]):9.5f}  "
                      f"act={np.median([g['n_active'] for g in grp]):.1f}  "
                      f"spur={np.median([g['spurious'] for g in grp]):.1f}  "
                      f"miss={np.median([g['missed'] for g in grp]):.1f}")

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {out}")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="gate_calib_hopf.csv")
    args = p.parse_args()

    sweep(
        concentrations=[0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0],
        sample_counts=[50, 100, 250, 1000],
        noises=[1e-3, 1e-2],
        seeds=range(6),
        out=args.out,
    )
