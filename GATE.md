# The CV convergence criterion as a per-refit acceptance gate

Larrañaga's ultra-low-data preprint uses the mean coefficient of variation over
active terms, with a 5% cutoff, as a **convergence criterion** — a stopping rule
for an active-learning loop — and notes the criterion has not been examined in
depth.

This tests it as a **per-refit acceptance gate**: computed on every refit and
used to accept or reject that individual update, rather than to terminate a
loop. The motivation is a vehicle refitting during operation, where a refit on
uninformative data can move the model the wrong way and there is no loop to
terminate.

Headline: it does not work well, and the reason is structural rather than a
matter of threshold choice.

## Setup

Hopf normal form, which is the system the preprint uses for its on-attractor
result:

```
xdot = mu*x - omega*y - A*x*(x^2 + y^2)
ydot = omega*x + mu*y - A*y*(x^2 + y^2)
```

with `mu=0.5, omega=1.0, A=1.0`, giving a stable limit cycle at radius
`sqrt(mu/A) ≈ 0.707`. Trajectories are integrated with `solve_ivp` and a degree-3
polynomial library is evaluated on the resulting states (9 terms, no bias).
Derivatives are taken analytically from the ODE, then corrupted with Gaussian
noise. The model has 8 nonzero true coefficients across the two equations.

The knob is where initial conditions live:

- `concentration = 0` — radii spread from near the origin out past the cycle.
  Well-conditioned library.
- `concentration = 1` — every trajectory starts exactly on the limit cycle.
  All samples lie on a 1-D curve in a 2-D state space, the polynomial library
  loses rank, condition number reaches ~1e11. This is the on-attractor case.

Fitting uses the E-SINDy path from this repo: STLSQ at `threshold=7e-3`,
`alpha=5e-5`, B=20 bootstrap members, one library column dropped per member,
element-wise median aggregation.

Scoring is relative L2 error of the **median** (deployed) model against the
known analytic coefficients, so spurious terms are penalized alongside wrong
values. 336 runs.

## Finding 1: bootstrap spread does not scale with the severity of rank deficiency

Within a single cell (`N=50`, `noise=0.001`):

| concentration | cond | rel. error | terms missed | mean_cv |
|---|---|---|---|---|
| 0.90 | 2.9e3 | 0.262 | 0 | 3.40 |
| 1.00 | 9.6e11 | 0.852 | 6 of 8 | 0.92 |

The catastrophic case — six of the eight true coefficients lost — produces a CV
**3.7x lower** than a case with 3.3x less error. This inversion appears at every
N: `mean_cv` at `concentration=1.0` sits between 0.86 and 1.27 across the whole
sweep while errors an order of magnitude smaller score far higher elsewhere.

The mechanism is that the two quantities measure different things. Bootstrap
resampling perturbs *which samples* are drawn. Every resample of on-attractor
data is still on-attractor, so every member inherits the same rank deficiency,
makes the same mistake, and agrees with the others. Ensemble spread is a measure
of sampling variability; loss of rank in the data manifold is not sampling
variability.

Two secondary effects reinforce this. Under rank deficiency the fit drops true
terms rather than scattering them, so `n_active` falls from 9-16 down to 6 and
the statistic is averaged over fewer, more-agreed-upon terms. And the surviving
terms are precisely the ones the degenerate geometry determines consistently.

This is the finding that matters for the paper. Ensemble disagreement is well
matched to *conditioning that varies across resamples* and poorly matched to
*conditioning that is a property of the trajectory*.

## Finding 2: the gate works where it is not needed

Lift over the majority-class baseline, per (N, noise) cell, `err_tol=0.05`,
`mean_cv`:

| N | noise | baseline | accuracy | lift | Spearman(CV, err) |
|---|---|---|---|---|---|
| 50 | 0.001 | 71.4% | 73.8% | +2.4% | −0.077 |
| 50 | 0.01 | 100% | — | one class | −0.253 |
| 100 | 0.001 | 52.4% | 61.9% | +9.5% | +0.371 |
| 100 | 0.01 | 78.6% | 76.2% | **−2.4%** | −0.209 |
| 250 | 0.001 | 66.7% | 76.2% | +9.5% | +0.277 |
| 250 | 0.01 | 73.8% | 73.8% | **+0.0%** | −0.355 |
| 1000 | 0.001 | 61.9% | 90.5% | +28.6% | +0.738 |
| 1000 | 0.01 | 66.7% | 88.1% | +21.4% | +0.643 |

At N=1000 the statistic is genuinely informative: lift above +20% and rank
correlation above 0.6. At N ≤ 250 it is near-useless, with two cells at zero or
negative lift and rank correlations that are negative in four of six.

The criterion was proposed for the ultra-low-data limit. That is the regime
where, as a per-refit gate, it performs worst.

## Finding 3: the pooled gate is real but small, and survives held-out validation

Threshold chosen on even seeds, scored on odd:

| statistic | baseline | in-sample acc | in-sample lift | held-out acc | held-out lift |
|---|---|---|---|---|---|
| mean_cv | 71.4% | 78.3% | +6.8% | 70.8% | +6.5% |
| median_cv | 71.4% | 72.0% | +0.6% | 64.9% | +0.6% |

`mean_cv` loses almost nothing out of sample, so the +6.8% is not threshold
overfitting. It is also not much of a gate. `median_cv` is worthless here; its
selected threshold of 0.0528 rejects nearly everything and scores baseline plus
noise.

Held-out validation holds up across all three sweeps in this document —
in-sample and out-of-sample lift differ by under 4 points everywhere — so none
of these numbers are threshold overfitting. They are simply small.

## Finding 4: statistic choice does not transfer between problems

On the linear synthetic (below) `median_cv` beat `mean_cv` consistently. On the
Hopf data the ordering reverses and `mean_cv` wins by a wide margin. Any
preference between the two was an artifact of the problem it was tuned on.

## Scaffolding: what was tried first and did not work

Before the Hopf port, the same questions were asked of a linear synthetic — a
standard-normal design matrix with one planted correlated column, collinearity
swept via that correlation. That synthetic is not a SINDy library, and neither
is `refit_profile.make_buffer`, which evaluates a polynomial library on iid
Gaussian points where `x` and `x^2`, and `x^2` and `y^2`, are uncorrelated by
construction. Both give near-orthogonal libraries. Recorded here because the
negative results were informative:

- **N=8000, noise=1e-3 produced zero bad runs in 120 attempts** even at
  condition number 142. Collinearity costs nothing when data is plentiful; the
  bootstrap median absorbs it. The pathology is genuinely a low-data phenomenon.
- **No absolute threshold transfers across operating regimes.** A run at
  `d=6, N=250, noise=0.01` scored `mean_cv=11.32` at 1.4% error (good), while
  `d=27, N=250, noise=0.1` scored 1.34 at 16.6% error (bad). The CV's scale
  moves with library size, buffer size and noise.
- **Library-size dilution.** At matched collinearity, d=27 CVs ran 5-10x below
  d=6 CVs. The mean over active terms averages one bad term against however many
  clean ones exist.
- **Baseline-relative gating underperformed a plain absolute threshold.** On the
  low-noise sweep it reached +3.1% and +5.2% lift, against +18.8% and +21.4%
  for a pooled absolute threshold on the same runs. The regime dependence is not
  a simple multiplicative offset.
- **Noise-scoping did not isolate the effect.** Holding noise at 0.01 across 384
  runs gave 78.4% and 81.0% raw accuracy, close to the mixed-noise sweep.
- **Tolerance sensitivity.** Varying the definition of "bad" from 2% to 20%
  error left `mean_cv` flat (81.2 / 81.2 / 81.2 / 83.3%) while `median_cv`
  climbed (79.2 / 83.3 / 85.4 / 89.6%), suggesting the statistic separated gross
  failures better than marginal ones. **This did not survive the Hopf port** —
  see Finding 1, where the grossest failure in the sweep is scored low.

Re-scored with the current analysis script, the three sweeps give:

| sweep | n | baseline | mean_cv held-out lift | median_cv held-out lift | median Spearman |
|---|---|---|---|---|---|
| linear, mixed noise | 320 | 75.0% | +4.4% | +5.0% | 0.549 |
| linear, low noise only | 384 | 59.6% | +18.2% | +20.3% | 0.665 |
| Hopf trajectories | 336 | 71.4% | +6.5% | +0.6% | 0.277 |

The low-noise linear sweep is the outlier, not the rule. Its +20% lift is the
most favourable number produced anywhere in this work, and it came from the
least realistic problem: a design matrix with a single planted correlated column
and no dynamics. The mixed-noise linear sweep already falls to +5%, in line with
the trajectory data.

Rank correlation between CV and actual error orders as 0.67, 0.55, 0.28 across
those three sweeps, which is consistent with the statistic degrading as the
problem becomes more realistic. That reading is suggestive rather than
established: three configurations, and the two linear sweeps differ in noise
composition as well as in how closely they resemble a SINDy problem.

The 76.9% and 81.2% figures quoted below for the pre-support-restriction code
predate that change to `gate.py` and are not reproducible from the current
revision without reverting it.

## Two implementation choices not specified in the paper

**Condition the CV on term retention.** Under library dropout a member shows a
zero coefficient for two different reasons: the fit killed the term, or the
member was never offered it. Averaging over all members conflates these. The CV
here is computed only over members that were offered a term and kept it.

**Restrict scoring to the median's support.** Retention conditioning alone is
not enough. At low N members disagree violently on junk terms that the
element-wise median then discards, so the CV goes large while the deployed model
is fine. Scoring only terms nonzero in the median puts the statistic on the same
object the error metric measures. On the linear synthetic this raised pooled
accuracy from 76.9% to 80.6% (`mean_cv`) and 81.2% to 83.4% (`median_cv`).

Both choices are forced by the per-refit framing. A stopping rule watches one
sequence converge and can ignore them; a gate scoring an individual fit cannot.

## Limitations

- **N and conditioning are confounded in the Hopf sweep.** Trajectory count
  scales with N at fixed points-per-trajectory, so condition number at
  `concentration=0` falls from 852 at N=50 to 7.4 at N=1000. "The gate works at
  high N" and "the gate works when the data is well-conditioned" are not
  separated. Decoupling needs one sweep varying points-per-trajectory at fixed
  trajectory count and another doing the reverse.
- **Derivatives are analytic, not finite-differenced.** This is optimistic; real
  derivative estimation adds error correlated with the state.
- **One system, one library degree, one dropout column.** N_DROP=1 out of 9 may
  probe rank deficiency poorly — dropping a single column from an already
  rank-deficient library changes little.
- **No closed loop.** Everything here is offline. Whether these results hold
  when ill-conditioning arises endogenously from a controller making the input a
  function of the state is untested, and that is the case that motivated the
  work.
- **Thresholds outside Finding 3 and the re-scored table are in-sample** and
  should be read as upper bounds.

## Open question this raises

If ensemble disagreement cannot detect on-attractor degeneracy, a per-refit gate
needs a second signal that reads the data geometry directly. Condition number of
the library on the current buffer is the obvious candidate, and it separated
these cases cleanly (7.4 to 1e11 across the concentration knob) where CV did
not. Whether a cheap conditioning statistic can be computed inside a refit
budget, and whether it composes with the CV rather than replacing it, is the
next thing to test.

## Reproduce

```
python test_gate.py                                     # sanity check
python sweep_hopf.py  --out gate_calib_hopf.csv         # Hopf sweep
python analyze_calib.py --csv gate_calib_hopf.csv       # lift + held-out

python sweep_gate.py  --out gate_calib.csv              # linear scaffolding
python analyze_calib.py --csv gate_calib.csv
```
