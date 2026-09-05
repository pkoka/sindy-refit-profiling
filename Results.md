# Results

E-SINDy refit cost under member-level parallelism, on a laptop and on embedded
flight-class hardware.

Predictions in `PREDICTIONS.md` were written and committed before any Jetson run
and have not been edited since. This file scores them.

## Hardware

| | M5 Pro | Jetson Orin Nano Super |
|---|---|---|
| cores | 15 | 6x Cortex-A78AE @ 1.728 GHz |
| RAM | shared | 7.3 GB LPDDR5 |
| OS | macOS 26.5.1 | Ubuntu 24.04.4, L4T 6.8.12-1021-tegra |
| power mode | plugged in, Low Power off | MAXN_SUPER (mode 2), `jetson_clocks` applied |
| BLAS | Accelerate | OpenBLAS |

Identical package versions on both: numpy 2.5.2, scipy 1.18.1, scikit-learn
1.9.0, pysindy 2.1.0, joblib 1.5.3.

Jetson runs were done headless over SSH with the graphical target stopped
(`systemctl isolate multi-user.target`), no containers running, and no other
users on the box.

Timing protocol: one warm-up call, then min and p90 over 7 reps. `min` estimates
what the machine can do. `p90` is what decides whether a control-loop deadline is
met. Both are reported for every row.

## Headline

**Best refit rate achieved anywhere: 1.55 Hz** (Jetson, anchor config, loky,
n_jobs=6, `min` of 7 reps; 1.44 Hz at p90).

The published target cadence is 10 Hz. That is **6.45x short**, and hitting it
from the measured 1.990 s serial time would require a 19.9x speedup on 6 cores.
The large config reaches 0.129 Hz, roughly 78x short.

For reference the 15-core M5 Pro peaked at 5.72 Hz, still 1.7x short.

[Certain] Naive member-level parallelism does not reach a 10 Hz E-SINDy refit
cadence on this hardware. The gap is an order of magnitude, not a tuning problem.

## Scorecard

| quantity | predicted | actual | verdict |
|---|---|---|---|
| anchor serial | 1.5 s (1.24-1.71) | 1.990 s | miss, high |
| anchor peak speedup | 1.9x (1.6-2.2) | 3.09x loky @6 | miss, high |
| anchor best Hz (min) | 1.3 (0.9-1.8) | 1.55 | **hit** |
| anchor best Hz (p90) | 1.2 (0.8-1.7) | 1.44 | **hit** |
| anchor winning backend | threading | **loky**, by 1.50x | **wrong** |
| large serial | 17.0 s (14.2-19.5) | 24.278 s | miss, high |
| large peak speedup | 3.6x (3.0-4.2) | 3.11x | hit, low edge |
| large best Hz | 0.21 (0.15-0.30) | 0.129 | miss, low |
| large winning backend | loky, wider margin | tie (0.7%) | **wrong** |
| claim 1 (crossover driven by d) | narrows on Jetson | **inverts** | **falsified** |
| claim 2 (M5 p90 blowup is Apple-specific) | no pathology on Jetson | max 1.45, no pathology | **confirmed** |
| claim 3 (BLAS pinning is a null) | null within 10% | 34.8% effect | **falsified** |

Three of four registered falsifiers fired. The headline prediction held.

The per-core slowdown factor R was predicted at 4.8 (4.0-5.5). Measured 6.42x on
the anchor config and 6.84x on the large config, both outside the band.

## Finding 1: the backend crossover inverted

Predicted: threading's anchor-config advantage narrows on the Jetson.
Observed: it reverses.

Best time by backend, both machines, BLAS pinned:

| config | M5 loky | M5 threading | Jetson loky | Jetson threading |
|---|---|---|---|---|
| anchor N=16000 d=27 B=20 | 0.1876 s | **0.1747 s** | **0.645 s** | 0.957 s |
| large N=32000 d=83 B=40 | **1.1810 s** | 1.2966 s | 7.800 s | **7.749 s** |

On the anchor config loky is 1.50x faster than threading on the Jetson, having
been 7% slower on the M5. On the large config the two are within 0.7%, where on
the M5 loky led by 9%.

The shape is more telling than the ratio. Jetson threading on the anchor config
saturates at n_jobs=4 (2.00x, 2.07x, 2.05x at 4, 5, 6) while loky is still
climbing at n_jobs=6 (2.62x, 2.94x, 3.09x) and has not saturated at the core
count. That is the signature of a serialized region, not of transfer cost.

[Likely] Mechanism: interpreter-bound and LAPACK-bound code do not slow down by
the same factor on this hardware. Same-code cross-machine ratios:

- `refit_parallel` path (thin Python wrapper around lstsq): R = 6.42x
- `refit_profile` path (`EnsembleOptimizer`, more Python per member): R = 8.97x

[Guessing] The two are measured at different N (16000 vs 8000), so cache effects
are not excluded and the gap is suggestive rather than conclusive.

If interpreter time inflates faster than LAPACK time, then relative to compute,
loky's pickling overhead *shrinks* on the Jetson while GIL-held Python time in
the STLSQ thresholding loop *grows*. The anchor config has the higher GIL-held
fraction because its lstsq (d=27) is small relative to the surrounding Python.
That predicts exactly what was observed: threading saturates early on the anchor
config and not on the large one.

This supersedes the transfer-cost account offered when only M5 data existed.
That account was already weak, since bytes-per-work was nearly identical across
the two M5 configs (4.6 vs 4.2 ms per MB) and so could not explain a crossover.

## Finding 2: parallel efficiency is higher on the slower machine

Efficiency at n_jobs=6 on both machines, so cores are matched:

| config, backend | M5 Pro | Jetson |
|---|---|---|
| anchor, loky | 24.3% | **51.4%** |
| anchor, threading | 27.5% | 34.2% |
| large, loky | 47.1% | **51.7%** |
| large, threading | 45.5% | **51.7%** |

[Certain] The Jetson parallelizes this workload more efficiently than the M5 Pro
at equal core count, on three of four config-backend pairs.

[Likely] Not because its cores are better. Because they are slow enough that
fixed per-dispatch overhead stops dominating. Per-member work rises from 15.6 ms
to 99.5 ms on the anchor config while joblib's dispatch cost is roughly
platform-constant, so the same overhead is amortized over 6.4x more compute.

[Guessing] A second contributor is that 6 homogeneous cores schedule more
predictably than the M5's heterogeneous mix. Not separately measured.

This inverts the naive reading of the M5 numbers. The 10.7% efficiency observed
there was not a property of the algorithm. It was a property of running a small
workload on fast cores.

## Finding 3: the M5 tail-latency pathology is Apple-specific

Registered claim 2 predicted p90/min would stay under about 1.3 on the Jetson
with no loky pathology, falsified if any row exceeded 2x.

| | max p90/min | where |
|---|---|---|
| M5 Pro, pinned | **4.29x** | anchor, loky, n_jobs=12 (min 0.1939 s, p90 0.8326 s) |
| M5 Pro, unpinned | **4.52x** | anchor, loky, n_jobs=15 |
| Jetson, pinned | 1.12x | anchor, loky, n_jobs=5 |
| Jetson, unpinned | 1.45x | large, threading, n_jobs=4 (oversubscribed, see Finding 4) |

[Certain] Confirmed. Across all 48 Jetson rows nothing approaches the 2x
falsifier, and the worst case sits in the deliberately pathological unpinned
threading configuration rather than in normal operation. In the pinned
configuration, which is the valid one, the worst tail is 1.12x.

[Likely] On the M5 the blowup appears only at n_jobs >= 10, past the P-core
count, i.e. once workers spill onto E-cores. [Guessing] The exact P/E split was
not verified, so the attribution to core heterogeneity is inference from the
n_jobs threshold, not a measurement.

**This must be reported as an Apple heterogeneous-scheduling artifact, not as a
general property of process pools.** Reporting it the other way would have been
the natural conclusion from the M5 data alone, and would have been wrong. This is
what pre-registering the falsifier bought.

## Finding 4: BLAS pinning, a real effect and a measurement artifact

Registered claim 3 predicted a null within 10%, falsified above 25%.

Serial (n_jobs=1) times, pinned vs unpinned:

| config | pinned | unpinned | effect |
|---|---|---|---|
| Jetson large (d=83) | 24.278 s | 18.014 s | unpinned **34.8% faster** |
| Jetson anchor (d=27) | 1.990 s | 2.169 s | unpinned 9.0% *slower* |
| M5 large | 3.5474 s | 3.5644 s | 0.5%, null |
| M5 anchor | 0.3102 s | 0.3325 s | 7.2%, null |

[Certain] Falsified on the Jetson large config. [Likely] The sign flip between
configs is the expected one: a 32000x82 lstsq is large enough for multithreaded
OpenBLAS to pay off, a 16000x26 one is not, and thread setup costs more than it
returns. [Guessing] The M5 null is attributable to Accelerate's different
threading policy, not separately verified.

### The artifact, and which CSV is authoritative

The 34.8% figure does **not** propagate to any achieved result:

| config | pinned best | unpinned best |
|---|---|---|
| anchor, loky | 0.645 s @6 | 0.643 s @6 |
| large, loky | 7.800 s @5 | 7.760 s @6 |

[Certain] Best achievable times are invariant to pinning at n_jobs >= 2, to
within 0.5%. Only the n_jobs=1 baseline moves.

[Certain] Cause: joblib dispatches `n_jobs=1` through `SequentialBackend`,
in-process, with no worker spawn. Unpinned, that baseline inherits all 6
OpenBLAS threads, while every `n_jobs >= 2` loky worker has its inner thread count
limited automatically. The n_jobs=1 row therefore measures a different execution
model from every other row in its own sweep.

Consequence: the unpinned speedup column is inflated at the baseline, not at the
top. Loky large config reads 3.11x pinned and 2.32x unpinned while the underlying
best times differ by 0.5%.

**`parallel_times_jetson_pinned.csv` is the authoritative dataset.**
`parallel_times_jetson_unpinned.csv` is retained as a methods note and as the
source of Finding 5. Any speedup figure computed against an unpinned baseline
should be disregarded.

## Finding 5: threading plus unpinned BLAS is catastrophic

Unrequested and the cleanest single result in the set. It exists only because the
falsifier for claim 3 required running the leg.

Jetson, large config, threading backend, BLAS unpinned:

| n_jobs | min | speedup |
|---|---|---|
| 1 | 17.969 s | 1.00x |
| 2 | 19.630 s | 0.92x |
| 3 | 22.186 s | 0.81x |
| 4 | 25.132 s | 0.71x |
| 5 | 35.437 s | 0.51x |
| 6 | 38.655 s | **0.46x** |

[Certain] Monotonically worse with every added worker, ending 2.15x slower than
its own serial baseline.

[Certain] Cause: joblib limits inner BLAS threads in loky workers but does **not**
do so under the threading backend. Six joblib threads each spawning up to six
OpenBLAS threads is 36 threads on 6 cores.

[Likely] The same effect appears in attenuated form on the anchor config
(threading peaks at 1.31x unpinned versus 2.07x pinned), smaller because d=27
gives OpenBLAS less to thread over.

Practical consequence: **`prefer="threads"` in joblib requires explicit BLAS
thread limiting.** Without it, adding workers makes the workload slower, and the
degradation is silent.

## Thermal

Peak CPU temperature across all sweeps: **59.562 C** (`tegrastats_run.log`,
1 s interval). Orin Nano throttles in the 95-100 C region.

[Certain] Thermal throttling is excluded as a confound. Passive cooling was
sufficient for a run of this length, consistent with the p90/min ratios never
exceeding 1.12 in the pinned configuration.

## Correctness

Both gates passed on the Jetson before any timing was collected.

- `--check`, parallel implementation vs `EnsembleOptimizer` over 10 trials:
  stable support 20 vs 20, zero support disagreement in either direction,
  median relative difference 0.0000 (max 0.0001), member-spread ratio median
  1.123 (p10 0.652, p90 1.627). PASS.
- `--check-backends`, loky vs threading: bit-identical, max absolute difference
  0.000e+00. PASS.

[Certain] The backend comparison is therefore a pure timing comparison. Backend
choice cannot change the answer, only how long it takes.

## Open items

1. **The 10 Hz target itself is unverified.** SINDy-RL Table S4's refit cadence
   has not been confirmed as per-control-step rather than per-episode. If it is
   per-episode, "6.45x short" is measured against a target that does not exist in
   the assumed form. This is the cheapest open item and it conditions the
   headline.
2. **`n_subset` inconsistency between harnesses.** `refit_profile.time_refit`
   constructs `EnsembleOptimizer` without `n_subset`, while `check_correctness`
   passes `n_subset=n_samples` and `fit_one_member` bootstraps at full size. If
   pysindy defaults to a fraction, `refit_times*.csv` and `parallel_times*.csv`
   are not directly comparable. Speedup columns within `parallel_times*.csv` are
   unaffected, since their baseline is n_jobs=1 of the same function. Cross-machine
   R ratios are also unaffected, since each compares a harness to itself.
3. **Finding 1's mechanism is inferred, not isolated.** Separating GIL-held time
   from LAPACK time would need per-phase instrumentation inside the STLSQ loop.
   The interpreter-vs-LAPACK ratio argument is consistent with the data but is
   not a direct measurement.

## Files

| file | contents |
|---|---|
| `parallel_times_jetson_pinned.csv` | authoritative Jetson sweep, 24 rows |
| `parallel_times_jetson_unpinned.csv` | methods note and Finding 5, 24 rows |
| `parallel_times_m5_pinned.csv` | M5 Pro sweep, 40 rows |
| `parallel_times_m5_unpinned.csv` | M5 Pro sweep, 40 rows |
| `refit_times_jetson_quick.csv` | smoke test, N=2000 and N=8000 |
| `refit_times.csv` | M5 Pro sequential grid |
| `tegrastats_run.log` | 1 s thermal and power trace, full Jetson session |
| `PREDICTIONS.md` | pre-registered, unedited |
| `jetson_setup.sh` | environment setup, non-destructive to the host |
