# Jetson Orin Nano predictions

Written 2026-09-03, before the first Jetson run. Numbers derived from
parallel_times_m5_pinned.csv. Not revised after seeing Jetson results.

## Reference (M5 Pro, 15 cores, BLAS pinned)

| config | d | B | serial s | ms/member | peak speedup | best Hz (min) |
|---|---|---|---|---|---|---|
| anchor  N=16000 deg=2 | 27 | 20 | 0.3102 | 15.6 | 1.79x threading @15 | 5.72 |
| large   N=32000 deg=3 | 83 | 40 | 3.5474 | 88.7 | 2.99x loky @8      | 0.84 |

At n_jobs=6, the Jetson's core count:
anchor threading 0.1894s / loky 0.2128s. large loky 1.2565s / threading 1.3382s.

## Model

  jetson_serial = m5_serial * R,  R = per-core slowdown, A78AE @1.5GHz vs M5 P-core
  jetson_best   = jetson_serial / S,  S = peak speedup on 6 homogeneous cores

R = 4.8 central (4.0 to 5.5). Clock ratio ~3x, plus IPC gap, plus memory
bandwidth 68 GB/s vs ~150 GB/s, and the bootstrap row gather is bandwidth heavy.

S: 6 equal cores beats the M5's 5P+10E mix on efficiency, but has less
bandwidth headroom.

## Predicted, anchor (N=16000, d=27, B=20)

- serial refit_s        1.5 s        (1.24 to 1.71)
- peak speedup          1.9x         (1.6 to 2.2), at n_jobs = 5 or 6
- best Hz (min)         1.3          (0.9 to 1.8)
- best Hz (p90)         1.2          (0.8 to 1.7)
- winning backend       threading, but by a smaller margin than on the M5

## Predicted, large (N=32000, d=83, B=40)

- serial refit_s        17.0 s       (14.2 to 19.5)
- peak speedup          3.6x         (3.0 to 4.2), at n_jobs = 6
- best Hz (min)         0.21         (0.15 to 0.30)
- winning backend       loky, by a wider margin than on the M5

## Mechanism claims, each independently falsifiable

1. The backend crossover is driven by d, not by compute-per-byte.
   Bytes-per-work is nearly identical across the two M5 configs
   (4.6 vs 4.2 ms per MB transferred), so transfer cost cannot explain why
   threading wins one and loses the other. The remaining candidate is
   GIL-held Python work in the STLSQ thresholding loop, which scales with d.
   PREDICTS: on the Jetson, where the interpreter is relatively slower and
   the GIL-held fraction rises for both configs, threading's anchor margin
   narrows and loky's large-config margin widens.
   FALSIFIED IF: threading's anchor margin widens, or loky wins the anchor.

2. Loky's p90 blowup is an Apple heterogeneous-scheduling artifact, not a
   general property of process pools. On the M5 it appears only at
   n_jobs >= 10, well past the 5 P-cores, i.e. once workers spill onto
   E-cores. Worst case pinned: anchor loky n_jobs=12, min 0.1939s but
   p90 0.8326s, a 4.3x tail.
   PREDICTS: on 6 homogeneous A78AE cores with the grid capped at 6, the
   p90/min ratio stays under ~1.3 for both backends and loky shows no
   pathology.
   FALSIFIED IF: loky p90/min exceeds 2x anywhere on the Jetson.
   THIS IS THE CLAIM MOST AT RISK OF BEING OVERSTATED. If it holds, the
   tail-latency result is about Apple silicon and must be reported that way,
   not as "process pools have bad tail latency."

3. BLAS oversubscription is not a factor. Pinned vs unpinned serial on the
   M5: anchor 0.3102 vs 0.3325, large 3.5474 vs 3.5644.
   PREDICTS: same null on aarch64 OpenBLAS, within 10%.
   FALSIFIED IF: the Jetson unpinned run differs from pinned by >25%.

## Headline falsifier

Best result anywhere in 40 M5 Pro rows is 5.72 Hz, on 15 cores, on the
easier config. If the Jetson reaches >= 10 Hz on the anchor config, the
dispatch-overhead account is wrong and this whole writeup needs rebuilding.

If the Jetson lands where predicted, the conclusion is:
naive member-level parallelism does not reach a 10 Hz E-SINDy refit cadence
on embedded flight hardware, and the gap is roughly an order of magnitude,
not a tuning problem.