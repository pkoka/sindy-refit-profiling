# bench_njobs.py
import sys, time
from refit_profile import make_buffer
from refit_parallel import ensemble_parallel

def bench(theta, y, n_models, nj, reps=3):
    ensemble_parallel(theta, y, n_models, nj)    # warm-up: pays pool creation
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        ensemble_parallel(theta, y, n_models, nj)
        ts.append(time.perf_counter() - t0)
    return min(ts)

if __name__ == "__main__":
    N        = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
    degree   = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    n_models = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    theta, y = make_buffer(N, degree)
    print(f"N={N} degree={degree} B={n_models} d={theta.shape[1]}")
    for nj in (1, 2, 3, 4, 5, 6, 8, 10):
        print(f"{nj:3d}  {bench(theta, y, n_models, nj):.3f}s")