import time
import numpy as np

def fake_one_event(rng):
    N, M, K = rng.integers(1, 5), rng.integers(1, 8), 20
    hist = rng.random((N, K, 4), dtype=np.float32)
    cand = rng.random((M, 4), dtype=np.float32)
    ctx = rng.random((N, 16, 6), dtype=np.float32)
    ch = rng.random((M, 128), dtype=np.float32)
    return hist, cand, ctx, ch

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 5000
    t0 = time.time()
    for _ in range(n):
        fake_one_event(rng)
    dt = time.time() - t0
    print(f"{n / dt:.0f} events/s (1 process)")
