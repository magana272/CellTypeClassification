import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

SRC = 'data/10x/matrix.npy'
OUT = 'data/10x/bench_out.npy'
N_ROWS = 7653
SEED = 0
TRIALS = 3

def run_single(X, rows):
    mm = np.lib.format.open_memmap(OUT, mode='w+', dtype=X.dtype, shape=(len(rows), X.shape[1]))
    np.take(X, rows, axis=0, out=mm)
    del mm

def run_threaded(X, rows, n_threads):
    mm = np.lib.format.open_memmap(OUT, mode='w+', dtype=X.dtype, shape=(len(rows), X.shape[1]))
    bounds = np.linspace(0, len(rows), n_threads + 1, dtype=int)
    def copy(s, e):
        np.take(X, rows[s:e], axis=0, out=mm[s:e])
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        list(ex.map(lambda b: copy(*b), zip(bounds[:-1], bounds[1:])))
    del mm

def time_it(fn):
    if os.path.exists(OUT):
        os.remove(OUT)
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    os.remove(OUT)
    return dt

def main():
    X = np.load(SRC, mmap_mode='r')
    print(f'Source: {X.shape} {X.dtype} ({X.nbytes/1e9:.2f} GB)')
    rng = np.random.default_rng(SEED)
    rows = np.sort(rng.choice(X.shape[0], size=N_ROWS, replace=False)).astype(np.int64)
    out_gb = N_ROWS * X.shape[1] * 4 / 1e9
    print(f'Output per trial: {N_ROWS} rows ({out_gb:.2f} GB)')
    print()

    configs = [
        ('single', lambda: run_single(X, rows)),
        ('threads=4', lambda: run_threaded(X, rows, 4)),
        ('threads=8', lambda: run_threaded(X, rows, 8)),
        ('threads=16', lambda: run_threaded(X, rows, 16)),
    ]

    results = {name: [] for name, _ in configs}
    for trial in range(TRIALS):
        order = configs if trial % 2 == 0 else list(reversed(configs))
        for name, fn in order:
            dt = time_it(fn)
            results[name].append(dt)
            print(f'trial {trial} {name:12s}: {dt:6.2f}s')
        print()

    print('=== summary (seconds) ===')
    for name, _ in configs:
        ts = results[name]
        print(f'{name:12s} min={min(ts):5.2f} mean={sum(ts)/len(ts):5.2f} max={max(ts):5.2f} all={[round(t,2) for t in ts]}')

if __name__ == '__main__':
    main()
