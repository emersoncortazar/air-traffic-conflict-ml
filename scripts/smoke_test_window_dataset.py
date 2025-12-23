from time import perf_counter

from atcml.ml.window_dataset import WindowDatasetConfig, load_window_sequences

cfg = WindowDatasetConfig(
    traj_csv="data/trajectories.csv",
    windows_csv="data/windows.csv",
    history_s=60.0,
    dt_s=1.0,
)

print("load_window_sequences from:", load_window_sequences.__code__.co_filename)

t0 = perf_counter()
X, y, meta = load_window_sequences(cfg)
t1 = perf_counter()

print(f"Loaded in {t1 - t0:.2f}s")
print("X.shape:", X.shape, "dtype:", X.dtype)
print("y.shape:", y.shape, "dtype:", y.dtype, "pos_rate:", float(y.mean()))
print("meta.shape:", meta.shape)
print("first meta:", meta[0] if len(meta) else None)

# sanity checks
assert X.ndim == 3, "X should be (N, T, F)"
assert len(X) == len(y) == len(meta), "X, y, meta should align"
assert X.shape[1] > 1 and X.shape[2] > 0, "T and F should be > 0"

print("OK ✅")
