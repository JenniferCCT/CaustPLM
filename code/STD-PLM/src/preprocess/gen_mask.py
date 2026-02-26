import os
import numpy as np

# ===== 你只需要改這裡 =====
true_path = "../../../../data/traffic/miss_data/PEMS03/true_data_SR-TR_0.9_v1.npz"
miss_path = "../../../../data/traffic/miss_data/PEMS03/miss_data_SR-TR_0.9_v1.npz"
missing_ratio = 0.9
seed = 2024
# ==========================

z = np.load(true_path, allow_pickle=True)
if "data" not in z.files:
    raise KeyError(f"'data' not found in {true_path}. keys={z.files}")

X = z["data"].astype(np.float32)
if X.ndim != 3:
    raise ValueError(f"Expect data shape (T,N,F), got {X.shape}")

T, N, F = X.shape

rng = np.random.default_rng(seed)

# mask: 1=observed, 0=missing
mask = (rng.random((T, N, F)) >= missing_ratio).astype(np.int64)

os.makedirs(os.path.dirname(miss_path), exist_ok=True)
np.savez_compressed(miss_path, mask=mask)

obs_ratio = mask.mean()
print("Saved:", miss_path)
print(f"mask shape={mask.shape}, dtype={mask.dtype}")
print(f"obs_ratio={obs_ratio:.6f}, missing_ratio={1-obs_ratio:.6f}")