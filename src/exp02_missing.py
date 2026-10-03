import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb
from common import load, cross_validate, TARGET, apply_masks, test_mask_patterns, NUM
from features import make_features
from exp01_baselines import Wrap
tr, te = load(); P = test_mask_patterns(te)
pats_nonzero = P[P.any(1)]
def aug(frac, mult=1):
    def f(d, seed):
        rng = np.random.default_rng(seed); parts=[d]
        for i in range(mult):
            s = d.sample(frac=frac, random_state=seed+i)
            parts.append(apply_masks_nz(s, seed+i))
        return pd.concat(parts, ignore_index=True)
    return f
def apply_masks_nz(df, seed):
    rng = np.random.default_rng(seed); m = pats_nonzero[rng.integers(0, len(pats_nonzero), len(df))]
    out = df.copy(); v = out[NUM].to_numpy(float); v[m] = np.nan; out[NUM] = v; return out
L = lambda: Wrap(lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.01, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4))
if __name__ == "__main__":
    cross_validate("m01_aug_nz_0.3", L, tr, te, augment=aug(0.3))
    cross_validate("m02_aug_nz_1.0", L, tr, te, augment=aug(1.0))
    cross_validate("m03_aug_nz_1.0x2", L, tr, te, augment=aug(1.0, 2))
