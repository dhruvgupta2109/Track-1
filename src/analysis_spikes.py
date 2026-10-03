import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
pd.set_option("display.width", 200)
from evalkit import segment_frame, seg, Z, TR
d = segment_frame("final_cmp_V3_5x4", use_masked=False)
for c in ["previous_usage", "occupancy", "temperature"]:
    zc = d["z_" + c]
    d["zb"] = pd.cut(zc, [-99, -5, -3, -2, 2, 3, 5, 99])
    print(f"\n== clean OOF residual by z of {c}"); print(seg(d, "zb")[["n", "rmse", "bias", "bias_se", "y", "pred"]])
# co-occurrence of spikes
for nm, df in [("train", Z)]:
    s = (df.abs() > 3)
    print("\nspike co-occurrence train (rows):"); print(pd.crosstab(s.previous_usage, s.occupancy))
    print("corr of z (spike rows only):"); print(df[(df.abs() > 3).any(axis=1)].corr().round(2))
