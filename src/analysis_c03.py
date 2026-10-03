import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
from evalkit import segment_frame, seg, summary
N = "final_cmp_V3_5x4"
print(summary(N))
d = segment_frame(N, use_masked=True)
for q in ["previous_usage", "occupancy", "temperature", "humidity"]:
    d[q + "_q"] = pd.qcut(d[q], 10, duplicates="drop")
def show(by, minn=0, sort=None):
    t = seg(d, by); t = t[t.n >= minn]
    t["flag"] = np.where(np.abs(t.bias) > 2.5 * t.bias_se, "*", "")
    print(f"\n== by {by}"); print(t.sort_values(sort) if sort else t)
for by in ["building_id", "building_type", "hour", "weekend", "month", "pattern", "n_missing", "miss_previous_usage",
           "miss_occupancy", "spike", "previous_usage_q", "occupancy_q", "temperature_q", "humidity_q"]:
    show(by)
show(["spike", "miss_occupancy"]); show(["spike", "miss_previous_usage"]); show(["weekend", "spike"])
show(["building_type", "spike"])
show(["building_type", "weekend"])
t = seg(d, ["building_type", "hour"]); t["flag"] = np.where(np.abs(t.bias) > 3 * t.bias_se, "*", "")
print("\n== building_type x hour: cells with |bias|>3se"); print(t[t.flag == "*"])
d.to_pickle("experiments/seg_c03.pkl")
