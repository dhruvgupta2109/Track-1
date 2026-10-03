import numpy as np, pandas as pd
tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv")
D = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
A = pd.concat([tr.assign(src="train"), te.assign(src="test")], ignore_index=True)
A["d"] = A.day_of_week.map(D.index)
A["n"] = A.id.str[2:].astype(int)
def nxt(h, d, off):
    hh = h + off; return hh % 24, (d + hh // 24) % 7
key = lambda b, m, d, h: (b, m, d, h)
idx = {}
for i, r in A.iterrows():
    idx.setdefault(key(r.building_id, r.month, r.d, r.hour), []).append(i)
prev = A.previous_usage.values
def stat(off, sample=tr.index):
    diffs, exact, has = [], 0, 0
    for i in sample:
        r = A.loc[i]; h2, d2 = nxt(r.hour, r.d, off)
        c = [j for j in idx.get(key(r.building_id, r.month, d2, h2), []) if j != i and not np.isnan(prev[j])]
        if not c: continue
        has += 1; dd = np.min(np.abs(prev[c] - r.energy_usage)); diffs.append(dd); exact += dd < 0.05
    diffs = np.array(diffs)
    return has, exact, np.round(np.quantile(diffs, [.1, .25, .5]), 2) if len(diffs) else None
for off in [1, 2, 5, -1, 0]:
    has, ex, q = stat(off)
    print(f"offset {off:+d}h: train rows with a candidate {has}, exact (|d|<0.05) {ex} ({ex/max(has,1):.3f}), min|prev_j - y_i| quantiles 10/25/50%: {q}")
# also: does prev_i equal energy of the same building's row at hour-1 (backward check, train-only targets)
