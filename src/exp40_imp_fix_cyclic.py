import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from proxy import evaluate, IMP_PARAMS
from exp34_devfeat import multi_member, DEEP, SLOW
L = [["building", "hour"], ["building"]]
def cyc(member):
    """wrap a member so imputers also see sin/cos of hour and month"""
    def add(F):
        G = F.copy(); h = G["hour"].to_numpy(dtype=float); m = G["month"].to_numpy(dtype=float)
        G["h_sin"], G["h_cos"] = np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)
        G["m_sin"], G["m_cos"] = np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12); return G
    def fit(F0, col, w):
        f = member(add(F0), col, w); return lambda G: f(add(G))
    return fit
g = sys.argv[1]
TRIO = lambda: [multi_member(IMP_PARAMS, L, 11, 5), multi_member(DEEP, L, 12, 5), multi_member(SLOW, L, 13, 5)]
if g == "fixed": evaluate(TRIO(), 2.0, label="c09 cfg, mask order FIXED")
if g == "cyclic": evaluate([cyc(m) for m in TRIO()], 2.0, label="c09 cfg fixed + cyclic hour/month")
