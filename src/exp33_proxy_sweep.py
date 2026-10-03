import sys; sys.path.insert(0, "src")
from proxy import evaluate, lgb_member, IMP_PARAMS
DEEP = dict(IMP_PARAMS, num_leaves=63, min_child_samples=5, n_estimators=1000, learning_rate=0.02)
SLOW = dict(IMP_PARAMS, n_estimators=1500, learning_rate=0.01)
TRIO = [lgb_member(IMP_PARAMS, 11), lgb_member(DEEP, 12), lgb_member(SLOW, 13)]
grp = sys.argv[1]
if grp == "A":
    evaluate(TRIO, 1.0, label="c06 trio w1 (LB 3.238)")
    evaluate(TRIO, 2.0, label="c07 trio w2 (LB 3.231)")
    evaluate(TRIO, 3.0, label="trio w3")
    evaluate(TRIO, 4.0, label="trio w4")
elif grp == "B":
    evaluate(TRIO, 6.0, label="trio w6")
    evaluate(TRIO, 10.0, label="trio w10")
    evaluate(TRIO, 1.0, train_w=0.0, use_dom=False, label="trio test-only")
elif grp == "C":
    evaluate([lgb_member(IMP_PARAMS, 11, 2), lgb_member(DEEP, 12, 2), lgb_member(SLOW, 13, 2)], 2.0, label="trio w2 aug x2")
    evaluate([lgb_member(p, s) for p in (IMP_PARAMS, DEEP, SLOW) for s in (21, 22, 23)], 2.0, label="trio x3 seeds w2")
if grp == "D":
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor
    from imputed import _mask_aug
    from proxy import PN
    def et_member(seed=31):
        def fit(F0, col, w):
            F = _mask_aug(F0, col, PN, 1, seed=seed); ww = np.tile(w, 2); ok = F[col].notna().to_numpy()
            enc = lambda G: G.assign(building=G.building.cat.codes, btype=G.btype.cat.codes).fillna(-999)
            m = ExtraTreesRegressor(300, min_samples_leaf=3, max_features=0.7, n_jobs=3, random_state=seed).fit(enc(F.drop(columns=[col]))[ok], F.loc[ok, col], sample_weight=ww[ok])
            return lambda G: m.predict(enc(G.drop(columns=[col])))
        return fit
    evaluate([lgb_member(IMP_PARAMS, 11)], 1.0, use_dom=False, label="c04 pooled single (LB 3.337)")
    REG = dict(IMP_PARAMS, num_leaves=31, min_child_samples=30, reg_lambda=10, n_estimators=1500, learning_rate=0.015, colsample_bytree=0.7)
    evaluate(TRIO + [lgb_member(REG, 14)], 2.0, label="trio+reg w2")
    evaluate(TRIO + [et_member()], 2.0, label="trio+ET w2")
if grp == "E":
    evaluate([lgb_member(IMP_PARAMS, 11, 3), lgb_member(DEEP, 12, 3), lgb_member(SLOW, 13, 3)], 2.0, label="trio w2 aug x3")
if grp == "F":
    evaluate([lgb_member(IMP_PARAMS, 11, 5), lgb_member(DEEP, 12, 5), lgb_member(SLOW, 13, 5)], 2.0, label="trio w2 aug x5")
if grp == "G":
    evaluate([lgb_member(IMP_PARAMS, 11, 8), lgb_member(DEEP, 12, 8), lgb_member(SLOW, 13, 8)], 2.0, label="trio w2 aug x8")
if grp == "H":
    evaluate([lgb_member(p, s, 5) for p in (IMP_PARAMS, DEEP, SLOW) for s in (21, 22, 23)], 2.0, label="9 members w2 aug x5")
if grp == "I":
    evaluate([lgb_member(IMP_PARAMS, 11, 5), lgb_member(DEEP, 12, 5), lgb_member(SLOW, 13, 5)], 1.0, label="trio w1 aug x5")
