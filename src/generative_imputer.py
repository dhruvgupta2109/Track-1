"""Conditional Gaussian mixture with explicit deterministic feature schedules.

Fit on incomplete feature rows using EM conditional moments; no energy target.
Shared residual components, domain/hour-band mixture probabilities, robust
per-building scales. Intended to test a generator hypothesis, not assert it.
"""
import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from features import BUILDINGS, TYPES, BUILDING_TYPE, NUMERIC, make_features


def schedule(F):
    b = F.building.cat.codes.to_numpy(); h = F.hour.to_numpy(); dom = F.dom.to_numpy()[:, None]
    B = (b[:, None] == np.arange(12)).astype(float)
    T = B @ np.array([[BUILDING_TYPE[b] == t for t in TYPES] for b in BUILDINGS], float)
    H = (h[:, None] == np.arange(24)).astype(float)
    we = F.weekend.to_numpy()[:, None]
    m = F.month.to_numpy()
    return np.concatenate([B, B*dom, (T[:, :, None]*H[:, None, :]).reshape(len(F), -1),
                           B*we, (T[:, :, None]*(H*we)[:, None, :]).reshape(len(F), -1)*.5,
                           (T[:, :, None]*(H*dom)[:, None, :]).reshape(len(F), -1)*.3,
                           np.sin(2*np.pi*m/12)[:, None], np.cos(2*np.pi*m/12)[:, None]], 1)


def groups(F):
    h = F.hour.to_numpy()
    band = np.digitize(h, [4, 7, 10, 19, 22])
    return F.dom.to_numpy().astype(int)*6 + band


def moments(X, means, covs, probs, group, need_cov=True):
    """Posterior responsibilities and conditional first/second moments."""
    n, d = X.shape; k = len(means)
    logp = np.zeros((n, k)); ex = np.zeros((n, k, d)); vc = np.zeros((n, k, d, d)) if need_cov else None
    masks = np.isfinite(X)
    for mask in np.unique(masks, axis=0):
        rows = np.flatnonzero(np.all(masks == mask, axis=1)); obs = np.flatnonzero(mask); mis = np.flatnonzero(~mask)
        Z = X[np.ix_(rows, obs)]
        for j in range(k):
            C = covs[j]; ex[rows, j] = means[j]
            if len(obs):
                S = C[np.ix_(obs, obs)]; delta = Z-means[j, obs]
                gain = np.linalg.solve(S, C[np.ix_(obs, mis)]).T
                ex[rows[:, None], j, obs] = Z
                ex[rows[:, None], j, mis] = means[j, mis] + delta @ gain.T
                logp[rows, j] = -.5*(len(obs)*np.log(2*np.pi) + np.linalg.slogdet(S)[1] +
                                            np.sum(delta*np.linalg.solve(S, delta.T).T, axis=1))
                if need_cov and len(mis):
                    V = C[np.ix_(mis, mis)]-gain@C[np.ix_(obs, mis)]
                    vc[rows[:, None, None], j, mis[None, :, None], mis[None, None, :]] = V
            elif need_cov:
                vc[rows, j] = C
    logp += np.log(np.maximum(probs[group], 1e-12))
    z = logsumexp(logp, axis=1); resp = np.exp(logp-z[:, None])
    return resp, ex, vc, z


class GeneratorImputer:
    def __init__(self, components=8):
        self.components = components

    def fit(self, rows, domain):
        F = make_features(rows).assign(dom=domain); D = schedule(F); X = F[NUMERIC].to_numpy()
        b = F.building.cat.codes.to_numpy(); weights = np.where(np.asarray(domain) == 1, 2., 1.)
        self.coef = []; self.intercept = []; scale = np.ones((12, 4))
        mu = np.zeros_like(X)
        for j in range(4):
            ok = np.isfinite(X[:, j]); sw = weights.copy()
            for iteration in range(5):
                model = Ridge(alpha=10).fit(D[ok], X[ok, j], sample_weight=sw[ok])
                mu[:, j] = model.predict(D)
                r = X[:, j]-mu[:, j]
                for bb in range(12):
                    v = r[(b == bb) & ok]; med = np.median(v)
                    scale[bb, j] = max(float(np.median(np.abs(v-med))*1.4826), .2)
                sw = weights*np.minimum(1., 1.5*scale[b, j]/np.maximum(np.abs(r), 1e-6))
            self.coef.append(model.coef_); self.intercept.append(model.intercept_)
        self.scale = scale; R = (X-mu)/scale[b]
        # Complete rows only initialize; EM then incorporates every observed entry.
        complete = R[np.isfinite(R).all(axis=1)]
        init = GaussianMixture(self.components, random_state=20261003, n_init=1,
                               reg_covar=.02, max_iter=100).fit(complete)
        means = init.means_; covs = init.covariances_; probs = np.tile(init.weights_, (12, 1)); grp = groups(F)
        last = -np.inf
        for it in range(60):
            resp, ex, vc, ll = moments(R, means, covs, probs, grp)
            rw = resp*weights[:, None]; nk = rw.sum(axis=0)
            means = np.einsum("nk,nkd->kd", rw, ex)/nk[:, None]
            second = np.einsum("nk,nkde->kde", rw, vc) + np.einsum("nk,nkd,nke->kde", rw, ex, ex)
            covs = second/nk[:, None, None] - np.einsum("kd,ke->kde", means, means)
            covs += np.eye(4)[None]*.02
            prior = nk/nk.sum()
            for g in range(12):
                count = rw[grp == g].sum(axis=0)+40*prior
                probs[g] = count/count.sum()
            loss = float(np.average(ll, weights=weights))
            if it > 5 and abs(loss-last) < 1e-5:
                break
            last = loss
        self.means, self.covs, self.probs = means, covs, probs
        return self

    def predict(self, rows, domain=1):
        F = make_features(rows).assign(dom=domain); D = schedule(F)
        mu = D@np.asarray(self.coef).T+np.asarray(self.intercept)
        b = F.building.cat.codes.to_numpy(); X = F[NUMERIC].to_numpy(); R = (X-mu)/self.scale[b]
        resp, ex, _, _ = moments(R, self.means, self.covs, self.probs, groups(F), need_cov=False)
        return np.einsum("nk,nkd->nd", resp, ex)*self.scale[b]+mu
