"""Frozen-feature alignment. Row-vector convention: L @ W -> E; I @ V -> E."""
import warnings
import numpy as np
from scipy.special import logsumexp


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class PCA:
    def __init__(self, d, whiten=0.0):
        self.d, self.whiten = d, whiten

    def fit(self, x):
        if not 0 <= self.whiten <= 1:
            raise ValueError('whiten must be in [0,1]')
        self.mean = x.mean(0)
        _, s, vt = np.linalg.svd(x - self.mean, full_matrices=False)
        if self.d > min(x.shape[0]-1, x.shape[1]) or s[self.d-1] < 1e-10:
            raise ValueError('PCA dimension exceeds training rank')
        scale = np.maximum(s[:self.d] / np.sqrt(len(x)-1), 1e-6)
        self.proj = vt[:self.d].T / scale**self.whiten
        return self

    def transform(self, x):
        return unit((x-self.mean) @ self.proj)

    @classmethod
    def fit_shared(cls, d, arrays, whiten=0.0):
        """One projection for spaces that already share coordinates (CLIP text and image towers).

        Each space is centered with ITS OWN mean (this removes CLIP's modality gap); the principal axes are fitted on
        the stacked centered data, so both spaces keep CLIP's common coordinates after the reduction.
        """
        if len({x.shape[1] for x in arrays}) != 1:
            raise ValueError('Shared PCA needs equal feature widths (E and I from the same CLIP model)')
        means = [x.mean(0) for x in arrays]
        stacked = np.vstack([x - m for x, m in zip(arrays, means)])
        fitted = cls(d, whiten).fit(stacked)  # mean ~ 0; reuse the rank checks and whitening of fit()
        out = []
        for m in means:
            p = cls(d, whiten)
            p.mean, p.proj = m, fitted.proj
            out.append(p)
        return out


def fit_space_pcas(data, d, whiten=0.0, shared_ei=False, joint_ei=False):
    """Training-only PCA. For native CLIP, jointly fit ONE mean and ONE projection for E and I.

    The legacy shared_ei option has a shared projection but separate modality means.
    joint_ei=True preserves relative CLIP coordinates and is recommended for native CLIP.
    """
    if shared_ei and joint_ei:
        raise ValueError("Choose only one of shared_ei or joint_ei")
    pcas = {s: PCA(d, whiten).fit(data[s+'_fit']) for s in 'lei'}
    if shared_ei:
        pcas['e'], pcas['i'] = PCA.fit_shared(d, [data['e_fit'], data['i_fit']], whiten)
    elif joint_ei:
        combined = PCA(d, whiten).fit(np.concatenate([data['e_fit'], data['i_fit']], axis=0))
        pcas['e'] = pcas['i'] = combined
    return pcas


def polar(m):
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    return u @ vt


def procrustes(x, y):
    return polar(x.T @ y)


def sinkhorn(x, y, epsilon=.08, tau=None, steps=500, tol=1e-7):
    """Balanced or KL-unbalanced OT with uniform reference marginals.
    Returns the raw coupling (unbalanced mass is NOT renormalized).
    """
    if epsilon <= 0 or (tau is not None and tau <= 0):
        raise ValueError('epsilon and tau must be positive')
    c = np.maximum((x*x).sum(1)[:, None] + (y*y).sum(1)[None, :] - 2*x@y.T, 0)
    k = -c / epsilon
    a = np.full(len(x), -np.log(len(x)))
    b = np.full(len(y), -np.log(len(y)))
    u, v = np.zeros(len(x)), np.zeros(len(y))
    power = 1 if tau is None else tau/(tau+epsilon)
    for _ in range(steps):
        old = u.copy()
        u = power * (a-logsumexp(k+v[None, :], axis=1))
        v = power * (b-logsumexp(k+u[:, None], axis=0))
        if np.max(np.abs(u-old)) < tol:
            break
    return np.exp(k+u[:, None]+v[None, :])


def neighbor_scores(x, y, k=10):
    s = unit(x) @ unit(y).T
    if k == 0:
        return s
    kx, ky = min(k, len(y)), min(k, len(x))
    rx = np.partition(s, -kx, axis=1)[:, -kx:].mean(1)
    ry = np.partition(s, -ky, axis=0)[-ky:, :].mean(0)
    return 2*s-rx[:, None]-ry[None, :]


def votes(lw, e, images_in_e, k=10):
    """e[j] is the caption paired with image[j]. Mutual text NN + route agreement."""
    if len(e) != len(images_in_e):
        raise ValueError('Vote bank must contain row-aligned English/image pairs')
    scores = neighbor_scores(lw, e, k)
    direct = scores.argmax(1)
    via = neighbor_scores(lw, images_in_e, k).argmax(1)
    back = scores.argmax(0)
    keep = np.where((direct == via) & (back[direct] == np.arange(len(lw))))[0]
    return keep, direct[keep]


def fit(lpair, epair, lpool, epool, image_bank, caption_bank,
        method='visual', seed=0, iterations=20, batch=256,
        lambda_e=.25, lambda_i=.25, lambda_p=.1, epsilon=.08,
        tau=None, cycle=True, extra_e=None, csls=10):
    """Stochastic alternating OT / exact orthogonal update per sampled subproblem.
    Not a monotone optimizer of the full-pool objective.
    """
    if method not in {'supervised', 'text', 'visual', 'bank_text', 'extra_text', 'gaussian', 'broken_pairs'}:
        raise ValueError(method)
    if len(lpair) < lpair.shape[1]:
        warnings.warn('Anchor budget < dimension: initialization is rank deficient; reduce d.')
    rng = np.random.default_rng(seed)
    w = procrustes(lpair, epair)
    anchors = lpair.T @ epair / len(lpair)
    trace = []
    control_rng = np.random.default_rng(seed + 100000)
    target = image_bank
    captions = caption_bank
    if method == 'bank_text':
        if len(caption_bank) != len(image_bank):
            raise ValueError('bank_text needs an English caption for every image-bank row')
        # Same image IDs and content as visual, but English text embeddings.
        target = caption_bank
    if method == 'extra_text':
        if extra_e is None or len(extra_e) < len(image_bank):
            raise ValueError('extra_text needs a disjoint English pool >= image bank size')
        target = extra_e[control_rng.choice(len(extra_e), len(image_bank), replace=False)]
    if method == 'gaussian':
        target = unit(control_rng.normal(size=image_bank.shape))
    if method == 'broken_pairs':
        captions = caption_bank[control_rng.permutation(len(caption_bank))]
    if method == 'supervised':
        return w, trace
    for step in range(iterations):
        li = rng.choice(len(lpool), min(batch, len(lpool)), replace=False)
        ei = rng.choice(len(epool), min(batch, len(epool)), replace=False)
        vi = rng.choice(len(target), min(batch, len(target)), replace=False)
        x, y, z = lpool[li], epool[ei], target[vi]
        p = sinkhorn(x@w, y, epsilon, tau)
        m = anchors + lambda_e * x.T @ p @ y
        if method != 'text':
            p_i = sinkhorn(x@w, z, epsilon, tau)
            m += lambda_i * x.T @ p_i @ z
        count = 0
        # Shared caption control uses the SAME pseudo-pair weight and selection rule.
        # For extra_text, the third bank is paired with itself (text-only route).
        if cycle and method != 'text':
            bank_e = target[vi] if method in {'extra_text', 'bank_text'} else captions[vi]
            ix, iy = votes(x@w, bank_e, z, csls)
            count = len(ix)
            if count:
                m += lambda_p * x[ix].T @ bank_e[iy] / count
        w = polar(m)
        trace.append({'iteration':step, 'pseudo_count':count,
                      'anchor_mse':float(np.mean(np.sum((lpair@w-epair)**2, axis=1)))})
    return w, trace


def retrieval(x, y, xids, yids, k=10):
    """Multiple captions per image supported through shared semantic IDs.
    Dense evaluation: use moderate candidate sets (e.g. XM3600).
    """
    if not set(xids).issubset(set(yids)):
        raise ValueError('Every query must have a relevant candidate')
    order = np.argsort(-neighbor_scores(x, y, k), axis=1)
    out, hits = {}, {}
    for rank in (1, 5, 10):
        h = (yids[order[:, :rank]] == xids[:, None]).any(1)
        out[f'R@{rank}'] = float(h.mean())
        hits[f'R@{rank}'] = h
    return out, hits


def paired_bootstrap(a, b, seed=0, repeats=2000):
    """Paired query bootstrap. For repeated image IDs, aggregate by image first."""
    delta = np.asarray(a, float)-np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(repeats)])
    return {'difference':float(delta.mean()), 'ci95':np.quantile(means, [.025,.975]).tolist()}


def linear_cka(x, y):
    """Requires semantically paired rows; arbitrary unpaired ordering is invalid."""
    x, y = x-x.mean(0), y-y.mean(0)
    return float(np.linalg.norm(x.T@y)**2 / max(np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y), 1e-12))
