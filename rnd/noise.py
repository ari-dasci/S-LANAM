"""Exact noise generators of the design.

All randomness comes from numpy Generators seeded with the streams fixed by the design:
  labels      rng_L = default_rng([s, 1, r])
  selection   rng_A = default_rng([s, 2, r])
  corruption  rng_C = default_rng([s, 3, r, level])   (traversed in sorted(S_A) order)
  EE noise    default_rng([s, 4, r])
  H4: S       default_rng([s, 5])
  H4: C       default_rng([s, 6])
  E2 (round 1): noisy test selection  default_rng([s, 7, r]); its corruption  default_rng([s, 8, r, level])
  OOS (round 5): two-fold cross-fitting split  default_rng([s, 13, r_L, r_A])
with r = round(100 * rho).
"""
import io
import numpy as np


def r_of(rho):
    return int(round(100 * rho))


# ----------------------------------------------------------------------------- label noise (4.1)
def symmetric_T(K, rho):
    """T[i, j] = P(noisy = j | clean = i) for flips to another class chosen uniformly."""
    T = np.full((K, K), rho / (K - 1)) if K > 1 else np.zeros((1, 1))
    np.fill_diagonal(T, 1.0 - rho)
    return T


def flip_labels(y, K, n_flip, rng):
    """Select n_flip examples without replacement and move each label to another class uniformly.

    Used by L/J (rng_L) and by condition S of H4 (default_rng([s, 5])).
    """
    y = np.asarray(y)
    n = len(y)
    S = rng.choice(n, size=n_flip, replace=False)
    S = np.sort(S)
    y_new = y.copy()
    for i in S:
        y_new[i] = rng.choice([k for k in range(K) if k != y[i]])
    return y_new, S


def label_noise(y, K, rho, seed):
    """Condition L_rho / J_rho: returns noisy labels, sorted S_L and the true transition matrix."""
    if rho <= 0:
        return np.asarray(y).copy(), np.array([], dtype=int), np.eye(K)
    rng = np.random.default_rng([seed, 1, r_of(rho)])
    y_new, S = flip_labels(y, K, int(round(rho * len(y))), rng)
    return y_new, S, symmetric_T(K, rho)


def pair_T(K, rho):
    """Pair noise: T[i, i] = 1 - rho and T[i, (i + 1) mod K] = rho."""
    T = (1.0 - rho) * np.eye(K)
    for i in range(K):
        T[i, (i + 1) % K] += rho
    return T


def pair_label_noise(y, K, rho, seed):
    """round 4 P_rho / JP_rho: the examples selected exactly as in L_rho (rng_L, same choice) move to class
    (y + 1) mod K. Returns noisy labels, sorted S_L and the true (pair) transition matrix."""
    y = np.asarray(y)
    if rho <= 0:
        return y.copy(), np.array([], dtype=int), np.eye(K)
    rng = np.random.default_rng([seed, 1, r_of(rho)])
    S = np.sort(rng.choice(len(y), size=int(round(rho * len(y))), replace=False))
    y_new = y.copy()
    y_new[S] = (y[S] + 1) % K
    return y_new, S, pair_T(K, rho)


def dependent_label_noise(y, K, S_A, rho, seed):
    """round 4 JD_rho: the labels of exactly the examples whose inputs are corrupted (S_L = S_A) move to another
    class uniformly, with the stream default_rng([s, 12, r]); the marginal transition matrix is symmetric."""
    y = np.asarray(y)
    rng = np.random.default_rng([seed, 12, r_of(rho)])
    y_new = y.copy()
    for i in np.sort(S_A):
        y_new[i] = rng.choice([k for k in range(K) if k != y[i]])
    return y_new, np.sort(np.asarray(S_A)), symmetric_T(K, rho)


def crossfit_folds(y_noisy, K, rho_L, rho_A, seed):
    """OOS: two folds stratified by the noisy label. In each noisy class a permutation drawn
    from default_rng([s, 13, r_L, r_A]); its first floor(n_k / 2) examples go to fold 0, the rest to fold 1."""
    rng = np.random.default_rng([seed, 13, r_of(rho_L), r_of(rho_A)])
    y_noisy = np.asarray(y_noisy)
    fold = np.ones(len(y_noisy), dtype=np.int64)
    for k in range(K):
        idx = rng.permutation(np.flatnonzero(y_noisy == k))
        fold[idx[: len(idx) // 2]] = 0
    return fold


def select_inputs(n, rho, seed):
    """S_A for A_rho / J_rho (independent of the level)."""
    if rho <= 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng([seed, 2, r_of(rho)])
    return np.sort(rng.choice(n, size=int(round(rho * n)), replace=False))


def corruption_rng(seed, rho, level):
    return np.random.default_rng([seed, 3, r_of(rho), level])


def select_test(n_test, rho, seed):
    """E2: test examples whose inputs are corrupted (depends only on the seed, not on the condition)."""
    rng = np.random.default_rng([seed, 7, r_of(rho)])
    return np.sort(rng.choice(n_test, size=int(round(rho * n_test)), replace=False))


def test_corruption_rng(seed, rho, level):
    return np.random.default_rng([seed, 8, r_of(rho), level])


# ----------------------------------------------------------------------------- input noise (4.2)
def corrupt_tabular(X, S_A, sd_clean, s_level, rng):
    """x_j + s * sigma_j * z, z ~ N(0,1) independently per attribute; no rounding or clipping."""
    X = np.array(X, dtype=np.float64, copy=True)
    for i in S_A:
        X[i] = X[i] + s_level * sd_clean * rng.standard_normal(X.shape[1])
    return X


def corrupt_images(X, S_A, params, rng):
    """X: uint8 (n, 32, 32, 3). One corruption per image chosen uniformly among gauss, blur, jpeg.

    Returns the corrupted copy and the type index of each selected image (0 gauss, 1 blur, 2 jpeg).
    """
    from PIL import Image, ImageFilter
    X = np.array(X, copy=True)
    types = np.zeros(len(S_A), dtype=np.int8)
    for k, i in enumerate(S_A):
        t = int(rng.integers(3))
        types[k] = t
        if t == 0:
            x = X[i].astype(np.float64) / 255.0
            x = np.clip(x + rng.normal(0.0, params["sigma"], size=x.shape), 0.0, 1.0)
            X[i] = np.round(x * 255.0).astype(np.uint8)
        elif t == 1:
            img = Image.fromarray(X[i]).filter(ImageFilter.GaussianBlur(radius=params["radius"]))
            X[i] = np.asarray(img, dtype=np.uint8)
        else:
            buf = io.BytesIO()
            Image.fromarray(X[i]).save(buf, format="JPEG", quality=int(params["quality"]))
            buf.seek(0)
            X[i] = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
    return X, types


def _is_letter(ch):
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def corrupt_string(s, p, rng):
    """Left-to-right pass; each ASCII letter is perturbed with probability p by sub, del, ins or swap.

    swap exchanges the letter with the next character if it exists and is a letter (the next
    character is then consumed); otherwise nothing is done. Other characters are untouched.
    """
    out = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if _is_letter(ch) and rng.random() < p:
            op = int(rng.integers(4))
            if op == 0:                       # sub
                out.append(chr(97 + int(rng.integers(26))))
            elif op == 1:                     # del
                pass
            elif op == 2:                     # ins
                out.append(ch)
                out.append(chr(97 + int(rng.integers(26))))
            else:                             # swap
                if i + 1 < n and _is_letter(s[i + 1]):
                    out.append(s[i + 1])
                    out.append(ch)
                    i += 2
                    continue
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def corrupt_texts(texts, S_A, p, rng):
    texts = list(texts)
    for i in S_A:
        texts[i] = corrupt_string(texts[i], p, rng)
    return texts


def corrupt_series(X, S_A, snr_db, rng):
    """X: raw windows (n, C, T). Per channel: P = var(x_c) (population), sigma = sqrt(P / 10^(SNR/10))."""
    X = np.array(X, dtype=np.float64, copy=True)
    for i in S_A:
        for c in range(X.shape[1]):
            P = X[i, c].var()
            sigma = np.sqrt(P / 10 ** (snr_db / 10.0))
            X[i, c] = X[i, c] + sigma * rng.standard_normal(X.shape[2])
    return X


def corrupt_graph_features(X, nodes, q, rng):
    """X: binary (N, F); each feature of each selected node flips independently with probability q."""
    X = np.array(X, copy=True)
    for v in nodes:
        flips = rng.random(X.shape[1]) < q
        X[v] = np.where(flips, 1 - X[v], X[v])
    return X


def corrupt_waves(waves, S_A, snr_db, rng):
    """waves: list of 1-D float arrays (unpadded). P = mean(x^2); clips with P = 0 are left unchanged
    and no random numbers are drawn for them. Returns a new list and the indices left unchanged."""
    waves = list(waves)
    silent = []
    for i in S_A:
        x = np.asarray(waves[i], dtype=np.float64)
        P = float(np.mean(x ** 2))
        if P == 0.0:
            silent.append(int(i))
            continue
        sigma = np.sqrt(P / 10 ** (snr_db / 10.0))
        waves[i] = (x + sigma * rng.standard_normal(len(x))).astype(np.float32)
    return waves, silent


# ----------------------------------------------------------------------------- embedding noise
def embedding_noise(E, S_A, sigma_E, seed, rho):
    """EE: e_i + eps_i, eps_i ~ N(0, sigma_E^2 I_d) for i in sorted(S_A); no renormalisation."""
    rng = np.random.default_rng([seed, 4, r_of(rho)])
    E = np.array(E, dtype=np.float32, copy=True)
    for i in S_A:
        E[i] = E[i] + (sigma_E * rng.standard_normal(E.shape[1])).astype(np.float32)
    return E


# ----------------------------------------------------------------------------- H4
def annotator_stats(y, yhat, K):
    """r_F and T_F (rows: true class, row-normalised empirical confusion of the annotator)."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    r_F = float(np.mean(yhat != y))
    T = np.zeros((K, K))
    for a, b in zip(y, yhat):
        T[a, b] += 1
    rows = T.sum(1, keepdims=True)
    T_F = np.divide(T, rows, out=np.zeros_like(T), where=rows > 0)
    return r_F, T_F


def fm_condition(cond, y, yhat, K, seed, r_F, T_F):
    """Labels and true transition matrix of condition S, C or F."""
    y = np.asarray(y)
    if cond == "F":
        return np.asarray(yhat).copy(), T_F
    if cond == "S":
        rng = np.random.default_rng([seed, 5])
        y_new, _ = flip_labels(y, K, int(round(r_F * len(y))), rng)
        return y_new, symmetric_T(K, r_F)
    if cond == "C":
        rng = np.random.default_rng([seed, 6])
        y_new = np.array([rng.choice(K, p=T_F[y[i]]) for i in range(len(y))], dtype=y.dtype)
        return y_new, T_F
    raise ValueError(cond)
