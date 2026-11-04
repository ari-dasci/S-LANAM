"""Calibrated extension of the fifth round (block OOSC): FC with temperature-calibrated out-of-sample predictions.

OOSC  for every cell of the OOS block (rnd/v17.py) whose FC run is on this card, the saved out-of-sample predictions
      p_hat_oos are calibrated with one temperature per half model, fitted by maximum likelihood against the noisy
      labels of the fold that model did not see (the target of the anchor estimator is P(noisy Y | X)); T_hat_cal
      uses the class-wise rule of the OOS block on the calibrated predictions; FC is trained with T_hat_cal on the
      whole noisy training set with the grid recipe and seed (method FCcal, pipeline OOS), on the same card as the
      CE and FC runs of the cell. Diagnostics: inverse temperatures, noisy-label NLL before and after, mean diagonal
      of the estimates, and the error of T_hat_cal with the class-wise, global and argmax rules.
"""
import math
import os

import numpy as np
import torch

from . import config as C
from . import noise as N
from . import train as TR
from .experiment import _common, done, log, write_row
from .v17 import PIPE, _err, _paths

METHOD = "FCcal"
LOG_MIN = 1e-38                               # smallest normal float32: clip before taking logs


def _logits(p):
    return np.log(np.clip(np.asarray(p, dtype=np.float64), LOG_MIN, None))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def noisy_nll(z, y, beta):
    """Mean negative log-likelihood of the labels y under softmax(beta * z)."""
    s = beta * z
    m = s.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(s - m).sum(axis=1, keepdims=True)))[:, 0]
    return float(np.mean(lse - s[np.arange(len(y)), y]))


def fit_inverse_temperature(z, y, lo=1e-3, hi=1e3, tol=1e-7, it=300):
    """Inverse temperature beta minimising noisy_nll (convex in beta, so unimodal in log beta): golden-section
    search on log beta in [log lo, log hi]."""
    y = np.asarray(y)
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = noisy_nll(z, y, math.exp(c)), noisy_nll(z, y, math.exp(d))
    for _ in range(it):
        if b - a < tol:
            break
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = noisy_nll(z, y, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = noisy_nll(z, y, math.exp(d))
    return math.exp((a + b) / 2)


def calibrate(p_oos, y_noisy, fold):
    """Calibrated out-of-sample predictions: on fold f, softmax(beta_f * log p_hat) with beta_f fitted on fold f,
    whose predictions come from the half model trained on the other fold. Returns (p_cal, betas, nll_before,
    nll_after), the last three per fold."""
    z = _logits(p_oos)
    y_noisy = np.asarray(y_noisy)
    p_cal = np.zeros_like(z)
    betas, before, after = [], [], []
    for f in (0, 1):
        I = np.flatnonzero(fold == f)
        beta = fit_inverse_temperature(z[I], y_noisy[I])
        p_cal[I] = _softmax(beta * z[I])
        betas.append(beta)
        before.append(noisy_nll(z[I], y_noisy[I], 1.0))
        after.append(noisy_nll(z[I], y_noisy[I], beta))
    return p_cal, betas, before, after


def run_oos_cal(b, seed, conds, device):
    ds = b.ds
    for cond in conds:
        if done(ds, PIPE, METHOD, cond, seed):
            continue
        P = _paths(ds, cond, seed)
        if not (done(ds, PIPE, "FC", cond, seed) and os.path.exists(P["oos"])):
            continue                                         # the cell ran on the other card
        rho_L, rho_A, level = C.CONDITIONS[cond]
        y_noisy, S_L, T_true = N.label_noise(b.y_train, b.K, rho_L, seed)
        fold = N.crossfit_folds(y_noisy, b.K, rho_L, rho_A, seed)
        p_oos = np.load(P["oos"])
        p_cal, betas, nll0, nll1 = calibrate(p_oos, y_noisy, fold)
        T_cal = TR.estimate_T_classwise(p_cal, y_noisy)
        T_oos = np.load(P["That_oos"])
        p_that = P["That_oos"].replace("_That_oos.npy", "_That_cal.npy")
        np.save(p_that, T_cal)
        K = b.K
        diag = {"errT_cal_class": _err(T_cal, T_true, K), "errT_cal_global": _err(TR.estimate_T(p_cal), T_true, K),
                "errT_cal_argmax": _err(TR.estimate_T_argmax(p_cal), T_true, K),
                "errT_OOS_class": _err(T_oos, T_true, K), "errT_identity": _err(np.eye(K), T_true, K),
                "beta0": betas[0], "beta1": betas[1], "nll_before0": nll0[0], "nll_before1": nll0[1],
                "nll_after0": nll1[0], "nll_after1": nll1[1],
                "diag_OOS": float(np.mean(np.diag(T_oos))), "diag_cal": float(np.mean(np.diag(T_cal)))}
        X, info = b.noisy_inputs(rho_A, level, seed)
        Xt = X if b.kind == "GRA" else b.task(X, device)
        common = _common(b, y_noisy, S_L, info["S_A"], device)
        r = b.fit(Xt, y_noisy, seed, device, T=T_cal)
        write_row(ds, PIPE, METHOD, cond, seed, **r, **common, T_hat_path=os.path.relpath(p_that, C.RUNS),
                  T_hat_frob_err=diag["errT_cal_class"], notes=";".join(f"{k}={v:.6f}" for k, v in diag.items()))
        log(f"{ds} OOS FCcal {cond} s{seed} acc={r['test_acc']:.4f} errT_cal={diag['errT_cal_class']:.4f} "
            f"errT_oos={diag['errT_OOS_class']:.4f} beta={betas[0]:.3f},{betas[1]:.3f}")
        del Xt
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
