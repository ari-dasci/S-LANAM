"""Exploratory blocks of the fourth exploratory round.

H2B  H2 with pair label noise and with label noise placed on the corrupted inputs (TAB, TS, GRA, TXT):
     P_rho   pair noise (selection of L_rho, label moved to (y + 1) mod K);
     JP_rho  P_rho plus the independent input noise of A_rho (as J_rho);
     JD_rho  input noise of A_rho and symmetric label noise on exactly the same examples (S_L = S_A);
     CE, FC (anchor T_hat from the CE model of the same condition) and FC* (pair T for P/JP, symmetric T for JD),
     pipeline N, fixed budget, last epoch; references c0, L, A and J of the grid on the same card.
SC   standardisation control (TAB, TS): the grid conditions with input noise, standardised with the mean and
     standard deviation of the clean training set instead of the noisy one (pipeline SC); c0 and L are unchanged
     (their training inputs are clean), so their grid runs are the references.
"""
import os

import numpy as np
import torch

from . import config as C
from . import noise as N
from . import train as TR
from .experiment import _common, _probs_path, _save_noise, done, log, write_row


def _h2b_labels(b, cond, seed, S_A):
    rho_L = C.CONDITIONS[cond][0]
    if cond.startswith("JD"):
        return N.dependent_label_noise(b.y_train, b.K, S_A, rho_L, seed)
    return N.pair_label_noise(b.y_train, b.K, rho_L, seed)


def run_h2b(b, seed, conds, methods, device):
    ds = b.ds
    for cond in conds:
        todo = [m for m in methods if m in ("CE", "FC", "FCstar") and not done(ds, "N", m, cond, seed)]
        if not todo:
            continue
        _, rho_A, level = C.CONDITIONS[cond]
        X, info = b.noisy_inputs(rho_A, level, seed)
        y_noisy, S_L, T_true = _h2b_labels(b, cond, seed, info["S_A"])
        _save_noise(ds, cond, seed, S_L, y_noisy, info)
        Xt = X if b.kind == "GRA" else b.task(X, device)
        common = _common(b, y_noisy, S_L, info["S_A"], device)
        pp = _probs_path(ds, "N", cond, seed)
        if "CE" in todo or ("FC" in todo and not os.path.exists(pp)):
            r = b.fit(Xt, y_noisy, seed, device, need_train_probs=True)
            os.makedirs(os.path.dirname(pp), exist_ok=True)
            np.save(pp, r.pop("train_probs").astype(np.float32))
            if not done(ds, "N", "CE", cond, seed):
                write_row(ds, "N", "CE", cond, seed, **r, **common)
                log(f"{ds} H2B CE {cond} s{seed} acc={r['test_acc']:.4f} t={r['train_time_s']:.0f}s")
        if "FC" in todo:
            T_hat = TR.estimate_T(np.load(pp))
            tp = pp.replace(".npy", "_That.npy")
            np.save(tp, T_hat)
            err = float(np.linalg.norm(T_hat - T_true) / b.K)
            r = b.fit(Xt, y_noisy, seed, device, T=T_hat)
            write_row(ds, "N", "FC", cond, seed, **r, **common, T_hat_path=os.path.relpath(tp, C.RUNS),
                      T_hat_frob_err=err)
            log(f"{ds} H2B FC {cond} s{seed} acc={r['test_acc']:.4f} errT={err:.3f}")
        if "FCstar" in todo:
            r = b.fit(Xt, y_noisy, seed, device, T=T_true)
            write_row(ds, "N", "FCstar", cond, seed, **r, **common)
            log(f"{ds} H2B FCstar {cond} s{seed} acc={r['test_acc']:.4f}")
        del Xt
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def _sc_task(b, X, device):
    if b.kind == "TAB":
        return TR.task_tab(X, b.d["X_test"], device, K=b.K, stats_X=b.d["X_train"])
    if b.kind == "TS":
        return TR.task_ts(X, b.d["X_test"], device, K=b.K, stats_X=b.d["X_train"])
    raise ValueError("SC is defined for TAB and TS only")


def run_sc(b, seed, conds, methods, device):
    from . import v14
    ds = b.ds
    for cond in conds:
        todo = [m for m in methods if cond in C.SC_CONDS.get(m, []) and not done(ds, "SC", m, cond, seed)]
        if cond == "AT40" and b.kind != "TAB":
            todo = []
        if not todo:
            continue
        rho_L, rho_A, level = C.CONDITIONS[cond]
        y_noisy, S_L, T_true = N.label_noise(b.y_train, b.K, rho_L, seed)
        if cond == "AT40":
            X, info = v14.at40_inputs(b, seed)
        else:
            X, info = b.noisy_inputs(rho_A, level, seed)
        Xt = _sc_task(b, X, device)
        S_A = info.get("S_A", np.arange(0))
        common = _common(b, y_noisy, S_L, S_A, device)
        for m in todo:
            r = b.fit(Xt, y_noisy, seed, device, T=T_true if m == "FCstar" else None)
            write_row(ds, "SC", m, cond, seed, **r, **common, notes="clean-set standardisation")
            log(f"{ds} SC {m} {cond} s{seed} acc={r['test_acc']:.4f}")
        del Xt
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
