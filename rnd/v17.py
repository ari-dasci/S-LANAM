"""Exploratory block of the fifth exploratory round.

OOS  forward correction with a transition matrix estimated from out-of-sample predictions, grid conditions
     L20, L40, J20 and J40 (same noise streams as the grid), the six grid datasets and the six second datasets.
     Per dataset, condition and seed, on one card (pipeline OOS):
     1. CE on the whole noisy training set (grid recipe): the reference A_CE(u) and in-sample predictions;
     2. two-fold cross-fitting: folds stratified by the noisy label (noise.crossfit_folds), CE trained on each fold
        with the grid recipe (same epochs, batch, optimiser and seed) and predictions on the other fold, so that
        every training example gets an out-of-sample p_hat (GRA: the whole graph stays, only labels are hidden);
     3. T_hat_OOS with the class-wise rule (train.estimate_T_classwise);
     4. FC with T_hat_OOS on the whole noisy training set.
     Diagnostics without extra training: the error of T_hat for {in-sample, out-of-sample} x {global, class-wise}
     percentile and the out-of-sample argmax rule, and the accuracy of each half model on the held-out fold
     against the clean labels.
"""
import os

import numpy as np
import torch

from . import config as C
from . import noise as N
from . import train as TR
from .experiment import _common, _save_noise, done, log, write_row

PIPE = "OOS"


def _paths(ds, cond, seed):
    base = os.path.join(C.PROBS, ds, PIPE)
    return {k: os.path.join(base, f"{cond}_s{seed}_{k}.npy") for k in ("is", "oos", "That_oos")}


def _subtask(task, I_tr, I_ho):
    """The task restricted to the training examples I_tr, with the held-out examples I_ho as its test split."""
    itr = torch.as_tensor(I_tr, dtype=torch.long)
    iho = torch.as_tensor(I_ho, dtype=torch.long)

    def get(split, idx, aug, g):
        if split == "train":
            return task["get"]("train", itr[idx], aug, g)
        return task["get"]("train", iho[idx], False, g)

    sub = dict(n=len(I_tr), model_fn=task["model_fn"], get=get, fwd=task["fwd"], n_test=len(I_ho), has_noisy=False)
    if "pred_bs" in task:
        sub["pred_bs"] = task["pred_bs"]
    return sub


def _fit_half(b, Xt, y_noisy, I_tr, I_ho, seed, device):
    cfg = dict(C.DATASETS[b.ds])
    if b.kind == "GRA":
        tr = b.d["train_idx"]
        return TR.fit_predict_graph(Xt, b.d["y"], y_noisy[I_tr], b.d["edge_index"], tr[I_tr], tr[I_ho], cfg, seed,
                                    device, need_test_probs=True)
    return TR.fit_predict(_subtask(Xt, I_tr, I_ho), y_noisy[I_tr], b.y_train[I_ho], cfg, seed, device,
                          need_test_probs=True)


def _err(T_hat, T, K):
    return float(np.linalg.norm(T_hat - T) / K)


def run_oos(b, seed, conds, device):
    ds = b.ds
    for cond in conds:
        if done(ds, PIPE, "FC", cond, seed):
            continue
        rho_L, rho_A, level = C.CONDITIONS[cond]
        y_noisy, S_L, T_true = N.label_noise(b.y_train, b.K, rho_L, seed)
        X, info = b.noisy_inputs(rho_A, level, seed)
        _save_noise(ds, cond, seed, S_L, y_noisy, info)
        Xt = X if b.kind == "GRA" else b.task(X, device)
        common = _common(b, y_noisy, S_L, info["S_A"], device)
        P = _paths(ds, cond, seed)
        os.makedirs(os.path.dirname(P["is"]), exist_ok=True)
        # 1. CE on the whole noisy training set
        if not done(ds, PIPE, "CE", cond, seed) or not os.path.exists(P["is"]):
            r = b.fit(Xt, y_noisy, seed, device, need_train_probs=True)
            np.save(P["is"], r.pop("train_probs").astype(np.float32))
            write_row(ds, PIPE, "CE", cond, seed, **r, **common)
            log(f"{ds} OOS CE {cond} s{seed} acc={r['test_acc']:.4f} t={r['train_time_s']:.0f}s")
        # 2. two-fold cross-fitting
        if not os.path.exists(P["oos"]):
            fold = N.crossfit_folds(y_noisy, b.K, rho_L, rho_A, seed)
            p_oos = np.zeros((b.n, b.K), dtype=np.float32)
            for f in (0, 1):
                I_ho, I_tr = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
                r = _fit_half(b, Xt, y_noisy, I_tr, I_ho, seed, device)
                p_oos[I_ho] = r.pop("test_probs")
                write_row(ds, PIPE, f"CEhalf{f}", cond, seed, **r, **common,
                          notes="trained on the other fold; test_acc on this held-out fold against clean labels")
                log(f"{ds} OOS CEhalf{f} {cond} s{seed} heldout_acc={r['test_acc']:.4f} t={r['train_time_s']:.0f}s")
            np.save(P["oos"], p_oos)
        # 3. estimates of T and diagnostics
        p_is, p_oos = np.load(P["is"]), np.load(P["oos"])
        T_oos = TR.estimate_T_classwise(p_oos, y_noisy)
        np.save(P["That_oos"], T_oos)
        K = b.K
        diag = {"errT_IS_global": _err(TR.estimate_T(p_is), T_true, K),
                "errT_IS_class": _err(TR.estimate_T_classwise(p_is, y_noisy), T_true, K),
                "errT_OOS_global": _err(TR.estimate_T(p_oos), T_true, K),
                "errT_OOS_class": _err(T_oos, T_true, K),
                "errT_OOS_argmax": _err(TR.estimate_T_argmax(p_oos), T_true, K)}
        # 4. FC with T_hat_OOS
        r = b.fit(Xt, y_noisy, seed, device, T=T_oos)
        write_row(ds, PIPE, "FC", cond, seed, **r, **common, T_hat_path=os.path.relpath(P["That_oos"], C.RUNS),
                  T_hat_frob_err=diag["errT_OOS_class"],
                  notes=";".join(f"{k}={v:.6f}" for k, v in diag.items()))
        log(f"{ds} OOS FC {cond} s{seed} acc={r['test_acc']:.4f} errT_oos={diag['errT_OOS_class']:.3f} "
            f"errT_is={diag['errT_IS_global']:.3f}")
        del Xt
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
