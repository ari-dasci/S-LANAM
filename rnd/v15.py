"""Exploratory blocks of the third exploratory round, outside the grid and the round 2 blocks.

Blocks (run.py --blocks):
  TE    transition matrix estimated from an early-stopped CE model: CE trained on 90% of the noisy training
        set with the noisy-validation split of ES, the weights of the best validation epoch predict the whole
        training set, T_hat_ES comes from the anchor estimator of the grid, and FC with T_hat_ES is trained on the
        whole noisy training set with the grid's fixed budget (methods CEES and FCES, pipeline TE).
  ANNX  two more zero-shot annotators per modality, stored in noise/<DS>_F1.npz and <DS>_F2.npz.
  H4X   conditions S_a, C_a and F_a of annotator a = 1, 2, as S, C and F of the grid.
The ten-fold audio study uses datasets AUDF1-AUDF10 with the N, ANN and H4 blocks of the grid; all
UrbanSound8K clips are annotated once by annotate_all_us8k.
"""
import os

import numpy as np
import torch

from . import config as C
from . import noise as N
from . import train as TR
from .experiment import (_common, _probs_path, _save_noise, code_version, done, gpu_name, log, write_row)

H4X_CONDS = ["S1", "C1", "F1", "S2", "C2", "F2"]


# ============================================================================= TE
def run_te(b, seed, conds, device):
    from . import v14
    ds = b.ds
    for cond in conds:
        todo = [m for m in ("CEES", "FCES") if not done(ds, "TE", m, cond, seed)]
        if not todo:
            continue
        rho_L, rho_A, level = C.CONDITIONS[cond]
        y_noisy, S_L, T_true = N.label_noise(b.y_train, b.K, rho_L, seed)
        X, info = b.noisy_inputs(rho_A, level, seed)
        _save_noise(ds, cond, seed, S_L, y_noisy, info)
        Xt = X if b.kind == "GRA" else b.task(X, device)
        common = _common(b, y_noisy, S_L, info["S_A"], device)
        pp = _probs_path(ds, "TE", cond, seed)
        tp = pp.replace(".npy", "_That.npy")
        if "CEES" in todo or not os.path.exists(tp):
            use, val = v14.es_split(b.n, seed)
            r = v14._fit(b, Xt, y_noisy, seed, device, use_idx=use, val_idx=val, es_probs=True)
            P = r.pop("train_probs_es").astype(np.float32)
            os.makedirs(os.path.dirname(pp), exist_ok=True)
            np.save(pp, P)
            T_hat = TR.estimate_T(P)
            np.save(tp, T_hat)
            err = float(np.linalg.norm(T_hat - T_true) / b.K)
            extra = v14._save_extra(ds, "TE", "CEES", cond, seed, r, S_L, info["S_A"])
            if not done(ds, "TE", "CEES", cond, seed):
                write_row(ds, "TE", "CEES", cond, seed, **r, **dict(common, n_train=len(use)),
                          T_hat_path=os.path.relpath(tp, C.RUNS), T_hat_frob_err=err, extra_path=extra)
                log(f"{ds} TE CEES {cond} s{seed} acc={r['test_acc']:.4f} es={r['test_acc_es']:.4f}@{r['best_epoch']}"
                    f" errT={err:.3f}")
        if not done(ds, "TE", "FCES", cond, seed):
            T_hat = np.load(tp)
            err = float(np.linalg.norm(T_hat - T_true) / b.K)
            r = b.fit(Xt, y_noisy, seed, device, T=T_hat)
            write_row(ds, "TE", "FCES", cond, seed, **r, **common, T_hat_path=os.path.relpath(tp, C.RUNS),
                      T_hat_frob_err=err)
            log(f"{ds} TE FCES {cond} s{seed} acc={r['test_acc']:.4f} errT={err:.3f}")
        del Xt
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


# ============================================================================= annotators
def annotate_all_us8k(device, chunk=500):
    """Labels of all UrbanSound8K clips by the annotator of the grid (LAION-CLAP 630k-audioset-best), once."""
    from . import annotate as A
    from . import data as D
    p = os.path.join(C.NOISE, "AUD_Fall.npz")
    if not os.path.exists(p):
        n = len(D.us8k_meta())
        cfg = C.ANNOTATORS["AUD"]
        model = A.load_clap("630k-audioset-best.pt", False, device)
        yh, S, rev = [], [], None
        for a in range(0, n, chunk):
            w = D.us8k_waves(np.arange(a, min(a + chunk, n)), cfg["sr"])
            y_, s_, rev = A.annotate_aud(w, device, cfg=cfg, model=model)
            yh.append(y_)
            S.append(s_)
            log(f"AUD_Fall {min(a + chunk, n)}/{n}")
        os.makedirs(C.NOISE, exist_ok=True)
        np.savez_compressed(p, yhat=np.concatenate(yh), scores=np.concatenate(S), revision=rev)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    z = np.load(p, allow_pickle=True)
    return z["yhat"], str(z["revision"])


def annotate_x(b, device):
    """Annotators 1 and 2 of C.ANNOTATORS_X for IMG, TXT or AUD (training set of the grid split)."""
    from . import annotate as A
    from . import data as D
    ds = b.ds
    for a, cfg in enumerate(C.ANNOTATORS_X[ds], start=1):
        p = os.path.join(C.NOISE, f"{ds}_F{a}.npz")
        if os.path.exists(p):
            continue
        if ds == "IMG":
            yhat, S, rev = A.annotate_img(b.d["X_train"], device, cfg=cfg)
        elif ds == "TXT":
            yhat, S, rev = A.annotate_txt(b.d["X_train"], device, cfg=cfg)
        elif ds == "AUD":
            model = A.load_clap(cfg["ckpt"], cfg["fusion"], device, cfg.get("amodel", "HTSAT-tiny"))
            yh, Ss, rows = [], [], b.d["train_rows"]
            for i in range(0, len(rows), 500):
                y_, s_, rev = A.annotate_aud(D.us8k_waves(rows[i:i + 500], cfg["sr"]), device, cfg=cfg, model=model)
                yh.append(y_)
                Ss.append(s_)
            yhat, S = np.concatenate(yh), np.concatenate(Ss)
            del model
        else:
            raise ValueError(ds)
        r_F, T_F = N.annotator_stats(b.y_train, yhat, b.K)
        os.makedirs(C.NOISE, exist_ok=True)
        np.savez_compressed(p, yhat=yhat, scores=S, revision=rev, r_F=r_F, T_F=T_F)
        log(f"{ds} annotator {a} {rev}: r_F={r_F:.4f}")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


# ============================================================================= H4X
def run_h4x(b, seed, conds, methods, device):
    ds = b.ds
    Xt = None
    for cond in conds:
        a = int(cond[1:])
        todo = [m for m in methods if not done(ds, "N", m, cond, seed)]
        if not todo:
            continue
        z = np.load(os.path.join(C.NOISE, f"{ds}_F{a}.npz"), allow_pickle=True)
        yhat, r_F, T_F, rev = z["yhat"], float(z["r_F"]), z["T_F"], str(z["revision"])
        y_noisy, T = N.fm_condition(cond[0], b.y_train, yhat, b.K, seed, r_F, T_F)
        if Xt is None:
            Xt = b.task(b.d["X_train"], device)
        common = dict(n_train=b.n, n_label_noisy=int(np.sum(y_noisy != b.y_train)), n_input_noisy=0, n_overlap=0,
                      git_commit=code_version(), gpu=gpu_name(device), annotator=rev, r_F=r_F, rho_L=r_F)
        for m in todo:
            r = b.fit(Xt, y_noisy, seed, device, T=None if m == "CE" else T)
            write_row(ds, "N", m, cond, seed, **r, **common)
            log(f"{ds} H4X {m} {cond} s{seed} acc={r['test_acc']:.4f} rF={r_F:.3f}")
