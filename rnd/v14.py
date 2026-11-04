"""Exploratory blocks of the second exploratory round, outside the grid.

Blocks (run.py --blocks):
  LS   CE under c0, L40, A40, J40 with per-example training losses recorded at 20%, 50% and 100% of the epochs
       (loss separability); its rows are also the same-GPU references of H1C and H2P.
  H1C  controls of H1: R40 removes the examples of S_A, X40 blends their inputs with an input of another
       class, AT40 (tabular) replaces 40% of the values of every attribute with uniform values.
  H2P  loss correction with an anchor-free estimate of T (VolMinNet, method VOL) and Co-teaching (COT) under
       L40 and J40.
  ES   early stopping: 10% of the noisy training set is held out; test accuracy at the epoch of best
       noisy-validation accuracy, CE under c0, L40, A40, J40 and FC* under L40, J40.
  CN   CIFAR-10N real labels and symmetric noise at the same rate, CE, FC and FC*.
  CD   Cora diagnosis: CE and FC* under L40 for 200 and 400 epochs with per-epoch curves.
Random streams (new): [s, 9, 40] for X40 partners, [s, 10, 40, a] for AT40 attribute a, [s, 11] for the
validation split of ES.
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from . import noise as N
from . import train as TR
from .experiment import (_common, _probs_path, _save_noise, code_version, done, gpu_name, log, write_row)


# ============================================================================= noise of the H1 controls
def x40_inputs(b, seed, alpha=None):
    """X40: x_i <- blend of x_i and the input of a random training example of another (true) class, for i in S_A.
    Labels are unchanged. TAB/TS/IMG: (1-a) x_i + a x_j; AUD: x_j rescaled to the RMS of x_i, cut or padded to its
    length; TXT: every word of x_i replaced, with probability a, by a random word of x_j; GRA: every feature bit
    taken from node j with probability a."""
    alpha = C.X40_ALPHA if alpha is None else alpha
    S_A = N.select_inputs(b.n, 0.4, seed)
    rng = np.random.default_rng([seed, 9, 40])
    y = b.y_train
    partner = {}
    for i in S_A:
        others = np.flatnonzero(y != y[i])
        partner[int(i)] = int(others[rng.integers(len(others))])
    k = b.kind
    if k == "GRA":
        X = np.array(b.d["X"], copy=True)
        tr = b.d["train_idx"]
        for i, j in partner.items():
            take = rng.random(X.shape[1]) < alpha
            X[tr[i], take] = b.d["X"][tr[j], take]
        return X, dict(S_A=S_A, partner=np.array([partner[int(i)] for i in S_A]))
    Xtr = b.d["X_train"]
    if k in ("TAB", "TS"):
        X = np.array(Xtr, dtype=np.float64, copy=True)
        for i, j in partner.items():
            X[i] = (1 - alpha) * Xtr[i] + alpha * Xtr[j]
    elif k == "IMG":
        X = np.array(Xtr, copy=True)
        for i, j in partner.items():
            X[i] = np.round((1 - alpha) * Xtr[i].astype(np.float64) + alpha * Xtr[j].astype(np.float64)).astype(np.uint8)
    elif k == "AUD":
        X = list(Xtr)
        for i, j in partner.items():
            xi, xj = np.asarray(Xtr[i], np.float32), np.asarray(Xtr[j], np.float32)
            xj = np.resize(xj, len(xi)) if len(xj) else np.zeros_like(xi)
            pi, pj = np.sqrt(np.mean(xi ** 2)), np.sqrt(np.mean(xj ** 2))
            if pj > 0:
                xj = xj * (pi / pj)
            X[i] = ((1 - alpha) * xi + alpha * xj).astype(np.float32)
    elif k == "TXT":
        X = list(Xtr)
        for i, j in partner.items():
            wi, wj = Xtr[i].split(), Xtr[j].split()
            if not wi or not wj:
                continue
            X[i] = " ".join(wj[rng.integers(len(wj))] if rng.random() < alpha else w for w in wi)
    else:
        raise ValueError(k)
    return X, dict(S_A=S_A, partner=np.array([partner[int(i)] for i in S_A]))


def at40_inputs(b, seed, rho=0.4):
    """AT40 (tabular): for every attribute a, round(rho n) values chosen with default_rng([s, 10, 40, a]) are
    replaced by values drawn uniformly from [min, max] of that attribute in the clean training set."""
    X0 = b.d["X_train"]
    X = np.array(X0, dtype=np.float64, copy=True)
    n, d = X.shape
    lo, hi = X0.min(axis=0), X0.max(axis=0)
    m = int(round(rho * n))
    for a in range(d):
        rng = np.random.default_rng([seed, 10, N.r_of(rho), a])
        rows = rng.choice(n, size=m, replace=False)
        X[rows, a] = rng.uniform(lo[a], hi[a], size=m)
    return X, dict(S_A=np.arange(0), n_values=np.array(m * d))


def es_split(n, seed):
    """ES: validation positions, round(ES_VAL n) drawn with default_rng([s, 11]); the rest trains."""
    rng = np.random.default_rng([seed, 11])
    val = np.sort(rng.choice(n, size=int(round(C.ES_VAL * n)), replace=False))
    return np.setdiff1d(np.arange(n), val), val


def loss_epochs(E):
    return sorted({max(1, int(round(f * E))) for f in C.LOSS_EPOCHS})


# ============================================================================= VolMinNet transition layer
class VolT(nn.Module):
    """T = row-normalise(I + sigmoid(W) * (1 - I)), W initialised at -2 (Li et al., 2021)."""

    def __init__(self, K):
        super().__init__()
        self.w = nn.Parameter(-2.0 * torch.ones(K, K))
        self.register_buffer("eye", torch.eye(K))

    def forward(self):
        T = self.eye + torch.sigmoid(self.w) * (1 - self.eye)
        return T / T.sum(1, keepdim=True)


def _vol_loss(logits, y, Tm):
    T = Tm()
    logq = torch.logsumexp(F.log_softmax(logits.float(), 1)[:, :, None] + torch.log(T)[None, :, :], dim=1)
    return F.nll_loss(logq, y) + C.VOL["lam"] * torch.linalg.slogdet(T)[1], T


# ============================================================================= extended training loop
@torch.no_grad()
def _predict_idx(model, task, split, idx, device, amp, bs=1024):
    model.eval()
    out = []
    idx = torch.as_tensor(np.asarray(idx))
    for b in range(0, len(idx), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out.append(torch.softmax(task["fwd"](model, task["get"](split, idx[b:b + bs], False, None)).float(), 1).cpu())
    return torch.cat(out).numpy()


def fit_x(task, y_train, y_test, cfg, seed, device, T=None, use_idx=None, val_idx=None, loss_at=(), vol=False,
          cot_rate=None, need_train_probs=False, es_probs=False):
    """As train.fit_predict, plus: training restricted to use_idx; per-epoch noisy-validation and test accuracy
    with the test accuracy at the best validation epoch (first maximum); per-example training losses (eval mode,
    noisy labels) after the epochs in loss_at; VolMinNet (vol=True) or Co-teaching (cot_rate = forget rate).
    es_probs (round 3, TE): also return the predictions on the whole training set of the model of the best epoch."""
    TR.seed_all(seed)
    model = task["model_fn"]().to(device)
    models_ = [model]
    if cot_rate is not None:
        torch.manual_seed(seed + 10007)
        models_.append(task["model_fn"]().to(device))
    pool = np.arange(task["n"]) if use_idx is None else np.asarray(use_idx)
    n, B, E = len(pool), cfg["batch"], cfg["epochs"]
    steps = E * math.ceil(n / B)
    opts = [TR._optimizer(cfg, m, steps) for m in models_]
    amp = bool(cfg.get("amp")) and device.startswith("cuda")
    y = torch.as_tensor(y_train, dtype=torch.long, device=device)
    Tt = None if T is None else torch.tensor(T, dtype=torch.float32, device=device)
    Tm = optT = None
    if vol:
        Tm = VolT(task["K"] if "K" in task else int(y.max().item()) + 1).to(device)
        optT = torch.optim.SGD(Tm.parameters(), lr=C.VOL["lr"], momentum=C.VOL["momentum"])
    tk = max(1, math.ceil(C.COT["tk_frac"] * E))
    gen = torch.Generator().manual_seed(seed)
    agen = torch.Generator(device=device).manual_seed(seed)
    pool_t = torch.as_tensor(pool)
    yt = np.asarray(y_test)
    curve, losses, best = [], {}, (-1.0, None, None)
    t0 = time.time()
    for ep in range(E):
        for m in models_:
            m.train()
        perm = pool_t[torch.randperm(n, generator=gen)]
        R = 0.0 if cot_rate is None else cot_rate * min(ep / tk, 1.0)
        for b in range(0, n, B):
            idx = perm[b:b + B]
            yi = y[idx.to(device)]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                xb = task["get"]("train", idx, True, agen)
                outs = [task["fwd"](m, xb) for m in models_]
            if cot_rate is not None:
                la = F.cross_entropy(outs[0].float(), yi, reduction="none")
                lb = F.cross_entropy(outs[1].float(), yi, reduction="none")
                k = max(1, int(math.ceil((1 - R) * len(idx))))
                ia, ib = torch.argsort(la.detach())[:k], torch.argsort(lb.detach())[:k]
                losses_ = [F.cross_entropy(outs[0].float()[ib], yi[ib]), F.cross_entropy(outs[1].float()[ia], yi[ia])]
            elif vol:
                loss, _ = _vol_loss(outs[0], yi, Tm)
                losses_ = [loss]
            else:
                losses_ = [TR.forward_loss(outs[0], yi, Tt)]
            for (opt, sched), loss in zip(opts, losses_):
                opt.zero_grad(set_to_none=True)
            if optT is not None:
                optT.zero_grad(set_to_none=True)
            sum(losses_).backward()
            for (opt, sched) in opts:
                opt.step()
                if sched is not None:
                    sched.step()
            if optT is not None:
                optT.step()
        if val_idx is not None:
            pv = _predict_idx(model, task, "train", val_idx, device, amp).argmax(1)
            va = float(np.mean(pv == np.asarray(y_train)[val_idx]))
            pt = _predict_idx(model, task, "test", np.arange(task["n_test"]), device, amp).argmax(1)
            ta = float(np.mean(pt == yt))
            curve.append((ep + 1, va, ta))
            if va > best[0]:
                best = (va, ep + 1, ta)
                if es_probs:
                    best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        if (ep + 1) in loss_at:
            p = TR._predict(model, task, "train", device, amp)
            yy = np.asarray(y_train)
            losses[ep + 1] = -np.log(np.clip(p[np.arange(len(yy)), yy], 1e-12, None)).astype(np.float32)
    train_time = time.time() - t0
    pte = TR._predict(model, task, "test", device, amp)
    pred = pte.argmax(1)
    res = dict(test_acc=float(np.mean(pred == yt)), test_macro_f1=TR.macro_f1(yt, pred, pte.shape[1]),
               train_time_s=train_time, epochs=E)
    if val_idx is not None:
        res.update(test_acc_es=best[2], best_epoch=best[1], curve=np.array(curve))
    if loss_at:
        res["losses"] = losses
    if vol:
        res["T_vol"] = Tm().detach().cpu().numpy()
    if need_train_probs:
        res["train_probs"] = TR._predict(model, task, "train", device, amp)
    if es_probs and val_idx is not None:
        model.load_state_dict(best_state)
        res["train_probs_es"] = TR._predict(model, task, "train", device, amp)
    for m in models_:
        del m
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return res


def fit_graph_x(X, y_full, y_train_noisy, edge_index, train_idx, test_idx, cfg, seed, device, T=None, use_pos=None,
                val_pos=None, loss_at=(), vol=False, cot_rate=None, track=False, need_train_probs=False, epochs=None,
                es_probs=False):
    """Graph counterpart of fit_x (full batch). Positions refer to train_idx; track records, per epoch, test
    accuracy, training loss and training accuracy against the noisy and the true labels."""
    from . import models as M
    TR.seed_all(seed)
    Xf = TR._row_norm(X)
    x = torch.tensor(Xf, device=device)
    A = M.gcn_adjacency(edge_index, len(Xf), device)
    K = cfg["K"]
    models_ = [M.GCN(Xf.shape[1], K).to(device)]
    if cot_rate is not None:
        torch.manual_seed(seed + 10007)
        models_.append(M.GCN(Xf.shape[1], K).to(device))
    opts = [torch.optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"]) for m in models_]
    pos = np.arange(len(train_idx)) if use_pos is None else np.asarray(use_pos)
    tr = torch.as_tensor(np.asarray(train_idx)[pos], device=device)
    yn = torch.as_tensor(np.asarray(y_train_noisy), dtype=torch.long, device=device)
    yp = yn[torch.as_tensor(pos, device=device)]
    Tt = None if T is None else torch.tensor(T, dtype=torch.float32, device=device)
    Tm = optT = None
    if vol:
        Tm = VolT(K).to(device)
        optT = torch.optim.SGD(Tm.parameters(), lr=C.VOL["lr"], momentum=C.VOL["momentum"])
    E = epochs or cfg["epochs"]
    tk = max(1, math.ceil(C.COT["tk_frac"] * E))
    yt = np.asarray(y_full)[test_idx]
    ytrue_tr = np.asarray(y_full)[train_idx]
    curve, losses, best = [], {}, (-1.0, None, None)
    t0 = time.time()
    for ep in range(E):
        for m in models_:
            m.train()
        outs = [m(x, A)[tr] for m in models_]
        if cot_rate is not None:
            R = cot_rate * min(ep / tk, 1.0)
            la = F.cross_entropy(outs[0], yp, reduction="none")
            lb = F.cross_entropy(outs[1], yp, reduction="none")
            k = max(1, int(math.ceil((1 - R) * len(yp))))
            ia, ib = torch.argsort(la.detach())[:k], torch.argsort(lb.detach())[:k]
            loss = F.cross_entropy(outs[0][ib], yp[ib]) + F.cross_entropy(outs[1][ia], yp[ia])
        elif vol:
            loss, _ = _vol_loss(outs[0], yp, Tm)
        else:
            loss = TR.forward_loss(outs[0], yp, Tt)
        for o in opts:
            o.zero_grad(set_to_none=True)
        if optT is not None:
            optT.zero_grad(set_to_none=True)
        loss.backward()
        for o in opts:
            o.step()
        if optT is not None:
            optT.step()
        if val_pos is not None or track or (ep + 1) in loss_at:
            models_[0].eval()
            with torch.no_grad():
                p = torch.softmax(models_[0](x, A), 1).cpu().numpy()
            ptr = p[train_idx]
            if val_pos is not None:
                va = float(np.mean(ptr[val_pos].argmax(1) == np.asarray(y_train_noisy)[val_pos]))
                ta = float(np.mean(p[test_idx].argmax(1) == yt))
                curve.append((ep + 1, va, ta))
                if va > best[0]:
                    best = (va, ep + 1, ta)
                    if es_probs:
                        best_ptr = ptr.copy()
            elif track:
                ynp = np.asarray(y_train_noisy)
                curve.append((ep + 1, float(loss.item()), float(np.mean(p[test_idx].argmax(1) == yt)),
                              float(np.mean(ptr.argmax(1) == ynp)), float(np.mean(ptr.argmax(1) == ytrue_tr))))
            if (ep + 1) in loss_at:
                ynp = np.asarray(y_train_noisy)
                losses[ep + 1] = -np.log(np.clip(ptr[np.arange(len(ynp)), ynp], 1e-12, None)).astype(np.float32)
    train_time = time.time() - t0
    m0 = models_[0]
    m0.eval()
    with torch.no_grad():
        p = torch.softmax(m0(x, A), 1).cpu().numpy()
    pred = p[test_idx].argmax(1)
    res = dict(test_acc=float(np.mean(pred == yt)), test_macro_f1=TR.macro_f1(yt, pred, K), train_time_s=train_time,
               epochs=E)
    if val_pos is not None:
        res.update(test_acc_es=best[2], best_epoch=best[1], curve=np.array(curve))
    elif track:
        res["curve"] = np.array(curve)
    if loss_at:
        res["losses"] = losses
    if vol:
        res["T_vol"] = Tm().detach().cpu().numpy()
    if need_train_probs:
        res["train_probs"] = p[train_idx]
    if es_probs and val_pos is not None:
        res["train_probs_es"] = best_ptr
    return res


# ============================================================================= run units
def _extra_dir(ds):
    p = os.path.join(C.RUNS, "extra", ds)
    os.makedirs(p, exist_ok=True)
    return p


def _save_extra(ds, pl, m, cond, seed, r, S_L=None, S_A=None):
    """Curves, losses and learned T go to extra/<ds>/<pl>_<m>_<cond>_s<seed>.npz; returns the relative path."""
    keep = {}
    for k in ("curve", "T_vol"):
        if k in r:
            keep[k] = r.pop(k)
    if "losses" in r:
        for ep, v in r.pop("losses").items():
            keep[f"loss_ep{ep}"] = v
    if not keep:
        return ""
    if S_L is not None:
        keep["S_L"] = np.asarray(S_L)
    if S_A is not None:
        keep["S_A"] = np.asarray(S_A)
    p = os.path.join(_extra_dir(ds), f"{pl}_{m}_{cond}_s{seed}.npz")
    np.savez_compressed(p, **keep)
    return os.path.relpath(p, C.RUNS)


def _fit(b, X_or_task, y_noisy, seed, device, **kw):
    cfg = dict(C.DATASETS[b.ds])
    if b.kind == "GRA":
        use = kw.pop("use_idx", None)
        val = kw.pop("val_idx", None)
        return fit_graph_x(X_or_task, b.d["y"], y_noisy, b.d["edge_index"], b.d["train_idx"], b.d["test_idx"], cfg,
                           seed, device, use_pos=use, val_pos=val, **kw)
    task = X_or_task
    task.setdefault("K", b.K)
    return fit_x(task, y_noisy, b.d["y_test"], cfg, seed, device, **kw)


def _inputs(b, cond, seed):
    """Training inputs and record for a round 2 condition (or a grid condition)."""
    rho_L, rho_A, level = C.CONDITIONS[cond]
    if cond == "R40":
        X = b.d["X"] if b.kind == "GRA" else b.d["X_train"]
        return X, dict(S_A=N.select_inputs(b.n, 0.4, seed), removed=np.array(1))
    if cond == "X40":
        return x40_inputs(b, seed)
    if cond == "AT40":
        assert b.kind == "TAB", "AT40 is defined for tabular data only"
        return at40_inputs(b, seed)
    return b.noisy_inputs(rho_A, level, seed)


def _task(b, X, device):
    return X if b.kind == "GRA" else b.task(X, device)


def run_block(b, seed, block, conds, methods, device):
    ds = b.ds
    for cond in conds:
        if block == "CN":
            _run_cn(b, seed, cond, methods, device)
            continue
        if block == "CD":
            _run_cd(b, seed, cond, methods, device)
            continue
        pl = "ES" if block == "ES" else "N"
        allowed = {"LS": ["CE"], "H1C": ["CE"], "H2P": ["VOL", "COT"], "ES": ["CE", "FCstar"]}[block]
        todo = [m for m in methods if m in allowed and not done(ds, pl, m, cond, seed)]
        if block == "ES" and cond not in ("L40", "J40"):
            todo = [m for m in todo if m != "FCstar"]
        if not todo:
            continue
        base = cond if cond in ("c0", "L40", "A40", "J40") else None
        rho_L = C.CONDITIONS[cond][0]
        y_noisy, S_L, T_true = N.label_noise(b.y_train, b.K, rho_L, seed)
        X, info = _inputs(b, cond, seed)
        _save_noise(ds, cond, seed, S_L, y_noisy, info)
        Xt = _task(b, X, device)
        S_A = info.get("S_A", np.arange(0))
        common = _common(b, y_noisy, S_L, S_A if cond not in ("R40", "AT40") else np.arange(0), device)
        kw = {}
        if cond == "R40":
            kw["use_idx"] = np.setdiff1d(np.arange(b.n), S_A)
            common["n_train"] = b.n - len(S_A)
        if block == "ES":
            use, val = es_split(b.n, seed)
            kw.update(use_idx=use, val_idx=val)
        for m in todo:
            mkw = dict(kw)
            if block == "LS":
                mkw["loss_at"] = loss_epochs(C.DATASETS[ds]["epochs"])
            if m == "FCstar":
                mkw["T"] = T_true
            if m == "VOL":
                mkw["vol"] = True
            if m == "COT":
                mkw["cot_rate"] = rho_L
            r = _fit(b, Xt, y_noisy, seed, device, **mkw)
            extra = _save_extra(ds, pl, m, cond, seed, r, S_L, S_A)
            if m == "VOL":
                common_m = dict(common, T_hat_frob_err=float(np.linalg.norm(np.load(os.path.join(C.RUNS, extra))["T_vol"]
                                                                              - T_true) / b.K))
            else:
                common_m = common
            write_row(ds, pl, m, cond, seed, **r, **common_m, extra_path=extra)
            log(f"{ds} {pl} {m} {cond} s{seed} acc={r['test_acc']:.4f}"
                + (f" es={r['test_acc_es']:.4f}@{r['best_epoch']}" if "test_acc_es" in r else ""))
        del Xt
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- CIFAR-10N
CN_KEYS = {"agg": "aggre_label", "r1": "random_label1", "worst": "worse_label"}


def cn_labels(b, cond, seed):
    """Nagg/Nr1/Nworst: the human labels; Sagg/Sr1/Sworst: symmetric noise at the realised rate of that set."""
    from . import data as D
    z = D.load_cifar10n()
    clean = z["clean_label"]
    assert np.array_equal(clean, b.y_train), "CIFAR-10N labels are not in the torchvision order"
    key = CN_KEYS[cond[1:]]
    yn = z[key]
    rate = float(np.mean(yn != clean))
    if cond.startswith("N"):
        Tn = np.zeros((b.K, b.K))
        np.add.at(Tn, (clean, yn), 1.0)
        Tn /= Tn.sum(1, keepdims=True)
        return yn, np.flatnonzero(yn != clean), Tn, rate
    ys, S, _ = N.label_noise(b.y_train, b.K, rate, seed)
    return ys, S, N.symmetric_T(b.K, rate), rate


def _run_cn(b, seed, cond, methods, device):
    ds = b.ds
    assert b.ds == "IMG"
    todo = [m for m in methods if m in ("CE", "FC", "FCstar") and not done(ds, "N", m, cond, seed)]
    if not todo:
        return
    y_noisy, S_L, T_true, rate = cn_labels(b, cond, seed)
    X = b.d["X_train"]
    Xt = b.task(X, device)
    common = dict(n_train=b.n, n_label_noisy=int(np.sum(y_noisy != b.y_train)), n_input_noisy=0, n_overlap=0,
                  git_commit=code_version(), gpu=gpu_name(device), notes=f"rate={rate:.4f}", rho_L=round(rate, 4))
    pp = _probs_path(ds, "N", cond, seed)
    if "CE" in todo or ("FC" in todo and not os.path.exists(pp)):
        r = b.fit(Xt, y_noisy, seed, device, need_train_probs=True)
        os.makedirs(os.path.dirname(pp), exist_ok=True)
        np.save(pp, r.pop("train_probs").astype(np.float32))
        if not done(ds, "N", "CE", cond, seed):
            write_row(ds, "N", "CE", cond, seed, **r, **common)
            log(f"{ds} N CE {cond} s{seed} acc={r['test_acc']:.4f}")
    if "FC" in todo:
        T_hat = TR.estimate_T(np.load(pp))
        tp = pp.replace(".npy", "_That.npy")
        np.save(tp, T_hat)
        r = b.fit(Xt, y_noisy, seed, device, T=T_hat)
        write_row(ds, "N", "FC", cond, seed, **r, **common, T_hat_path=os.path.relpath(tp, C.RUNS),
                  T_hat_frob_err=float(np.linalg.norm(T_hat - T_true) / b.K))
        log(f"{ds} N FC {cond} s{seed} acc={r['test_acc']:.4f}")
    if "FCstar" in todo:
        r = b.fit(Xt, y_noisy, seed, device, T=T_true)
        write_row(ds, "N", "FCstar", cond, seed, **r, **common)
        log(f"{ds} N FCstar {cond} s{seed} acc={r['test_acc']:.4f}")


# ----------------------------------------------------------------------------- Cora diagnosis
def _run_cd(b, seed, cond, methods, device):
    """cond 'L40' (200 epochs) or 'L40e400' (400 epochs); per-epoch curves."""
    ds = b.ds
    assert b.kind == "GRA"
    todo = [m for m in methods if m in ("CE", "FCstar") and not done(ds, "CD", m, cond, seed)]
    if not todo:
        return
    epochs = 400 if cond.endswith("e400") else None
    y_noisy, S_L, T_true = N.label_noise(b.y_train, b.K, 0.4, seed)
    common = _common(b, y_noisy, S_L, np.arange(0), device)
    for m in todo:
        r = fit_graph_x(b.d["X"], b.d["y"], y_noisy, b.d["edge_index"], b.d["train_idx"], b.d["test_idx"],
                        dict(C.DATASETS[ds]), seed, device, T=T_true if m == "FCstar" else None, track=True,
                        epochs=epochs)
        extra = _save_extra(ds, "CD", m, cond, seed, r, S_L)
        write_row(ds, "CD", m, cond, seed, **r, **common, extra_path=extra)
        log(f"{ds} CD {m} {cond} s{seed} acc={r['test_acc']:.4f}")
