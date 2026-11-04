"""Training and evaluation of CE, FC and FC* on every pipeline.

A "task" bundles the tensors of one (dataset, condition, seed) and how to feed them to a model;
fit_predict trains from scratch with the dataset's fixed hyperparameters and returns the clean-test
accuracy and macro-F1 after the last epoch, plus the softmax outputs on the training set (eval mode,
no augmentation) that FC needs to estimate its transition matrix.
"""
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from . import models as M


# ----------------------------------------------------------------------------- utilities
def seed_all(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def macro_f1(y, pred, K):
    f1 = []
    for k in range(K):
        tp = np.sum((pred == k) & (y == k))
        fp = np.sum((pred == k) & (y != k))
        fn = np.sum((pred != k) & (y == k))
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def forward_loss(logits, y, T):
    """CE when T is None; otherwise -log((softmax(f(x)) T)_y) (forward correction), computed in log space as
    -logsumexp_j(log_softmax(f(x))_j + log T[j, y]). This is exact, keeps the gradient of confidently wrong
    examples (log(softmax) with a floor zeroes it) and equals CE when T = I."""
    logits = logits.float()
    if T is None:
        return F.cross_entropy(logits, y)
    logq = torch.logsumexp(F.log_softmax(logits, 1)[:, :, None] + torch.log(T)[None, :, :], dim=1)   # (B, K)
    return F.nll_loss(logq, y)


def estimate_T(probs, q=0.97):
    """Anchor points (Patrini et al.): row i is p_hat at the example in position ceil(q n) of the
    ascending order of p_hat_i (1-based), i.e. the 97th percentile."""
    n, K = probs.shape
    pos = int(math.ceil(q * n)) - 1
    T = np.zeros((K, K))
    for i in range(K):
        T[i] = probs[np.argsort(probs[:, i], kind="stable")[pos]]
    return T


def estimate_T_classwise(probs, y_noisy, q=0.97):
    """OOS: row i is p_hat at the example in position ceil(q n_i) of the ascending order of
    p_hat_i among the n_i examples whose noisy label is i (the global rule of estimate_T if n_i = 0)."""
    n, K = probs.shape
    y_noisy = np.asarray(y_noisy)
    T = estimate_T(probs, q)
    for i in range(K):
        cand = np.flatnonzero(y_noisy == i)
        if len(cand):
            pos = int(math.ceil(q * len(cand))) - 1
            T[i] = probs[cand[np.argsort(probs[cand, i], kind="stable")[pos]]]
    return T


def estimate_T_argmax(probs):
    """Anchor rule without filtering (Patrini et al.): row i is p_hat at the example with the largest p_hat_i."""
    return probs[np.argmax(probs, axis=0)].copy()


# ----------------------------------------------------------------------------- tasks per modality
def _std(X, axis):
    mu = X.mean(axis=axis, keepdims=True)
    sd = X.std(axis=axis, keepdims=True)
    sd[sd == 0] = 1.0
    return mu, sd


def task_tab(Xtr, Xte, device, Xte_noisy=None, K=None, stats_X=None):
    K = K or C.DATASETS["TAB"]["K"]
    # statistics of the training set after noise injection (grid), or of stats_X (round 4 SC: clean training set)
    mu, sd = _std(Xtr if stats_X is None else stats_X, 0)
    S = {"train": torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=device),
         "test": torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=device)}
    if Xte_noisy is not None:                    # E2: same normalisation as the clean test set
        S["test_noisy"] = torch.tensor((Xte_noisy - mu) / sd, dtype=torch.float32, device=device)
    return dict(n=len(S["train"]), model_fn=lambda: M.MLP(S["train"].shape[1], K),
                get=lambda split, idx, aug, g: S[split][idx],
                fwd=lambda m, x: m(x), n_test=len(S["test"]), has_noisy=Xte_noisy is not None)


def task_ts(Xtr, Xte, device, Xte_noisy=None, K=None, stats_X=None):
    K = K or C.DATASETS["TS"]["K"]
    # per channel over all windows and steps of the noisy training set (grid), or of stats_X (round 4 SC)
    mu, sd = _std(Xtr if stats_X is None else stats_X, (0, 2))
    S = {"train": torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=device),
         "test": torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=device)}
    if Xte_noisy is not None:
        S["test_noisy"] = torch.tensor((Xte_noisy - mu) / sd, dtype=torch.float32, device=device)
    return dict(n=len(S["train"]), model_fn=lambda: M.FCN(S["train"].shape[1], K),
                get=lambda split, idx, aug, g: S[split][idx],
                fwd=lambda m, x: m(x), n_test=len(S["test"]), has_noisy=Xte_noisy is not None)


def img_stats(X_clean):
    x = X_clean.astype(np.float64) / 255.0
    return x.mean(axis=(0, 1, 2)), x.std(axis=(0, 1, 2))


def task_img(Xtr, Xte, stats, device, Xte_noisy=None, K=None):
    K = K or C.DATASETS["IMG"]["K"]
    S = {"train": torch.tensor(Xtr, device=device).permute(0, 3, 1, 2).contiguous(),      # uint8 (n, 3, 32, 32)
         "test": torch.tensor(Xte, device=device).permute(0, 3, 1, 2).contiguous()}
    if Xte_noisy is not None:
        S["test_noisy"] = torch.tensor(Xte_noisy, device=device).permute(0, 3, 1, 2).contiguous()
    tr, te = S["train"], S["test"]
    mean = torch.tensor(stats[0], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(stats[1], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    ar = torch.arange(32, device=device)

    def get(split, idx, aug, g):
        x = S[split][idx.to(device)].float() / 255.0
        if aug:                                   # random crop 32 with 4-pixel zero padding, h-flip p = 0.5
            B = x.shape[0]
            x = F.pad(x, (4, 4, 4, 4))
            ox = torch.randint(0, 9, (B,), generator=g, device=device)
            oy = torch.randint(0, 9, (B,), generator=g, device=device)
            rows = (oy[:, None] + ar[None, :])[:, None, :, None]
            cols = (ox[:, None] + ar[None, :])[:, None, None, :]
            x = x[torch.arange(B, device=device)[:, None, None, None],
                  torch.arange(3, device=device)[None, :, None, None], rows, cols]
            flip = torch.rand(B, generator=g, device=device) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
        return ((x - mean) / std).contiguous(memory_format=torch.channels_last)

    return dict(n=len(tr), model_fn=lambda: M.resnet18_cifar(K).to(
        memory_format=torch.channels_last), get=get, fwd=lambda m, x: m(x), n_test=len(te),
        has_noisy=Xte_noisy is not None)


def task_txt(texts_tr, texts_te, device, tokenizer=None, texts_te_noisy=None, K=None):
    K = K or C.DATASETS["TXT"]["K"]
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = "distilbert-base-uncased"
    tok = tokenizer or AutoTokenizer.from_pretrained(name)
    L = C.DATASETS["TXT"]["max_len"]

    def enc(t):
        e = tok(list(t), truncation=True, max_length=L, padding="max_length", return_tensors="pt")
        return e["input_ids"].to(device), e["attention_mask"].to(device)
    S = {"train": enc(texts_tr), "test": enc(texts_te)}
    if texts_te_noisy is not None:
        S["test_noisy"] = enc(texts_te_noisy)
    return dict(n=len(S["train"][0]),
                model_fn=lambda: AutoModelForSequenceClassification.from_pretrained(name, num_labels=K),
                get=lambda split, idx, aug, g: tuple(t[idx.to(device)] for t in S[split]),
                fwd=lambda m, x: m(input_ids=x[0], attention_mask=x[1]).logits, n_test=len(S["test"][0]),
                has_noisy=texts_te_noisy is not None)


def task_aud(waves_tr, waves_te, device, batch=512, waves_te_noisy=None, K=None, T=64000):
    K = K or C.DATASETS["AUD"]["K"]
    fb = M.mel_filterbank()

    def feats(waves):
        out = []
        for b in range(0, len(waves), batch):
            chunk = np.zeros((len(waves[b:b + batch]), T), dtype=np.float32)
            for j, w in enumerate(waves[b:b + batch]):     # pad with zeros or truncate to 4 s after noise
                chunk[j, :min(T, len(w))] = w[:T]
            out.append(M.log_mel(torch.tensor(chunk, device=device), fb).half())
        return torch.cat(out)
    S = {"train": feats(waves_tr), "test": feats(waves_te)}
    if waves_te_noisy is not None:
        S["test_noisy"] = feats(waves_te_noisy)
    return dict(n=len(S["train"]), model_fn=lambda: M.AudioCNN(K),
                get=lambda split, idx, aug, g: S[split][idx.to(device)].float(),
                fwd=lambda m, x: m(x), n_test=len(S["test"]), has_noisy=waves_te_noisy is not None)


def task_head(Etr, Ete, K, device):
    tr = torch.tensor(Etr, dtype=torch.float32, device=device)
    te = torch.tensor(Ete, dtype=torch.float32, device=device)
    return dict(n=len(tr), model_fn=lambda: M.LinearHead(tr.shape[1], K),
                get=lambda split, idx, aug, g: (tr if split == "train" else te)[idx.to(device)],
                fwd=lambda m, x: m(x), n_test=len(te))


# ----------------------------------------------------------------------------- generic loop
def _optimizer(cfg, model, steps):
    if cfg["opt"] == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                              nesterov=cfg["nesterov"], weight_decay=cfg["wd"])
    elif cfg["opt"] == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = None
    if cfg.get("sched") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.0)
    elif cfg.get("sched") == "linear":
        warm = int(round(cfg["warmup"] * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    return opt, sched


@torch.no_grad()
def _predict(model, task, split, device, amp, bs=1024):
    model.eval()
    n = task["n"] if split == "train" else task["n_test"]
    out = []
    for b in range(0, n, bs):
        idx = torch.arange(b, min(n, b + bs))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out.append(torch.softmax(task["fwd"](model, task["get"](split, idx, False, None)).float(), 1).cpu())
    return torch.cat(out).numpy()


def fit_predict(task, y_train, y_test, cfg, seed, device, T=None, need_train_probs=False, need_test_probs=False):
    """Train from scratch (same seed for CE and FC) and evaluate after the last epoch."""
    seed_all(seed)
    model = task["model_fn"]().to(device)
    n, B, E = task["n"], cfg["batch"], cfg["epochs"]
    steps = E * math.ceil(n / B)
    opt, sched = _optimizer(cfg, model, steps)
    amp = bool(cfg.get("amp")) and device.startswith("cuda")
    y = torch.as_tensor(y_train, dtype=torch.long, device=device)
    Tt = None if T is None else torch.tensor(T, dtype=torch.float32, device=device)
    gen = torch.Generator().manual_seed(seed)                     # data order
    agen = torch.Generator(device=device).manual_seed(seed)       # augmentation
    t0 = time.time()
    for _ in range(E):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for b in range(0, n, B):
            idx = perm[b:b + B]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = task["fwd"](model, task["get"]("train", idx, True, agen))
            loss = forward_loss(logits, y[idx.to(device)], Tt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
    train_time = time.time() - t0
    pbs = task.get("pred_bs", 1024)                # round 2 H3C: smaller prediction batches for fine-tuned encoders
    pte = _predict(model, task, "test", device, amp, bs=pbs)
    pred = pte.argmax(1)
    yt = np.asarray(y_test)
    res = dict(test_acc=float(np.mean(pred == yt)), test_macro_f1=macro_f1(yt, pred, pte.shape[1]),
               train_time_s=train_time, epochs=E)
    if task.get("has_noisy"):                     # E2: same trained model, noisy test set
        pn = _predict(model, task, "test_noisy", device, amp, bs=pbs).argmax(1)
        res.update(test_acc_noisy=float(np.mean(pn == yt)), test_macro_f1_noisy=macro_f1(yt, pn, pte.shape[1]))
    if need_train_probs:
        res["train_probs"] = _predict(model, task, "train", device, amp, bs=pbs)
    if need_test_probs:                           # round 5 OOS: predictions on the held-out fold
        res["test_probs"] = pte
    del model
    torch.cuda.empty_cache() if device.startswith("cuda") else None
    return res


def _row_norm(X):
    Xf = X.astype(np.float32)
    rs = Xf.sum(1, keepdims=True)
    return np.divide(Xf, rs, out=np.zeros_like(Xf), where=rs > 0)


def fit_predict_graph(X, y_full, y_train_noisy, edge_index, train_idx, test_idx, cfg, seed, device, T=None,
                      need_train_probs=False, X_noisy_test=None, need_test_probs=False):
    """GRA: full-batch transductive training; features row-normalised after noise injection."""
    seed_all(seed)
    Xf = _row_norm(X)
    x = torch.tensor(Xf, device=device)
    A = M.gcn_adjacency(edge_index, len(Xf), device)
    model = M.GCN(Xf.shape[1], cfg["K"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    tr = torch.as_tensor(train_idx, device=device)
    y = torch.as_tensor(y_train_noisy, dtype=torch.long, device=device)
    Tt = None if T is None else torch.tensor(T, dtype=torch.float32, device=device)
    t0 = time.time()
    for _ in range(cfg["epochs"]):
        model.train()
        loss = forward_loss(model(x, A)[tr], y, Tt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    train_time = time.time() - t0
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(x, A), 1).cpu().numpy()
    yt = np.asarray(y_full)[test_idx]
    pred = p[test_idx].argmax(1)
    res = dict(test_acc=float(np.mean(pred == yt)), test_macro_f1=macro_f1(yt, pred, cfg["K"]),
               train_time_s=train_time, epochs=cfg["epochs"])
    if X_noisy_test is not None:                  # E2: test-node features corrupted at evaluation only
        with torch.no_grad():
            pn = model(torch.tensor(_row_norm(X_noisy_test), device=device), A).argmax(1).cpu().numpy()[test_idx]
        res.update(test_acc_noisy=float(np.mean(pn == yt)), test_macro_f1_noisy=macro_f1(yt, pn, cfg["K"]))
    if need_train_probs:
        res["train_probs"] = p[train_idx]
    if need_test_probs:                           # round 5 OOS: predictions on the held-out training nodes
        res["test_probs"] = p[test_idx]
    return res
