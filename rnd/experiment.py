"""Run units of the grid: native pipeline (N, H1-H2), frozen encoders (H3) and
labels from foundation models (H4). Every run writes one CSV; existing CSVs are skipped (resume)."""
import csv
import glob
import hashlib
import json
import os
import time

import numpy as np
import torch

from . import config as C
from . import data as D
from . import noise as N
from . import train as TR


# ----------------------------------------------------------------------------- bookkeeping
def code_version():
    h = hashlib.sha1()
    for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))):
        h.update(open(f, "rb").read())
    return "code:" + h.hexdigest()[:10]


def gpu_name(device):
    return torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else "cpu"


def csv_path(ds, pipeline, method, cond, seed):
    return os.path.join(C.RUNS, ds, pipeline, method, f"{cond}_s{seed}.csv")


def done(*a):
    return os.path.exists(csv_path(*a))


def write_row(ds, pipeline, method, cond, seed, **kw):
    row = {k: "" for k in C.CSV_COLUMNS}
    rho_L, rho_A, level = C.CONDITIONS.get(cond, (kw.pop("rho_L", ""), 0.0, ""))
    row.update(dataset=ds, modality=C.DATASETS[ds]["modality"], pipeline=pipeline, method=method,
               condition=cond, rho_L=rho_L, rho_A=rho_A, level=level, seed=seed, deterministic=1)
    row.update({k: v for k, v in kw.items() if k in row})
    p = csv_path(ds, pipeline, method, cond, seed)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=C.CSV_COLUMNS)
        w.writeheader()
        w.writerow(row)
    os.replace(tmp, p)


def log(msg):
    os.makedirs(C.RUNS, exist_ok=True)
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(C.RUNS, "run.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------------------- noisy native training sets
class Bundle:
    """Loaded dataset plus cached per-condition inputs."""

    def __init__(self, ds):
        self.ds = ds
        self.kind = C.KIND[ds]                  # modality of the dataset (round 2: second datasets share it)
        self.d = D.load(ds)
        self.K = C.DATASETS[ds]["K"]
        if self.kind == "GRA":
            self.y_train = self.d["y"][self.d["train_idx"]]
            self.n = len(self.d["train_idx"])
        else:
            self.y_train = np.asarray(self.d["y_train"])
            self.n = len(self.y_train)
        if self.kind == "TAB":
            self.sd_clean = self.d["X_train"].std(axis=0)           # ddof = 0, clean training set
        if self.kind == "IMG":
            self.stats = TR.img_stats(self.d["X_train"])
        self.tokenizer = None

    def noisy_inputs(self, rho_A, level, seed):
        """Training inputs after input noise and the record of what was corrupted."""
        S_A = N.select_inputs(self.n, rho_A, seed)
        info = dict(S_A=S_A)
        if rho_A <= 0:
            X = self.d["X"] if self.kind == "GRA" else self.d["X_train"]
            return X, info
        rng = N.corruption_rng(seed, rho_A, level)
        lv = C.LEVELS[self.kind][level]
        if self.kind == "TAB":
            X = N.corrupt_tabular(self.d["X_train"], S_A, self.sd_clean, lv, rng)
        elif self.kind == "IMG":
            X, info["types"] = N.corrupt_images(self.d["X_train"], S_A, lv, rng)
        elif self.kind == "TXT":
            X = N.corrupt_texts(self.d["X_train"], S_A, lv, rng)
        elif self.kind == "TS":
            X = N.corrupt_series(self.d["X_train"], S_A, lv, rng)
        elif self.kind == "GRA":
            X = N.corrupt_graph_features(self.d["X"], self.d["train_idx"][S_A], lv, rng)
        elif self.kind == "AUD":
            X, info["silent"] = N.corrupt_waves(self.d["X_train"], S_A, lv, rng)
        return X, info

    def noisy_test(self, seed, X_graph=None):
        """E2 (round 1): test inputs with a fraction NOISY_TEST['rho'] corrupted at level NOISY_TEST['level'].

        Depends only on the seed. For GRA, X_graph is the feature matrix used in training (after any training
        input noise) and only the selected test nodes are corrupted, at evaluation time."""
        rho, level = C.NOISY_TEST["rho"], C.NOISY_TEST["level"]
        n_test = len(self.d["test_idx"]) if self.kind == "GRA" else len(self.d["y_test"])
        S_T = N.select_test(n_test, rho, seed)
        rng = N.test_corruption_rng(seed, rho, level)
        lv = C.LEVELS[self.kind][level]
        if self.kind == "TAB":
            return N.corrupt_tabular(self.d["X_test"], S_T, self.sd_clean, lv, rng)
        if self.kind == "IMG":
            return N.corrupt_images(self.d["X_test"], S_T, lv, rng)[0]
        if self.kind == "TXT":
            return N.corrupt_texts(self.d["X_test"], S_T, lv, rng)
        if self.kind == "TS":
            return N.corrupt_series(self.d["X_test"], S_T, lv, rng)
        if self.kind == "GRA":
            return N.corrupt_graph_features(X_graph, self.d["test_idx"][S_T], lv, rng)
        if self.kind == "AUD":
            return N.corrupt_waves(self.d["X_test"], S_T, lv, rng)[0]
        raise ValueError(self.ds)

    def task(self, X, device, Xte_noisy=None):
        if self.kind == "TAB":
            return TR.task_tab(X, self.d["X_test"], device, Xte_noisy, K=self.K)
        if self.kind == "IMG":
            return TR.task_img(X, self.d["X_test"], self.stats, device, Xte_noisy, K=self.K)
        if self.kind == "TXT":
            if self.tokenizer is None:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            return TR.task_txt(X, self.d["X_test"], device, self.tokenizer, Xte_noisy, K=self.K)
        if self.kind == "TS":
            return TR.task_ts(X, self.d["X_test"], device, Xte_noisy, K=self.K)
        if self.kind == "AUD":
            return TR.task_aud(X, self.d["X_test"], device, waves_te_noisy=Xte_noisy, K=self.K,
                               T=C.DATASETS[self.ds].get("wave_len", 64000))
        raise ValueError(self.ds)

    def fit(self, X_or_task, y_noisy, seed, device, T=None, need_train_probs=False, X_noisy_test=None):
        cfg = dict(C.DATASETS[self.ds])
        if self.kind == "GRA":
            return TR.fit_predict_graph(X_or_task, self.d["y"], y_noisy, self.d["edge_index"], self.d["train_idx"],
                                        self.d["test_idx"], cfg, seed, device, T, need_train_probs, X_noisy_test)
        return TR.fit_predict(X_or_task, y_noisy, self.d["y_test"], cfg, seed, device, T, need_train_probs)


def _save_noise(ds, cond, seed, S_L, y_noisy, info):
    os.makedirs(C.NOISE, exist_ok=True)
    p = os.path.join(C.NOISE, f"{ds}_{cond}_s{seed}.npz")
    if not os.path.exists(p):
        extra = {k: np.asarray(v) for k, v in info.items()}
        np.savez_compressed(p, S_L=S_L, new_labels=y_noisy[S_L], **extra)


def _probs_path(ds, pipeline, cond, seed):
    return os.path.join(C.PROBS, ds, pipeline, f"{cond}_s{seed}.npy")


def _common(b, y_noisy, S_L, S_A, device):
    return dict(n_train=b.n, n_label_noisy=int(np.sum(y_noisy != b.y_train)), n_input_noisy=len(S_A),
                n_overlap=len(np.intersect1d(S_L, S_A)), git_commit=code_version(), gpu=gpu_name(device))


# ----------------------------------------------------------------------------- N block (H1, H2)
def run_native(b, seed, conds, methods, device, noisy_test=False):
    ds = b.ds
    for cond in conds:
        todo = [m for m in methods if (cond in C.GRID_N[m] or cond in C.GRID_E1[m]) and not done(ds, "N", m, cond, seed)]
        if not todo:
            continue
        rho_L, rho_A, level = C.CONDITIONS[cond]
        y_noisy, S_L, T_true = N.label_noise(b.y_train, b.K, rho_L, seed)
        X, info = b.noisy_inputs(rho_A, level, seed)
        _save_noise(ds, cond, seed, S_L, y_noisy, info)
        notes = f"silent_clips={len(info.get('silent', []))}" if "silent" in info else ""
        Xg_noisy = None
        if b.kind == "GRA":
            Xt = X
            Xg_noisy = b.noisy_test(seed, X) if noisy_test else None
        else:
            Xt = b.task(X, device, b.noisy_test(seed) if noisy_test else None)
        common = _common(b, y_noisy, S_L, info["S_A"], device)
        pp = _probs_path(ds, "N", cond, seed)
        if "CE" in todo or ("FC" in todo and not os.path.exists(pp)):
            r = b.fit(Xt, y_noisy, seed, device, need_train_probs=True, X_noisy_test=Xg_noisy)
            os.makedirs(os.path.dirname(pp), exist_ok=True)
            np.save(pp, r.pop("train_probs").astype(np.float32))
            if not done(ds, "N", "CE", cond, seed):
                write_row(ds, "N", "CE", cond, seed, **r, **common, notes=notes)
                log(f"{ds} N CE {cond} s{seed} acc={r['test_acc']:.4f} t={r['train_time_s']:.0f}s")
        if "FC" in todo:
            T_hat = TR.estimate_T(np.load(pp))
            tp = pp.replace(".npy", "_That.npy")
            np.save(tp, T_hat)
            err = float(np.linalg.norm(T_hat - T_true) / b.K)
            r = b.fit(Xt, y_noisy, seed, device, T=T_hat, X_noisy_test=Xg_noisy)
            write_row(ds, "N", "FC", cond, seed, **r, **common, T_hat_path=os.path.relpath(tp, C.RUNS),
                      T_hat_frob_err=err, notes=notes)
            log(f"{ds} N FC {cond} s{seed} acc={r['test_acc']:.4f} errT={err:.3f}")
        if "FCstar" in todo:
            r = b.fit(Xt, y_noisy, seed, device, T=T_true, X_noisy_test=Xg_noisy)
            write_row(ds, "N", "FCstar", cond, seed, **r, **common, notes=notes)
            log(f"{ds} N FCstar {cond} s{seed} acc={r['test_acc']:.4f}")
        del Xt
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- H4 block
def run_fm(b, seed, conds, methods, device):
    ds = b.ds
    z = np.load(os.path.join(C.NOISE, f"{ds}_F.npz"), allow_pickle=True)
    yhat, r_F, T_F, rev = z["yhat"], float(z["r_F"]), z["T_F"], str(z["revision"])
    Xt = None
    for cond in conds:
        todo = [m for m in methods if not done(ds, "N", m, cond, seed)]
        if not todo:
            continue
        y_noisy, T = N.fm_condition(cond, b.y_train, yhat, b.K, seed, r_F, T_F)
        if Xt is None:
            X = b.d["X"] if ds == "GRA" else b.d["X_train"]
            Xt = X if ds == "GRA" else b.task(X, device)
        common = dict(n_train=b.n, n_label_noisy=int(np.sum(y_noisy != b.y_train)), n_input_noisy=0, n_overlap=0,
                      git_commit=code_version(), gpu=gpu_name(device), annotator=rev, r_F=r_F, rho_L=r_F)
        for m in todo:
            r = b.fit(Xt, y_noisy, seed, device, T=None if m == "CE" else T)
            write_row(ds, "N", m, cond, seed, **r, **common)
            log(f"{ds} H4 {m} {cond} s{seed} acc={r['test_acc']:.4f} rF={r_F:.3f}")


def annotate(b, device):
    from . import annotate as A
    p = os.path.join(C.NOISE, f"{b.ds}_F.npz")
    if os.path.exists(p):
        return
    if b.ds in C.AUD_FOLDS:                     # round 3: all clips annotated once, then each fold
        from . import v15
        yhat_all, rev = v15.annotate_all_us8k(device)
        yhat = yhat_all[b.d["train_rows"]]
        r_F, T_F = N.annotator_stats(b.y_train, yhat, b.K)
        np.savez_compressed(p, yhat=yhat, revision=rev, r_F=r_F, T_F=T_F, rows=b.d["train_rows"])
        log(f"{b.ds} annotated by {rev} (all clips): r_F={r_F:.4f}")
        return
    if b.ds == "AUD":
        X = D.us8k_waves(b.d["train_rows"], C.ANNOTATORS["AUD"]["sr"])
    else:
        X = b.d["X_train"]
    yhat, r_F, T_F, rev = A.run(b.ds, X, b.y_train, device)
    log(f"{b.ds} annotated by {rev}: r_F={r_F:.4f}")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()             # release the annotator's memory before the H4 runs


# ----------------------------------------------------------------------------- H3 block
def _emb_dir(ds):
    p = os.path.join(C.EMB, ds)
    os.makedirs(p, exist_ok=True)
    return p


def clean_embeddings(b, enc, tag=""):
    p = os.path.join(_emb_dir(b.ds + tag), "clean.npz")
    if not os.path.exists(p):
        if b.ds == "AUD":
            sr = C.ENCODERS["AUD"]["sr"]
            Etr = enc(D.us8k_waves(b.d["train_rows"], sr))
            Ete = enc(D.us8k_waves(b.d["test_rows"], sr))
        else:
            Etr, Ete = enc(b.d["X_train"]), enc(b.d["X_test"])
        np.savez(p, train=Etr, test=Ete, revision=enc.revision)
    z = np.load(p)
    return z["train"], z["test"]


def en_embeddings(b, enc, seed, rho=0.4, level=2, tag=""):
    """Embeddings of the corrupted inputs of S_A (same S_A and rng_C as the native pipeline).
    tag (round 2 H3C) selects another cache folder, e.g. "_RN" for the randomly initialised encoder."""
    p = os.path.join(_emb_dir(b.ds + tag), f"EN_r{N.r_of(rho)}_l{level}_s{seed}.npz")
    if not os.path.exists(p):
        S_A = N.select_inputs(b.n, rho, seed)
        rng = N.corruption_rng(seed, rho, level)
        lv = C.LEVELS[b.ds][level]
        if b.ds == "IMG":
            Xc, _ = N.corrupt_images(b.d["X_train"][S_A], np.arange(len(S_A)), lv, rng)
            E = enc(Xc)
        elif b.ds == "TXT":
            sel = [b.d["X_train"][i] for i in S_A]
            E = enc(N.corrupt_texts(sel, np.arange(len(S_A)), lv, rng))
        elif b.ds == "AUD":                                     # noise added at 32 kHz with the SNR of 4.2
            w32 = D.us8k_waves(b.d["train_rows"][S_A], C.ENCODERS["AUD"]["sr"])
            w32, _ = N.corrupt_waves(w32, np.arange(len(S_A)), lv, rng)
            E = enc(w32)
        np.savez(p, S_A=S_A, E=E)
    z = np.load(p)
    return z["S_A"], z["E"]


def run_frozen(b, seed, conds, methods, device, enc=None):
    ds = b.ds
    need = [(pl, m, c) for c in conds for m in methods for pl in (["EN"] if c in ("c0", "L40") else ["EN", "EE"])
            if not done(ds, pl, m, c, seed)]
    if not need:
        return
    if enc is None:
        from .embed import get_encoder
        enc = get_encoder(ds, device)
    Etr, Ete = clean_embeddings(b, enc)
    S_A, E_en = en_embeddings(b, enc, seed)
    delta = float(np.mean(np.linalg.norm(E_en - Etr[S_A], axis=1)))
    sigma_E = delta / np.sqrt(Etr.shape[1])
    head = dict(opt="adam", lr=C.HEAD["lr"], wd=C.HEAD["wd"], batch=C.HEAD["batch"], epochs=C.HEAD["epochs"], K=b.K)
    for cond in conds:
        rho_L, rho_A, level = C.CONDITIONS[cond]
        y_noisy, S_L, T_true = N.label_noise(b.y_train, b.K, rho_L, seed)
        for pl in (["EN"] if cond in ("c0", "L40") else ["EN", "EE"]):
            todo = [m for m in methods if not done(ds, pl, m, cond, seed)]
            if not todo:
                continue
            E = Etr.copy()
            if rho_A > 0 and pl == "EN":
                E[S_A] = E_en
            elif rho_A > 0 and pl == "EE":
                E = N.embedding_noise(Etr, S_A, sigma_E, seed, rho_A)
            task = TR.task_head(E, Ete, b.K, device)
            common = dict(n_train=b.n, n_label_noisy=len(S_L), n_input_noisy=len(S_A) if rho_A > 0 else 0,
                          n_overlap=len(np.intersect1d(S_L, S_A)) if rho_A > 0 else 0, git_commit=code_version(),
                          gpu=gpu_name(device), delta=delta if pl == "EE" else "", sigma_E=sigma_E if pl == "EE" else "",
                          notes=f"encoder={enc.revision}")
            pp = _probs_path(ds, pl, cond, seed)
            if "CE" in todo or not os.path.exists(pp):
                r = TR.fit_predict(task, y_noisy, b.d["y_test"], head, seed, device, need_train_probs=True)
                os.makedirs(os.path.dirname(pp), exist_ok=True)
                np.save(pp, r.pop("train_probs").astype(np.float32))
                if not done(ds, pl, "CE", cond, seed):
                    write_row(ds, pl, "CE", cond, seed, **r, **common)
                    log(f"{ds} {pl} CE {cond} s{seed} acc={r['test_acc']:.4f}")
            if "FC" in todo:
                T_hat = TR.estimate_T(np.load(pp))
                tp = pp.replace(".npy", "_That.npy")
                np.save(tp, T_hat)
                r = TR.fit_predict(task, y_noisy, b.d["y_test"], head, seed, device, T=T_hat)
                write_row(ds, pl, "FC", cond, seed, **r, **common, T_hat_path=os.path.relpath(tp, C.RUNS),
                          T_hat_frob_err=float(np.linalg.norm(T_hat - T_true) / b.K))
                log(f"{ds} {pl} FC {cond} s{seed} acc={r['test_acc']:.4f}")
    return enc
