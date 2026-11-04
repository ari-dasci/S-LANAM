"""H3C: controls of H3 on IMG, TXT and AUD.

RN  the encoder of H3 with random weights (its library's initialisation after torch.manual_seed(0)) and a
    linear head on its L2-normalised embeddings; CE and FC.
MH  the pre-trained embeddings of EN with an MLP head (512 hidden units, ReLU), same optimiser, batch and
    epochs as the linear head; CE and FC.
FT  the pre-trained encoder fine-tuned end to end with a linear head on its (unnormalised) output; CE.
Conditions c0, L40, A40 and J40. Input noise as in EN: the inputs of S_A are corrupted at level 2 with the
same S_A and corruption stream as the native pipeline, before encoding.
"""
import os

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from . import data as D
from . import noise as N
from . import train as TR
from .experiment import (_probs_path, clean_embeddings, code_version, done, en_embeddings, gpu_name, log,
                         write_row)

H3C_CONDS = ["c0", "L40", "A40", "J40"]
H3C_METHODS = {"RN": ["CE", "FC"], "MH": ["CE", "FC"], "FT": ["CE"]}
FT_CFG = {
    "IMG": dict(opt="adamw", lr=1e-5, wd=0.1, batch=128, epochs=3, sched="cosine", amp=True),
    "TXT": dict(opt="adamw", lr=2e-5, wd=0.01, batch=32, epochs=3, sched="linear", warmup=0.06, amp=True,
                max_len=128),
    "AUD": dict(opt="adam", lr=1e-4, wd=0.0, batch=32, epochs=10, amp=False, sr=32000, clip_s=4.0),
}
MH_HIDDEN = 512


# ----------------------------------------------------------------------------- heads on embeddings (RN, MH)
class MLPHead(nn.Module):
    def __init__(self, d, K, h=MH_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, K))

    def forward(self, x):
        return self.net(x)


def task_mlp_head(Etr, Ete, K, device):
    t = TR.task_head(Etr, Ete, K, device)
    d = Etr.shape[1]
    t["model_fn"] = lambda: MLPHead(d, K)
    return t


def reinit_bert_(module, std=0.02):
    """In place: Linear and Embedding weights ~ N(0, std) with zero biases (padding row zeroed), LayerNorm 1/0."""
    import torch.nn as nn
    n = 0
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, nn.Linear):
                m.weight.normal_(0.0, std)
                if m.bias is not None:
                    m.bias.zero_()
                n += 1
            elif isinstance(m, nn.Embedding):
                m.weight.normal_(0.0, std)
                if m.padding_idx is not None:
                    m.weight[m.padding_idx].zero_()
                n += 1
            elif isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)
                m.bias.zero_()
                n += 1
    return n


def check_random_text_encoder(texts, device, enc_random, n=1000, max_cos=0.5):
    """round 3 check: mean cosine between random and pre-trained embeddings of the same texts must be low."""
    from . import embed as EM
    pre = EM.TextEncoder(device)
    a, b = pre(list(texts[:n])), enc_random(list(texts[:n]))
    cos = float(np.mean(np.sum(a * b, axis=1)))       # both L2-normalised
    del pre
    assert cos < max_cos, f"random text encoder too close to the pre-trained one (mean cosine {cos:.3f})"
    return cos


def random_encoder(ds, device):
    """The H3 encoder with its pre-processing but freshly initialised weights (torch.manual_seed(0))."""
    from . import embed as EM
    if ds == "IMG":
        import open_clip
        enc = EM.ImageEncoder(device)
        torch.manual_seed(0)
        m = open_clip.create_model(C.ENCODERS["IMG"]["arch"], pretrained=None, force_quick_gelu=True)
        enc.model = m.to(device).eval()
    elif ds == "TXT":
        # round 3: the round 2 version replaced tm.auto_model, which did not change the module that
        # Sentence-Transformers runs, so the encoder stayed pre-trained. The weights of every module of the
        # Hugging Face model are now re-initialised in place with the BERT initialisation.
        enc = EM.TextEncoder(device)
        torch.manual_seed(0)
        n_init = reinit_bert_(enc.model)
        assert n_init > 0, "no module re-initialised"
    elif ds == "AUD":
        from panns_inference.models import Cnn14
        enc = EM.AudioEncoder(device)
        torch.manual_seed(0)
        m = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000,
                  classes_num=527)
        dev = enc.at.device
        enc.at.model = torch.nn.DataParallel(m.to(dev)) if "cuda" in str(dev) else m
    else:
        raise ValueError(ds)
    enc.revision += ":random-init-seed0"
    return enc


def run_heads(b, seed, pl, conds, methods, device, enc):
    """RN (random encoder, linear head) or MH (pre-trained embeddings, MLP head); EN-style input noise."""
    ds = b.ds
    todo_any = [(m, c) for c in conds for m in methods if not done(ds, pl, m, c, seed)]
    if not todo_any:
        return
    tag = "_RN" if pl == "RN" else ""
    Etr, Ete = clean_embeddings(b, enc, tag=tag)
    S_A, E_en = en_embeddings(b, enc, seed, tag=tag)
    head = dict(opt="adam", lr=C.HEAD["lr"], wd=C.HEAD["wd"], batch=C.HEAD["batch"], epochs=C.HEAD["epochs"], K=b.K)
    for cond in conds:
        todo = [m for m in methods if not done(ds, pl, m, cond, seed)]
        if not todo:
            continue
        rho_L, rho_A, level = C.CONDITIONS[cond]
        y_noisy, S_L, T_true = N.label_noise(b.y_train, b.K, rho_L, seed)
        E = Etr.copy()
        if rho_A > 0:
            E[S_A] = E_en
        task = (TR.task_head if pl == "RN" else task_mlp_head)(E, Ete, b.K, device)
        common = dict(n_train=b.n, n_label_noisy=len(S_L), n_input_noisy=len(S_A) if rho_A > 0 else 0,
                      n_overlap=len(np.intersect1d(S_L, S_A)) if rho_A > 0 else 0, git_commit=code_version(),
                      gpu=gpu_name(device), notes=f"encoder={enc.revision};head={'linear' if pl == 'RN' else 'mlp512'}")
        pp = _probs_path(ds, pl, cond, seed)
        if "CE" in todo or ("FC" in todo and not os.path.exists(pp)):
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


# ----------------------------------------------------------------------------- end-to-end fine-tuning (FT)
class _ClipFT(nn.Module):
    def __init__(self, K):
        super().__init__()
        import open_clip
        cfg = C.ENCODERS["IMG"]
        m = open_clip.create_model(cfg["arch"], pretrained=cfg["pretrained"], force_quick_gelu=True)
        self.enc = m.visual
        self.head = nn.Linear(cfg["dim"], K)

    def forward(self, x):
        return self.head(self.enc(x).float())


class _E5FT(nn.Module):
    def __init__(self, K):
        super().__init__()
        from transformers import AutoModel
        cfg = C.ENCODERS["TXT"]
        self.enc = AutoModel.from_pretrained(cfg["name"])
        self.head = nn.Linear(cfg["dim"], K)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        m = mask.unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1).clamp_min(1.0))       # mean pooling, as E5


class _Cnn14FT(nn.Module):
    def __init__(self, K):
        super().__init__()
        from panns_inference.models import Cnn14
        self.enc = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000,
                         classes_num=527)
        ckpt = os.path.join(os.path.expanduser("~"), "panns_data", "Cnn14_mAP=0.431.pth")
        self.enc.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
        self.enc.spec_augmenter = nn.Identity()          # no augmentation
        self.head = nn.Linear(C.ENCODERS["AUD"]["dim"], K)

    def forward(self, x):
        return self.head(self.enc(x, None)["embedding"])


_WAVE_CACHE = {}
_IMG_PRE = []


def _waves32(b, rows_key):
    key = (b.ds, rows_key)
    if key not in _WAVE_CACHE:
        _WAVE_CACHE[key] = [np.asarray(w, dtype=np.float32) for w in D.us8k_waves(b.d[rows_key], FT_CFG["AUD"]["sr"])]
    return _WAVE_CACHE[key]


def _img_preprocess():
    """The evaluation transform of open_clip used by the H3 image encoder (built once)."""
    if not _IMG_PRE:
        import open_clip
        cfg = C.ENCODERS["IMG"]
        _, _, pre = open_clip.create_model_and_transforms(cfg["arch"], pretrained=cfg["pretrained"],
                                                          force_quick_gelu=True)
        _IMG_PRE.append(pre)
    return _IMG_PRE[0]


def _fix_len(waves, L):
    out = np.zeros((len(waves), L), dtype=np.float32)
    for j, w in enumerate(waves):
        out[j, :min(L, len(w))] = np.asarray(w, dtype=np.float32)[:L]
    return out


def ft_inputs(b, seed, rho_A, level=2):
    """Training inputs for FT with the inputs of S_A corrupted exactly as in en_embeddings."""
    S_A = N.select_inputs(b.n, rho_A, seed)
    if b.ds == "AUD":
        X = list(_waves32(b, "train_rows"))
    else:
        X = list(b.d["X_train"]) if b.ds == "TXT" else np.array(b.d["X_train"], copy=True)
    if rho_A <= 0:
        return X, S_A
    rng = N.corruption_rng(seed, rho_A, level)
    lv = C.LEVELS[b.ds][level]
    if b.ds == "IMG":
        Xc, _ = N.corrupt_images(b.d["X_train"][S_A], np.arange(len(S_A)), lv, rng)
        X[S_A] = Xc
    elif b.ds == "TXT":
        Xc = N.corrupt_texts([b.d["X_train"][i] for i in S_A], np.arange(len(S_A)), lv, rng)
        for j, i in enumerate(S_A):
            X[i] = Xc[j]
    elif b.ds == "AUD":
        Xc, _ = N.corrupt_waves([X[i] for i in S_A], np.arange(len(S_A)), lv, rng)
        for j, i in enumerate(S_A):
            X[i] = Xc[j]
    return X, S_A


def ft_task(b, X, device):
    K, ds = b.K, b.ds
    if ds == "IMG":
        from PIL import Image
        pre = _img_preprocess()
        S = {"train": X, "test": b.d["X_test"]}

        def get(split, idx, aug, g):
            ims = S[split][idx.cpu().numpy()]
            return torch.stack([pre(Image.fromarray(im)) for im in ims]).to(device)
        return dict(n=len(X), model_fn=lambda: _ClipFT(K), get=get, fwd=lambda m, x: m(x),
                    n_test=len(S["test"]), pred_bs=256)
    if ds == "TXT":
        from transformers import AutoTokenizer
        cfg = C.ENCODERS["TXT"]
        tok = AutoTokenizer.from_pretrained(cfg["name"])
        L = FT_CFG["TXT"]["max_len"]

        def enc(t):
            e = tok([cfg["prefix"] + s for s in t], truncation=True, max_length=L, padding="max_length",
                    return_tensors="pt")
            return e["input_ids"].to(device), e["attention_mask"].to(device)
        S = {"train": enc(X), "test": enc(b.d["X_test"])}
        return dict(n=len(X), model_fn=lambda: _E5FT(K),
                    get=lambda split, idx, aug, g: tuple(t[idx.to(device)] for t in S[split]),
                    fwd=lambda m, x: m(*x), n_test=len(S["test"][0]), pred_bs=256)
    if ds == "AUD":
        L = int(FT_CFG["AUD"]["sr"] * FT_CFG["AUD"]["clip_s"])
        S = {"train": torch.from_numpy(_fix_len(X, L)).half(),
             "test": torch.from_numpy(_fix_len(_waves32(b, "test_rows"), L)).half()}
        return dict(n=len(X), model_fn=lambda: _Cnn14FT(K),
                    get=lambda split, idx, aug, g: S[split][idx.cpu()].float().to(device),
                    fwd=lambda m, x: m(x), n_test=len(S["test"]), pred_bs=64)
    raise ValueError(ds)


def run_ft(b, seed, conds, device):
    ds = b.ds
    for cond in conds:
        if done(ds, "FT", "CE", cond, seed):
            continue
        rho_L, rho_A, level = C.CONDITIONS[cond]
        y_noisy, S_L, _ = N.label_noise(b.y_train, b.K, rho_L, seed)
        X, S_A = ft_inputs(b, seed, rho_A, level)
        task = ft_task(b, X, device)
        cfg = {k: v for k, v in FT_CFG[ds].items() if k not in ("max_len", "sr", "clip_s")}
        r = TR.fit_predict(task, y_noisy, b.d["y_test"], cfg, seed, device)
        write_row(ds, "FT", "CE", cond, seed, **r, n_train=b.n, n_label_noisy=len(S_L),
                  n_input_noisy=len(S_A) if rho_A > 0 else 0,
                  n_overlap=len(np.intersect1d(S_L, S_A)) if rho_A > 0 else 0, git_commit=code_version(),
                  gpu=gpu_name(device), notes="FT:" + ";".join(f"{k}={v}" for k, v in FT_CFG[ds].items()))
        log(f"{ds} FT CE {cond} s{seed} acc={r['test_acc']:.4f} t={r['train_time_s']:.0f}s")
        del task
        torch.cuda.empty_cache() if device.startswith("cuda") else None


class _CachedEncoder:
    """Stands for the pre-trained encoder when all its embeddings are already cached (MH)."""

    def __init__(self, revision):
        self.revision = revision

    def __call__(self, *a, **k):
        raise RuntimeError("embedding cache incomplete")


def _pretrained_encoder(b, seeds, device):
    d = os.path.join(C.EMB, b.ds)
    need = [os.path.join(d, "clean.npz")] + [os.path.join(d, f"EN_r{N.r_of(0.4)}_l2_s{s}.npz") for s in seeds]
    if all(os.path.exists(p) for p in need):
        return _CachedEncoder(str(np.load(need[0])["revision"]))
    from .embed import get_encoder
    return get_encoder(b.ds, device)


def run_block(b, seeds, pls, conds, device):
    """H3C for one dataset: RN and MH share nothing with FT; encoders are built once per dataset."""
    if b.ds not in C.H3_DATASETS:
        return
    for pl in pls:
        methods = H3C_METHODS[pl]
        if pl in ("RN", "MH"):
            if all(done(b.ds, pl, m, c, s) for s in seeds for c in conds for m in methods):
                continue
            enc = random_encoder(b.ds, device) if pl == "RN" else _pretrained_encoder(b, seeds, device)
            if pl == "RN" and b.ds == "TXT":
                log(f"TXT RN check: mean cosine to the pre-trained embeddings "
                    f"{check_random_text_encoder(b.d['X_train'], device, enc):.3f}")
            for s in seeds:
                run_heads(b, s, pl, conds, methods, device, enc)
            del enc
            torch.cuda.empty_cache() if device.startswith("cuda") else None
        elif pl == "FT":
            for s in seeds:
                run_ft(b, s, conds, device)
