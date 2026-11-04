"""Pre-run and post-pilot checks. Writes a report to <RUNS>/checks/report.md and sample files.

  python checks.py --ds TAB,TS --stage pre      # generators, sets, samples and SNR (before any run)
  python checks.py --ds TAB,TS --stage post     # accuracy ranges and FC* after the pilot
  python checks.py --ds IMG --stage h4          # annotator determinism, T_F, S and C
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from rnd import config as C
from rnd import data as D
from rnd import experiment as X
from rnd import noise as N

MIN_ACC = {"TAB": 0.90, "IMG": 0.90, "TXT": 0.88, "TS": 0.85, "GRA": 0.80, "AUD": 0.60}


def out_dir():
    p = os.path.join(C.RUNS, "checks")
    os.makedirs(p, exist_ok=True)
    return p


def report(lines, ds, stage):
    with open(os.path.join(out_dir(), "report.md"), "a", encoding="utf-8") as f:
        f.write(f"\n## {ds} — {stage}\n\n" + "\n".join(f"- {l}" for l in lines) + "\n")
    print("\n".join(lines))


def snr_db(clean, noisy):
    clean, noise = np.asarray(clean, float), np.asarray(noisy, float) - np.asarray(clean, float)
    return 10 * np.log10(np.mean(clean ** 2) / np.mean(noise ** 2)) if np.mean(noise ** 2) > 0 else np.inf


def pre(ds):
    b = X.Bundle(ds)
    L = []
    ok = True
    for rho in (0.2, 0.4):
        for s in C.SEEDS:
            yn, S_L, _ = N.label_noise(b.y_train, b.K, rho, s)
            changed = int(np.sum(yn != b.y_train))
            if changed != int(round(rho * b.n)) or np.any(yn[S_L] == b.y_train[S_L]):
                ok = False
            S1 = N.select_inputs(b.n, rho, s)
            S2 = N.select_inputs(b.n, rho, s)
            if not np.array_equal(S1, S2):
                ok = False
    L.append(f"label flips exactly round(rho n) and none keeps its class; S_A reproducible: {'OK' if ok else 'FAIL'}")
    # samples and measured SNR at each level
    S_A = N.select_inputs(b.n, 0.4, 0)
    for lv in (1, 2, 3):
        X_noisy, info = b.noisy_inputs(0.4, lv, 0)
        idx = S_A[:5]
        if ds in ("TS", "AUD"):
            Xc = b.d["X_train"]
            if ds == "TS":
                snrs = [snr_db(Xc[i, c] - Xc[i, c].mean(), X_noisy[i, c] - Xc[i, c].mean())
                        for i in S_A[:200] for c in range(Xc.shape[1])]
            else:
                snrs = [snr_db(Xc[i], X_noisy[i]) for i in S_A[:200] if np.mean(np.asarray(Xc[i]) ** 2) > 0]
            nominal = C.LEVELS[ds][lv]
            m = float(np.mean(snrs))
            L.append(f"level {lv}: mean measured SNR {m:.2f} dB vs nominal {nominal} dB "
                     f"({'OK' if abs(m - nominal) <= 0.5 else 'CHECK'})")
            if ds == "AUD":
                import soundfile as sf
                for i in idx:
                    sf.write(os.path.join(out_dir(), f"AUD_l{lv}_{i}.wav"), np.asarray(X_noisy[i]), 16000)
        elif ds == "IMG":
            from PIL import Image
            grid = np.concatenate([np.concatenate([b.d["X_train"][i], X_noisy[i]], 0) for i in idx], 1)
            Image.fromarray(grid).resize((grid.shape[1] * 4, grid.shape[0] * 4), Image.NEAREST).save(
                os.path.join(out_dir(), f"IMG_l{lv}.png"))
            L.append(f"level {lv}: types of the first 5 corrupted images {list(info['types'][:5])} (0 gauss, 1 blur, 2 jpeg)")
        elif ds == "TXT":
            with open(os.path.join(out_dir(), f"TXT_l{lv}.txt"), "w", encoding="utf-8") as f:
                for i in idx:
                    f.write(f"[{i}] {b.d['X_train'][i]}\n  -> {X_noisy[i]}\n\n")
            L.append(f"level {lv}: sample texts written to checks/TXT_l{lv}.txt")
        elif ds == "TAB":
            d = (X_noisy[S_A] - b.d["X_train"][S_A]) / b.sd_clean
            L.append(f"level {lv}: s.d. of the added noise in clean-s.d. units {d.std():.3f} (nominal {C.LEVELS[ds][lv]})")
        elif ds == "GRA":
            nodes = b.d["train_idx"][S_A]
            frac = float(np.mean(X_noisy[nodes] != b.d["X"][nodes]))
            L.append(f"level {lv}: fraction of flipped features {frac:.4f} (nominal {C.LEVELS[ds][lv]})")
    report(L, ds, "pre")


def post(datasets):
    files = glob.glob(os.path.join(C.RUNS, "*", "N", "*", "*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files]) if files else pd.DataFrame()
    L = []
    better = []
    for ds in datasets:
        a = df[(df.dataset == ds) & (df.method == "CE") & (df.condition == "c0")].test_acc
        if len(a):
            L.append(f"{ds}: CE accuracy on c0 {a.mean():.4f} (threshold {MIN_ACC[ds]}) "
                     f"{'OK' if a.mean() > MIN_ACC[ds] else 'REVIEW THE IMPLEMENTATION'}")
        fs = df[(df.dataset == ds) & (df.method == "FCstar") & (df.condition == "L40")].set_index("seed").test_acc
        ce = df[(df.dataset == ds) & (df.method == "CE") & (df.condition == "L40")].set_index("seed").test_acc
        if len(fs) and len(ce):
            g = float((fs - ce).mean())
            better.append(g > 0)
            L.append(f"{ds}: FC* - CE on L40 = {100 * g:.2f} points")
    if better:
        L.append(f"FC* beats CE on L40 in at least one dataset: {'OK' if any(better) else 'REVIEW THE CORRECTED LOSS'}")
    report(L, ",".join(datasets), "post")


def h4(ds, device, n_check=500):
    from rnd import annotate as A
    b = X.Bundle(ds)
    z = np.load(os.path.join(C.NOISE, f"{ds}_F.npz"), allow_pickle=True)
    yhat, r_F, T_F = z["yhat"], float(z["r_F"]), z["T_F"]
    L = [f"r_F = {r_F:.4f}" + (" (below 0.05: H4 uninformative here)" if r_F < 0.05 else "")]
    L.append(f"rows of T_F sum to 1: {'OK' if np.allclose(T_F.sum(1), 1) else 'FAIL'}")
    X500 = D.us8k_waves(b.d["train_rows"][:n_check], C.ANNOTATORS["AUD"]["sr"]) if ds == "AUD" else b.d["X_train"][:n_check]
    fn = {"IMG": A.annotate_img, "TXT": A.annotate_txt, "AUD": A.annotate_aud}[ds]
    y1 = fn(X500, device)[0]
    y2 = fn(X500, device)[0]
    L.append(f"determinism on {n_check} examples: repeat identical {np.array_equal(y1, y2)}, "
             f"identical to stored labels {np.array_equal(y1, yhat[:n_check])}")
    nS, rates = [], []
    for s in C.SEEDS:
        yS, _ = N.fm_condition("S", b.y_train, yhat, b.K, s, r_F, T_F)
        yC, _ = N.fm_condition("C", b.y_train, yhat, b.K, s, r_F, T_F)
        nS.append(int(np.sum(yS != b.y_train)))
        rates.append(float(np.mean(yC != b.y_train)))
    L.append(f"S changes round(r_F n) = {int(round(r_F * b.n))} labels in every seed: {'OK' if set(nS) == {int(round(r_F * b.n))} else nS}")
    L.append(f"C: mean realised rate {np.mean(rates):.4f} vs r_F {r_F:.4f} "
             f"({'OK' if abs(np.mean(rates) - r_F) <= 0.01 else 'CHECK'})")
    report(L, ds, "h4")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ds", required=True)
    p.add_argument("--stage", choices=["pre", "post", "h4"], required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    dss = a.ds.split(",")
    if a.stage == "pre":
        for ds in dss:
            pre(ds)
    elif a.stage == "post":
        post(dss)
    else:
        for ds in dss:
            h4(ds, a.device)
