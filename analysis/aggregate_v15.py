"""Aggregation of the round 3 blocks: TE, CV (AUDF1-AUDF10), RN of text, H4X and DR.

Usage: python analysis/aggregate_v15.py --v15 <runs_v15 dir> [<runs_v15 dir> ...] --grid <runs dir>
       --e12 <runs_e12 dir> --out <dir>
Every statistic pairs runs by seed (or fold) on one GPU; references (CE of the grid, of E1/E2 or repeated in
runs_v15) are taken from the run of the same dataset, condition, seed and GPU.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats


def load(roots):
    fs = [f for r in roots for f in glob.glob(os.path.join(r, "*", "*", "*", "*.csv"))]
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    if len(df):
        df["card"] = np.where(df.gpu.astype(str).str.contains("5090"), "5090",
                              np.where(df.gpu.astype(str).str.contains("4090"), "4090", "cpu"))
    return df


def ci(x, test_train=None):
    """t interval over seeds or folds. With test_train = n_test/n_train (k-fold cross-validation), the variance
    factor 1/k becomes 1/k + n_test/n_train (corrected resampled t test, Nadeau and Bengio 2003), because the
    training sets of the folds overlap; the uncorrected interval is kept in lo_naive/hi_naive."""
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    n = len(x)
    if n < 2:
        return dict(mean=float(x.mean()) if n else np.nan, sd=np.nan, lo=np.nan, hi=np.nan, n=n, conclusion="n/a")
    m, s = x.mean(), x.std(ddof=1)
    t = stats.t.ppf(0.975, n - 1)
    h = t * s / np.sqrt(n)
    out = {}
    if test_train is not None:
        out = dict(lo_naive=m - h, hi_naive=m + h)
        h = t * s * np.sqrt(1 / n + test_train)
    c = "positive" if m - h > 0 else ("negative" if m + h < 0 else "not distinguishable")
    return dict(mean=m, sd=s, lo=m - h, hi=m + h, n=n, conclusion=c, **out)


class Refs:
    def __init__(self, *frames):
        self.df = pd.concat([f for f in frames if len(f)], ignore_index=True)

    def row(self, ds, pl, m, cond, seed, card):
        d = self.df
        r = d[(d.dataset == ds) & (d.pipeline == pl) & (d.method == m) & (d.condition == cond) & (d.seed == seed)
              & (d.card == card)]
        return r.iloc[0] if len(r) else None

    def acc(self, ds, pl, m, cond, seed, card):
        r = self.row(ds, pl, m, cond, seed, card)
        return np.nan if r is None else float(r.test_acc)


def rla(a0, a):
    return (a0 - a) / a0


def te(v15, ref):
    out = []
    t = v15[v15.pipeline == "TE"] if len(v15) else v15
    if not len(t):
        return pd.DataFrame()
    for ds in sorted(t.dataset.unique()):
        for rho in ("20", "40"):
            G, GJ, D, errc, errg, acces = [], [], [], [], [], []
            for s in sorted(t[t.dataset == ds].seed.unique()):
                rL = t[(t.dataset == ds) & (t.method == "FCES") & (t.condition == f"L{rho}") & (t.seed == s)]
                rJ = t[(t.dataset == ds) & (t.method == "FCES") & (t.condition == f"J{rho}") & (t.seed == s)]
                if not len(rL) or not len(rJ):
                    continue
                card = rL.card.iloc[0]
                gl = rL.test_acc.iloc[0] - ref.acc(ds, "N", "CE", f"L{rho}", s, card)
                gj = rJ.test_acc.iloc[0] - ref.acc(ds, "N", "CE", f"J{rho}", s, card)
                G.append(gl)
                GJ.append(gj)
                D.append(gj - gl)
                ec, eg = [], []            # one value per seed (mean over L and J), not two pseudo-replicates
                for cond in (f"L{rho}", f"J{rho}"):
                    fr = ref.row(ds, "N", "FC", cond, s, card)
                    er = t[(t.dataset == ds) & (t.method == "FCES") & (t.condition == cond) & (t.seed == s)]
                    if fr is not None and len(er):
                        ec.append(float(er.T_hat_frob_err.iloc[0]) - float(fr.T_hat_frob_err))
                        eg.append(float(er.T_hat_frob_err.iloc[0]))
                if ec:
                    errc.append(float(np.mean(ec)))
                    errg.append(float(np.mean(eg)))
                ce = t[(t.dataset == ds) & (t.method == "CEES") & (t.condition == f"L{rho}") & (t.seed == s)]
                if len(ce):
                    acces.append(float(ce.test_acc_es.iloc[0]))
            for name, x in ((f"G_ES(L{rho})", G), (f"G_ES(J{rho})", GJ), (f"Delta_ES_{rho}", D),
                            (f"errT_ES(L,J{rho})", errg), (f"errT_ES-errT_grid({rho})", errc),
                            (f"acc_CEES(L{rho})", acces)):
                out.append(dict(block="TE", dataset=ds, statistic=name, **ci(x)))
    o = pd.DataFrame(out)
    for ds in o.dataset.unique():                      # H2 verdict with the precondition G_ES(L) > 0
        for rho in ("20", "40"):
            g = o[(o.dataset == ds) & (o.statistic == f"G_ES(L{rho})")]
            i = o[(o.dataset == ds) & (o.statistic == f"Delta_ES_{rho}")].index
            if len(g) and g.conclusion.iloc[0] != "positive":
                o.loc[i, "conclusion"] = "not evaluable"
    return o


def cv(v15):
    """AUDF1-AUDF10: statistics of the grid and of H4 over the ten folds (one seed each)."""
    f = v15[v15.dataset.astype(str).str.startswith("AUDF")] if len(v15) else v15
    if not len(f):
        return pd.DataFrame()

    def acc(ds, m, cond):
        r = f[(f.dataset == ds) & (f.pipeline == "N") & (f.method == m) & (f.condition == cond)]
        return float(r.test_acc.iloc[0]) if len(r) else np.nan

    folds = sorted(f.dataset.unique(), key=lambda s: int(s[4:]))
    S = {k: [] for k in ["acc_c0", "D_0.2", "D_0.4", "I_0.2", "I_0.4", "G(L20)", "G(L40)", "Delta_0.2", "Delta_0.4",
                         "G*(L20)", "G*(L40)", "Delta*_0.2", "Delta*_0.4", "D_l1", "D_l3", "RLA(L40)", "RLA(A40)",
                         "E", "E_class", "E_instance", "G*(C)", "Delta_F", "r_F"]}
    for ds in folds:
        a0 = acc(ds, "CE", "c0")
        S["acc_c0"].append(a0)
        for rho in ("20", "40"):
            L, A, J = (rla(a0, acc(ds, "CE", c)) for c in (f"L{rho}", f"A{rho}", f"J{rho}"))
            S[f"D_0.{rho[0]}"].append(L - A)
            S[f"I_0.{rho[0]}"].append(J - L - A)
            gl = acc(ds, "FC", f"L{rho}") - acc(ds, "CE", f"L{rho}")
            gj = acc(ds, "FC", f"J{rho}") - acc(ds, "CE", f"J{rho}")
            S[f"G(L{rho})"].append(gl)
            S[f"Delta_0.{rho[0]}"].append(gj - gl)
            gsl = acc(ds, "FCstar", f"L{rho}") - acc(ds, "CE", f"L{rho}")
            gsj = acc(ds, "FCstar", f"J{rho}") - acc(ds, "CE", f"J{rho}")
            S[f"G*(L{rho})"].append(gsl)
            S[f"Delta*_0.{rho[0]}"].append(gsj - gsl)
        L40 = rla(a0, acc(ds, "CE", "L40"))
        S["RLA(L40)"].append(L40)
        S["RLA(A40)"].append(rla(a0, acc(ds, "CE", "A40")))
        for lv in ("1", "3"):
            S[f"D_l{lv}"].append(L40 - rla(a0, acc(ds, "CE", f"A40-l{lv}")))
        rF, rS, rC = (rla(a0, acc(ds, "CE", c)) for c in ("F", "S", "C"))
        S["E"].append(rF - rS)
        S["E_class"].append(rC - rS)
        S["E_instance"].append(rF - rC)
        gc = acc(ds, "FCstar", "C") - acc(ds, "CE", "C")
        gf = acc(ds, "FCstar", "F") - acc(ds, "CE", "F")
        S["G*(C)"].append(gc)
        S["Delta_F"].append(gf - gc)
        rr = f[(f.dataset == ds) & (f.condition == "F")]
        S["r_F"].append(float(rr.r_F.iloc[0]) if len(rr) else np.nan)
    # folds of UrbanSound8K: test about 1/10 and training about 9/10 of the clips, so n_test/n_train is about 1/9
    o = pd.DataFrame([dict(block="CV", dataset="AUD (10 folds)", statistic=k, **ci(v, test_train=1 / 9))
                      for k, v in S.items()])
    for pre, st in (("G(L20)", "Delta_0.2"), ("G(L40)", "Delta_0.4"), ("G*(L20)", "Delta*_0.2"),
                    ("G*(L40)", "Delta*_0.4"), ("G*(C)", "Delta_F")):
        if o[o.statistic == pre].conclusion.iloc[0] != "positive":
            o.loc[o.statistic == st, "conclusion"] = "not evaluable"
    return o


def rn_txt(v15):
    r = v15[(v15.dataset == "TXT") & (v15.pipeline == "RN")] if len(v15) else v15
    if not len(r):
        return pd.DataFrame()

    def acc(m, c, s):
        x = r[(r.method == m) & (r.condition == c) & (r.seed == s)]
        return float(x.test_acc.iloc[0]) if len(x) else np.nan

    S = {"acc(c0)": [], "RLA(L40)": [], "D_0.4": [], "G(L40)": [], "Delta_0.4": []}
    for s in sorted(r.seed.unique()):
        a0 = acc("CE", "c0", s)
        L, A = rla(a0, acc("CE", "L40", s)), rla(a0, acc("CE", "A40", s))
        S["acc(c0)"].append(a0)
        S["RLA(L40)"].append(L)
        S["D_0.4"].append(L - A)
        gl = acc("FC", "L40", s) - acc("CE", "L40", s)
        S["G(L40)"].append(gl)
        S["Delta_0.4"].append(acc("FC", "J40", s) - acc("CE", "J40", s) - gl)
    o = pd.DataFrame([dict(block="RN-TXT", dataset="TXT", statistic=k, **ci(v)) for k, v in S.items()])
    if o[o.statistic == "G(L40)"].conclusion.iloc[0] != "positive":
        o.loc[o.statistic == "Delta_0.4", "conclusion"] = "not evaluable"
    return o


def h4x(v15, ref):
    out = []
    if not len(v15):
        return pd.DataFrame()
    h = v15[(v15.pipeline == "N") & v15.condition.isin(["S1", "C1", "F1", "S2", "C2", "F2"])]
    for ds in sorted(h.dataset.unique()):
        for a in ("1", "2"):
            S = {"E": [], "E_class": [], "E_instance": [], "G*(C)": [], "Delta_F": [], "r_F": []}
            rev = ""
            for s in sorted(h[h.dataset == ds].seed.unique()):
                def acc(m, c):
                    x = h[(h.dataset == ds) & (h.method == m) & (h.condition == c + a) & (h.seed == s)]
                    return (float(x.test_acc.iloc[0]), x.card.iloc[0]) if len(x) else (np.nan, None)
                (aF, card), (aS, _), (aC, _) = acc("CE", "F"), acc("CE", "S"), acc("CE", "C")
                if card is None:
                    continue
                a0 = ref.acc(ds, "N", "CE", "c0", s, card)
                rF, rS, rC = rla(a0, aF), rla(a0, aS), rla(a0, aC)
                S["E"].append(rF - rS)
                S["E_class"].append(rC - rS)
                S["E_instance"].append(rF - rC)
                gc = acc("FCstar", "C")[0] - aC
                gf = acc("FCstar", "F")[0] - aF
                S["G*(C)"].append(gc)
                S["Delta_F"].append(gf - gc)
                x = h[(h.dataset == ds) & (h.condition == "F" + a) & (h.seed == s)]
                S["r_F"].append(float(x.r_F.iloc[0]))
                rev = str(x.annotator.iloc[0])
            if not S["E"]:
                continue
            o = pd.DataFrame([dict(block="H4X", dataset=ds, annotator=a, revision=rev, statistic=k, **ci(v))
                              for k, v in S.items()])
            if o[o.statistic == "G*(C)"].conclusion.iloc[0] != "positive":
                o.loc[o.statistic == "Delta_F", "conclusion"] = "not evaluable"
            out.append(o)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def dr(v15, ref):
    """RLA(A40-l) for l = 4-6 (round 3) with l = 1-3 (grid) and l = 7 (E1), D_l = RLA(L40) - RLA(A40-l), RLA(L40)."""
    out = []
    if not len(v15):
        return pd.DataFrame()
    d = v15[(v15.pipeline == "N") & v15.condition.isin(["A40-l4", "A40-l5", "A40-l6"])]
    for ds in sorted(d.dataset.unique()):
        R = {lv: [] for lv in range(1, 8)}
        Dl = {lv: [] for lv in range(1, 8)}
        RL = []
        for s in sorted(d[d.dataset == ds].seed.unique()):
            card = d[(d.dataset == ds) & (d.seed == s)].card.iloc[0]
            other = "4090" if card == "5090" else "5090"
            a0c = ref.acc(ds, "N", "CE", "c0", s, card)
            RL.append(rla(a0c, ref.acc(ds, "N", "CE", "L40", s, card)))
            for lv in range(1, 8):
                cond = "A40" if lv == 2 else f"A40-l{lv}"
                if lv in (4, 5, 6):
                    x = d[(d.dataset == ds) & (d.condition == cond) & (d.seed == s)]
                    a, c = (float(x.test_acc.iloc[0]) if len(x) else np.nan), card
                else:                                   # same card if available, else the card of the grid
                    c = card if np.isfinite(ref.acc(ds, "N", "CE", cond, s, card)) else other
                    a = ref.acc(ds, "N", "CE", cond, s, c)
                a0 = ref.acc(ds, "N", "CE", "c0", s, c)
                L = rla(a0, ref.acc(ds, "N", "CE", "L40", s, c))
                R[lv].append(rla(a0, a))
                Dl[lv].append(L - rla(a0, a))
        out.append(dict(block="DR", dataset=ds, statistic="RLA(L40)", **ci(RL)))
        for lv in range(1, 8):
            out.append(dict(block="DR", dataset=ds, statistic=f"RLA(A40-l{lv})", **ci(R[lv])))
            out.append(dict(block="DR", dataset=ds, statistic=f"D_l{lv}", **ci(Dl[lv])))
    return pd.DataFrame(out)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--v15", nargs="+", required=True)
    a.add_argument("--grid", required=True)
    a.add_argument("--e12", required=True)
    a.add_argument("--out", required=True)
    o = a.parse_args()
    v15 = load(o.v15)
    ref = Refs(load([o.grid]), load([o.e12]), v15)      # v15: CE c0/L40 of IMG on the 4090
    os.makedirs(o.out, exist_ok=True)
    parts = {"te": te(v15, ref), "cv": cv(v15), "rn_txt": rn_txt(v15), "h4x": h4x(v15, ref), "dr": dr(v15, ref)}
    for k, df in parts.items():
        df.to_csv(os.path.join(o.out, f"v15_{k}.csv"), index=False)
        print(f"== {k}: {len(df)} rows")
        if len(df):
            print(df.drop(columns=[c for c in ("sd", "revision") if c in df]).to_string(
                index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
