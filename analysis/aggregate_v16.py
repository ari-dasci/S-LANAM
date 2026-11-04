"""Aggregation of the round 4 blocks: H2B (pair and input-dependent label noise) and SC
(standardisation with clean-set statistics).

Usage: python analysis/aggregate_v16.py --v16 <runs_v16 dir> [<runs_v16 dir> ...] --grid <runs dir> --out <dir>
Every statistic pairs runs by seed on one card; c0, L, A and J references come from the grid run of the same
dataset, condition, seed and card.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from aggregate_v15 import Refs, ci, load, rla  # noqa: E402

RHOS = ("20", "40")


def _acc(df, ds, pl, m, cond, seed, card):
    r = df[(df.dataset == ds) & (df.pipeline == pl) & (df.method == m) & (df.condition == cond) & (df.seed == seed)
           & (df.card == card)]
    return float(r.test_acc.iloc[0]) if len(r) else np.nan


def _rows(block, ds, S, pre):
    o = pd.DataFrame([dict(block=block, dataset=ds, statistic=k, **ci(v)) for k, v in S.items()])
    for need, st in pre:                      # precondition: the gain under the reference condition is positive
        if st in set(o.statistic):
            ok = (o[o.statistic == need].conclusion == "positive").any()
            if not ok:
                o.loc[o.statistic == st, "conclusion"] = "not evaluable"
    return o


def h2b(v16, ref):
    out = []
    for ds in sorted(set(v16.dataset)):
        f = v16[(v16.dataset == ds) & (v16.pipeline == "N")]
        if not len(f):
            continue
        S = {}
        for seed in sorted(set(f.seed)):
            card = f[f.seed == seed].card.iloc[0]
            a = lambda m, c: _acc(f, ds, "N", m, c, seed, card)            # noqa: E731
            g = lambda m, c: ref.acc(ds, "N", m, c, seed, card)             # noqa: E731
            c0 = g("CE", "c0")
            for r in RHOS:
                P, JP, JD = f"P{r}", f"JP{r}", f"JD{r}"
                GP, GsP = a("FC", P) - a("CE", P), a("FCstar", P) - a("CE", P)
                GJP, GsJP = a("FC", JP) - a("CE", JP), a("FCstar", JP) - a("CE", JP)
                GJD, GsJD = a("FC", JD) - a("CE", JD), a("FCstar", JD) - a("CE", JD)
                GL, GsL = g("FC", f"L{r}") - g("CE", f"L{r}"), g("FCstar", f"L{r}") - g("CE", f"L{r}")
                GsJ = g("FCstar", f"J{r}") - g("CE", f"J{r}")
                vals = {f"G(P{r})": GP, f"G*(P{r})": GsP, f"G(JP{r})": GJP, f"G*(JP{r})": GsJP,
                        f"Delta^P_{r}": GJP - GP, f"Delta*^P_{r}": GsJP - GsP,
                        f"G(L{r})": GL, f"G*(L{r})": GsL, f"G(JD{r})": GJD, f"G*(JD{r})": GsJD,
                        f"Delta^D_{r}": GJD - GL, f"Delta*^D_{r}": GsJD - GsL, f"G*(JD{r})-G*(J{r})": GsJD - GsJ,
                        f"RLA(P{r})": rla(c0, a("CE", P)), f"D^P_{r}": rla(c0, a("CE", P)) - rla(c0, g("CE", f"A{r}")),
                        f"RLA(JD{r})": rla(c0, a("CE", JD)), f"RLA(J{r})": rla(c0, g("CE", f"J{r}")),
                        # where the change of G*(JD) comes from (accuracy of CE and of FC* under JD vs L and J)
                        f"A_CE(JD{r})-A_CE(J{r})": a("CE", JD) - g("CE", f"J{r}"),
                        f"A_FC*(JD{r})-A_FC*(J{r})": a("FCstar", JD) - g("FCstar", f"J{r}"),
                        f"A_CE(JD{r})-A_CE(L{r})": a("CE", JD) - g("CE", f"L{r}"),
                        f"A_FC*(JD{r})-A_FC*(L{r})": a("FCstar", JD) - g("FCstar", f"L{r}")}
                for c in (P, JP, JD):
                    e = f[(f.method == "FC") & (f.condition == c) & (f.seed == seed)].T_hat_frob_err
                    vals[f"errT({c})"] = float(e.iloc[0]) if len(e) else np.nan
                for k, v in vals.items():
                    S.setdefault(k, []).append(v)
        pre = []
        for r in RHOS:
            pre += [(f"G(P{r})", f"Delta^P_{r}"), (f"G*(P{r})", f"Delta*^P_{r}"), (f"G(L{r})", f"Delta^D_{r}"),
                    (f"G*(L{r})", f"Delta*^D_{r}")]
        out.append(_rows("H2B", ds, S, pre))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def sc(v16, ref, hyp):
    out = []
    for ds in sorted(set(v16.dataset)):
        f = v16[(v16.dataset == ds) & (v16.pipeline == "SC")]
        if not len(f):
            continue
        S = {}
        for seed in sorted(set(f.seed)):
            card = f[f.seed == seed].card.iloc[0]
            a = lambda m, c: _acc(f, ds, "SC", m, c, seed, card)            # noqa: E731
            g = lambda m, c: ref.acc(ds, "N", m, c, seed, card)             # noqa: E731
            c0 = g("CE", "c0")
            vals = {}
            for r in RHOS:
                L = rla(c0, g("CE", f"L{r}"))
                vals[f"D^SC_0.{r[0]}"] = L - rla(c0, a("CE", f"A{r}"))
                vals[f"D_0.{r[0]} (noisy-set scaling)"] = L - rla(c0, g("CE", f"A{r}"))
                for c in (f"A{r}", f"J{r}"):
                    vals[f"RLA^SC({c})-RLA({c})"] = rla(c0, a("CE", c)) - rla(c0, g("CE", c))
                GsL = g("FCstar", f"L{r}") - g("CE", f"L{r}")
                vals[f"G*(L{r})"] = GsL
                vals[f"Delta*^SC_0.{r[0]}"] = (a("FCstar", f"J{r}") - a("CE", f"J{r}")) - GsL
                vals[f"Delta*_0.{r[0]} (noisy-set scaling)"] = (g("FCstar", f"J{r}") - g("CE", f"J{r}")) - GsL
            L40 = rla(c0, g("CE", "L40"))
            for lv in ("1", "3"):
                vals[f"D^SC_l{lv}"] = L40 - rla(c0, a("CE", f"A40-l{lv}"))
                vals[f"D_l{lv} (noisy-set scaling)"] = L40 - rla(c0, g("CE", f"A40-l{lv}"))
                vals[f"RLA^SC(A40-l{lv})-RLA(A40-l{lv})"] = rla(c0, a("CE", f"A40-l{lv}")) - rla(c0, g("CE", f"A40-l{lv}"))
            if ds == "TAB":
                vals["RLA^SC(AT40)"] = rla(c0, a("CE", "AT40"))
                vals["D^SC_AT"] = L40 - vals["RLA^SC(AT40)"]
            for k, v in vals.items():
                S.setdefault(k, []).append(v)
        o = _rows("SC", ds, S, [("G*(L20)", "Delta*^SC_0.2"), ("G*(L40)", "Delta*^SC_0.4")])
        # grid verdicts for comparison (hypotheses.csv of the grid)
        comp = {"D^SC_0.2": ("D", 0.2), "D^SC_0.4": ("D", 0.4), "D^SC_l1": ("D_l1", 0.4), "D^SC_l3": ("D_l3", 0.4),
                "Delta*^SC_0.2": ("Delta*", 0.2), "Delta*^SC_0.4": ("Delta*", 0.4)}
        gm, gc = [], []
        for st in o.statistic:
            if st in comp:
                h = hyp[(hyp.dataset == ds) & (hyp.pipeline == "N") & (hyp.statistic == comp[st][0])
                        & (hyp.rho == comp[st][1])]
                gm.append(float(h["mean"].iloc[0]) if len(h) else np.nan)
                gc.append(str(h["conclusion"].iloc[0]) if len(h) else "")
            else:
                gm.append(np.nan)
                gc.append("")
        o["grid_mean"], o["grid_conclusion"] = gm, gc
        out.append(o)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--v16", nargs="+", required=True)
    a.add_argument("--grid", required=True)
    a.add_argument("--out", required=True)
    o = a.parse_args()
    v16 = load(o.v16)
    grid = load([o.grid])
    ref = Refs(grid, v16)          # TAB and TS references repeated on the RTX 4090 (pipeline N)
    hyp = pd.read_csv(os.path.join(o.grid, "analysis", "hypotheses.csv"))
    os.makedirs(o.out, exist_ok=True)
    for k, df in {"h2b": h2b(v16, ref), "sc": sc(v16, ref, hyp)}.items():
        df.to_csv(os.path.join(o.out, f"v16_{k}.csv"), index=False)
        print(f"== {k}: {len(df)} rows")
        if len(df):
            cols = [c for c in df.columns if c not in ("sd", "block")]
            print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
