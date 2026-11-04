"""Aggregation of the round 5 block: forward correction with T_hat from out-of-sample predictions,
and of its round 5 extension (the design, block OOSC): the same predictions calibrated by temperature.

Usage: python analysis/aggregate_v17.py --v17 <runs_v17 dir> [<runs_v17 dir> ...] --grid <runs dir> --out <dir>
       [--v14 <v14_stats csv> ...]
Every primary statistic pairs CE and FC-OOS runs of the same dataset, condition and seed, run on one card in the
block. G(u) of the grid (second datasets: round 2) is reported only descriptively, as its runs may come from
another card.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from aggregate_v15 import ci, load  # noqa: E402

RHOS = ("20", "40")
DIAG = ("errT_IS_global", "errT_IS_class", "errT_OOS_global", "errT_OOS_class", "errT_OOS_argmax")
DIAG_CAL = ("errT_cal_class", "errT_cal_global", "errT_cal_argmax", "errT_identity", "diag_OOS", "diag_cal")


def _card(f, cond, seed):
    """Card of the FC run of a cell; the CE runs of the cell are taken from the same card."""
    r = f[(f.method == "FC") & (f.condition == cond) & (f.seed == seed)]
    return r.card.iloc[0] if len(r) else None


def _acc(f, m, cond, seed):
    card = _card(f, cond, seed)
    r = f[(f.method == m) & (f.condition == cond) & (f.seed == seed) & (f.card == card)]
    return float(r.test_acc.iloc[0]) if len(r) else np.nan


def _diag(f, cond, seed, method="FC", keys=DIAG):
    r = f[(f.method == method) & (f.condition == cond) & (f.seed == seed)]
    if not len(r):
        return {k: np.nan for k in keys}
    kv = dict(x.split("=") for x in str(r.notes.iloc[0]).split(";") if "=" in x)
    return {k: float(kv.get(k, np.nan)) for k in keys}


def _cal(f, seeds):
    """round 5: FC with T_hat from temperature-calibrated out-of-sample predictions (method FCcal), paired
    with the CE and FC runs of the same cell and card; only seeds whose four FCcal runs exist."""
    S = {}
    for s in seeds:
        if any(not len(f[(f.method == "FCcal") & (f.condition == f"{u}{r}") & (f.seed == s)])
               for r in RHOS for u in ("L", "J")):
            continue
        vals = {}
        for r in RHOS:
            for u in (f"L{r}", f"J{r}"):
                a_cal = _acc(f, "FCcal", u, s)
                vals[f"G_cal({u})"] = a_cal - _acc(f, "CE", u, s)
                vals[f"G_cal-G_OOS({u})"] = a_cal - _acc(f, "FC", u, s)
                d = _diag(f, u, s, "FCcal", DIAG_CAL + ("beta0", "beta1"))
                for k in DIAG_CAL:
                    vals[f"{k}({u})"] = d[k]
                vals[f"beta({u})"] = (d["beta0"] + d["beta1"]) / 2
                vals[f"errT_cal-errT_OOS({u})"] = d["errT_cal_class"] - _diag(f, u, s)["errT_OOS_class"]
            vals[f"Delta_cal_{r}"] = vals[f"G_cal(J{r})"] - vals[f"G_cal(L{r})"]
        for k, v in vals.items():
            S.setdefault(k, []).append(v)
    return S


def _grid_G(grid, ds, cond):
    """Mean G(u) of the grid (pipeline N, in-sample anchors), descriptive."""
    if grid is None or not len(grid):
        return np.nan
    g = grid[(grid.dataset == ds) & (grid.pipeline == "N") & (grid.condition == cond)]
    out = []
    for s in sorted(set(g.seed)):
        fc = g[(g.method == "FC") & (g.seed == s)].test_acc
        ce = g[(g.method == "CE") & (g.seed == s)].test_acc
        if len(fc) and len(ce):
            out.append(float(fc.iloc[0]) - float(ce.iloc[0]))
    return float(np.mean(out)) if out else np.nan


def oos(v17, grid, v14):
    rows = []
    for ds in sorted(set(v17.dataset)):
        f = v17[(v17.dataset == ds) & (v17.pipeline == "OOS")]
        S = {}
        seeds = sorted(set(f[f.method == "FC"].seed))
        if not seeds:                                    # no completed cell yet
            continue
        for s in seeds:
            vals = {}
            for r in RHOS:
                for u in (f"L{r}", f"J{r}"):
                    vals[f"G_OOS({u})"] = _acc(f, "FC", u, s) - _acc(f, "CE", u, s)
                    d = _diag(f, u, s)
                    for k in DIAG:
                        vals[f"{k}({u})"] = d[k]
                    vals[f"errT_OOS-errT_IS({u})"] = d["errT_OOS_class"] - d["errT_IS_global"]
                    vals[f"heldout_acc({u})"] = np.nanmean([_acc(f, f"CEhalf{h}", u, s) for h in (0, 1)])
                vals[f"Delta_OOS_{r}"] = vals[f"G_OOS(J{r})"] - vals[f"G_OOS(L{r})"]
            for k, v in vals.items():
                S.setdefault(k, []).append(v)
        o = pd.DataFrame([dict(block="OOS", dataset=ds, statistic=k, **ci(v)) for k, v in S.items()])
        for r in RHOS:                                   # precondition of H2: G_OOS(L) positive
            ok = (o[o.statistic == f"G_OOS(L{r})"].conclusion == "positive").any()
            if not ok:
                o.loc[o.statistic == f"Delta_OOS_{r}", "conclusion"] = "not evaluable"
        Sc = _cal(f, seeds)
        if Sc:
            oc = pd.DataFrame([dict(block="OOSC", dataset=ds, statistic=k, **ci(v)) for k, v in Sc.items()])
            for r in RHOS:                               # same precondition for Delta_cal
                if not (oc[oc.statistic == f"G_cal(L{r})"].conclusion == "positive").any():
                    oc.loc[oc.statistic == f"Delta_cal_{r}", "conclusion"] = "not evaluable"
            o = pd.concat([o, oc], ignore_index=True)
        # descriptive: G of the grid / round 2 second datasets with in-sample anchors
        desc = []
        for r in RHOS:
            for u in (f"L{r}", f"J{r}"):
                gm = _grid_G(grid, ds, u)
                if np.isnan(gm) and v14 is not None and len(v14):
                    st = f"G(L)_0.{r[0]}" if u.startswith("L") else None
                    h = v14[(v14.dataset == ds) & (v14.statistic == st)] if st else []
                    gm = float(h["mean"].iloc[0]) if len(h) else np.nan
                desc.append(dict(block="OOS", dataset=ds, statistic=f"G_grid({u})", mean=gm, sd=np.nan, lo=np.nan,
                                 hi=np.nan, n=np.nan, conclusion="descriptive"))
        o = pd.concat([o, pd.DataFrame(desc)], ignore_index=True)
        o["card"] = ",".join(sorted(set(f.card)))
        rows.append(o)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--v17", nargs="+", required=True)
    a.add_argument("--grid", required=True)
    a.add_argument("--v14", nargs="*", default=[])
    a.add_argument("--out", required=True)
    o = a.parse_args()
    v17 = load(o.v17)
    grid = load([o.grid])
    v14 = pd.concat([pd.read_csv(p) for p in o.v14], ignore_index=True) if o.v14 else None
    os.makedirs(o.out, exist_ok=True)
    df = oos(v17, grid, v14)
    df.to_csv(os.path.join(o.out, "v17_oos.csv"), index=False)
    print(f"== oos: {len(df)} rows")
    if len(df):
        keep = df.statistic.str.match(r"G_OOS|Delta_OOS|errT_OOS-errT_IS|G_grid|G_cal|Delta_cal|errT_cal-errT_OOS|beta")
        print(df[keep][["dataset", "statistic", "mean", "lo", "hi", "n", "conclusion"]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
