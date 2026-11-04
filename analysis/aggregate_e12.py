"""Exploratory experiments E1 and E2 (design in Appendix B of the online supplement).

E1, stronger input noise: H2 and the prediction of Iacovissi et al. when input noise is calibrated to harm about as
much as label noise. E2, noisy test inputs: H1 when a fraction of the test inputs is corrupted too.
Neither is part of the main grid. E1 compares with c0 and L40 on the same GPU: the E2 runs (CE) and, where
they exist, the L40 FC/FC* runs of the E1/E2 folder; otherwise the grid runs given with --grid.

  python analysis/aggregate_e12.py --pilot results/round1_e12/pilot --grid results/grid   # calibration (step 1)
  python analysis/aggregate_e12.py --runs results/round1_e12 --grid results/grid          # E1 and E2 (step 2)

Outputs in <pilot or runs>/analysis/:
  e1_pilot.csv   mean RLA of A40-l4..l7 (CE, pilot seeds) against RLA(L40) of the grid, and the level the rule picks
  e1.csv         per dataset: D_s, I_s, G(L40), G(J40-s), Delta_s, G*(L40), G*(J40-s), Delta*_s, G(A40-s), errT
  e2.csv         per dataset: noisy-test accuracies, RLA~, D~, I~, cost of the noisy test, reproduction check
  e12_verdicts.csv, emp_e12.tex
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aggregate import ci, conclusion, load_runs, with_precondition, SUPPORTED, NOT_SUPPORTED  # noqa: E402
from rnd import config as C  # noqa: E402

E1_LEVELS = (4, 5, 6, 7)


def series(df, ds, m, cond, name="test_acc"):
    s = df[(df.dataset == ds) & (df.pipeline == "N") & (df.method == m) & (df.condition == cond)]
    return s.set_index("seed")[name].astype(float).sort_index()


def base(e, g, ds, m, cond, name="test_acc"):
    """Reference runs (c0, L40) for E1: the run of the E1/E2 folder when it exists for that seed, which is on the
    same GPU as the E1 runs, and otherwise the grid run, which then is on that GPU too."""
    return series(e, ds, m, cond, name).combine_first(series(g, ds, m, cond, name))


def same_gpu(e, g, ds, m, cond):
    """Seeds whose E1/E2 run and grid run were trained on the same GPU."""
    a = e[(e.dataset == ds) & (e.pipeline == "N") & (e.method == m) & (e.condition == cond)].set_index("seed").gpu
    b = g[(g.dataset == ds) & (g.pipeline == "N") & (g.method == m) & (g.condition == cond)].set_index("seed").gpu
    j = a.to_frame("a").join(b.to_frame("b"), how="inner")
    return j.index[j.a == j.b]


def rla(a0, a):
    return ((a0 - a) / a0).dropna()


def pick_level(rla_by_level, target):
    """Calibration rule (round 1): the mildest level whose mean RLA(A40-lk) reaches the grid RLA(L40);
    level 7 if none does."""
    for lv in E1_LEVELS:
        if lv in rla_by_level and rla_by_level[lv] >= target:
            return lv, True
    return 7, False


def pilot(pilot_dir, grid_dir):
    p, g = load_runs(pilot_dir), load_runs(grid_dir)
    rows = []
    for ds in C.BASE:
        target = rla(series(g, ds, "CE", "c0"), series(g, ds, "CE", "L40")).mean()
        a0 = series(p, ds, "CE", "c0")
        r = dict(dataset=ds, rla_L40_grid=target, n_seeds=len(a0))
        by = {}
        for lv in E1_LEVELS:
            x = rla(a0, series(p, ds, "CE", f"A40-l{lv}"))
            if len(x):
                by[lv] = x.mean()
                r[f"rla_A40_l{lv}"] = by[lv]
        if by:
            lv, reached = pick_level(by, target)
            r.update(level=lv, reached=reached, rla_chosen=by.get(lv, np.nan))
        rows.append(r)
    out = os.path.join(pilot_dir, "analysis")
    os.makedirs(out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "e1_pilot.csv"), index=False)
    print(df.to_string(index=False))
    print("E1_LEVEL =", {r["dataset"]: int(r["level"]) for r in rows if "level" in r})
    return df


def e1_e2(runs, grid_dir):
    e, g = load_runs(runs), load_runs(grid_dir)
    out = os.path.join(runs, "analysis")
    os.makedirs(out, exist_ok=True)
    H, V = [], []

    def add(rows, ds, stat, x, **kw):
        c = ci(x)
        rows.append(dict(dataset=ds, statistic=stat, **c, conclusion=conclusion(c), **kw))
        return conclusion(c)

    # E1
    E1 = []
    for ds in C.BASE:
        lv = C.E1_LEVEL.get(ds)
        if lv is None:
            continue
        A, J = f"A40-l{lv}", f"J40-l{lv}"
        if not len(series(e, ds, "CE", A)):
            continue
        a0 = base(e, g, ds, "CE", "c0")
        rL, rA, rJ = rla(a0, base(e, g, ds, "CE", "L40")), rla(a0, series(e, ds, "CE", A)), rla(a0, series(e, ds, "CE", J))
        kw = dict(level=lv)
        add(E1, ds, "RLA(L40)", rL, **kw)
        add(E1, ds, "RLA(A40-s)", rA, **kw)
        add(E1, ds, "RLA(J40-s)", rJ, **kw)
        add(E1, ds, "D_s", rL - rA, **kw)
        add(E1, ds, "I_s", rJ - rL - rA, **kw)
        GL = base(e, g, ds, "FC", "L40") - base(e, g, ds, "CE", "L40")
        GJ = series(e, ds, "FC", J) - series(e, ds, "CE", J)
        GsL = base(e, g, ds, "FCstar", "L40") - base(e, g, ds, "CE", "L40")
        GsJ = series(e, ds, "FCstar", J) - series(e, ds, "CE", J)
        cGL = add(E1, ds, "G(L40)", GL, **kw)
        add(E1, ds, "G(J40-s)", GJ, **kw)
        cD = add(E1, ds, "Delta_s", GJ - GL, **kw)
        cGsL = add(E1, ds, "G*(L40)", GsL, **kw)
        add(E1, ds, "G*(J40-s)", GsJ, **kw)
        cDs = add(E1, ds, "Delta*_s", GsJ - GsL, **kw)
        add(E1, ds, "G(A40-s)", series(e, ds, "FC", A) - series(e, ds, "CE", A), **kw)
        add(E1, ds, "errT(J40-s)-errT(L40)", series(e, ds, "FC", J, "T_hat_frob_err") -
            base(e, g, ds, "FC", "L40", "T_hat_frob_err"), **kw)
        V.append(dict(dataset=ds, experiment="E1", level=lv,
                      calibrated="yes" if conclusion(ci(rL - rA)) != "positive" else "no",
                      H2_s=with_precondition(cGL, cD), H2star_s=with_precondition(cGsL, cDs)))
    pd.DataFrame(E1).to_csv(os.path.join(out, "e1.csv"), index=False)

    # E2
    E2 = []
    if "test_acc_noisy" in e:
        for ds in C.BASE:
            At = {c: series(e, ds, "CE", c, "test_acc_noisy").dropna() for c in ("c0", "L40", "A40", "J40")}
            if not len(At["c0"]):
                continue
            for c in At:
                add(E2, ds, f"Acc~({c})", At[c])
                # reproduction check: clean-test accuracy of the E2 run against the grid run (same seed, same GPU)
                sg = same_gpu(e, g, ds, "CE", c)
                diff = (series(e, ds, "CE", c) - series(g, ds, "CE", c)).reindex(sg).dropna()
                E2.append(dict(dataset=ds, statistic=f"max|Acc_e2-Acc_grid|({c})", mean=float(diff.abs().max())
                               if len(diff) else np.nan, n=len(diff), conclusion="n/a",
                               note=f"same GPU for {len(sg)} of {len(At[c])} seeds"))
            rt = {c: rla(At["c0"], At[c]) for c in ("L40", "A40", "J40")}
            for c in rt:
                add(E2, ds, f"RLA~({c})", rt[c])
            cDt = add(E2, ds, "D~", rt["L40"] - rt["A40"])
            add(E2, ds, "I~", rt["J40"] - rt["L40"] - rt["A40"])
            add(E2, ds, "Acc~(c0)-Acc(c0)", At["c0"] - series(e, ds, "CE", "c0"))
            V.append(dict(dataset=ds, experiment="E2", H1_noisy_test=SUPPORTED if cDt == "positive" else NOT_SUPPORTED))
    pd.DataFrame(E2).to_csv(os.path.join(out, "e2.csv"), index=False)
    vd = pd.DataFrame(V)
    vd.to_csv(os.path.join(out, "e12_verdicts.csv"), index=False)

    # numbers for the appendix table (layout finalised in the manuscript)
    allh = pd.DataFrame(E1 + E2)

    def cell(ds, st, scale=100):
        s = allh[(allh.dataset == ds) & (allh.statistic == st)] if len(allh) else allh
        if not len(s) or np.isnan(s["mean"].iloc[0]):
            return "--"
        mark = {"positive": "$^{+}$", "negative": "$^{-}$"}.get(s.conclusion.iloc[0], "")
        return f"{scale * s['mean'].iloc[0]:.1f}{mark}"

    cols = ["RLA(L40)", "RLA(A40-s)", "D_s", "G*(L40)", "Delta*_s", "Acc~(c0)-Acc(c0)", "RLA~(L40)", "RLA~(A40)", "D~"]
    lines = [r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
             r"Dataset & $s$ & $\mathrm{RLA}(L_{0.4})$ & $\mathrm{RLA}(A^{s}_{0.4})$ & $D^{s}$ & $G^{*}(L_{0.4})$ & "
             r"$\Delta^{*}_{s}$ & $\tilde A(c_0)-A(c_0)$ & $\widetilde{\mathrm{RLA}}(L_{0.4})$ & "
             r"$\widetilde{\mathrm{RLA}}(A_{0.4})$ & $\tilde D$ \\", r"\midrule"]
    for ds in C.BASE:
        lv = C.E1_LEVEL.get(ds)
        lines.append(" & ".join([ds, str(lv) if lv else "--"] + [cell(ds, c) for c in cols]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "emp_e12.tex"), "w").write("\n".join(lines) + "\n")
    print(vd.to_string(index=False))
    print(f"aggregate_e12: {len(e)} runs (+{len(g)} grid) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="runs folder of the five-seed grid (c0, L40, FC, FC* reused)")
    ap.add_argument("--pilot", help="runs folder of the E1 calibration pilot")
    ap.add_argument("--runs", help="runs folder of E1 and E2")
    a = ap.parse_args()
    if a.pilot:
        pilot(a.pilot, a.grid)
    if a.runs:
        e1_e2(a.runs, a.grid)
