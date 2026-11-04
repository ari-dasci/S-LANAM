"""Aggregation of the exploratory round 2 blocks (design in Appendix B of the online supplement).

Usage: python analysis/aggregate_v14.py --runs results/round2_v14/5090 [--runs results/round2_v14/4090] --out <dir>
Each --runs folder holds <ds>/<pipeline>/<method>/<cond>_s<seed>.csv and extra/<ds>/*.npz. Statistics compare runs
of the same folder (same GPU) only; datasets never span folders.

Outputs in --out:
  v14_summary.csv   accuracy (and early-stopping accuracy) per dataset, pipeline, method and condition
  v14_stats.csv     every statistic with its 95% interval (Student's t, n-1 df) and sign
  v14_verdicts.csv  second datasets: verdicts of H1, H2 and H2*
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from rnd import config as C  # noqa: E402
from aggregate import (NOT_EVALUABLE, NOT_SUPPORTED, SUPPORTED, ci, conclusion,  # noqa: E402
                       with_precondition)


def load(folders):
    parts = []
    for r in folders:
        files = glob.glob(os.path.join(r, "*", "*", "*", "*.csv"))
        files = [f for f in files if os.sep + "analysis" not in f]
        if files:
            d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
            d["runs"] = r
            parts.append(d)
    if not parts:
        raise SystemExit("no run CSVs found")
    return pd.concat(parts, ignore_index=True)


def series(df, ds, pl, m, cond, name="test_acc"):
    s = df[(df.dataset == ds) & (df.pipeline == pl) & (df.method == m) & (df.condition == cond)]
    return s.set_index("seed")[name].astype(float).sort_index()


def rla_of(a, a0):
    return ((a0 - a) / a0).dropna()


def auroc(scores, pos):
    """Probability that a positive example has a larger score than a negative one (ties count 1/2)."""
    scores = np.asarray(scores, dtype=float)
    y = np.zeros(len(scores), dtype=bool)
    y[np.asarray(pos, dtype=int)] = True
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(scores)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(folders, out):
    df = load(folders)
    os.makedirs(out, exist_ok=True)
    rows = []
    for (ds, pl, m, cond), g in df.groupby(["dataset", "pipeline", "method", "condition"]):
        r = dict(dataset=ds, pipeline=pl, method=m, condition=cond, runs=g.runs.iloc[0], gpu=g.gpu.iloc[0])
        r.update({f"acc_{k}": v for k, v in ci(g.test_acc).items()})
        if "test_acc_es" in g and g.test_acc_es.notna().any():
            r.update({f"acc_es_{k}": v for k, v in ci(g.test_acc_es).items()})
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(out, "v14_summary.csv"), index=False)

    S = []

    def add(ds, block, stat, v, note=""):
        c = ci(v)
        S.append(dict(dataset=ds, block=block, statistic=stat, **c, sign=conclusion(c), note=note))
        return conclusion(c)

    present = sorted(df.dataset.unique())
    for ds in present:
        acc = lambda pl, m, c, name="test_acc": series(df, ds, pl, m, c, name)   # noqa: E731
        a0 = acc("N", "CE", "c0")
        rla = lambda c, pl="N", m="CE": rla_of(acc(pl, m, c), a0)                 # noqa: E731
        kind = C.KIND.get(ds, ds)

        # LS: loss separability
        rd = df[(df.dataset == ds) & (df.pipeline == "N") & (df.method == "CE")]
        E = C.DATASETS[ds]["epochs"] if ds in C.DATASETS else None
        if E is not None and ds in C.BASE:
            eps = sorted({max(1, int(round(f * E))) for f in C.LOSS_EPOCHS})
            au = {}
            for cond in ("L40", "J40", "A40"):
                for _, row in rd[rd.condition == cond].iterrows():
                    ep_ = row.get("extra_path")
                    if not isinstance(ep_, str) or not ep_:
                        continue
                    z = np.load(os.path.join(row.runs, ep_))
                    for ep in eps:
                        k = f"loss_ep{ep}"
                        if k not in z:
                            continue
                        if cond in ("L40", "J40"):
                            au.setdefault(("L", cond, ep), {})[row.seed] = auroc(z[k], z["S_L"])
                        if cond in ("A40", "J40"):
                            au.setdefault(("A", cond, ep), {})[row.seed] = auroc(z[k], z["S_A"])
            for (w, cond, ep), d in sorted(au.items()):
                add(ds, "LS", f"AUROC_{w}({cond})@ep{ep}", pd.Series(d))
            if ("L", "J40", eps[0]) in au and ("L", "L40", eps[0]) in au:
                add(ds, "LS", f"dAUROC@ep{eps[0]}",
                    pd.Series(au[("L", "J40", eps[0])]) - pd.Series(au[("L", "L40", eps[0])]))

        # H1C
        if len(acc("N", "CE", "R40")):
            add(ds, "H1C", "Q=RLA(A40)-RLA(R40)", rla("A40") - rla("R40"))
            add(ds, "H1C", "D_X=RLA(L40)-RLA(X40)", rla("L40") - rla("X40"))
            add(ds, "H1C", "RLA(R40)", rla("R40"))
            add(ds, "H1C", "RLA(X40)", rla("X40"))
        if len(acc("N", "CE", "AT40")):
            add(ds, "H1C", "D_AT=RLA(L40)-RLA(AT40)", rla("L40") - rla("AT40"))
            add(ds, "H1C", "RLA(AT40)", rla("AT40"))

        # H2P
        for m in ("VOL", "COT"):
            if not len(acc("N", m, "L40")):
                continue
            GL = acc("N", m, "L40") - acc("N", "CE", "L40")
            GJ = acc("N", m, "J40") - acc("N", "CE", "J40")
            pre = add(ds, "H2P", f"G_{m}(L40)", GL)
            add(ds, "H2P", f"G_{m}(J40)", GJ)
            main_ = add(ds, "H2P", f"Delta_{m}", GJ - GL)
            S[-1]["note"] = with_precondition(pre, main_)
            if m == "VOL":
                add(ds, "H2P", "errT_VOL(L40)", acc("N", m, "L40", "T_hat_frob_err"))
                add(ds, "H2P", "errT_VOL(J40)", acc("N", m, "J40", "T_hat_frob_err"))

        # ES
        if len(acc("ES", "CE", "c0", "test_acc_es")):
            e0 = acc("ES", "CE", "c0", "test_acc_es")
            rES = lambda c, m="CE": rla_of(acc("ES", m, c, "test_acc_es"), e0)   # noqa: E731
            add(ds, "ES", "D^ES_0.4", rES("L40") - rES("A40"))
            add(ds, "ES", "RLA^ES(L40)", rES("L40"))
            add(ds, "ES", "RLA^ES(A40)", rES("A40"))
            GL = acc("ES", "FCstar", "L40", "test_acc_es") - acc("ES", "CE", "L40", "test_acc_es")
            GJ = acc("ES", "FCstar", "J40", "test_acc_es") - acc("ES", "CE", "J40", "test_acc_es")
            pre = add(ds, "ES", "G*^ES(L40)", GL)
            main_ = add(ds, "ES", "Delta*^ES_0.4", GJ - GL)
            S[-1]["note"] = with_precondition(pre, main_)
            add(ds, "ES", "acc_es-acc_last CE(L40)", acc("ES", "CE", "L40", "test_acc_es") - acc("ES", "CE", "L40"))

        # CN (CIFAR-10N)
        if len(acc("N", "CE", "Nagg")):
            for k in ("agg", "r1", "worst"):
                rN, rS = rla(f"N{k}"), rla(f"S{k}")
                add(ds, "CN", f"E_N({k})=RLA(N)-RLA(S)", rN - rS)
                gN = acc("N", "FCstar", f"N{k}") - acc("N", "CE", f"N{k}")
                gS = acc("N", "FCstar", f"S{k}") - acc("N", "CE", f"S{k}")
                add(ds, "CN", f"G*(N{k})", gN)
                add(ds, "CN", f"G*(S{k})", gS)
                add(ds, "CN", f"Delta_N({k})", gN - gS)
                add(ds, "CN", f"G(N{k})", acc("N", "FC", f"N{k}") - acc("N", "CE", f"N{k}"))
                add(ds, "CN", f"errT(N{k})", acc("N", "FC", f"N{k}", "T_hat_frob_err"))
                add(ds, "CN", f"errT(S{k})", acc("N", "FC", f"S{k}", "T_hat_frob_err"))

        # CD (descriptive)
        if len(acc("CD", "CE", "L40")):
            for cond in ("L40", "L40e400"):
                add(ds, "CD", f"acc CE {cond}", acc("CD", "CE", cond))
                add(ds, "CD", f"acc FC* {cond}", acc("CD", "FCstar", cond))
                add(ds, "CD", f"G* {cond}", acc("CD", "FCstar", cond) - acc("CD", "CE", cond))

        # H3C
        for pl in ("RN", "MH", "FT"):
            p0 = acc(pl, "CE", "c0")
            if not len(p0):
                continue
            rp = lambda c, m="CE": rla_of(acc(pl, m, c), p0)                     # noqa: E731
            add(ds, f"H3C-{pl}", "acc(c0)", p0)
            add(ds, f"H3C-{pl}", "RLA(L40)", rp("L40"))
            add(ds, f"H3C-{pl}", "RLA(A40)", rp("A40"))
            add(ds, f"H3C-{pl}", "D_0.4", rp("L40") - rp("A40"))
            add(ds, f"H3C-{pl}", "I_0.4", rp("J40") - rp("L40") - rp("A40"))
            if len(acc(pl, "FC", "L40")):
                GL = acc(pl, "FC", "L40") - acc(pl, "CE", "L40")
                GJ = acc(pl, "FC", "J40") - acc(pl, "CE", "J40")
                pre = add(ds, f"H3C-{pl}", "G(L40)", GL)
                main_ = add(ds, f"H3C-{pl}", "Delta_0.4", GJ - GL)
                S[-1]["note"] = with_precondition(pre, main_)

        # second datasets: statistics on the N pipeline
        if ds in C.SECOND:
            for rho in (0.2, 0.4):
                r = int(round(100 * rho))
                L, A, J = f"L{r}", f"A{r}", f"J{r}"
                add(ds, "D2", f"D_{rho}", rla(L) - rla(A))
                add(ds, "D2", f"I_{rho}", rla(J) - rla(L) - rla(A))
                add(ds, "D2", f"RLA({L})", rla(L))
                add(ds, "D2", f"RLA({A})", rla(A))
                for m, tag in (("FC", ""), ("FCstar", "*")):
                    GL = acc("N", m, L) - acc("N", "CE", L)
                    GJ = acc("N", m, J) - acc("N", "CE", J)
                    add(ds, "D2", f"G{tag}(L)_{rho}", GL)
                    add(ds, "D2", f"G{tag}(J)_{rho}", GJ)
                    add(ds, "D2", f"Delta{tag}_{rho}", GJ - GL)
        _ = kind

    st = pd.DataFrame(S)
    st.to_csv(os.path.join(out, "v14_stats.csv"), index=False)

    V = []
    for ds in [d for d in present if d in C.SECOND]:
        def sg(stat):
            s = st[(st.dataset == ds) & (st.statistic == stat)]
            return s.sign.iloc[0] if len(s) else "n/a"
        v = dict(dataset=ds, first=C.KIND[ds])
        for rho in (0.2, 0.4):
            v[f"H1_{rho}"] = SUPPORTED if sg(f"D_{rho}") == "positive" else NOT_SUPPORTED
            v[f"H2_{rho}"] = with_precondition(sg(f"G(L)_{rho}"), sg(f"Delta_{rho}"))
            v[f"H2*_{rho}"] = with_precondition(sg(f"G*(L)_{rho}"), sg(f"Delta*_{rho}"))
        V.append(v)
    pd.DataFrame(V, columns=["dataset", "first"] + [f"{h}_{r}" for r in (0.2, 0.4) for h in ("H1", "H2", "H2*")]
                 ).to_csv(os.path.join(out, "v14_verdicts.csv"), index=False)
    _ = NOT_EVALUABLE
    print(f"{len(df)} runs, {len(st)} statistics -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.runs, a.out)
