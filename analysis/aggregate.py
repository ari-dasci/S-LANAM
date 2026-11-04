"""Aggregation of the per-run CSVs (design in Appendix B of the online supplement).

Outputs in <RUNS>/analysis/:
  summary.csv          accuracy and RLA per (dataset, pipeline, method, condition)
  hypotheses.csv       every statistic with its 95% interval and conclusion
  verdicts.csv         per dataset: supported / not supported / not evaluable (precondition not met)
  verdict_counts.csv   number of datasets per verdict (aggregation)
  that_error.csv       error of the estimated transition matrix by condition (exploratory, grid)
  annotators.csv       r_F, diagonal of T_F and label marginals of each foundation-model annotator
  emp_results.tex, emp_h2.tex, emp_h3.tex, emp_h4.tex   numbers for the tables of Section 8
95% intervals over seeds use Student's t with n-1 degrees of freedom (2.776 for five seeds).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rnd import config as C  # noqa: E402

SUPPORTED, NOT_SUPPORTED, NOT_EVALUABLE = "supported", "not supported", "not evaluable"


def ci(v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return dict(mean=float(v.mean()) if n else np.nan, sd=np.nan, lo=np.nan, hi=np.nan, n=n)
    m, sd = v.mean(), v.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return dict(mean=m, sd=sd, lo=m - h, hi=m + h, n=n)


def conclusion(c):
    if np.isnan(c["lo"]):
        return "n/a"
    return "positive" if c["lo"] > 0 else "negative" if c["hi"] < 0 else "null"


def load_runs(runs):
    files = [f for f in glob.glob(os.path.join(runs, "*", "*", "*", "*.csv"))]
    if not files:
        raise SystemExit("no run CSVs found under " + runs)
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def col(df, ds, pl, m, cond, name="test_acc"):
    """A column per seed as a Series indexed by seed. c0 and L40 of EE are shared with EN."""
    if pl == "EE" and cond in ("c0", "L40"):
        pl = "EN"
    s = df[(df.dataset == ds) & (df.pipeline == pl) & (df.method == m) & (df.condition == cond)]
    return s.set_index("seed")[name].astype(float).sort_index()


def acc(df, ds, pl, m, cond):
    return col(df, ds, pl, m, cond)


def rla(df, ds, pl, m, cond):
    base_m = "CE" if m == "FCstar" else m
    a0 = acc(df, ds, pl, base_m, "c0")
    a = acc(df, ds, pl, m, cond)
    return ((a0 - a) / a0).dropna()


def with_precondition(pre, main):
    """Hypotheses with a precondition (H2, H2*, H4b): the main statistic must be negative."""
    if pre != "positive":
        return NOT_EVALUABLE
    return SUPPORTED if main == "negative" else NOT_SUPPORTED


def main(runs):
    df = load_runs(runs)
    out = os.path.join(runs, "analysis")
    os.makedirs(out, exist_ok=True)

    # summary: accuracy and RLA per (dataset, pipeline, method, condition)
    rows = []
    for (ds, pl, m, cond), g in df.groupby(["dataset", "pipeline", "method", "condition"]):
        r = dict(dataset=ds, pipeline=pl, method=m, condition=cond)
        c = ci(g.test_acc)
        r.update({f"acc_{k}": v for k, v in c.items()})
        rr = ci(rla(df, ds, pl, m, cond))
        r.update({f"rla_{k}": v for k, v in rr.items() if k != "n"})
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(out, "summary.csv"), index=False)

    # hypotheses
    H = []

    def add(ds, pl, stat, rho, series, note=""):
        c = ci(series)
        H.append(dict(dataset=ds, pipeline=pl, statistic=stat, rho=rho, **c, conclusion=conclusion(c), note=note))
        return conclusion(c)

    for ds in C.BASE:
        for pl in ["N", "EN", "EE"]:
            if pl != "N" and ds not in C.H3_DATASETS:
                continue
            rhos = [0.2, 0.4] if pl == "N" else [0.4]
            for rho in rhos:
                r = int(round(100 * rho))
                L, A, J = f"L{r}", f"A{r}", f"J{r}"
                add(ds, pl, "D", rho, rla(df, ds, pl, "CE", L) - rla(df, ds, pl, "CE", A))
                add(ds, pl, "I", rho, rla(df, ds, pl, "CE", J) - rla(df, ds, pl, "CE", L) - rla(df, ds, pl, "CE", A))
                for m, tag in (("FC", ""), ("FCstar", "*")):
                    if pl != "N" and m == "FCstar":
                        continue
                    GL = acc(df, ds, pl, m, L) - acc(df, ds, pl, "CE", L)
                    GJ = acc(df, ds, pl, m, J) - acc(df, ds, pl, "CE", J)
                    add(ds, pl, f"G{tag}(L)", rho, GL)
                    add(ds, pl, f"G{tag}(J)", rho, GJ)
                    add(ds, pl, f"Delta{tag}", rho, GJ - GL,
                        note="exploratory (v1.2)" if tag == "*" else "")
                    if m == "FC":
                        add(ds, pl, "G(A)", rho, acc(df, ds, pl, "FC", A) - acc(df, ds, pl, "CE", A))
                        # exploratory (grid): does joint noise degrade the estimate of T?
                        add(ds, pl, "errT(J)-errT(L)", rho,
                            col(df, ds, pl, "FC", J, "T_hat_frob_err") - col(df, ds, pl, "FC", L, "T_hat_frob_err"),
                            note="exploratory (v1.2)")
            if pl == "N":
                for lv in ("l1", "l3"):
                    add(ds, pl, f"D_{lv}", 0.4, rla(df, ds, pl, "CE", "L40") - rla(df, ds, pl, "CE", f"A40-{lv}"))
        if ds in C.H4_DATASETS:
            rS, rC, rF = (rla(df, ds, "N", "CE", c) for c in ("S", "C", "F"))
            add(ds, "N", "E", np.nan, rF - rS)
            add(ds, "N", "E_class", np.nan, rC - rS)
            add(ds, "N", "E_instance", np.nan, rF - rC)
            G = {c: acc(df, ds, "N", "FCstar", c) - acc(df, ds, "N", "CE", c) for c in ("S", "C", "F")}
            for c in G:
                add(ds, "N", f"G*({c})", np.nan, G[c])
            add(ds, "N", "Delta_F", np.nan, G["F"] - G["C"])
    hyp = pd.DataFrame(H)
    hyp.to_csv(os.path.join(out, "hypotheses.csv"), index=False)

    # error of the estimated transition matrix by condition (exploratory, grid)
    E = []
    fc = df[(df.method == "FC") & df.T_hat_frob_err.notna()] if "T_hat_frob_err" in df else df.iloc[:0]
    for (ds, pl, cond), g in fc.groupby(["dataset", "pipeline", "condition"]):
        E.append(dict(dataset=ds, pipeline=pl, condition=cond, **ci(g.set_index("seed").T_hat_frob_err)))
    pd.DataFrame(E, columns=["dataset", "pipeline", "condition", "mean", "sd", "lo", "hi", "n"]).to_csv(
        os.path.join(out, "that_error.csv"), index=False)

    # verdicts per dataset
    def concl(ds, pl, stat, rho=None):
        s = hyp[(hyp.dataset == ds) & (hyp.pipeline == pl) & (hyp.statistic == stat)]
        if rho is not None:
            s = s[np.isclose(s.rho.astype(float), rho)]
        return s.conclusion.iloc[0] if len(s) else "n/a"

    verdicts = []
    for ds in C.BASE:
        v = dict(dataset=ds)
        for rho in (0.2, 0.4):
            v[f"H1_{rho}"] = SUPPORTED if concl(ds, "N", "D", rho) == "positive" else NOT_SUPPORTED
            v[f"H2_{rho}"] = with_precondition(concl(ds, "N", "G(L)", rho), concl(ds, "N", "Delta", rho))
            v[f"H2*_{rho}"] = with_precondition(concl(ds, "N", "G*(L)", rho), concl(ds, "N", "Delta*", rho))
        if ds in C.H3_DATASETS:
            diff = []
            for pl in ("EN", "EE"):
                for st in ("D", "Delta"):
                    if concl(ds, pl, st, 0.4) != concl(ds, "N", st, 0.4):
                        diff.append(f"{st}:{pl}")
            v["H3"] = (SUPPORTED + " (" + ";".join(diff) + ")") if diff else NOT_SUPPORTED
        if ds in C.H4_DATASETS:
            v["H4a"] = SUPPORTED if concl(ds, "N", "E") == "positive" else NOT_SUPPORTED
            v["H4b"] = with_precondition(concl(ds, "N", "G*(C)"), concl(ds, "N", "Delta_F"))
        verdicts.append(v)
    vd = pd.DataFrame(verdicts)
    vd.to_csv(os.path.join(out, "verdicts.csv"), index=False)
    counts = []
    for k in vd.columns[1:]:
        lab = vd[k].dropna().astype(str).map(lambda x: NOT_EVALUABLE if x.startswith(NOT_EVALUABLE) else
                                              NOT_SUPPORTED if x.startswith(NOT_SUPPORTED) else SUPPORTED)
        counts.append(dict(hypothesis=k, n=len(lab), **{x: int((lab == x).sum()) for x in
                                                      (SUPPORTED, NOT_SUPPORTED, NOT_EVALUABLE)}))
    pd.DataFrame(counts).to_csv(os.path.join(out, "verdict_counts.csv"), index=False)

    # annotators
    ann = []
    for ds in C.H4_DATASETS:
        p = os.path.join(runs, "noise", f"{ds}_F.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            yhat = z["yhat"]
            ann.append(dict(dataset=ds, revision=str(z["revision"]), r_F=float(z["r_F"]),
                            diag=" ".join(f"{x:.3f}" for x in np.diag(z["T_F"])),
                            marginal=" ".join(f"{x:.3f}" for x in np.bincount(yhat, minlength=len(z["T_F"])) / len(yhat))))
    ann = pd.DataFrame(ann)
    ann.to_csv(os.path.join(out, "annotators.csv"), index=False)

    # LaTeX tables for Section 8 (numbers only; the layout is finalised in the manuscript)
    def cell(ds, pl, st, rho, scale=100):
        s = hyp[(hyp.dataset == ds) & (hyp.pipeline == pl) & (hyp.statistic == st)]
        s = s[s.rho.isna()] if rho is None else s[np.isclose(s.rho.astype(float), rho)]
        if not len(s) or np.isnan(s["mean"].iloc[0]):
            return "--"
        mark = {"positive": "$^{+}$", "negative": "$^{-}$", "null": ""}.get(s.conclusion.iloc[0], "")
        return f"{scale * s['mean'].iloc[0]:.1f}{mark}"

    def table(path, head, body):
        n = head.count("&")
        lines = [r"\begin{tabular}{l" + "r" * n + "}", r"\toprule", head + r" \\", r"\midrule"]
        lines += [" & ".join(row) + r" \\" for row in body]
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(out, path), "w").write("\n".join(lines) + "\n")

    body = []
    for ds in C.BASE:
        a0 = acc(df, ds, "N", "CE", "c0")
        body.append([ds, f"{100 * a0.mean():.1f}" if len(a0) else "--", cell(ds, "N", "D", 0.2), cell(ds, "N", "D", 0.4),
                     cell(ds, "N", "I", 0.4), cell(ds, "N", "G(L)", 0.2), cell(ds, "N", "G(L)", 0.4),
                     cell(ds, "N", "Delta", 0.2), cell(ds, "N", "Delta", 0.4)])
    table("emp_results.tex", r"Dataset & $A_{0\%}$ & $D_{0.2}$ & $D_{0.4}$ & $I_{0.4}$ & $G(L_{0.2})$ & $G(L_{0.4})$ & "
          r"$\Delta_{0.2}$ & $\Delta_{0.4}$", body)

    body = [[ds] + [cell(ds, "N", st, rho) for tag in ("", "*") for rho in (0.2, 0.4)
                    for st in (f"G{tag}(L)", f"Delta{tag}")] for ds in C.BASE]
    table("emp_h2.tex", r"Dataset & $G(L_{0.2})$ & $\Delta_{0.2}$ & $G(L_{0.4})$ & $\Delta_{0.4}$ & "
          r"$G^{*}(L_{0.2})$ & $\Delta^{*}_{0.2}$ & $G^{*}(L_{0.4})$ & $\Delta^{*}_{0.4}$", body)

    body = [[ds] + [cell(ds, pl, st, 0.4) for st in ("D", "Delta") for pl in ("N", "EN", "EE")]
            for ds in C.H3_DATASETS]
    table("emp_h3.tex", r"Dataset & $D^{N}_{0.4}$ & $D^{EN}_{0.4}$ & $D^{EE}_{0.4}$ & $\Delta^{N}_{0.4}$ & "
          r"$\Delta^{EN}_{0.4}$ & $\Delta^{EE}_{0.4}$", body)

    body = []
    for ds in C.H4_DATASETS:
        rf = ann[ann.dataset == ds].r_F if len(ann) else []
        body.append([ds, f"{100 * float(rf.iloc[0]):.1f}" if len(rf) else "--"] +
                    [cell(ds, "N", st, None) for st in ("E", "E_class", "E_instance", "G*(C)", "G*(F)", "Delta_F")])
    table("emp_h4.tex", r"Dataset & $r_F$ & $E$ & $E_{\mathrm{class}}$ & $E_{\mathrm{inst}}$ & $G^{*}(C)$ & "
          r"$G^{*}(F)$ & $\Delta_F$", body)
    print(f"aggregate: {len(df)} runs -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=C.RUNS)
    main(p.parse_args().runs)
