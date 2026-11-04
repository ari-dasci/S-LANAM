"""Multiplicity (Holm in the families of the main hypotheses, Benjamini-Hochberg in exploratory
blocks) and equivalence (TOST, margin 1 point) over the reported statistics; round 5 adds the OOSC block
and a sensitivity analysis with exploratory families split by kind of statistic; --h3c adds the post hoc paired
contrasts of H3 between pipelines as an exploratory family.

Inputs are the aggregated CSVs (mean, sd, n per statistic); every interval of the study is a t interval over
seeds, so p = 2 P(T_{n-1} > |mean| / (sd / sqrt n)) reproduces the reported 95% verdicts.
Usage: python analysis/multiplicity_v15.py <grid hypotheses.csv> <v14_stats.csv>... [--e1 e1.csv]
       [--v15 <agg dir>] [--h3c h3_narrowing.csv] --out <dir>
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

MARGIN = 0.01          # one percentage point (statistics are fractions of accuracy or of RLA)


def p_two_sided(mean, sd, n):
    if not np.isfinite(sd) or sd == 0 or n < 2:
        return np.nan
    t = mean / (sd / np.sqrt(n))
    return float(2 * stats.t.sf(abs(t), n - 1))


def tost_equivalent(mean, sd, n, margin=MARGIN):
    """Both one-sided tests at 5%: the 90% t interval lies within (-margin, margin)."""
    if not np.isfinite(sd) or n < 2:
        return False
    h = stats.t.ppf(0.95, n - 1) * sd / np.sqrt(n)
    return bool(mean - h > -margin and mean + h < margin)


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    if not m:
        return out
    idx = np.argsort(p[ok])
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


def bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    if not m:
        return out
    idx = np.argsort(p[ok])
    ranked = p[ok][idx] * m / (np.arange(m) + 1)
    adj = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


def grid_rows(path):
    h = pd.read_csv(path)
    rows = []

    def get(ds, pl, st, rho):
        same_rho = h.rho.isna() if pd.isna(rho) else (h.rho == rho)       # H4 statistics have no rate
        r = h[(h.dataset == ds) & (h.pipeline == pl) & (h.statistic == st) & same_rho]
        return r.iloc[0] if len(r) else None

    for _, r in h.iterrows():
        ds, pl, st, rho = r.dataset, r.pipeline, r.statistic, r.rho
        base = dict(label=f"{ds} {pl} {st} rho={rho}", mean=r["mean"], sd=r["sd"], n=int(r["n"]))
        if pl == "N" and st == "D":
            rows.append(dict(base, family="H1", kind="main"))
        elif pl == "N" and st == "Delta":
            g = get(ds, "N", "G(L)", rho)
            if g is not None and g["conclusion"] == "positive":
                rows.append(dict(base, family="H2 (evaluable)", kind="main"))
        elif pl == "N" and st == "Delta*":
            g = get(ds, "N", "G*(L)", rho)
            if g is not None and g["conclusion"] == "positive":
                rows.append(dict(base, family="H2* (evaluable)", kind="exp"))
        elif pl in ("EN", "EE") and st == "D":
            rows.append(dict(base, family="H3", kind="main"))
        elif pl in ("EN", "EE") and st == "Delta":
            g = get(ds, pl, "G(L)", rho)
            if g is not None and g["conclusion"] == "positive":
                rows.append(dict(base, family="H3", kind="main"))
        elif pl == "N" and st == "E":
            rows.append(dict(base, family="H4a", kind="main"))
        elif pl == "N" and st == "Delta_F":
            g = get(ds, "N", "G*(C)", rho)
            if g is not None and g["conclusion"] == "positive":
                rows.append(dict(base, family="H4b (evaluable)", kind="main"))
        elif pl == "N" and st == "I":
            rows.append(dict(base, family="Interaction I", kind="exp"))
        elif pl == "N" and st in ("D_l1", "D_l3"):
            rows.append(dict(base, family="H1 other severities", kind="exp"))
    return rows


V14_TESTS = {          # statistics of the round 2 blocks tested against zero in the manuscript
    "H1C": ["Q=RLA(A40)-RLA(R40)", "D_X=RLA(L40)-RLA(X40)", "D_AT=RLA(L40)-RLA(AT40)"],
    "LS": ["dAUROC"],
    "ES": ["D^ES_0.4", "Delta*^ES_0.4", "acc_es-acc_last CE(L40)"],
    "H2P": ["G_VOL(L40)", "Delta_VOL", "G_COT(L40)", "Delta_COT"],
    "CN": ["E_N(agg)=RLA(N)-RLA(S)", "E_N(r1)=RLA(N)-RLA(S)", "E_N(worst)=RLA(N)-RLA(S)", "Delta_N(worst)"],
    "H3C-RN": ["D_0.4"], "H3C-MH": ["D_0.4"], "H3C-FT": ["D_0.4"],
    "D2": ["D_0.2", "D_0.4", "Delta_0.4", "Delta*_0.4", "Delta*_0.2"],
    "CD": ["G* L40", "G* L40e400"],
}
LS_EPOCH = {"TAB": 20, "IMG": 8, "TXT": 1, "TS": 20, "GRA": 40, "AUD": 10}


def v14_rows(paths):
    rows = []
    for p in paths:
        v = pd.read_csv(p)

        def positive(ds, block, st):
            g = v[(v.dataset == ds) & (v.block == block) & (v.statistic == st)]
            return bool(len(g)) and float(g.lo.iloc[0]) > 0

        for _, r in v.iterrows():
            st = str(r.statistic)
            want = V14_TESTS.get(r.block, [])
            if not any(st == w or st.startswith(w.split("=")[0] + "=") or st == w.split("=")[0]
                       or (w == "dAUROC" and st.startswith("dAUROC@")) for w in want):
                continue
            if r.block == "LS" and st != f"dAUROC@ep{LS_EPOCH.get(r.dataset, -1)}":
                continue
            if r.block == "H3C-RN" and r.dataset == "TXT":          # invalid run of round 2, excluded and redone in round 3
                continue
            if isinstance(r.get("note", ""), str) and "not evaluable" in r.get("note", ""):
                continue
            if r.block == "D2" and st.startswith("Delta"):              # H2 and H2* need a positive gain under L
                rho = st.split("_")[-1]
                pre = "G*(L)_" if st.startswith("Delta*") else "G(L)_"
                if not positive(r.dataset, "D2", pre + rho):
                    continue
            rows.append(dict(family=f"v1.4 {r.block}", kind="exp", label=f"{r.dataset} {r.block} {st}",
                             mean=r["mean"], sd=r["sd"], n=int(r["n"])))
    return rows


def e1_rows(path):
    e = pd.read_csv(path)
    rows = []
    # Delta*_s is evaluable only where G*(L40) is positive (e1.csv does not apply the precondition)
    ok = set(e[(e.statistic == "G*(L40)") & (e.conclusion == "positive")].dataset)
    for _, r in e.iterrows():
        st = str(r.statistic)
        if st in ("D_s", "I_s") or (st == "Delta*_s" and r.dataset in ok
                                    and str(r.get("conclusion", "")) != "not evaluable"):
            rows.append(dict(family="E1", kind="exp", label=f"{r.dataset} E1 {st}", mean=r["mean"], sd=r["sd"],
                             n=int(r["n"])))
    return rows


def v15_rows(agg):
    """round 3 blocks: TE (Delta_ES where evaluable), CV (as the grid families), H4X (E, Delta_F), RN-TXT (D), DR (D_l)."""
    rows = []

    def rd(k):
        p = os.path.join(agg, f"v15_{k}.csv")
        return pd.read_csv(p) if os.path.exists(p) and os.path.getsize(p) > 2 else pd.DataFrame()

    te = rd("te")
    for _, r in te.iterrows():
        if str(r.statistic).startswith("Delta_ES") and r.conclusion != "not evaluable":
            rows.append(dict(family="v1.5 TE", kind="exp", label=f"{r.dataset} {r.statistic}", mean=r["mean"],
                             sd=r["sd"], n=int(r["n"])))
    cv = rd("cv")
    for _, r in cv.iterrows():
        if r.statistic in ("D_0.2", "D_0.4", "I_0.2", "I_0.4", "E") or (r.statistic in ("Delta_0.2", "Delta_0.4", "Delta*_0.2",
                                                                       "Delta*_0.4", "Delta_F")
                                                      and r.conclusion != "not evaluable"):
            k = int(r["n"])     # corrected resampled t (Nadeau and Bengio): sd scaled so that sd/sqrt(k) = sd*sqrt(1/k+1/9)
            rows.append(dict(family="v1.5 CV", kind="exp", label=f"AUD10 {r.statistic}", mean=r["mean"],
                             sd=r["sd"] * np.sqrt(1 + k / 9), n=k))
    hx = rd("h4x")
    if len(hx):                       # annotators with r_F < 5% are uninformative (Appendix B of the online supplement)
        rf = hx[hx.statistic == "r_F"].set_index(["dataset", "annotator"])["mean"]
        hx = hx[[rf.get((d, a), 1) >= 0.05 for d, a in zip(hx.dataset, hx.annotator)]]
    for _, r in hx.iterrows():
        if r.statistic == "E" or (r.statistic == "Delta_F" and r.conclusion != "not evaluable"):
            rows.append(dict(family="v1.5 H4X", kind="exp", label=f"{r.dataset} a{r.annotator} {r.statistic}",
                             mean=r["mean"], sd=r["sd"], n=int(r["n"])))
    rn = rd("rn_txt")
    for _, r in rn.iterrows():
        if r.statistic == "D_0.4":
            rows.append(dict(family="v1.4 H3C-RN", kind="exp", label="TXT H3C-RN D_0.4 (v1.5)", mean=r["mean"],
                             sd=r["sd"], n=int(r["n"])))
    dr = rd("dr")
    for _, r in dr.iterrows():
        if r.statistic in ("D_l4", "D_l5", "D_l6"):
            rows.append(dict(family="v1.5 DR", kind="exp", label=f"{r.dataset} {r.statistic}", mean=r["mean"],
                             sd=r["sd"], n=int(r["n"])))
    return rows


def v16_rows(agg):
    """round 4 blocks: H2B (Delta^P, Delta*^P, Delta^D, Delta*^D where evaluable; G*(JD)-G*(J)) and SC
    (D^SC at both rates and severities 1 and 3, D^SC_AT, Delta*^SC where evaluable)."""
    rows = []

    def rd(k):
        p = os.path.join(agg, f"v16_{k}.csv")
        return pd.read_csv(p) if os.path.exists(p) and os.path.getsize(p) > 2 else pd.DataFrame()

    for _, r in rd("h2b").iterrows():
        st = str(r.statistic)
        if (st.startswith("Delta") and r.conclusion != "not evaluable") or st.startswith("G*(JD") and "-G*(J" in st:
            rows.append(dict(family="v1.6 H2B", kind="exp", label=f"{r.dataset} {st}", mean=r["mean"], sd=r["sd"],
                             n=int(r["n"])))
    for _, r in rd("sc").iterrows():
        st = str(r.statistic)
        if st.startswith("D^SC") or (st.startswith("Delta*^SC") and r.conclusion != "not evaluable"):
            rows.append(dict(family="v1.6 SC", kind="exp", label=f"{r.dataset} {st}", mean=r["mean"], sd=r["sd"],
                             n=int(r["n"])))
    return rows


def v17_rows(agg):
    """round 5 block: G_OOS under L and J, Delta_OOS where evaluable, and the change of the error of T_hat."""
    p = os.path.join(agg, "v17_oos.csv")
    if not os.path.exists(p) or os.path.getsize(p) <= 2:
        return []
    rows = []
    for _, r in pd.read_csv(p).iterrows():
        st = str(r.statistic)
        if st.startswith("G_OOS(") or st.startswith("errT_OOS-errT_IS(") or (st.startswith("Delta_OOS")
                                                                            and r.conclusion != "not evaluable"):
            if np.isfinite(r["sd"]) and int(r["n"]) > 1:
                rows.append(dict(family="v1.7 OOS", kind="exp", label=f"{r.dataset} {st}", mean=r["mean"],
                                 sd=r["sd"], n=int(r["n"])))
    return rows


def v18_rows(agg):
    """round 5 block OOSC, split into two families by kind of statistic: accuracy differences (G_cal,
    G_cal - G_OOS and Delta_cal where evaluable) and changes of the error of T_hat."""
    p = os.path.join(agg, "v17_oos.csv")
    if not os.path.exists(p) or os.path.getsize(p) <= 2:
        return []
    d = pd.read_csv(p)
    if "block" not in d.columns:
        return []
    rows = []
    for _, r in d[d.block == "OOSC"].iterrows():
        st = str(r.statistic)
        if st.startswith("G_cal") or (st.startswith("Delta_cal") and r.conclusion != "not evaluable"):
            fam = "v1.8 OOSC accuracy"
        elif st.startswith("errT_cal-errT_OOS("):
            fam = "v1.8 OOSC error of T_hat"
        else:
            continue
        if np.isfinite(r["sd"]) and int(r["n"]) > 1:
            rows.append(dict(family=fam, kind="exp", label=f"{r.dataset} {st}", mean=r["mean"], sd=r["sd"],
                             n=int(r["n"])))
    return rows


def h3c_rows(path):
    """Post hoc paired contrasts of H3 (diagnostics.py, h3_narrowing.csv, in points): an exploratory family."""
    d = pd.read_csv(path)
    return [dict(family="H3 paired contrasts", kind="exp", label=f"{r.dataset} {r.pipeline} D_N-D_frozen",
                 mean=r["mean"] / 100, sd=r["sd"] / 100, n=int(r["n"])) for _, r in d.iterrows()]


def stat_kind(label):
    """Kind of statistic of a test label: its last token (not a rate or version tag) without numbers and with the
    content of parentheses unified, for the sensitivity analysis that splits each exploratory family by kind of
    statistic: e.g. G_OOS(L20) and G_OOS(J40) are one kind, errT_OOS-errT_IS(L20) another."""
    toks = [t for t in label.split(" ")[1:] if not t.startswith("rho=") and not t.startswith("(v")]
    k = toks[-1] if toks else label
    if len(toks) > 1 and re.match(r"^[LJAc][0-9]", k):      # a condition after the statistic (e.g. "G* L40")
        k = toks[-2]
    k = re.sub(r"\([^)]*\)", "(.)", k)
    return re.sub(r"[0-9.]+", "", k)


def analyse(rows):
    df = pd.DataFrame(rows)
    df["p"] = [p_two_sided(m, s, n) for m, s, n in zip(df["mean"], df["sd"], df["n"])]
    df["sig_raw"] = df.p < 0.05
    df["p_adj"] = np.nan
    for fam, g in df.groupby("family"):
        adj = holm(g.p.values) if g.kind.iloc[0] == "main" else bh(g.p.values)
        df.loc[g.index, "p_adj"] = adj
    df["sig_adj"] = df.p_adj < 0.05
    # equivalence within one point only for accuracy and RLA statistics: changes of the error of T_hat are in
    # another unit and are left out of the equivalence counts
    df["equiv_1pt"] = [(not s) and ("errT" not in lab) and tost_equivalent(m, sd, n) for s, lab, m, sd, n in
                       zip(df.sig_raw, df.label, df["mean"], df["sd"], df["n"])]
    # sensitivity: Benjamini-Hochberg within each exploratory family split by kind of statistic
    df["stat_kind"] = [stat_kind(l) for l in df.label]
    df["p_adj_by_kind"] = df.p_adj
    for (fam, sk), g in df[df.kind == "exp"].groupby(["family", "stat_kind"]):
        df.loc[g.index, "p_adj_by_kind"] = bh(g.p.values)
    summ = df.groupby(["family", "kind"]).agg(tests=("label", "size"), sig_raw=("sig_raw", "sum"),
                                              sig_adj=("sig_adj", "sum"),
                                              not_sig=("sig_raw", lambda s: int((~s).sum())),
                                              equiv=("equiv_1pt", "sum")).reset_index()
    summ["lost"] = summ.sig_raw - summ.sig_adj
    return df, summ


def main():
    a = argparse.ArgumentParser()
    a.add_argument("grid")
    a.add_argument("v14", nargs="*")
    a.add_argument("--e1")
    a.add_argument("--v15")
    a.add_argument("--v16")
    a.add_argument("--v17")
    a.add_argument("--h3c")
    a.add_argument("--out", required=True)
    o = a.parse_args()
    rows = grid_rows(o.grid) + v14_rows(o.v14)
    if o.e1:
        rows += e1_rows(o.e1)
    if o.v15:
        rows += v15_rows(o.v15)
    if o.v16:
        rows += v16_rows(o.v16)
    if o.v17:
        rows += v17_rows(o.v17) + v18_rows(o.v17)
    if o.h3c:
        rows += h3c_rows(o.h3c)
    df, summ = analyse(rows)
    os.makedirs(o.out, exist_ok=True)
    df.to_csv(os.path.join(o.out, "multiplicity_tests.csv"), index=False)
    summ.to_csv(os.path.join(o.out, "multiplicity_summary.csv"), index=False)
    print(summ.to_string(index=False))
    lost = df[df.sig_raw & ~df.sig_adj]
    flip = df[(df.kind == "exp") & ((df.p_adj < 0.05) != (df.p_adj_by_kind < 0.05))]
    print("\nexploratory verdicts that change when families are split by kind of statistic:\n" +
          (flip[["family", "label", "mean", "p_adj", "p_adj_by_kind"]].to_string(index=False) if len(flip) else "none"))
    print("\nverdicts lost after correction:\n" + (lost[["family", "label", "mean", "p", "p_adj"]].to_string(index=False)
                                                  if len(lost) else "none"))


if __name__ == "__main__":
    main()
