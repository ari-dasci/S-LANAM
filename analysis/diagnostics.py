"""Checks quoted in the text of the article and supplement that are not in its tables.

Outputs in --out:
  h3_narrowing.csv          paired change of D_0.4 (CE) from the native pipeline (N) to each frozen pipeline (EN, EE)
                            on images, text and audio: D_0.4(N) - D_0.4(EN or EE) per seed, in points, with its 95%
                            interval (read by multiplicity_v15.py --h3c)
  audio_cards.csv           accuracy of CE on audio (pipeline N) for seeds 0-1 of the grid, run on the RTX 4090, and for
                            the same seeds repeated on the RTX 5090 in the ten-seed audio runs
  random_text_encoder.csv   excluded round-2 runs of the text encoder with random weights, whose initialisation did not
                            take effect, against the grid runs of the pre-trained encoder (pipeline EN)

Usage: python analysis/diagnostics.py --grid results/grid --aud10 results/audio_ten_seeds
                                      --excluded results/round2_v14_excluded --out out/diagnostics
"""
import argparse
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from aggregate import ci, load_runs, rla  # noqa: E402

H3 = ["IMG", "TXT", "AUD"]
D_CONDS = ("c0", "L40", "A40")


def d40(df, ds, pl):
    return rla(df, ds, pl, "CE", "L40") - rla(df, ds, pl, "CE", "A40")


def gpus(df, ds, pl, seed):
    """GPU models of the runs behind D_0.4 for one seed (c0 and L40 of EE are those of EN, as in aggregate.col)."""
    out = set()
    for c in D_CONDS:
        p = "EN" if pl == "EE" and c in ("c0", "L40") else pl
        s = df[(df.dataset == ds) & (df.pipeline == p) & (df.method == "CE") & (df.condition == c) & (df.seed == seed)]
        out |= set(s.gpu)
    return out


def h3_narrowing(df):
    rows = []
    for ds in H3:
        for pl in ("EN", "EE"):
            diff = (d40(df, ds, "N") - d40(df, ds, pl)).dropna()
            mixed = sum(len(gpus(df, ds, "N", s) | gpus(df, ds, pl, s)) > 1 for s in diff.index)
            c = ci(100 * diff)
            rows.append(dict(dataset=ds, pipeline=pl, mean=c["mean"], sd=c["sd"], lo=c["lo"], hi=c["hi"], n=c["n"],
                             pairs_on_two_gpu_models=mixed))
    return pd.DataFrame(rows)


def audio_ce(runs):
    fs = glob.glob(os.path.join(runs, "AUD", "N", "CE", "*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).set_index(["condition", "seed"])


def audio_cards(grid, aud10):
    g, a = audio_ce(grid), audio_ce(aud10)
    rows = []
    for (cond, seed), r in g.iterrows():
        if (cond, seed) in a.index:
            rows.append(dict(condition=cond, seed=seed, gpu_grid=r.gpu, gpu_ten_seeds=a.loc[(cond, seed), "gpu"],
                             acc_grid=r.test_acc, acc_ten_seeds=a.loc[(cond, seed), "test_acc"],
                             abs_diff_points=100 * abs(r.test_acc - a.loc[(cond, seed), "test_acc"])))
    out = pd.DataFrame(rows).sort_values(["seed", "condition"])
    seed_pairs = [100 * abs(a.loc[(c, s), "test_acc"] - a.loc[(c, t), "test_acc"])
                  for c in sorted({i[0] for i in a.index})
                  for s, t in itertools.combinations(sorted({i[1] for i in a.index if i[0] == c}), 2)]
    return out, np.array(seed_pairs)


def random_text_encoder(grid_df, excluded):
    fs = glob.glob(os.path.join(excluded, "TXT", "RN", "*", "*.csv"))
    rn = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    pt = grid_df[(grid_df.dataset == "TXT") & (grid_df.pipeline == "EN")]
    m = rn.merge(pt, on=["method", "condition", "seed"], suffixes=("_random", "_pretrained"))
    m["identical"] = m.test_acc_random == m.test_acc_pretrained
    return m[["method", "condition", "seed", "test_acc_random", "test_acc_pretrained", "identical"]]


def main(a):
    os.makedirs(a.out, exist_ok=True)
    df = load_runs(a.grid)

    h3 = h3_narrowing(df)
    h3.to_csv(os.path.join(a.out, "h3_narrowing.csv"), index=False)
    print("D_0.4(N) - D_0.4(frozen), CE, points (paired by seed):")
    for _, r in h3.iterrows():
        print(f"  {r.dataset} {r.pipeline}: {r['mean']:+.1f} [{r.lo:+.1f}, {r.hi:+.1f}] "
              f"(pairs on two GPU models: {r.pairs_on_two_gpu_models} of {r.n})")

    cards, seed_pairs = audio_cards(a.grid, a.aud10)
    cards.to_csv(os.path.join(a.out, "audio_cards.csv"), index=False)
    two = cards[cards.gpu_grid != cards.gpu_ten_seeds].abs_diff_points
    same = cards[cards.gpu_grid == cards.gpu_ten_seeds].abs_diff_points
    print(f"audio, CE: same seed on the two GPU models, {len(two)} configurations: |difference| from {two.min():.1f} "
          f"to {two.max():.1f} points (median {two.median():.1f}); same seed on the same GPU model, {len(same)} "
          f"configurations: largest |difference| {same.max():.1f}")
    print(f"audio, CE: two seeds on the RTX 5090, {len(seed_pairs)} pairs: median |difference| "
          f"{np.median(seed_pairs):.1f} points")

    rt = random_text_encoder(df, a.excluded)
    rt.to_csv(os.path.join(a.out, "random_text_encoder.csv"), index=False)
    print(f"text encoder with random weights (excluded round-2 runs): {int(rt.identical.sum())} of {len(rt)} accuracies "
          f"identical to those of the pre-trained encoder")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--aud10", required=True)
    ap.add_argument("--excluded", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
