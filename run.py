"""Command line for the cross-modal study (design in Appendix B of the online supplement).

Examples
  python run.py --ds TAB,TS,GRA --blocks N                 # whole native grid of three datasets
  python run.py --ds IMG --blocks ANN,H4 --seeds 0-4       # annotate with CLIP RN50, then S/C/F runs
  python run.py --ds TXT --blocks H3                       # frozen E5-large-v2 + linear head
  python run.py --ds AUD --pilot                           # pilot: one seed, key conditions
  python run.py --ds TAB --conds c0,A40-l4,A40-l5,A40-l6,A40-l7 --methods CE --seeds 100,101   # E1 calibration
  python run.py --ds TAB --conds c0,L40,A40,J40 --methods CE --noisy-test                        # E2
  python run.py --ds TAB --blocks LS,H1C,H2P,ES                 # round 2 exploratory blocks (CN: IMG; CD: GRA)
  python run.py --ds TAB2 --conds c0,L20,L40,A20,A40,J20,J40 --methods CE   # round 2 second datasets, then FC/FCstar on L/J
Runs whose CSV already exists are skipped, so an interrupted job can simply be relaunched.
"""
import argparse
import sys
import traceback

from rnd import config as C
from rnd import experiment as X


def seeds_arg(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in s.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ds", required=True, help="comma list among TAB,IMG,TXT,TS,GRA,AUD")
    p.add_argument("--blocks", default="N", help="comma list among N,ANN,H4,H3, v1.4 LS,H1C,H2P,ES,CN,CD,H3C "
                                                  "v1.5 TE,ANNX,H4X, v1.6 H2B,SC, v1.7 OOS and v1.8 OOSC")
    p.add_argument("--seeds", default="0-4")
    p.add_argument("--conds", default=None, help="restrict conditions (N/H3: c0,L40,...; H4: S,C,F)")
    p.add_argument("--methods", default=None, help="restrict methods (CE,FC,FCstar)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pilot", action="store_true", help="seed 0 and the conditions needed by the checks")
    p.add_argument("--noisy-test", action="store_true", help="E2 (v1.3): also evaluate on the noisy test set")
    p.add_argument("--pipelines", default="RN,MH,FT", help="H3C (v1.4): comma list among RN,MH,FT")
    a = p.parse_args()

    seeds = [0] if a.pilot else seeds_arg(a.seeds)
    blocks = a.blocks.split(",") if not a.pilot else ["N", "ANN", "H4", "H3"]
    for ds in a.ds.split(","):
        X.log(f"=== {ds}: blocks {blocks}, seeds {seeds}, code {X.code_version()}")
        b = X.Bundle(ds)
        for blk in blocks:
            try:
                if blk == "N":
                    conds = a.conds.split(",") if a.conds else (["c0", "L40", "A40", "J40"] if a.pilot else C.GRID_N["CE"])
                    methods = a.methods.split(",") if a.methods else ["CE", "FC", "FCstar"]
                    for s in seeds:
                        X.run_native(b, s, conds, methods, a.device, noisy_test=a.noisy_test)
                elif blk == "ANN" and (ds in C.H4_DATASETS or ds in C.AUD_FOLDS):
                    X.annotate(b, a.device)
                elif blk == "H4" and (ds in C.H4_DATASETS or ds in C.AUD_FOLDS):
                    conds = a.conds.split(",") if a.conds else C.GRID_H4["conditions"]
                    methods = a.methods.split(",") if a.methods else C.GRID_H4["methods"]
                    for s in seeds:
                        X.run_fm(b, s, conds, methods, a.device)
                elif blk == "OOS" and ds in C.OOS_DATASETS:           # round 5
                    from rnd import v17
                    conds = a.conds.split(",") if a.conds else C.OOS_CONDS
                    for s in seeds:
                        v17.run_oos(b, s, conds, a.device)
                elif blk == "OOSC" and ds in C.OOS_DATASETS:          # round 5
                    from rnd import v18
                    conds = a.conds.split(",") if a.conds else C.OOS_CONDS
                    for s in seeds:
                        v18.run_oos_cal(b, s, conds, a.device)
                elif blk in ("H2B", "SC"):                            # round 4
                    from rnd import v16
                    if blk == "H2B" and ds in C.H2B_DATASETS:
                        conds = a.conds.split(",") if a.conds else C.H2B_CONDS
                        methods = a.methods.split(",") if a.methods else ["CE", "FC", "FCstar"]
                        for s in seeds:
                            v16.run_h2b(b, s, conds, methods, a.device)
                    elif blk == "SC" and ds in C.SC_DATASETS:
                        conds = a.conds.split(",") if a.conds else sorted({c for v in C.SC_CONDS.values() for c in v})
                        methods = a.methods.split(",") if a.methods else ["CE", "FCstar"]
                        for s in seeds:
                            v16.run_sc(b, s, conds, methods, a.device)
                elif blk in ("TE", "ANNX", "H4X"):                    # round 3
                    from rnd import v15
                    if blk == "TE":
                        conds = a.conds.split(",") if a.conds else C.TE_CONDS
                        for s in seeds:
                            v15.run_te(b, s, conds, a.device)
                    elif blk == "ANNX" and ds in C.H4_DATASETS:
                        v15.annotate_x(b, a.device)
                    elif blk == "H4X" and ds in C.H4_DATASETS:
                        conds = a.conds.split(",") if a.conds else v15.H4X_CONDS
                        methods = a.methods.split(",") if a.methods else C.GRID_H4["methods"]
                        for s in seeds:
                            v15.run_h4x(b, s, conds, methods, a.device)
                elif blk == "H3C":                                  # round 2 (IMG, TXT, AUD)
                    from rnd import h3c
                    conds = a.conds.split(",") if a.conds else h3c.H3C_CONDS
                    h3c.run_block(b, seeds, a.pipelines.split(","), conds, a.device)
                elif blk in C.BLOCK_DEFAULTS:                       # round 2 exploratory blocks
                    from rnd import v14
                    dconds, dmeth = C.BLOCK_DEFAULTS[blk]
                    if blk == "H1C" and b.kind != "TAB":
                        dconds = [c for c in dconds if c != "AT40"]
                    conds = a.conds.split(",") if a.conds else dconds
                    methods = a.methods.split(",") if a.methods else dmeth
                    for s in seeds:
                        v14.run_block(b, s, blk, conds, methods, a.device)
                elif blk == "H3" and ds in C.H3_DATASETS:
                    conds = a.conds.split(",") if a.conds else C.GRID_H3["conditions"]
                    methods = a.methods.split(",") if a.methods else C.GRID_H3["methods"]
                    enc = None
                    for s in seeds:
                        enc = X.run_frozen(b, s, conds, methods, a.device, enc)
            except Exception:
                X.log(f"ERROR in {ds} {blk}:\n{traceback.format_exc()}")
                if a.pilot:
                    raise
        X.log(f"=== {ds}: finished")


if __name__ == "__main__":
    sys.exit(main())
