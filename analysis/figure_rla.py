"""Bar chart of the RLA of CE under label, input and joint noise in the grid (native inputs), Figure S1 of the
electronic supplement, written as a pgfplots figure from the summary of analysis/aggregate.py.

Usage: python analysis/figure_rla.py --analysis results/grid/analysis --out figures/emp_rla.tex
Bars: mean RLA (%) over the five seeds; whiskers: half-width of the 95% Student-t interval.
"""
import argparse
import os

import pandas as pd

MODS = [("TAB", "Tabular"), ("IMG", "Images"), ("TXT", "Text"), ("TS", "Time series"), ("GRA", "Graphs"),
        ("AUD", "Audio")]

TEMPLATE = r"""\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[ybar, width=0.53\textwidth, height=3.9cm, bar width=5pt, enlarge x limits=0.09,
  symbolic x coords={Tabular,Images,Text,Time series,Graphs,Audio}, xtick=data, x tick label style={font=\scriptsize, rotate=30, anchor=east},
  y tick label style={font=\scriptsize}, ymin=-20, ymax=70, ytick={-20,-10,0,10,20,30,40,50,60,70},
  ylabel={RLA (\%)}, ylabel style={font=\scriptsize}, title={\small $\rho=0.2$}, title style={yshift=-1.5ex},
  error bars/y dir=both, error bars/y explicit, error bars/error bar style={line width=0.4pt}, error bars/error mark options={rotate=90, mark size=1.2pt},
  axis x line*=bottom, axis y line*=left, ymajorgrids, grid style={black!12}, extra y ticks={0}, extra y tick style={grid style={black!60}}, legend style={font=\scriptsize, draw=none, fill=white, fill opacity=0.85, text opacity=1, at={(0.02,0.98)}, anchor=north west, legend columns=1}, legend image code/.code={\draw[#1] (0cm,-0.08cm) rectangle (0.22cm,0.12cm);}]
\addplot[fill=black!75, draw=black, mark=none, error bars/.cd, y dir=both, y explicit, error bar style={black, line width=0.4pt}] coordinates {<<L20>>};
\addplot[fill=black!15, draw=black, mark=none, error bars/.cd, y dir=both, y explicit, error bar style={black, line width=0.4pt}] coordinates {<<A20>>};
\addplot[pattern=north east lines, draw=black, mark=none, error bars/.cd, y dir=both, y explicit, error bar style={black, line width=0.4pt}] coordinates {<<J20>>};
\legend{Label noise, Input noise, Both}
\end{axis}
\begin{scope}[xshift=0.5\textwidth]
\begin{axis}[ybar, width=0.53\textwidth, height=3.9cm, bar width=5pt, enlarge x limits=0.09,
  symbolic x coords={Tabular,Images,Text,Time series,Graphs,Audio}, xtick=data, x tick label style={font=\scriptsize, rotate=30, anchor=east},
  y tick label style={font=\scriptsize}, ymin=-20, ymax=70, ytick={-20,-10,0,10,20,30,40,50,60,70},
  ylabel={}, ylabel style={font=\scriptsize}, title={\small $\rho=0.4$}, title style={yshift=-1.5ex},
  error bars/y dir=both, error bars/y explicit, error bars/error bar style={line width=0.4pt}, error bars/error mark options={rotate=90, mark size=1.2pt},
  axis x line*=bottom, axis y line*=left, ymajorgrids, grid style={black!12}, extra y ticks={0}, extra y tick style={grid style={black!60}},]
\addplot[fill=black!75, draw=black, mark=none, error bars/.cd, y dir=both, y explicit, error bar style={black, line width=0.4pt}] coordinates {<<L40>>};
\addplot[fill=black!15, draw=black, mark=none, error bars/.cd, y dir=both, y explicit, error bar style={black, line width=0.4pt}] coordinates {<<A40>>};
\addplot[pattern=north east lines, draw=black, mark=none, error bars/.cd, y dir=both, y explicit, error bar style={black, line width=0.4pt}] coordinates {<<J40>>};
\end{axis}
\end{scope}
\end{tikzpicture}
\caption{RLA (\%) of CE training under label noise, input noise and both at the same rate $\rho$ in the six modalities of the study (native inputs, clean test sets, five seeds; whiskers show 95\% Student-$t$ intervals). Under the operators of \cref{tab:emp-noise} and fixed-budget, final-epoch training, label noise costs more than input noise at the same rate in every modality (H1; \cref{tab:emp-results}). On time series and audio, input noise did not lower mean accuracy.}
\Description{Two grouped bar charts, for noise rates 0.2 and 0.4, with three bars per modality (label noise, input noise, both) for tabular data, images, text, time series, graphs and audio. Label-noise and joint-noise bars are higher than input-noise bars everywhere; audio shows the largest label-noise loss and a negative input-noise loss.}
\label{fig:emp-rla}
\end{figure}
"""


def coords(s, cond):
    out = []
    for ds, name in MODS:
        r = s[(s.dataset == ds) & (s.pipeline == "N") & (s.method == "CE") & (s.condition == cond)].iloc[0]
        out.append(f"({name},{100 * r.rla_mean:.2f}) +- (0,{100 * (r.rla_hi - r.rla_mean):.2f})")
    return " ".join(out)


def main(analysis, out):
    s = pd.read_csv(os.path.join(analysis, "summary.csv"))
    tex = TEMPLATE
    for cond in ("L20", "A20", "J20", "L40", "A40", "J40"):
        tex = tex.replace(f"<<{cond}>>", coords(s, cond))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w", encoding="utf-8").write(tex)
    print("written", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.analysis, a.out)
