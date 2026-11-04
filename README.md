# S-LANAM: Label and Attribute Noise Across Modalities

Code, per-seed results and annotator labels of the comparative study in

> Julián Luengo and Salvador García. *Label and Attribute Noise Across Data Modalities: A Survey and Cross-Modal Study.* Submitted to ACM Computing Surveys.

The study trains standard models on six data modalities (tabular data, images, text, time series, graphs and audio) under label noise, input (attribute) noise and both, and compares standard training (CE) with forward loss correction using an estimated (FC) or the true (FC\*) transition matrix. Section 7 of the article reports the main grid; Appendix B of the electronic supplement describes every setting and the exploratory rounds.

## Contents

| Path | Content |
|---|---|
| `rnd/` | Datasets and fixed splits, noise generators, models, training of CE, FC and FC\*, zero-shot annotators and run units |
| `analysis/` | Aggregation of the per-run CSVs, confidence intervals, multiplicity and equivalence tests, LaTeX tables |
| `run.py` | Command line for every block of runs |
| `prepare.py` | Downloads the datasets and model weights |
| `checks.py` | Checks of the noise generators and data before the runs, and of accuracy ranges after the pilot |
| `tests/` | Unit tests on synthetic data (`python -m pytest tests`) |
| `results/` | One CSV per run (5,455 files), by round and, where two GPUs were used, by card |
| `annotations/` | Labels of the zero-shot annotators of H4 (20 files) |

## Rounds of runs

| Folder | Round | Blocks (`run.py --blocks`) |
|---|---|---|
| `results/grid` | Main grid (H1–H4) | `N`, `ANN`, `H4`, `H3` |
| `results/round1_e12` | First exploratory round: stronger input noise (E1, with its calibration pilot in `pilot/`) and noisy test inputs (E2) | `N` with `--conds` and `--noisy-test` |
| `results/round2_v14` | Second round: loss separability, controls of H1, Co-teaching and VolMinNet, early stopping, CIFAR-10N, Cora at 400 epochs, controls of H3, second datasets | `LS`, `H1C`, `H2P`, `ES`, `CN`, `CD`, `H3C`; `N` on `TAB2` … `AUD2` |
| `results/round3_v15` | Third round: transition matrix from an early-stopped model, ten folds of UrbanSound8K, more annotators, other severities of input noise, repeated random-weight text control | `TE`, `ANNX`, `H4X`; `N`, `ANN`, `H4` on `AUDF1` … `AUDF10`; `N` with `--conds A40-l4,A40-l5,A40-l6`; `H3C --pipelines RN` on `TXT` |
| `results/round4_v16` | Fourth round: pair and input-dependent label noise, clean-set standardisation | `H2B`, `SC` |
| `results/round5_v17` | Fifth round: forward correction with a transition matrix from out-of-sample predictions, raw (`FC`) and calibrated by temperature (`FCcal`) | `OOS`, then `OOSC` |
| `results/audio_ten_seeds` | Ten seeds of the audio grid on one GPU | `N` with `--seeds 0-9` |
| `results/round2_v14_excluded` | Invalid runs of the random-weight text control of the second round, used in no statistic | `H3C --pipelines RN` on `TXT` |

The 78 CSVs of seeds 2–4 in `results/audio_ten_seeds` are copies of the audio runs of `results/grid` (same card), so that folder can be aggregated on its own; the release holds 5,337 distinct runs used in the statistics and the 40 excluded runs.

Module and file names keep the internal labels of the rounds: `v14` is the second round, `v15` the third, `v16` the fourth, and `v17` and `v18` the fifth (`OOS` and its calibrated extension `OOSC`). The forty runs of the random-weight text control of the second round were invalid (the encoder kept its pre-trained weights, and every accuracy equals that of the pre-trained encoder); they are kept in `results/round2_v14_excluded`, and the third round repeats them.

## Installation

A CUDA GPU is needed for images, text and audio. The runs used Python 3.12 and PyTorch 2.13 on Windows 11 with an NVIDIA RTX 4090 and an RTX 5090.

```
conda create -n lanam python=3.12
conda activate lanam
pip install -r requirements.txt
```

The analysis scripts need only numpy, scipy and pandas (`pip install -r requirements-analysis.txt`), so every table can be regenerated from the released CSVs without a GPU or PyTorch; the unit tests need PyTorch. Commands are written for a POSIX shell; on Windows, use WSL or Git Bash. Training all runs took about 193 GPU-hours.

`torchlibrosa`, `panns_inference` and `laion_clap` may need `pip install --no-deps`, as `laion_clap` pins an old version of numpy.

## Data

Datasets and model weights are downloaded by `prepare.py` from their original sources (UCI, Zenodo, Hugging Face, the UCR archive and the TensorFlow Speech Commands release) and are not redistributed here. Three environment variables set the folders:

```
export RND_DATA=/path/to/data     # datasets and caches
export RND_RUNS=/path/to/runs     # one CSV per run, plus noise/, probs/ and emb/
export HF_HOME=/path/to/hf        # Hugging Face cache
python prepare.py --what TAB,IMG,TXT,TS,GRA,AUD --models distilbert,e5,panns,qwen,clap,clip
python prepare.py --what CIFAR10N,TAB2,IMG2,TXT2,TS2,GRA2,AUD2
```

The annotators of H4 (CLIP, Qwen2.5 and LAION-CLAP) label the clean training sets once and deterministically. Their labels are in `annotations/`; copy them to `$RND_RUNS/noise/` to reuse them instead of running the `ANN` and `ANNX` blocks, and to `results/grid/noise/` before running the analysis scripts (see below).

## Running the experiments

Every run writes one CSV, and existing CSVs are skipped, so an interrupted job can be relaunched. Examples:

```
python checks.py --ds TAB,TS --stage pre                       # generators, splits and samples
python run.py --ds TAB,TS,GRA --blocks N                       # native grid of three datasets
python run.py --ds IMG --blocks ANN,H4                         # annotate with CLIP RN50, then conditions S, C and F
python run.py --ds TXT --blocks H3                             # frozen E5-large-v2 with a linear head
python run.py --ds TAB --blocks LS,H1C,H2P,ES                  # second round
python run.py --ds TAB2 --conds c0,L20,L40,A20,A40,J20,J40     # a second dataset
python run.py --ds TAB --blocks TE                             # third round
python run.py --ds TAB --blocks H2B,SC                         # fourth round
python run.py --ds TAB --blocks OOS,OOSC                       # fifth round (OOSC reads the predictions saved by OOS)
```

`python run.py --help` lists every option. Noise is injected from fixed random streams of the seed, so the noisy training sets are identical across methods, pipelines and machines.

## Reproducing the statistics and tables

The analysis scripts read the CSVs of `results/` and write CSV summaries and the LaTeX tables of the article and supplement. For example:

```
mkdir -p results/grid/noise && cp annotations/*.npz results/grid/noise/            # annotator labels, read by the aggregation of H4
python analysis/aggregate.py --runs results/grid                                   # H1-H4: results/grid/analysis
python analysis/aggregate.py --runs results/audio_ten_seeds                        # ten audio seeds (other datasets: empty rows)
python analysis/aggregate_e12.py --runs results/round1_e12 --grid results/grid
python analysis/aggregate_v14.py --runs results/round2_v14/4090 --runs results/round2_v14/5090 --out out/v14
python analysis/aggregate_v15.py --v15 results/round3_v15/4090 results/round3_v15/5090 --grid results/grid --e12 results/round1_e12 --out out/v15
python analysis/aggregate_v16.py --v16 results/round4_v16/4090 --grid results/grid --out out/v16
python analysis/aggregate_v17.py --v17 results/round5_v17/4090 results/round5_v17/5090 --grid results/grid --v14 out/v14/v14_stats.csv --out out/v17
python analysis/diagnostics.py --grid results/grid --aud10 results/audio_ten_seeds --excluded results/round2_v14_excluded --out out/diagnostics
python analysis/multiplicity_v15.py results/grid/analysis/hypotheses.csv out/v14/v14_stats.csv --e1 results/round1_e12/analysis/e1.csv --v15 out/v15 --v16 out/v16 --v17 out/v17 --h3c out/diagnostics/h3_narrowing.csv --out out/mult
python analysis/tables.py --analysis results/grid/analysis --e12 results/round1_e12/analysis --out tables
python analysis/tables_v14.py out/v14/v14_stats.csv --out tables/emp_v14.tex
python analysis/tables_v15.py out/v15 --grid results/grid/analysis --mult out/mult --out tables
python analysis/tables_v16.py out/v16 --out tables
python analysis/tables_v17.py out/v17 --out tables
python analysis/figure_rla.py --analysis results/grid/analysis --out figures/emp_rla.tex
```

The generated figure and table bodies match the article and its supplement; captions, and the layout of Table S5 (from `emp_v14.tex`), were edited for publication, and the text cell of its random-weight row, marked †, is the third-round repeat in `out/v15/v15_rn_txt.csv`.

Statistics are paired by seed, and each paired difference uses runs of one GPU model; in the grid, audio seeds 0–1 ran on the RTX 4090 and seeds 2–4 on the RTX 5090, so the audio intervals pool both cards. Intervals are 95% Student-*t* intervals over five seeds (over ten folds with the corrected resampled *t* for UrbanSound8K).

| Table or figure of the article and supplement | Script (LaTeX file) |
|---|---|
| Results of H1–H4, complete results, error of the transition matrix, E1 and E2 | `analysis/tables.py` (`emp_results`, `emp_h4`, `emp_that`, `emp_full_native`, `emp_full_frozen`, `emp_e12`) |
| Second round | `analysis/tables_v14.py` (`emp_v14`; the article's layout was edited by hand) |
| Third round, ten folds and more annotators, multiplicity | `analysis/tables_v15.py` (`emp_v15`, `emp_cv_h4x`, `emp_mult`) |
| Fourth round | `analysis/tables_v16.py` (`emp_v16`) |
| Fifth round | `analysis/tables_v17.py` (`emp_v17`) |
| Relative accuracy loss of the grid (Figure S1) | `analysis/figure_rla.py` (`emp_rla`) |
| Numbers quoted in the text but not tabulated: paired narrowing of H3, audio seeds on the two GPU models, invalid random-weight text runs | `analysis/diagnostics.py` (CSVs in `out/diagnostics`) |

## Reproducibility

Seeds, data order and augmentation are fixed, and every CSV records the GPU, the code version and the training time. Runs reproduce on the same GPU model; on Cora (graphs), repeated runs on one GPU can differ by one test node because of non-deterministic scatter operations. Different GPU models give slightly different accuracies, so statistics pair runs of one card; the one exception is the paired narrowing of H3 on audio, which pairs native runs of seeds 2–4 (RTX 5090) with frozen-encoder runs (RTX 4090).

## Results format

Each CSV holds one run: dataset, modality, pipeline (`N` native; `EN`, `EE` frozen encoders; block names for the exploratory rounds), method (`CE`, `FC`, `FCstar` and others), condition (for example `L40`: label noise on 40% of the training examples; `A40`: input noise on 40%; `J40`: both), noise rates and severity, seed, number of noisy examples, test accuracy and macro-F1, training time, error of the estimated transition matrix, GPU and code version. In the `CEhalf0` and `CEhalf1` rows of the fifth round (models trained on one fold), `test_acc` is the accuracy on the held-out fold against the clean labels and `n_train` records the size of the whole training set. The `extra/` files hold per-epoch training curves and the indices of the noisy examples. In the fifth round, `T_hat_path` points to the estimated transition matrices, released under `probs/` of each card folder (`*_That_oos.npy` and `*_That_cal.npy`, rows indexed by the noisy class), and the `notes` field holds the diagnostics of each estimate. The code version is a hash of the development version of `rnd/*.py` at the time of the run; the released code differs from it only in comments and docstrings.

## License and citation

- Code (`rnd/`, `analysis/`, `run.py`, `prepare.py`, `checks.py`, `tests/`): MIT License, see `LICENSE`.
- Per-seed results (`results/`) and annotator labels (`annotations/`): Creative Commons Attribution 4.0 International (CC BY 4.0), see `LICENSE-data`.

Third-party material keeps its own terms. The datasets are not redistributed and are subject to the licenses of their providers (for example, UrbanSound8K is licensed for non-commercial use only). The annotator labels were produced by pre-trained models released under their own licenses (CLIP, Qwen2.5-7B-Instruct, Qwen2.5-3B-Instruct, Phi-3.5-mini-instruct and LAION-CLAP); check those licenses before reusing the labels.

If you use this code or these results, please cite the article above.
