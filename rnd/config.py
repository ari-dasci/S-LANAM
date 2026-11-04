"""Configuration of the cross-modal study (design in Appendix B of the online supplement).

Paths are taken from environment variables so that each machine keeps data and
runs outside the synchronised Google Drive folder.
"""
import os

DATA = os.environ.get("RND_DATA", os.path.abspath("data"))      # raw downloads and caches
RUNS = os.environ.get("RND_RUNS", os.path.abspath("runs"))      # one CSV per run
NOISE = os.path.join(RUNS, "noise")                              # noise/<dataset>_<cond>_s<s>.npz
PROBS = os.path.join(RUNS, "probs")                              # CE predictions on the training set (for FC)
EMB = os.path.join(RUNS, "emb")                                  # cached embeddings (H3)

SEEDS = [0, 1, 2, 3, 4]

# --------------------------------------------------------------------------- datasets
DATASETS = {
    "TAB": dict(modality="tabular", K=26, epochs=100, batch=128, opt="adam", lr=1e-3, wd=0.0),
    "IMG": dict(modality="image", K=10, epochs=40, batch=128, opt="sgd", lr=0.1, wd=5e-4,
                momentum=0.9, nesterov=True, sched="cosine", amp=True),
    "TXT": dict(modality="text", K=4, epochs=3, batch=32, opt="adamw", lr=2e-5, wd=0.01,
                warmup=0.06, sched="linear", max_len=128, amp=True),
    "TS": dict(modality="timeseries", K=6, epochs=100, batch=64, opt="adam", lr=1e-3, wd=0.0),
    "GRA": dict(modality="graph", K=7, epochs=200, batch=None, opt="adam", lr=0.01, wd=5e-4),
    "AUD": dict(modality="audio", K=10, epochs=50, batch=64, opt="adam", lr=1e-3, wd=0.0),
}

# round 2: one second dataset per modality, same model and recipe as the first one of its modality.
DATASETS.update({
    "TAB2": dict(DATASETS["TAB"], K=7),                        # Covertype (UCI 31), 50,000 / 20,000 fixed rows
    "IMG2": dict(DATASETS["IMG"], K=100),                      # CIFAR-100
    "TXT2": dict(DATASETS["TXT"], K=14),                       # DBpedia-14, 1,430 per class / 500 per class
    "TS2": dict(DATASETS["TS"], K=7),                          # ElectricDevices (UCR), univariate, length 96
    "GRA2": dict(DATASETS["GRA"], K=6),                        # CiteSeer (Planetoid), 60/40 stratified
    "AUD2": dict(DATASETS["AUD"], K=10, wave_len=16000),       # Speech Commands v2, 10 words, 1-s clips
})
for _k in ("TAB2", "IMG2", "TXT2", "TS2", "GRA2", "AUD2"):
    DATASETS[_k]["modality"] = DATASETS[_k[:-1]]["modality"]
KIND = {k: (k[:-1] if k.endswith("2") else k) for k in DATASETS}
BASE = ["TAB", "IMG", "TXT", "TS", "GRA", "AUD"]
SECOND = ["TAB2", "IMG2", "TXT2", "TS2", "GRA2", "AUD2"]
# round 3: UrbanSound8K with each official fold as test set; fold k uses seed k - 1.
AUD_FOLDS = [f"AUDF{k}" for k in range(1, 11)]
for _k in AUD_FOLDS:
    DATASETS[_k] = dict(DATASETS["AUD"])
    KIND[_k] = "AUD"

# --------------------------------------------------------------------------- input-noise levels
LEVELS = {
    "TAB": {1: 0.5, 2: 1.0, 3: 2.0},                         # s (standard deviations)
    "IMG": {1: dict(sigma=0.04, radius=0.5, quality=80),
            2: dict(sigma=0.08, radius=1.0, quality=50),
            3: dict(sigma=0.12, radius=1.5, quality=20)},
    "TXT": {1: 0.05, 2: 0.10, 3: 0.20},                       # p per letter
    "TS": {1: 20.0, 2: 10.0, 3: 0.0},                         # SNR dB
    "GRA": {1: 0.005, 2: 0.01, 3: 0.02},                      # flip probability
    "AUD": {1: 20.0, 2: 10.0, 3: 0.0},                        # SNR dB
}
# E1 (round 1): ladder of stronger severities, levels 4-7, same corruptions in the same native units.
LEVELS_E1 = {
    "TAB": {4: 4.0, 5: 8.0, 6: 16.0, 7: 32.0},
    "IMG": {4: dict(sigma=0.20, radius=2.5, quality=10), 5: dict(sigma=0.35, radius=4.0, quality=5),
            6: dict(sigma=0.50, radius=6.0, quality=2), 7: dict(sigma=0.80, radius=10.0, quality=1)},
    "TXT": {4: 0.4, 5: 0.6, 6: 0.8, 7: 1.0},
    "TS": {4: -5.0, 5: -10.0, 6: -15.0, 7: -20.0},
    "GRA": {4: 0.05, 5: 0.10, 6: 0.20, 7: 0.50},
    "AUD": {4: -5.0, 5: -10.0, 6: -15.0, 7: -20.0},
}
for _ds, _lv in LEVELS_E1.items():
    LEVELS[_ds].update(_lv)
# Level chosen by the E1 pilot calibration (seeds 100, 101); fixed before the E1 grid and not changed afterwards.
E1_LEVEL = {"TAB": 7, "IMG": 7, "TXT": 7, "TS": 7, "GRA": 7, "AUD": 7}   # no level reached RLA(L40), so 7
E1_PILOT_SEEDS = [100, 101]
for _k in SECOND + AUD_FOLDS:                                   # same levels in the same native units
    LEVELS[_k] = LEVELS[KIND[_k]]
# E2 (round 1): noisy test set, a fraction of the test inputs corrupted at the middle severity.
NOISY_TEST = dict(rho=0.4, level=2)

# --------------------------------------------------------------------------- conditions
# name: (rho_L, rho_A, level)
CONDITIONS = {
    "c0": (0.0, 0.0, 2),
    "L20": (0.2, 0.0, 2), "L40": (0.4, 0.0, 2),
    "A20": (0.0, 0.2, 2), "A40": (0.0, 0.4, 2),
    "J20": (0.2, 0.2, 2), "J40": (0.4, 0.4, 2),
    "A40-l1": (0.0, 0.4, 1), "A40-l3": (0.0, 0.4, 3),
}
E1_CONDS = {lv: (f"A40-l{lv}", f"J40-l{lv}") for lv in (4, 5, 6, 7)}
for _lv, (_a, _j) in E1_CONDS.items():
    CONDITIONS[_a] = (0.0, 0.4, _lv)
    CONDITIONS[_j] = (0.4, 0.4, _lv)
FM_CONDITIONS = ["S", "C", "F"]          # H4, labels only, clean inputs
# round 2: conditions of the exploratory blocks (name: rho_L, rho_A, level), all at the middle severity.
CONDITIONS.update({
    "R40": (0.0, 0.4, 2),     # H1 control: the 40% of examples selected for A40 are removed (data loss only)
    "X40": (0.0, 0.4, 2),     # H1 control: the inputs of S_A are blended with an input of another class (alpha 0.5)
    "AT40": (0.0, 0.4, 2),    # H1 control (tabular): 40% of the values of every attribute replaced uniformly
    "L40e400": (0.4, 0.0, 2), # Cora diagnosis: L40 trained for 400 epochs
})
BLOCK_DEFAULTS = {            # round 2 blocks of run.py: default conditions and methods
    "LS": (["c0", "L40", "A40", "J40"], ["CE"]),
    "H1C": (["R40", "X40", "AT40"], ["CE"]),
    "H2P": (["L40", "J40"], ["VOL", "COT"]),
    "ES": (["c0", "L40", "A40", "J40"], ["CE", "FCstar"]),
    "CN": (["Nagg", "Nr1", "Nworst", "Sagg", "Sr1", "Sworst"], ["CE", "FC", "FCstar"]),
    "CD": (["L40", "L40e400"], ["CE", "FCstar"]),
}
CN_CONDITIONS = ["Nagg", "Nr1", "Nworst", "Sagg", "Sr1", "Sworst"]   # CIFAR-10N real labels and matched symmetric
X40_ALPHA = 0.5
ES_VAL = 0.1                  # early stopping: fraction of the (noisy) training set held out for validation
VOL = dict(lam=1e-4, lr=0.01, momentum=0.9)          # VolMinNet (Li et al., 2021)
COT = dict(tk_frac=0.1)                              # Co-teaching: forget rate reaches rho_L after 10% of epochs
LOSS_EPOCHS = (0.2, 0.5, 1.0)                        # loss recording for separability, as fractions of epochs

GRID_N = {
    "CE": ["c0", "L20", "L40", "A20", "A40", "J20", "J40", "A40-l1", "A40-l3"],
    "FC": ["c0", "L20", "L40", "A20", "A40", "J20", "J40"],
    "FCstar": ["L20", "L40", "J20", "J40"],
}
# E1 (round 1): conditions allowed per method outside the main grid; run only when named with --conds.
GRID_E1 = {"CE": [c for p in E1_CONDS.values() for c in p], "FC": [c for p in E1_CONDS.values() for c in p],
           "FCstar": [j for _, j in E1_CONDS.values()]}
GRID_H3 = {"methods": ["CE", "FC"], "conditions": ["c0", "L40", "A40", "J40"], "pipelines": ["EN", "EE"]}
GRID_H4 = {"methods": ["CE", "FCstar"], "conditions": ["S", "C", "F"]}
H3_DATASETS = ["IMG", "TXT", "AUD"]
H4_DATASETS = ["IMG", "TXT", "AUD"]

# --------------------------------------------------------------------------- frozen encoders
ENCODERS = {
    "IMG": dict(kind="open_clip", arch="ViT-B-32", pretrained="openai", dim=512),
    "TXT": dict(kind="sentence_transformers", name="intfloat/e5-large-v2", prefix="query: ",
                max_len=512, dim=1024),
    "AUD": dict(kind="panns", sr=32000, dim=2048, min_len=32000),
}
HEAD = dict(lr=1e-3, batch=256, epochs=100, wd=0.0)

# --------------------------------------------------------------------------- annotators
ANNOTATORS = {
    "IMG": dict(kind="open_clip", arch="RN50", pretrained="openai", template="a photo of a {}."),
    "TXT": dict(kind="causal_lm", name="Qwen/Qwen2.5-7B-Instruct",
                prompt=("Classify the topic of the following news article as one of: World, Sports, "
                        "Business, Science and Technology.\n\nArticle: {text}\n\nAnswer with the topic only."),
                verbalizers=["World", "Sports", "Business", "Science and Technology"]),
    "AUD": dict(kind="laion_clap", template="This is a sound of {}.", sr=48000),
}

# round 3: two more zero-shot annotators per modality, same template, prompt and scoring as above.
ANNOTATORS_X = {
    "IMG": [dict(ANNOTATORS["IMG"], arch="ViT-B-16"), dict(ANNOTATORS["IMG"], arch="ViT-L-14")],
    "TXT": [dict(ANNOTATORS["TXT"], name="Qwen/Qwen2.5-3B-Instruct"),
            dict(ANNOTATORS["TXT"], name="microsoft/Phi-3.5-mini-instruct")],
    "AUD": [dict(ANNOTATORS["AUD"], ckpt="630k-best.pt", fusion=False),
            # the fusion checkpoint needs torchaudio, absent from the environment, so the second audio annotator
            # is the general-audio HTSAT-base checkpoint of the same repository (non-fusion)
            dict(ANNOTATORS["AUD"], ckpt="music_speech_audioset_epoch_15_esc_89.98.pt", fusion=False,
                 amodel="HTSAT-base")],
}
TE_CONDS = ["L20", "L40", "J20", "J40"]              # round 3
# round 4: H2B, pair label noise (P) alone and with independent input noise (JP), and symmetric label
# noise placed on the examples whose inputs are corrupted (JD); SC, the grid conditions with input noise
# standardised with statistics of the clean training set (TAB and TS).
CONDITIONS.update({"P20": (0.2, 0.0, 2), "P40": (0.4, 0.0, 2), "JP20": (0.2, 0.2, 2), "JP40": (0.4, 0.4, 2),
                   "JD20": (0.2, 0.2, 2), "JD40": (0.4, 0.4, 2)})
H2B_CONDS = ["P20", "P40", "JP20", "JP40", "JD20", "JD40"]
H2B_DATASETS = ["TAB", "TS", "GRA", "TXT"]
SC_CONDS = {"CE": ["A20", "A40", "J20", "J40", "A40-l1", "A40-l3", "AT40"], "FCstar": ["J20", "J40"]}
SC_DATASETS = ["TAB", "TS"]
# round 5: forward correction with T_hat from two-fold cross-fitted (out-of-sample) predictions.
OOS_CONDS = ["L20", "L40", "J20", "J40"]
OOS_DATASETS = BASE + SECOND

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
US8K_CLASSES = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
                "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

# --------------------------------------------------------------------------- results
CSV_COLUMNS = ["dataset", "modality", "pipeline", "method", "condition", "rho_L", "rho_A", "level", "seed",
               "n_train", "n_label_noisy", "n_input_noisy", "n_overlap", "test_acc", "test_macro_f1",
               "train_time_s", "epochs", "T_hat_path", "T_hat_frob_err", "sigma_E", "delta", "annotator",
               "r_F", "git_commit", "gpu", "deterministic", "notes", "test_acc_noisy", "test_macro_f1_noisy",
               "test_acc_es", "best_epoch", "extra_path"]
