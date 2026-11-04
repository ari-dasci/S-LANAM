"""Datasets, fixed splits and caches.

Splits never depend on the seed. Raw files are downloaded by prepare.py (after approval) into
RND_DATA; this module only reads them and builds cached arrays.
"""
import os
import zipfile
import numpy as np

from . import config as C

URLS = {
    "TAB": "https://archive.ics.uci.edu/static/public/59/letter+recognition.zip",
    "TS": "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "AUD": "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1",
}


def ddir(name):
    p = os.path.join(C.DATA, name)
    os.makedirs(p, exist_ok=True)
    return p


# ----------------------------------------------------------------------------- tabular: Letter (UCI 59)
def load_tab():
    d = ddir("TAB")
    path = os.path.join(d, "letter-recognition.data")
    if not os.path.exists(path):
        with zipfile.ZipFile(os.path.join(d, "letter+recognition.zip")) as z:
            z.extract("letter-recognition.data", d)
    rows = [l.strip().split(",") for l in open(path) if l.strip()]
    assert len(rows) == 20000, len(rows)
    y = np.array([ord(r[0]) - ord("A") for r in rows], dtype=np.int64)
    X = np.array([[float(v) for v in r[1:]] for r in rows], dtype=np.float64)
    return dict(X_train=X[:16000], y_train=y[:16000], X_test=X[16000:], y_test=y[16000:])


# ----------------------------------------------------------------------------- images: CIFAR-10
def load_img():
    from torchvision.datasets import CIFAR10
    d = ddir("IMG")
    tr = CIFAR10(d, train=True, download=False)
    te = CIFAR10(d, train=False, download=False)
    return dict(X_train=np.asarray(tr.data, dtype=np.uint8), y_train=np.asarray(tr.targets, dtype=np.int64),
                X_test=np.asarray(te.data, dtype=np.uint8), y_test=np.asarray(te.targets, dtype=np.int64))


# ----------------------------------------------------------------------------- text: AG News (20,000 balanced)
def load_txt():
    d = ddir("TXT")
    cache = os.path.join(d, "agnews_20k.npz")
    if not os.path.exists(cache):
        from datasets import load_dataset
        ds = load_dataset("fancyzhx/ag_news")
        tr_text, tr_y = list(ds["train"]["text"]), np.asarray(ds["train"]["label"], dtype=np.int64)
        rng = np.random.default_rng(0)
        idx = []
        for k in range(4):                      # classes in order, one draw each, without replacement
            idx.append(rng.choice(np.where(tr_y == k)[0], size=5000, replace=False))
        idx = np.sort(np.concatenate(idx))      # keep the original order of the training file
        np.savez_compressed(cache, train_idx=idx,
                            X_train=np.array([tr_text[i] for i in idx], dtype=object), y_train=tr_y[idx],
                            X_test=np.array(list(ds["test"]["text"]), dtype=object),
                            y_test=np.asarray(ds["test"]["label"], dtype=np.int64))
    z = np.load(cache, allow_pickle=True)
    return dict(X_train=list(z["X_train"]), y_train=z["y_train"], X_test=list(z["X_test"]), y_test=z["y_test"],
                train_idx=z["train_idx"])


# ----------------------------------------------------------------------------- time series: UCI HAR inertial signals
HAR_CHANNELS = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z",
                "total_acc_x", "total_acc_y", "total_acc_z"]


def load_ts():
    d = ddir("TS")
    cache = os.path.join(d, "har.npz")
    if not os.path.exists(cache):
        outer = os.path.join(d, "human+activity+recognition+using+smartphones.zip")
        with zipfile.ZipFile(outer) as z:
            inner = [n for n in z.namelist() if n.endswith("UCI HAR Dataset.zip")][0]
            z.extract(inner, d)
        with zipfile.ZipFile(os.path.join(d, inner)) as z:
            z.extractall(d)
        root = os.path.join(d, "UCI HAR Dataset")

        def part(split):
            X = np.stack([np.loadtxt(os.path.join(root, split, "Inertial Signals", f"{c}_{split}.txt"))
                          for c in HAR_CHANNELS], axis=1)
            y = np.loadtxt(os.path.join(root, split, f"y_{split}.txt")).astype(np.int64) - 1
            return X, y
        Xtr, ytr = part("train")
        Xte, yte = part("test")
        np.savez_compressed(cache, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)
    z = np.load(cache)
    assert z["X_train"].shape == (7352, 9, 128) and z["X_test"].shape == (2947, 9, 128)
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- graphs: Cora (PyG Planetoid)
def load_gra():
    from torch_geometric.datasets import Planetoid
    d = ddir("GRA")
    data = Planetoid(d, "Cora")[0]                 # no transform: binary features
    X = data.x.numpy().astype(np.uint8)
    y = data.y.numpy().astype(np.int64)
    rng = np.random.default_rng(0)
    tr = []
    for k in range(int(y.max()) + 1):               # stratified 60/40, classes in order
        idx = rng.permutation(np.where(y == k)[0])
        tr.append(idx[: int(round(0.6 * len(idx)))])
    train_idx = np.sort(np.concatenate(tr))
    test_idx = np.setdiff1d(np.arange(len(y)), train_idx)
    return dict(X=X, y=y, edge_index=data.edge_index.numpy(), train_idx=train_idx, test_idx=test_idx)


# ----------------------------------------------------------------------------- audio: UrbanSound8K
def _read_mono(path):
    """Mono float32 waveform at the file's own rate; librosa (audioread) only for files libsndfile rejects."""
    import soundfile as sf
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        return x.mean(axis=1), sr
    except Exception:
        import librosa
        x, sr = librosa.load(path, sr=None, mono=True)
        return np.asarray(x, dtype=np.float32), sr


def resample(x, sr_in, sr_out):
    from math import gcd
    from scipy.signal import resample_poly
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def us8k_root():
    d = ddir("AUD")
    root = os.path.join(d, "UrbanSound8K")
    if not os.path.isdir(root):
        import tarfile
        with tarfile.open(os.path.join(d, "UrbanSound8K.tar.gz")) as t:
            t.extractall(d)
    return root


def us8k_meta():
    import pandas as pd
    root = us8k_root()
    meta = pd.read_csv(os.path.join(root, "metadata", "UrbanSound8K.csv"))
    meta["path"] = [os.path.join(root, "audio", f"fold{f}", n) for f, n in zip(meta.fold, meta.slice_file_name)]
    return meta


def _wave_cache(sr):
    d = ddir("AUD")
    cache = os.path.join(d, f"waves_{sr}.npz")
    if not os.path.exists(cache):
        meta = us8k_meta()
        chunks, offs = [], [0]
        for p in meta.path:
            x, s = _read_mono(p)
            w = resample(x, s, sr)
            chunks.append(w)
            offs.append(offs[-1] + len(w))
        np.savez(cache, wave=np.concatenate(chunks), offsets=np.array(offs, dtype=np.int64),
                 fold=meta.fold.values, label=meta.classID.values.astype(np.int64))
    return cache


def load_aud(sr=16000, test_fold=10):
    """Mono waveforms at sr, unpadded, in metadata order; fold test_fold is the test set, the other nine train
    (the grid uses fold 10; round 3 uses every fold in turn)."""
    z = np.load(_wave_cache(sr))
    w, o = z["wave"], z["offsets"]
    waves = [w[o[i]:o[i + 1]] for i in range(len(o) - 1)]
    fold, y = z["fold"], z["label"]
    tr, te = np.where(fold != test_fold)[0], np.where(fold == test_fold)[0]
    return dict(X_train=[waves[i] for i in tr], y_train=y[tr], X_test=[waves[i] for i in te], y_test=y[te],
                train_rows=tr, test_rows=te)


def us8k_waves(rows, sr):
    """Read and resample the clips of the given metadata rows on the fly (32 kHz for PANNs, 48 kHz for CLAP)."""
    meta = us8k_meta()
    out = []
    for i in rows:
        x, s = _read_mono(meta.path.iloc[int(i)])
        out.append(resample(x, s, sr))
    return out


# ============================================================================= round 2: second datasets and CIFAR-10N
URLS.update({
    "TAB2": "https://archive.ics.uci.edu/static/public/31/covertype.zip",
    "TS2": "https://www.timeseriesclassification.com/aeon-toolkit/ElectricDevices.zip",
    "CIFAR10N": "https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/main/data/CIFAR-10_human.pt",
    "AUD2": "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
})


def load_tab2():
    """Covertype (UCI 31): one fixed permutation (default_rng(0)); first 50,000 rows train, next 20,000 test."""
    d = ddir("TAB2")
    cache = os.path.join(d, "covtype_70k.npz")
    if not os.path.exists(cache):
        import gzip
        with zipfile.ZipFile(os.path.join(d, "covertype.zip")) as z:
            z.extract("covtype.data.gz", d)
        with gzip.open(os.path.join(d, "covtype.data.gz"), "rt") as f:
            A = np.loadtxt(f, delimiter=",")
        assert A.shape == (581012, 55), A.shape
        perm = np.random.default_rng(0).permutation(len(A))
        tr, te = perm[:50000], perm[50000:70000]
        X, y = A[:, :54], A[:, 54].astype(np.int64) - 1
        np.savez_compressed(cache, X_train=X[tr], y_train=y[tr], X_test=X[te], y_test=y[te], rows_train=tr, rows_test=te)
    z = np.load(cache)
    return dict(X_train=z["X_train"], y_train=z["y_train"], X_test=z["X_test"], y_test=z["y_test"])


def load_img2():
    from torchvision.datasets import CIFAR100
    d = ddir("IMG2")
    tr = CIFAR100(d, train=True, download=False)
    te = CIFAR100(d, train=False, download=False)
    return dict(X_train=np.asarray(tr.data, dtype=np.uint8), y_train=np.asarray(tr.targets, dtype=np.int64),
                X_test=np.asarray(te.data, dtype=np.uint8), y_test=np.asarray(te.targets, dtype=np.int64))


def load_txt2():
    """DBpedia-14: title + content; 1,430 training and 500 test articles per class, drawn per class in class order
    with default_rng(0) and default_rng(1), indices then sorted."""
    d = ddir("TXT2")
    cache = os.path.join(d, "dbpedia_20k.npz")
    if not os.path.exists(cache):
        from datasets import load_dataset
        ds = load_dataset("fancyzhx/dbpedia_14")

        def pick(split, per, seed):
            y = np.asarray(ds[split]["label"], dtype=np.int64)
            rng = np.random.default_rng(seed)
            idx = np.sort(np.concatenate([rng.choice(np.where(y == k)[0], size=per, replace=False) for k in range(14)]))
            t, c = ds[split]["title"], ds[split]["content"]
            return np.array([f"{t[i]}. {c[i]}" for i in idx], dtype=object), y[idx], idx
        Xtr, ytr, itr = pick("train", 1430, 0)
        Xte, yte, ite = pick("test", 500, 1)
        np.savez_compressed(cache, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte, train_idx=itr, test_idx=ite)
    z = np.load(cache, allow_pickle=True)
    return dict(X_train=list(z["X_train"]), y_train=z["y_train"], X_test=list(z["X_test"]), y_test=z["y_test"])


def _read_ts_file(path):
    """aeon/sktime .ts file, univariate, equal length, integer or string class labels."""
    X, y, data = [], [], False
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("@data"):
            data = True
            continue
        if not data or line.startswith("@"):
            continue
        *dims, lab = line.split(":")
        X.append([[float(v) if v != "?" else np.nan for v in dim.split(",")] for dim in dims])
        y.append(lab)
    return np.array(X, dtype=np.float64), np.array(y)


def load_ts2():
    """ElectricDevices (UCR): official train/test split, labels remapped to 0..6 in sorted order."""
    d = ddir("TS2")
    cache = os.path.join(d, "electricdevices.npz")
    if not os.path.exists(cache):
        with zipfile.ZipFile(os.path.join(d, "ElectricDevices.zip")) as z:
            z.extractall(d)
        Xtr, ytr = _read_ts_file(os.path.join(d, "ElectricDevices_TRAIN.ts"))
        Xte, yte = _read_ts_file(os.path.join(d, "ElectricDevices_TEST.ts"))
        classes = sorted(set(ytr), key=lambda v: float(v))
        m = {c: i for i, c in enumerate(classes)}
        np.savez_compressed(cache, X_train=Xtr, y_train=np.array([m[v] for v in ytr], dtype=np.int64),
                            X_test=Xte, y_test=np.array([m[v] for v in yte], dtype=np.int64))
    z = np.load(cache)
    assert z["X_train"].shape[1:] == (1, 96), z["X_train"].shape
    return {k: z[k] for k in z.files}


def load_gra2():
    from torch_geometric.datasets import Planetoid
    d = ddir("GRA2")
    data = Planetoid(d, "CiteSeer")[0]
    X = data.x.numpy().astype(np.uint8)
    y = data.y.numpy().astype(np.int64)
    rng = np.random.default_rng(0)
    tr = []
    for k in range(int(y.max()) + 1):
        idx = rng.permutation(np.where(y == k)[0])
        tr.append(idx[: int(round(0.6 * len(idx)))])
    train_idx = np.sort(np.concatenate(tr))
    test_idx = np.setdiff1d(np.arange(len(y)), train_idx)
    return dict(X=X, y=y, edge_index=data.edge_index.numpy(), train_idx=train_idx, test_idx=test_idx)


SC_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


def load_aud2():
    """Speech Commands v0.02, ten words: 2,000 training clips per word drawn with default_rng(0) from the official
    training subset, and the whole official test subset; 16 kHz, 1 s."""
    d = ddir("AUD2")
    cache = os.path.join(d, "sc10.npz")
    if not os.path.exists(cache):
        import glob
        root = os.path.join(d, "SpeechCommands", "speech_commands_v0.02")
        if not os.path.isdir(root):
            import tarfile
            os.makedirs(root, exist_ok=True)
            with tarfile.open(os.path.join(d, "speech_commands_v0.02.tar.gz")) as t:
                t.extractall(root)
        test = set(open(os.path.join(root, "testing_list.txt")).read().split())
        val = set(open(os.path.join(root, "validation_list.txt")).read().split())
        rng = np.random.default_rng(0)
        tr_items, te_items = [], []
        for k, w in enumerate(SC_WORDS):
            files = sorted(os.path.relpath(p, root).replace("\\", "/") for p in glob.glob(os.path.join(root, w, "*.wav")))
            train = [f for f in files if f not in test and f not in val]
            pick = sorted(rng.choice(len(train), size=2000, replace=False))
            tr_items += [(train[i], k) for i in pick]
            te_items += [(f, k) for f in files if f in test]

        def read(items):
            ws = [_read_mono(os.path.join(root, f))[0] for f, _ in items]
            o = np.cumsum([0] + [len(w) for w in ws])
            return np.concatenate(ws).astype(np.float32), o.astype(np.int64), np.array([k for _, k in items], np.int64)
        wtr, otr, ytr = read(tr_items)
        wte, ote, yte = read(te_items)
        np.savez(cache, wtr=wtr, otr=otr, ytr=ytr, wte=wte, ote=ote, yte=yte)
    z = np.load(cache)

    def split(w, o):
        return [w[o[i]:o[i + 1]] for i in range(len(o) - 1)]
    return dict(X_train=split(z["wtr"], z["otr"]), y_train=z["ytr"], X_test=split(z["wte"], z["ote"]), y_test=z["yte"])


def load_cifar10n():
    """CIFAR-10N human labels (Wei et al., 2022) in the torchvision order of the CIFAR-10 training set."""
    import torch
    z = torch.load(os.path.join(ddir("IMG"), "CIFAR-10_human.pt"), weights_only=False)
    return {k: np.asarray(v, dtype=np.int64) for k, v in z.items()}


LOADERS = {"TAB": load_tab, "IMG": load_img, "TXT": load_txt, "TS": load_ts, "GRA": load_gra, "AUD": load_aud,
           "TAB2": load_tab2, "IMG2": load_img2, "TXT2": load_txt2, "TS2": load_ts2, "GRA2": load_gra2,
           "AUD2": load_aud2}


for _k in range(1, 11):                     # round 3: AUDF1-AUDF10
    LOADERS[f"AUDF{_k}"] = (lambda k: (lambda: load_aud(test_fold=k)))(_k)


def load(ds):
    return LOADERS[ds]()
