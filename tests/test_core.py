"""Unit and smoke tests that need no downloads (synthetic data, CPU)."""
import os
import sys
import tempfile

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rnd import config as C, models as M, noise as N, train as TR  # noqa: E402


def test_label_noise_exact():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 10, 5000)
    for rho in (0.2, 0.4):
        for s in range(5):
            yn, S, T = N.label_noise(y, 10, rho, s)
            assert np.sum(yn != y) == round(rho * len(y))
            assert np.all(yn[S] != y[S])
            assert np.allclose(T.sum(1), 1) and np.isclose(T[0, 0], 1 - rho)
            yn2, S2, _ = N.label_noise(y, 10, rho, s)
            assert np.array_equal(yn, yn2) and np.array_equal(S, S2)


def test_selection_independent_of_level_and_type():
    S1 = N.select_inputs(1000, 0.4, 3)
    S2 = N.select_inputs(1000, 0.4, 3)
    assert np.array_equal(S1, S2) and len(S1) == 400 and np.all(np.diff(S1) > 0)
    yn, SL, _ = N.label_noise(np.zeros(1000, int), 5, 0.4, 3)
    assert not np.array_equal(np.sort(SL), S1)          # labels and inputs use independent streams


def test_string_corruption():
    rng = np.random.default_rng(0)
    s = "Hello, world! 123"
    assert N.corrupt_string(s, 0.0, rng) == s
    out = N.corrupt_string(s, 1.0, np.random.default_rng(1))
    assert "123" in out and "," in out and "!" in out      # non-letters untouched
    t = N.corrupt_texts(["abc def"] * 3, [1], 0.5, np.random.default_rng(2))
    assert t[0] == "abc def" and t[2] == "abc def"


def test_series_and_audio_snr():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 9, 128))
    Xn = N.corrupt_series(X, np.arange(50), 10.0, np.random.default_rng(1))
    snr = [10 * np.log10(X[i, c].var() / (Xn[i, c] - X[i, c]).var()) for i in range(50) for c in range(9)]
    assert abs(np.mean(snr) - 10.0) < 0.5
    waves = [rng.standard_normal(16000).astype(np.float32), np.zeros(800, np.float32)]
    wn, silent = N.corrupt_waves(waves, [0, 1], 0.0, np.random.default_rng(3))
    assert silent == [1] and wn[1] is waves[1]
    assert abs(10 * np.log10(np.mean(waves[0] ** 2) / np.mean((wn[0] - waves[0]) ** 2))) < 0.2


def test_images_and_graph():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 256, (20, 32, 32, 3), dtype=np.uint8)
    Xn, types = N.corrupt_images(X, np.arange(0, 20, 2), C.LEVELS["IMG"][2], np.random.default_rng(1))
    assert Xn.dtype == np.uint8 and np.array_equal(Xn[1], X[1]) and len(types) == 10
    assert not np.array_equal(Xn[0], X[0])
    G = (rng.random((30, 1433)) < 0.02).astype(np.uint8)
    Gn = N.corrupt_graph_features(G, np.array([0, 5]), 0.01, np.random.default_rng(2))
    assert np.array_equal(Gn[1], G[1]) and 0 < np.sum(Gn[0] != G[0]) < 60


def test_fm_conditions():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 4000)
    yhat = np.where(rng.random(4000) < 0.2, (y + 1) % 4, y)
    r_F, T_F = N.annotator_stats(y, yhat, 4)
    assert np.allclose(T_F.sum(1), 1) and abs(r_F - 0.2) < 0.03
    yS, TS = N.fm_condition("S", y, yhat, 4, 0, r_F, T_F)
    assert np.sum(yS != y) == round(r_F * 4000)
    rates = [np.mean(N.fm_condition("C", y, yhat, 4, s, r_F, T_F)[0] != y) for s in range(5)]
    assert abs(np.mean(rates) - r_F) < 0.01
    yF, TF = N.fm_condition("F", y, yhat, 4, 0, r_F, T_F)
    assert np.array_equal(yF, yhat) and TF is T_F


def test_estimate_T_and_forward_loss():
    p = np.random.default_rng(0).dirichlet(np.ones(3), size=1000)
    T = TR.estimate_T(p)
    assert T.shape == (3, 3) and np.allclose(T.sum(1), 1)
    logits = torch.randn(8, 3)
    y = torch.randint(0, 3, (8,))
    assert torch.isclose(TR.forward_loss(logits, y, torch.eye(3)), TR.forward_loss(logits, y, None), atol=1e-5)
    # same value as the direct formula on moderate logits
    Tt = torch.tensor(T, dtype=torch.float32)
    direct = F.nll_loss(torch.log(torch.softmax(logits, 1) @ Tt), y)
    assert torch.isclose(TR.forward_loss(logits, y, Tt), direct, atol=1e-5)
    # T = I equals CE, value and gradient, even for confidently wrong logits
    big = torch.tensor([[60.0, -60.0, 0.0], [0.0, 80.0, -80.0]], requires_grad=True)
    yb = torch.tensor([1, 2])
    TR.forward_loss(big, yb, torch.eye(3)).backward()
    g_fc = big.grad.clone()
    big.grad = None
    F.cross_entropy(big, yb).backward()
    assert torch.allclose(g_fc, big.grad, atol=1e-6) and g_fc.abs().sum() > 0.5


def test_models_shapes():
    assert M.MLP()(torch.randn(4, 16)).shape == (4, 26)
    assert M.FCN()(torch.randn(4, 9, 128)).shape == (4, 6)
    assert M.AudioCNN()(torch.randn(2, 64, 401)).shape == (2, 10)
    A = M.gcn_adjacency(np.array([[0, 1, 2], [1, 2, 3]]), 5, "cpu")
    assert M.GCN(10, 3)(torch.randn(5, 10), A).shape == (5, 3)
    fb = M.mel_filterbank()
    assert M.log_mel(torch.randn(2, 64000), fb).shape == (2, 64, 401)


def test_fit_predict_learns_tab_and_ts():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((26, 16)) * 3
    y = rng.integers(0, 26, 2000)
    X = centers[y] + rng.standard_normal((2000, 16))
    task = TR.task_tab(X[:1500], X[1500:], "cpu")
    cfg = dict(C.DATASETS["TAB"], epochs=5)
    r = TR.fit_predict(task, y[:1500], y[1500:], cfg, 0, "cpu", need_train_probs=True)
    assert r["test_acc"] > 0.9 and r["train_probs"].shape == (1500, 26)
    r2 = TR.fit_predict(task, y[:1500], y[1500:], cfg, 0, "cpu")
    assert r2["test_acc"] == r["test_acc"]                 # deterministic given the seed
    Xs = rng.standard_normal((300, 9, 128))
    ys = rng.integers(0, 6, 300)
    Xs[np.arange(300), ys % 9] += 2.0
    r3 = TR.fit_predict(TR.task_ts(Xs[:200], Xs[200:], "cpu"), ys[:200], ys[200:], dict(C.DATASETS["TS"], epochs=10),
                        0, "cpu", T=N.symmetric_T(6, 0.2))
    assert 0.0 <= r3["test_acc"] <= 1.0


def test_graph_fit():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, 300)
    X = (rng.random((300, 50)) < 0.05).astype(np.uint8)
    X[np.arange(300), y] = 1
    order = np.argsort(y, kind='stable')                    # homophilous ring within each class
    ei = np.array([[order[i], order[i + 1]] for i in range(299) if y[order[i]] == y[order[i + 1]]]).T
    tr, te = np.arange(200), np.arange(200, 300)
    cfg = dict(C.DATASETS["GRA"], epochs=50)
    r = TR.fit_predict_graph(X, y, y[tr], ei, tr, te, cfg, 0, "cpu", need_train_probs=True)
    assert r["test_acc"] > 0.5 and r["train_probs"].shape == (200, 7)


def test_img_task_augmentation_cpu():
    X = np.random.default_rng(0).integers(0, 256, (16, 32, 32, 3), dtype=np.uint8)
    t = TR.task_img(X, X[:4], TR.img_stats(X), "cpu")
    g = torch.Generator().manual_seed(0)
    xb = t["get"]("train", torch.arange(8), True, g)
    assert xb.shape == (8, 3, 32, 32)
    assert torch.allclose(t["get"]("test", torch.arange(4), False, None), t["get"]("train", torch.arange(4), False, None))


def test_aggregate_on_fake_runs():
    from rnd import experiment as X
    with tempfile.TemporaryDirectory() as d:
        C.RUNS = d
        rng = np.random.default_rng(0)
        for s in range(5):
            for cond in C.GRID_N["CE"]:
                X.write_row("TAB", "N", "CE", cond, s, test_acc=0.95 - 0.1 * ("L" in cond) - 0.03 * ("A" in cond)
                            + 0.01 * rng.standard_normal())
            for cond in C.GRID_N["FC"]:
                X.write_row("TAB", "N", "FC", cond, s, test_acc=0.95 - 0.05 * ("L" in cond) + 0.01 * rng.standard_normal(),
                            T_hat_frob_err=0.05 + 0.02 * ("J" in cond) + 0.001 * rng.standard_normal())
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
        import aggregate
        aggregate.main(d)
        import pandas as pd
        a = os.path.join(d, "analysis")
        h = pd.read_csv(os.path.join(a, "hypotheses.csv"))
        D = h[(h.dataset == "TAB") & (h.statistic == "D") & (h.rho == 0.4)].iloc[0]
        assert D.conclusion == "positive"
        e = h[(h.dataset == "TAB") & (h.statistic == "errT(J)-errT(L)") & (h.rho == 0.4)].iloc[0]
        assert e.conclusion == "positive"
        v = pd.read_csv(os.path.join(a, "verdicts.csv")).set_index("dataset").loc["TAB"]
        # FC gains on L40 (G(L) > 0) but not on J40 (Delta < 0): supported; no FC* runs: H2* not evaluable
        assert v["H1_0.4"] == "supported" and v["H2_0.4"] == "supported" and v["H2*_0.4"] == "not evaluable"
        assert len(pd.read_csv(os.path.join(a, "that_error.csv"))) == 7
        for f in ("emp_results.tex", "emp_h2.tex", "emp_h3.tex", "emp_h4.tex", "verdict_counts.csv"):
            assert os.path.exists(os.path.join(a, f))


def test_experiment_flow_synthetic(monkeypatch):
    """N, H3 and H4 run units end to end on synthetic TAB/TXT data with a fake encoder."""
    from rnd import data as D, experiment as X
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((26, 16)) * 3
    ytr, yte = rng.integers(0, 26, 600), rng.integers(0, 26, 200)
    monkeypatch.setitem(D.LOADERS, "TAB", lambda: dict(X_train=centers[ytr] + rng.standard_normal((600, 16)),
                                                         y_train=ytr, X_test=centers[yte] + rng.standard_normal((200, 16)),
                                                         y_test=yte))
    words = ["alpha", "beta", "gamma", "delta"]
    yt, ye = rng.integers(0, 4, 400), rng.integers(0, 4, 100)
    monkeypatch.setitem(D.LOADERS, "TXT", lambda: dict(X_train=[f"{words[k]} news item" for k in yt], y_train=yt,
                                                         X_test=[f"{words[k]} news item" for k in ye], y_test=ye))
    monkeypatch.setitem(C.DATASETS, "TAB", dict(C.DATASETS["TAB"], epochs=3))
    monkeypatch.setattr(C, "HEAD", dict(C.HEAD, epochs=5))

    class FakeEnc:
        revision = "fake"

        def __call__(self, texts):
            E = np.array([[hash(t + str(j)) % 1000 / 1000.0 for j in range(8)] for t in texts], dtype=np.float32)
            for i, t in enumerate(texts):
                for k, w in enumerate(words):
                    E[i, k] += 3.0 * (w in t)
            return E / np.linalg.norm(E, axis=1, keepdims=True)

    with tempfile.TemporaryDirectory() as d:
        for attr, sub in (("RUNS", ""), ("NOISE", "noise"), ("PROBS", "probs"), ("EMB", "emb")):
            monkeypatch.setattr(C, attr, os.path.join(d, sub) if sub else d)
        b = X.Bundle("TAB")
        X.run_native(b, 0, ["c0", "L40", "A40", "J40"], ["CE", "FC", "FCstar"], "cpu")
        csvs = sorted(os.path.relpath(p, d) for p in __import__("glob").glob(os.path.join(d, "TAB", "*", "*", "*.csv")))
        assert len(csvs) == 4 + 4 + 2, csvs
        X.run_native(b, 0, ["c0"], ["CE"], "cpu")                       # resume: nothing to do
        bt = X.Bundle("TXT")
        X.run_frozen(bt, 0, ["c0", "L40", "A40", "J40"], ["CE", "FC"], "cpu", FakeEnc())
        h3 = __import__("glob").glob(os.path.join(d, "TXT", "E*", "*", "*.csv"))
        assert len(h3) == 12, len(h3)
        yhat = np.where(rng.random(400) < 0.25, (yt + 1) % 4, yt)
        r_F, T_F = N.annotator_stats(yt, yhat, 4)
        os.makedirs(C.NOISE, exist_ok=True)
        np.savez(os.path.join(C.NOISE, "TAB_F.npz"), yhat=np.where(rng.random(600) < 0.2, (ytr + 3) % 26, ytr),
                 r_F=0.2, T_F=N.annotator_stats(ytr, np.where(rng.random(600) < 0.2, (ytr + 3) % 26, ytr), 26)[1],
                 revision="fake")
        X.run_fm(b, 0, ["S", "C", "F"], ["CE", "FCstar"], "cpu")
        import pandas as pd
        rows = pd.concat([pd.read_csv(p) for p in __import__("glob").glob(os.path.join(d, "TAB", "N", "*", "[SCF]_s0.csv"))])
        assert len(rows) == 6 and set(rows.annotator) == {"fake"}


# ----------------------------------------------------------------------------- E1 / E2 (round 1)
def test_e1_levels_and_conditions():
    for ds, lv in C.LEVELS.items():
        assert set(lv) == {1, 2, 3, 4, 5, 6, 7}, ds
    for ds in ("TS", "AUD"):                                   # lower SNR is stronger
        assert all(C.LEVELS[ds][k] > C.LEVELS[ds][k + 1] for k in range(1, 7))
    for ds in ("TAB", "TXT", "GRA"):
        assert all(C.LEVELS[ds][k] < C.LEVELS[ds][k + 1] for k in range(1, 7))
    assert all(C.LEVELS["IMG"][k]["sigma"] < C.LEVELS["IMG"][k + 1]["sigma"] for k in range(1, 7))
    for lv in (4, 5, 6, 7):
        assert C.CONDITIONS[f"A40-l{lv}"] == (0.0, 0.4, lv) and C.CONDITIONS[f"J40-l{lv}"] == (0.4, 0.4, lv)
        assert f"J40-l{lv}" in C.GRID_E1["FCstar"] and f"A40-l{lv}" not in C.GRID_E1["FCstar"]
        assert all(f"A40-l{lv}" not in g for g in C.GRID_N.values())      # main grid unchanged
    # the training selection S_A does not depend on the level, so A40-l5 corrupts the same examples as A40
    assert np.array_equal(N.select_inputs(1000, 0.4, 2), N.select_inputs(1000, 0.4, 2))


def test_select_test_streams():
    S = N.select_test(1000, 0.4, 3)
    assert len(S) == 400 and np.all(np.diff(S) > 0) and np.array_equal(S, N.select_test(1000, 0.4, 3))
    assert not np.array_equal(S, N.select_inputs(1000, 0.4, 3))       # independent of the training stream
    a = N.test_corruption_rng(3, 0.4, 2).random(5)
    assert np.allclose(a, N.test_corruption_rng(3, 0.4, 2).random(5))
    assert not np.allclose(a, N.corruption_rng(3, 0.4, 2).random(5))


def test_noisy_test_does_not_change_training():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((26, 16)) * 3
    y = rng.integers(0, 26, 1200)
    X = centers[y] + rng.standard_normal((1200, 16))
    Xte = X[900:]
    Xn = N.corrupt_tabular(Xte, N.select_test(300, 0.4, 0), X[:900].std(0), 4.0, N.test_corruption_rng(0, 0.4, 2))
    cfg = dict(C.DATASETS["TAB"], epochs=3)
    r0 = TR.fit_predict(TR.task_tab(X[:900], Xte, "cpu"), y[:900], y[900:], cfg, 0, "cpu")
    r1 = TR.fit_predict(TR.task_tab(X[:900], Xte, "cpu", Xn), y[:900], y[900:], cfg, 0, "cpu")
    r2 = TR.fit_predict(TR.task_tab(X[:900], Xte, "cpu", Xte), y[:900], y[900:], cfg, 0, "cpu")
    assert "test_acc_noisy" not in r0 and r1["test_acc"] == r0["test_acc"]
    assert r2["test_acc_noisy"] == r2["test_acc"] == r0["test_acc"]     # identity corruption: same accuracy
    assert r1["test_acc_noisy"] < r1["test_acc"]
    # graph: evaluation-time corruption of the test nodes only
    yg = rng.integers(0, 7, 300)
    G = (rng.random((300, 50)) < 0.05).astype(np.uint8)
    G[np.arange(300), yg] = 1
    order = np.argsort(yg, kind="stable")
    ei = np.array([[order[i], order[i + 1]] for i in range(299) if yg[order[i]] == yg[order[i + 1]]]).T
    tr, te = np.arange(200), np.arange(200, 300)
    gcfg = dict(C.DATASETS["GRA"], epochs=30)
    g0 = TR.fit_predict_graph(G, yg, yg[tr], ei, tr, te, gcfg, 0, "cpu")
    g1 = TR.fit_predict_graph(G, yg, yg[tr], ei, tr, te, gcfg, 0, "cpu", X_noisy_test=G)
    assert g1["test_acc"] == g0["test_acc"] == g1["test_acc_noisy"]


def test_run_native_noisy_test_synthetic(monkeypatch):
    from rnd import data as D, experiment as X
    import pandas as pd
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((26, 16)) * 3
    ytr, yte = rng.integers(0, 26, 600), rng.integers(0, 26, 200)
    monkeypatch.setitem(D.LOADERS, "TAB", lambda: dict(X_train=centers[ytr] + rng.standard_normal((600, 16)),
                                                         y_train=ytr, X_test=centers[yte] + rng.standard_normal((200, 16)),
                                                         y_test=yte))
    monkeypatch.setitem(C.DATASETS, "TAB", dict(C.DATASETS["TAB"], epochs=3))
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as d2:
        for attr, sub in (("RUNS", ""), ("NOISE", "noise"), ("PROBS", "probs"), ("EMB", "emb")):
            monkeypatch.setattr(C, attr, os.path.join(d, sub) if sub else d)
        b = X.Bundle("TAB")
        X.run_native(b, 0, ["c0", "A40-l5", "J40-l5"], ["CE", "FC", "FCstar"], "cpu", noisy_test=True)
        rows = pd.concat([pd.read_csv(p) for p in __import__("glob").glob(os.path.join(d, "TAB", "N", "*", "*.csv"))])
        assert len(rows) == 2 + 2 + 3 and rows.test_acc_noisy.notna().all()     # FC also on c0
        assert set(rows[rows.condition == "J40-l5"].level) == {5}
        clean = rows.set_index(["method", "condition"]).test_acc
        for attr, sub in (("RUNS", ""), ("NOISE", "noise"), ("PROBS", "probs")):
            monkeypatch.setattr(C, attr, os.path.join(d2, sub) if sub else d2)
        X.run_native(b, 0, ["c0", "A40-l5"], ["CE"], "cpu")               # without --noisy-test
        r2 = pd.concat([pd.read_csv(p) for p in __import__("glob").glob(os.path.join(d2, "TAB", "N", "*", "*.csv"))])
        assert r2.test_acc_noisy.isna().all()
        for _, r in r2.iterrows():                                       # training and clean test unchanged
            assert r.test_acc == clean[(r.method, r.condition)]


def test_aggregate_e12_on_fake_runs(monkeypatch):
    from rnd import experiment as X
    import pandas as pd
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
    import aggregate_e12 as AE
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as grid, tempfile.TemporaryDirectory() as pil, \
            tempfile.TemporaryDirectory() as e12:
        acc = {"c0": 0.95, "L40": 0.80}
        monkeypatch.setattr(C, "RUNS", grid)
        for s in range(5):
            for m, d in (("CE", 0.0), ("FC", 0.0), ("FCstar", 0.05)):
                for c in ("c0", "L40"):
                    if m == "FCstar" and c == "c0":
                        continue
                    X.write_row("TAB", "N", m, c, s, test_acc=acc[c] + d * (c == "L40") + 0.005 * rng.standard_normal(),
                                T_hat_frob_err=0.3 if m == "FC" else None)
        monkeypatch.setattr(C, "RUNS", pil)                       # pilot: RLA(A40-lk) = 0.05, 0.12, 0.2, 0.3
        for s in C.E1_PILOT_SEEDS:
            X.write_row("TAB", "N", "CE", "c0", s, test_acc=0.95)
            for lv, r in zip((4, 5, 6, 7), (0.05, 0.12, 0.2, 0.3)):
                X.write_row("TAB", "N", "CE", f"A40-l{lv}", s, test_acc=0.95 * (1 - r))
        p = AE.pilot(pil, grid).set_index("dataset").loc["TAB"]
        assert p.level == 6 and p.reached                             # RLA(L40) ~ 0.158: l6 is the mildest above
        assert AE.pick_level({4: 0.0, 5: 0.01, 6: 0.02, 7: 0.03}, 0.2) == (7, False)
        monkeypatch.setitem(C.E1_LEVEL, "TAB", 6)
        monkeypatch.setattr(C, "RUNS", e12)
        for s in range(5):
            X.write_row("TAB", "N", "CE", "A40-l6", s, test_acc=0.76 + 0.005 * rng.standard_normal())
            X.write_row("TAB", "N", "FC", "A40-l6", s, test_acc=0.76 + 0.005 * rng.standard_normal())
            X.write_row("TAB", "N", "CE", "J40-l6", s, test_acc=0.60 + 0.005 * rng.standard_normal())
            X.write_row("TAB", "N", "FC", "J40-l6", s, test_acc=0.60 + 0.005 * rng.standard_normal(), T_hat_frob_err=0.5)
            X.write_row("TAB", "N", "FCstar", "J40-l6", s, test_acc=0.60 + 0.005 * rng.standard_normal())
            for c, a in (("c0", 0.95), ("L40", 0.80), ("A40", 0.94), ("J40", 0.79)):
                X.write_row("TAB", "N", "CE", c, s, test_acc=a, test_acc_noisy=a - 0.10 + 0.08 * (c in ("A40", "J40"))
                            + 0.003 * rng.standard_normal())
        AE.e1_e2(e12, grid)
        h1 = pd.read_csv(os.path.join(e12, "analysis", "e1.csv")).set_index("statistic")
        assert h1.loc["Delta*_s", "conclusion"] == "negative" and h1.loc["G*(L40)", "conclusion"] == "positive"
        assert h1.loc["errT(J40-s)-errT(L40)", "conclusion"] == "positive"
        v = pd.read_csv(os.path.join(e12, "analysis", "e12_verdicts.csv"))
        v1 = v[v.experiment == "E1"].iloc[0]
        assert v1.H2star_s == "supported" and v1.H2_s == "not evaluable" and v1.calibrated == "yes"
        h2 = pd.read_csv(os.path.join(e12, "analysis", "e2.csv")).set_index("statistic")
        assert h2.loc["D~", "conclusion"] == "positive" and h2.loc["Acc~(c0)-Acc(c0)", "conclusion"] == "negative"
        assert v[v.experiment == "E2"].iloc[0].H1_noisy_test == "supported"
        assert os.path.exists(os.path.join(e12, "analysis", "emp_e12.tex"))
        import tables
        tables.e12(os.path.join(e12, "analysis"), e12)
        t = open(os.path.join(e12, "emp_e12.tex")).read()
        assert r"\label{tab:emp-e12}" in t and t.count(r"\\") == 8 and "Tabular" in t


# ----------------------------------------------------------------------------- round 2
def _synthetic_tab(monkeypatch, name="TAB", K=26, n=600, nte=200, d=16):
    from rnd import data as D
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((K, d)) * 3
    ytr, yte = rng.integers(0, K, n), rng.integers(0, K, nte)
    monkeypatch.setitem(D.LOADERS, name, lambda: dict(X_train=centers[ytr] + rng.standard_normal((n, d)), y_train=ytr,
                                                       X_test=centers[yte] + rng.standard_normal((nte, d)), y_test=yte))
    monkeypatch.setitem(C.DATASETS, name, dict(C.DATASETS[name], epochs=4))


def _runs_tmp(monkeypatch, d):
    for attr, sub in (("RUNS", ""), ("NOISE", "noise"), ("PROBS", "probs"), ("EMB", "emb")):
        monkeypatch.setattr(C, attr, os.path.join(d, sub) if sub else d)


def test_v14_second_dataset_kind(monkeypatch):
    from rnd import experiment as X
    import pandas as pd
    _synthetic_tab(monkeypatch, "TAB2", K=7, d=54)
    with tempfile.TemporaryDirectory() as d:
        _runs_tmp(monkeypatch, d)
        b = X.Bundle("TAB2")
        assert b.kind == "TAB" and b.K == 7 and hasattr(b, "sd_clean")
        X.run_native(b, 0, ["c0", "L40", "A40"], ["CE", "FC"], "cpu")
        rows = pd.concat([pd.read_csv(p) for p in __import__("glob").glob(os.path.join(d, "TAB2", "N", "*", "*.csv"))])
        assert len(rows) == 6 and set(rows.dataset) == {"TAB2"}


def test_v14_fit_x_matches_fit_predict():
    from rnd import v14
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((26, 16)) * 3
    y = rng.integers(0, 26, 800)
    X = centers[y] + rng.standard_normal((800, 16))
    task = TR.task_tab(X[:600], X[600:], "cpu")
    cfg = dict(C.DATASETS["TAB"], epochs=3)
    r0 = TR.fit_predict(task, y[:600], y[600:], cfg, 0, "cpu")
    r1 = v14.fit_x(dict(task, K=26), y[:600], y[600:], cfg, 0, "cpu", loss_at=(1, 3))
    assert r1["test_acc"] == r0["test_acc"] and set(r1["losses"]) == {1, 3} and r1["losses"][3].shape == (600,)
    use, val = v14.es_split(600, 0)
    assert len(val) == 60 and len(np.intersect1d(use, val)) == 0
    r2 = v14.fit_x(dict(task, K=26), y[:600], y[600:], cfg, 0, "cpu", use_idx=use, val_idx=val)
    assert 1 <= r2["best_epoch"] <= 3 and r2["curve"].shape == (3, 3) and 0 <= r2["test_acc_es"] <= 1
    r3 = v14.fit_x(dict(task, K=26), y[:600], y[600:], cfg, 0, "cpu", vol=True)
    assert np.allclose(r3["T_vol"].sum(1), 1) and r3["T_vol"].shape == (26, 26)
    r4 = v14.fit_x(dict(task, K=26), y[:600], y[600:], cfg, 0, "cpu", cot_rate=0.4)
    assert 0 <= r4["test_acc"] <= 1


def test_v14_h1_controls(monkeypatch):
    from rnd import experiment as X, v14
    _synthetic_tab(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        _runs_tmp(monkeypatch, d)
        b = X.Bundle("TAB")
        Xx, info = v14.x40_inputs(b, 0)
        S = info["S_A"]
        assert len(S) == 240 and np.all(b.y_train[info["partner"]] != b.y_train[S])
        keep = np.setdiff1d(np.arange(b.n), S)
        assert np.array_equal(Xx[keep], b.d["X_train"][keep]) and not np.allclose(Xx[S], b.d["X_train"][S])
        Xa, _ = v14.at40_inputs(b, 0)
        frac = np.mean(Xa != b.d["X_train"], axis=0)
        assert np.all(np.abs(frac - 0.4) < 0.01)                     # 40% of the values of every attribute
        v14.run_block(b, 0, "LS", ["c0", "L40"], ["CE"], "cpu")
        v14.run_block(b, 0, "H1C", ["R40", "X40", "AT40"], ["CE"], "cpu")
        v14.run_block(b, 0, "H2P", ["L40"], ["VOL", "COT"], "cpu")
        v14.run_block(b, 0, "ES", ["c0", "L40"], ["CE", "FCstar"], "cpu")
        import pandas as pd
        rows = pd.concat([pd.read_csv(p) for p in __import__("glob").glob(os.path.join(d, "TAB", "*", "*", "*.csv"))])
        got = set(zip(rows.pipeline, rows.method, rows.condition))
        assert got == {("N", "CE", "c0"), ("N", "CE", "L40"), ("N", "CE", "R40"), ("N", "CE", "X40"), ("N", "CE", "AT40"),
                       ("N", "VOL", "L40"), ("N", "COT", "L40"), ("ES", "CE", "c0"), ("ES", "CE", "L40"),
                       ("ES", "FCstar", "L40")}, got
        r40 = rows[rows.condition == "R40"].iloc[0]
        assert r40.n_train == b.n - 240
        ls = np.load(os.path.join(d, rows[(rows.condition == "L40") & (rows.pipeline == "N") & (rows.method == "CE")]
                                  .extra_path.iloc[0]))
        assert {"loss_ep1", "loss_ep2", "loss_ep4", "S_L"} <= set(ls.files)
        es = rows[(rows.pipeline == "ES")]
        assert es.test_acc_es.notna().all() and es.best_epoch.notna().all()


def test_v14_x40_text_image_audio_graph(monkeypatch):
    from rnd import data as D, experiment as X, v14
    rng = np.random.default_rng(0)
    words = ["alpha", "beta", "gamma", "delta"]
    yt = rng.integers(0, 4, 100)
    monkeypatch.setitem(D.LOADERS, "TXT", lambda: dict(X_train=[f"{words[k]} news item about {words[k]}" for k in yt],
                                                       y_train=yt, X_test=["alpha news"], y_test=np.array([0])))
    b = X.Bundle("TXT")
    Xx, info = v14.x40_inputs(b, 1)
    changed = [i for i in range(100) if Xx[i] != b.d["X_train"][i]]
    assert set(changed) <= set(info["S_A"]) and len(changed) > 20
    yi = rng.integers(0, 10, 50)
    monkeypatch.setitem(D.LOADERS, "IMG", lambda: dict(X_train=rng.integers(0, 256, (50, 32, 32, 3), dtype=np.uint8),
                                                       y_train=yi, X_test=np.zeros((2, 32, 32, 3), np.uint8),
                                                       y_test=np.array([0, 1])))
    bi = X.Bundle("IMG")
    Xi, info = v14.x40_inputs(bi, 0)
    assert Xi.dtype == np.uint8 and len(info["S_A"]) == 20
    ya = rng.integers(0, 10, 30)
    monkeypatch.setitem(D.LOADERS, "AUD", lambda: dict(X_train=[rng.standard_normal(800).astype(np.float32) for _ in ya],
                                                       y_train=ya, X_test=[np.zeros(800, np.float32)], y_test=np.array([0])))
    ba = X.Bundle("AUD")
    Xa, info = v14.x40_inputs(ba, 0)
    i = info["S_A"][0]
    assert len(Xa[i]) == 800 and not np.allclose(Xa[i], ba.d["X_train"][i])
    yg = rng.integers(0, 7, 60)
    G = (rng.random((60, 30)) < 0.2).astype(np.uint8)
    ei = np.array([[i, i + 1] for i in range(59)]).T
    monkeypatch.setitem(D.LOADERS, "GRA", lambda: dict(X=G, y=yg, edge_index=ei, train_idx=np.arange(40),
                                                       test_idx=np.arange(40, 60)))
    bg = X.Bundle("GRA")
    Xg, info = v14.x40_inputs(bg, 0)
    assert Xg.dtype == np.uint8 and np.array_equal(Xg[40:], G[40:])


def test_v14_graph_blocks(monkeypatch):
    from rnd import data as D, experiment as X, v14
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, 300)
    G = (rng.random((300, 50)) < 0.05).astype(np.uint8)
    G[np.arange(300), y] = 1
    order = np.argsort(y, kind="stable")
    ei = np.array([[order[i], order[i + 1]] for i in range(299) if y[order[i]] == y[order[i + 1]]]).T
    monkeypatch.setitem(D.LOADERS, "GRA", lambda: dict(X=G, y=y, edge_index=ei, train_idx=np.arange(200),
                                                       test_idx=np.arange(200, 300)))
    monkeypatch.setitem(C.DATASETS, "GRA", dict(C.DATASETS["GRA"], epochs=20))
    with tempfile.TemporaryDirectory() as d:
        _runs_tmp(monkeypatch, d)
        b = X.Bundle("GRA")
        for blk, conds, ms in (("LS", ["L40", "A40"], ["CE"]), ("H1C", ["R40", "X40"], ["CE"]),
                               ("H2P", ["J40"], ["VOL", "COT"]), ("ES", ["L40"], ["CE", "FCstar"]),
                               ("CD", ["L40", "L40e400"], ["CE", "FCstar"])):
            v14.run_block(b, 0, blk, conds, ms, "cpu")
        import pandas as pd
        rows = pd.concat([pd.read_csv(p) for p in __import__("glob").glob(os.path.join(d, "GRA", "*", "*", "*.csv"))])
        assert len(rows) == 2 + 2 + 2 + 2 + 4, len(rows)
        cd = rows[(rows.pipeline == "CD") & (rows.condition == "L40e400")]
        assert set(cd.epochs) == {400}
        c = np.load(os.path.join(d, cd.extra_path.iloc[0]))["curve"]
        assert c.shape == (400, 5)


def test_v14_cifar10n_labels(monkeypatch):
    from rnd import data as D, experiment as X, v14
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 500)
    worst = np.where(rng.random(500) < 0.4, (y + 1) % 10, y)
    monkeypatch.setitem(D.LOADERS, "IMG", lambda: dict(X_train=np.zeros((500, 32, 32, 3), np.uint8), y_train=y,
                                                       X_test=np.zeros((2, 32, 32, 3), np.uint8), y_test=np.array([0, 1])))
    monkeypatch.setattr(D, "load_cifar10n", lambda: dict(clean_label=y, aggre_label=y, random_label1=worst, worse_label=worst))
    b = X.Bundle("IMG")
    yn, S, T, rate = v14.cn_labels(b, "Nworst", 0)
    assert np.array_equal(yn, worst) and np.allclose(T.sum(1), 1) and abs(rate - np.mean(worst != y)) < 1e-12
    ys, S2, T2, rate2 = v14.cn_labels(b, "Sworst", 0)
    assert rate2 == rate and np.sum(ys != y) == round(rate * 500)


# ----------------------------------------------------------------------------- round 2 H3C
def test_h3c_mlp_head_learns():
    from rnd import h3c
    rng = np.random.default_rng(0)
    K, d = 3, 16
    mu = rng.normal(size=(K, d)) * 3
    y = rng.integers(0, K, 600)
    E = (mu[y] + rng.normal(size=(600, d))).astype(np.float32)
    task = h3c.task_mlp_head(E[:500], E[500:], K, "cpu")
    assert isinstance(task["model_fn"](), h3c.MLPHead)
    cfg = dict(opt="adam", lr=1e-3, wd=0.0, batch=64, epochs=20)
    r = TR.fit_predict(task, y[:500], y[500:], cfg, 0, "cpu")
    assert r["test_acc"] > 0.9


def test_h3c_ft_inputs_match_en_corruption():
    from rnd import h3c

    class Fake:
        pass
    b = Fake()
    b.ds, b.n = "TXT", 60
    b.d = {"X_train": [f"sample text number {i} with some words" for i in range(60)]}
    X, S_A = h3c.ft_inputs(b, 3, 0.4, 2)
    assert np.array_equal(S_A, N.select_inputs(60, 0.4, 3)) and len(S_A) == 24
    ref = N.corrupt_texts([b.d["X_train"][i] for i in S_A], np.arange(len(S_A)), C.LEVELS["TXT"][2],
                          N.corruption_rng(3, 0.4, 2))
    assert [X[i] for i in S_A] == ref
    rest = np.setdiff1d(np.arange(60), S_A)
    assert all(X[i] == b.d["X_train"][i] for i in rest)
    X0, S0 = h3c.ft_inputs(b, 3, 0.0, 2)
    assert X0 == b.d["X_train"] and len(S0) == 0


# ----------------------------------------------------------------------------- round 3
def test_v15_es_probs_best_epoch():
    from rnd import v14
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((26, 16)) * 3
    y = rng.integers(0, 26, 800)
    X = centers[y] + rng.standard_normal((800, 16))
    task = dict(TR.task_tab(X[:600], X[600:], "cpu"), K=26)
    use, val = v14.es_split(600, 0)
    cfg1 = dict(C.DATASETS["TAB"], epochs=1)
    r = v14.fit_x(task, y[:600], y[600:], cfg1, 0, "cpu", use_idx=use, val_idx=val, es_probs=True,
                  need_train_probs=True)
    assert r["best_epoch"] == 1 and r["train_probs_es"].shape == (600, 26)
    assert np.allclose(r["train_probs_es"], r["train_probs"], atol=1e-6)        # one epoch: best = last
    cfg3 = dict(C.DATASETS["TAB"], epochs=3)
    r3 = v14.fit_x(task, y[:600], y[600:], cfg3, 0, "cpu", use_idx=use, val_idx=val, es_probs=True)
    assert np.allclose(r3["train_probs_es"].sum(1), 1, atol=1e-4)


def test_v15_te_tab_and_graph(monkeypatch):
    from rnd import data as D, experiment as X, v15
    import pandas as pd
    _synthetic_tab(monkeypatch, "TAB", K=26, d=16)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, 300)
    G = (rng.random((300, 50)) < 0.05).astype(np.uint8)
    G[np.arange(300), y] = 1
    order = np.argsort(y, kind="stable")
    ei = np.array([[order[i], order[i + 1]] for i in range(299) if y[order[i]] == y[order[i + 1]]]).T
    monkeypatch.setitem(D.LOADERS, "GRA", lambda: dict(X=G, y=y, edge_index=ei, train_idx=np.arange(200),
                                                       test_idx=np.arange(200, 300)))
    monkeypatch.setitem(C.DATASETS, "GRA", dict(C.DATASETS["GRA"], epochs=20))
    with tempfile.TemporaryDirectory() as d:
        _runs_tmp(monkeypatch, d)
        for ds in ("TAB", "GRA"):
            b = X.Bundle(ds)
            v15.run_te(b, 0, ["L40", "J40"], "cpu")
            v15.run_te(b, 0, ["L40"], "cpu")                                 # resume: nothing to do
            rows = pd.concat([pd.read_csv(p) for p in __import__("glob").glob(os.path.join(d, ds, "TE", "*", "*.csv"))])
            assert len(rows) == 4 and set(rows.method) == {"CEES", "FCES"}
            assert rows.T_hat_frob_err.notna().all() and rows[rows.method == "CEES"].best_epoch.notna().all()
            assert os.path.exists(os.path.join(d, "probs", ds, "TE", "L40_s0_That.npy"))


def test_v15_h4x_flow(monkeypatch):
    from rnd import experiment as X, v15
    import pandas as pd
    _synthetic_tab(monkeypatch, "TAB", K=26, d=16)
    with tempfile.TemporaryDirectory() as d:
        _runs_tmp(monkeypatch, d)
        b = X.Bundle("TAB")
        rng = np.random.default_rng(1)
        yhat = np.where(rng.random(b.n) < 0.2, (b.y_train + 3) % 26, b.y_train)
        r_F, T_F = N.annotator_stats(b.y_train, yhat, 26)
        os.makedirs(C.NOISE, exist_ok=True)
        np.savez(os.path.join(C.NOISE, "TAB_F1.npz"), yhat=yhat, r_F=r_F, T_F=T_F, revision="fake1")
        v15.run_h4x(b, 0, ["S1", "C1", "F1"], ["CE", "FCstar"], "cpu")
        rows = pd.concat([pd.read_csv(p) for p in __import__("glob").glob(os.path.join(d, "TAB", "N", "*", "*1_s0.csv"))])
        assert len(rows) == 6 and set(rows.annotator) == {"fake1"} and set(rows.condition) == {"S1", "C1", "F1"}
        assert np.allclose(rows.rho_L, r_F)


def test_v15_aud_folds(monkeypatch):
    from rnd import data as D
    assert len(C.AUD_FOLDS) == 10 and all(C.KIND[k] == "AUD" for k in C.AUD_FOLDS)
    assert all(k in D.LOADERS for k in C.AUD_FOLDS) and C.LEVELS["AUDF4"] == C.LEVELS["AUD"]
    with tempfile.TemporaryDirectory() as d:
        n = 50
        fold = np.repeat(np.arange(1, 11), 5)
        p = os.path.join(d, "w.npz")
        np.savez(p, wave=np.arange(n * 3, dtype=np.float32), offsets=np.arange(0, n * 3 + 1, 3), fold=fold,
                 label=np.arange(n) % 10)
        monkeypatch.setattr(D, "_wave_cache", lambda sr: p)
        for k in (1, 7, 10):
            z = D.LOADERS[f"AUDF{k}"]()
            assert np.all(fold[z["test_rows"]] == k) and np.all(fold[z["train_rows"]] != k)
            assert len(z["train_rows"]) + len(z["test_rows"]) == n
        z10, zg = D.load_aud(test_fold=10), D.load_aud()
        assert np.array_equal(z10["train_rows"], zg["train_rows"])


def test_v15_reinit_bert():
    from rnd import h3c
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Embedding(100, 64, padding_idx=0), torch.nn.Linear(64, 64), torch.nn.LayerNorm(64))
    w0 = m[1].weight.detach().clone()
    with torch.no_grad():
        m[2].weight.fill_(3.0)
    torch.manual_seed(0)
    assert h3c.reinit_bert_(m) == 3
    assert not torch.allclose(w0, m[1].weight) and abs(float(m[1].weight.std()) - 0.02) < 0.003
    assert torch.all(m[2].weight == 1) and torch.all(m[0].weight[0] == 0)


# ----------------------------------------------------------------------------- round 4
def test_v16_pair_and_dependent_noise():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 6, 3000)
    for rho in (0.2, 0.4):
        _, S_sym, _ = N.label_noise(y, 6, rho, 3)
        yp, S, T = N.pair_label_noise(y, 6, rho, 3)
        assert np.array_equal(S, S_sym)                                  # same selection as L_rho
        assert np.all(yp[S] == (y[S] + 1) % 6) and np.sum(yp != y) == round(rho * len(y))
        assert np.allclose(T.sum(1), 1) and np.isclose(T[0, 1], rho) and np.isclose(T[0, 0], 1 - rho)
        S_A = N.select_inputs(len(y), rho, 3)
        yd, SL, Td = N.dependent_label_noise(y, 6, S_A, rho, 3)
        assert np.array_equal(SL, S_A) and np.all(yd[S_A] != y[S_A])
        assert np.sum(yd != y) == len(S_A) and np.allclose(Td, N.symmetric_T(6, rho))


def test_v16_h2b_and_sc_flow(monkeypatch):
    from rnd import experiment as X, v16
    import pandas as pd
    _synthetic_tab(monkeypatch, "TAB", K=26, d=16)
    with tempfile.TemporaryDirectory() as d:
        _runs_tmp(monkeypatch, d)
        b = X.Bundle("TAB")
        v16.run_h2b(b, 0, ["P40", "JD40"], ["CE", "FC", "FCstar"], "cpu")
        v16.run_sc(b, 0, ["A40", "AT40", "J40"], ["CE", "FCstar"], "cpu")
        g = __import__("glob").glob
        h = pd.concat([pd.read_csv(p) for p in g(os.path.join(d, "TAB", "N", "*", "*.csv"))])
        assert len(h) == 6 and set(h.condition) == {"P40", "JD40"}
        jd = h[(h.condition == "JD40") & (h.method == "CE")].iloc[0]
        assert jd.n_overlap == jd.n_input_noisy == jd.n_label_noisy             # labels flipped exactly on S_A
        sc = pd.concat([pd.read_csv(p) for p in g(os.path.join(d, "TAB", "SC", "*", "*.csv"))])
        assert set(zip(sc.method, sc.condition)) == {("CE", "A40"), ("CE", "AT40"), ("CE", "J40"), ("FCstar", "J40")}


def test_v16_sc_uses_clean_statistics():
    rng = np.random.default_rng(0)
    Xc = rng.normal(size=(200, 4))
    Xn = Xc * 3.0
    t = TR.task_tab(Xn, Xc[:10], "cpu", K=3, stats_X=Xc)
    mu, sd = Xc.mean(0), Xc.std(0)
    assert np.allclose(t["get"]("test", np.arange(10), False, None).numpy(), (Xc[:10] - mu) / sd, atol=1e-5)


# ----------------------------------------------------------------------------- round 5 OOS
def test_crossfit_folds_stratified_and_deterministic():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 7, 1001)
    f1 = N.crossfit_folds(y, 7, 0.4, 0.4, 2)
    f2 = N.crossfit_folds(y, 7, 0.4, 0.4, 2)
    assert np.array_equal(f1, f2) and set(np.unique(f1)) == {0, 1}
    for k in range(7):
        nk = np.sum(y == k)
        assert np.sum((y == k) & (f1 == 0)) == nk // 2
    assert not np.array_equal(f1, N.crossfit_folds(y, 7, 0.2, 0.0, 2))


def test_estimate_T_classwise_recovers_T_and_avoids_other_classes():
    K, n_per = 100, 50                       # many classes: the global 97th percentile leaves class i
    T = N.symmetric_T(K, 0.4)
    y_true = np.repeat(np.arange(K), n_per)
    probs = T[y_true].copy()                 # perfect out-of-sample predictions: p_hat(x) = T[y(x)]
    y_noisy = y_true.copy()                  # anchors exist in every noisy class
    Tc = TR.estimate_T_classwise(probs, y_noisy)
    assert np.allclose(Tc, T)
    Tg = TR.estimate_T(probs)
    assert not np.allclose(Tg, T)            # the global rule picks examples of other classes
    assert np.allclose(TR.estimate_T_argmax(probs), T)


def test_fit_predict_test_probs_on_subtask():
    from rnd import v17
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    task = TR.task_tab(X, X[:20], "cpu", K=2)
    I_tr, I_ho = np.arange(0, 200, 2), np.arange(1, 200, 2)
    sub = v17._subtask(task, I_tr, I_ho)
    cfg = dict(C.DATASETS["TAB"], K=2, epochs=2)
    r = TR.fit_predict(sub, y[I_tr], y[I_ho], cfg, 0, "cpu", need_test_probs=True)
    assert r["test_probs"].shape == (100, 2) and np.allclose(r["test_probs"].sum(1), 1, atol=1e-5)


# ----------------------------------------------------------------------------- round 5 OOSC
def test_fit_inverse_temperature_recovers_scale():
    from rnd import v18
    rng = np.random.default_rng(0)
    K, n = 5, 20000
    z_true = rng.normal(scale=2.0, size=(n, K))
    p = v18._softmax(z_true)
    y = np.array([rng.choice(K, p=pi) for pi in p])          # labels drawn from the true probabilities
    beta = v18.fit_inverse_temperature(3.0 * z_true, y)     # overconfident logits: beta should be close to 1/3
    assert abs(beta - 1 / 3) < 0.02
    assert v18.noisy_nll(3.0 * z_true, y, beta) <= v18.noisy_nll(3.0 * z_true, y, 1.0)


def test_calibrate_per_fold_and_shapes():
    from rnd import v18
    rng = np.random.default_rng(1)
    K, n = 4, 4000
    z = rng.normal(size=(n, K))
    y = rng.integers(0, K, n)
    fold = N.crossfit_folds(y, K, 0.4, 0.0, 0)
    p = v18._softmax(4.0 * z)
    p_cal, betas, before, after = v18.calibrate(p.astype(np.float32), y, fold)
    assert p_cal.shape == (n, K) and np.allclose(p_cal.sum(1), 1)
    assert len(betas) == 2 and all(a <= b + 1e-12 for a, b in zip(after, before))
    T = TR.estimate_T_classwise(p_cal, y)
    assert T.shape == (K, K)


def test_calibrate_handles_zero_probabilities():
    from rnd import v18
    p = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]] * 50, dtype=np.float32)
    y = np.array([0, 1, 1, 0] * 50)
    fold = np.array([0, 1] * 100)
    p_cal, betas, _, _ = v18.calibrate(p, y, fold)
    assert np.all(np.isfinite(p_cal)) and all(np.isfinite(betas))
