"""Unit / integration tests for router.py (T4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from content import multiplex_streams
from lockin import SAMPLES_PER_SYMBOL
from router import (
    DEFAULT_READOUT_PATH,
    DEFAULT_WEIGHTS_PATH,
    FEATURE_DIM,
    HIDDEN,
    N_HIST_BINS,
    Router,
    evaluate_router_path_acc,
    fit_patch_readouts,
    load_frozen_readouts,
    load_frozen_router,
    make_clean_router_dataset,
    phase_histogram,
    router_features,
    sha256_file,
    train_and_freeze,
)
from sheet import N_PATCHES, Sheet

RESULTS = Path(__file__).resolve().parents[1] / "results"
WEIGHTS = RESULTS / "router_weights.npz"
READOUTS = RESULTS / "router_readouts.npz"
STATS = RESULTS / "T4_router_stats.txt"


def test_phase_histogram_shape_and_norm():
    w = multiplex_streams(SAMPLES_PER_SYMBOL, {0: 1.0})
    h = phase_histogram(w, n_bins=32)
    assert h.shape == (32,)
    assert abs(h.sum() - 1.0) < 1e-9
    assert np.all(h >= 0)


def test_router_features_dim():
    w = multiplex_streams(SAMPLES_PER_SYMBOL, {3: 1.0, 5: 1.0})
    f = router_features(w, d_star=3)
    assert f.shape == (FEATURE_DIM,)
    assert FEATURE_DIM == N_HIST_BINS + N_PATCHES == 32
    assert abs(f[:N_HIST_BINS].sum() - 1.0) < 1e-9
    assert f[N_HIST_BINS + 3] == 1.0
    assert f[N_HIST_BINS:].sum() == 1.0


def test_param_count_and_macs():
    r = Router(seed=0)
    # (64, 32) + 64 + (8, 64) + 8 = 2048 + 64 + 512 + 8 = 2632
    assert r.param_count() == HIDDEN * FEATURE_DIM + HIDDEN + N_PATCHES * HIDDEN + N_PATCHES
    assert r.macs_per_decision() == FEATURE_DIM * HIDDEN + HIDDEN * N_PATCHES


def test_sheet_gain_mode_fixed_does_not_break_default():
    s0 = Sheet(seed=0)
    assert s0.gain_mode == "oscillatory"
    g0 = s0.gain.copy()
    assert np.all(g0 >= 0.0) and np.all(g0 <= 1.0)
    s1 = Sheet(seed=0, gain_mode="fixed")
    assert np.allclose(s1.gain, 1.0)
    s1.step(0.5, None)
    assert np.allclose(s1.gain, 1.0)


def test_make_control_opens_exactly_topk():
    r = Router(seed=0)
    ctrl = r.make_control([2, 5])
    assert ctrl.pilot_signal is None
    assert ctrl.inhibition[2] == 0.0 and ctrl.inhibition[5] == 0.0
    assert ctrl.inhibition.sum() == N_PATCHES - 2


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory):
    """Train once into results/ (shared) if missing, else reuse."""
    if not WEIGHTS.exists() or not READOUTS.exists():
        train_and_freeze(
            n_train=2000,
            n_eval=300,
            seed=20260921,
            weights_path=WEIGHTS,
            readout_path=READOUTS,
            stats_path=STATS,
            verbose=False,
        )
    return WEIGHTS, READOUTS


def test_frozen_router_clean_acc(trained_artifacts):
    weights_path, readout_path = trained_artifacts
    router = load_frozen_router(weights_path)
    readouts = load_frozen_readouts(readout_path)
    assert router.frozen
    metrics = evaluate_router_path_acc(
        router, readouts, n_symbols=400, seed=99, use_mlp=True
    )
    assert metrics["routing_acc"] >= 0.95, metrics
    assert metrics["acc"] >= 0.95, metrics
    digest = sha256_file(weights_path)
    assert len(digest) == 64
    assert STATS.exists()
    text = STATS.read_text()
    assert "param_count" in text or "param_count:" in text
    assert "MACs_per_decision" in text
    assert digest[:8] in text or "weights_sha256" in text


def test_save_load_roundtrip(tmp_path):
    feats, labels, _ = make_clean_router_dataset(256, seed=3)
    r = Router(seed=3)
    r.train(feats, labels, n_epochs=20, batch_size=32, lr=0.1, seed=3)
    r.freeze()
    path = tmp_path / "w.npz"
    h1 = r.save(path)
    r2 = Router.load(path)
    x = feats[0]
    assert np.allclose(r.probs(x), r2.probs(x), atol=1e-9)
    assert h1 == sha256_file(path)
