"""
Spec for Algorithm 1's Main (paper lines 59-69):

    function Main()
        chi_i <- Preprocessing(D)
        theta0 <- BIPInitilize(D)
        while Optimize do
            ComputeLoss
        # BIP_PosteriorEstimate(theta_hat)   # @todo  <- unresolved in the paper itself
        return null

The paper's own pseudocode marks the final posterior-estimate step
`# @todo`, so Main() is acknowledged-incomplete even in the spec -- these
tests only check the orchestration wiring it DOES define: Preprocessing
runs on every snapshot, BIPInitialize produces theta0, Optimize is called
with that theta0, and its result is what Main() returns. They mock out
preprocessing/bip_initialize/optimize so this doesn't require any of the
underlying (still-unimplemented) numerics to be real -- see
multsc_grn_inference.main for the target contract (main() raises
NotImplementedError for now).
"""
from __future__ import annotations

import numpy as np
import pytest

import multsc_grn_inference.main as main_module
from multsc_grn_inference.theta import Theta

G = 2
N_SNAPS = 3


def _fake_snapshots():
    rng = np.random.default_rng(0)
    return [rng.normal(size=(50, G)) for _ in range(N_SNAPS)]


def _fake_mu():
    return [np.array([2.0, 2.0])] * (N_SNAPS - 1)


def test_main_preprocesses_every_snapshot(monkeypatch):
    D = _fake_snapshots()
    seen_inputs = []

    def fake_preprocessing(X):
        seen_inputs.append(X)
        return f"chi_{len(seen_inputs) - 1}"

    monkeypatch.setattr(main_module, "preprocessing", fake_preprocessing)
    monkeypatch.setattr(main_module, "bip_initialize", lambda D, mu, dt: "theta0")
    monkeypatch.setattr(main_module, "optimize", lambda theta0, D, chi, dt: "theta_hat")

    main_module.main(D, _fake_mu(), dt=0.3)

    assert len(seen_inputs) == len(D)
    for seen, expected in zip(seen_inputs, D):
        np.testing.assert_array_equal(seen, expected)


def test_main_initialises_theta_via_bip_before_optimizing(monkeypatch):
    D = _fake_snapshots()
    mu = _fake_mu()
    calls = []

    monkeypatch.setattr(main_module, "preprocessing", lambda X: "chi")
    monkeypatch.setattr(main_module, "bip_initialize", lambda D, mu, dt: calls.append(("bip", D, mu, dt)) or "theta0")
    monkeypatch.setattr(main_module, "optimize", lambda theta0, D, chi, dt: calls.append(("optimize", theta0, D, chi, dt)) or "theta_hat")

    main_module.main(D, mu, dt=0.3)

    assert calls[0][0] == "bip"
    assert calls[1][0] == "optimize"
    assert calls[1][1] == "theta0", "Optimize must be initialised at BIPInitialize's theta0"


def test_main_returns_optimizes_result(monkeypatch):
    D = _fake_snapshots()
    theta_hat = Theta(A=np.eye(G), mu=_fake_mu(), sigma=0.2)

    monkeypatch.setattr(main_module, "preprocessing", lambda X: "chi")
    monkeypatch.setattr(main_module, "bip_initialize", lambda D, mu, dt: "theta0")
    monkeypatch.setattr(main_module, "optimize", lambda theta0, D, chi, dt: theta_hat)

    result = main_module.main(D, _fake_mu(), dt=0.3)

    assert result is theta_hat
