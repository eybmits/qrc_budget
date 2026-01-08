#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QRC Trajectory-vs-EV Paper Evaluation (Mackey–Glass, TFIM, weak measurement)
===========================================================================

This single script reproduces the *full paper-style evaluation* for a
physically-motivated quantum reservoir computing (QRC) model:

  • 4-qubit TFIM-like Hamiltonian (ring ZZ coupling + transverse field disorder)
  • Exact unitary propagation per time step
  • Continuous weak measurement (discrete sampling) producing shot trajectories
  • Completely-positive (CP) measurement update for Pauli-Z (backaction)

We compare two *budget-matched* learning pipelines using the SAME simulated shots:

EV baseline
  - average currents across shots -> tapped-delay features -> ridge regression

Trajectory-level
  - tapped-delay features per shot -> stack all shots for ridge regression
  - test-time prediction averaging across shots

Stopping condition (already met by the provided "winning config"):
  - NRMSE_EV < 1.0
  - NRMSE_traj < NRMSE_EV

This script:
  1) evaluates the winning configuration (single run + time-series plot)
  2) runs ridge-λ sweeps (shows regime dependence / tradeoff)
  3) runs lag/window sweeps (sample-ratio heuristic)
  4) runs multi-seed robustness experiments
  5) runs a backaction ablation (backaction ON vs OFF)
  6) generates a noise×shots phase diagram
  7) writes all results to CSV and all figures to PNG
  8) builds a 4-panel "paper figure" as a collage (no matplotlib subplots)

Outputs are written to:  ./qrc_mg_paper_artifacts/

Dependencies:
  numpy, pandas, matplotlib, scikit-learn, pillow
"""

from __future__ import annotations

import os
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from PIL import Image


# -----------------------------
# Metrics / utilities
# -----------------------------

def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NRMSE = RMSE / std(y_true). Lower is better."""
    return math.sqrt(mean_squared_error(y_true, y_pred)) / (np.std(y_true) + 1e-12)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Mackey–Glass generator
# -----------------------------

def generate_mackey_glass(
    t_max: int = 10000,
    tau: int = 17,
    seed: int = 42,
    dt: float = 1.0,
    burn_in: int = 800,
) -> np.ndarray:
    """
    Mackey–Glass delay equation (Euler discretization).
    Returns standardized sequence after burn-in.
    """
    rng = np.random.default_rng(seed)
    beta, gamma, n = 0.2, 0.1, 10
    steps = int(t_max / dt)
    delay = int(tau / dt)

    x = np.zeros(steps, dtype=float)
    x[0] = 1.2 + 0.05 * rng.standard_normal()

    for t in range(delay, steps - 1):
        x_tau = x[t - delay]
        x[t + 1] = x[t] + dt * (beta * x_tau / (1 + x_tau**n) - gamma * x[t])

    x = x[burn_in:]
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    return x


# -----------------------------
# Pauli operators + helpers
# -----------------------------

I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)


def op_on_qubit(op: np.ndarray, q: int, n_qubits: int) -> np.ndarray:
    ops = [I2] * n_qubits
    ops[q] = op
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def unitary_from_H(H: np.ndarray, dt: float) -> np.ndarray:
    evals, evecs = np.linalg.eigh(H)
    return evecs @ np.diag(np.exp(-1j * evals * dt)) @ evecs.conj().T


def pure_state_rho(n_qubits: int, bitstring: str) -> np.ndarray:
    dim = 2**n_qubits
    idx = 0
    for b in bitstring:
        idx = (idx << 1) | (1 if b == "1" else 0)
    ket = np.zeros((dim, 1), dtype=complex)
    ket[idx, 0] = 1.0
    return ket @ ket.conj().T


# -----------------------------
# Feature map: tapped delay (window)
# -----------------------------

def tapped_features(currents: np.ndarray, lags: int) -> np.ndarray:
    """
    currents: (T, C)
    returns:  (T, C*lags + 1) with explicit bias column
    """
    T, C = currents.shape
    d = C * lags + 1
    X = np.zeros((T, d), dtype=float)

    col = 0
    for ch in range(C):
        for lag in range(lags):
            if lag == 0:
                X[:, col] = currents[:, ch]
            else:
                X[lag:, col] = currents[:-lag, ch]
            col += 1

    X[:, -1] = 1.0
    return X


# -----------------------------
# TFIM weak-measurement reservoir
# -----------------------------

class TFIM_QRC:
    """
    n-qubit TFIM-like reservoir with weak Z measurement.

    Dynamics:
      ρ <- U_t ρ U_t†, U_t = exp(-i (H0 + u_t Hin) dt)

    Measurement record for each measured Z_q:
      dY = 2 κ <Z_q> dt + dW, dW ~ N(0, σ^2 dt)
      current I = dY/dt

    CP backaction update specialized to Pauli-Z:
      M(dY) ∝ exp(κ Z dY) (diagonal in Z basis)
      ρ <- M ρ M / Tr(M ρ M)
    """

    def __init__(
        self,
        n_qubits: int,
        hx: float,
        hz: float,
        J: float,
        input_scale: float,
        dt: float,
        kappa: float,
        noise_sigma: float,
        measure_mode: str,
        input_axis: str,
        init_state: str,
        reservoir_seed: int,
        backaction: bool = True,
    ):
        self.n = int(n_qubits)
        self.dim = 2**self.n
        self.dt = float(dt)
        self.kappa = float(kappa)
        self.noise_sigma = float(noise_sigma)
        self.measure_mode = str(measure_mode)
        self.input_axis = str(input_axis).upper()
        self.backaction = bool(backaction)

        rng = np.random.default_rng(reservoir_seed)

        self.Z_ops = [op_on_qubit(Z2, i, self.n) for i in range(self.n)]
        self.X_ops = [op_on_qubit(X2, i, self.n) for i in range(self.n)]
        self.Y_ops = [op_on_qubit(Y2, i, self.n) for i in range(self.n)]

        # TFIM-like H0 with mild disorder
        hx_i = hx * (1 + 0.1 * rng.standard_normal(self.n))
        hz_i = hz * (1 + 0.1 * rng.standard_normal(self.n))

        H0 = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(self.n):
            H0 += hx_i[i] * self.X_ops[i] + hz_i[i] * self.Z_ops[i]
        for i in range(self.n):
            j = (i + 1) % self.n
            H0 += J * (self.Z_ops[i] @ self.Z_ops[j])

        # Normalize spectrum for stable dt scaling
        H0 /= np.max(np.abs(np.linalg.eigvalsh(H0)) + 1e-12)
        self.H0 = H0

        # Input Hamiltonian Hin on X/Y/Z axis (non-commuting X is often useful with Z measurement)
        mask = rng.choice([-1.0, 1.0], size=self.n)
        if self.input_axis == "Z":
            Hin = sum(mask[i] * self.Z_ops[i] for i in range(self.n))
        elif self.input_axis == "X":
            Hin = sum(mask[i] * self.X_ops[i] for i in range(self.n))
        elif self.input_axis == "Y":
            Hin = sum(mask[i] * self.Y_ops[i] for i in range(self.n))
        else:
            raise ValueError("input_axis must be one of: X, Y, Z")

        Hin = Hin / (np.linalg.norm(Hin) + 1e-12)
        self.Hin = Hin * float(input_scale)

        # Initial state
        if init_state == "mixed":
            self.rho0 = np.eye(self.dim, dtype=complex) / self.dim
        elif init_state == "zero":
            self.rho0 = pure_state_rho(self.n, "0" * self.n)
        else:
            raise ValueError("init_state must be 'mixed' or 'zero'")

        # Precompute Z eigenvalue signs in computational basis for each qubit
        self.z_signs: List[np.ndarray] = []
        for q in range(self.n):
            s = np.empty(self.dim, dtype=float)
            for b in range(self.dim):
                bit = (b >> (self.n - 1 - q)) & 1
                s[b] = 1.0 if bit == 0 else -1.0
            self.z_signs.append(s)

    def pick_measured_qubits(self, rng: np.random.Generator) -> np.ndarray:
        if self.measure_mode == "all":
            return np.arange(self.n, dtype=int)
        if self.measure_mode == "one":
            return np.array([rng.integers(0, self.n)], dtype=int)
        if self.measure_mode == "z0":
            return np.array([0], dtype=int)
        # default: half
        k = max(1, self.n // 2)
        return rng.choice(np.arange(self.n), size=k, replace=False)

    def measure_update(self, rho: np.ndarray, q: int, dY: float) -> np.ndarray:
        if not self.backaction:
            return rho

        # CP update: M(dY) ∝ exp(kappa Z dY) (diagonal in Z basis)
        a = self.kappa * dY
        s = self.z_signs[q]
        m = np.exp(s * a)

        rho = (m[:, None] * rho) * m[None, :]
        tr = float(rho.trace().real)
        if tr <= 0.0 or not np.isfinite(tr):
            rho = self.rho0.copy()
        else:
            rho /= tr

        rho = 0.5 * (rho + rho.conj().T)
        return rho

    def precompute_unitaries(self, inputs: np.ndarray) -> np.ndarray:
        u = np.tanh(inputs.astype(float))
        T = len(u)
        U = np.empty((T, self.dim, self.dim), dtype=complex)
        for t, val in enumerate(u):
            Ht = self.H0 + val * self.Hin
            U[t] = unitary_from_H(Ht, self.dt)
        return U

    def simulate_currents(
        self,
        inputs: np.ndarray,
        n_shots: int,
        seed_base: int,
        U: Optional[np.ndarray] = None,
        noise_sigma_override: Optional[float] = None,
    ) -> np.ndarray:
        """
        Simulate weak-measurement currents.
        Returns: currents with shape (n_shots, T, n_qubits)
        """
        if U is None:
            U = self.precompute_unitaries(inputs)

        if noise_sigma_override is None:
            noise_sigma = self.noise_sigma
        else:
            noise_sigma = float(noise_sigma_override)

        T = len(inputs)
        C = self.n
        currents = np.zeros((n_shots, T, C), dtype=float)

        for s in range(n_shots):
            rng = np.random.default_rng(seed_base + s)
            rho = self.rho0.copy()

            for t in range(T):
                rho = U[t] @ rho @ U[t].conj().T
                diag = rho.diagonal().real

                for q in self.pick_measured_qubits(rng):
                    expZ = float((self.z_signs[q] * diag).sum())

                    dW = rng.normal(0.0, noise_sigma * math.sqrt(self.dt))
                    dY = 2.0 * self.kappa * expZ * self.dt + dW
                    currents[s, t, q] = dY / self.dt

                    rho = self.measure_update(rho, q, dY)
                    diag = rho.diagonal().real

        return currents


# -----------------------------
# Experiment configuration
# -----------------------------

@dataclass
class Config:
    # Reservoir / physics
    n_qubits: int = 4
    hx: float = 1.0
    hz: float = 0.0
    J: float = 1.0
    input_scale: float = 1.0
    dt: float = 0.1

    # Measurement
    kappa: float = 0.5
    noise_sigma: float = 0.10
    measure_mode: str = "all"      # "all", "half", "one", "z0"
    input_axis: str = "X"          # "X", "Y", or "Z"
    init_state: str = "zero"       # "mixed" or "zero"
    backaction: bool = True

    # Readout / features
    lags: int = 20
    ridge_alpha: float = 1e-2

    # Data split
    washout: int = 50
    train_len: int = 300
    test_len: int = 500

    # Shots / seeds
    n_shots: int = 5
    reservoir_seed: int = 17
    noise_seed: int = 1234
    series_seed: int = 42


def make_dataset(series_seed: int, total_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (u, y) where u_t = x_t, y_t = x_{t+1}.
    """
    series = generate_mackey_glass(t_max=total_steps + 2000, seed=series_seed)
    u = series[:total_steps]
    y = series[1 : total_steps + 1]
    return u, y


def fit_eval_from_currents(
    currents: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    lags: int,
    ridge_alpha: float,
    start_index: int,
) -> Dict[str, object]:
    """
    Given currents (N, T, C) and target y (T,),
    compute EV and trajectory NRMSE.
    """
    N, T, C = currents.shape

    idx_tr = slice(start_index, start_index + cfg.train_len)
    idx_te = slice(start_index + cfg.train_len, start_index + cfg.train_len + cfg.test_len)
    ytr = y[idx_tr]
    yte = y[idx_te]

    # EV baseline
    cur_mean = currents.mean(axis=0)
    X_ev = tapped_features(cur_mean, lags)
    mod_ev = Ridge(alpha=ridge_alpha).fit(X_ev[idx_tr], ytr)
    pred_ev = mod_ev.predict(X_ev[idx_te])
    err_ev = nrmse(yte, pred_ev)

    # Trajectory-level (stacked training)
    X_list = [tapped_features(currents[s], lags) for s in range(N)]
    Xtr_stack = np.concatenate([X_list[s][idx_tr] for s in range(N)], axis=0)
    ytr_stack = np.tile(ytr, N)
    mod_traj = Ridge(alpha=ridge_alpha).fit(Xtr_stack, ytr_stack)

    preds = np.stack([mod_traj.predict(X_list[s][idx_te]) for s in range(N)], axis=0)
    pred_traj = preds.mean(axis=0)
    err_traj = nrmse(yte, pred_traj)

    out = dict(
        err_ev=float(err_ev),
        err_traj=float(err_traj),
        gap=float(err_ev - err_traj),
        cond_ev_lt_1=bool(err_ev < 1.0),
        cond_traj_lt_ev=bool(err_traj < err_ev),
        cond_both=bool((err_ev < 1.0) and (err_traj < err_ev)),
    )
    return out | {"yte": yte, "pred_ev": pred_ev, "pred_traj": pred_traj}


def simulate_and_eval(cfg: Config) -> Dict[str, object]:
    """
    Convenience: simulate currents for the cfg and compute EV/traj errors.
    """
    start = max(cfg.washout, cfg.lags)
    total_steps = start + cfg.train_len + cfg.test_len
    u, y = make_dataset(cfg.series_seed, total_steps)

    qrc = TFIM_QRC(
        n_qubits=cfg.n_qubits,
        hx=cfg.hx,
        hz=cfg.hz,
        J=cfg.J,
        input_scale=cfg.input_scale,
        dt=cfg.dt,
        kappa=cfg.kappa,
        noise_sigma=cfg.noise_sigma,
        measure_mode=cfg.measure_mode,
        input_axis=cfg.input_axis,
        init_state=cfg.init_state,
        reservoir_seed=cfg.reservoir_seed,
        backaction=cfg.backaction,
    )

    t0 = time.time()
    U = qrc.precompute_unitaries(u)
    currents = qrc.simulate_currents(u, n_shots=cfg.n_shots, seed_base=cfg.noise_seed, U=U)
    sim_time = time.time() - t0

    metrics = fit_eval_from_currents(currents, y, cfg, cfg.lags, cfg.ridge_alpha, start)
    metrics["sim_time_s"] = float(sim_time)
    metrics["feature_dim"] = int(cfg.n_qubits * cfg.lags + 1)
    metrics["start_index"] = int(start)
    metrics["total_steps"] = int(total_steps)
    return metrics | {"currents": currents, "u": u, "y": y, "U": U}


# -----------------------------
# Paper evaluation suite
# -----------------------------

def plot_timeseries(yte: np.ndarray, pred_ev: np.ndarray, pred_traj: np.ndarray, path: str, title: str, n_show: int = 300) -> None:
    n_show = min(n_show, len(yte))
    plt.figure(figsize=(10, 3.8))
    plt.plot(yte[:n_show], label="Target", lw=2, alpha=0.5)
    plt.plot(pred_ev[:n_show], "--", label="EV")
    plt.plot(pred_traj[:n_show], label="Traj")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_ridge_sweep(df: pd.DataFrame, path: str, title: str) -> None:
    plt.figure(figsize=(7.5, 3.8))
    plt.semilogx(df["ridge_alpha"], df["err_ev"], "o-", label="NRMSE_EV")
    plt.semilogx(df["ridge_alpha"], df["err_traj"], "o-", label="NRMSE_traj")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("ridge_alpha (λ)")
    plt.ylabel("NRMSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_lags_sweep(df: pd.DataFrame, path: str, title: str) -> None:
    plt.figure(figsize=(7.5, 3.8))
    plt.plot(df["lags"], df["gap"], "o-")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("lags (window depth per channel)")
    plt.ylabel("gap = NRMSE_EV − NRMSE_traj")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_gap_hist(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str, path: str, title: str) -> None:
    plt.figure(figsize=(7.5, 3.8))
    plt.hist(df_a["gap"], bins=12, alpha=0.6, label=label_a)
    plt.hist(df_b["gap"], bins=12, alpha=0.6, label=label_b)
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("gap = NRMSE_EV − NRMSE_traj")
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_phase_heatmap(pivot: pd.DataFrame, path: str, title: str) -> None:
    plt.figure(figsize=(7.6, 4.8))
    im = plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
    plt.yticks(np.arange(len(pivot.index)), [f"{v:.3f}" for v in pivot.index])
    plt.xlabel("n_shots")
    plt.ylabel("noise_sigma")
    plt.title(title)
    plt.colorbar(im, label="gap (EV − traj)")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def collage_2x2(img_paths: List[str], out_path: str, labels: Optional[List[str]] = None, pad: int = 10) -> None:
    """
    Create a 2×2 collage using PIL. img_paths must have length 4.
    No matplotlib subplots needed.
    """
    imgs = [Image.open(p).convert("RGB") for p in img_paths]
    widths = [im.size[0] for im in imgs]
    heights = [im.size[1] for im in imgs]
    w = max(widths)
    h = max(heights)

    # resize each to common size (preserve aspect by fitting inside)
    resized = []
    for im in imgs:
        im2 = im.copy()
        im2.thumbnail((w, h))
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        ox = (w - im2.size[0]) // 2
        oy = (h - im2.size[1]) // 2
        canvas.paste(im2, (ox, oy))
        resized.append(canvas)

    out = Image.new("RGB", (2 * w + 3 * pad, 2 * h + 3 * pad), (255, 255, 255))

    # paste
    out.paste(resized[0], (pad, pad))
    out.paste(resized[1], (2 * pad + w, pad))
    out.paste(resized[2], (pad, 2 * pad + h))
    out.paste(resized[3], (2 * pad + w, 2 * pad + h))

    out.save(out_path, quality=95)


def run_multiseed(cfg: Config, reservoir_seeds: List[int], series_seeds: List[int], noise_seeds: List[int], tag: str) -> pd.DataFrame:
    """
    Multi-seed reruns. To speed up, cache U per (reservoir_seed, series_seed).
    """
    rows = []
    for rseed in reservoir_seeds:
        for sseed in series_seeds:
            # Build dataset length from cfg's lags and washout
            start = max(cfg.washout, cfg.lags)
            total_steps = start + cfg.train_len + cfg.test_len
            u, y = make_dataset(sseed, total_steps)

            qrc = TFIM_QRC(
                n_qubits=cfg.n_qubits,
                hx=cfg.hx,
                hz=cfg.hz,
                J=cfg.J,
                input_scale=cfg.input_scale,
                dt=cfg.dt,
                kappa=cfg.kappa,
                noise_sigma=cfg.noise_sigma,
                measure_mode=cfg.measure_mode,
                input_axis=cfg.input_axis,
                init_state=cfg.init_state,
                reservoir_seed=rseed,
                backaction=cfg.backaction,
            )
            U = qrc.precompute_unitaries(u)

            for nseed in noise_seeds:
                currents = qrc.simulate_currents(u, n_shots=cfg.n_shots, seed_base=nseed, U=U)
                m = fit_eval_from_currents(currents, y, cfg, cfg.lags, cfg.ridge_alpha, start)

                rows.append(
                    dict(
                        tag=tag,
                        reservoir_seed=rseed,
                        series_seed=sseed,
                        noise_seed=nseed,
                        noise_sigma=cfg.noise_sigma,
                        n_shots=cfg.n_shots,
                        lags=cfg.lags,
                        ridge_alpha=cfg.ridge_alpha,
                        err_ev=m["err_ev"],
                        err_traj=m["err_traj"],
                        gap=m["gap"],
                        cond_ev_lt_1=m["cond_ev_lt_1"],
                        cond_traj_lt_ev=m["cond_traj_lt_ev"],
                        cond_both=m["cond_both"],
                    )
                )
    return pd.DataFrame(rows)


def paper_evaluation(out_dir: str = "qrc_mg_paper_artifacts") -> None:
    ensure_dir(out_dir)
    fig_dir = os.path.join(out_dir, "figs")
    ensure_dir(fig_dir)

    # --- Winning configuration (requested) ---
    WIN = Config(
        n_qubits=4,
        hx=1.0,
        hz=0.0,
        J=1.0,
        input_scale=1.0,
        dt=0.1,
        kappa=0.5,
        noise_sigma=0.10,
        measure_mode="all",
        input_axis="X",
        init_state="zero",
        backaction=True,
        lags=20,
        ridge_alpha=1e-2,
        washout=50,
        train_len=300,
        test_len=500,
        n_shots=5,
        reservoir_seed=17,
        noise_seed=1234,
        series_seed=42,
    )

    # --- Robust variant (same, but lower measurement noise) ---
    ROBUST = Config(**(asdict(WIN) | {"noise_sigma": 0.05}))

    # Save configs
    with open(os.path.join(out_dir, "winning_config.json"), "w") as f:
        json.dump(asdict(WIN), f, indent=2)
    with open(os.path.join(out_dir, "robust_config_sigma005.json"), "w") as f:
        json.dump(asdict(ROBUST), f, indent=2)

    master_rows: List[Dict[str, object]] = []

    # ==========================
    # 1) Single-run evaluation (winning config)
    # ==========================
    res_win = simulate_and_eval(WIN)
    print("\n[Winning config]")
    print(f"NRMSE_EV   = {res_win['err_ev']:.6f}")
    print(f"NRMSE_traj = {res_win['err_traj']:.6f}")
    print(f"Gap        = {res_win['gap']:.6f}")
    print(f"Feasible?  = {res_win['cond_both']}")

    # Export timeseries
    ts_path = os.path.join(out_dir, "winning_timeseries.csv")
    pd.DataFrame(
        {
            "t": np.arange(len(res_win["yte"])),
            "target": res_win["yte"],
            "pred_ev": res_win["pred_ev"],
            "pred_traj": res_win["pred_traj"],
        }
    ).to_csv(ts_path, index=False)

    figA = os.path.join(fig_dir, "A_timeseries_winning.png")
    plot_timeseries(
        res_win["yte"],
        res_win["pred_ev"],
        res_win["pred_traj"],
        figA,
        title=f"A) Winning config (EV={res_win['err_ev']:.3f}, Traj={res_win['err_traj']:.3f})",
        n_show=300,
    )

    master_rows.append(dict(section="single_run", tag="WIN", **asdict(WIN), err_ev=res_win["err_ev"], err_traj=res_win["err_traj"], gap=res_win["gap"]))

    # Also evaluate robust single-run (same seeds)
    res_rob = simulate_and_eval(ROBUST)
    master_rows.append(dict(section="single_run", tag="ROBUST", **asdict(ROBUST), err_ev=res_rob["err_ev"], err_traj=res_rob["err_traj"], gap=res_rob["gap"]))

    # ==========================
    # 2) Ridge sweep (λ sweep) on the SAME currents (winning config)
    # ==========================
    ridge_grid = np.logspace(-5, 0, 13)
    curr = res_win["currents"]
    y = res_win["y"]
    start = max(WIN.washout, WIN.lags)

    ridge_rows = []
    for lam in ridge_grid:
        m = fit_eval_from_currents(curr, y, WIN, WIN.lags, float(lam), start)
        ridge_rows.append(dict(ridge_alpha=float(lam), err_ev=m["err_ev"], err_traj=m["err_traj"], gap=m["gap"], cond_both=m["cond_both"]))
    ridge_df = pd.DataFrame(ridge_rows)
    ridge_df.to_csv(os.path.join(out_dir, "ridge_sweep_winning.csv"), index=False)

    figB = os.path.join(fig_dir, "B_ridge_sweep.png")
    plot_ridge_sweep(ridge_df, figB, title="B) Ridge sweep on winning currents")

    for r in ridge_rows:
        row = dict(section="ridge_sweep", tag="WIN", **asdict(WIN))
        row["ridge_alpha"] = r["ridge_alpha"]
        row["err_ev"] = r["err_ev"]
        row["err_traj"] = r["err_traj"]
        row["gap"] = r["gap"]
        row["cond_both"] = r["cond_both"]
        master_rows.append(row)

    # ==========================
    # 3) Lags sweep (window size) using a single simulation long enough for max lags
    # ==========================
    lags_list = [5, 10, 20, 35, 50, 80]
    max_lags = max(lags_list)
    start_max = max(WIN.washout, max_lags)
    total_steps = start_max + WIN.train_len + WIN.test_len
    u_long, y_long = make_dataset(WIN.series_seed, total_steps)

    qrc_long = TFIM_QRC(
        n_qubits=WIN.n_qubits, hx=WIN.hx, hz=WIN.hz, J=WIN.J,
        input_scale=WIN.input_scale, dt=WIN.dt,
        kappa=WIN.kappa, noise_sigma=WIN.noise_sigma,
        measure_mode=WIN.measure_mode, input_axis=WIN.input_axis,
        init_state=WIN.init_state, reservoir_seed=WIN.reservoir_seed,
        backaction=WIN.backaction,
    )
    U_long = qrc_long.precompute_unitaries(u_long)
    currents_long = qrc_long.simulate_currents(u_long, n_shots=WIN.n_shots, seed_base=WIN.noise_seed, U=U_long)

    lags_rows = []
    for L in lags_list:
        start_L = max(WIN.washout, L)
        m = fit_eval_from_currents(currents_long, y_long, WIN, lags=int(L), ridge_alpha=WIN.ridge_alpha, start_index=start_L)
        d = WIN.n_qubits * int(L) + 1
        rho_ev = WIN.train_len / d
        rho_traj = WIN.train_len * WIN.n_shots / d
        lags_rows.append(dict(lags=int(L), err_ev=m["err_ev"], err_traj=m["err_traj"], gap=m["gap"], rho_ev=rho_ev, rho_traj=rho_traj))
    lags_df = pd.DataFrame(lags_rows)
    lags_df.to_csv(os.path.join(out_dir, "lags_sweep_winning.csv"), index=False)

    figC1 = os.path.join(fig_dir, "C1_lags_gap.png")
    plot_lags_sweep(lags_df, figC1, title="C1) Regime dependence: gap vs lags (winning params)")

    # Gap vs rho_ev (sample-ratio heuristic plot)
    figC2 = os.path.join(fig_dir, "C2_gap_vs_rhoev.png")
    plt.figure(figsize=(7.5, 3.8))
    plt.plot(lags_df["rho_ev"], lags_df["gap"], "o-")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("rho_EV = N_tr / d")
    plt.ylabel("gap = EV − traj")
    plt.title("C2) Sample-ratio heuristic: gap vs rho_EV")
    plt.tight_layout()
    plt.savefig(figC2, dpi=180)
    plt.close()

    for r in lags_rows:
        row = dict(section="lags_sweep", tag="WIN", **asdict(WIN))
        row["lags"] = r["lags"]
        row["err_ev"] = r["err_ev"]
        row["err_traj"] = r["err_traj"]
        row["gap"] = r["gap"]
        row["rho_ev"] = r["rho_ev"]
        row["rho_traj"] = r["rho_traj"]
        master_rows.append(row)

    # ==========================
    # 4) Multi-seed robustness (WIN vs ROBUST)
    # ==========================
    reservoir_seeds = [17, 23]
    series_seeds = [42, 43]
    noise_seeds = list(range(2000, 2005))  # 5 -> total 2*2*5=20 runs

    t0 = time.time()
    ms_win = run_multiseed(WIN, reservoir_seeds, series_seeds, noise_seeds, tag="WIN_sigma010")
    ms_rob = run_multiseed(ROBUST, reservoir_seeds, series_seeds, noise_seeds, tag="ROBUST_sigma005")
    print(f"\n[Multi-seed] done in {time.time()-t0:.1f}s")

    ms_win.to_csv(os.path.join(out_dir, "multiseed_win_sigma010.csv"), index=False)
    ms_rob.to_csv(os.path.join(out_dir, "multiseed_robust_sigma005.csv"), index=False)

    # Summary table
    def summarize(df: pd.DataFrame) -> Dict[str, float]:
        return dict(
            n_runs=len(df),
            err_ev_mean=float(df["err_ev"].mean()),
            err_ev_std=float(df["err_ev"].std(ddof=1)),
            err_traj_mean=float(df["err_traj"].mean()),
            err_traj_std=float(df["err_traj"].std(ddof=1)),
            gap_mean=float(df["gap"].mean()),
            gap_std=float(df["gap"].std(ddof=1)),
            frac_ev_lt_1=float((df["err_ev"] < 1.0).mean()),
            frac_traj_lt_ev=float((df["err_traj"] < df["err_ev"]).mean()),
            frac_both=float(((df["err_ev"] < 1.0) & (df["err_traj"] < df["err_ev"])).mean()),
        )

    ms_summary = pd.DataFrame([
        dict(tag="WIN_sigma010", **summarize(ms_win)),
        dict(tag="ROBUST_sigma005", **summarize(ms_rob)),
    ])
    ms_summary.to_csv(os.path.join(out_dir, "multiseed_summary.csv"), index=False)
    print("\n[Multi-seed summary]")
    print(ms_summary.to_string(index=False))

    figD = os.path.join(fig_dir, "D_gap_hist_multiseed.png")
    plot_gap_hist(ms_win, ms_rob, "WIN σ=0.10", "ROBUST σ=0.05", figD, title="D) Multi-seed gap distribution")

    # add to master
    for _, row in pd.concat([ms_win, ms_rob], axis=0).iterrows():
        base = asdict(WIN).copy()
        base.update(row.to_dict())  # overwrites seeds/noise_sigma/etc. safely
        base["section"] = "multiseed"
        master_rows.append(base)

    # ==========================
    # 5) Backaction control (winning config): backaction ON vs OFF (same seeds)
    # ==========================
    WIN_NOBA = Config(**(asdict(WIN) | {"backaction": False}))
    res_noba = simulate_and_eval(WIN_NOBA)

    back_df = pd.DataFrame([
        dict(mode="backaction_on", err_ev=res_win["err_ev"], err_traj=res_win["err_traj"], gap=res_win["gap"]),
        dict(mode="backaction_off", err_ev=res_noba["err_ev"], err_traj=res_noba["err_traj"], gap=res_noba["gap"]),
    ])
    back_df.to_csv(os.path.join(out_dir, "backaction_control.csv"), index=False)

    figE = os.path.join(fig_dir, "E_backaction_control.png")
    plt.figure(figsize=(6.8, 3.8))
    x = np.arange(len(back_df))
    plt.plot(x, back_df["err_ev"], "o-", label="NRMSE_EV")
    plt.plot(x, back_df["err_traj"], "o-", label="NRMSE_traj")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xticks(x, back_df["mode"], rotation=10)
    plt.ylabel("NRMSE")
    plt.title("E) Backaction control (same seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figE, dpi=180)
    plt.close()

    for _, row in back_df.iterrows():
        master_rows.append(dict(section="backaction", tag="WIN", mode=row["mode"], **asdict(WIN), err_ev=row["err_ev"], err_traj=row["err_traj"], gap=row["gap"]))

    # ==========================
    # 6) Phase diagram (noise_sigma × n_shots) for ROBUST config
    # ==========================
    # Simulate max shots once per noise; slice for fewer shots.
    noise_list = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]
    shots_list = [1, 2, 3, 5, 8, 10]
    max_shots = max(shots_list)

    # For phase diagram: ensure dataset long enough for ROBUST.lags
    startR = max(ROBUST.washout, ROBUST.lags)
    total_stepsR = startR + ROBUST.train_len + ROBUST.test_len
    uR, yR = make_dataset(ROBUST.series_seed, total_stepsR)

    qrcR = TFIM_QRC(
        n_qubits=ROBUST.n_qubits, hx=ROBUST.hx, hz=ROBUST.hz, J=ROBUST.J,
        input_scale=ROBUST.input_scale, dt=ROBUST.dt,
        kappa=ROBUST.kappa, noise_sigma=ROBUST.noise_sigma,
        measure_mode=ROBUST.measure_mode, input_axis=ROBUST.input_axis,
        init_state=ROBUST.init_state, reservoir_seed=ROBUST.reservoir_seed,
        backaction=ROBUST.backaction,
    )
    UR = qrcR.precompute_unitaries(uR)

    phase_rows = []
    for noise in noise_list:
        currents_noise = qrcR.simulate_currents(uR, n_shots=max_shots, seed_base=ROBUST.noise_seed, U=UR, noise_sigma_override=float(noise))
        for N in shots_list:
            cur_slice = currents_noise[:N]
            m = fit_eval_from_currents(cur_slice, yR, ROBUST, ROBUST.lags, ROBUST.ridge_alpha, startR)
            phase_rows.append(dict(noise_sigma=float(noise), n_shots=int(N), err_ev=m["err_ev"], err_traj=m["err_traj"], gap=m["gap"]))

    phase_df = pd.DataFrame(phase_rows)
    phase_df.to_csv(os.path.join(out_dir, "phase_noise_vs_shots.csv"), index=False)

    pivot_gap = phase_df.pivot(index="noise_sigma", columns="n_shots", values="gap").sort_index()
    pivot_gap = pivot_gap.reindex(columns=shots_list)

    figF = os.path.join(fig_dir, "F_phase_gap_heatmap.png")
    plot_phase_heatmap(pivot_gap, figF, title="F) Phase diagram (ROBUST): gap = EV − traj")

    for _, row in phase_df.iterrows():
        base = asdict(ROBUST).copy()
        base.update(dict(noise_sigma=float(row["noise_sigma"]), n_shots=int(row["n_shots"])))
        base.update(dict(section="phase", tag="ROBUST", err_ev=float(row["err_ev"]), err_traj=float(row["err_traj"]), gap=float(row["gap"])))
        master_rows.append(base)

    # ==========================
    # 7) Build 4-panel paper figure (collage)
    # Choose panels: A timeseries, B ridge sweep, D gap hist, F phase heatmap
    # ==========================
    final_4panel = os.path.join(out_dir, "final_4panel.png")
    collage_2x2([figA, figB, figD, figF], final_4panel)

    # ==========================
    # 8) Write master CSV
    # ==========================
    master_df = pd.DataFrame(master_rows)
    master_df.to_csv(os.path.join(out_dir, "paper_results_master.csv"), index=False)

    # Write a short README
    readme = f"""QRC paper evaluation artifacts
=============================

This folder was generated by qrc_mg_paper_full.py.

Key files:
  - winning_config.json
  - robust_config_sigma005.json
  - winning_timeseries.csv
  - ridge_sweep_winning.csv
  - lags_sweep_winning.csv
  - multiseed_win_sigma010.csv
  - multiseed_robust_sigma005.csv
  - multiseed_summary.csv
  - backaction_control.csv
  - phase_noise_vs_shots.csv
  - paper_results_master.csv
  - final_4panel.png

Figures are in ./figs/
"""
    with open(os.path.join(out_dir, "README.txt"), "w") as f:
        f.write(readme)

    print("\nArtifacts written to:", os.path.abspath(out_dir))
    print("Final 4-panel figure:", os.path.abspath(final_4panel))


def main() -> None:
    paper_evaluation(out_dir="qrc_mg_paper_artifacts")


if __name__ == "__main__":
    main()
