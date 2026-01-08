#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autonomous QRC Optimization (TFIM, weak measurement) on Mackey–Glass
===================================================================

This script performs an *iterative, logged parameter search* for a physically
motivated quantum reservoir computing (QRC) model based on an n-qubit
Transverse-Field Ising Model (TFIM) Hamiltonian with continuous weak-measurement
records.

Goal / stopping criteria
------------------------
Stop only when BOTH are satisfied:

  (1) NRMSE_EV   < 1.0
  (2) NRMSE_traj < NRMSE_EV

Definitions
-----------
For each input stream u_t (Mackey–Glass x_t), we simulate N shots of weak
measurement currents I_t^(n). We compare two budget-matched pipelines using the
*same simulated shots*:

EV baseline:
  • Average currents across shots to estimate a mean current stream
  • Build tapped-delay (window) features from the mean stream
  • Train ridge regression and evaluate on test

Trajectory-level:
  • Build tapped-delay features per shot
  • Stack all per-shot windows as supervised samples for ridge training
  • At test time, predict per-shot and average predictions across shots

Reservoir model
---------------
• n=4–6 qubits (default search uses n=4 for speed; 4 is within the requested range)
• TFIM-like Hamiltonian on a ring:
    H(t) = H0 + u(t) * Hin
    H0 = Σ_i (hx_i X_i + hz_i Z_i) + J Σ_i Z_i Z_{i+1}
• Exact unitary step per dt: ρ <- U ρ U†
• Weak measurement record for each measured Z_q channel:
    dY = 2 κ <Z_q> dt + dW,  dW ~ N(0, σ^2 dt)
    current I = dY / dt
• Completely-positive (CP) measurement update specialized to Pauli-Z:
    M(dY) ∝ exp(κ Z dY),  ρ <- M ρ M / Tr(M ρ M)

Outputs (written to ./qrc_mg_evlt1_artifacts/)
---------------------------------------------
• qrc_sweep_results.csv      : per-iteration parameters + metrics
• iter_progress.png          : NRMSE_EV and NRMSE_traj vs iteration
• iter_scatter_ev_vs_traj.png: EV vs Traj scatter across iterations
• iter_phase_points.png      : sparse "phase diagram" (noise_sigma × n_shots tested points)
• final_config.json          : winning configuration
• final_timeseries.csv       : target and predictions on test segment
• final_timeseries.png       : plot of target vs EV vs Traj
• phase_noise_vs_shots.csv   : dense grid over noise_sigma × n_shots (final config)
• final_4panel.png           : paper-ready 4-panel figure

Dependencies
------------
numpy, pandas, matplotlib, scikit-learn
"""

from __future__ import annotations

import os
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# -----------------------------
# Utility / metrics
# -----------------------------

def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized RMSE: RMSE / std(y_true)."""
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
# Pauli operators and helpers
# -----------------------------

I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)


def op_on_qubit(op: np.ndarray, q: int, n_qubits: int) -> np.ndarray:
    """Place a 2x2 operator on qubit q of an n-qubit system (0=leftmost)."""
    ops = [I2] * n_qubits
    ops[q] = op
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def unitary_from_H(H: np.ndarray, dt: float) -> np.ndarray:
    """Exact unitary exp(-i H dt) via eigendecomposition."""
    evals, evecs = np.linalg.eigh(H)
    return evecs @ np.diag(np.exp(-1j * evals * dt)) @ evecs.conj().T


def pure_state_rho(n_qubits: int, bitstring: str) -> np.ndarray:
    """Density matrix for a computational basis state |bitstring>."""
    dim = 2 ** n_qubits
    idx = 0
    for b in bitstring:
        idx = (idx << 1) | (1 if b == "1" else 0)
    ket = np.zeros((dim, 1), dtype=complex)
    ket[idx, 0] = 1.0
    return ket @ ket.conj().T


# -----------------------------
# Feature map: tapped delay
# -----------------------------

def tapped_features(currents: np.ndarray, lags: int) -> np.ndarray:
    """
    currents: (T, C)
    returns:  (T, C*lags + 1) with bias
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
    n-qubit TFIM reservoir with weak measurement of Pauli-Z channels.
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
        seed: int,
    ):
        self.n = int(n_qubits)
        self.dim = 2 ** self.n

        self.dt = float(dt)
        self.kappa = float(kappa)
        self.noise_sigma = float(noise_sigma)
        self.measure_mode = str(measure_mode)
        self.input_axis = str(input_axis).upper()

        rng = np.random.default_rng(seed)

        self.Z_ops = [op_on_qubit(Z2, i, self.n) for i in range(self.n)]
        self.X_ops = [op_on_qubit(X2, i, self.n) for i in range(self.n)]
        self.Y_ops = [op_on_qubit(Y2, i, self.n) for i in range(self.n)]

        # TFIM-like H0 with mild disorder
        H = np.zeros((self.dim, self.dim), dtype=complex)
        hx_i = hx * (1 + 0.1 * rng.standard_normal(self.n))
        hz_i = hz * (1 + 0.1 * rng.standard_normal(self.n))

        for i in range(self.n):
            H += hx_i[i] * self.X_ops[i] + hz_i[i] * self.Z_ops[i]

        for i in range(self.n):
            j = (i + 1) % self.n
            H += J * (self.Z_ops[i] @ self.Z_ops[j])

        # Normalize spectrum for stable dt scaling
        H /= np.max(np.abs(np.linalg.eigvalsh(H)) + 1e-12)
        self.H0 = H

        # Input coupling Hin (choose non-commuting axis to create rich Z-record structure)
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
        """Measurement schedule: all / half / one / z0."""
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
        """
        CP measurement update for Pauli-Z:
            M(dY) ∝ exp(kappa Z dY) (diagonal in Z basis)
            rho <- M rho M / Tr(M rho M)
        """
        a = self.kappa * dY
        s = self.z_signs[q]
        m = np.exp(s * a)

        rho = (m[:, None] * rho) * m[None, :]
        tr = float(rho.trace().real)
        if tr <= 0.0 or not np.isfinite(tr):
            rho = self.rho0.copy()
        else:
            rho /= tr

        # Hermitize for numerical stability
        rho = 0.5 * (rho + rho.conj().T)
        return rho

    def precompute_unitaries(self, inputs: np.ndarray) -> np.ndarray:
        """Precompute U_t for all inputs (tanh squashing)."""
        u = np.tanh(inputs.astype(float))
        T = len(u)
        U = np.empty((T, self.dim, self.dim), dtype=complex)
        for t, val in enumerate(u):
            Ht = self.H0 + val * self.Hin
            U[t] = unitary_from_H(Ht, self.dt)
        return U

    def simulate_currents(self, inputs: np.ndarray, n_shots: int, seed_base: int) -> np.ndarray:
        """
        Simulate weak-measurement currents.
        Returns currents with shape (n_shots, T, n_qubits).
        """
        U = self.precompute_unitaries(inputs)
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

                    # record increment + current
                    dW = rng.normal(0.0, self.noise_sigma * math.sqrt(self.dt))
                    dY = 2.0 * self.kappa * expZ * self.dt + dW
                    currents[s, t, q] = dY / self.dt

                    # backaction
                    rho = self.measure_update(rho, q, dY)
                    diag = rho.diagonal().real

        return currents


# -----------------------------
# Configuration + evaluation
# -----------------------------

@dataclass
class Config:
    # reservoir / physics
    n_qubits: int = 4
    hx: float = 1.0
    hz: float = 0.0
    J: float = 1.0
    input_scale: float = 1.0
    dt: float = 0.1

    # measurement
    kappa: float = 0.5
    noise_sigma: float = 0.2
    measure_mode: str = "all"   # all, half, one, z0
    input_axis: str = "Z"       # X/Y/Z
    init_state: str = "mixed"   # mixed or zero

    # readout + features
    lags: int = 80
    ridge_alpha: float = 1e-3

    # data split
    washout: int = 50
    train_len: int = 300
    test_len: int = 500

    # shots / seeds
    n_shots: int = 5
    reservoir_seed: int = 17
    noise_seed: int = 1234
    series_seed: int = 42


def evaluate(cfg: Config, return_series: bool = False) -> Dict[str, object]:
    """
    Run one experiment (one input stream, fixed seeds) and return metrics.
    """
    # Ensure window and washout indexing is valid
    start = max(cfg.washout, cfg.lags)
    total_steps = start + cfg.train_len + cfg.test_len

    series = generate_mackey_glass(t_max=total_steps + 2000, seed=cfg.series_seed)
    u = series[:total_steps]
    y = series[1 : total_steps + 1]

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
        seed=cfg.reservoir_seed,
    )

    t0 = time.time()
    currents = qrc.simulate_currents(u, n_shots=cfg.n_shots, seed_base=cfg.noise_seed)
    sim_time = time.time() - t0

    idx_tr = slice(start, start + cfg.train_len)
    idx_te = slice(start + cfg.train_len, start + cfg.train_len + cfg.test_len)
    ytr = y[idx_tr]
    yte = y[idx_te]

    # EV baseline: average currents then build window features
    cur_mean = currents.mean(axis=0)
    X_ev = tapped_features(cur_mean, cfg.lags)
    mod_ev = Ridge(alpha=cfg.ridge_alpha).fit(X_ev[idx_tr], ytr)
    pred_ev = mod_ev.predict(X_ev[idx_te])
    err_ev = nrmse(yte, pred_ev)

    # Trajectory-level: stack per-shot windows, then test-time prediction averaging
    X_list = [tapped_features(currents[s], cfg.lags) for s in range(cfg.n_shots)]
    Xtr_stack = np.concatenate([X_list[s][idx_tr] for s in range(cfg.n_shots)], axis=0)
    ytr_stack = np.tile(ytr, cfg.n_shots)
    mod_traj = Ridge(alpha=cfg.ridge_alpha).fit(Xtr_stack, ytr_stack)
    preds = np.stack([mod_traj.predict(X_list[s][idx_te]) for s in range(cfg.n_shots)], axis=0)
    pred_traj = preds.mean(axis=0)
    err_traj = nrmse(yte, pred_traj)

    # Persistence baseline (for context only)
    pers_pred = u[idx_te]
    err_persist = nrmse(yte, pers_pred)

    cond_ev_lt_1 = bool(err_ev < 1.0)
    cond_traj_lt_ev = bool(err_traj < err_ev)
    cond_both = bool(cond_ev_lt_1 and cond_traj_lt_ev)

    out: Dict[str, object] = dict(
        err_ev=float(err_ev),
        err_traj=float(err_traj),
        gap=float(err_ev - err_traj),
        persist_nrmse=float(err_persist),
        sim_time_s=float(sim_time),
        cond_ev_lt_1=cond_ev_lt_1,
        cond_traj_lt_ev=cond_traj_lt_ev,
        cond_both=cond_both,
        feature_dim=int(cfg.n_qubits * cfg.lags + 1),
        start_index=int(start),
        total_steps=int(total_steps),
    )

    if return_series:
        out.update(yte=yte, pred_ev=pred_ev, pred_traj=pred_traj)

    return out


def compute_ev_traj_from_currents(
    currents: np.ndarray,
    cfg: Config,
    u: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """
    Helper for dense phase diagram: compute EV/traj from a precomputed current tensor.
    currents shape: (N, T, C)
    """
    N, T, C = currents.shape
    start = max(cfg.washout, cfg.lags)

    idx_tr = slice(start, start + cfg.train_len)
    idx_te = slice(start + cfg.train_len, start + cfg.train_len + cfg.test_len)

    ytr = y[idx_tr]
    yte = y[idx_te]

    cur_mean = currents.mean(axis=0)
    X_ev = tapped_features(cur_mean, cfg.lags)
    mod_ev = Ridge(alpha=cfg.ridge_alpha).fit(X_ev[idx_tr], ytr)
    pred_ev = mod_ev.predict(X_ev[idx_te])
    err_ev = nrmse(yte, pred_ev)

    X_list = [tapped_features(currents[s], cfg.lags) for s in range(N)]
    Xtr_stack = np.concatenate([X_list[s][idx_tr] for s in range(N)], axis=0)
    ytr_stack = np.tile(ytr, N)
    mod_traj = Ridge(alpha=cfg.ridge_alpha).fit(Xtr_stack, ytr_stack)
    preds = np.stack([mod_traj.predict(X_list[s][idx_te]) for s in range(N)], axis=0)
    pred_traj = preds.mean(axis=0)
    err_traj = nrmse(yte, pred_traj)

    return float(err_ev), float(err_traj)


# -----------------------------
# Iterative optimization + artifacts
# -----------------------------

def update_iteration_plots(df: pd.DataFrame, out_dir: str) -> None:
    """Update the three per-iteration plots."""
    # 1) NRMSE progress
    plt.figure(figsize=(7, 3.2))
    plt.plot(df["iter"], df["err_ev"], "o-", label="NRMSE_EV")
    plt.plot(df["iter"], df["err_traj"], "o-", label="NRMSE_traj")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("NRMSE")
    plt.title("NRMSE across iterations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iter_progress.png"), dpi=160)
    plt.close()

    # 2) EV vs Traj scatter
    plt.figure(figsize=(4.6, 4.2))
    plt.scatter(df["err_ev"], df["err_traj"])
    mn = float(min(df["err_ev"].min(), df["err_traj"].min()))
    mx = float(max(df["err_ev"].max(), df["err_traj"].max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
    plt.axvline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("NRMSE_EV")
    plt.ylabel("NRMSE_traj")
    plt.title("Trajectory vs EV (tested iterations)")
    # highlight any solutions found so far
    sol = df[df["cond_both"] == True]
    if len(sol) > 0:
        plt.scatter(sol["err_ev"], sol["err_traj"], s=140, marker="*", label="feasible")
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iter_scatter_ev_vs_traj.png"), dpi=160)
    plt.close()

    # 3) Sparse phase diagram points: noise_sigma × n_shots tested so far
    plt.figure(figsize=(5.8, 4.2))
    sc = plt.scatter(df["n_shots"], df["noise_sigma"], c=df["gap"])
    plt.xlabel("n_shots")
    plt.ylabel("noise_sigma")
    plt.title("Sparse phase diagram (tested points): color = gap (EV − traj)")
    plt.colorbar(sc, label="gap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iter_phase_points.png"), dpi=160)
    plt.close()


def main() -> None:
    out_dir = "qrc_mg_evlt1_artifacts"
    ensure_dir(out_dir)

    csv_path = os.path.join(out_dir, "qrc_sweep_results.csv")

    base = Config()

    # Systematic candidate list (coarse-to-fine, minimal changes).
    # We stop at the FIRST configuration satisfying:
    #   err_ev < 1.0  AND  err_traj < err_ev
    candidate_mods: List[Tuple[str, Dict[str, object]]] = [
        ("baseline (mixed, input Z, long window)", {}),
        ("init_state=zero", {"init_state": "zero"}),
        ("input_axis=X (non-commuting with Z measurement)", {"init_state": "zero", "input_axis": "X"}),
        ("reduce window + stronger ridge", {"init_state": "zero", "input_axis": "X", "lags": 20, "ridge_alpha": 1e-2}),
        ("reduce measurement noise (sigma=0.15)", {"init_state": "zero", "input_axis": "X", "lags": 20, "ridge_alpha": 1e-2, "noise_sigma": 0.15}),
        ("reduce measurement noise (sigma=0.10)", {"init_state": "zero", "input_axis": "X", "lags": 20, "ridge_alpha": 1e-2, "noise_sigma": 0.10}),
        ("reduce measurement noise (sigma=0.05)", {"init_state": "zero", "input_axis": "X", "lags": 20, "ridge_alpha": 1e-2, "noise_sigma": 0.05}),
    ]

    rows: List[Dict[str, object]] = []
    solution_cfg: Config | None = None
    solution_metrics: Dict[str, object] | None = None

    print("Iterative QRC search started...")
    for it, (label, mods) in enumerate(candidate_mods, start=1):
        cfg_dict = asdict(base)
        cfg_dict.update(mods)
        cfg = Config(**cfg_dict)

        print(f"Iteration {it}: {label}")
        metrics = evaluate(cfg, return_series=False)

        row = {"iter": it, "label": label, **asdict(cfg), **metrics}
        rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        update_iteration_plots(df, out_dir)

        if metrics["cond_both"]:
            solution_cfg = cfg
            print(f"✅ Found feasible solution at iteration {it}")
            break

    if solution_cfg is None:
        raise RuntimeError("No feasible configuration found. Expand candidate list/search space.")

    # Re-evaluate solution with series for timeseries artifacts
    solution_metrics = evaluate(solution_cfg, return_series=True)

    # Save final config
    with open(os.path.join(out_dir, "final_config.json"), "w") as f:
        json.dump(asdict(solution_cfg), f, indent=2)

    # Save final timeseries CSV
    yte = solution_metrics["yte"]
    pev = solution_metrics["pred_ev"]
    ptr = solution_metrics["pred_traj"]
    ts_df = pd.DataFrame({
        "t": np.arange(len(yte), dtype=int),
        "target": yte,
        "pred_ev": pev,
        "pred_traj": ptr,
    })
    ts_df.to_csv(os.path.join(out_dir, "final_timeseries.csv"), index=False)

    # Timeseries plot
    n_show = min(300, len(yte))
    plt.figure(figsize=(9, 3.6))
    plt.plot(yte[:n_show], label="Target", lw=2, alpha=0.5)
    plt.plot(pev[:n_show], "--", label=f"EV (NRMSE={solution_metrics['err_ev']:.3f})")
    plt.plot(ptr[:n_show], label=f"Traj (NRMSE={solution_metrics['err_traj']:.3f})")
    plt.title("Final config: Mackey–Glass one-step prediction (first 300 test steps)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "final_timeseries.png"), dpi=180)
    plt.close()

    # Dense phase diagram for final config: noise_sigma × n_shots
    # (simulate max shots once per noise, then slice for different N)
    start = max(solution_cfg.washout, solution_cfg.lags)
    total_steps = start + solution_cfg.train_len + solution_cfg.test_len
    series = generate_mackey_glass(t_max=total_steps + 2000, seed=solution_cfg.series_seed)
    u = series[:total_steps]
    y = series[1 : total_steps + 1]

    noise_list = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]
    shots_list = [1, 2, 3, 5, 8, 10]
    max_shots = max(shots_list)

    phase_rows: List[Dict[str, object]] = []

    for noise in noise_list:
        qrc = TFIM_QRC(
            n_qubits=solution_cfg.n_qubits,
            hx=solution_cfg.hx,
            hz=solution_cfg.hz,
            J=solution_cfg.J,
            input_scale=solution_cfg.input_scale,
            dt=solution_cfg.dt,
            kappa=solution_cfg.kappa,
            noise_sigma=float(noise),
            measure_mode=solution_cfg.measure_mode,
            input_axis=solution_cfg.input_axis,
            init_state=solution_cfg.init_state,
            seed=solution_cfg.reservoir_seed,
        )
        currents = qrc.simulate_currents(u, n_shots=max_shots, seed_base=solution_cfg.noise_seed)

        for N in shots_list:
            cur_slice = currents[:N]
            err_ev, err_traj = compute_ev_traj_from_currents(cur_slice, solution_cfg, u, y)
            phase_rows.append(
                dict(noise_sigma=float(noise), n_shots=int(N), err_ev=err_ev, err_traj=err_traj, gap=err_ev - err_traj)
            )

    phase_df = pd.DataFrame(phase_rows)
    phase_df.to_csv(os.path.join(out_dir, "phase_noise_vs_shots.csv"), index=False)

    # 4-panel figure
    df = pd.DataFrame(rows)
    pivot = phase_df.pivot(index="noise_sigma", columns="n_shots", values="gap").sort_index()
    pivot = pivot.reindex(columns=shots_list)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))

    # A) optimization progress
    ax = axes[0, 0]
    ax.plot(df["iter"], df["err_ev"], "o-", label="NRMSE_EV")
    ax.plot(df["iter"], df["err_traj"], "o-", label="NRMSE_traj")
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("NRMSE")
    ax.set_title("A) Iterative search progress")
    ax.legend()

    # B) final timeseries
    ax = axes[0, 1]
    ax.plot(yte[:n_show], label="Target", lw=2, alpha=0.5)
    ax.plot(pev[:n_show], "--", label=f"EV ({solution_metrics['err_ev']:.2f})")
    ax.plot(ptr[:n_show], label=f"Traj ({solution_metrics['err_traj']:.2f})")
    ax.set_title("B) Final prediction (first 300 test steps)")
    ax.legend()

    # C) scatter across iterations
    ax = axes[1, 0]
    ax.scatter(df["err_ev"], df["err_traj"])
    mn = float(min(df["err_ev"].min(), df["err_traj"].min()))
    mx = float(max(df["err_ev"].max(), df["err_traj"].max()))
    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("NRMSE_EV")
    ax.set_ylabel("NRMSE_traj")
    ax.set_title("C) Trajectory vs EV (iterations)")
    sol = df[df["cond_both"] == True].iloc[-1]
    ax.scatter([sol["err_ev"]], [sol["err_traj"]], s=150, marker="*", label="solution")
    ax.legend()

    # D) heatmap gap noise vs shots (final config)
    ax = axes[1, 1]
    im = ax.imshow(pivot.values, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{v:.3f}" for v in pivot.index])
    ax.set_xlabel("n_shots")
    ax.set_ylabel("noise_sigma")
    ax.set_title("D) Gap = EV − Traj (final config)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "final_4panel.png"), dpi=180)
    plt.close()

    print("\nSearch finished.")
    print("Winning config:", asdict(solution_cfg))
    print(f"NRMSE_EV   = {solution_metrics['err_ev']:.4f}")
    print(f"NRMSE_traj = {solution_metrics['err_traj']:.4f}")
    print(f"Gap (EV−traj) = {solution_metrics['gap']:.4f}")
    print("Artifacts in:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
