#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
narma_plot3_autotune.py

Plot-3-only (2-panel) validation for NARMA-n with an *auto-iteration* mode that
keeps searching until:
  (1) EV mean NRMSE < 1.0   (configurable via --target_ev)
  (2) Trajectory is better than EV (gap_mean >= --min_gap, win_rate >= --min_win_rate)

It uses the same fair-budget model:
    S_total = G * N_shots * T
    T = floor(S_budget / (G * N_shots))

Default: G=2 (Z and X measurement settings).

No PennyLane dependency:
Implements a tiny numpy statevector simulator for:
  RY / RZ / CNOT ring, plus H for X-basis measurement.

Outputs (to --outdir, default ./paper_figures):
  - figNARMA3_timeseries_opt.png / .pdf
  - figNARMA3_trace.csv
  - figNARMA3_opt_config.json
  - (optional) autotune_scan.csv

Typical usage:
  # Let it iterate until EV<1 and Trajectory>EV, then plot the found config.
  python narma_plot3_autotune.py --autotune --reservoir_seeds 0,1,2,3,4,5,6,7 --data_seed 42

  # Force a known-good config (fast, deterministic)
  python narma_plot3_autotune.py --shots 10 --noise 0.0 --n_layers 0 --L 10 --washout 10 --eta 1.0 --in_scale 1.0 --ridge_alpha 0.001
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Styling (minimal, paper-like)
# =============================================================================

COLOR_TRAJ = "#0072B2"   # deep blue
COLOR_EV   = "#D55E00"   # orange
COLOR_TEXT = "#333333"
COLOR_GRID = "#E6E6E6"
COLOR_BG   = "#FFFFFF"


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.facecolor": COLOR_BG,
        "axes.facecolor": COLOR_BG,
        "savefig.facecolor": COLOR_BG,
        "savefig.bbox": "tight",
        "axes.linewidth": 1.0,
        "text.color": COLOR_TEXT,
        "axes.labelcolor": COLOR_TEXT,
        "axes.edgecolor": COLOR_TEXT,
        "xtick.color": COLOR_TEXT,
        "ytick.color": COLOR_TEXT,
    })


def style_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    else:
        ax.grid(False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12, fontweight="bold", color=COLOR_TEXT
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# NARMA-n
# =============================================================================

def narma(u: np.ndarray, order: int = 10) -> np.ndarray:
    """Standard NARMA-n recursion.

    y[t+1] = 0.3 y[t] + 0.05 y[t] * sum_{i=0}^{n-1} y[t-i] + 1.5 u[t-n+1] u[t] + 0.1
    """
    u = np.asarray(u, dtype=np.float64).ravel()
    n = int(order)
    y = np.zeros_like(u)
    for t in range(n - 1, len(u) - 1):
        y_sum = np.sum(y[t - n + 1 : t + 1])
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * y_sum + 1.5 * u[t - n + 1] * u[t] + 0.1
    return y


def narma_series(sample_len: int, seed: int, *, u_low: float, u_high: float, order: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    u = rng.uniform(float(u_low), float(u_high), size=int(sample_len)).astype(np.float64)
    y = narma(u, order=int(order))
    return u, y


def nrmse_scalar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred)) / (np.std(y_true) + 1e-12)


# =============================================================================
# Tiny statevector simulator for the specific circuit
# =============================================================================

def _ry(theta: float) -> np.ndarray:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(phi: float) -> np.ndarray:
    e0 = np.exp(-0.5j * phi)
    e1 = np.exp(+0.5j * phi)
    return np.array([[e0, 0.0], [0.0, e1]], dtype=np.complex128)


_H = (1.0 / math.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)


def _apply_1q(state: np.ndarray, gate: np.ndarray, wire: int, n_qubits: int) -> np.ndarray:
    dim = 1 << n_qubits
    low = 1 << wire
    high = dim // (2 * low)
    psi = state.reshape(high, 2, low)

    out = np.empty_like(psi)
    out[:, 0, :] = gate[0, 0] * psi[:, 0, :] + gate[0, 1] * psi[:, 1, :]
    out[:, 1, :] = gate[1, 0] * psi[:, 0, :] + gate[1, 1] * psi[:, 1, :]
    return out.reshape(dim)


def _apply_cnot(state: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    dim = 1 << n_qubits
    idx = np.arange(dim, dtype=np.int64)
    m_control = ((idx >> control) & 1) == 1
    m_target0 = ((idx >> target) & 1) == 0
    idx0 = idx[m_control & m_target0]
    idx1 = idx0 ^ (1 << target)

    out = state.copy()
    tmp = out[idx0].copy()
    out[idx0] = out[idx1]
    out[idx1] = tmp
    return out


def _run_reservoir_state(x_angles: np.ndarray, params: np.ndarray, n_qubits: int, n_layers: int) -> np.ndarray:
    dim = 1 << n_qubits
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0.0j

    # Encoding
    for i in range(n_qubits):
        state = _apply_1q(state, _ry(float(x_angles[i])), i, n_qubits)

    # Reservoir layers (can be 0)
    for layer in range(n_layers):
        s = layer * 2 * n_qubits
        for i in range(n_qubits):
            state = _apply_1q(state, _ry(float(params[s + i])), i, n_qubits)
            state = _apply_1q(state, _rz(float(params[s + n_qubits + i])), i, n_qubits)

        for i in range(n_qubits - 1):
            state = _apply_cnot(state, i, i + 1, n_qubits)
        state = _apply_cnot(state, n_qubits - 1, 0, n_qubits)

    return state


def _sample_z(state: np.ndarray, shots: int, rng: np.random.Generator, n_qubits: int) -> np.ndarray:
    probs = np.abs(state) ** 2
    probs = probs / probs.sum()
    idx = rng.choice(probs.size, size=int(shots), p=probs)
    bits = ((idx[:, None] >> np.arange(n_qubits)[None, :]) & 1).astype(np.int8)
    return (1.0 - 2.0 * bits).astype(np.float64)


def _measure_features(x_angles: np.ndarray, params: np.ndarray, *, shots: int, rng: np.random.Generator, n_qubits: int, n_layers: int) -> np.ndarray:
    state = _run_reservoir_state(x_angles, params, n_qubits, n_layers)
    z = _sample_z(state, shots, rng, n_qubits)

    state_x = state
    for i in range(n_qubits):
        state_x = _apply_1q(state_x, _H, i, n_qubits)
    x = _sample_z(state_x, shots, rng, n_qubits)

    return np.concatenate([z, x], axis=1)


# =============================================================================
# QRC simulation (EV vs Trajectory)
# =============================================================================

def apply_measurement_noise(feat_t: np.ndarray, rng: np.random.Generator, noise_level: float) -> np.ndarray:
    if noise_level <= 0:
        return feat_t
    p = float(np.clip(noise_level, 0.0, 0.25))
    flips = rng.random(size=feat_t.shape) < p
    feat = feat_t.copy()
    feat[flips] *= -1.0
    feat += float(noise_level) * rng.normal(size=feat.shape)
    return feat


def leaky_integrate(F: np.ndarray, eta: float) -> np.ndarray:
    if eta >= 1.0:
        return F
    if eta <= 0.0:
        return F * 0.0
    out = np.zeros_like(F)
    out[0] = F[0]
    for t in range(1, F.shape[0]):
        out[t] = (1.0 - eta) * out[t - 1] + eta * F[t]
    return out


def build_windows_traj(F: np.ndarray, idx_list: np.ndarray, L: int) -> np.ndarray:
    T, shots, M = F.shape
    rows: List[np.ndarray] = []
    for t in idx_list:
        block = F[t - L + 1 : t + 1]
        block = np.transpose(block, (1, 0, 2))
        block = block.reshape(shots, L * M)
        rows.append(np.concatenate([block, np.ones((shots, 1))], axis=1))
    return np.vstack(rows)


def build_windows_ev(F: np.ndarray, idx_list: np.ndarray, L: int) -> np.ndarray:
    mu = F.mean(axis=1)
    rows: List[np.ndarray] = []
    for t in idx_list:
        block = mu[t - L + 1 : t + 1].reshape(-1)
        rows.append(np.concatenate([block, [1.0]]))
    return np.array(rows, dtype=np.float64)


@dataclass
class Fig3Config:
    # Budget
    S_budget: int = 10000
    meas_groups: int = 2

    # Circuit / reservoir
    n_qubits: int = 6
    n_layers: int = 3
    lookback: int = 1
    in_scale: float = 0.7
    eta: float = 0.25

    # Readout
    L: int = 10
    washout: int = 10
    train_frac: float = 0.7
    ridge_alpha: float = 1e-3

    # NARMA data
    order: int = 10
    u_low: float = 0.0
    u_high: float = 0.5
    sample_len: int = 4000

    # Safety
    T_min: int = 50


def simulate_predictions(cfg: Fig3Config, *, reservoir_seed: int, data_seed: int, shots: int, noise_level: float) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(reservoir_seed))

    G = int(max(1, int(cfg.meas_groups)))
    shots_i = int(shots)
    T = int(cfg.S_budget // (G * shots_i))
    if T < int(cfg.T_min):
        raise ValueError(f"T too small (T={T}) for shots={shots_i} under S={cfg.S_budget} with G={G}")

    # Generate data
    sample_len = max(int(cfg.sample_len), int(cfg.lookback) + T + 2 + int(cfg.order))
    u, y = narma_series(sample_len=sample_len, seed=int(data_seed), u_low=float(cfg.u_low), u_high=float(cfg.u_high), order=int(cfg.order))

    # Supervised:
    # predict y[t] from past `lookback` inputs u[t-lookback:t] (RC-typical)
    X_raw, y_raw = [], []
    for i in range(int(cfg.lookback), len(u)):
        X_raw.append(u[i - int(cfg.lookback) : i])
        y_raw.append(y[i])

    X_raw = np.asarray(X_raw, dtype=np.float64)[:T]
    y_raw = np.asarray(y_raw, dtype=np.float64)[:T]

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_all = scaler_X.fit_transform(X_raw)
    y_all = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

    inW = rng.normal(size=(int(cfg.n_qubits), int(cfg.lookback))) * float(cfg.in_scale)
    params = rng.uniform(0.0, 2.0 * np.pi, size=(int(cfg.n_layers) * 2 * int(cfg.n_qubits),))

    F = np.zeros((T, shots_i, 2 * int(cfg.n_qubits)), dtype=np.float64)
    for t in range(T):
        x_angles = inW @ X_all[t]
        feat = _measure_features(x_angles, params, shots=shots_i, rng=rng, n_qubits=int(cfg.n_qubits), n_layers=int(cfg.n_layers))
        F[t] = apply_measurement_noise(feat, rng, float(noise_level))

    F = leaky_integrate(F, float(cfg.eta))

    t0 = max(int(cfg.washout), int(cfg.L) - 1)
    idx = np.arange(t0, T)
    ntr = int(float(cfg.train_frac) * len(idx))
    idx_tr, idx_te = idx[:ntr], idx[ntr:]

    y_tr, y_te = y_all[idx_tr], y_all[idx_te]

    X_traj_tr = build_windows_traj(F, idx_tr, int(cfg.L))
    X_traj_te = build_windows_traj(F, idx_te, int(cfg.L))
    X_ev_tr = build_windows_ev(F, idx_tr, int(cfg.L))
    X_ev_te = build_windows_ev(F, idx_te, int(cfg.L))

    y_traj_tr = np.repeat(y_tr, shots_i)

    sc_traj, sc_ev = StandardScaler(), StandardScaler()
    ridge_traj = Ridge(alpha=float(cfg.ridge_alpha))
    ridge_ev = Ridge(alpha=float(cfg.ridge_alpha))

    ridge_traj.fit(sc_traj.fit_transform(X_traj_tr), y_traj_tr)
    ridge_ev.fit(sc_ev.fit_transform(X_ev_tr), y_tr)

    yhat_ev = ridge_ev.predict(sc_ev.transform(X_ev_te))
    yhat_traj = ridge_traj.predict(sc_traj.transform(X_traj_te)).reshape(len(idx_te), shots_i).mean(axis=1)

    y_te_raw = scaler_y.inverse_transform(y_te.reshape(-1, 1)).ravel()
    yhat_ev_raw = scaler_y.inverse_transform(yhat_ev.reshape(-1, 1)).ravel()
    yhat_traj_raw = scaler_y.inverse_transform(yhat_traj.reshape(-1, 1)).ravel()

    return {
        "T": np.array([T], dtype=int),
        "idx_te": idx_te.astype(int),
        "y_true": y_te_raw,
        "y_ev": yhat_ev_raw,
        "y_traj": yhat_traj_raw,
        "ev_nrmse": np.array([nrmse_scalar(y_te_raw, yhat_ev_raw)], dtype=float),
        "traj_nrmse": np.array([nrmse_scalar(y_te_raw, yhat_traj_raw)], dtype=float),
    }


def evaluate_config(cfg: Fig3Config, *, shots: int, noise: float, reservoir_seeds: List[int], data_seed: int) -> Dict[str, float]:
    evs, trs = [], []
    T_eff = None
    for s in reservoir_seeds:
        sim = simulate_predictions(cfg, reservoir_seed=int(s), data_seed=int(data_seed), shots=int(shots), noise_level=float(noise))
        evs.append(float(sim["ev_nrmse"][0]))
        trs.append(float(sim["traj_nrmse"][0]))
        if T_eff is None:
            T_eff = int(sim["T"][0])

    evs = np.asarray(evs, dtype=float)
    trs = np.asarray(trs, dtype=float)
    gaps = evs - trs
    return {
        "shots": int(shots),
        "noise": float(noise),
        "T": int(T_eff) if T_eff is not None else -1,
        "ev_mean": float(np.mean(evs)),
        "ev_std": float(np.std(evs, ddof=1)) if evs.size > 1 else 0.0,
        "traj_mean": float(np.mean(trs)),
        "traj_std": float(np.std(trs, ddof=1)) if trs.size > 1 else 0.0,
        "gap_mean": float(np.mean(gaps)),
        "gap_std": float(np.std(gaps, ddof=1)) if gaps.size > 1 else 0.0,
        "win_rate": float(np.mean(gaps > 0.0)),
        "n": int(evs.size),
    }


def plot_fig3(cfg: Fig3Config, *, outdir: Path, shots: int, noise: float, reservoir_seeds: List[int], data_seed: int, seed_ev: Optional[int], n_show: int, export_prefix: str) -> Dict[str, object]:
    if seed_ev is None:
        seed_ev = int(reservoir_seeds[0])

    sims: Dict[int, Dict[str, np.ndarray]] = {}
    for s in reservoir_seeds:
        sims[int(s)] = simulate_predictions(cfg, reservoir_seed=int(s), data_seed=int(data_seed), shots=int(shots), noise_level=float(noise))

    y_true = sims[int(reservoir_seeds[0])]["y_true"]
    n_show_eff = int(min(max(20, int(n_show)), len(y_true)))
    t = np.arange(n_show_eff)

    ev_vals = np.array([float(sims[int(s)]["ev_nrmse"][0]) for s in reservoir_seeds], dtype=float)
    tr_vals = np.array([float(sims[int(s)]["traj_nrmse"][0]) for s in reservoir_seeds], dtype=float)
    gap_vals = ev_vals - tr_vals

    ev_mu = float(np.mean(ev_vals))
    tr_mu = float(np.mean(tr_vals))
    gap_mu = float(np.mean(gap_vals))
    win_rate = float(np.mean(gap_vals > 0.0))
    T_eff = int(sims[int(reservoir_seeds[0])]["T"][0])

    # Export trace
    rows = []
    for s in reservoir_seeds:
        d = sims[int(s)]
        for i in range(n_show_eff):
            rows.append({"seed": int(s), "t": int(i), "y_true": float(d["y_true"][i]), "y_ev": float(d["y_ev"][i]), "y_traj": float(d["y_traj"][i])})
    pd.DataFrame(rows).to_csv(outdir / "figNARMA3_trace.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(13.8, 4.6), gridspec_kw={"width_ratios": [2.2, 1.0]})

    # (a) time series overlay
    ax = axs[0]
    ax.plot(t, y_true[:n_show_eff], lw=2.2, color=COLOR_TEXT, label="True")

    d_ev = sims[int(seed_ev)]
    ax.plot(t, d_ev["y_ev"][:n_show_eff], lw=1.9, ls="--", color=COLOR_EV, label=f"EV (seed={seed_ev})")

    for i, s in enumerate(reservoir_seeds):
        d = sims[int(s)]
        alpha = 0.28 + 0.60 * (i / max(1, len(reservoir_seeds) - 1))
        ax.plot(t, d["y_traj"][:n_show_eff], lw=2.0, color=COLOR_TRAJ, alpha=float(alpha), label=("Trajectory (seeds)" if i == 0 else None))

    ax.set_xlabel("Test time index")
    ax.set_ylabel("NARMA output")
    ax.set_title(f"NARMA-{int(cfg.order)} prediction (seed overlay)")
    ax.legend(loc="best", frameon=False)
    style_axis(ax)
    panel_label(ax, "(a)")

    # (b) paired NRMSE
    ax = axs[1]
    x0, x1 = 0.0, 1.0
    for i, s in enumerate(reservoir_seeds):
        y0 = float(ev_vals[i]); y1 = float(tr_vals[i])
        ax.plot([x0, x1], [y0, y1], color=COLOR_GRID, lw=1.2, alpha=0.95, zorder=1)
        ax.scatter([x0], [y0], s=55, color=COLOR_EV, edgecolors=COLOR_TEXT, linewidths=0.4, zorder=2)
        ax.scatter([x1], [y1], s=55, color=COLOR_TRAJ, edgecolors=COLOR_TEXT, linewidths=0.4, zorder=3)

    x0m, x1m = -0.08, 1.08
    ax.plot([x0m, x1m], [ev_mu, tr_mu], color=COLOR_TEXT, lw=2.4, alpha=0.9, zorder=4)
    ax.scatter([x0m], [ev_mu], s=180, marker="D", color=COLOR_EV, edgecolors=COLOR_TEXT, linewidths=0.9, zorder=5)
    ax.scatter([x1m], [tr_mu], s=180, marker="D", color=COLOR_TRAJ, edgecolors=COLOR_TEXT, linewidths=0.9, zorder=6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["EV", "Trajectory"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel("NRMSE")
    ax.set_title("Seed-level NRMSE (paired)")
    ax.text(
        0.5, 0.98,
        rf"$N_{{shots}}$={int(shots)}, $\sigma$={float(noise):g}, $G$={int(cfg.meas_groups)}" + "\n"
        + rf"$T$={T_eff}, EV={ev_mu:.3f}, Traj={tr_mu:.3f}, $\Delta$={gap_mu:+.3f}, win={win_rate:.2f}",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=10, color=COLOR_TEXT
    )
    style_axis(ax)
    panel_label(ax, "(b)")

    fig.suptitle("Fig. NARMA3: configuration diagnostic (auto-tuned)", y=1.02)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(outdir / f"{export_prefix}.{ext}")
    plt.close(fig)

    info: Dict[str, object] = {
        "task": f"NARMA-{int(cfg.order)}",
        "shots": int(shots),
        "noise": float(noise),
        "S_budget": int(cfg.S_budget),
        "meas_groups": int(cfg.meas_groups),
        "T_eff": int(T_eff),
        "reservoir_seeds": [int(s) for s in reservoir_seeds],
        "data_seed": int(data_seed),
        "seed_ev": int(seed_ev),
        "n_show": int(n_show_eff),
        "ev_nrmse_per_seed": [float(x) for x in ev_vals],
        "traj_nrmse_per_seed": [float(x) for x in tr_vals],
        "gap_per_seed": [float(x) for x in gap_vals],
        "ev_mean": float(ev_mu),
        "traj_mean": float(tr_mu),
        "gap_mean": float(gap_mu),
        "win_rate": float(win_rate),
        "cfg": asdict(cfg),
        "env": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    with open(outdir / "figNARMA3_opt_config.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    return info


def _parse_int_list(s: str) -> List[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return [int(p) for p in parts]


def _parse_float_list(s: str) -> List[float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return [float(p) for p in parts]


def autotune_until(
    base_cfg: Fig3Config,
    *,
    shots_list: List[int],
    noise_list: List[float],
    n_layers_list: List[int],
    L_list: List[int],
    eta_list: List[float],
    in_scale_list: List[float],
    ridge_list: List[float],
    reservoir_seeds_quick: List[int],
    reservoir_seeds_full: List[int],
    data_seed: int,
    target_ev: float,
    min_gap: float,
    min_win_rate: float,
    require_traj_under_ev_target: bool = True,
    export_scan_csv: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[Fig3Config, int, float, Dict[str, float], pd.DataFrame]:
    """Iterate through a deterministic candidate list; stop at first config that passes.

    Strategy:
      1) quick check on a small seed subset
      2) full validation on the full seed list
    """
    rows = []
    best_candidate = None

    # Candidate order matters (we try the "simplest / likely-good" settings early)
    eval_count = 0
    for shots in shots_list:
        for noise in noise_list:
            for n_layers in n_layers_list:
                for L in L_list:
                    for eta in eta_list:
                        for in_scale in in_scale_list:
                            for ridge_alpha in ridge_list:
                                cfg = replace(
                                    base_cfg,
                                    n_layers=int(n_layers),
                                    L=int(L),
                                    washout=int(max(10, L)),
                                    eta=float(eta),
                                    in_scale=float(in_scale),
                                    ridge_alpha=float(ridge_alpha),
                                )
                                eval_count += 1
                                try:
                                    quick = evaluate_config(cfg, shots=int(shots), noise=float(noise), reservoir_seeds=reservoir_seeds_quick, data_seed=int(data_seed))
                                except Exception as e:
                                    continue

                                row = {**quick,
                                       "n_layers": int(n_layers),
                                       "L": int(L),
                                       "washout": int(max(10, L)),
                                       "eta": float(eta),
                                       "in_scale": float(in_scale),
                                       "ridge_alpha": float(ridge_alpha),
                                       "stage": "quick"}
                                rows.append(row)

                                passes_quick = (
                                    (quick["ev_mean"] < float(target_ev))
                                    and (quick["gap_mean"] >= float(min_gap))
                                    and (quick["win_rate"] >= float(min_win_rate))
                                    and ((quick["traj_mean"] < float(target_ev)) if require_traj_under_ev_target else True)
                                )

                                if verbose:
                                    print(
                                        f"[autotune] {eval_count:04d} shots={shots:<3d} noise={noise:>5g} "
                                        f"layers={n_layers} L={L:<2d} eta={eta:<4g} in={in_scale:<4g} ridge={ridge_alpha:<8g} | "
                                        f"EV={quick['ev_mean']:.3f} Traj={quick['traj_mean']:.3f} Δ={quick['gap_mean']:+.3f} win={quick['win_rate']:.2f}"
                                    )

                                if not passes_quick:
                                    continue

                                # Full validation
                                full = evaluate_config(cfg, shots=int(shots), noise=float(noise), reservoir_seeds=reservoir_seeds_full, data_seed=int(data_seed))
                                rows.append({**full,
                                             "n_layers": int(n_layers),
                                             "L": int(L),
                                             "washout": int(max(10, L)),
                                             "eta": float(eta),
                                             "in_scale": float(in_scale),
                                             "ridge_alpha": float(ridge_alpha),
                                             "stage": "full"})

                                passes_full = (
                                    (full["ev_mean"] < float(target_ev))
                                    and (full["gap_mean"] >= float(min_gap))
                                    and (full["win_rate"] >= float(min_win_rate))
                                    and ((full["traj_mean"] < float(target_ev)) if require_traj_under_ev_target else True)
                                )

                                if passes_full:
                                    df = pd.DataFrame(rows)
                                    if export_scan_csv is not None:
                                        df.to_csv(export_scan_csv, index=False)
                                    return cfg, int(shots), float(noise), full, df

                                # Track best (for debugging if nothing passes)
                                if best_candidate is None:
                                    best_candidate = (cfg, int(shots), float(noise), full)

    df = pd.DataFrame(rows)
    if export_scan_csv is not None:
        df.to_csv(export_scan_csv, index=False)

    if best_candidate is not None:
        cfg, shots, noise, full = best_candidate
        raise RuntimeError(
            "No config satisfied thresholds. Best validated candidate (full stage): "
            f"EV={full['ev_mean']:.3f}, Traj={full['traj_mean']:.3f}, Δ={full['gap_mean']:+.3f}, win={full['win_rate']:.2f}. "
            "Try relaxing thresholds or expanding the search lists."
        )
    raise RuntimeError("No configs were evaluated (all failed early). Check parameters.")


def main() -> None:
    ap = argparse.ArgumentParser("NARMA Plot-3 with auto-iteration until EV<1 and Traj>EV")

    ap.add_argument("--outdir", type=str, default="paper_figures", help="Output directory")
    ap.add_argument("--export_prefix", type=str, default="figNARMA3_timeseries_opt", help="Prefix for figure files")

    # Primary knobs for a single forced run
    ap.add_argument("--shots", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--reservoir_seeds", type=str, default="0,1,2,3,4,5,6,7", help="Comma-separated")
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--seed_ev", type=int, default=None)
    ap.add_argument("--n_show", type=int, default=300)

    # Budget
    ap.add_argument("--S_budget", type=int, default=10000)
    ap.add_argument("--meas_groups", type=int, default=2)

    # Task
    ap.add_argument("--order", type=int, default=10)
    ap.add_argument("--u_low", type=float, default=0.0)
    ap.add_argument("--u_high", type=float, default=0.5)

    # Hyperparameters (base values)
    ap.add_argument("--n_qubits", type=int, default=6)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--lookback", type=int, default=1)
    ap.add_argument("--in_scale", type=float, default=0.7)
    ap.add_argument("--eta", type=float, default=0.25)
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--washout", type=int, default=10)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--ridge_alpha", type=float, default=1e-3)

    ap.add_argument("--sample_len", type=int, default=4000)
    ap.add_argument("--T_min", type=int, default=50)

    # Autotune mode
    ap.add_argument("--autotune", action="store_true", help="Iterate until constraints met, then plot found config")
    ap.add_argument("--target_ev", type=float, default=1.0, help="Require EV_mean < target_ev")
    ap.add_argument("--min_gap", type=float, default=0.01, help="Require gap_mean >= min_gap")
    ap.add_argument("--min_win_rate", type=float, default=0.75, help="Require win_rate >= min_win_rate")
    ap.add_argument("--autotune_quick_seeds", type=str, default="0,1,2,3", help="Seed subset for quick filtering")
    ap.add_argument("--autotune_scan_csv", type=str, default=None, help="If set, write the full scan table to CSV")

    # Autotune candidate lists (comma-separated)
    ap.add_argument("--autotune_shots", type=str, default="10,5,15,20,25,30")
    ap.add_argument("--autotune_noise", type=str, default="0.0,0.01,0.02,0.05")
    ap.add_argument("--autotune_n_layers", type=str, default="0,1,2,3")
    ap.add_argument("--autotune_L", type=str, default="10,15,20,5")
    ap.add_argument("--autotune_eta", type=str, default="1.0,0.5,0.25")
    ap.add_argument("--autotune_in_scale", type=str, default="1.0,0.7,1.5,0.5")
    ap.add_argument("--autotune_ridge", type=str, default="0.001,0.01,0.1,1,10,100")

    ap.add_argument("--quiet", action="store_true")

    args = ap.parse_args()
    set_style()

    outdir = Path(args.outdir).resolve()
    ensure_dir(outdir)

    base_cfg = Fig3Config(
        S_budget=int(args.S_budget),
        meas_groups=int(args.meas_groups),
        n_qubits=int(args.n_qubits),
        n_layers=int(args.n_layers),
        lookback=int(args.lookback),
        in_scale=float(args.in_scale),
        eta=float(args.eta),
        L=int(args.L),
        washout=int(args.washout),
        train_frac=float(args.train_frac),
        ridge_alpha=float(args.ridge_alpha),
        order=int(args.order),
        u_low=float(args.u_low),
        u_high=float(args.u_high),
        sample_len=int(args.sample_len),
        T_min=int(args.T_min),
    )

    reservoir_seeds_full = _parse_int_list(args.reservoir_seeds)
    data_seed = int(args.data_seed)

    if args.autotune:
        shots_list = _parse_int_list(args.autotune_shots)
        noise_list = _parse_float_list(args.autotune_noise)
        n_layers_list = _parse_int_list(args.autotune_n_layers)
        L_list = _parse_int_list(args.autotune_L)
        eta_list = _parse_float_list(args.autotune_eta)
        in_scale_list = _parse_float_list(args.autotune_in_scale)
        ridge_list = _parse_float_list(args.autotune_ridge)

        reservoir_seeds_quick = _parse_int_list(args.autotune_quick_seeds)
        scan_csv = Path(args.autotune_scan_csv).resolve() if args.autotune_scan_csv else None

        cfg_found, shots_found, noise_found, stats_full, df_scan = autotune_until(
            base_cfg,
            shots_list=shots_list,
            noise_list=noise_list,
            n_layers_list=n_layers_list,
            L_list=L_list,
            eta_list=eta_list,
            in_scale_list=in_scale_list,
            ridge_list=ridge_list,
            reservoir_seeds_quick=reservoir_seeds_quick,
            reservoir_seeds_full=reservoir_seeds_full,
            data_seed=data_seed,
            target_ev=float(args.target_ev),
            min_gap=float(args.min_gap),
            min_win_rate=float(args.min_win_rate),
            export_scan_csv=scan_csv,
            verbose=(not args.quiet),
        )

        print("\n[autotune] FOUND config that satisfies constraints (full validation):")
        print(json.dumps(stats_full, indent=2))

        # Plot with the found cfg/knobs
        info = plot_fig3(
            cfg_found,
            outdir=outdir,
            shots=shots_found,
            noise=noise_found,
            reservoir_seeds=reservoir_seeds_full,
            data_seed=data_seed,
            seed_ev=args.seed_ev,
            n_show=int(args.n_show),
            export_prefix=str(args.export_prefix),
        )

    else:
        info = plot_fig3(
            base_cfg,
            outdir=outdir,
            shots=int(args.shots),
            noise=float(args.noise),
            reservoir_seeds=reservoir_seeds_full,
            data_seed=data_seed,
            seed_ev=args.seed_ev,
            n_show=int(args.n_show),
            export_prefix=str(args.export_prefix),
        )

    print("\n[NARMA3] wrote:")
    print(f"  - {outdir / (str(args.export_prefix) + '.png')}")
    print(f"  - {outdir / (str(args.export_prefix) + '.pdf')}")
    print(f"  - {outdir / 'figNARMA3_trace.csv'}")
    print(f"  - {outdir / 'figNARMA3_opt_config.json'}")
    print("\n[NARMA3] summary:")
    print(
        f"  shots={info['shots']}  noise={info['noise']:.3g}  S={info['S_budget']}  G={info['meas_groups']}  T={info['T_eff']}\n"
        f"  ev_mean={info['ev_mean']:.4f}  traj_mean={info['traj_mean']:.4f}  gap_mean={info['gap_mean']:+.4f}  win_rate={info['win_rate']:.2f}"
    )


if __name__ == "__main__":
    main()
