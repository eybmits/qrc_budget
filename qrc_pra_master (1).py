#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRA-ready QRC evaluation suite: EV vs Trajectory on Mackey–Glass
===============================================================

This single script merges two working codebases (minimal changes) into one
"paper pipeline" that produces reviewer-ready artifacts for Physical Review A.

It contains two backends:

(A) Circuit-QRC (PennyLane, shot sampling) under FAIR sampling budget
    S = N_shots * T  (T is automatically adjusted as S//N_shots).
    -> fast regime maps (noise × shots) + uncertainty across random seeds.

(B) Physical-QRC (TFIM Hamiltonian + weak measurement trajectories, density matrix)
    -> stateful quantum reservoir + measurement backaction validation.
    -> "freeze config" seed sweeps + phase diagram with uncertainty.

Key PRA-oriented outputs
------------------------
* Raw per-run CSVs (so results are auditable)
* Mean/Std/Win-rate (Traj beats EV) statistics
* Bootstrap confidence intervals + sign-flip permutation p-value for the gap
* Clear separation of:
    - Search / tuning (optional, produces final_config.json)
    - Evaluation / robustness (frozen config, many independent seeds)

Outputs are written to:
    <outdir>/circuit_qrc/
    <outdir>/tfim_qrc/

Typical "paper run"
------------------
1) Use your already-found TFIM config (final_config.json) and run everything:
   python qrc_pra_master.py --mode all --outdir paper_artifacts --preset paper --tfim_config final_config.json

2) If you want to re-run the TFIM coarse search first:
   python qrc_pra_master.py --mode tfim --tfim_mode search_then_eval --outdir paper_artifacts

Notes
-----
- Circuit-QRC part requires: pennylane, seaborn
- TFIM-QRC part requires: numpy, pandas, matplotlib, scikit-learn

The code intentionally favors clarity/reproducibility over speed.
"""

from __future__ import annotations

import os
import json
import math
import time
import argparse
from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Optional deps for circuit-QRC plotting and QNodes
try:
    import seaborn as sns
except Exception:
    sns = None  # allow TFIM-only runs
try:
    import pennylane as qml
except Exception:
    qml = None  # allow TFIM-only runs


# =============================================================================
# Common utilities
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: object, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized RMSE: RMSE / std(y_true)."""
    return math.sqrt(mean_squared_error(y_true, y_pred)) / (np.std(y_true) + 1e-12)


def parse_list(s: str, cast=float) -> List:
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def bootstrap_ci(
    x: np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Simple nonparametric bootstrap CI for a 1D sample.

    Returns (lo, hi) quantiles.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        samp = x[rng.integers(0, n, size=n)]
        stats[b] = stat(samp)
    alpha = (1 - ci) / 2
    lo = float(np.quantile(stats, alpha))
    hi = float(np.quantile(stats, 1 - alpha))
    return lo, hi


def signflip_permutation_pvalue(d: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    """
    Paired sign-flip permutation test for mean(d)=0.

    d: paired differences (e.g., gap = EV - Traj) across independent seeds.

    Returns two-sided p-value.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = float(np.mean(d))
    # Under H0, sign of each paired difference is symmetric.
    cnt = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(d))
        val = float(np.mean(signs * d))
        if abs(val) >= abs(obs) - 1e-15:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


# =============================================================================
# (A) Circuit-based QRC suite (PennyLane) under fair budget S = shots * T
# =============================================================================

@dataclass
class CircuitConfig:
    outdir: str
    n_seeds: int = 10

    shots_list: List[int] = None
    noise_list: List[float] = None
    S_budget: int = 10000

    n_qubits: int = 6
    n_layers: int = 3
    lookback: int = 3
    L: int = 20
    washout: int = 30
    train_frac: float = 0.7
    ridge_alpha: float = 1.0
    in_scale: float = 0.7
    eta: float = 0.25
    tau: int = 17
    sample_len: int = 4000
    T_min: int = 50

    # plotting
    make_dashboard: bool = True
    make_phase_contour: bool = True
    make_slices: bool = True
    make_extra_stats_maps: bool = True

    def __post_init__(self):
        if self.shots_list is None:
            self.shots_list = [5, 10, 15, 20, 25, 30, 40, 50]
        if self.noise_list is None:
            self.noise_list = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]


def circuit_set_style() -> None:
    if sns is None:
        return
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 300,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "image.cmap": "magma_r",
    })


def circuit_mackey_glass(sample_len: int = 4000, tau: int = 17, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    delta_t = 1.0
    x = np.zeros(sample_len, dtype=np.float64)
    x[:tau] = 1.5
    for i in range(tau, sample_len):
        x_tau = x[i - tau]
        x[i] = x[i - 1] + delta_t * (0.2 * x_tau / (1.0 + x_tau**10) - 0.1 * x[i - 1])
    return x


def make_qnode(n_qubits: int, n_layers: int, shots: int):
    if qml is None:
        raise RuntimeError("pennylane is required for circuit-QRC mode, but could not be imported.")
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    @qml.qnode(dev)
    def qnode(x_angles, params):
        for i in range(n_qubits):
            qml.RY(float(x_angles[i]), wires=i)
        for layer in range(n_layers):
            s = layer * 2 * n_qubits
            for i in range(n_qubits):
                qml.RY(float(params[s + i]), wires=i)
                qml.RZ(float(params[s + n_qubits + i]), wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
        meas = []
        for i in range(n_qubits):
            meas.append(qml.sample(qml.PauliZ(i)))
        for i in range(n_qubits):
            meas.append(qml.sample(qml.PauliX(i)))
        return meas

    return qnode


def build_windows_traj(F: np.ndarray, idx_list: np.ndarray, L: int) -> np.ndarray:
    T, shots, M = F.shape
    rows = []
    for t in idx_list:
        block = F[t - L + 1 : t + 1]
        block = np.transpose(block, (1, 0, 2))
        block = block.reshape(shots, L * M)
        bias = np.ones((shots, 1), dtype=np.float64)
        rows.append(np.concatenate([block, bias], axis=1))
    return np.vstack(rows)


def build_windows_ev(F: np.ndarray, idx_list: np.ndarray, L: int) -> np.ndarray:
    mu = F.mean(axis=1)
    rows = []
    for t in idx_list:
        block = mu[t - L + 1 : t + 1].reshape(-1)
        rows.append(np.concatenate([block, [1.0]]))
    return np.array(rows, dtype=np.float64)


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
    Ff = np.zeros_like(F)
    Ff[0] = F[0]
    for t in range(1, F.shape[0]):
        Ff[t] = (1.0 - eta) * Ff[t - 1] + eta * F[t]
    return Ff


def circuit_run_once(cfg: CircuitConfig, seed: int, shots: int, noise_level: float) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    T = int(cfg.S_budget // shots)
    if T < cfg.T_min:
        return dict(ev=np.nan, traj=np.nan, gap=np.nan, T=T, S=cfg.S_budget)

    sample_len = max(cfg.sample_len, cfg.lookback + T + 2)
    data = circuit_mackey_glass(sample_len=sample_len, tau=cfg.tau, seed=seed)

    X_raw, y_raw = [], []
    for i in range(cfg.lookback, len(data) - 1):
        X_raw.append(data[i - cfg.lookback : i])
        y_raw.append(data[i])
    X_raw = np.array(X_raw)[:T]
    y_raw = np.array(y_raw)[:T]

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_all = scaler_X.fit_transform(X_raw)
    y_all = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

    inW = rng.normal(size=(cfg.n_qubits, cfg.lookback)) * cfg.in_scale
    params = rng.uniform(0, 2 * np.pi, size=(cfg.n_layers * 2 * cfg.n_qubits,))
    qnode = make_qnode(cfg.n_qubits, cfg.n_layers, shots)

    F = np.zeros((T, shots, 2 * cfg.n_qubits), dtype=np.float64)
    for t in range(T):
        x_angles = (inW @ X_all[t])
        meas_list = qnode(x_angles, params)
        feat_t = np.stack(meas_list, axis=1)
        F[t] = apply_measurement_noise(feat_t, rng, noise_level)

    F = leaky_integrate(F, cfg.eta)

    t0 = max(cfg.washout, cfg.L - 1)
    idx = np.arange(t0, T)
    if len(idx) < 10:
        return dict(ev=np.nan, traj=np.nan, gap=np.nan, T=T, S=cfg.S_budget)

    ntr = int(cfg.train_frac * len(idx))
    idx_tr, idx_te = idx[:ntr], idx[ntr:]
    if len(idx_te) < 5:
        return dict(ev=np.nan, traj=np.nan, gap=np.nan, T=T, S=cfg.S_budget)

    y_tr, y_te = y_all[idx_tr], y_all[idx_te]

    X_traj_tr = build_windows_traj(F, idx_tr, cfg.L)
    X_traj_te = build_windows_traj(F, idx_te, cfg.L)
    X_ev_tr = build_windows_ev(F, idx_tr, cfg.L)
    X_ev_te = build_windows_ev(F, idx_te, cfg.L)

    # Supervised samples for Trajectory method
    y_traj_tr = np.repeat(y_tr, shots)

    sc_traj, sc_ev = StandardScaler(), StandardScaler()
    ridge_traj = Ridge(alpha=cfg.ridge_alpha)
    ridge_ev = Ridge(alpha=cfg.ridge_alpha)

    ridge_traj.fit(sc_traj.fit_transform(X_traj_tr), y_traj_tr)
    ridge_ev.fit(sc_ev.fit_transform(X_ev_tr), y_tr)

    yhat_ev = ridge_ev.predict(sc_ev.transform(X_ev_te))
    yhat_traj = ridge_traj.predict(sc_traj.transform(X_traj_te)).reshape(len(idx_te), shots).mean(axis=1)

    ev = nrmse(y_te, yhat_ev)
    traj = nrmse(y_te, yhat_traj)
    return dict(ev=float(ev), traj=float(traj), gap=float(ev - traj), T=T, S=cfg.S_budget)


def circuit_pivot(df_stats: pd.DataFrame, metric: str, shots_list: List[int], noise_list: List[float]) -> pd.DataFrame:
    pv = df_stats.pivot(index="noise", columns="shots", values=metric)
    pv = pv.reindex(index=noise_list, columns=shots_list)
    return pv


def circuit_plot_heatmap(
    data: pd.DataFrame,
    outpath: str,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str,
    annot_fmt: str = ".3f",
    center: Optional[float] = None,
    cbar_label: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    if sns is None:
        raise RuntimeError("seaborn is required for circuit-QRC plots, but could not be imported.")
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        data,
        annot=True,
        fmt=annot_fmt,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        cbar_kws={"label": cbar_label} if cbar_label else None,
    )
    ax.set_title(title, pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def circuit_plot_individual_heatmaps(df_stats: pd.DataFrame, cfg: CircuitConfig, outdir: str) -> None:
    shots_list, noise_list = cfg.shots_list, cfg.noise_list

    ev_data = circuit_pivot(df_stats, "ev_mean", shots_list, noise_list)
    traj_data = circuit_pivot(df_stats, "traj_mean", shots_list, noise_list)
    gap_data = circuit_pivot(df_stats, "gap_mean", shots_list, noise_list)

    circuit_plot_heatmap(
        ev_data,
        os.path.join(outdir, "heatmap_ev_mean.png"),
        title="Baseline EV (mean NRMSE)",
        xlabel=r"Number of Shots ($N_{shots}$)",
        ylabel=r"Noise Level ($\sigma$)",
        cmap="magma_r",
        annot_fmt=".3f",
        cbar_label="NRMSE",
    )
    circuit_plot_heatmap(
        traj_data,
        os.path.join(outdir, "heatmap_traj_mean.png"),
        title="Trajectory Method (mean NRMSE)",
        xlabel=r"Number of Shots ($N_{shots}$)",
        ylabel=r"Noise Level ($\sigma$)",
        cmap="magma_r",
        annot_fmt=".3f",
        cbar_label="NRMSE",
    )

    max_abs = max(abs(np.nanmin(gap_data.values)), abs(np.nanmax(gap_data.values)))
    circuit_plot_heatmap(
        gap_data,
        os.path.join(outdir, "heatmap_gap_mean.png"),
        title=r"Gap $\Delta = \mathrm{EV} - \mathrm{Traj}$ (mean)",
        xlabel=r"Number of Shots ($N_{shots}$)",
        ylabel=r"Noise Level ($\sigma$)",
        cmap="RdBu",
        annot_fmt=".3f",
        center=0.0,
        vmin=-max_abs,
        vmax=max_abs,
        cbar_label="NRMSE difference",
    )

    if cfg.make_extra_stats_maps:
        gap_std = circuit_pivot(df_stats, "gap_std", shots_list, noise_list)
        circuit_plot_heatmap(
            gap_std,
            os.path.join(outdir, "heatmap_gap_std.png"),
            title=r"Gap std across seeds: $\mathrm{std}(\Delta)$",
            xlabel=r"Number of Shots ($N_{shots}$)",
            ylabel=r"Noise Level ($\sigma$)",
            cmap="magma_r",
            annot_fmt=".3f",
            cbar_label="std(Δ)",
        )

        win = circuit_pivot(df_stats, "win_rate", shots_list, noise_list)
        circuit_plot_heatmap(
            win,
            os.path.join(outdir, "heatmap_win_rate.png"),
            title=r"Win rate across seeds: $\Pr(\Delta>0)$",
            xlabel=r"Number of Shots ($N_{shots}$)",
            ylabel=r"Noise Level ($\sigma$)",
            cmap="magma_r",
            annot_fmt=".2f",
            cbar_label="fraction",
            vmin=0.0,
            vmax=1.0,
        )


def circuit_plot_dashboard(df_stats: pd.DataFrame, cfg: CircuitConfig, outdir: str) -> None:
    if sns is None:
        return
    shots_list, noise_list = cfg.shots_list, cfg.noise_list

    ev_data = circuit_pivot(df_stats, "ev_mean", shots_list, noise_list)
    traj_data = circuit_pivot(df_stats, "traj_mean", shots_list, noise_list)
    gap_data = circuit_pivot(df_stats, "gap_mean", shots_list, noise_list)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    sns.heatmap(ev_data, ax=axes[0], annot=True, fmt=".3f", cmap="magma_r", cbar_kws={"label": "NRMSE"}, linewidths=0.5)
    axes[0].set_title("Baseline: EV", pad=10)
    axes[0].set_ylabel(r"Noise Level ($\sigma$)")
    axes[0].set_xlabel(r"$N_{shots}$")
    axes[0].invert_yaxis()

    sns.heatmap(traj_data, ax=axes[1], annot=True, fmt=".3f", cmap="magma_r", cbar_kws={"label": "NRMSE"}, linewidths=0.5)
    axes[1].set_title("Method: Trajectory", pad=10)
    axes[1].set_xlabel(r"$N_{shots}$")
    axes[1].set_ylabel("")
    axes[1].invert_yaxis()

    max_abs = max(abs(np.nanmin(gap_data.values)), abs(np.nanmax(gap_data.values)))
    sns.heatmap(
        gap_data,
        ax=axes[2],
        annot=True,
        fmt=".3f",
        cmap="RdBu",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        cbar_kws={"label": r"$\Delta$ (EV − Traj)"},
        linewidths=0.5,
    )
    axes[2].set_title("Performance Gap", pad=10)
    axes[2].set_xlabel(r"$N_{shots}$")
    axes[2].set_ylabel("")
    axes[2].invert_yaxis()

    plt.suptitle(f"Circuit-QRC: Mackey–Glass (Fixed budget S={cfg.S_budget})", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dashboard_overview.png"), dpi=300, bbox_inches="tight")
    plt.close()


def circuit_plot_phase_contour(df_stats: pd.DataFrame, cfg: CircuitConfig, outdir: str) -> None:
    shots_list, noise_list = cfg.shots_list, cfg.noise_list
    gap = circuit_pivot(df_stats, "gap_mean", shots_list, noise_list).values
    X_grid, Y_grid = np.meshgrid(shots_list, noise_list)

    fig, ax = plt.subplots(figsize=(7, 6))
    limit = float(max(abs(np.nanmin(gap)), abs(np.nanmax(gap))))
    levels = np.linspace(-limit, limit, 100)

    cp = ax.contourf(X_grid, Y_grid, gap, levels=levels, cmap="RdBu", extend="both")
    ax.contour(X_grid, Y_grid, gap, levels=[0], colors="k", linewidths=1.5, linestyles="--")
    fig.colorbar(cp, ax=ax, label=r"$\Delta$ NRMSE (EV − Traj)")

    ax.set_xlabel(r"Number of Shots ($N_{shots}$)")
    ax.set_ylabel(r"Measurement Noise ($\sigma$)")
    ax.set_title("Circuit-QRC phase diagram: Trajectory vs EV", pad=15)

    text_str = f"Budget S={cfg.S_budget}\nT ≈ S/N_shots"
    ax.text(
        0.95,
        0.95,
        text_str,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "phase_diagram_contour.png"), dpi=300)
    plt.close()


def circuit_plot_slices(df_runs: pd.DataFrame, cfg: CircuitConfig, outdir: str) -> None:
    """
    Uncertainty slices: plot Traj mean±std and EV mean at a few selected noise levels.

    (Minimal change from your original 'rigorous_slices' plot.)
    """
    import matplotlib.ticker as ticker

    shots_list = cfg.shots_list
    noise_list = cfg.noise_list

    fig, ax = plt.subplots(figsize=(8, 6))

    # pick 3 noise levels (low/mid/high)
    if len(noise_list) > 3:
        idxs = [0, len(noise_list) // 2, len(noise_list) - 1]
        sel_noise = [noise_list[i] for i in idxs]
    else:
        sel_noise = noise_list

    markers = ["o", "s", "^", "D"]

    for i, noise in enumerate(sel_noise):
        sub = df_runs[df_runs["noise"] == noise]
        stats = sub.groupby("shots").agg(
            traj_mean=("traj_nrmse", "mean"),
            traj_std=("traj_nrmse", "std"),
            ev_mean=("ev_nrmse", "mean"),
            ev_std=("ev_nrmse", "std"),
        ).reset_index()

        m = markers[i % len(markers)]

        # Trajectory (solid + band)
        ax.plot(stats["shots"], stats["traj_mean"], marker=m, linestyle="-", lw=2, label=f"Traj (σ={noise})")
        ax.fill_between(
            stats["shots"],
            stats["traj_mean"] - stats["traj_std"],
            stats["traj_mean"] + stats["traj_std"],
            alpha=0.15,
        )

        # EV (dashed + band)
        ax.plot(stats["shots"], stats["ev_mean"], linestyle="--", lw=1.8, alpha=0.8, label=f"EV (σ={noise})")
        ax.fill_between(
            stats["shots"],
            stats["ev_mean"] - stats["ev_std"],
            stats["ev_mean"] + stats["ev_std"],
            alpha=0.08,
        )

    ax.set_xlabel(r"Number of Shots ($N_{shots}$)")
    ax.set_ylabel("NRMSE (log scale)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_title("Circuit-QRC: scaling & uncertainty across seeds", pad=15)
    ax.legend(loc="best", fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "rigorous_slices.png"), dpi=300)
    plt.close()


def run_circuit_suite(cfg: CircuitConfig) -> None:
    if qml is None or sns is None:
        raise RuntimeError(
            "Circuit-QRC suite requires pennylane and seaborn. "
            "Install them or run with --mode tfim."
        )

    circuit_set_style()
    outdir = cfg.outdir
    ensure_dir(outdir)

    rows = []
    total_runs = cfg.n_seeds * len(cfg.shots_list) * len(cfg.noise_list)
    cnt = 0

    print("\n=== Circuit-QRC suite ===")
    print(f"Output:   {outdir}")
    print(f"Budget S: {cfg.S_budget}")
    print(f"Seeds:    {cfg.n_seeds}")
    print(f"Grid:     {len(cfg.noise_list)} noise × {len(cfg.shots_list)} shots = {len(cfg.noise_list)*len(cfg.shots_list)} points")

    for si in range(cfg.n_seeds):
        seed = 42 + si
        for shots in cfg.shots_list:
            for noise in cfg.noise_list:
                cnt += 1
                res = circuit_run_once(cfg, seed=seed, shots=shots, noise_level=noise)
                rows.append(
                    dict(
                        seed=seed,
                        shots=shots,
                        noise=noise,
                        ev_nrmse=res["ev"],
                        traj_nrmse=res["traj"],
                        gap=res["gap"],
                        T=res["T"],
                        S_budget=res["S"],
                    )
                )
                if cnt % 5 == 0 or cnt == total_runs:
                    print(f"\rProgress: {cnt}/{total_runs} | last gap={res['gap']:+.4f}", end="", flush=True)

    print("\nData generation complete. Aggregating + plotting...")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "simulation_data.csv"), index=False)

    # Aggregated stats across seeds
    def _win_rate(g):
        gg = np.asarray(g, dtype=float)
        gg = gg[np.isfinite(gg)]
        return float(np.mean(gg > 0)) if len(gg) else float("nan")

    df_stats = df.groupby(["shots", "noise"], as_index=False).agg(
        ev_mean=("ev_nrmse", "mean"),
        ev_std=("ev_nrmse", "std"),
        traj_mean=("traj_nrmse", "mean"),
        traj_std=("traj_nrmse", "std"),
        gap_mean=("gap", "mean"),
        gap_std=("gap", "std"),
        win_rate=("gap", _win_rate),
        n=("gap", "count"),
        T=("T", "mean"),
    )

    df_stats.to_csv(os.path.join(outdir, "summary_stats.csv"), index=False)

    # Core plots
    circuit_plot_individual_heatmaps(df_stats, cfg, outdir)
    if cfg.make_dashboard:
        circuit_plot_dashboard(df_stats, cfg, outdir)
    if cfg.make_phase_contour:
        circuit_plot_phase_contour(df_stats, cfg, outdir)
    if cfg.make_slices:
        circuit_plot_slices(df, cfg, outdir)

    print(f"Done. Artifacts in: {os.path.abspath(outdir)}")


# =============================================================================
# (B) Physical TFIM + weak measurement QRC suite
# =============================================================================

# --- Pauli operators ---
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
    dim = 2**n_qubits
    idx = 0
    for b in bitstring:
        idx = (idx << 1) | (1 if b == "1" else 0)
    ket = np.zeros((dim, 1), dtype=complex)
    ket[idx, 0] = 1.0
    return ket @ ket.conj().T


def tfim_generate_mackey_glass(
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
        exp_clip: float = 50.0,
    ):
        self.n = int(n_qubits)
        self.dim = 2 ** self.n

        self.dt = float(dt)
        self.kappa = float(kappa)
        self.noise_sigma = float(noise_sigma)
        self.measure_mode = str(measure_mode)
        self.input_axis = str(input_axis).upper()
        self.exp_clip = float(exp_clip)

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
        # numerical guard (minimal, preserves behavior in stable regime)
        ex = np.clip(s * a, -self.exp_clip, self.exp_clip)
        m = np.exp(ex)

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


@dataclass
class TFIMConfig:
    # reservoir / physics
    n_qubits: int = 4
    hx: float = 1.0
    hz: float = 0.0
    J: float = 1.0
    input_scale: float = 1.0
    dt: float = 0.1

    # measurement
    kappa: float = 0.5
    noise_sigma: float = 0.1
    measure_mode: str = "all"   # all, half, one, z0
    input_axis: str = "X"       # X/Y/Z
    init_state: str = "zero"    # mixed or zero

    # readout + features
    lags: int = 20
    ridge_alpha: float = 1e-2

    # data split
    washout: int = 50
    train_len: int = 300
    test_len: int = 500

    # shots / seeds
    n_shots: int = 5
    reservoir_seed: int = 17
    noise_seed: int = 1234
    series_seed: int = 42

    # analysis knobs
    standardize_features: bool = False  # keep default behavior identical to your working script


def tfim_config_from_json(path: str) -> TFIMConfig:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    # allow a plain dict (like your final_config.json)
    return TFIMConfig(**d)


def tfim_evaluate(cfg: TFIMConfig, return_series: bool = False) -> Dict[str, object]:
    """
    Run one experiment (one input stream, fixed seeds) and return metrics.
    """
    start = max(cfg.washout, cfg.lags)
    total_steps = start + cfg.train_len + cfg.test_len

    series = tfim_generate_mackey_glass(t_max=total_steps + 2000, seed=cfg.series_seed)
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

    # Trajectory-level features
    X_list = [tapped_features(currents[s], cfg.lags) for s in range(cfg.n_shots)]

    if cfg.standardize_features:
        sc_ev = StandardScaler().fit(X_ev[idx_tr])
        X_ev_tr = sc_ev.transform(X_ev[idx_tr])
        X_ev_te = sc_ev.transform(X_ev[idx_te])

        Xtr_stack = np.concatenate([X_list[s][idx_tr] for s in range(cfg.n_shots)], axis=0)
        sc_tr = StandardScaler().fit(Xtr_stack)
        Xtr_stack_sc = sc_tr.transform(Xtr_stack)
        Xte_sc = np.stack([sc_tr.transform(X_list[s][idx_te]) for s in range(cfg.n_shots)], axis=0)
    else:
        X_ev_tr = X_ev[idx_tr]
        X_ev_te = X_ev[idx_te]
        Xtr_stack = np.concatenate([X_list[s][idx_tr] for s in range(cfg.n_shots)], axis=0)
        Xtr_stack_sc = Xtr_stack
        Xte_sc = np.stack([X_list[s][idx_te] for s in range(cfg.n_shots)], axis=0)

    mod_ev = Ridge(alpha=cfg.ridge_alpha).fit(X_ev_tr, ytr)
    pred_ev = mod_ev.predict(X_ev_te)
    err_ev = nrmse(yte, pred_ev)

    # Trajectory training: stack shots
    ytr_stack = np.tile(ytr, cfg.n_shots)
    mod_traj = Ridge(alpha=cfg.ridge_alpha).fit(Xtr_stack_sc, ytr_stack)

    preds = np.stack([mod_traj.predict(Xte_sc[s]) for s in range(cfg.n_shots)], axis=0)
    pred_traj = preds.mean(axis=0)
    err_traj = nrmse(yte, pred_traj)

    # Persistence baseline (context only)
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


def tfim_compute_ev_traj_from_currents(
    currents: np.ndarray,
    cfg: TFIMConfig,
    u: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """
    Helper for phase diagram: compute EV/traj from a precomputed current tensor.
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

    X_list = [tapped_features(currents[s], cfg.lags) for s in range(N)]

    if cfg.standardize_features:
        sc_ev = StandardScaler().fit(X_ev[idx_tr])
        X_ev_tr = sc_ev.transform(X_ev[idx_tr])
        X_ev_te = sc_ev.transform(X_ev[idx_te])

        Xtr_stack = np.concatenate([X_list[s][idx_tr] for s in range(N)], axis=0)
        sc_tr = StandardScaler().fit(Xtr_stack)
        Xtr_stack_sc = sc_tr.transform(Xtr_stack)
        Xte_sc = np.stack([sc_tr.transform(X_list[s][idx_te]) for s in range(N)], axis=0)
    else:
        X_ev_tr = X_ev[idx_tr]
        X_ev_te = X_ev[idx_te]
        Xtr_stack = np.concatenate([X_list[s][idx_tr] for s in range(N)], axis=0)
        Xtr_stack_sc = Xtr_stack
        Xte_sc = np.stack([X_list[s][idx_te] for s in range(N)], axis=0)

    mod_ev = Ridge(alpha=cfg.ridge_alpha).fit(X_ev_tr, ytr)
    pred_ev = mod_ev.predict(X_ev_te)
    err_ev = nrmse(yte, pred_ev)

    ytr_stack = np.tile(ytr, N)
    mod_traj = Ridge(alpha=cfg.ridge_alpha).fit(Xtr_stack_sc, ytr_stack)
    preds = np.stack([mod_traj.predict(Xte_sc[s]) for s in range(N)], axis=0)
    pred_traj = preds.mean(axis=0)
    err_traj = nrmse(yte, pred_traj)

    return float(err_ev), float(err_traj)


# -----------------------------
# TFIM: reviewer-ready evaluation add-ons
# -----------------------------

def tfim_seed_sweep(
    base_cfg: TFIMConfig,
    out_dir: str,
    sweep_field: str,
    n_seeds: int,
    seed_start: int = 1000,
) -> pd.DataFrame:
    """
    Sweep one seed field (noise_seed / reservoir_seed / series_seed) while freezing all other params.

    Returns a DataFrame with one row per seed.
    """
    assert sweep_field in {"noise_seed", "reservoir_seed", "series_seed"}
    rows = []
    for k in range(n_seeds):
        val = seed_start + k
        cfg = replace(base_cfg, **{sweep_field: val})
        m = tfim_evaluate(cfg, return_series=False)
        rows.append({"k": k, sweep_field: val, **asdict(cfg), **m})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"seed_sweep_{sweep_field}.csv"), index=False)
    return df


def tfim_plot_seed_sweep(df: pd.DataFrame, out_dir: str, label: str) -> None:
    """
    Make 3 basic plots: EV vs Traj scatter, gap histogram, EV+Traj vs seed index.
    """
    # Progress vs seed index
    plt.figure(figsize=(7, 3.2))
    plt.plot(df["k"], df["err_ev"], "o-", label="NRMSE_EV")
    plt.plot(df["k"], df["err_traj"], "o-", label="NRMSE_traj")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Seed index")
    plt.ylabel("NRMSE")
    plt.title(f"TFIM seed sweep ({label}): NRMSE across seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"seed_sweep_{label}_progress.png"), dpi=170)
    plt.close()

    # Scatter
    plt.figure(figsize=(4.8, 4.4))
    plt.scatter(df["err_ev"], df["err_traj"], alpha=0.8)
    mn = float(min(df["err_ev"].min(), df["err_traj"].min()))
    mx = float(max(df["err_ev"].max(), df["err_traj"].max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
    plt.axvline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("NRMSE_EV")
    plt.ylabel("NRMSE_traj")
    plt.title(f"TFIM seed sweep ({label}): Trajectory vs EV")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"seed_sweep_{label}_scatter.png"), dpi=170)
    plt.close()

    # Gap histogram
    gaps = df["gap"].astype(float).values
    plt.figure(figsize=(6.2, 3.4))
    plt.hist(gaps[np.isfinite(gaps)], bins=20)
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Gap = EV − Traj")
    plt.ylabel("Count")
    plt.title(f"TFIM seed sweep ({label}): gap distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"seed_sweep_{label}_gap_hist.png"), dpi=170)
    plt.close()


def tfim_summarize_gap(df: pd.DataFrame, out_dir: str, label: str) -> Dict[str, object]:
    gaps = df["gap"].astype(float).values
    gaps = gaps[np.isfinite(gaps)]
    ev = df["err_ev"].astype(float).values
    tr = df["err_traj"].astype(float).values
    ev = ev[np.isfinite(ev)]
    tr = tr[np.isfinite(tr)]

    summary = {
        "label": label,
        "n": int(len(gaps)),
        "gap_mean": float(np.mean(gaps)) if len(gaps) else float("nan"),
        "gap_std": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else float("nan"),
        "gap_median": float(np.median(gaps)) if len(gaps) else float("nan"),
        "win_rate_gap_gt_0": float(np.mean(gaps > 0)) if len(gaps) else float("nan"),
        "p_signflip_two_sided": float(signflip_permutation_pvalue(gaps, seed=1)) if len(gaps) else float("nan"),
        "gap_ci95_bootstrap": bootstrap_ci(gaps, stat=np.mean, n_boot=4000, seed=2),
        "ev_mean": float(np.mean(ev)) if len(ev) else float("nan"),
        "traj_mean": float(np.mean(tr)) if len(tr) else float("nan"),
        "ev_ci95_bootstrap": bootstrap_ci(ev, stat=np.mean, n_boot=4000, seed=3),
        "traj_ci95_bootstrap": bootstrap_ci(tr, stat=np.mean, n_boot=4000, seed=4),
    }
    save_json(summary, os.path.join(out_dir, f"seed_sweep_{label}_summary.json"))
    return summary


def tfim_phase_diagram_with_uncertainty(
    base_cfg: TFIMConfig,
    out_dir: str,
    noise_list: List[float],
    shots_list: List[int],
    n_seeds: int = 10,
    seed_start: int = 2000,
    seed_field: str = "noise_seed",
) -> pd.DataFrame:
    """
    Build a phase diagram over (noise_sigma, n_shots) and aggregate across seeds.

    We keep the reservoir fixed (base_cfg.reservoir_seed) unless you choose seed_field="reservoir_seed".
    For each seed, we simulate max_shots once per noise value, then slice for each n_shots.
    """
    assert seed_field in {"noise_seed", "reservoir_seed", "series_seed"}

    ensure_dir(out_dir)
    max_shots = int(max(shots_list))

    # fixed dataset for phase diagram: same as tfim_evaluate (uses cfg.series_seed)
    start = max(base_cfg.washout, base_cfg.lags)
    total_steps = start + base_cfg.train_len + base_cfg.test_len
    series = tfim_generate_mackey_glass(t_max=total_steps + 2000, seed=base_cfg.series_seed)
    u = series[:total_steps]
    y = series[1 : total_steps + 1]

    rows = []
    for k in range(n_seeds):
        seed_val = seed_start + k
        cfg_seed = replace(base_cfg, **{seed_field: seed_val})
        for noise in noise_list:
            # instantiate QRC for this noise
            qrc = TFIM_QRC(
                n_qubits=cfg_seed.n_qubits,
                hx=cfg_seed.hx,
                hz=cfg_seed.hz,
                J=cfg_seed.J,
                input_scale=cfg_seed.input_scale,
                dt=cfg_seed.dt,
                kappa=cfg_seed.kappa,
                noise_sigma=float(noise),
                measure_mode=cfg_seed.measure_mode,
                input_axis=cfg_seed.input_axis,
                init_state=cfg_seed.init_state,
                seed=cfg_seed.reservoir_seed,
            )
            currents = qrc.simulate_currents(u, n_shots=max_shots, seed_base=cfg_seed.noise_seed)

            for N in shots_list:
                cur_slice = currents[: int(N)]
                err_ev, err_traj = tfim_compute_ev_traj_from_currents(cur_slice, cfg_seed, u, y)
                rows.append(
                    dict(
                        seed_index=k,
                        seed_field=seed_field,
                        seed_value=seed_val,
                        noise_sigma=float(noise),
                        n_shots=int(N),
                        err_ev=float(err_ev),
                        err_traj=float(err_traj),
                        gap=float(err_ev - err_traj),
                    )
                )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"phase_raw_{seed_field}.csv"), index=False)

    # Aggregate
    def _win_rate(g):
        gg = np.asarray(g, dtype=float)
        gg = gg[np.isfinite(gg)]
        return float(np.mean(gg > 0)) if len(gg) else float("nan")

    stats = df.groupby(["noise_sigma", "n_shots"], as_index=False).agg(
        gap_mean=("gap", "mean"),
        gap_std=("gap", "std"),
        win_rate=("gap", _win_rate),
        ev_mean=("err_ev", "mean"),
        traj_mean=("err_traj", "mean"),
        n=("gap", "count"),
    )
    stats.to_csv(os.path.join(out_dir, f"phase_stats_{seed_field}.csv"), index=False)

    # Plot: mean gap heatmap + contour
    pivot = stats.pivot(index="noise_sigma", columns="n_shots", values="gap_mean").sort_index()
    pivot = pivot.reindex(columns=sorted(shots_list))

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{v:.3f}" for v in pivot.index])
    ax.set_xlabel("n_shots")
    ax.set_ylabel("noise_sigma")
    ax.set_title("TFIM phase diagram: mean gap (EV − Traj)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean gap")

    # Contour for boundary (mean gap = 0)
    # Need grid in same order
    X, Y = np.meshgrid(np.arange(len(pivot.columns)), np.arange(len(pivot.index)))
    Z = pivot.values
    try:
        ax.contour(X, Y, Z, levels=[0.0], colors="k", linewidths=1.5, linestyles="--")
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"phase_mean_gap_{seed_field}.png"), dpi=180)
    plt.close()

    # Plot: win-rate heatmap
    pivot_w = stats.pivot(index="noise_sigma", columns="n_shots", values="win_rate").sort_index()
    pivot_w = pivot_w.reindex(columns=sorted(shots_list))

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(pivot_w.values, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(pivot_w.columns)))
    ax.set_xticklabels(pivot_w.columns)
    ax.set_yticks(np.arange(len(pivot_w.index)))
    ax.set_yticklabels([f"{v:.3f}" for v in pivot_w.index])
    ax.set_xlabel("n_shots")
    ax.set_ylabel("noise_sigma")
    ax.set_title("TFIM phase diagram: win rate P(gap>0)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="win rate")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"phase_win_rate_{seed_field}.png"), dpi=180)
    plt.close()

    # Plot: gap std heatmap
    pivot_s = stats.pivot(index="noise_sigma", columns="n_shots", values="gap_std").sort_index()
    pivot_s = pivot_s.reindex(columns=sorted(shots_list))

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(pivot_s.values, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(pivot_s.columns)))
    ax.set_xticklabels(pivot_s.columns)
    ax.set_yticks(np.arange(len(pivot_s.index)))
    ax.set_yticklabels([f"{v:.3f}" for v in pivot_s.index])
    ax.set_xlabel("n_shots")
    ax.set_ylabel("noise_sigma")
    ax.set_title("TFIM phase diagram: std(gap) across seeds")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="std gap")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"phase_gap_std_{seed_field}.png"), dpi=180)
    plt.close()

    return stats


def tfim_timeseries_artifact(cfg: TFIMConfig, out_dir: str, n_show: int = 300) -> None:
    res = tfim_evaluate(cfg, return_series=True)
    yte = np.asarray(res["yte"])
    pev = np.asarray(res["pred_ev"])
    ptr = np.asarray(res["pred_traj"])

    # CSV
    ts_df = pd.DataFrame({"t": np.arange(len(yte)), "target": yte, "pred_ev": pev, "pred_traj": ptr})
    ts_df.to_csv(os.path.join(out_dir, "final_timeseries.csv"), index=False)

    # plot
    n_show = min(int(n_show), len(yte))
    plt.figure(figsize=(9, 3.6))
    plt.plot(yte[:n_show], label="Target", lw=2, alpha=0.5)
    plt.plot(pev[:n_show], "--", label=f"EV (NRMSE={res['err_ev']:.3f})")
    plt.plot(ptr[:n_show], label=f"Traj (NRMSE={res['err_traj']:.3f})")
    plt.title("TFIM-QRC: Mackey–Glass one-step prediction (test segment)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "final_timeseries.png"), dpi=180)
    plt.close()


def tfim_ablation_sweep(
    base_cfg: TFIMConfig,
    out_dir: str,
    field: str,
    values: List[float],
    n_seeds: int = 15,
    seed_start: int = 3000,
) -> pd.DataFrame:
    """
    Small ablation sweep for reviewer robustness:
    - vary ridge_alpha or lags, and for each setting evaluate across many noise_seeds.

    Outputs:
      ablation_<field>.csv
      ablation_<field>_plot.png
    """
    assert field in {"ridge_alpha", "lags"}
    rows = []
    for v in values:
        cfg_v = replace(base_cfg, **{field: int(v) if field == "lags" else float(v)})
        df = tfim_seed_sweep(cfg_v, out_dir=out_dir, sweep_field="noise_seed", n_seeds=n_seeds, seed_start=seed_start)
        gaps = df["gap"].astype(float).values
        gaps = gaps[np.isfinite(gaps)]
        rows.append(
            dict(
                field=field,
                value=v,
                n=len(gaps),
                gap_mean=float(np.mean(gaps)) if len(gaps) else float("nan"),
                gap_std=float(np.std(gaps, ddof=1)) if len(gaps) > 1 else float("nan"),
                win_rate=float(np.mean(gaps > 0)) if len(gaps) else float("nan"),
            )
        )
        seed_start += 10000  # keep sweeps independent

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, f"ablation_{field}.csv"), index=False)

    # plot
    plt.figure(figsize=(6.5, 3.6))
    plt.errorbar(out["value"], out["gap_mean"], yerr=out["gap_std"], fmt="o-")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel(field)
    plt.ylabel("gap mean ± std")
    plt.title(f"TFIM ablation: effect of {field} on gap (EV − Traj)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"ablation_{field}.png"), dpi=170)
    plt.close()

    return out


# -----------------------------
# Optional TFIM coarse search (kept close to your working script)
# -----------------------------

def tfim_coarse_search_then_eval(
    out_dir: str,
    base: TFIMConfig,
) -> TFIMConfig:
    """
    Minimal reproduction of your candidate_mods search.
    Stops at the first configuration satisfying (EV<1) and (Traj<EV).
    Saves qrc_sweep_results.csv and iteration plots.

    Returns the found TFIMConfig.
    """
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "qrc_sweep_results.csv")

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
    solution_cfg: Optional[TFIMConfig] = None

    print("\n=== TFIM coarse search ===")
    for it, (label, mods) in enumerate(candidate_mods, start=1):
        cfg_dict = asdict(base)
        cfg_dict.update(mods)
        cfg = TFIMConfig(**cfg_dict)
        metrics = tfim_evaluate(cfg, return_series=False)

        row = {"iter": it, "label": label, **asdict(cfg), **metrics}
        rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        # plots (minimal)
        plt.figure(figsize=(7, 3.2))
        plt.plot(df["iter"], df["err_ev"], "o-", label="NRMSE_EV")
        plt.plot(df["iter"], df["err_traj"], "o-", label="NRMSE_traj")
        plt.axhline(1.0, linestyle="--", linewidth=1)
        plt.xlabel("Iteration")
        plt.ylabel("NRMSE")
        plt.title("TFIM coarse search progress")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "iter_progress.png"), dpi=160)
        plt.close()

        plt.figure(figsize=(4.8, 4.4))
        plt.scatter(df["err_ev"], df["err_traj"])
        mn = float(min(df["err_ev"].min(), df["err_traj"].min()))
        mx = float(max(df["err_ev"].max(), df["err_traj"].max()))
        plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
        plt.axvline(1.0, linestyle="--", linewidth=1)
        plt.xlabel("NRMSE_EV")
        plt.ylabel("NRMSE_traj")
        plt.title("TFIM coarse search: Trajectory vs EV")
        sol = df[df["cond_both"] == True]
        if len(sol) > 0:
            plt.scatter(sol["err_ev"], sol["err_traj"], s=140, marker="*", label="feasible")
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "iter_scatter_ev_vs_traj.png"), dpi=160)
        plt.close()

        if metrics["cond_both"]:
            solution_cfg = cfg
            print(f"✅ Found feasible solution at iteration {it}: {label}")
            break

    if solution_cfg is None:
        raise RuntimeError("No feasible configuration found in coarse search. Expand candidate_mods.")

    save_json(asdict(solution_cfg), os.path.join(out_dir, "final_config.json"))
    return solution_cfg


def run_tfim_suite(
    outdir: str,
    cfg: TFIMConfig,
    tfim_mode: str,
    seed_sweep_noise: int,
    seed_sweep_reservoir: int,
    seed_sweep_series: int,
    phase_seeds: int,
    phase_noise_list: List[float],
    phase_shots_list: List[int],
    ablation_ridge: List[float],
    ablation_lags: List[int],
) -> None:
    ensure_dir(outdir)
    print("\n=== TFIM-QRC suite ===")
    print(f"Output: {outdir}")

    if tfim_mode == "search_then_eval":
        cfg = tfim_coarse_search_then_eval(outdir, base=cfg)
    else:
        # always store the frozen config used
        save_json(asdict(cfg), os.path.join(outdir, "final_config_used.json"))

    # Single-run timeseries artifact (uses cfg seeds)
    tfim_timeseries_artifact(cfg, outdir)

    # Seed sweeps (robustness)
    if seed_sweep_noise > 0:
        df_noise = tfim_seed_sweep(cfg, out_dir=outdir, sweep_field="noise_seed", n_seeds=seed_sweep_noise, seed_start=5000)
        tfim_plot_seed_sweep(df_noise, outdir, label="noise_seed")
        tfim_summarize_gap(df_noise, outdir, label="noise_seed")

    if seed_sweep_reservoir > 0:
        df_res = tfim_seed_sweep(cfg, out_dir=outdir, sweep_field="reservoir_seed", n_seeds=seed_sweep_reservoir, seed_start=6000)
        tfim_plot_seed_sweep(df_res, outdir, label="reservoir_seed")
        tfim_summarize_gap(df_res, outdir, label="reservoir_seed")

    if seed_sweep_series > 0:
        df_ser = tfim_seed_sweep(cfg, out_dir=outdir, sweep_field="series_seed", n_seeds=seed_sweep_series, seed_start=7000)
        tfim_plot_seed_sweep(df_ser, outdir, label="series_seed")
        tfim_summarize_gap(df_ser, outdir, label="series_seed")

    # Phase diagram with uncertainty (seeded)
    if phase_seeds > 0:
        tfim_phase_diagram_with_uncertainty(
            base_cfg=cfg,
            out_dir=outdir,
            noise_list=phase_noise_list,
            shots_list=phase_shots_list,
            n_seeds=phase_seeds,
            seed_start=8000,
            seed_field="noise_seed",
        )

    # Minimal ablations
    if ablation_ridge:
        tfim_ablation_sweep(cfg, out_dir=outdir, field="ridge_alpha", values=ablation_ridge, n_seeds=min(15, max(seed_sweep_noise, 10)))
    if ablation_lags:
        tfim_ablation_sweep(cfg, out_dir=outdir, field="lags", values=ablation_lags, n_seeds=min(15, max(seed_sweep_noise, 10)))

    print(f"Done. Artifacts in: {os.path.abspath(outdir)}")


# =============================================================================
# Combined helper plot (optional)
# =============================================================================

def make_combined_summary_figure(outdir: str, circuit_dir: str, tfim_dir: str) -> None:
    """
    Create a very simple combined figure that shows:
    - Circuit mean gap phase diagram
    - TFIM mean gap phase diagram

    This is optional and only works if the expected files exist.
    """
    c_path = os.path.join(circuit_dir, "phase_diagram_contour.png")
    t_path = os.path.join(tfim_dir, "phase_mean_gap_noise_seed.png")

    if not (os.path.exists(c_path) and os.path.exists(t_path)):
        return

    import matplotlib.image as mpimg

    img1 = mpimg.imread(c_path)
    img2 = mpimg.imread(t_path)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))
    axes[0].imshow(img1)
    axes[0].axis("off")
    axes[0].set_title("Circuit-QRC (fair budget)")

    axes[1].imshow(img2)
    axes[1].axis("off")
    axes[1].set_title("TFIM-QRC (physical validation)")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "combined_phase_summary.png"), dpi=180)
    plt.close()


# =============================================================================
# CLI / presets
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="PRA-ready QRC EV vs Trajectory suite (Mackey–Glass).")
    ap.add_argument("--mode", type=str, default="all", choices=["all", "circuit", "tfim"], help="Which suite(s) to run.")
    ap.add_argument("--outdir", type=str, default="paper_artifacts", help="Root output directory.")
    ap.add_argument("--preset", type=str, default="paper", choices=["paper", "fast"], help="Parameter preset.")

    # --- Circuit knobs (keep close to your original script) ---
    ap.add_argument("--circuit_n_seeds", type=int, default=None)
    ap.add_argument("--circuit_shots_list", type=str, default=None)
    ap.add_argument("--circuit_noise_list", type=str, default=None)
    ap.add_argument("--circuit_S_budget", type=int, default=None)

    # --- TFIM knobs ---
    ap.add_argument("--tfim_config", type=str, default=None, help="Path to TFIM config JSON (e.g., final_config.json).")
    ap.add_argument("--tfim_mode", type=str, default="eval", choices=["eval", "search_then_eval"])
    ap.add_argument("--tfim_standardize", action="store_true", help="Standardize features before ridge (optional).")

    ap.add_argument("--tfim_seed_sweep_noise", type=int, default=None, help="How many noise_seed values to sweep.")
    ap.add_argument("--tfim_seed_sweep_reservoir", type=int, default=None)
    ap.add_argument("--tfim_seed_sweep_series", type=int, default=None)

    ap.add_argument("--tfim_phase_seeds", type=int, default=None)
    ap.add_argument("--tfim_phase_noise_list", type=str, default=None)
    ap.add_argument("--tfim_phase_shots_list", type=str, default=None)

    ap.add_argument("--tfim_ablation_ridge", type=str, default=None, help="Comma list of ridge_alpha values for ablation.")
    ap.add_argument("--tfim_ablation_lags", type=str, default=None, help="Comma list of lags values for ablation.")

    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    root_out = args.outdir
    ensure_dir(root_out)

    # -----------------------------
    # Presets
    # -----------------------------
    if args.preset == "paper":
        # Circuit: still heavy; choose a reviewer-friendly but realistic default.
        circuit_defaults = dict(
            n_seeds=10 if args.circuit_n_seeds is None else args.circuit_n_seeds,
            shots_list=parse_list(args.circuit_shots_list, int) if args.circuit_shots_list else [5, 10, 15, 20, 25, 30, 40, 50],
            noise_list=parse_list(args.circuit_noise_list, float) if args.circuit_noise_list else [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10],
            S_budget=args.circuit_S_budget if args.circuit_S_budget is not None else 10000,
        )
        # TFIM: robust evaluation defaults
        tfim_defaults = dict(
            seed_sweep_noise=50 if args.tfim_seed_sweep_noise is None else args.tfim_seed_sweep_noise,
            seed_sweep_reservoir=20 if args.tfim_seed_sweep_reservoir is None else args.tfim_seed_sweep_reservoir,
            seed_sweep_series=20 if args.tfim_seed_sweep_series is None else args.tfim_seed_sweep_series,
            phase_seeds=15 if args.tfim_phase_seeds is None else args.tfim_phase_seeds,
            phase_noise_list=parse_list(args.tfim_phase_noise_list, float) if args.tfim_phase_noise_list else [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30],
            phase_shots_list=parse_list(args.tfim_phase_shots_list, int) if args.tfim_phase_shots_list else [1, 2, 3, 5, 8, 10],
            ablation_ridge=parse_list(args.tfim_ablation_ridge, float) if args.tfim_ablation_ridge else [1e-4, 1e-2, 1e0],
            ablation_lags=parse_list(args.tfim_ablation_lags, int) if args.tfim_ablation_lags else [10, 20, 40],
        )
    else:
        # fast preset
        circuit_defaults = dict(
            n_seeds=3 if args.circuit_n_seeds is None else args.circuit_n_seeds,
            shots_list=parse_list(args.circuit_shots_list, int) if args.circuit_shots_list else [5, 20, 50],
            noise_list=parse_list(args.circuit_noise_list, float) if args.circuit_noise_list else [0.0, 0.05, 0.10],
            S_budget=args.circuit_S_budget if args.circuit_S_budget is not None else 6000,
        )
        tfim_defaults = dict(
            seed_sweep_noise=15 if args.tfim_seed_sweep_noise is None else args.tfim_seed_sweep_noise,
            seed_sweep_reservoir=0 if args.tfim_seed_sweep_reservoir is None else args.tfim_seed_sweep_reservoir,
            seed_sweep_series=0 if args.tfim_seed_sweep_series is None else args.tfim_seed_sweep_series,
            phase_seeds=5 if args.tfim_phase_seeds is None else args.tfim_phase_seeds,
            phase_noise_list=parse_list(args.tfim_phase_noise_list, float) if args.tfim_phase_noise_list else [0.05, 0.10, 0.20, 0.30],
            phase_shots_list=parse_list(args.tfim_phase_shots_list, int) if args.tfim_phase_shots_list else [1, 3, 5, 10],
            ablation_ridge=parse_list(args.tfim_ablation_ridge, float) if args.tfim_ablation_ridge else [],
            ablation_lags=parse_list(args.tfim_ablation_lags, int) if args.tfim_ablation_lags else [],
        )

    # -----------------------------
    # Build configs
    # -----------------------------
    circuit_cfg = CircuitConfig(
        outdir=os.path.join(root_out, "circuit_qrc"),
        n_seeds=circuit_defaults["n_seeds"],
        shots_list=circuit_defaults["shots_list"],
        noise_list=circuit_defaults["noise_list"],
        S_budget=circuit_defaults["S_budget"],
    )

    # TFIM config loading: prefer user-provided path; otherwise try ./final_config.json; otherwise defaults.
    tfim_cfg: TFIMConfig
    if args.tfim_config:
        tfim_cfg = tfim_config_from_json(args.tfim_config)
    else:
        local = "final_config.json"
        if os.path.exists(local):
            tfim_cfg = tfim_config_from_json(local)
        else:
            tfim_cfg = TFIMConfig()

    if args.tfim_standardize:
        tfim_cfg.standardize_features = True

    tfim_out = os.path.join(root_out, "tfim_qrc")

    # -----------------------------
    # Run
    # -----------------------------
    if args.mode in ("all", "circuit"):
        run_circuit_suite(circuit_cfg)

    if args.mode in ("all", "tfim"):
        run_tfim_suite(
            outdir=tfim_out,
            cfg=tfim_cfg,
            tfim_mode=args.tfim_mode,
            seed_sweep_noise=tfim_defaults["seed_sweep_noise"],
            seed_sweep_reservoir=tfim_defaults["seed_sweep_reservoir"],
            seed_sweep_series=tfim_defaults["seed_sweep_series"],
            phase_seeds=tfim_defaults["phase_seeds"],
            phase_noise_list=tfim_defaults["phase_noise_list"],
            phase_shots_list=tfim_defaults["phase_shots_list"],
            ablation_ridge=tfim_defaults["ablation_ridge"],
            ablation_lags=tfim_defaults["ablation_lags"],
        )

    # Optional combined summary figure
    if args.mode == "all":
        make_combined_summary_figure(root_out, circuit_cfg.outdir, tfim_out)

    print("\nALL DONE.")


if __name__ == "__main__":
    main()
