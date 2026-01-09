#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pra_one_run_paper_figures_hiimpact.py

ONE-RUN pipeline for a PRA-style methods paper:

1) Exp. 1: Run the *fair-budget* sweep (via qrc_pra_master*.py) if needed.
2) Fig. 1 (a–c): Phase-diagram heatmaps (EV / Trajectory / Gap).
3) Fig. 2 (a–c): Mechanism via ρ_EV collapse:
   (a) Collapse curves Δ(ρ_EV; σ) with orange→blue styling (no viridis).
   (b) Extracted "trajectory-win window" [ρ⁻(σ), ρ⁺(σ)] where Δ>0 (design-rule summary).
   (c) Interpolated contour in collapsed coordinates (ρ_EV, σ) with orange-blue Δ colormap,
       WITHOUT black boundary line and WITHOUT black point overlay.
4) Fig. 3 (a–b): Optimal config from Fig. 1:
   (a) True + 4 trajectory seeds (one curve per seed) + one EV baseline curve (representative seed).
   (b) Paired seed-level NRMSE (EV vs Trajectory) + mean markers.

Outputs:
  - Sweep artifacts:   ./paper_artifacts/circuit_qrc/summary_stats.csv  (and more)
  - Paper figures:     ./paper_figures/fig1_phase_diagram.(png/pdf)
                      ./paper_figures/fig2_rho_collapse.(png/pdf)
                      ./paper_figures/fig3_timeseries_opt.(png/pdf)
  - Audit tables:      ./paper_figures/fig2_rho_collapse_points.csv
                      ./paper_figures/fig2_rho_crossings_table.csv
                      ./paper_figures/fig2_rho_winband_table.csv
                      ./paper_figures/fig3_timeseries_trace.csv
                      ./paper_figures/fig3_opt_config.json

Notes
-----
- Fig. 3 uses PennyLane (qml.sample). Install: pip install pennylane scikit-learn numpy pandas matplotlib
- If you already have a sweep CSV, pass --in_csv or --exp1_dir to skip the sweep.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap, Normalize

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Optional: PennyLane only needed for Fig.3
try:
    import pennylane as qml
except Exception:
    qml = None


# =============================================================================
# High-impact style (Nature/Science-like)
# =============================================================================

# Consistent method palette
COLOR_TRAJ = "#0072B2"   # deep blue (Trajectory)
COLOR_EV   = "#D55E00"   # vivid orange (EV baseline)

# UI colors
COLOR_TEXT = "#333333"
COLOR_GRID = "#E6E6E6"
COLOR_BG   = "#FFFFFF"
COLOR_ZERO = "#F7F7F7"


def set_high_impact_style() -> None:
    """Global matplotlib style."""
    plt.rcParams.update({
        # Typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "text.color": COLOR_TEXT,
        "axes.labelcolor": COLOR_TEXT,
        "axes.edgecolor": COLOR_TEXT,
        "xtick.color": COLOR_TEXT,
        "ytick.color": COLOR_TEXT,

        # Sizes
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,

        # Figure quality
        "figure.dpi": 300,
        "savefig.dpi": 300,

        # Lines / ticks
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,

        # Background
        "figure.facecolor": COLOR_BG,
        "axes.facecolor": COLOR_BG,
        "savefig.facecolor": COLOR_BG,
        "savefig.bbox": "tight",
    })


def style_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    """Open-axis look + subtle grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_TEXT)
    ax.spines["bottom"].set_color(COLOR_TEXT)
    ax.tick_params(axis="both", which="both", color=COLOR_TEXT)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    else:
        ax.grid(False)


def panel_label(ax: plt.Axes, label: str) -> None:
    """Clean panel label (a)/(b)/(c) in the upper-left, no box."""
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12, fontweight="bold", color=COLOR_TEXT,
    )


def cmap_seq(color_hex: str) -> LinearSegmentedColormap:
    """Sequential colormap from white -> color."""
    return LinearSegmentedColormap.from_list(f"seq_{color_hex}", [COLOR_ZERO, color_hex], N=256)


def cmap_div() -> LinearSegmentedColormap:
    """Diverging colormap EV(orange) -> white -> Traj(blue)."""
    return LinearSegmentedColormap.from_list("div_ev_traj", [COLOR_EV, COLOR_ZERO, COLOR_TRAJ], N=256)


def cmap_sigma_ob() -> LinearSegmentedColormap:
    """Sigma colormap: orange -> blue (no viridis)."""
    return LinearSegmentedColormap.from_list("sigma_orange_blue", [COLOR_EV, COLOR_TRAJ], N=256)


# =============================================================================
# IO helpers
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def infer_csv(exp1_dir: Path) -> Path:
    """Pick the best-guess CSV file from a directory."""
    for name in ["summary_stats.csv", "simulation_data.csv", "summary_mean.csv"]:
        p = exp1_dir / name
        if p.exists():
            return p
    cands = list(exp1_dir.glob("*.csv"))
    if len(cands) == 1:
        return cands[0]
    raise FileNotFoundError(
        f"Could not find a suitable CSV in {exp1_dir}. Expected summary_stats.csv or simulation_data.csv."
    )


def load_stats(csv_path: Path) -> pd.DataFrame:
    """Load aggregated stats (preferred) or raw per-run data; return aggregated table."""
    df = pd.read_csv(csv_path)

    if "Nshots" in df.columns and "shots" not in df.columns:
        df = df.rename(columns={"Nshots": "shots"})
    if "sigma" in df.columns and "noise" not in df.columns:
        df = df.rename(columns={"sigma": "noise"})

    if {"ev_mean", "traj_mean", "gap_mean"}.issubset(df.columns):
        out = df.copy()
        if "gap" in out.columns and "gap_mean" not in out.columns:
            out = out.rename(columns={"gap": "gap_mean"})
        return out

    if {"ev", "traj", "gap"}.issubset(df.columns) and "ev_mean" not in df.columns:
        return df.rename(columns={"ev": "ev_mean", "traj": "traj_mean", "gap": "gap_mean"}).copy()

    if not {"shots", "noise"}.issubset(df.columns):
        raise ValueError(f"CSV missing required columns. Found: {sorted(df.columns)}")

    if {"ev_nrmse", "traj_nrmse", "gap"}.issubset(df.columns):
        ev_col, tr_col, gap_col = "ev_nrmse", "traj_nrmse", "gap"
    elif {"ev", "traj", "gap"}.issubset(df.columns):
        ev_col, tr_col, gap_col = "ev", "traj", "gap"
    else:
        raise ValueError("Raw CSV must include (ev_nrmse, traj_nrmse, gap) or (ev, traj, gap).")

    def win_rate(g: pd.Series) -> float:
        gg = np.asarray(g, dtype=float)
        gg = gg[np.isfinite(gg)]
        return float(np.mean(gg > 0.0)) if len(gg) else float("nan")

    agg = (
        df.groupby(["shots", "noise"], as_index=False)
          .agg(
              ev_mean=(ev_col, "mean"),
              ev_std=(ev_col, "std"),
              traj_mean=(tr_col, "mean"),
              traj_std=(tr_col, "std"),
              gap_mean=(gap_col, "mean"),
              gap_std=(gap_col, "std"),
              win_rate=(gap_col, win_rate),
              n=(gap_col, "count"),
              T=("T", "mean") if "T" in df.columns else (gap_col, "count"),
          )
    )
    return agg


def parse_int_list(s: Optional[str], *, default: List[int]) -> List[int]:
    """Parse comma-separated ints, or return default if None/empty."""
    if s is None:
        return list(default)
    txt = str(s).strip()
    if not txt:
        return list(default)
    parts = [p.strip() for p in txt.split(",") if p.strip()]
    if not parts:
        return list(default)
    return [int(p) for p in parts]


# =============================================================================
# ρ_EV collapse helpers
# =============================================================================

def compute_rho_ev(
    df_stats: pd.DataFrame,
    *,
    alpha: float,
    n_qubits: int,
    L: int,
    washout: int,
    S_budget: int,
    use_T_from_csv: bool = True,
) -> pd.DataFrame:
    """Attach rho_EV to a stats table."""
    out = df_stats.copy()

    M = 2 * int(n_qubits)
    d = int(L) * int(M) + 1
    t0 = max(int(washout), int(L) - 1)

    if use_T_from_csv and "T" in out.columns and out["T"].notna().any():
        T = out["T"].astype(float)
    else:
        T = np.floor(float(S_budget) / out["shots"].astype(float))

    out["rho_ev"] = (float(alpha) / float(d)) * (T - float(t0))
    out["d_readout"] = d
    out["t0"] = t0
    out["T_eff"] = T
    return out


def find_zero_crossings(rho: np.ndarray, gap: np.ndarray) -> List[float]:
    """Interpolated rho values where gap crosses zero."""
    rho = np.asarray(rho, dtype=float)
    gap = np.asarray(gap, dtype=float)
    mask = np.isfinite(rho) & np.isfinite(gap)
    rho, gap = rho[mask], gap[mask]
    if len(rho) < 2:
        return []
    order = np.argsort(rho)
    rho, gap = rho[order], gap[order]

    out: List[float] = []
    for i in range(len(rho) - 1):
        g0, g1 = gap[i], gap[i + 1]
        if g0 == 0.0:
            out.append(float(rho[i]))
        if g0 * g1 < 0.0:
            r0, r1 = rho[i], rho[i + 1]
            rc = r0 + (r1 - r0) * (-g0) / (g1 - g0)
            out.append(float(rc))

    out = sorted(out)
    dedup: List[float] = []
    for v in out:
        if not dedup or abs(v - dedup[-1]) > 1e-6:
            dedup.append(v)
    return dedup


def _interp_zero(r0: float, g0: float, r1: float, g1: float) -> float:
    """Linear interpolation location where g crosses 0 between (r0,g0) and (r1,g1)."""
    if g1 == g0:
        return 0.5 * (r0 + r1)
    return r0 + (r1 - r0) * (-g0) / (g1 - g0)


def extract_positive_window(
    rho: np.ndarray,
    gap: np.ndarray,
    *,
    rho_ref: float = 1.0,
) -> Tuple[float, float]:
    """Extract one contiguous window [rho_left, rho_right] where gap>0.

    We pick the window that contains rho_ref if possible; otherwise, the closest window
    (tie-breaker: larger mean gap inside the window).

    Returns (nan, nan) if no positive region exists.
    """
    rho = np.asarray(rho, dtype=float)
    gap = np.asarray(gap, dtype=float)
    m = np.isfinite(rho) & np.isfinite(gap)
    rho, gap = rho[m], gap[m]
    if rho.size < 2:
        return (float("nan"), float("nan"))

    order = np.argsort(rho)
    rho, gap = rho[order], gap[order]
    pos = gap > 0.0
    if not np.any(pos):
        return (float("nan"), float("nan"))

    segments = []
    i = 0
    n = rho.size
    while i < n:
        if not pos[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and pos[j + 1]:
            j += 1

        # left boundary
        if i == 0:
            left = float(rho[i])
        else:
            left = float(_interp_zero(float(rho[i - 1]), float(gap[i - 1]), float(rho[i]), float(gap[i])))

        # right boundary
        if j == n - 1:
            right = float(rho[j])
        else:
            right = float(_interp_zero(float(rho[j]), float(gap[j]), float(rho[j + 1]), float(gap[j + 1])))

        if right < left:
            left, right = right, left

        contains = (left <= float(rho_ref) <= right)
        dist = 0.0 if contains else float(min(abs(rho_ref - left), abs(rho_ref - right)))
        mean_gap = float(np.mean(gap[i:j + 1])) if j >= i else -1e9

        # sort key: prefer contains, then smallest dist, then largest mean_gap
        segments.append((contains, dist, -mean_gap, left, right))
        i = j + 1

    segments.sort(key=lambda x: (not x[0], x[1], x[2]))
    return float(segments[0][3]), float(segments[0][4])


# =============================================================================
# Fig. 1: Phase-diagram heatmaps
# =============================================================================

def plot_fig1_phase_diagram(
    df_stats: pd.DataFrame,
    outdir: Path,
    *,
    S_budget: int,
    shots_order: Optional[List[int]] = None,
    noise_order: Optional[List[float]] = None,
) -> None:
    if shots_order is None:
        shots_order = sorted(df_stats["shots"].unique())
    if noise_order is None:
        noise_order = sorted(df_stats["noise"].unique())

    ev = df_stats.pivot(index="noise", columns="shots", values="ev_mean").reindex(index=noise_order, columns=shots_order)
    tr = df_stats.pivot(index="noise", columns="shots", values="traj_mean").reindex(index=noise_order, columns=shots_order)
    gap = df_stats.pivot(index="noise", columns="shots", values="gap_mean").reindex(index=noise_order, columns=shots_order)

    vmin_nrmse = float(np.nanmin([np.nanmin(ev.values), np.nanmin(tr.values)]))
    vmax_nrmse = float(np.nanmax([np.nanmax(ev.values), np.nanmax(tr.values)]))
    vmax_gap = float(np.nanmax(np.abs(gap.values)))
    if not np.isfinite(vmax_gap) or vmax_gap <= 0:
        vmax_gap = 1.0

    fig, axs = plt.subplots(1, 3, figsize=(14.8, 4.6))

    im0 = axs[0].imshow(ev.values, origin="lower", aspect="auto",
                        cmap=cmap_seq(COLOR_EV), vmin=vmin_nrmse, vmax=vmax_nrmse)
    axs[0].set_title("EV baseline")
    axs[0].set_xlabel(r"Shots $N_{\mathrm{shots}}$")
    axs[0].set_ylabel(r"Noise level $\sigma$")
    axs[0].set_xticks(np.arange(len(shots_order)))
    axs[0].set_xticklabels([str(s) for s in shots_order])
    axs[0].set_yticks(np.arange(len(noise_order)))
    axs[0].set_yticklabels([f"{x:g}" for x in noise_order])
    c0 = fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    c0.set_label("NRMSE")
    style_axis(axs[0], grid=False)
    panel_label(axs[0], "(a)")

    im1 = axs[1].imshow(tr.values, origin="lower", aspect="auto",
                        cmap=cmap_seq(COLOR_TRAJ), vmin=vmin_nrmse, vmax=vmax_nrmse)
    axs[1].set_title("Trajectory-level")
    axs[1].set_xlabel(r"Shots $N_{\mathrm{shots}}$")
    axs[1].set_xticks(np.arange(len(shots_order)))
    axs[1].set_xticklabels([str(s) for s in shots_order])
    axs[1].set_yticks(np.arange(len(noise_order)))
    axs[1].set_yticklabels([f"{x:g}" for x in noise_order])
    c1 = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    c1.set_label("NRMSE")
    style_axis(axs[1], grid=False)
    panel_label(axs[1], "(b)")

    im2 = axs[2].imshow(gap.values, origin="lower", aspect="auto",
                        cmap=cmap_div(), vmin=-vmax_gap, vmax=vmax_gap)
    axs[2].set_title(r"Gap $\Delta=\mathrm{NRMSE}_{EV}-\mathrm{NRMSE}_{traj}$")
    axs[2].set_xlabel(r"Shots $N_{\mathrm{shots}}$")
    axs[2].set_xticks(np.arange(len(shots_order)))
    axs[2].set_xticklabels([str(s) for s in shots_order])
    axs[2].set_yticks(np.arange(len(noise_order)))
    axs[2].set_yticklabels([f"{x:g}" for x in noise_order])
    c2 = fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    c2.set_label(r"$\Delta$")
    style_axis(axs[2], grid=False)
    panel_label(axs[2], "(c)")

    fig.suptitle(rf"Fig. 1: Fair-budget phase diagram ($S=N_{{shots}}\times T={S_budget}$)", y=1.02)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(outdir / f"fig1_phase_diagram.{ext}")
    plt.close(fig)


# =============================================================================
# Fig. 2: ρ_EV collapse (FIXED)
# =============================================================================

def plot_fig2_rho_collapse(
    df_rho: pd.DataFrame,
    outdir: Path,
    *,
    rho_ref: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    noises = sorted(df_rho["noise"].unique())

    df_rho_sorted = df_rho.sort_values(["noise", "rho_ev", "shots"]).reset_index(drop=True)
    df_rho_sorted.to_csv(outdir / "fig2_rho_collapse_points.csv", index=False)

    # Crossing table (audit)
    cross_rows = []
    for sigma in noises:
        sub = df_rho_sorted[df_rho_sorted["noise"] == sigma].sort_values("rho_ev")
        for j, rc in enumerate(find_zero_crossings(sub["rho_ev"].to_numpy(), sub["gap_mean"].to_numpy())):
            cross_rows.append({"noise": float(sigma), "cross_idx": int(j), "rho_cross": float(rc)})
    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(outdir / "fig2_rho_crossings_table.csv", index=False)

    # Extract trajectory-win window per sigma (design-rule summary for panel b)
    band_rows = []
    for sigma in noises:
        sub = df_rho_sorted[df_rho_sorted["noise"] == sigma].sort_values("rho_ev")
        left, right = extract_positive_window(
            sub["rho_ev"].to_numpy(dtype=float),
            sub["gap_mean"].to_numpy(dtype=float),
            rho_ref=float(rho_ref),
        )
        band_rows.append({
            "noise": float(sigma),
            "rho_left": float(left),
            "rho_right": float(right),
            "rho_width": float(right - left) if np.isfinite(left) and np.isfinite(right) else float("nan"),
        })
    band_df = pd.DataFrame(band_rows)
    band_df.to_csv(outdir / "fig2_rho_winband_table.csv", index=False)

    fig, axs = plt.subplots(1, 3, figsize=(15.2, 4.6))

    # -------------------------
    # (a) Lines: Δ(ρ_EV; σ) — orange→blue, no viridis
    # -------------------------
    ax = axs[0]
    sig_cmap = cmap_sigma_ob()
    sig_norm = Normalize(vmin=float(min(noises)) if noises else 0.0,
                         vmax=float(max(noises)) if noises else 1.0)

    for sigma in noises:
        sub = df_rho_sorted[df_rho_sorted["noise"] == sigma].sort_values("rho_ev")
        col = sig_cmap(sig_norm(float(sigma)))
        ax.plot(
            sub["rho_ev"],
            sub["gap_mean"],
            marker="o",
            lw=1.8,
            ms=4.0,
            color=col,
            label=fr"$\sigma$={sigma:g}",
        )
    ax.axhline(0.0, color=COLOR_TEXT, lw=1.0)
    ax.axvline(rho_ref, color=COLOR_TEXT, lw=1.0, ls="--")
    ax.set_xlabel(r"EV sample ratio $\rho_{EV}$")
    ax.set_ylabel(r"$\Delta=\mathrm{NRMSE}_{EV}-\mathrm{NRMSE}_{traj}$")
    ax.set_title(r"Collapse curves: $\Delta(\rho_{EV};\sigma)$")
    ax.legend(loc="best", frameon=False, ncol=2)
    style_axis(ax)
    panel_label(ax, "(a)")

    # -------------------------
    # (b) NEW: extracted win-window [ρ⁻(σ), ρ⁺(σ)]
    # -------------------------
    ax = axs[1]
    bf = band_df.copy()
    mask = np.isfinite(bf["rho_left"].to_numpy()) & np.isfinite(bf["rho_right"].to_numpy())
    bf = bf[mask].sort_values("noise")

    if len(bf) > 0:
        # Fill the window (trajectory-win region) in soft blue
        ax.fill_betweenx(
            bf["noise"].to_numpy(dtype=float),
            bf["rho_left"].to_numpy(dtype=float),
            bf["rho_right"].to_numpy(dtype=float),
            color=COLOR_TRAJ,
            alpha=0.14,
            linewidth=0,
            label=r"Window where $\Delta>0$",
        )
        # Left boundary in orange, right boundary in blue
        ax.plot(
            bf["rho_left"].to_numpy(dtype=float),
            bf["noise"].to_numpy(dtype=float),
            color=COLOR_EV,
            lw=1.8,
            marker="o",
            ms=4.0,
            label=r"$\rho^{-}(\sigma)$",
        )
        ax.plot(
            bf["rho_right"].to_numpy(dtype=float),
            bf["noise"].to_numpy(dtype=float),
            color=COLOR_TRAJ,
            lw=1.8,
            marker="o",
            ms=4.0,
            label=r"$\rho^{+}(\sigma)$",
            alpha=0.95,
        )

    ax.axvline(rho_ref, color=COLOR_TEXT, lw=1.0, ls="--")
    ax.set_xlabel(r"EV sample ratio $\rho_{EV}$")
    ax.set_ylabel(r"Noise level $\sigma$")
    ax.set_title(r"Extracted trajectory-win window")
    ax.legend(loc="best", frameon=False)
    style_axis(ax)
    panel_label(ax, "(b)")

    # -------------------------
    # (c) Interpolated contour — NO black lines, NO black points
    # -------------------------
    ax = axs[2]
    x = df_rho_sorted["rho_ev"].to_numpy(dtype=float)
    y = df_rho_sorted["noise"].to_numpy(dtype=float)
    z = df_rho_sorted["gap_mean"].to_numpy(dtype=float)

    tri = mtri.Triangulation(x, y)

    vmax = float(np.nanmax(np.abs(z)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    levels = np.linspace(-vmax, vmax, 15)

    cf = ax.tricontourf(tri, z, levels=levels, cmap=cmap_div(), extend="both")
    ax.set_xlabel(r"EV sample ratio $\rho_{EV}$")
    ax.set_ylabel(r"Noise level $\sigma$")
    ax.set_title(r"Trajectory-win band (collapsed coordinate)")
    cb2 = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb2.set_label(r"$\Delta$")

    style_axis(ax, grid=False)
    panel_label(ax, "(c)")

    # Harmonize x-limits
    rho_min = float(np.nanmin(df_rho_sorted["rho_ev"]))
    rho_max = float(np.nanmax(df_rho_sorted["rho_ev"]))
    pad = 0.05 * (rho_max - rho_min + 1e-12)
    for a in axs:
        a.set_xlim(rho_min - pad, rho_max + pad)

    fig.suptitle(r"Fig. 2: Mechanism / design rule via $\rho_{EV}$ collapse", y=1.02)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(outdir / f"fig2_rho_collapse.{ext}")
    plt.close(fig)

    return df_rho_sorted, cross_df, band_df


# =============================================================================
# Fig. 3: 4-seed overlay + paired NRMSE
# =============================================================================

@dataclass
class CircuitSimConfig:
    S_budget: int = 10000
    n_qubits: int = 6
    n_layers: int = 3
    lookback: int = 3
    L: int = 20
    washout: int = 30
    train_frac: float = 0.7

    ridge_alpha: float = 1.0
    eta: float = 0.25
    in_scale: float = 0.7

    tau: int = 17
    sample_len: int = 4000
    T_min: int = 50


def mackey_glass(sample_len: int = 4000, tau: int = 17, seed: int = 42) -> np.ndarray:
    delta_t = 1.0
    x = np.zeros(sample_len, dtype=np.float64)
    x[:tau] = 1.5
    rng = np.random.default_rng(seed)
    _ = rng.random()
    for i in range(tau, sample_len):
        x_tau = x[i - tau]
        x[i] = x[i - 1] + delta_t * (0.2 * x_tau / (1.0 + x_tau ** 10) - 0.1 * x[i - 1])
    return x


def nrmse_scalar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred)) / (np.std(y_true) + 1e-12)


def make_qnode(n_qubits: int, n_layers: int, shots: int, *, seed: Optional[int] = None):
    if qml is None:
        raise RuntimeError("PennyLane is required for Fig.3. Install: pip install pennylane")

    try:
        dev = qml.device("default.qubit", wires=n_qubits, shots=shots, seed=seed)
    except TypeError:
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
    rows = []
    for t in idx_list:
        block = F[t - L + 1: t + 1]
        block = np.transpose(block, (1, 0, 2))
        block = block.reshape(shots, L * M)
        rows.append(np.concatenate([block, np.ones((shots, 1))], axis=1))
    return np.vstack(rows)


def build_windows_ev(F: np.ndarray, idx_list: np.ndarray, L: int) -> np.ndarray:
    mu = F.mean(axis=1)
    rows = []
    for t in idx_list:
        block = mu[t - L + 1: t + 1].reshape(-1)
        rows.append(np.concatenate([block, [1.0]]))
    return np.array(rows, dtype=np.float64)


def simulate_predictions(
    cfg: CircuitSimConfig,
    *,
    reservoir_seed: int,
    data_seed: int,
    shots: int,
    noise_level: float,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(reservoir_seed)

    T = int(cfg.S_budget // shots)
    if T < cfg.T_min:
        raise ValueError(f"T too small (T={T}) for shots={shots} under S={cfg.S_budget}.")

    sample_len = max(cfg.sample_len, cfg.lookback + T + 2)
    data = mackey_glass(sample_len=sample_len, tau=cfg.tau, seed=data_seed)

    X_raw, y_raw = [], []
    for i in range(cfg.lookback, len(data) - 1):
        X_raw.append(data[i - cfg.lookback: i])
        y_raw.append(data[i])
    X_raw = np.array(X_raw)[:T]
    y_raw = np.array(y_raw)[:T]

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_all = scaler_X.fit_transform(X_raw)
    y_all = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

    inW = rng.normal(size=(cfg.n_qubits, cfg.lookback)) * cfg.in_scale
    params = rng.uniform(0, 2 * np.pi, size=(cfg.n_layers * 2 * cfg.n_qubits,))

    qnode = make_qnode(cfg.n_qubits, cfg.n_layers, shots, seed=int(reservoir_seed))

    F = np.zeros((T, shots, 2 * cfg.n_qubits), dtype=np.float64)
    for t in range(T):
        x_angles = (inW @ X_all[t])
        meas_list = qnode(x_angles, params)
        feat_t = np.stack(meas_list, axis=1)
        F[t] = apply_measurement_noise(feat_t, rng, noise_level)

    F = leaky_integrate(F, cfg.eta)

    t0 = max(cfg.washout, cfg.L - 1)
    idx = np.arange(t0, T)
    ntr = int(cfg.train_frac * len(idx))
    idx_tr, idx_te = idx[:ntr], idx[ntr:]

    y_tr, y_te = y_all[idx_tr], y_all[idx_te]

    X_traj_tr = build_windows_traj(F, idx_tr, cfg.L)
    X_traj_te = build_windows_traj(F, idx_te, cfg.L)
    X_ev_tr = build_windows_ev(F, idx_tr, cfg.L)
    X_ev_te = build_windows_ev(F, idx_te, cfg.L)

    y_traj_tr = np.repeat(y_tr, shots)

    sc_traj, sc_ev = StandardScaler(), StandardScaler()
    ridge_traj = Ridge(alpha=cfg.ridge_alpha)
    ridge_ev = Ridge(alpha=cfg.ridge_alpha)

    ridge_traj.fit(sc_traj.fit_transform(X_traj_tr), y_traj_tr)
    ridge_ev.fit(sc_ev.fit_transform(X_ev_tr), y_tr)

    yhat_ev = ridge_ev.predict(sc_ev.transform(X_ev_te))
    yhat_traj = ridge_traj.predict(sc_traj.transform(X_traj_te)).reshape(len(idx_te), shots).mean(axis=1)

    y_te_raw = scaler_y.inverse_transform(y_te.reshape(-1, 1)).ravel()
    yhat_ev_raw = scaler_y.inverse_transform(yhat_ev.reshape(-1, 1)).ravel()
    yhat_traj_raw = scaler_y.inverse_transform(yhat_traj.reshape(-1, 1)).ravel()

    return {
        "idx_te": idx_te.astype(int),
        "y_true": y_te_raw,
        "y_ev": yhat_ev_raw,
        "y_traj": yhat_traj_raw,
        "ev_nrmse": np.array([nrmse_scalar(y_te_raw, yhat_ev_raw)]),
        "traj_nrmse": np.array([nrmse_scalar(y_te_raw, yhat_traj_raw)]),
    }


def plot_fig3_seed_overlay(
    df_stats: pd.DataFrame,
    cfg: CircuitSimConfig,
    outdir: Path,
    *,
    criterion: str = "gap_mean",
    data_seed: int = 42,
    seed_list: List[int] = None,
    seed_ev: Optional[int] = None,
    n_show: int = 300,
) -> Dict[str, float]:
    if seed_list is None:
        seed_list = [0, 1, 2, 3]
    if seed_ev is None:
        seed_ev = int(seed_list[0])

    if criterion not in df_stats.columns:
        raise ValueError(f"criterion '{criterion}' not found in df_stats columns: {sorted(df_stats.columns)}")

    minimize_metrics = {"ev_mean", "traj_mean", "ev_std", "traj_std"}
    ascending = True if criterion in minimize_metrics else False
    opt = df_stats.sort_values(criterion, ascending=ascending).iloc[0]
    shots_opt = int(opt["shots"])
    noise_opt = float(opt["noise"])

    sims: Dict[int, Dict[str, np.ndarray]] = {}
    for s in seed_list:
        sims[int(s)] = simulate_predictions(
            cfg,
            reservoir_seed=int(s),
            data_seed=int(data_seed),
            shots=shots_opt,
            noise_level=noise_opt,
        )

    sim_ev = sims[int(seed_ev)]
    y_true = sims[int(seed_list[0])]["y_true"]

    n_show_eff = int(min(max(20, n_show), len(y_true)))
    t = np.arange(n_show_eff)

    ev_vals = np.array([float(sims[int(s)]["ev_nrmse"][0]) for s in seed_list], dtype=float)
    tr_vals = np.array([float(sims[int(s)]["traj_nrmse"][0]) for s in seed_list], dtype=float)
    gap_vals = ev_vals - tr_vals

    ev_mu = float(np.mean(ev_vals))
    tr_mu = float(np.mean(tr_vals))
    gap_mu = float(np.mean(gap_vals))

    rows = []
    for s in seed_list:
        d = sims[int(s)]
        for i in range(n_show_eff):
            rows.append({
                "seed": int(s),
                "t": int(i),
                "y_true": float(d["y_true"][i]),
                "y_ev": float(d["y_ev"][i]),
                "y_traj": float(d["y_traj"][i]),
            })
    pd.DataFrame(rows).to_csv(outdir / "fig3_timeseries_trace.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(13.8, 4.6), gridspec_kw={"width_ratios": [2.2, 1.0]})

    ax = axs[0]
    ax.plot(t, y_true[:n_show_eff], lw=2.2, color=COLOR_TEXT, label="True")
    ax.plot(t, sim_ev["y_ev"][:n_show_eff], lw=1.8, ls="--", color=COLOR_EV,
            label=fr"EV (seed={seed_ev})")

    for i, s in enumerate(seed_list):
        d = sims[int(s)]
        alpha = 0.35 + 0.45 * (i / max(1, len(seed_list) - 1))
        ax.plot(
            t,
            d["y_traj"][:n_show_eff],
            lw=1.9,
            color=COLOR_TRAJ,
            alpha=float(alpha),
            label=(fr"Trajectory (seeds)" if i == 0 else None),
        )

    ax.set_xlabel("Test time index")
    ax.set_ylabel("Mackey–Glass value")
    ax.set_title("Mackey–Glass prediction (4 seeds)")
    ax.legend(loc="best", frameon=False)
    style_axis(ax)
    panel_label(ax, "(a)")

    ax = axs[1]
    x0, x1 = 0.0, 1.0
    for i, s in enumerate(seed_list):
        y0 = float(ev_vals[i])
        y1 = float(tr_vals[i])
        ax.plot([x0, x1], [y0, y1], color=COLOR_GRID, lw=1.2, zorder=1)
        ax.scatter([x0], [y0], s=55, color=COLOR_EV, edgecolors=COLOR_TEXT, linewidths=0.4, zorder=2)
        ax.scatter([x1], [y1], s=55, color=COLOR_TRAJ, edgecolors=COLOR_TEXT, linewidths=0.4, zorder=3)

    ax.scatter([x0], [ev_mu], s=140, color=COLOR_EV, edgecolors=COLOR_TEXT, linewidths=0.8, zorder=4)
    ax.scatter([x1], [tr_mu], s=140, color=COLOR_TRAJ, edgecolors=COLOR_TEXT, linewidths=0.8, zorder=5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["EV", "Trajectory"])
    ax.set_ylabel("NRMSE")
    ax.set_title("Seed-level NRMSE (paired)")
    ax.text(
        0.5, 0.98,
        rf"$N_{{shots}}$={shots_opt}, $\sigma$={noise_opt:g}" + "\n"
        + rf"$\Delta$ (mean over seeds) = {gap_mu:+.3f}",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=10,
        color=COLOR_TEXT,
    )
    style_axis(ax)
    panel_label(ax, "(b)")

    fig.suptitle("Fig. 3: Optimal configuration (4 seeds)", y=1.02)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(outdir / f"fig3_timeseries_opt.{ext}")
    plt.close(fig)

    info = {
        "criterion": str(criterion),
        "shots_opt": int(shots_opt),
        "noise_opt": float(noise_opt),
        "fig3_data_seed": int(data_seed),
        "fig3_seed_list": [int(s) for s in seed_list],
        "fig3_seed_ev": int(seed_ev),
        "fig3_n_show": int(n_show),
        "fig3_ridge_alpha": float(cfg.ridge_alpha),
        "fig3_eta": float(cfg.eta),
        "fig3_in_scale": float(cfg.in_scale),
        "fig3_tau": int(cfg.tau),
        "ev_nrmse_per_seed": [float(x) for x in ev_vals],
        "traj_nrmse_per_seed": [float(x) for x in tr_vals],
        "gap_per_seed": [float(x) for x in gap_vals],
        "ev_mean": float(ev_mu),
        "traj_mean": float(tr_mu),
        "gap_mean": float(gap_mu),
    }
    with open(outdir / "fig3_opt_config.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    return info


# =============================================================================
# Sweep runner (Exp. 1)
# =============================================================================

def path_depth(p: Path) -> int:
    try:
        return len(p.resolve().parts)
    except Exception:
        return len(p.parts)


def candidate_search_roots(run_dir: Path) -> List[Path]:
    run_dir = run_dir.resolve()
    roots: List[Path] = [run_dir, run_dir.parent, run_dir.parent.parent, Path(__file__).resolve().parent]
    for r in list(roots):
        roots.append(r / "qrc_budget-main")
        roots.append(r / "qrc_budget-main" / "qrc_budget-main")
    out: List[Path] = []
    seen: set = set()
    for r in roots:
        try:
            rr = r.resolve()
        except Exception:
            rr = r
        if rr in seen:
            continue
        seen.add(rr)
        if rr.exists() and rr.is_dir():
            out.append(rr)
    return out


def find_qrc_pra_master(sweep_script: Optional[str], run_dir: Path) -> Path:
    if sweep_script:
        p = Path(sweep_script).expanduser()
        if p.is_dir():
            cands = sorted(p.glob("qrc_pra_master*.py"))
            if not cands:
                cands = sorted(p.glob("**/qrc_pra_master*.py"))
            if not cands:
                raise FileNotFoundError(f"No qrc_pra_master*.py found under: {p}")
            return cands[0].resolve()
        if not p.exists():
            raise FileNotFoundError(f"--sweep_script not found: {p}")
        return p.resolve()

    roots = candidate_search_roots(run_dir)
    cands: List[Path] = []
    for r in roots:
        cands += list(r.glob("qrc_pra_master*.py"))
    if not cands:
        for r in roots:
            cands += list(r.glob("**/qrc_pra_master*.py"))

    cands = [c for c in cands if c.is_file() and c.name.startswith("qrc_pra_master")]
    if not cands:
        raise FileNotFoundError(
            "Could not auto-find qrc_pra_master*.py. "
            "Run this script from the repo directory or pass --sweep_script /path/to/qrc_pra_master.py"
        )
    cands = sorted(cands, key=lambda p: (path_depth(p), len(str(p))))
    return cands[0].resolve()


def run_circuit_sweep(
    *,
    pra_master_path: Path,
    artifacts_root: Path,
    preset: str,
    S_budget: int,
    circuit_n_seeds: Optional[int],
    circuit_shots_list: Optional[str],
    circuit_noise_list: Optional[str],
    verbose: bool,
) -> None:
    ensure_dir(artifacts_root)

    cmd: List[str] = [
        sys.executable,
        str(pra_master_path),
        "--mode", "circuit",
        "--preset", str(preset),
        "--outdir", str(artifacts_root),
        "--circuit_S_budget", str(int(S_budget)),
    ]
    if circuit_n_seeds is not None:
        cmd += ["--circuit_n_seeds", str(int(circuit_n_seeds))]
    if circuit_shots_list is not None:
        cmd += ["--circuit_shots_list", str(circuit_shots_list)]
    if circuit_noise_list is not None:
        cmd += ["--circuit_noise_list", str(circuit_noise_list)]

    if verbose:
        print("\n[Exp.1] Running circuit sweep:")
        print("  ", " ".join(cmd))
        print(f"  cwd = {pra_master_path.parent}")
        print(f"  out = {artifacts_root}")

    subprocess.run(cmd, check=True, cwd=str(pra_master_path.parent))


def ensure_exp1_csv(
    *,
    run_dir: Path,
    artifacts_root: Path,
    preset: str,
    S_budget: int,
    force_sweep: bool,
    skip_sweep: bool,
    sweep_script: Optional[str],
    circuit_n_seeds: Optional[int],
    circuit_shots_list: Optional[str],
    circuit_noise_list: Optional[str],
    verbose: bool,
) -> Path:
    exp1_dir = artifacts_root / "circuit_qrc"
    expected = exp1_dir / "summary_stats.csv"

    if skip_sweep:
        if exp1_dir.exists():
            return infer_csv(exp1_dir).resolve()
        raise FileNotFoundError("--skip_sweep set, but artifacts directory does not exist.")

    if expected.exists() and not force_sweep:
        if verbose:
            print(f"[Exp.1] Found existing sweep CSV: {expected}")
        return expected.resolve()

    pra_master_path = find_qrc_pra_master(sweep_script, run_dir)
    run_circuit_sweep(
        pra_master_path=pra_master_path,
        artifacts_root=artifacts_root,
        preset=preset,
        S_budget=S_budget,
        circuit_n_seeds=circuit_n_seeds,
        circuit_shots_list=circuit_shots_list,
        circuit_noise_list=circuit_noise_list,
        verbose=verbose,
    )

    if expected.exists():
        return expected.resolve()
    if exp1_dir.exists():
        return infer_csv(exp1_dir).resolve()

    raise FileNotFoundError(f"Sweep finished but no CSV found in {exp1_dir}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="ONE-RUN: run Exp.1 sweep if needed, then generate Fig.1–3 with high-impact styling."
    )

    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--in_csv", type=str, default=None, help="Use this CSV and skip the sweep.")
    src.add_argument("--exp1_dir", type=str, default=None, help="Use CSV from this directory and skip the sweep.")

    ap.add_argument("--run_dir", type=str, default=".", help="Base directory (default: current directory).")
    ap.add_argument("--artifacts_dir", type=str, default="paper_artifacts", help="Sweep outputs root (default: paper_artifacts).")
    ap.add_argument("--outdir", type=str, default="paper_figures", help="Figure output dir (default: paper_figures).")

    ap.add_argument("--preset", type=str, default="paper", choices=["paper", "fast"], help="Sweep preset for qrc_pra_master.")
    ap.add_argument("--force_sweep", action="store_true", help="Re-run the sweep even if CSV exists.")
    ap.add_argument("--skip_sweep", action="store_true", help="Do not run sweep; require --in_csv or --exp1_dir.")
    ap.add_argument("--sweep_script", type=str, default=None, help="Path to qrc_pra_master*.py (optional).")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")

    # Budget / collapse parameters
    ap.add_argument("--S_budget", type=int, default=10000, help="Fair budget S (default: 10000).")
    ap.add_argument("--train_frac", type=float, default=0.7, help="Train fraction alpha (default: 0.7).")
    ap.add_argument("--n_qubits", type=int, default=6, help="Qubits Q (default: 6).")
    ap.add_argument("--L", type=int, default=20, help="Window length L (default: 20).")
    ap.add_argument("--washout", type=int, default=30, help="Washout steps (default: 30).")
    ap.add_argument("--rho_ref", type=float, default=1.0, help="Reference line ρ_EV (default: 1).")

    # Sweep overrides
    ap.add_argument("--circuit_n_seeds", type=int, default=None, help="Override sweep seeds.")
    ap.add_argument("--circuit_shots_list", type=str, default=None, help="Override shots list, e.g. '5,10,20,25,30,50'.")
    ap.add_argument("--circuit_noise_list", type=str, default=None, help="Override noise list, e.g. '0,0.02,0.05,0.1'.")

    # Fig.3 settings
    ap.add_argument("--skip_fig3", action="store_true", help="Skip Fig.3 (no PennyLane required).")
    ap.add_argument("--criterion", type=str, default="gap_mean", help="How to pick optimal config (default: gap_mean).")
    ap.add_argument("--fig3_n_show", type=int, default=300, help="Points to show in time series (default: 300).")
    ap.add_argument("--fig3_data_seed", type=int, default=42, help="Fixed MG task seed for Fig.3 (default: 42).")
    ap.add_argument("--fig3_seeds", type=str, default="0,1,2,3", help="Reservoir seeds for Fig.3 overlay (default: 0,1,2,3).")
    ap.add_argument("--fig3_seed_ev", type=int, default=None, help="Representative seed for EV curve in Fig.3(a).")

    # Fig.3 hyperparameters
    ap.add_argument("--fig3_ridge_alpha", type=float, default=1.0, help="Ridge alpha for Fig.3 (default: 1.0).")
    ap.add_argument("--fig3_eta", type=float, default=0.25, help="Leak rate η for Fig.3 (default: 0.25).")
    ap.add_argument("--fig3_in_scale", type=float, default=0.7, help="Input scale for Fig.3 (default: 0.7).")
    ap.add_argument("--fig3_tau", type=int, default=17, help="Mackey-Glass delay tau for Fig.3 (default: 17).")

    args = ap.parse_args()

    set_high_impact_style()

    run_dir = Path(args.run_dir).expanduser().resolve()
    artifacts_root = (run_dir / args.artifacts_dir).resolve()
    outdir = (run_dir / args.outdir).resolve()
    ensure_dir(outdir)

    if args.in_csv:
        csv_path = Path(args.in_csv).expanduser().resolve()
    elif args.exp1_dir:
        csv_path = infer_csv(Path(args.exp1_dir).expanduser().resolve())
    else:
        csv_path = ensure_exp1_csv(
            run_dir=run_dir,
            artifacts_root=artifacts_root,
            preset=args.preset,
            S_budget=args.S_budget,
            force_sweep=args.force_sweep,
            skip_sweep=args.skip_sweep,
            sweep_script=args.sweep_script,
            circuit_n_seeds=args.circuit_n_seeds,
            circuit_shots_list=args.circuit_shots_list,
            circuit_noise_list=args.circuit_noise_list,
            verbose=args.verbose,
        )

    if args.verbose:
        print(f"[IO] Using stats CSV: {csv_path}")

    df_stats = load_stats(csv_path)

    # Fig.1
    plot_fig1_phase_diagram(df_stats, outdir, S_budget=args.S_budget)

    # Fig.2
    df_rho = compute_rho_ev(
        df_stats,
        alpha=args.train_frac,
        n_qubits=args.n_qubits,
        L=args.L,
        washout=args.washout,
        S_budget=args.S_budget,
        use_T_from_csv=True,
    )
    plot_fig2_rho_collapse(df_rho, outdir, rho_ref=args.rho_ref)

    # Fig.3
    if not args.skip_fig3:
        seeds = parse_int_list(args.fig3_seeds, default=[0, 1, 2, 3])
        seed_ev = args.fig3_seed_ev if args.fig3_seed_ev is not None else int(seeds[0])

        cfg = CircuitSimConfig(
            S_budget=args.S_budget,
            n_qubits=args.n_qubits,
            n_layers=3,
            lookback=3,
            L=args.L,
            washout=args.washout,
            train_frac=args.train_frac,
            ridge_alpha=float(args.fig3_ridge_alpha),
            eta=float(args.fig3_eta),
            in_scale=float(args.fig3_in_scale),
            tau=int(args.fig3_tau),
        )
        plot_fig3_seed_overlay(
            df_stats,
            cfg,
            outdir,
            criterion=args.criterion,
            data_seed=int(args.fig3_data_seed),
            seed_list=seeds,
            seed_ev=int(seed_ev),
            n_show=int(args.fig3_n_show),
        )

    if args.verbose:
        print(f"[DONE] Figures written to: {outdir}")


if __name__ == "__main__":
    main()
