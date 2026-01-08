#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
THE ULTIMATE MACKEY-GLASS VISUALIZATION SUITE
---------------------------------------------
Generates:
1. Individual High-Res Heatmaps (EV, Traj, Gap)
2. Combined Dashboard (All 3 in one view)
3. Physics-Style Phase Diagram (Contour Plot)
4. Rigorous Uncertainty Analysis (Slices with Error Bands)

Method: QRC on Mackey-Glass with FAIR BUDGET constraints.
"""

import os
import argparse
import numpy as np
import pandas as pd
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# -----------------------------
# 1. GLOBAL STYLE SETTINGS (Publication Quality)
# -----------------------------
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',       # LaTeX-like math
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,              # High resolution for print
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'image.cmap': 'magma_r'         # Default cmap
})

# -----------------------------
# 2. CORE LOGIC (Mackey-Glass & QRC)
# -----------------------------
def mackey_glass(sample_len=4000, tau=17, seed=42):
    rng = np.random.default_rng(seed)
    delta_t = 1.0
    x = np.zeros(sample_len, dtype=np.float64)
    x[:tau] = 1.5
    for i in range(tau, sample_len):
        x_tau = x[i - tau]
        x[i] = x[i - 1] + delta_t * (0.2 * x_tau / (1.0 + x_tau**10) - 0.1 * x[i - 1])
    return x

def nrmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (np.std(y_true) + 1e-12)

def make_qnode(n_qubits, n_layers, shots):
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
    @qml.qnode(dev)
    def qnode(x_angles, params):
        for i in range(n_qubits):
            qml.RY(float(x_angles[i]), wires=i)
        for layer in range(n_layers):
            s = layer * 2 * n_qubits
            for i in range(n_qubits):
                qml.RY(float(params[s+i]), wires=i)
                qml.RZ(float(params[s+n_qubits+i]), wires=i)
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

def build_windows_traj(F, idx_list, L):
    T, shots, M = F.shape
    rows = []
    for t in idx_list:
        block = F[t - L + 1:t + 1]
        block = np.transpose(block, (1, 0, 2))
        block = block.reshape(shots, L * M)
        bias = np.ones((shots, 1), dtype=np.float64)
        rows.append(np.concatenate([block, bias], axis=1))
    return np.vstack(rows)

def build_windows_ev(F, idx_list, L):
    mu = F.mean(axis=1)
    rows = []
    for t in idx_list:
        block = mu[t - L + 1:t + 1].reshape(-1)
        rows.append(np.concatenate([block, [1.0]]))
    return np.array(rows, dtype=np.float64)

def apply_measurement_noise(feat_t, rng, noise_level):
    if noise_level <= 0: return feat_t
    p = float(np.clip(noise_level, 0.0, 0.25))
    flips = rng.random(size=feat_t.shape) < p
    feat = feat_t.copy()
    feat[flips] *= -1.0
    feat += float(noise_level) * rng.normal(size=feat.shape)
    return feat

def leaky_integrate(F, eta):
    if eta >= 1.0: return F
    if eta <= 0.0: return F * 0.0
    Ff = np.zeros_like(F)
    Ff[0] = F[0]
    for t in range(1, F.shape[0]):
        Ff[t] = (1.0 - eta) * Ff[t - 1] + eta * F[t]
    return Ff

# -----------------------------
# 3. EXECUTION LOOP
# -----------------------------
def run_once(cfg, seed, shots, noise_level):
    rng = np.random.default_rng(seed)
    T = int(cfg.S_budget // shots)
    if T < cfg.T_min:
        return np.nan, np.nan, np.nan, T, cfg.S_budget

    sample_len = max(cfg.sample_len, cfg.lookback + T + 2)
    data = mackey_glass(sample_len=sample_len, tau=cfg.tau, seed=seed)

    X_raw, y_raw = [], []
    for i in range(cfg.lookback, len(data) - 1):
        X_raw.append(data[i - cfg.lookback:i])
        y_raw.append(data[i])
    X_raw = np.array(X_raw)[:T]
    y_raw = np.array(y_raw)[:T]

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_all = scaler_X.fit_transform(X_raw)
    y_all = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

    inW = rng.normal(size=(cfg.n_qubits, cfg.lookback)) * cfg.in_scale
    params = rng.uniform(0, 2 * np.pi, size=(cfg.n_layers * 2 * cfg.n_qubits,))
    qnode = make_qnode(cfg.n_qubits, cfg.n_layers, shots)
    
    F = np.zeros((T, shots, 2*cfg.n_qubits))
    for t in range(T):
        x_angles = (inW @ X_all[t])
        meas_list = qnode(x_angles, params)
        feat_t = np.stack(meas_list, axis=1)
        F[t] = apply_measurement_noise(feat_t, rng, noise_level)
    
    F = leaky_integrate(F, cfg.eta)

    t0 = max(cfg.washout, cfg.L - 1)
    idx = np.arange(t0, T)
    if len(idx) < 10: return np.nan, np.nan, np.nan, T, cfg.S_budget

    ntr = int(cfg.train_frac * len(idx))
    idx_tr, idx_te = idx[:ntr], idx[ntr:]
    if len(idx_te) < 5: return np.nan, np.nan, np.nan, T, cfg.S_budget

    y_tr, y_te = y_all[idx_tr], y_all[idx_te]
    
    X_traj_tr = build_windows_traj(F, idx_tr, cfg.L)
    X_traj_te = build_windows_traj(F, idx_te, cfg.L)
    X_ev_tr = build_windows_ev(F, idx_tr, cfg.L)
    X_ev_te = build_windows_ev(F, idx_te, cfg.L)
    y_traj_tr = np.repeat(y_tr, shots)

    sc_traj, sc_ev = StandardScaler(), StandardScaler()
    ridge_traj, ridge_ev = Ridge(alpha=cfg.ridge_alpha), Ridge(alpha=cfg.ridge_alpha)

    ridge_traj.fit(sc_traj.fit_transform(X_traj_tr), y_traj_tr)
    ridge_ev.fit(sc_ev.fit_transform(X_ev_tr), y_tr)

    yhat_ev = ridge_ev.predict(sc_ev.transform(X_ev_te))
    yhat_traj = ridge_traj.predict(sc_traj.transform(X_traj_te)).reshape(len(idx_te), shots).mean(axis=1)

    return nrmse(y_te, yhat_ev), nrmse(y_te, yhat_traj), nrmse(y_te, yhat_ev)-nrmse(y_te, yhat_traj), T, cfg.S_budget

# -----------------------------
# 4. PLOTTING SUITE
# -----------------------------

def pivot_helper(df_mean, metric, shots_list, noise_list):
    pv = df_mean.pivot(index="noise", columns="shots", values=metric)
    pv = pv.reindex(index=noise_list, columns=shots_list)
    return pv

def plot_individual_heatmaps(df_mean, shots_list, noise_list, outdir):
    """Generates separate files for EV, Traj, and Gap (Improved Version)."""
    
    # Define the 3 maps to generate
    maps = [
        ("ev_nrmse_mean", "magma_r", "Baseline EV (NRMSE)", "heatmap_ev.png", None),
        ("traj_nrmse_mean", "magma_r", "Trajectory Method (NRMSE)", "heatmap_traj.png", None),
        ("gap_mean", "RdBu", r"Gap ($\Delta = EV - Traj$)", "heatmap_gap.png", 0)
    ]
    
    for metric, cmap, title, filename, center_val in maps:
        data = pivot_helper(df_mean, metric, shots_list, noise_list)
        
        plt.figure(figsize=(8, 6))
        
        # Determine color limits
        vmin, vmax = None, None
        if center_val is not None:
            max_abs = max(abs(data.min().min()), abs(data.max().max()))
            vmin, vmax = -max_abs, max_abs

        ax = sns.heatmap(data, annot=True, fmt=".3f", cmap=cmap, center=center_val,
                         vmin=vmin, vmax=vmax, linewidths=.5,
                         cbar_kws={'label': 'NRMSE' if 'Gap' not in title else 'Difference'})
        
        ax.set_title(r"\textbf{" + title + "}", pad=15)
        ax.set_xlabel(r"Number of Shots ($N_{shots}$)")
        ax.set_ylabel(r"Noise Level ($\sigma$)")
        ax.invert_yaxis() # Ensure standard Cartesian y-axis (low noise bottom)
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, filename), dpi=300)
        plt.close()
        print(f"Saved: {filename}")

def plot_dashboard_overview(df_mean, shots_list, noise_list, outdir, S_budget):
    """The 3-in-1 Dashboard."""
    ev_data = pivot_helper(df_mean, "ev_nrmse_mean", shots_list, noise_list)
    traj_data = pivot_helper(df_mean, "traj_nrmse_mean", shots_list, noise_list)
    gap_data = pivot_helper(df_mean, "gap_mean", shots_list, noise_list)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    
    # 1. EV
    sns.heatmap(ev_data, ax=axes[0], annot=True, fmt=".3f", cmap="magma_r", 
                cbar_kws={'label': 'NRMSE'}, linewidths=.5)
    axes[0].set_title(r"\textbf{Baseline: EV}", pad=10)
    axes[0].set_ylabel(r"Noise Level ($\sigma$)")
    axes[0].set_xlabel("N Shots")
    axes[0].invert_yaxis()

    # 2. Traj
    sns.heatmap(traj_data, ax=axes[1], annot=True, fmt=".3f", cmap="magma_r", 
                cbar_kws={'label': 'NRMSE'}, linewidths=.5)
    axes[1].set_title(r"\textbf{Method: Trajectory}", pad=10)
    axes[1].set_xlabel("N Shots")
    axes[1].set_ylabel("")
    axes[1].invert_yaxis()

    # 3. Gap
    max_abs = max(abs(gap_data.min().min()), abs(gap_data.max().max()))
    sns.heatmap(gap_data, ax=axes[2], annot=True, fmt=".3f", 
                cmap="RdBu", center=0, vmin=-max_abs, vmax=max_abs,
                cbar_kws={'label': r'$\Delta$ (Pos = Traj Wins)'}, linewidths=.5)
    axes[2].set_title(r"\textbf{Performance Gap}", pad=10)
    axes[2].set_xlabel("N Shots")
    axes[2].set_ylabel("")
    axes[2].invert_yaxis()

    plt.suptitle(f"Dashboard: Mackey-Glass QRC (Fixed Budget $S={S_budget}$)", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dashboard_overview.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: dashboard_overview.png")

def plot_phase_diagram_contour(df_mean, shots_list, noise_list, outdir, S_budget):
    """Physics-style Contour Plot."""
    X_grid, Y_grid = np.meshgrid(shots_list, noise_list)
    Z = pivot_helper(df_mean, "gap_mean", shots_list, noise_list).values

    fig, ax = plt.subplots(figsize=(7, 6))

    limit = max(abs(np.nanmin(Z)), abs(np.nanmax(Z)))
    levels = np.linspace(-limit, limit, 100)
    
    cp = ax.contourf(X_grid, Y_grid, Z, levels=levels, cmap="RdBu", extend='both')
    ax.contour(X_grid, Y_grid, Z, levels=[0], colors='k', linewidths=1.5, linestyles='--')

    cbar = fig.colorbar(cp, ax=ax, label=r"$\Delta$ NRMSE (EV $-$ Traj)")
    
    ax.set_xlabel(r"Number of Shots ($N_{shots}$)")
    ax.set_ylabel(r"Measurement Noise ($\sigma_{noise}$)")
    ax.set_title(r"\textbf{Phase Diagram: Trajectory vs. EV}", pad=15)
    
    # Annotation
    text_str = f"Budget $S={S_budget}$\n$T \\approx S / N_{{shots}}$"
    ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "phase_diagram_contour.png"), dpi=300)
    plt.close()
    print("Saved: phase_diagram_contour.png")

def plot_rigorous_slices(df, shots_list, noise_list, outdir):
    """Line plots with Standard Deviation Bands."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("viridis", len(noise_list))
    
    # Pick 3 representative noise levels for clarity
    if len(noise_list) > 3:
        indices = [0, len(noise_list)//2, len(noise_list)-1]
        sel_noise = [noise_list[i] for i in indices]
        sel_colors = [colors[i] for i in indices]
    else:
        sel_noise = noise_list
        sel_colors = colors

    markers = ['o', 's', '^', 'D']
    
    for i, noise in enumerate(sel_noise):
        sub = df[df['noise'] == noise]
        stats = sub.groupby('shots').agg(
            traj_mean=('traj_nrmse', 'mean'),
            traj_std=('traj_nrmse', 'std'),
            ev_mean=('ev_nrmse', 'mean')
        ).reset_index()
        
        c = sel_colors[i]
        m = markers[i % len(markers)]
        
        # Trajectory (Solid with Band)
        ax.plot(stats['shots'], stats['traj_mean'], label=f"Traj ($\sigma={noise}$)", 
                color=c, marker=m, linestyle='-', lw=2)
        ax.fill_between(stats['shots'], 
                        stats['traj_mean'] - stats['traj_std'], 
                        stats['traj_mean'] + stats['traj_std'], 
                        color=c, alpha=0.15)
        
        # EV (Dashed, Thinner)
        ax.plot(stats['shots'], stats['ev_mean'], 
                color=c, linestyle='--', alpha=0.6, lw=1.5)

    ax.set_xlabel(r"Number of Shots ($N_{shots}$)")
    ax.set_ylabel("NRMSE (Log Scale)")
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    
    ax.set_title(r"\textbf{Performance Scaling & Uncertainty}", pad=15)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', lw=2, linestyle='-'),
                    Line2D([0], [0], color='black', lw=1.5, linestyle='--')]
    
    leg1 = ax.legend(custom_lines, ['Trajectory Method', 'EV Baseline'], loc='upper center', 
                     bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    ax.legend(loc='best', fontsize=10, frameon=True, edgecolor='k')
    ax.add_artist(leg1)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(outdir, "rigorous_slices.png"), dpi=300)
    plt.close()
    print("Saved: rigorous_slices.png")

# -----------------------------
# 5. MAIN
# -----------------------------
def parse_list(s, cast=float):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="mg_ultimate_suite")

    # statistisch sauber, aber noch bezahlbar
    ap.add_argument("--n_seeds", type=int, default=2)

    # zeigt klar den Übergang: low / mid / high shots (und damit T klein)
    ap.add_argument("--shots_list", type=str, default="5,10,15,20,25,30,40,50")

    # fein genug für “phase boundary”, aber nicht zu groß
    ap.add_argument("--noise_list", type=str, default="0,0.01,0.02,0.03,0.05,0.08,0.10")

    # fair-budget so, dass auch 100 shots noch T>=200 ist (bei T_min=200)
    ap.add_argument("--S_budget", type=int, default=10000)

    
    # Defaults
    ap.add_argument("--n_qubits", type=int, default=6)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--lookback", type=int, default=3)
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--washout", type=int, default=30)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument("--in_scale", type=float, default=0.7)
    ap.add_argument("--eta", type=float, default=0.25)
    ap.add_argument("--tau", type=int, default=17)
    ap.add_argument("--sample_len", type=int, default=4000)
    ap.add_argument("--T_min", type=int, default=50)

    cfg = ap.parse_args()
    os.makedirs(cfg.outdir, exist_ok=True)

    shots_list = parse_list(cfg.shots_list, int)
    noise_list = parse_list(cfg.noise_list, float)
    
    class C: pass
    for k,v in vars(cfg).items(): setattr(C, k, v)
    
    print(f"--- STARTING ULTIMATE RUN ---")
    print(f"Budget S: {C.S_budget}")
    print(f"Seeds:    {C.n_seeds}")
    print(f"Output:   {cfg.outdir}")
    print(f"-----------------------------")
    
    rows = []
    total_runs = cfg.n_seeds * len(shots_list) * len(noise_list)
    cnt = 0
    
    for si in range(cfg.n_seeds):
        seed = 42 + si
        for shots in shots_list:
            for noise in noise_list:
                cnt += 1
                ev, traj, gap, T, S = run_once(C, seed, shots, noise)
                rows.append({
                    "seed": seed, "shots": shots, "noise": noise,
                    "ev_nrmse": ev, "traj_nrmse": traj, "gap_mean": gap
                })
                print(f"\rProgress: {cnt}/{total_runs} | Last Gap: {gap:.4f}", end="", flush=True)
    
    print("\n\nData generation complete. Generating High-End Visualizations...")
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "simulation_data.csv"), index=False)
    
    # Aggregate
    df_mean = df.groupby(["shots", "noise"], as_index=False).agg(
        ev_nrmse_mean=("ev_nrmse", "mean"),
        traj_nrmse_mean=("traj_nrmse", "mean"),
        gap_mean=("gap_mean", "mean")
    )
    df_mean.to_csv(os.path.join(cfg.outdir, "summary_mean.csv"), index=False)
    
    # --- CALL ALL PLOTTING FUNCTIONS ---
    
    # 1. The Classics (Improved)
    plot_individual_heatmaps(df_mean, shots_list, noise_list, cfg.outdir)
    
    # 2. The Dashboard
    plot_dashboard_overview(df_mean, shots_list, noise_list, cfg.outdir, C.S_budget)
    
    # 3. The Physics Plot
    plot_phase_diagram_contour(df_mean, shots_list, noise_list, cfg.outdir, C.S_budget)
    
    # 4. The Stats Plot
    plot_rigorous_slices(df, shots_list, noise_list, cfg.outdir)
    
    print(f"\nDone! Check folder '{cfg.outdir}' for all 6 graphics.")

if __name__ == "__main__":
    main()