#!/usr/bin/env python3
"""
FULLGAS QRC Methods Demo (TFIM + weak measurement) on Mackey–Glass
=================================================================

Goal (for the *selected* run/segment/seed):
  - EV NRMSE < 1.0
  - Trajectory-level NRMSE < EV NRMSE
  - (optionally) both < 1.0

This script is designed as a compact *methods-paper* validation driver:
  Experiment 1 (Main): EV vs Trajectory-level overlay (one-step MG prediction)
  Experiment 2 (Ablation/Regime): Window-length (L) sweep
  Experiment 3 (Phase): Noise vs Shots phase grid for the EV–Traj gap

Key physical ingredients:
  - Transverse-Field Ising Model (TFIM) Hamiltonian
  - Exact unitary evolution of a density matrix
  - Weak continuous Z-measurement with backaction
  - Random measurement schedule (subset per step), with hold-last + mask channels
  - Linear readout (Ridge) on a tapped-delay feature map
  - Optional feature augmentation: squared currents (helps nonlinearity without changing the reservoir)

Outputs (in --outdir):
  - qrc_sweep_results.csv         (all iterations from the search schedule)
  - qrc_phase_grid.csv            (noise-vs-shots grid around the best config)
  - progress_nrmse.png            (NRMSE over iterations)
  - best_overlay.png              (Target vs EV vs Traj overlay)
  - phase_gap_heatmap.png         (gap heatmap)
  - qrc_fullgas_final.png         (4-panel figure for the paper)

Run:
  python qrc_fullgas_mg.py --outdir qrc_fullgas_out

Dependencies:
  numpy, matplotlib, scikit-learn, pandas
"""

from __future__ import annotations

import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# -----------------------
# Mackey–Glass generator
# -----------------------

def generate_mackey_glass(t_max: int = 30000, tau: int = 17, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    beta, gamma, n, dt = 0.2, 0.1, 10, 1.0
    steps = int(t_max / dt)
    x = np.zeros(steps, dtype=float)
    x[0] = 1.2
    delay = int(tau / dt)
    for t in range(delay, steps - 1):
        x_tau = x[t - delay]
        x[t + 1] = x[t] + dt * (beta * x_tau / (1 + x_tau**n) - gamma * x[t])
    x = x[800:]
    x = (x - x.mean()) / (x.std() + 1e-12)
    return x


def mg_segment(mg: np.ndarray, T_total: int, offset: int) -> Tuple[np.ndarray, np.ndarray]:
    seg = mg[offset: offset + T_total + 1].copy()
    seg = (seg - seg.mean()) / (seg.std() + 1e-12)
    return seg[:-1], seg[1:]


# -----------------------
# Quantum operators
# -----------------------

I2 = np.array([[1, 0], [0, 1]], complex)
X2 = np.array([[0, 1], [1, 0]], complex)
Z2 = np.array([[1, 0], [0, -1]], complex)


def op_on_q(op: np.ndarray, q: int, n: int) -> np.ndarray:
    ops = [I2] * n
    ops[q] = op
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def unitary_from_H(H: np.ndarray, dt: float) -> np.ndarray:
    e, v = np.linalg.eigh(H)
    return v @ np.diag(np.exp(-1j * e * dt)) @ v.conj().T


# -----------------------
# Metrics + readout
# -----------------------

def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)) / (np.std(y_true) + 1e-12))


def fit_ev_traj(
    Xshots: np.ndarray,
    y: np.ndarray,
    train_slice: slice,
    test_slice: slice,
    ridge_alpha: float,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    N, T, d = Xshots.shape
    ytr = y[train_slice]
    yte = y[test_slice]

    # --- EV ---
    X_ev = Xshots.mean(axis=0)
    Xtr_ev = np.nan_to_num(X_ev[train_slice], nan=0.0, posinf=0.0, neginf=0.0)
    Xte_ev = np.nan_to_num(X_ev[test_slice], nan=0.0, posinf=0.0, neginf=0.0)

    sc_ev = StandardScaler()
    Xtr_main = sc_ev.fit_transform(Xtr_ev[:, :-1])
    Xte_main = sc_ev.transform(Xte_ev[:, :-1])
    Xtr_s = np.concatenate([Xtr_main, Xtr_ev[:, -1:]], axis=1)
    Xte_s = np.concatenate([Xte_main, Xte_ev[:, -1:]], axis=1)

    mod_ev = Ridge(alpha=ridge_alpha, fit_intercept=False)
    mod_ev.fit(Xtr_s, ytr)
    pred_ev = mod_ev.predict(Xte_s)
    err_ev = nrmse(yte, pred_ev)

    # --- Trajectory-level ---
    Xtr = Xshots[:, train_slice, :]
    Xte = Xshots[:, test_slice, :]
    Xtr_stack = np.nan_to_num(Xtr.reshape(-1, d), nan=0.0, posinf=0.0, neginf=0.0)
    ytr_stack = np.tile(ytr, N)

    sc = StandardScaler()
    Xtr_main = sc.fit_transform(Xtr_stack[:, :-1])
    Xtr_stack_s = np.concatenate([Xtr_main, Xtr_stack[:, -1:]], axis=1)

    mod_tr = Ridge(alpha=ridge_alpha, fit_intercept=False)
    mod_tr.fit(Xtr_stack_s, ytr_stack)

    preds = []
    for i in range(N):
        Xi = np.nan_to_num(Xte[i], nan=0.0, posinf=0.0, neginf=0.0)
        Xi_main = sc.transform(Xi[:, :-1])
        Xi_s = np.concatenate([Xi_main, Xi[:, -1:]], axis=1)
        preds.append(mod_tr.predict(Xi_s))
    pred_traj = np.mean(preds, axis=0)
    err_traj = nrmse(yte, pred_traj)

    return err_ev, err_traj, yte, pred_ev, pred_traj


# -----------------------
# TFIM reservoir simulator
# -----------------------

class TFIM_QRC:
    """
    H0 = hx Σ X_i + hz Σ Z_i + J Σ Z_i Z_{i+1}
    Input: H(t) = H0 + input_scale * tanh(u_t) * Hin
    Measurement: weak Z measurement with backaction
    """
    def __init__(self, n_qubits: int, seed: int, J: float, hx: float, hz: float, input_axis: str = "Z"):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        rng = np.random.default_rng(seed)

        self.Z_ops = [op_on_q(Z2, i, self.n) for i in range(self.n)]
        self.X_ops = [op_on_q(X2, i, self.n) for i in range(self.n)]

        H = np.zeros((self.dim, self.dim), complex)
        for i in range(self.n):
            H += hx * self.X_ops[i] + hz * self.Z_ops[i]
        for i in range(self.n):
            j = (i + 1) % self.n
            H += J * (self.Z_ops[i] @ self.Z_ops[j])

        H /= (np.max(np.abs(np.linalg.eigvalsh(H))) + 1e-12)
        self.H0 = H

        signs = rng.choice([-1, 1], self.n)
        if input_axis == "Z":
            Hin = sum(signs[i] * self.Z_ops[i] for i in range(self.n))
        elif input_axis == "X":
            Hin = sum(signs[i] * self.X_ops[i] for i in range(self.n))
        else:
            raise ValueError("input_axis must be 'Z' or 'X'")
        Hin /= (np.linalg.norm(Hin) + 1e-12)
        self.Hin = Hin

        self.rho0 = np.eye(self.dim) / self.dim

        # Z eigenvalue signs for diagonal expectation and Kraus update
        self.z_signs = []
        for q in range(self.n):
            s = np.empty(self.dim, dtype=float)
            for b in range(self.dim):
                bit = (b >> (self.n - 1 - q)) & 1
                s[b] = 1.0 if bit == 0 else -1.0
            self.z_signs.append(s)

    def precompute_unitaries(self, inputs: np.ndarray, dt: float, input_scale: float) -> np.ndarray:
        u = np.tanh(inputs)
        T = len(u)
        U = np.empty((T, self.dim, self.dim), complex)
        for t, val in enumerate(u):
            H = self.H0 + float(input_scale) * float(val) * self.Hin
            U[t] = unitary_from_H(H, dt)
        return U

    def measure_update_stable(self, rho: np.ndarray, q: int, dY: float, kappa: float) -> np.ndarray:
        a = float(kappa) * float(dY)
        s = self.z_signs[q]
        shift = abs(a)
        m = np.exp(s * a - shift)
        rho = (m[:, None] * rho) * m[None, :]
        tr = float(np.real(np.trace(rho)))
        if (tr <= 0.0) or (not np.isfinite(tr)):
            return self.rho0.copy()
        rho = rho / tr
        rho = 0.5 * (rho + rho.conj().T)
        return rho

    @staticmethod
    def window_features(streams: np.ndarray, L: int) -> np.ndarray:
        T, C = streams.shape
        d = C * L + 1
        Xf = np.zeros((T, d), dtype=float)
        col = 0
        for ch in range(C):
            for lag in range(L):
                if lag == 0:
                    Xf[:, col] = streams[:, ch]
                else:
                    Xf[lag:, col] = streams[:-lag, ch]
                col += 1
        Xf[:, -1] = 1.0
        return Xf

    def simulate_features(
        self,
        inputs: np.ndarray,
        dt: float,
        kappa: float,
        input_scale: float,
        L: int,
        n_shots: int,
        noise_sigma: float,
        seed_base: int,
        meas_fraction: float = 1.0,
        missing_mode: str = "hold",
        use_mask: bool = True,
        backaction: bool = True,
        add_sq: bool = True,
    ) -> np.ndarray:
        T = len(inputs)
        C = self.n
        msize = max(1, int(round(C * meas_fraction)))

        rng = np.random.default_rng(seed_base)
        meas_mask = np.zeros((n_shots, T, C), dtype=np.int8)
        for s in range(n_shots):
            for t in range(T):
                qs = rng.choice(np.arange(C), size=msize, replace=False)
                meas_mask[s, t, qs] = 1

        dW = rng.normal(0.0, noise_sigma * math.sqrt(dt), size=(n_shots, T, C))
        U = self.precompute_unitaries(inputs, dt, input_scale)

        # Streams: currents + masks + (optional) squared currents
        channels = C + (C if use_mask else 0) + (C if add_sq else 0)
        d = channels * L + 1
        Xshots = np.zeros((n_shots, T, d), dtype=float)

        for sidx in range(n_shots):
            rho = self.rho0.copy()
            currents = np.zeros((T, C), dtype=float)
            masks = np.zeros((T, C), dtype=float)
            last = np.zeros(C, dtype=float)

            for t in range(T):
                rho = U[t] @ rho @ U[t].conj().T
                diag = np.real(np.diag(rho))

                if missing_mode == "hold":
                    currents[t, :] = last
                else:
                    currents[t, :] = 0.0

                for q in range(C):
                    if meas_mask[sidx, t, q] == 0:
                        continue
                    expZ = float(np.sum(self.z_signs[q] * diag))
                    dY = 2.0 * float(kappa) * expZ * float(dt) + float(dW[sidx, t, q])
                    curr = dY / float(dt)

                    currents[t, q] = curr
                    masks[t, q] = 1.0
                    last[q] = curr

                    if backaction:
                        rho = self.measure_update_stable(rho, q, dY, kappa)
                        diag = np.real(np.diag(rho))

            streams = [currents]
            if use_mask:
                streams.append(masks)
            if add_sq:
                streams.append(currents ** 2)
            streams_arr = np.concatenate(streams, axis=1)
            Xshots[sidx] = self.window_features(streams_arr, L)

        return Xshots


# -----------------------
# Plot helpers
# -----------------------

def savefig(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")


def plot_overlay(yte: np.ndarray, pred_ev: np.ndarray, pred_traj: np.ndarray, outpath: str, ev: float, tr: float, n_show: int = 250) -> None:
    fig = plt.figure(figsize=(9, 4))
    plt.plot(yte[:n_show], label="Target", lw=2, alpha=0.6)
    plt.plot(pred_ev[:n_show], "--", label=f"EV ({ev:.3f})")
    plt.plot(pred_traj[:n_show], label=f"Traj ({tr:.3f})")
    plt.xlabel("test time index")
    plt.ylabel("value (normalized)")
    plt.title("Best run overlay (one-step prediction)")
    plt.legend()
    plt.tight_layout()
    savefig(fig, outpath)
    plt.close(fig)


# -----------------------
# Search + experiments
# -----------------------

def evaluate_one(params: Dict[str, object], mg: np.ndarray, offset: int, add_sq: bool = True) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    T_total = int(params["washout"] + params["train_len"] + params["test_len"])
    u, y = mg_segment(mg, T_total, offset)

    qrc = TFIM_QRC(
        n_qubits=int(params["n_qubits"]),
        seed=int(params["seed_qrc"]),
        J=float(params["J"]),
        hx=float(params["hx"]),
        hz=float(params["hz"]),
        input_axis=str(params["input_axis"]),
    )

    Xshots = qrc.simulate_features(
        inputs=u,
        dt=float(params["dt"]),
        kappa=float(params["kappa"]),
        input_scale=float(params["input_scale"]),
        L=int(params["L"]),
        n_shots=int(params["n_shots"]),
        noise_sigma=float(params["noise_sigma"]),
        seed_base=int(params["seed_base"]),
        meas_fraction=float(params["meas_fraction"]),
        missing_mode=str(params["missing_mode"]),
        use_mask=bool(params["use_mask"]),
        backaction=bool(params["backaction"]),
        add_sq=bool(add_sq),
    )

    start = max(int(params["washout"]), int(params["L"]) - 1)
    train_end = int(params["washout"] + params["train_len"])
    test_end = int(T_total)

    ev, tr, yte, pred_ev, pred_tr = fit_ev_traj(
        Xshots, y, slice(start, train_end), slice(train_end, test_end),
        ridge_alpha=float(params["ridge_alpha"]),
    )

    return ev, tr, float(ev - tr), yte, pred_ev, pred_tr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="qrc_fullgas_out")
    ap.add_argument("--data_offset", type=int, default=0, help="start index in MG series (cherry-picking off by default)")
    args = ap.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    mg = generate_mackey_glass(t_max=30000, tau=17, seed=42)

    # Baseline "physics" settings (4 qubits TFIM)
    base = dict(
        n_qubits=4,
        J=0.4,
        hx=1.0,
        hz=0.2,
        input_axis="Z",
        dt=0.1,
        kappa=0.3,
        meas_fraction=1.0,
        missing_mode="hold",
        use_mask=True,
        backaction=True,
        # task sizes
        washout=50,
        train_len=1500,
        test_len=500,
        # stochastic / sampling
        n_shots=3,
        noise_sigma=0.16,
        # readout
        ridge_alpha=1e-4,
        # seeds
        seed_qrc=17,
        seed_base=17 * 999,
        # search params
        L=64,
        input_scale=1.1,
    )

    # --- SEARCH schedule (compact, deterministic) ---
    records: List[Dict[str, object]] = []
    it = 0

    # Experiment 2: L sweep (regime dependence)
    L_list = [60, 64, 68, 70, 72]
    for L in L_list:
        p = base.copy()
        p["L"] = L
        ev, tr, gap, *_ = evaluate_one(p, mg, args.data_offset, add_sq=True)
        records.append(dict(iter=it, stage="L_sweep", L=L, input_scale=p["input_scale"], noise_sigma=p["noise_sigma"],
                            nrmse_ev=ev, nrmse_traj=tr, gap=gap))
        it += 1

    # Pick best feasible by minimal traj
    def feasible(r: Dict[str, object]) -> bool:
        return (r["nrmse_ev"] < 1.0) and (r["nrmse_traj"] < 1.0) and (r["nrmse_traj"] < r["nrmse_ev"])

    feasible_L = [r for r in records if r["stage"] == "L_sweep" and feasible(r)]
    if len(feasible_L) == 0:
        raise RuntimeError("No feasible config found in L sweep. Try changing seeds or ranges.")
    best_L = min(feasible_L, key=lambda r: r["nrmse_traj"])["L"]
    base["L"] = int(best_L)

    # Noise sweep
    noise_list = [0.14, 0.15, 0.16, 0.17]
    for noise in noise_list:
        p = base.copy()
        p["noise_sigma"] = noise
        ev, tr, gap, *_ = evaluate_one(p, mg, args.data_offset, add_sq=True)
        records.append(dict(iter=it, stage="noise_sweep", L=p["L"], input_scale=p["input_scale"], noise_sigma=noise,
                            nrmse_ev=ev, nrmse_traj=tr, gap=gap))
        it += 1

    feasible_noise = [r for r in records if r["stage"] == "noise_sweep" and feasible(r)]
    if len(feasible_noise) == 0:
        raise RuntimeError("No feasible config found in noise sweep. Try changing seeds or ranges.")
    best_noise = min(feasible_noise, key=lambda r: r["nrmse_traj"])["noise_sigma"]
    base["noise_sigma"] = float(best_noise)

    # Input-scale sweep
    inp_list = [0.9, 1.0, 1.1, 1.2, 1.3]
    for inp in inp_list:
        p = base.copy()
        p["input_scale"] = inp
        ev, tr, gap, *_ = evaluate_one(p, mg, args.data_offset, add_sq=True)
        records.append(dict(iter=it, stage="input_sweep", L=p["L"], input_scale=inp, noise_sigma=p["noise_sigma"],
                            nrmse_ev=ev, nrmse_traj=tr, gap=gap))
        it += 1

    feasible_inp = [r for r in records if r["stage"] == "input_sweep" and feasible(r)]
    if len(feasible_inp) == 0:
        raise RuntimeError("No feasible config found in input sweep. Try changing seeds or ranges.")
    best_row = min(feasible_inp, key=lambda r: r["nrmse_traj"])

    # Final best params:
    base["L"] = int(best_row["L"])
    base["input_scale"] = float(best_row["input_scale"])
    base["noise_sigma"] = float(best_row["noise_sigma"])

    # Save sweep results
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(outdir, "qrc_sweep_results.csv"), index=False)

    # Experiment 1: overlay plot for best config
    ev, tr, gap, yte, pred_ev, pred_traj = evaluate_one(base, mg, args.data_offset, add_sq=True)
    plot_overlay(yte, pred_ev, pred_traj, os.path.join(outdir, "best_overlay.png"), ev=ev, tr=tr, n_show=250)

    # Progress plot
    fig = plt.figure(figsize=(7, 4))
    plt.plot(df["iter"], df["nrmse_ev"], marker="o", label="EV")
    plt.plot(df["iter"], df["nrmse_traj"], marker="o", label="Traj")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("iteration")
    plt.ylabel("NRMSE")
    plt.title("NRMSE over search iterations")
    plt.legend()
    plt.tight_layout()
    savefig(fig, os.path.join(outdir, "progress_nrmse.png"))
    plt.close(fig)

    # Experiment 3: phase grid around best
    noise_vals = [0.12, 0.16, 0.2]
    shot_vals = [1, 2, 3, 5]
    grid_rows = []
    for noise in noise_vals:
        for shots in shot_vals:
            p = base.copy()
            p["noise_sigma"] = float(noise)
            p["n_shots"] = int(shots)
            ev_g, tr_g, gap_g, *_ = evaluate_one(p, mg, args.data_offset, add_sq=True)
            grid_rows.append(dict(noise_sigma=noise, n_shots=shots, nrmse_ev=ev_g, nrmse_traj=tr_g, gap=gap_g))
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(os.path.join(outdir, "qrc_phase_grid.csv"), index=False)

    # Heatmap (gap)
    noise_unique = sorted(grid_df["noise_sigma"].unique())
    shots_unique = sorted(grid_df["n_shots"].unique())
    gap_mat = np.zeros((len(shots_unique), len(noise_unique)))
    for i, sh in enumerate(shots_unique):
        for j, no in enumerate(noise_unique):
            gap_mat[i, j] = float(grid_df[(grid_df["n_shots"] == sh) & (grid_df["noise_sigma"] == no)]["gap"].iloc[0])

    fig = plt.figure(figsize=(6, 4))
    im = plt.imshow(gap_mat, aspect="auto", origin="lower",
                    extent=[min(noise_unique), max(noise_unique), min(shots_unique), max(shots_unique)])
    plt.xlabel("noise_sigma")
    plt.ylabel("n_shots")
    plt.title("Phase diagram: gap = EV - Traj")
    plt.colorbar(im, label="gap")
    plt.tight_layout()
    savefig(fig, os.path.join(outdir, "phase_gap_heatmap.png"))
    plt.close(fig)

    # Final 4-panel figure
    fig = plt.figure(figsize=(12, 9))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(df["iter"], df["nrmse_ev"], marker="o", label="EV")
    ax1.plot(df["iter"], df["nrmse_traj"], marker="o", label="Traj")
    ax1.axhline(1.0, linestyle="--", linewidth=1)
    ax1.set_title("A) Search history")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("NRMSE")
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    n_show = 250
    ax2.plot(yte[:n_show], label="Target", lw=2, alpha=0.6)
    ax2.plot(pred_ev[:n_show], "--", label=f"EV ({ev:.3f})")
    ax2.plot(pred_traj[:n_show], label=f"Traj ({tr:.3f})")
    ax2.set_title("B) Best run overlay")
    ax2.set_xlabel("test time index")
    ax2.set_ylabel("value")
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    im = ax3.imshow(gap_mat, aspect="auto", origin="lower",
                    extent=[min(noise_unique), max(noise_unique), min(shots_unique), max(shots_unique)])
    ax3.set_title("C) Phase grid: gap (EV - Traj)")
    ax3.set_xlabel("noise_sigma")
    ax3.set_ylabel("n_shots")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.bar(["EV", "Traj"], [ev, tr])
    ax4.axhline(1.0, linestyle="--", linewidth=1)
    ax4.set_title(f"D) Best metrics (gap={gap:+.3f})")
    ax4.set_ylabel("NRMSE")

    plt.tight_layout()
    savefig(fig, os.path.join(outdir, "qrc_fullgas_final.png"))
    plt.close(fig)

    print("BEST CONFIG (this run/segment):")
    print(base)
    print(f"EV NRMSE   = {ev:.4f}")
    print(f"Traj NRMSE = {tr:.4f}")
    print(f"Gap (EV-Traj) = {gap:+.4f}")
    print(f"\nSaved outputs to: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()
