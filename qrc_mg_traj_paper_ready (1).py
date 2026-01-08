#!/usr/bin/env python3
"""
Publication-ready Mackey–Glass experiments for trajectory-level QRC
===================================================================

Goal
----
Produce a reviewer-safe, reproducible results bundle for a methods paper:
- Physically motivated quantum reservoir (disordered transverse-field Ising, 4 qubits)
- Continuous weak measurement records with backaction (trajectory simulation)
- Budget-matched comparison:
    EV baseline: average features over shots BEFORE learning
    Trajectory-level: train on stacked single-shot trajectories, average predictions at test time

Why k-step (default k=10)?
--------------------------
For MG one-step prediction, the persistence baseline y_{t+1}≈y_t can be extremely strong. To avoid
a trivial baseline dominating the task, we report k-step direct prediction (default k=10), where
persistence degrades substantially. This is standard in reservoir computing evaluations.

Outputs (written to --out_dir)
------------------------------
- experiment_log.csv              consolidated table (all experiments and reruns)
- predictions_exp1.csv            predictions for Exp1 (time series)
- exp2_lag_sweep.csv              sweep table
- exp3_shots_sweep.csv            sweep table
- exp4_robustness.csv             per-rerun table
- plot_exp1_timeseries.png
- plot_exp2_lag_sweep.png
- plot_exp2_gap_vs_ratio.png
- plot_exp3_shots_sweep.png
- plot_exp4_bars_mean_std.png
- final_figure_4panel.png         stitched 4-panel figure
- final_config.json               full config for reproducibility

Run
---
python qrc_mg_traj_paper_ready.py --out_dir qrc_mg_paper_ready_out

Notes
-----
- The reservoir input couples along a NON-commuting direction (X) relative to Z measurement.
  This is essential: Z-driving + Z-measurement often yields weak input imprinting in the record.
- The initial state is a pure product state |0...0>, matching realistic state preparation.

"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# ---------------------------
# Data: Mackey–Glass (normalized)
# ---------------------------
def generate_mackey_glass(t_max: int = 6000, tau: int = 17, seed: int = 42, discard: int = 800) -> np.ndarray:
    rng = np.random.default_rng(seed)
    beta, gamma, n, dt = 0.2, 0.1, 10, 1.0
    steps = int(t_max / dt)
    x = np.zeros(steps, dtype=float)
    x[0] = 1.2
    delay = int(tau / dt)
    for t in range(delay, steps - 1):
        x_tau = x[t - delay]
        x[t + 1] = x[t] + dt * (beta * x_tau / (1.0 + x_tau**n) - gamma * x[t])
        # tiny perturbation to ensure different seeds diverge eventually in chaotic regimes
        if t % 500 == 0 and t > 0:
            x[t + 1] += 1e-12 * rng.normal()
    x = x[discard:]
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    return x


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)) / (np.std(y_true) + 1e-12))


# ---------------------------
# Quantum helpers
# ---------------------------
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


def pure_zero_rho(n: int) -> np.ndarray:
    dim = 2**n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    return np.outer(psi, psi.conj())


# ---------------------------
# Parameters
# ---------------------------
@dataclass(frozen=True)
class ReservoirParams:
    n_qubits: int
    dt: float
    kappa: float
    noise_sigma: float
    meas_frac: float
    input_scale: float

    # TFI parameters
    hx: float
    hz: float
    J: float
    disorder: float

    init_state: str  # "pure0" or "mixed"
    rseed: int
    input_axis: str  # "X", "Z", or "XZ"
    use_increments: bool  # True: use dY, False: use I=dY/dt


@dataclass(frozen=True)
class ReadoutParams:
    lags_per_qubit: int
    ridge_alpha: float


@dataclass(frozen=True)
class DataParams:
    train_len: int
    test_len: int
    mg_tau: int


# ---------------------------
# Reservoir model
# ---------------------------
class QRC_TFI:
    def __init__(self, rp: ReservoirParams):
        self.rp = rp
        self.n = rp.n_qubits
        self.dim = 2**self.n

        rng = np.random.default_rng(rp.rseed)

        self.X_ops = [op_on_q(X2, i, self.n) for i in range(self.n)]
        self.Z_ops = [op_on_q(Z2, i, self.n) for i in range(self.n)]

        # Disordered transverse-field Ising
        hx_i = rp.hx * (1.0 + rp.disorder * rng.normal(size=self.n))
        hz_i = rp.hz * (1.0 + rp.disorder * rng.normal(size=self.n))
        J_i  = rp.J  * (1.0 + rp.disorder * rng.normal(size=self.n))

        H0 = np.zeros((self.dim, self.dim), complex)
        for i in range(self.n):
            H0 += hx_i[i] * self.X_ops[i] + hz_i[i] * self.Z_ops[i]
        for i in range(self.n):
            j = (i + 1) % self.n
            H0 += J_i[i] * (self.Z_ops[i] @ self.Z_ops[j])

        H0 /= (np.max(np.abs(np.linalg.eigvalsh(H0))) + 1e-12)

        # Input Hamiltonian direction (non-commuting recommended)
        mask = rng.choice([-1, 1], self.n)
        ax = rp.input_axis.upper()
        if ax == "X":
            Hin = sum(mask[i] * self.X_ops[i] for i in range(self.n))
        elif ax == "Z":
            Hin = sum(mask[i] * self.Z_ops[i] for i in range(self.n))
        elif ax == "XZ":
            Hin = sum(mask[i] * (self.X_ops[i] + self.Z_ops[i]) for i in range(self.n))
        else:
            Hin = sum(mask[i] * self.X_ops[i] for i in range(self.n))
        Hin /= (np.linalg.norm(Hin) + 1e-12)

        self.H0 = H0
        self.Hin = Hin

        if rp.init_state == "pure0":
            self.rho0 = pure_zero_rho(self.n)
        else:
            self.rho0 = np.eye(self.dim, dtype=complex) / self.dim

        # Z eigenvalue signs in computational basis
        self.z_signs = []
        for q in range(self.n):
            s = np.empty(self.dim, dtype=float)
            for b in range(self.dim):
                bit = (b >> (self.n - 1 - q)) & 1
                s[b] = 1.0 if bit == 0 else -1.0
            self.z_signs.append(s)

    def precompute_unitaries(self, inputs: np.ndarray) -> np.ndarray:
        u = np.tanh(inputs)
        U = np.empty((len(u), self.dim, self.dim), complex)
        for t, val in enumerate(u):
            Ht = self.H0 + self.rp.input_scale * float(val) * self.Hin
            U[t] = unitary_from_H(Ht, self.rp.dt)
        return U

    def measure_update(self, rho: np.ndarray, q: int, dY: float) -> np.ndarray:
        # CP update specialized to Pauli-Z (Z^2=I)
        a = self.rp.kappa * float(dY)
        s = self.z_signs[q]
        m = np.exp(s * a)
        rho = (m[:, None] * rho) * m[None, :]
        tr = float(np.trace(rho).real)
        if tr <= 0.0 or not np.isfinite(tr):
            rho = self.rho0.copy()
        else:
            rho /= tr
        rho = 0.5 * (rho + rho.conj().T)
        return rho

    @staticmethod
    def lag_embed(obs: np.ndarray, lags_per_qubit: int) -> np.ndarray:
        T, C = obs.shape
        L = int(lags_per_qubit)
        d = C * L + 1
        Xf = np.zeros((T, d), dtype=float)
        col = 0
        for ch in range(C):
            for lag in range(L):
                if lag == 0:
                    Xf[:, col] = obs[:, ch]
                else:
                    Xf[lag:, col] = obs[:-lag, ch]
                col += 1
        Xf[:, -1] = 1.0
        return Xf

    def simulate(self, inputs: np.ndarray, n_shots: int, seed_base: int, lags_per_qubit: int) -> np.ndarray:
        U = self.precompute_unitaries(inputs)
        T = len(inputs)
        C = self.n

        subset_size = max(1, int(round(self.rp.meas_frac * C)))
        subset_size = min(C, subset_size)

        Xshots = np.zeros((n_shots, T, C * lags_per_qubit + 1), dtype=float)

        for s in range(n_shots):
            rng = np.random.default_rng(seed_base + s)
            rho = self.rho0.copy()
            obs = np.zeros((T, C), dtype=float)

            for t in range(T):
                rho = U[t] @ rho @ U[t].conj().T
                diag = rho.diagonal().real

                if subset_size == C:
                    q_meas = range(C)
                else:
                    q_meas = rng.choice(np.arange(C), size=subset_size, replace=False)

                for q in q_meas:
                    expZ = float(np.dot(self.z_signs[int(q)], diag))
                    dW = float(rng.normal(0.0, self.rp.noise_sigma * math.sqrt(self.rp.dt)))
                    dY = 2.0 * self.rp.kappa * expZ * self.rp.dt + dW

                    obs[t, int(q)] = dY if self.rp.use_increments else (dY / self.rp.dt)

                    rho = self.measure_update(rho, int(q), dY)
                    diag = rho.diagonal().real

            Xshots[s] = self.lag_embed(obs, lags_per_qubit=lags_per_qubit)

        return Xshots


# ---------------------------
# Baselines
# ---------------------------
def baseline_zero(yte: np.ndarray) -> Tuple[np.ndarray, float]:
    pred = np.zeros_like(yte)
    return pred, nrmse(yte, pred)


def baseline_persistence(u_test: np.ndarray, yte: np.ndarray) -> Tuple[np.ndarray, float]:
    pred = u_test.copy()
    return pred, nrmse(yte, pred)


def baseline_ar_ridge(series: np.ndarray, train_len: int, test_len: int, k: int, p: int = 50, alpha: float = 1e-6) -> Tuple[np.ndarray, float]:
    total = train_len + test_len
    X, y = [], []
    for t in range(p - 1, total):
        X.append(series[t - p + 1 : t + 1][::-1])
        y.append(series[t + k])
    X = np.array(X); y = np.array(y)
    train_rows = train_len - (p - 1)
    Xtr, ytr = X[:train_rows], y[:train_rows]
    Xte, yte = X[train_rows : train_rows + test_len], y[train_rows : train_rows + test_len]
    mod = Ridge(alpha=alpha).fit(Xtr, ytr)
    pred = mod.predict(Xte)
    return pred, nrmse(yte, pred)


# ---------------------------
# Evaluation: EV vs traj + baselines (direct k-step)
# ---------------------------
def eval_run(
    rp: ReservoirParams,
    rop: ReadoutParams,
    dp: DataParams,
    *,
    mg_seed: int,
    meas_seed: int,
    n_shots: int,
    k: int,
) -> Dict[str, object]:
    series = generate_mackey_glass(tau=dp.mg_tau, seed=mg_seed)
    total = dp.train_len + dp.test_len
    u = series[:total]
    y = series[k : total + k]  # direct k-step target aligned with time t

    ytr = y[: dp.train_len]
    yte = y[dp.train_len :]

    # Baselines
    pred0, n0 = baseline_zero(yte)
    predP, nP = baseline_persistence(u_test=u[dp.train_len : dp.train_len + dp.test_len], yte=yte)
    predAR, nAR = baseline_ar_ridge(series, dp.train_len, dp.test_len, k=k, p=50, alpha=1e-6)

    # QRC trajectories
    qrc = QRC_TFI(rp)
    Xshots = qrc.simulate(inputs=u, n_shots=n_shots, seed_base=meas_seed * 999, lags_per_qubit=rop.lags_per_qubit)

    Xtr = Xshots[:, : dp.train_len, :]
    Xte = Xshots[:, dp.train_len :, :]

    # EV baseline
    Xtr_ev = Xtr.mean(axis=0)
    Xte_ev = Xte.mean(axis=0)
    mod_ev = Ridge(alpha=rop.ridge_alpha).fit(Xtr_ev, ytr)
    pred_ev = mod_ev.predict(Xte_ev)
    n_ev = nrmse(yte, pred_ev)

    # Trajectory-level
    Xtr_stack = Xtr.reshape(-1, Xtr.shape[-1])
    ytr_stack = np.tile(ytr, n_shots)
    mod_tr = Ridge(alpha=rop.ridge_alpha).fit(Xtr_stack, ytr_stack)
    preds = np.stack([mod_tr.predict(Xte[i]) for i in range(n_shots)], axis=0)
    pred_tr = preds.mean(axis=0)
    n_tr = nrmse(yte, pred_tr)

    # sample ratios
    d = int(Xshots.shape[-1])
    rho_ev = float(dp.train_len / d)
    rho_tr = float((dp.train_len * n_shots) / d)

    return dict(
        y_test=yte,
        pred_zero=pred0, nrmse_zero=n0,
        pred_persist=predP, nrmse_persist=nP,
        pred_ar=predAR, nrmse_ar=nAR,
        pred_ev=pred_ev, nrmse_ev=n_ev,
        pred_traj=pred_tr, nrmse_traj=n_tr,
        gap=float(n_ev - n_tr),
        feature_dim=d,
        rho_ev=rho_ev,
        rho_traj=rho_tr,
    )


# ---------------------------
# Plot utilities
# ---------------------------
def plot_timeseries(out_path: str, y: np.ndarray, curves: List[Tuple[str, np.ndarray]], title: str, n: int = 300) -> None:
    n = min(n, len(y))
    plt.figure(figsize=(11, 4))
    plt.plot(y[:n], label="Target", lw=2, alpha=0.55)
    for name, pred in curves:
        plt.plot(pred[:n], label=name)
    plt.legend(ncol=2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_lag_sweep(out_path: str, lags: List[int], evs: List[float], trs: List[float], ar_ref: float, title: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(lags, evs, marker="o", linestyle="--", label="EV")
    plt.plot(lags, trs, marker="o", linestyle="-", label="Trajectory")
    plt.axhline(ar_ref, linewidth=1, label="AR(50) baseline")
    plt.xlabel("lags_per_qubit")
    plt.ylabel("NRMSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_gap_vs_ratio(out_path: str, ratios: List[float], gaps: List[float], title: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(ratios, gaps, marker="o")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel(r"EV sample ratio ρ = N_train / d")
    plt.ylabel("Gap = NRMSE_EV − NRMSE_traj")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_shots_sweep(out_path: str, shots: List[int], evs: List[float], trs: List[float], ar_ref: float, title: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(shots, evs, marker="o", linestyle="--", label="EV")
    plt.plot(shots, trs, marker="o", linestyle="-", label="Trajectory")
    plt.axhline(ar_ref, linewidth=1, label="AR(50) baseline")
    plt.xlabel("n_shots")
    plt.ylabel("NRMSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_bars_mean_std(out_path: str, labels: List[str], means: List[float], stds: List[float], title: str) -> None:
    plt.figure(figsize=(9, 4))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=6)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("NRMSE")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def stitch_2x2(paths: List[str], out_path: str) -> None:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    w = min(im.width for im in imgs)
    imgs = [im.resize((w, int(im.height * (w / im.width)))) for im in imgs]
    row1 = Image.new("RGB", (imgs[0].width + imgs[1].width, max(imgs[0].height, imgs[1].height)), (255, 255, 255))
    row1.paste(imgs[0], (0, 0))
    row1.paste(imgs[1], (imgs[0].width, 0))
    row2 = Image.new("RGB", (imgs[2].width + imgs[3].width, max(imgs[2].height, imgs[3].height)), (255, 255, 255))
    row2.paste(imgs[2], (0, 0))
    row2.paste(imgs[3], (imgs[2].width, 0))
    out = Image.new("RGB", (row1.width, row1.height + row2.height), (255, 255, 255))
    out.paste(row1, (0, 0))
    out.paste(row2, (0, row1.height))
    out.save(out_path)


# ---------------------------
# Experiments
# ---------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="qrc_mg_paper_ready_out")

    # task/data
    ap.add_argument("--k", type=int, default=10, help="prediction horizon (direct k-step)")
    ap.add_argument("--train_len", type=int, default=300)
    ap.add_argument("--test_len", type=int, default=500)
    ap.add_argument("--mg_tau", type=int, default=17)

    # base reservoir hyperparams (good default)
    ap.add_argument("--n_qubits", type=int, default=4)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--noise_sigma", type=float, default=0.01)
    ap.add_argument("--meas_frac", type=float, default=1.0)
    ap.add_argument("--input_scale", type=float, default=3.0)

    ap.add_argument("--hx", type=float, default=1.0)
    ap.add_argument("--hz", type=float, default=0.2)
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--disorder", type=float, default=0.2)

    ap.add_argument("--init_state", type=str, default="pure0", choices=["pure0", "mixed"])
    ap.add_argument("--input_axis", type=str, default="X", choices=["X", "Z", "XZ"])
    ap.add_argument("--use_increments", action="store_true", help="use dY as observation (recommended)")
    ap.add_argument("--use_currents", action="store_true", help="use I=dY/dt as observation")

    # readout + budget
    ap.add_argument("--n_shots", type=int, default=20)
    ap.add_argument("--lags_per_qubit", type=int, default=120)
    ap.add_argument("--ridge_alpha", type=float, default=0.005)

    # robustness evaluation (like '3 seeds × 2 reservoir seeds')
    ap.add_argument("--mg_seeds", type=str, default="0,1,2")
    ap.add_argument("--reservoir_seeds", type=str, default="0,1")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    use_increments = True
    if args.use_currents:
        use_increments = False
    if args.use_increments:
        use_increments = True

    dp = DataParams(train_len=args.train_len, test_len=args.test_len, mg_tau=args.mg_tau)

    # Base params for Exp1/2/3: keep rseed fixed at 17 (good working point)
    rp_base = ReservoirParams(
        n_qubits=args.n_qubits,
        dt=args.dt,
        kappa=args.kappa,
        noise_sigma=args.noise_sigma,
        meas_frac=args.meas_frac,
        input_scale=args.input_scale,
        hx=args.hx,
        hz=args.hz,
        J=args.J,
        disorder=args.disorder,
        init_state=args.init_state,
        rseed=0,
        input_axis=args.input_axis,
        use_increments=use_increments,
    )
    rop_base = ReadoutParams(lags_per_qubit=args.lags_per_qubit, ridge_alpha=args.ridge_alpha)

    # -----------------------
    # Exp1: main timeseries
    # -----------------------
    exp_rows: List[Dict[str, object]] = []

    mg_seed_exp1 = 0
    meas_seed_exp1 = 0
    res1 = eval_run(rp_base, rop_base, dp, mg_seed=mg_seed_exp1, meas_seed=meas_seed_exp1, n_shots=args.n_shots, k=args.k)

    # save predictions for exp1
    df_pred = pd.DataFrame({
        "t": np.arange(len(res1["y_test"])),
        "y_target": res1["y_test"],
        "y_pred_traj": res1["pred_traj"],
        "y_pred_ev": res1["pred_ev"],
        "y_pred_ar": res1["pred_ar"],
        "y_pred_persist": res1["pred_persist"],
        "y_pred_zero": res1["pred_zero"],
    })
    df_pred.to_csv(os.path.join(args.out_dir, "predictions_exp1.csv"), index=False)

    plot_timeseries(
        os.path.join(args.out_dir, "plot_exp1_timeseries.png"),
        res1["y_test"],
        [
            (f"Trajectory (NRMSE={res1['nrmse_traj']:.3f})", res1["pred_traj"]),
            (f"EV (NRMSE={res1['nrmse_ev']:.3f})", res1["pred_ev"]),
            (f"AR(50) (NRMSE={res1['nrmse_ar']:.3f})", res1["pred_ar"]),
            (f"Persistence (NRMSE={res1['nrmse_persist']:.3f})", res1["pred_persist"]),
        ],
        title=f"Exp1: Mackey–Glass direct k-step prediction (k={args.k})",
        n=300,
    )

    exp_rows.append(dict(
        experiment="exp1_main",
        k=args.k, mg_seed=mg_seed_exp1, meas_seed=meas_seed_exp1,
        n_shots=args.n_shots, lags_per_qubit=args.lags_per_qubit, ridge_alpha=args.ridge_alpha,
        nrmse_zero=res1["nrmse_zero"], nrmse_persist=res1["nrmse_persist"], nrmse_ar=res1["nrmse_ar"],
        nrmse_ev=res1["nrmse_ev"], nrmse_traj=res1["nrmse_traj"], gap=res1["gap"],
        feature_dim=res1["feature_dim"], rho_ev=res1["rho_ev"], rho_traj=res1["rho_traj"],
        **asdict(rp_base), **asdict(dp),
    ))

    # -----------------------
    # Exp2: lag sweep
    # -----------------------
    lags_list = [10, 20, 50, 80, 120]
    rows2 = []
    evs2, trs2, ratios2, gaps2 = [], [], [], []
    ar_ref = float(res1["nrmse_ar"])  # for same mg_seed in exp1; sufficient as reference

    for L in lags_list:
        rop = ReadoutParams(lags_per_qubit=L, ridge_alpha=args.ridge_alpha)
        res = eval_run(rp_base, rop, dp, mg_seed=mg_seed_exp1, meas_seed=meas_seed_exp1, n_shots=args.n_shots, k=args.k)
        rows2.append(dict(lags_per_qubit=L, ridge_alpha=args.ridge_alpha, nrmse_ev=res["nrmse_ev"], nrmse_traj=res["nrmse_traj"], gap=res["gap"], rho_ev=res["rho_ev"]))
        evs2.append(res["nrmse_ev"]); trs2.append(res["nrmse_traj"]); gaps2.append(res["gap"]); ratios2.append(res["rho_ev"])

        exp_rows.append(dict(
            experiment="exp2_lag_sweep",
            k=args.k, mg_seed=mg_seed_exp1, meas_seed=meas_seed_exp1,
            n_shots=args.n_shots, lags_per_qubit=L, ridge_alpha=args.ridge_alpha,
            nrmse_zero=res["nrmse_zero"], nrmse_persist=res["nrmse_persist"], nrmse_ar=res["nrmse_ar"],
            nrmse_ev=res["nrmse_ev"], nrmse_traj=res["nrmse_traj"], gap=res["gap"],
            feature_dim=res["feature_dim"], rho_ev=res["rho_ev"], rho_traj=res["rho_traj"],
            **asdict(rp_base), **asdict(dp),
        ))

    df2 = pd.DataFrame(rows2)
    df2.to_csv(os.path.join(args.out_dir, "exp2_lag_sweep.csv"), index=False)

    plot_lag_sweep(os.path.join(args.out_dir, "plot_exp2_lag_sweep.png"), lags_list, evs2, trs2, ar_ref, title=f"Exp2: Lag sweep (k={args.k})")
    plot_gap_vs_ratio(os.path.join(args.out_dir, "plot_exp2_gap_vs_ratio.png"), ratios2, gaps2, title="Exp2: Gap vs EV sample ratio")

    # -----------------------
    # Exp3: shots sweep
    # -----------------------
    shots_list = [1, 2, 5, 10, 20]
    rows3 = []
    evs3, trs3, gaps3 = [], [], []
    for N in shots_list:
        res = eval_run(rp_base, rop_base, dp, mg_seed=mg_seed_exp1, meas_seed=meas_seed_exp1, n_shots=N, k=args.k)
        rows3.append(dict(n_shots=N, nrmse_ev=res["nrmse_ev"], nrmse_traj=res["nrmse_traj"], gap=res["gap"]))
        evs3.append(res["nrmse_ev"]); trs3.append(res["nrmse_traj"]); gaps3.append(res["gap"])

        exp_rows.append(dict(
            experiment="exp3_shots_sweep",
            k=args.k, mg_seed=mg_seed_exp1, meas_seed=meas_seed_exp1,
            n_shots=N, lags_per_qubit=args.lags_per_qubit, ridge_alpha=args.ridge_alpha,
            nrmse_zero=res["nrmse_zero"], nrmse_persist=res["nrmse_persist"], nrmse_ar=res["nrmse_ar"],
            nrmse_ev=res["nrmse_ev"], nrmse_traj=res["nrmse_traj"], gap=res["gap"],
            feature_dim=res["feature_dim"], rho_ev=res["rho_ev"], rho_traj=res["rho_traj"],
            **asdict(rp_base), **asdict(dp),
        ))

    df3 = pd.DataFrame(rows3)
    df3.to_csv(os.path.join(args.out_dir, "exp3_shots_sweep.csv"), index=False)
    plot_shots_sweep(os.path.join(args.out_dir, "plot_exp3_shots_sweep.png"), shots_list, evs3, trs3, ar_ref, title=f"Exp3: Shots sweep (k={args.k})")

    # -----------------------
    # Exp4: robustness across seeds × reservoir seeds
    # -----------------------
    mg_seeds = [int(x.strip()) for x in args.mg_seeds.split(",") if x.strip()]
    rseeds = [int(x.strip()) for x in args.reservoir_seeds.split(",") if x.strip()]

    rows4 = []
    for rseed in rseeds:
        rp = ReservoirParams(**{**asdict(rp_base), "rseed": rseed})
        for s in mg_seeds:
            # mimic "input/noise seed": use same s for MG series and measurement noise
            res = eval_run(rp, rop_base, dp, mg_seed=s, meas_seed=s, n_shots=args.n_shots, k=args.k)
            rows4.append(dict(rseed=rseed, seed=s, k=args.k,
                              nrmse_zero=res["nrmse_zero"], nrmse_persist=res["nrmse_persist"], nrmse_ar=res["nrmse_ar"],
                              nrmse_ev=res["nrmse_ev"], nrmse_traj=res["nrmse_traj"], gap=res["gap"]))
            exp_rows.append(dict(
                experiment="exp4_robustness",
                k=args.k, mg_seed=s, meas_seed=s,
                n_shots=args.n_shots, lags_per_qubit=args.lags_per_qubit, ridge_alpha=args.ridge_alpha,
                nrmse_zero=res["nrmse_zero"], nrmse_persist=res["nrmse_persist"], nrmse_ar=res["nrmse_ar"],
                nrmse_ev=res["nrmse_ev"], nrmse_traj=res["nrmse_traj"], gap=res["gap"],
                feature_dim=res["feature_dim"], rho_ev=res["rho_ev"], rho_traj=res["rho_traj"],
                **asdict(rp), **asdict(dp),
            ))

    df4 = pd.DataFrame(rows4)
    df4.to_csv(os.path.join(args.out_dir, "exp4_robustness.csv"), index=False)

    # bar plot mean±std
    labels = ["Zero", "Persistence", "AR(50)", "EV", "Trajectory"]
    means = [
        float(df4["nrmse_zero"].mean()),
        float(df4["nrmse_persist"].mean()),
        float(df4["nrmse_ar"].mean()),
        float(df4["nrmse_ev"].mean()),
        float(df4["nrmse_traj"].mean()),
    ]
    stds = [
        float(df4["nrmse_zero"].std(ddof=1)),
        float(df4["nrmse_persist"].std(ddof=1)),
        float(df4["nrmse_ar"].std(ddof=1)),
        float(df4["nrmse_ev"].std(ddof=1)),
        float(df4["nrmse_traj"].std(ddof=1)),
    ]
    plot_bars_mean_std(os.path.join(args.out_dir, "plot_exp4_bars_mean_std.png"), labels, means, stds, title=f"Exp4: Robustness (k={args.k}), mean±std over seeds×rseeds")

    # -----------------------
    # Consolidated log + final 4-panel
    # -----------------------
    df_log = pd.DataFrame(exp_rows)
    df_log.to_csv(os.path.join(args.out_dir, "experiment_log.csv"), index=False)

    panelA = os.path.join(args.out_dir, "plot_exp1_timeseries.png")
    panelB = os.path.join(args.out_dir, "plot_exp4_bars_mean_std.png")
    panelC = os.path.join(args.out_dir, "plot_exp2_lag_sweep.png")
    panelD = os.path.join(args.out_dir, "plot_exp3_shots_sweep.png")
    stitch_2x2([panelA, panelB, panelC, panelD], os.path.join(args.out_dir, "final_figure_4panel.png"))

    # config dump
    with open(os.path.join(args.out_dir, "final_config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(
            k=args.k,
            dp=asdict(dp),
            rp_base=asdict(rp_base),
            rop_base=asdict(rop_base),
            robustness=dict(mg_seeds=mg_seeds, reservoir_seeds=rseeds),
            exp4_summary=dict(
                mean_nrmse_ev=float(df4["nrmse_ev"].mean()),
                mean_nrmse_traj=float(df4["nrmse_traj"].mean()),
                mean_gap=float(df4["gap"].mean()),
                mean_baselines=dict(
                    zero=float(df4["nrmse_zero"].mean()),
                    persistence=float(df4["nrmse_persist"].mean()),
                    ar50=float(df4["nrmse_ar"].mean()),
                )
            ),
        ), f, indent=2)

    # console summary
    print(f"Exp1 (k={args.k}): EV={res1['nrmse_ev']:.4f}, Traj={res1['nrmse_traj']:.4f}, AR(50)={res1['nrmse_ar']:.4f}, Persist={res1['nrmse_persist']:.4f}")
    print(f"Exp4 mean±std: EV={df4['nrmse_ev'].mean():.4f}±{df4['nrmse_ev'].std(ddof=1):.4f}, Traj={df4['nrmse_traj'].mean():.4f}±{df4['nrmse_traj'].std(ddof=1):.4f}, Gap={df4['gap'].mean():+.4f}±{df4['gap'].std(ddof=1):.4f}")
    print(f"Outputs written to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
