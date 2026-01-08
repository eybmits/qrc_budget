#!/usr/bin/env python3
"""
QRC (TFI reservoir) learns Mackey–Glass with trajectory-level training (v2)
==========================================================================
This is a physically motivated quantum reservoir computing (QRC) simulator:
- n = 4–6 qubits (density matrix simulation)
- disordered transverse-field Ising Hamiltonian H0
- input drives a non-commuting control Hamiltonian Hin (default: X-axis), crucial for learnability
- continuous weak measurement on Z of each qubit (discrete-time sampled), with backaction
- trajectory-level vs expectation-value (EV) baseline under matched shot budgets

Key improvements vs earlier near-chance runs:
---------------------------------------------
1) Input coupling uses a non-commuting operator (X) rather than Z.
   If input modulates Z and we only read out Z, populations barely change and the record can be
   almost uncorrelated with the input → NRMSE ~ 1.
2) Pure initialization |0...0><0...0| (physical: state prep) gives a strong measurement signal.
3) Use dY increments as the raw observation (option). For small dt, currents dY/dt amplify noise.

Experiments (3 main ones, paper-friendly):
------------------------------------------
Exp1 (Main): fixed configuration, plot target vs predictions (EV vs Traj) and save predictions CSV.
Exp2 (Lag sweep): vary lags_per_qubit, report NRMSE curves + gap.
Exp3 (Shots sweep): vary N_shots, report NRMSE curves + gap.

Outputs written to --out_dir:
-----------------------------
- experiment_log.csv              (all runs/metrics)
- predictions_exp1.csv            (per-time-step predictions for exp1)
- plot_exp1_timeseries.png
- plot_exp2_lag_sweep.png
- plot_exp2_gap_vs_lags.png
- plot_exp3_shots_sweep.png
- plot_exp3_gap_vs_shots.png
- final_figure_4panel.png         (stitched from 4 key plots)
- final_config.json               (best config used)

Run:
----
python qrc_mg_traj_methods_v2.py --out_dir qrc_mg_methods_v2_out
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
# Reservoir
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
    mg_seed: int


class QRC_TFI:
    def __init__(self, rp: ReservoirParams):
        self.rp = rp
        self.n = rp.n_qubits
        self.dim = 2**self.n

        rng = np.random.default_rng(rp.rseed)

        self.X_ops = [op_on_q(X2, i, self.n) for i in range(self.n)]
        self.Z_ops = [op_on_q(Z2, i, self.n) for i in range(self.n)]

        # Disordered TFI
        hx_i = rp.hx * (1.0 + rp.disorder * rng.normal(size=self.n))
        hz_i = rp.hz * (1.0 + rp.disorder * rng.normal(size=self.n))
        J_i = rp.J * (1.0 + rp.disorder * rng.normal(size=self.n))

        H0 = np.zeros((self.dim, self.dim), complex)
        for i in range(self.n):
            H0 += hx_i[i] * self.X_ops[i] + hz_i[i] * self.Z_ops[i]
        for i in range(self.n):
            j = (i + 1) % self.n
            H0 += J_i[i] * (self.Z_ops[i] @ self.Z_ops[j])

        # normalize spectrum
        H0 /= (np.max(np.abs(np.linalg.eigvalsh(H0))) + 1e-12)

        # Input Hamiltonian direction
        mask = rng.choice([-1, 1], self.n)
        if rp.input_axis.upper() == "X":
            Hin = sum(mask[i] * self.X_ops[i] for i in range(self.n))
        elif rp.input_axis.upper() == "Z":
            Hin = sum(mask[i] * self.Z_ops[i] for i in range(self.n))
        elif rp.input_axis.upper() == "XZ":
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
# EV vs trajectory evaluation
# ---------------------------
def eval_ev_vs_traj(
    rp: ReservoirParams,
    rop: ReadoutParams,
    dp: DataParams,
    n_shots: int,
    master_seed: int,
) -> Dict[str, object]:
    series = generate_mackey_glass(tau=dp.mg_tau, seed=dp.mg_seed)
    total = dp.train_len + dp.test_len
    u = series[:total]
    y = series[1 : total + 1]

    qrc = QRC_TFI(rp)
    Xshots = qrc.simulate(inputs=u, n_shots=n_shots, seed_base=master_seed * 999, lags_per_qubit=rop.lags_per_qubit)

    Xtr = Xshots[:, : dp.train_len, :]
    Xte = Xshots[:, dp.train_len :, :]
    ytr = y[: dp.train_len]
    yte = y[dp.train_len :]

    # EV baseline
    Xtr_ev = Xtr.mean(axis=0)
    Xte_ev = Xte.mean(axis=0)
    mod_ev = Ridge(alpha=rop.ridge_alpha).fit(Xtr_ev, ytr)
    pred_ev = mod_ev.predict(Xte_ev)
    err_ev = nrmse(yte, pred_ev)

    # trajectory-level
    Xtr_stack = Xtr.reshape(-1, Xtr.shape[-1])
    ytr_stack = np.tile(ytr, n_shots)
    mod_traj = Ridge(alpha=rop.ridge_alpha).fit(Xtr_stack, ytr_stack)
    preds = np.stack([mod_traj.predict(Xte[i]) for i in range(n_shots)], axis=0)
    pred_traj = preds.mean(axis=0)
    err_traj = nrmse(yte, pred_traj)

    return dict(
        y_test=yte,
        pred_ev=pred_ev,
        pred_traj=pred_traj,
        nrmse_ev=err_ev,
        nrmse_traj=err_traj,
        gap=float(err_ev - err_traj),
        feature_dim=int(Xshots.shape[-1]),
    )


# ---------------------------
# Plot utilities
# ---------------------------
def plot_exp1_timeseries(out_path: str, y: np.ndarray, pev: np.ndarray, ptr: np.ndarray, ev: float, tr: float, n: int = 250) -> None:
    n = min(n, len(y))
    plt.figure(figsize=(10, 4))
    plt.plot(y[:n], label="Target", lw=2, alpha=0.6)
    plt.plot(pev[:n], "--", label=f"EV (NRMSE={ev:.3f})")
    plt.plot(ptr[:n], label=f"Traj (NRMSE={tr:.3f})")
    plt.legend()
    plt.title("Exp1: Mackey–Glass next-step prediction")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_sweep(out_path: str, xs: List[int], evs: List[float], trs: List[float], xlabel: str, title: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(xs, evs, marker="o", linestyle="--", label="EV")
    plt.plot(xs, trs, marker="o", linestyle="-", label="Trajectory")
    plt.xlabel(xlabel)
    plt.ylabel("NRMSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_gap(out_path: str, xs: List[int], gaps: List[float], xlabel: str, title: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(xs, gaps, marker="o")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel("Gap = NRMSE_EV − NRMSE_traj")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
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
def exp1_main(rp: ReservoirParams, rop: ReadoutParams, dp: DataParams, n_shots: int, seed: int, out_dir: str) -> Dict[str, object]:
    res = eval_ev_vs_traj(rp, rop, dp, n_shots, seed)

    # save predictions
    dfp = pd.DataFrame({
        "t": np.arange(len(res["y_test"])),
        "y_target": res["y_test"],
        "y_pred_ev": res["pred_ev"],
        "y_pred_traj": res["pred_traj"],
    })
    dfp.to_csv(os.path.join(out_dir, "predictions_exp1.csv"), index=False)

    # plot
    plot_exp1_timeseries(
        os.path.join(out_dir, "plot_exp1_timeseries.png"),
        res["y_test"], res["pred_ev"], res["pred_traj"], res["nrmse_ev"], res["nrmse_traj"]
    )
    return res


def exp2_lag_sweep(rp: ReservoirParams, dp: DataParams, n_shots: int, seed: int, out_dir: str, lags_list: List[int], ridge_alpha: float) -> pd.DataFrame:
    rows = []
    evs, trs, gaps = [], [], []

    for L in lags_list:
        rop = ReadoutParams(lags_per_qubit=L, ridge_alpha=ridge_alpha)
        res = eval_ev_vs_traj(rp, rop, dp, n_shots, seed)
        rows.append(dict(lags_per_qubit=L, ridge_alpha=ridge_alpha, nrmse_ev=res["nrmse_ev"], nrmse_traj=res["nrmse_traj"], gap=res["gap"]))
        evs.append(res["nrmse_ev"]); trs.append(res["nrmse_traj"]); gaps.append(res["gap"])

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "exp2_lag_sweep.csv"), index=False)

    plot_sweep(os.path.join(out_dir, "plot_exp2_lag_sweep.png"), lags_list, evs, trs, "lags_per_qubit", "Exp2: Lag sweep")
    plot_gap(os.path.join(out_dir, "plot_exp2_gap_vs_lags.png"), lags_list, gaps, "lags_per_qubit", "Exp2: Gap vs lags")
    return df


def exp3_shots_sweep(rp: ReservoirParams, rop: ReadoutParams, dp: DataParams, seed: int, out_dir: str, shots_list: List[int]) -> pd.DataFrame:
    rows = []
    evs, trs, gaps = [], [], []

    for N in shots_list:
        res = eval_ev_vs_traj(rp, rop, dp, N, seed)
        rows.append(dict(n_shots=N, nrmse_ev=res["nrmse_ev"], nrmse_traj=res["nrmse_traj"], gap=res["gap"]))
        evs.append(res["nrmse_ev"]); trs.append(res["nrmse_traj"]); gaps.append(res["gap"])

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "exp3_shots_sweep.csv"), index=False)

    plot_sweep(os.path.join(out_dir, "plot_exp3_shots_sweep.png"), shots_list, evs, trs, "n_shots", "Exp3: Shots sweep")
    plot_gap(os.path.join(out_dir, "plot_exp3_gap_vs_shots.png"), shots_list, gaps, "n_shots", "Exp3: Gap vs shots")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="qrc_mg_methods_v2_out")
    ap.add_argument("--seed", type=int, default=17)

    # data split
    ap.add_argument("--train_len", type=int, default=300)
    ap.add_argument("--test_len", type=int, default=500)
    ap.add_argument("--mg_tau", type=int, default=17)
    ap.add_argument("--mg_seed", type=int, default=42)

    # main config (best found in this session)
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

    ap.add_argument("--n_shots", type=int, default=20)
    ap.add_argument("--lags_per_qubit", type=int, default=120)
    ap.add_argument("--ridge_alpha", type=float, default=0.005)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    use_increments = True
    if args.use_currents:
        use_increments = False
    if args.use_increments:
        use_increments = True

    dp = DataParams(train_len=args.train_len, test_len=args.test_len, mg_tau=args.mg_tau, mg_seed=args.mg_seed)

    rp = ReservoirParams(
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
        rseed=args.seed,
        input_axis=args.input_axis,
        use_increments=use_increments,
    )
    rop = ReadoutParams(lags_per_qubit=args.lags_per_qubit, ridge_alpha=args.ridge_alpha)

    # Exp1
    res1 = exp1_main(rp, rop, dp, n_shots=args.n_shots, seed=args.seed, out_dir=args.out_dir)

    # Exp2 (lags)
    lags_list = [10, 20, 50, 80, 120]
    df2 = exp2_lag_sweep(rp, dp, n_shots=args.n_shots, seed=args.seed, out_dir=args.out_dir, lags_list=lags_list, ridge_alpha=args.ridge_alpha)

    # Exp3 (shots)
    shots_list = [1, 2, 5, 10, 20]
    df3 = exp3_shots_sweep(rp, rop, dp, seed=args.seed, out_dir=args.out_dir, shots_list=shots_list)

    # consolidated log
    rows = []
    rows.append(dict(experiment="exp1_main", n_shots=args.n_shots, lags_per_qubit=args.lags_per_qubit, ridge_alpha=args.ridge_alpha,
                     nrmse_ev=res1["nrmse_ev"], nrmse_traj=res1["nrmse_traj"], gap=res1["gap"], feature_dim=res1["feature_dim"], **asdict(rp), **asdict(dp)))
    for _, r in df2.iterrows():
        rows.append(dict(experiment="exp2_lag_sweep", n_shots=args.n_shots, lags_per_qubit=int(r["lags_per_qubit"]), ridge_alpha=float(r["ridge_alpha"]),
                         nrmse_ev=float(r["nrmse_ev"]), nrmse_traj=float(r["nrmse_traj"]), gap=float(r["gap"]), feature_dim=int(args.n_qubits*int(r["lags_per_qubit"])+1),
                         **asdict(rp), **asdict(dp)))
    for _, r in df3.iterrows():
        rows.append(dict(experiment="exp3_shots_sweep", n_shots=int(r["n_shots"]), lags_per_qubit=args.lags_per_qubit, ridge_alpha=args.ridge_alpha,
                         nrmse_ev=float(r["nrmse_ev"]), nrmse_traj=float(r["nrmse_traj"]), gap=float(r["gap"]), feature_dim=res1["feature_dim"],
                         **asdict(rp), **asdict(dp)))

    df_log = pd.DataFrame(rows)
    df_log.to_csv(os.path.join(args.out_dir, "experiment_log.csv"), index=False)

    # stitch final 4-panel plot
    panel_paths = [
        os.path.join(args.out_dir, "plot_exp1_timeseries.png"),
        os.path.join(args.out_dir, "plot_exp2_lag_sweep.png"),
        os.path.join(args.out_dir, "plot_exp3_shots_sweep.png"),
        os.path.join(args.out_dir, "plot_exp2_gap_vs_lags.png"),
    ]
    stitch_2x2(panel_paths, os.path.join(args.out_dir, "final_figure_4panel.png"))

    # write final config
    with open(os.path.join(args.out_dir, "final_config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rp=asdict(rp), rop=asdict(rop), dp=asdict(dp), exp1_metrics=dict(nrmse_ev=res1["nrmse_ev"], nrmse_traj=res1["nrmse_traj"], gap=res1["gap"])), f, indent=2)

    print(f"Exp1: NRMSE_EV={res1['nrmse_ev']:.4f}  NRMSE_traj={res1['nrmse_traj']:.4f}  Gap={res1['gap']:+.4f}")
    print(f"Outputs written to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
