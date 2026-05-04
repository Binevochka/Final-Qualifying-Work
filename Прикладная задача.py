
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


# ============================================================
# Console helpers
# ============================================================

def get_int(prompt: str, default: int) -> int:
    try:
        raw = input(f"{prompt} [{default}]: ").strip()
        return int(raw) if raw else default
    except Exception:
        return default


def get_float(prompt: str, default: float) -> float:
    try:
        raw = input(f"{prompt} [{default}]: ").strip()
        return float(raw) if raw else default
    except Exception:
        return default


def get_choice(prompt: str, options: Sequence[str], default: str) -> str:
    print(prompt)
    for i, item in enumerate(options, start=1):
        print(f"  {i}. {item}")
    try:
        raw = input(f"Выберите [1-{len(options)}] [{default}]: ").strip()
        if not raw:
            return default
        idx = int(raw) - 1
        if 0 <= idx < len(options):
            return options[idx]
    except Exception:
        pass
    return default


# ============================================================
# Initial designs
# ============================================================

def init_uniform_grid(bounds: np.ndarray, n: int) -> np.ndarray:
    dim = bounds.shape[0]
    if n <= 1:
        return ((bounds[:, 0] + bounds[:, 1]) / 2.0).reshape(1, -1)

    per_dim = max(2, int(round(n ** (1.0 / dim))))
    axes = [np.linspace(bounds[i, 0], bounds[i, 1], per_dim) for i in range(dim)]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.column_stack([m.ravel() for m in mesh])

    if grid.shape[0] >= n:
        idx = np.linspace(0, grid.shape[0] - 1, n, dtype=int)
        return grid[idx]

    reps = math.ceil(n / grid.shape[0])
    return np.tile(grid, (reps, 1))[:n]


def init_lhs(bounds: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    dim = bounds.shape[0]
    X = np.zeros((n, dim), dtype=float)
    for j in range(dim):
        perm = rng.permutation(n)
        u = (perm + rng.random(n)) / n
        lo, hi = bounds[j]
        X[:, j] = lo + (hi - lo) * u
    return X


def init_uniform_random(bounds: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, bounds.shape[0]))


def init_chaos(bounds: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    dim = bounds.shape[0]
    z = rng.uniform(0.1, 0.9, size=dim)
    X = np.zeros((n, dim), dtype=float)
    for i in range(n):
        z = 4.0 * z * (1.0 - z)
        X[i] = bounds[:, 0] + z * (bounds[:, 1] - bounds[:, 0])
    return X


# ============================================================
# Aircraft control problem (fast but more stable)
# ============================================================

@dataclass
class AircraftParams:
    n0: float = 0.7
    n22: float = 2.5
    n32: float = 16.0
    n33: float = 2.2
    nb: float = 100.0


@dataclass
class AircraftControlProblemFast:
    bounds: np.ndarray
    aircraft: AircraftParams
    T: float
    dt: float
    memory_delta: float
    theta_ref_values: tuple[float, ...]
    theta0_values: tuple[float, ...]
    q0_values: tuple[float, ...]
    alpha0_values: tuple[float, ...]
    delta_b_limit: float
    integral_limit: float
    state_abs_limit: float
    qdot_limit: float
    diverged_penalty: float
    terminal_penalty_coef: float
    control_penalty_coef: float
    overshoot_penalty_coef: float

    @classmethod
    def create_default(cls) -> "AircraftControlProblemFast":
        # bounds narrowed around empirically stable region
        return cls(
            bounds=np.array([
                [-1.6, -0.2],   # Knp
                [ 0.00,  0.90], # KD1
                [-0.06,  0.08], # KD2
                [-0.80,  0.20], # KI
            ], dtype=float),
            aircraft=AircraftParams(),
            T=4.0,
            dt=0.02,
            memory_delta=0.20,
            theta_ref_values=(0.03, 0.07),
            theta0_values=(-0.02, 0.02),
            q0_values=(-0.02, 0.02),
            alpha0_values=(-0.02, 0.02),
            delta_b_limit=0.18,
            integral_limit=0.30,
            state_abs_limit=0.70,
            qdot_limit=6.0,
            diverged_penalty=200.0,
            terminal_penalty_coef=12.0,
            control_penalty_coef=0.03,
            overshoot_penalty_coef=2.5,
        )

    @property
    def dim(self) -> int:
        return 4

    def initial_states_matrix(self) -> np.ndarray:
        states = []
        for th0 in self.theta0_values:
            for q0 in self.q0_values:
                for a0 in self.alpha0_values:
                    states.append([th0, q0, a0])
        return np.asarray(states, dtype=float)

    def rhs_batch(self, X: np.ndarray, delta_b: np.ndarray) -> np.ndarray:
        theta = X[:, 0]
        q = X[:, 1]
        alpha = X[:, 2]
        alpha_dot = q - self.aircraft.n22 * alpha
        q_dot = -self.aircraft.n0 * alpha - self.aircraft.n32 * alpha_dot - self.aircraft.n33 * q - self.aircraft.nb * delta_b
        theta_dot = q
        return np.column_stack([theta_dot, q_dot, alpha_dot])

    def _simulate_one_ref_batch(self, gains: np.ndarray, theta_ref: float, X0: np.ndarray) -> float:
        Knp, KD1, KD2, KI = [float(v) for v in gains]
        dt = self.dt
        n_steps = int(round(self.T / dt)) + 1
        n_states = X0.shape[0]

        X = X0.copy()
        err_int = np.zeros(n_states, dtype=float)

        ise = np.zeros(n_states, dtype=float)
        control_energy = np.zeros(n_states, dtype=float)
        max_overshoot = np.zeros(n_states, dtype=float)

        bad_mask = np.zeros(n_states, dtype=bool)

        for _ in range(n_steps):
            theta = X[:, 0]
            q = X[:, 1]
            alpha = X[:, 2]

            e = theta_ref - theta
            err_int += e * dt
            err_int = np.clip(err_int, -self.integral_limit, self.integral_limit)

            alpha_dot = q - self.aircraft.n22 * alpha
            q_dot_free = -self.aircraft.n0 * alpha - self.aircraft.n32 * alpha_dot - self.aircraft.n33 * q

            u = Knp * e + KD1 * q + KD2 * q_dot_free + KI * err_int
            u = np.clip(u, -self.delta_b_limit, self.delta_b_limit)

            K1 = self.rhs_batch(X, u)
            K2 = self.rhs_batch(X + 0.5 * dt * K1, u)
            K3 = self.rhs_batch(X + 0.5 * dt * K2, u)
            K4 = self.rhs_batch(X + dt * K3, u)
            X = X + (dt / 6.0) * (K1 + 2 * K2 + 2 * K3 + K4)

            ise += (e ** 2) * dt
            control_energy += (u ** 2) * dt
            max_overshoot = np.maximum(max_overshoot, np.maximum(theta - theta_ref, 0.0))

            state_max = np.max(np.abs(X), axis=1)
            qdot_abs = np.abs(q_dot_free)
            bad_mask |= (
                ~np.isfinite(state_max) |
                (state_max > self.state_abs_limit) |
                (qdot_abs > self.qdot_limit)
            )

            if np.all(bad_mask):
                break

        terminal_error = np.abs(theta_ref - X[:, 0])

        cost = (
            ise
            + self.control_penalty_coef * control_energy
            + self.terminal_penalty_coef * terminal_error
            + self.overshoot_penalty_coef * max_overshoot
        )
        cost[bad_mask] += self.diverged_penalty
        return float(np.mean(cost))

    def __call__(self, gains: np.ndarray) -> float:
        gains = np.asarray(gains, dtype=float)
        X0 = self.initial_states_matrix()

        J_values = []
        for theta_ref in self.theta_ref_values:
            J_values.append(self._simulate_one_ref_batch(gains, theta_ref, X0))

        J = float(np.mean(J_values))
        return J if np.isfinite(J) else self.diverged_penalty

    def simulate_best_trajectory(self, gains: np.ndarray, theta_ref: float = 0.03):
        Knp, KD1, KD2, KI = [float(v) for v in gains]
        dt = self.dt
        n_steps = int(round(self.T / dt)) + 1
        t_grid = np.linspace(0.0, self.T, n_steps)

        x = np.array([0.0, 0.0, 0.0], dtype=float)
        err_int = 0.0

        theta_hist = np.zeros(n_steps, dtype=float)
        ref_hist = np.full(n_steps, theta_ref, dtype=float)
        u_hist = np.zeros(n_steps, dtype=float)

        for i in range(n_steps):
            theta, q, alpha = x
            e = theta_ref - theta
            err_int += e * dt
            err_int = float(np.clip(err_int, -self.integral_limit, self.integral_limit))

            alpha_dot = q - self.aircraft.n22 * alpha
            q_dot_free = -self.aircraft.n0 * alpha - self.aircraft.n32 * alpha_dot - self.aircraft.n33 * q
            u = Knp * e + KD1 * q + KD2 * q_dot_free + KI * err_int
            u = float(np.clip(u, -self.delta_b_limit, self.delta_b_limit))

            theta_hist[i] = theta
            u_hist[i] = u

            def rhs(xx: np.ndarray) -> np.ndarray:
                th, qq, aa = xx
                aa_dot = qq - self.aircraft.n22 * aa
                qq_dot = -self.aircraft.n0 * aa - self.aircraft.n32 * aa_dot - self.aircraft.n33 * qq - self.aircraft.nb * u
                th_dot = qq
                return np.array([th_dot, qq_dot, aa_dot], dtype=float)

            if i < n_steps - 1:
                k1 = rhs(x)
                k2 = rhs(x + 0.5 * dt * k1)
                k3 = rhs(x + 0.5 * dt * k2)
                k4 = rhs(x + dt * k3)
                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return t_grid, ref_hist, theta_hist, u_hist


# ============================================================
# RBF machinery
# ============================================================

class RBFKernel:
    def __init__(self, name: str = "multiquadric", epsilon: float = 1.0) -> None:
        self.name = name.lower()
        self.epsilon = float(epsilon)

    def __call__(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        er = self.epsilon * r
        if self.name == "gaussian":
            return np.exp(-(er ** 2))
        if self.name == "multiquadric":
            return np.sqrt(1.0 + er ** 2)
        if self.name == "inverse_quadratic":
            return 1.0 / (1.0 + er ** 2)
        if self.name == "thin_plate":
            out = np.zeros_like(r)
            mask = r > 1e-15
            out[mask] = (r[mask] ** 2) * np.log(r[mask])
            return out
        if self.name == "polyharmonic":
            return r ** 3
        raise ValueError(f"Unsupported kernel: {self.name}")


def build_poly_matrix(X: np.ndarray, degree: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n, dim = X.shape
    cols = [np.ones(n)]
    if degree >= 1:
        for j in range(dim):
            cols.append(X[:, j])
    return np.column_stack(cols)


@dataclass
class SolveResult:
    x_best: np.ndarray
    f_best: float
    n_evals: int
    n_iters: int
    msaa_triggered: int
    best_history: list[float]


class GLISMSAAFastSolver:
    def __init__(
        self,
        problem: AircraftControlProblemFast,
        n_basis: int,
        max_passes: int,
        rbf_type: str,
        epsilon_rbf: float,
        poly_degree: int,
        init_method: str,
        lambda_reg: float,
        alpha: float,
        delta: float,
        epsilon_deltaF: float,
        n_random_candidates: int,
        seed: int,
        patience: int,
        T0: float,
        C_boltz: float,
        beta_temp: float,
        N_msaa: int,
    ) -> None:
        self.problem = problem
        self.dim = problem.dim
        self.bounds = problem.bounds.copy()
        self.n_basis = int(n_basis)
        self.max_passes = int(max_passes)
        self.rbf = RBFKernel(rbf_type, epsilon_rbf)
        self.poly_degree = int(poly_degree)
        self.init_method = init_method.lower()
        self.lambda_reg = float(lambda_reg)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.epsilon_deltaF = float(epsilon_deltaF)
        self.n_random_candidates = int(n_random_candidates)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.patience = int(patience)
        self.T0 = float(T0)
        self.C_boltz = float(C_boltz)
        self.beta_temp = float(beta_temp)
        self.N_msaa = int(N_msaa)

        self.X = np.empty((0, self.dim))
        self.Y = np.empty((0,))
        self.n_evals = 0
        self.best_history: list[float] = []

    def _default_stable_points(self) -> np.ndarray:
        # a few hand-crafted stable starting points inside bounds
        pts = np.array([
            [-1.20, 0.20, 0.00, -0.20],
            [-1.00, 0.30, 0.02, -0.40],
            [-1.35, 0.35, 0.01, -0.55],
            [-0.90, 0.10, 0.00, -0.10],
        ], dtype=float)
        return np.clip(pts, self.bounds[:, 0], self.bounds[:, 1])

    def _initialize(self) -> None:
        n_random = max(0, self.n_basis - 4)
        if self.init_method == "lhs":
            X0 = init_lhs(self.bounds, n_random, self.rng) if n_random > 0 else np.empty((0, self.dim))
        elif self.init_method == "uniform_random":
            X0 = init_uniform_random(self.bounds, n_random, self.rng) if n_random > 0 else np.empty((0, self.dim))
        elif self.init_method == "uniform_grid":
            X0 = init_uniform_grid(self.bounds, n_random) if n_random > 0 else np.empty((0, self.dim))
        elif self.init_method == "chaos":
            X0 = init_chaos(self.bounds, n_random, self.rng) if n_random > 0 else np.empty((0, self.dim))
        else:
            raise ValueError(f"Unknown init method: {self.init_method}")

        seeds = self._default_stable_points()
        self.X = np.vstack([X0, seeds])[:self.n_basis]
        self.Y = np.array([self.problem(x) for x in self.X], dtype=float)
        self.n_evals += len(self.Y)
        self.best_history = [float(np.min(self.Y))]

    def _fit_surrogate(self):
        N = self.X.shape[0]
        D = cdist(self.X, self.X)
        Phi = self.rbf(D)
        P = build_poly_matrix(self.X, self.poly_degree)
        A11 = Phi.T @ Phi + self.lambda_reg * np.eye(N)
        A12 = Phi.T @ P
        A21 = P.T @ Phi
        A22 = P.T @ P
        b1 = Phi.T @ self.Y
        b2 = P.T @ self.Y
        A = np.block([[A11, A12], [A21, A22]])
        b = np.concatenate([b1, b2])
        coeffs = np.linalg.pinv(A) @ b
        return coeffs[:N], coeffs[N:]

    def _predict_batch(self, Xq: np.ndarray, w: np.ndarray, beta: np.ndarray) -> np.ndarray:
        D = cdist(Xq, self.X)
        return self.rbf(D) @ w + build_poly_matrix(Xq, self.poly_degree) @ beta

    def _uncertainty_batch(self, Xq: np.ndarray, w: np.ndarray, beta: np.ndarray) -> np.ndarray:
        D = np.maximum(cdist(Xq, self.X), 1e-10)
        W = 1.0 / D**2
        W /= np.sum(W, axis=1, keepdims=True)
        f_hat = self._predict_batch(Xq, w, beta)
        return np.sqrt(np.sum(W * (self.Y.reshape(1, -1) - f_hat.reshape(-1, 1))**2, axis=1))

    def _novelty_batch(self, Xq: np.ndarray) -> np.ndarray:
        D = np.maximum(cdist(Xq, self.X), 1e-10)
        raw = 1.0 / D**2
        prox = (2.0 / np.pi) * np.arctan(np.sum(raw, axis=1))
        return 1.0 - prox

    def _dedup_rows(self, X: np.ndarray) -> np.ndarray:
        rounded = np.round(X, decimals=6)
        _, idx = np.unique(rounded, axis=0, return_index=True)
        return X[np.sort(idx)]

    def _candidate_pool(self) -> np.ndarray:
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        width = hi - lo

        # Global candidates
        Xg = self.rng.uniform(lo, hi, size=(self.n_random_candidates, self.dim))

        # Local around best and top few
        best_idx = np.argsort(self.Y)[: min(4, len(self.Y))]
        centers = self.X[best_idx]
        blocks = [Xg]
        for c in centers:
            for scale, count in [(0.08, 100), (0.03, 80)]:
                block = c.reshape(1, -1) + self.rng.normal(scale=scale * width, size=(count, self.dim))
                blocks.append(np.clip(block, lo, hi))

        # Midpoints / recombinations of good points
        if len(centers) >= 2:
            mids = []
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    mids.append(0.5 * (centers[i] + centers[j]))
            mids = np.asarray(mids, dtype=float)
            if mids.size > 0:
                blocks.append(np.clip(mids, lo, hi))

        pool = np.vstack(blocks + [centers, self._default_stable_points()])
        return self._dedup_rows(pool)

    def _select_two_points(self, w: np.ndarray, beta: np.ndarray, k: int):
        pool = self._candidate_pool()
        yhat = self._predict_batch(pool, w, beta)
        s = self._uncertainty_batch(pool, w, beta)
        z = self._novelty_batch(pool)
        dF = float(np.max(self.Y) - np.min(self.Y) + self.epsilon_deltaF)

        y_norm = np.maximum((yhat - np.min(self.Y)) / dF, 0.0)
        s_norm = s / dF

        alpha_k = max(self.alpha * (1.0 - 0.4 * k / max(self.max_passes, 1)), 0.8)
        delta_k = max(self.delta * (1.0 - 0.5 * k / max(self.max_passes, 1)), 0.04)

        acq = y_norm - alpha_k * s_norm - delta_k * z

        order_y = np.argsort(yhat)
        order_a = np.argsort(acq)

        x_rs = pool[order_y[0]].copy()

        # make second point different from first
        x_acq = None
        for idx in order_a:
            cand = pool[idx]
            if np.linalg.norm(cand - x_rs) > 1e-6:
                x_acq = cand.copy()
                break
        if x_acq is None:
            x_acq = pool[order_a[0]].copy()

        return x_rs, x_acq

    def _update_basis(self, x1: np.ndarray, f1: float, x2: np.ndarray, f2: float) -> bool:
        old_best = float(np.min(self.Y))
        X_aug = np.vstack([self.X, x1.reshape(1, -1), x2.reshape(1, -1)])
        Y_aug = np.concatenate([self.Y, [f1, f2]])
        keep = np.argsort(Y_aug)[: self.n_basis]
        self.X = X_aug[keep]
        self.Y = Y_aug[keep]
        new_best = float(np.min(self.Y))
        self.best_history.append(new_best)
        return new_best < old_best - 1e-12

    def _run_msaa(self, x0: np.ndarray):
        x = x0.copy()
        fx = self.problem(x)
        self.n_evals += 1

        x_best = x.copy()
        f_best = fx
        T = self.T0
        width = self.bounds[:, 1] - self.bounds[:, 0]

        for _ in range(self.N_msaa):
            cand = x + self.rng.normal(scale=0.05 * width, size=self.dim)
            cand = np.clip(cand, self.bounds[:, 0], self.bounds[:, 1])
            fc = self.problem(cand)
            self.n_evals += 1

            df = fc - fx
            if df <= 0 or self.rng.random() < math.exp(-df / max(self.C_boltz * T, 1e-12)):
                x, fx = cand, fc
                if fx < f_best:
                    x_best, f_best = x.copy(), fx

            T *= self.beta_temp

        return x_best, f_best

    def solve(self, verbose: bool = True) -> SolveResult:
        self._initialize()
        no_improve = 0
        msaa_triggered = 0

        for k in range(1, self.max_passes + 1):
            w, beta = self._fit_surrogate()
            x_rs, x_acq = self._select_two_points(w, beta, k)

            f_rs = self.problem(x_rs)
            f_acq = self.problem(x_acq)
            self.n_evals += 2

            improved = self._update_basis(x_rs, f_rs, x_acq, f_acq)
            no_improve = 0 if improved else no_improve + 1

            if verbose:
                print(
                    f"[iter {k:02d}] f_RS={f_rs:.6f}, f_Acq={f_acq:.6f}, "
                    f"f_best={self.best_history[-1]:.6f}, no_improve={no_improve}"
                )

            if no_improve >= self.patience:
                msaa_triggered += 1
                x_best = self.X[np.argmin(self.Y)]
                x_m, f_m = self._run_msaa(x_best)
                current_best = float(np.min(self.Y))
                self._update_basis(x_m, f_m, x_best, current_best)
                no_improve = 0
                if verbose:
                    print(f"  -> MSAA: f_MSAA={f_m:.6f}, f_best={self.best_history[-1]:.6f}")

        ib = int(np.argmin(self.Y))
        return SolveResult(
            x_best=self.X[ib].copy(),
            f_best=float(self.Y[ib]),
            n_evals=self.n_evals,
            n_iters=self.max_passes,
            msaa_triggered=msaa_triggered,
            best_history=self.best_history.copy(),
        )


# ============================================================
# Reporting and plots
# ============================================================

def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_best_convergence(best_history: list[float], title: str, out_path: Path | None = None) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(len(best_history)), best_history, marker="o")
    plt.xlabel("Итерация")
    plt.ylabel("Лучшее значение J")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_best_trajectory(problem: AircraftControlProblemFast, gains: np.ndarray, out_path: Path | None = None) -> None:
    t, theta_ref, theta, u = problem.simulate_best_trajectory(gains, theta_ref=problem.theta_ref_values[0])

    plt.figure(figsize=(8, 4.5))
    plt.plot(t, theta_ref, "--", linewidth=1.8, label=r"$\vartheta_{zad}(t)$")
    plt.plot(t, theta, linewidth=2.0, label=r"$\vartheta(t)$")
    plt.xlabel("t")
    plt.ylabel("Угол тангажа")
    plt.title("Переходный процесс для лучших найденных параметров")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 100)
    print("УЛУЧШЕННАЯ прикладная задача: GLIS-MSAA для настройки регулятора продольного движения")
    print("=" * 100)
    print("Версия с 20 комбинациями и более устойчивой генерацией кандидатов.")

    run_mode = get_choice("Режим запуска:", ["все комбинации", "одна комбинация"], "все комбинации")

    problem = AircraftControlProblemFast.create_default()

    M = get_int("M (число базисных точек)", 12)
    P_max = get_int("P_max", 18)
    n_random_candidates = get_int("Число кандидатов", 350)
    patience = get_int("patience", 4)
    N_msaa = get_int("N_msaa", 35)
    base_seed = get_int("base_seed", 42)

    poly_degree = 1
    epsilon_rbf = 1.0
    lambda_reg = 2e-3
    alpha = 1.4
    delta = 0.12
    epsilon_deltaF = 1e-8
    T0 = 0.4
    C_boltz = 0.85
    beta_temp = 0.95

    all_inits = ["lhs", "uniform_random", "uniform_grid", "chaos"]
    all_kernels = ["multiquadric", "gaussian", "inverse_quadratic", "thin_plate", "polyharmonic"]

    if run_mode == "одна комбинация":
        init_methods = [get_choice("Метод инициализации:", all_inits, "lhs")]
        kernels = [get_choice("Ядро:", all_kernels, "gaussian")]
    else:
        init_methods = all_inits
        kernels = all_kernels

    print(f"\nБудут запущены все комбинации: {len(init_methods)} × {len(kernels)} = {len(init_methods) * len(kernels)}")

    out_dir = Path("aircraft_glis_msaa_improved_results")
    out_dir.mkdir(exist_ok=True)

    rows = []
    best_row = None
    best_history = None
    best_x = None

    combo_idx = 0
    total = len(init_methods) * len(kernels)

    for init_method in init_methods:
        for kernel in kernels:
            combo_idx += 1
            seed = base_seed + 31 * combo_idx

            print("\n" + "-" * 100)
            print(f"Комбинация {combo_idx:02d}/{total}: init={init_method}, kernel={kernel}, seed={seed}")

            solver = GLISMSAAFastSolver(
                problem=problem,
                n_basis=M,
                max_passes=P_max,
                rbf_type=kernel,
                epsilon_rbf=epsilon_rbf,
                poly_degree=poly_degree,
                init_method=init_method,
                lambda_reg=lambda_reg,
                alpha=alpha,
                delta=delta,
                epsilon_deltaF=epsilon_deltaF,
                n_random_candidates=n_random_candidates,
                seed=seed,
                patience=patience,
                T0=T0,
                C_boltz=C_boltz,
                beta_temp=beta_temp,
                N_msaa=N_msaa,
            )

            result = solver.solve(verbose=True)

            row = {
                "init_method": init_method,
                "kernel": kernel,
                "seed": seed,
                "J_best": result.f_best,
                "n_evals": result.n_evals,
                "n_iters": result.n_iters,
                "msaa_triggered": result.msaa_triggered,
                "Knp": result.x_best[0],
                "KD1": result.x_best[1],
                "KD2": result.x_best[2],
                "KI": result.x_best[3],
            }
            rows.append(row)

            if best_row is None or row["J_best"] < best_row["J_best"]:
                best_row = row
                best_history = result.best_history.copy()
                best_x = result.x_best.copy()

            print(
                f"Итог: J_best={result.f_best:.8f}, evals={result.n_evals}, "
                f"msaa_triggered={result.msaa_triggered}, x_best={np.round(result.x_best, 6)}"
            )

    rows_sorted = sorted(rows, key=lambda r: r["J_best"])

    print("\n" + "=" * 100)
    print("СВОДКА ПО ВСЕМ КОМБИНАЦИЯМ")
    print("=" * 100)
    for i, row in enumerate(rows_sorted, start=1):
        print(
            f"{i:02d}. init={row['init_method']:<14} kernel={row['kernel']:<18} "
            f"J_best={row['J_best']:.8f}  evals={row['n_evals']:<5} msaa={row['msaa_triggered']}"
        )

    if best_row is not None and best_history is not None and best_x is not None:
        print("\nЛУЧШАЯ КОМБИНАЦИЯ")
        print(
            f"init={best_row['init_method']}, kernel={best_row['kernel']}, "
            f"seed={best_row['seed']}, J_best={best_row['J_best']:.8f}"
        )
        print(f"Knp = {best_x[0]: .6f}")
        print(f"KD1 = {best_x[1]: .6f}")
        print(f"KD2 = {best_x[2]: .6f}")
        print(f"KI  = {best_x[3]: .6f}")

        csv_path = out_dir / "summary_all_combinations.csv"
        save_csv(csv_path, rows_sorted)

        conv_path = out_dir / "best_convergence.png"
        traj_path = out_dir / "best_theta_trajectory.png"

        plot_best_convergence(
            best_history,
            f"Сходимость лучшей комбинации: {best_row['init_method']} + {best_row['kernel']}",
            conv_path,
        )
        plot_best_trajectory(problem, best_x, traj_path)

        print(f"\nГрафики сохранены в папку: {out_dir.resolve()}")
        print(f"CSV-сводка сохранена: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
