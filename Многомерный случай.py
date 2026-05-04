from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence
import csv
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
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
# Problem definition
# ============================================================
def omega_from_k2(K2: float) -> float:
    return math.pi * K2 / 5.0


def base_component(z: np.ndarray, H: float, K2: float) -> np.ndarray:
    w = omega_from_k2(K2)
    z = np.asarray(z, dtype=float)
    return 0.5 * z ** 2 + H * (1.0 - np.cos(w * z))


@dataclass
class ProblemND:
    dim: int
    bounds: np.ndarray
    c1: float
    c2: float
    H: float
    K2: float
    mode: str
    seed: int
    x_star: np.ndarray
    Q: np.ndarray

    @classmethod
    def create(
            cls,
            dim: int,
            bounds: np.ndarray,
            c1: float,
            c2: float,
            H: float,
            K2: float,
            mode: str,
            seed: int,
    ) -> "ProblemND":
        rng = np.random.default_rng(seed + 17 * dim)
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (dim, 2):
            raise ValueError("bounds must have shape (dim, 2)")

        lo = bounds[:, 0]
        hi = bounds[:, 1]
        width = hi - lo

        # Internal non-grid optimum for meaningful globality tests
        frac = np.linspace(0.29, 0.71, dim) + 0.07 * np.sin(np.arange(1, dim + 1))
        frac = np.clip(frac, 0.18, 0.82)
        x_star = lo + frac * width

        # random orthogonal mixing matrix
        A = rng.normal(size=(dim, dim))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1.0

        if mode == "separable_boundary":
            x_star = hi.copy()
            Q = np.eye(dim)
        elif mode == "separable_interior_shifted":
            Q = np.eye(dim)
        elif mode == "strict_globality":
            pass
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return cls(
            dim=dim,
            bounds=bounds,
            c1=c1,
            c2=c2,
            H=H,
            K2=K2,
            mode=mode,
            seed=seed,
            x_star=x_star,
            Q=Q,
        )

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Z = (X - self.x_star) @ self.Q.T
        return Z

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        Z = self._transform(X)
        vals = 4.0 + np.mean(base_component(Z, self.H, self.K2), axis=1)
        return vals.astype(float)

    def __call__(self, x: np.ndarray) -> float:
        return float(self.evaluate_batch(np.asarray(x, dtype=float).reshape(1, -1))[0])

    @property
    def f_ref(self) -> float:
        return 4.0

    @property
    def x_ref(self) -> np.ndarray:
        return self.x_star.copy()


# ============================================================
# RBF kernel
# ============================================================
class RBFKernel:
    def __init__(self, name: str = "gaussian", epsilon: float = 1.0) -> None:
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
            mask = r > 0
            out[mask] = (r[mask] ** 2) * np.log(r[mask])
            return out
        if self.name == "polyharmonic":
            out = np.zeros_like(r)
            mask = r > 0
            out[mask] = r[mask] ** 3
            return out
        raise ValueError(f"Unsupported RBF kernel: {self.name}")


# ============================================================
# Initialization
# ============================================================
def make_bounds_matrix(dim: int, same_bounds: tuple[float, float] | None = None) -> np.ndarray:
    if same_bounds is None:
        raise ValueError("same_bounds is required")
    lo, hi = same_bounds
    bounds = np.zeros((dim, 2), dtype=float)
    bounds[:, 0] = lo
    bounds[:, 1] = hi
    return bounds


def init_uniform_grid_nd(bounds: np.ndarray, n: int) -> np.ndarray:
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
    out = np.tile(grid, (reps, 1))[:n]
    return out


def init_lhs_nd(bounds: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    dim = bounds.shape[0]
    X = np.zeros((n, dim), dtype=float)
    for j in range(dim):
        perm = rng.permutation(n)
        u = rng.random(n)
        X[:, j] = bounds[j, 0] + (perm + u) / n * (bounds[j, 1] - bounds[j, 0])
    return X


def init_uniform_random_nd(bounds: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return rng.uniform(lo, hi, size=(n, bounds.shape[0]))


def init_chaos_nd(bounds: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    dim = bounds.shape[0]
    X = np.zeros((n, dim), dtype=float)
    z = rng.random(dim)
    z = np.where(z <= 0.0, 0.123456789, z)
    for i in range(n):
        z = 4.0 * z * (1.0 - z)
        X[i] = bounds[:, 0] + z * (bounds[:, 1] - bounds[:, 0])
    return X


# ============================================================
# Solver internals
# ============================================================
def build_poly_matrix(X: np.ndarray, degree: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n, dim = X.shape
    cols = [np.ones(n)]
    if degree >= 1:
        for j in range(dim):
            cols.append(X[:, j])
    if degree >= 2:
        for j in range(dim):
            cols.append(X[:, j] ** 2)
    return np.column_stack(cols)


@dataclass
class SolveResult:
    x_best: np.ndarray
    f_best: float
    n_evals: int
    n_iters: int
    msaa_triggered: int
    initial_hit_global: int
    first_global_iter: int
    first_global_source: str
    msaa_improved_best: int
    best_history: list[float]
    basis_history: list[dict]  # Track basis changes


class GLISSolverND:
    def __init__(
            self,
            problem: ProblemND,
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
            n_restarts: int,
            n_random_candidates: int,
            seed: int,
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
        self.n_restarts = int(n_restarts)
        self.n_random_candidates = int(n_random_candidates)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.X = np.empty((0, self.dim))
        self.Y = np.empty((0,))
        self.x_best = np.zeros(self.dim)
        self.f_best = np.inf
        self.n_evals = 0
        self.best_history: list[float] = []
        self.basis_history: list[dict] = []

        width = np.mean(self.bounds[:, 1] - self.bounds[:, 0])
        self.duplicate_tol = 1e-5 * max(width, 1.0)

    def _initialize_basis(self) -> None:
        if self.init_method == "uniform_grid":
            X0 = init_uniform_grid_nd(self.bounds, self.n_basis)
        elif self.init_method == "lhs":
            X0 = init_lhs_nd(self.bounds, self.n_basis, self.rng)
        elif self.init_method == "uniform_random":
            X0 = init_uniform_random_nd(self.bounds, self.n_basis, self.rng)
        elif self.init_method == "chaos":
            X0 = init_chaos_nd(self.bounds, self.n_basis, self.rng)
        else:
            raise ValueError(f"Unknown init method: {self.init_method}")

        Y0 = self.problem.evaluate_batch(X0)
        self.n_evals += len(Y0)
        self.X = X0
        self.Y = Y0

        idx = int(np.argmin(self.Y))
        self.x_best = self.X[idx].copy()
        self.f_best = float(self.Y[idx])
        self.best_history = [self.f_best]

        # Track initial basis
        self.basis_history.append({
            'iter': 0,
            'stage': 'init',
            'basis_size': len(self.X),
            'x_best': self.x_best.copy(),
            'f_best': self.f_best,
            'basis_changed': True,
            'added_points': [],
            'removed_points': [],
            'added_count': 0,
            'removed_count': 0,
            'basis_y_sorted': np.sort(self.Y.copy()),
            'basis_y_min': float(np.min(self.Y)),
            'basis_y_mean': float(np.mean(self.Y)),
            'basis_y_max': float(np.max(self.Y)),
        })

    def _clip_many_to_bounds(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.bounds[:, 0], self.bounds[:, 1])

    def _is_too_close(self, x: np.ndarray, extra_points: Optional[list[np.ndarray]] = None) -> bool:
        x = np.asarray(x, dtype=float).reshape(1, -1)
        if self.X.shape[0] > 0:
            d0 = np.linalg.norm(self.X - x, axis=1)
            if np.any(d0 <= self.duplicate_tol):
                return True
        if extra_points:
            E = np.asarray(extra_points, dtype=float)
            if E.ndim == 1:
                E = E.reshape(1, -1)
            d1 = np.linalg.norm(E - x, axis=1)
            if np.any(d1 <= self.duplicate_tol):
                return True
        return False

    def _basis_signature(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            return X.reshape(0, self.dim)
        keys = tuple(X[:, j] for j in range(X.shape[1] - 1, -1, -1))
        order = np.lexsort(keys)
        return X[order]

    def _fit_surrogate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        ИСПРАВЛЕНИЕ: Используем псевдообратную матрицу Мур-Пенроуза
        вместо обычного решения СЛАУ для устойчивости
        """
        N = self.X.shape[0]
        D = cdist(self.X, self.X)
        Phi = self.rbf(D)
        P = build_poly_matrix(self.X, self.poly_degree)

        # Build the augmented system
        A11 = Phi.T @ Phi + self.lambda_reg * np.eye(N)
        A12 = Phi.T @ P
        A21 = P.T @ Phi
        A22 = P.T @ P

        b1 = Phi.T @ self.Y
        b2 = P.T @ self.Y

        A = np.block([[A11, A12], [A21, A22]])
        b = np.concatenate([b1, b2])

        # ИСПОЛЬЗУЕМ ПСЕВДООБРАТНУЮ МАТРИЦУ (Moore-Penrose)
        try:
            # Псевдообратная матрица для прямоугольных или плохо обусловленных матриц
            A_pinv = np.linalg.pinv(A)
            coeffs = A_pinv @ b
        except Exception:
            # Fallback to lstsq if pinv fails
            coeffs = np.linalg.lstsq(A, b, rcond=None)[0]

        w = coeffs[:N]
        beta = coeffs[N:]
        return w, beta

    def _predict(self, x: np.ndarray, w: np.ndarray, beta: np.ndarray) -> float:
        return float(self._predict_batch(np.asarray(x, dtype=float).reshape(1, -1), w, beta)[0])

    def _predict_batch(self, Xq: np.ndarray, w: np.ndarray, beta: np.ndarray) -> np.ndarray:
        Xq = np.asarray(Xq, dtype=float)
        D = cdist(Xq, self.X)
        Phi_q = self.rbf(D)
        P_q = build_poly_matrix(Xq, self.poly_degree)
        return Phi_q @ w + P_q @ beta

    def _idw_raw_batch(self, Xq: np.ndarray) -> np.ndarray:
        D = cdist(np.asarray(Xq, dtype=float), self.X)
        D = np.maximum(D, 1e-12)
        return np.exp(-(D ** 2)) / (D ** 2)

    def _idw_weights_batch(self, Xq: np.ndarray) -> np.ndarray:
        W = self._idw_raw_batch(Xq)
        sums = np.sum(W, axis=1, keepdims=True)
        return np.divide(W, np.maximum(sums, 1e-12))

    def _novelty_batch(self, Xq: np.ndarray) -> np.ndarray:
        Xq = np.asarray(Xq, dtype=float)
        W_raw = self._idw_raw_batch(Xq)
        proximity = (2.0 / np.pi) * np.arctan(np.sum(W_raw, axis=1))
        Z = 1.0 - proximity
        for i, x in enumerate(Xq):
            if self._is_too_close(x):
                Z[i] = 0.0
        return Z

    def _uncertainty_batch(self, Xq: np.ndarray, w: np.ndarray, beta: np.ndarray) -> np.ndarray:
        f_hat = self._predict_batch(Xq, w, beta)
        V = self._idw_weights_batch(Xq)
        residuals = (self.Y.reshape(1, -1) - f_hat.reshape(-1, 1)) ** 2
        return np.sqrt(np.sum(V * residuals, axis=1))

    def _acquisition_batch(self, Xq: np.ndarray, w: np.ndarray, beta: np.ndarray, k: int) -> np.ndarray:
        f_hat = self._predict_batch(Xq, w, beta)
        s_x = self._uncertainty_batch(Xq, w, beta)
        z_x = self._novelty_batch(Xq)

        y_min = float(np.min(self.Y))
        y_max = float(np.max(self.Y))
        delta_F = float(y_max - y_min + self.epsilon_deltaF)

        f_hat_norm = np.maximum((f_hat - y_min) / delta_F, 0.0)
        s_norm = s_x / delta_F

        alpha_k = self.alpha * (1.0 - 0.6 * (k / max(self.max_passes, 1)))
        alpha_k = max(alpha_k, 1.0)

        delta_k = self.delta * (1.0 - 0.7 * (k / max(self.max_passes, 1)))
        delta_k = max(delta_k, 0.03)

        promising = np.exp(-2.0 * f_hat_norm)
        bad_penalty = 1.2 * np.maximum(0.0, f_hat_norm - 0.9) ** 2

        dist_to_low = Xq - self.bounds[:, 0]
        dist_to_high = self.bounds[:, 1] - Xq
        dist_to_edge = np.min(np.minimum(dist_to_low, dist_to_high), axis=1)
        edge_zone = 0.03 * np.mean(self.bounds[:, 1] - self.bounds[:, 0])
        edge_penalty = np.where(
            dist_to_edge < edge_zone,
            0.15 * (edge_zone - dist_to_edge) / max(edge_zone, 1e-12),
            0.0,
        )

        min_dist = np.min(cdist(Xq, self.X), axis=1)
        duplicate_penalty = np.where(min_dist <= self.duplicate_tol, 1e6, 0.0)

        return f_hat_norm - alpha_k * s_norm - delta_k * z_x * promising + bad_penalty + edge_penalty + duplicate_penalty

    def _candidate_pool(self) -> np.ndarray:
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        width = hi - lo
        mean_width = float(np.mean(width))

        # 1) глобальный пул
        n_global = self.n_random_candidates
        global_pool = self.rng.uniform(lo, hi, size=(n_global, self.dim))

        # 2) несколько локальных облаков вокруг текущего лучшего решения
        local_blocks = []
        sigmas_best = [0.20 * width, 0.10 * width, 0.04 * width]
        counts_best = [
            max(300, self.n_random_candidates // 2),
            max(200, self.n_random_candidates // 3),
            max(120, self.n_random_candidates // 5),
        ]
        for sig, cnt in zip(sigmas_best, counts_best):
            block = self.x_best.reshape(1, -1) + self.rng.normal(scale=sig, size=(cnt, self.dim))
            local_blocks.append(self._clip_many_to_bounds(block))

        # 3) облака вокруг нескольких лучших точек базиса
        top_k = min(8, self.X.shape[0])
        top_idx = np.argsort(self.Y)[:top_k]
        top_centers = self.X[top_idx] if top_k > 0 else np.empty((0, self.dim))
        around_top = []
        if top_k > 0:
            per_top = max(80, self.n_random_candidates // 10)
            for center in top_centers:
                for sig in (0.12 * width, 0.05 * width):
                    block = center.reshape(1, -1) + self.rng.normal(scale=sig, size=(per_top, self.dim))
                    around_top.append(self._clip_many_to_bounds(block))
        top_pool = np.vstack(around_top) if around_top else np.empty((0, self.dim))

        # 4) смешивание и экстраполяция между лучшими точками
        mix_blocks = []
        if self.X.shape[0] >= 2:
            pair_n = max(200, self.n_random_candidates // 3)
            idx_a = self.rng.integers(0, self.X.shape[0], size=pair_n)
            idx_b = self.rng.integers(0, self.X.shape[0], size=pair_n)
            Xa = self.X[idx_a]
            Xb = self.X[idx_b]
            lam = self.rng.random((pair_n, 1))
            mixed = lam * Xa + (1.0 - lam) * Xb
            mixed += self.rng.normal(scale=0.03 * width, size=mixed.shape)
            mix_blocks.append(self._clip_many_to_bounds(mixed))

            # дифференциальная экстраполяция
            eta = self.rng.uniform(0.3, 1.2, size=(pair_n, 1))
            diff = Xa + eta * (Xa - Xb)
            diff += self.rng.normal(scale=0.02 * width, size=diff.shape)
            mix_blocks.append(self._clip_many_to_bounds(diff))

        # 5) специальные 2D-кандидаты для strict_globality
        ring_pool = np.empty((0, self.dim))
        if self.dim == 2 and getattr(self.problem, "mode", "") == "strict_globality":
            angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
            radii = [0.03 * mean_width, 0.08 * mean_width, 0.16 * mean_width, 0.28 * mean_width]
            circles = []
            for r in radii:
                circ = np.column_stack([
                    self.x_best[0] + r * np.cos(angles),
                    self.x_best[1] + r * np.sin(angles),
                ])
                circles.append(self._clip_many_to_bounds(circ))
            ring_pool = np.vstack(circles)

        # 6) якорные точки
        anchors = [self.x_best.reshape(1, -1)]
        if top_k > 0:
            anchors.append(top_centers)
        anchors.append(((lo + hi) / 2.0).reshape(1, -1))

        pool = np.vstack([
            global_pool,
            *local_blocks,
            top_pool,
            *mix_blocks,
            ring_pool,
            *anchors,
        ])

        # 7) грубая дедупликация
        rounded = np.round(pool / max(self.duplicate_tol, 1e-12), decimals=0)
        _, uniq_idx = np.unique(rounded, axis=0, return_index=True)
        pool = pool[np.sort(uniq_idx)]

        return pool

    def _minimize_nd_function(
            self,
            func: Callable[[np.ndarray], float],
            avoid_points: Optional[list[np.ndarray]],
            batch_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> tuple[np.ndarray, float]:
        pool = self._candidate_pool()
        vals = np.asarray(
            batch_func(pool) if batch_func is not None else [func(x) for x in pool],
            dtype=float
        ).reshape(-1)
        order = np.argsort(vals)

        # 1) элитные сырые кандидаты
        elite_raw: list[tuple[np.ndarray, float]] = []
        max_raw = min(len(order), max(20, 3 * self.n_restarts))
        for idx in order[:max_raw]:
            x0 = pool[idx]
            if self._is_too_close(x0, avoid_points):
                continue
            elite_raw.append((x0.copy(), float(vals[idx])))

        if not elite_raw:
            x0 = self.x_best.copy()
            return x0, float(func(x0))

        best_x = min(elite_raw, key=lambda t: t[1])[0].copy()
        best_val = min(elite_raw, key=lambda t: t[1])[1]

        # 2) старты для локальной доработки
        starts: list[np.ndarray] = [x.copy() for x, _ in elite_raw[:max(self.n_restarts, 10)]]
        starts.append(self.x_best.copy())
        top_basis_k = min(5, self.X.shape[0])
        if top_basis_k > 0:
            top_basis_idx = np.argsort(self.Y)[:top_basis_k]
            for idx in top_basis_idx:
                starts.append(self.X[idx].copy())

        # дедупликация стартов
        clean_starts: list[np.ndarray] = []
        for s in starts:
            if self._is_too_close(s, clean_starts):
                continue
            if self._is_too_close(s, avoid_points):
                continue
            clean_starts.append(s.copy())

        width = self.bounds[:, 1] - self.bounds[:, 0]

        # 3) локальная доработка
        for x0 in clean_starts[: max(self.n_restarts + 4, 12)]:
            res = minimize(
                lambda xx: float(func(xx)),
                x0=x0,
                method="L-BFGS-B",
                bounds=[tuple(b) for b in self.bounds],
                options={"maxiter": 40, "ftol": 1e-10},
            )
            cand_x = self._clip_many_to_bounds(np.asarray(res.x, dtype=float).reshape(1, -1))[0]
            if not self._is_too_close(cand_x, avoid_points):
                cand_val = float(func(cand_x))
                if cand_val < best_val:
                    best_val = cand_val
                    best_x = cand_x.copy()

        # 4) дополнительная доработка вокруг найденного локального минимума
        n_jitter = 8 if (self.dim == 2 and getattr(self.problem, "mode", "") == "strict_globality") else 4
        for scale in (0.05, 0.015):
            jitter_block = cand_x.reshape(1, -1) + self.rng.normal(scale=scale * width, size=(n_jitter, self.dim))
            jitter_block = self._clip_many_to_bounds(jitter_block)
            if batch_func is not None:
                jitter_vals = np.asarray(batch_func(jitter_block), dtype=float).reshape(-1)
            else:
                jitter_vals = np.array([float(func(z)) for z in jitter_block], dtype=float)
            j_idx = int(np.argmin(jitter_vals))
            j_x = jitter_block[j_idx]
            j_val = float(jitter_vals[j_idx])
            if not self._is_too_close(j_x, avoid_points) and j_val < best_val:
                best_val = j_val
                best_x = j_x.copy()

        # 5) fallback
        return best_x.copy(), float(best_val)

    def _update_basis_two_points(self, x_rs: np.ndarray, f_rs: float, x_acq: np.ndarray, f_acq: float) -> bool:
        """
        Добавляем две новые точки, удаляем две наихудшие и явно отслеживаем,
        какие точки реально вошли в новый базис и какие были удалены.
        """
        prev_best = self.f_best
        old_X = self.X.copy()
        old_sig = self._basis_signature(old_X)

        X_aug = np.vstack([self.X, x_rs.reshape(1, -1), x_acq.reshape(1, -1)])
        Y_aug = np.concatenate([self.Y, [f_rs, f_acq]])

        keep = np.argsort(Y_aug)[: self.n_basis]
        self.X = X_aug[keep]
        self.Y = Y_aug[keep]

        idx = int(np.argmin(self.Y))
        self.x_best = self.X[idx].copy()
        self.f_best = float(self.Y[idx])
        self.best_history.append(self.f_best)

        candidate_points = [x_rs.copy(), x_acq.copy()]
        added_points = [
            p.copy() for p in candidate_points
            if any(np.allclose(p, q, atol=self.duplicate_tol) for q in self.X)
        ]
        removed_points = [
            p.copy() for p in old_X
            if not any(np.allclose(p, q, atol=self.duplicate_tol) for q in self.X)
        ]
        basis_changed = not np.allclose(old_sig, self._basis_signature(self.X), atol=self.duplicate_tol)

        self.basis_history.append({
            'iter': len(self.basis_history),
            'stage': 'rs_acq',
            'basis_size': len(self.X),
            'x_best': self.x_best.copy(),
            'f_best': self.f_best,
            'basis_changed': basis_changed,
            'added_points': added_points,
            'removed_points': removed_points,
            'added_count': len(added_points),
            'removed_count': len(removed_points),
            'basis_y_sorted': np.sort(self.Y.copy()),
            'basis_y_min': float(np.min(self.Y)),
            'basis_y_mean': float(np.mean(self.Y)),
            'basis_y_max': float(np.max(self.Y)),
            'x_rs': x_rs.copy(),
            'f_rs': f_rs,
            'x_acq': x_acq.copy(),
            'f_acq': f_acq
        })

        return self.f_best < prev_best - 1e-12


class GLISMSAASolverND(GLISSolverND):
    def __init__(
            self,
            *args,
            patience: int,
            T0: float,
            C_boltz: float,
            beta_temp: float,
            N_msaa: int,
            constraint_mode: str,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.patience = int(patience)
        self.T0 = float(T0)
        self.C_boltz = float(C_boltz)
        self.beta_temp = float(beta_temp)
        self.N_msaa = int(N_msaa)
        self.constraint_mode = constraint_mode.lower()

    def _handle_constraint(self, x: np.ndarray) -> Optional[np.ndarray]:
        x = np.asarray(x, dtype=float)
        if self.constraint_mode == "reject":
            if np.any(x < self.bounds[:, 0]) or np.any(x > self.bounds[:, 1]):
                return None
            return x
        if self.constraint_mode == "random_reset":
            bad = (x < self.bounds[:, 0]) | (x > self.bounds[:, 1])
            if np.any(bad):
                x = x.copy()
                x[bad] = self.rng.uniform(self.bounds[bad, 0], self.bounds[bad, 1])
            return x
        return self._clip_many_to_bounds(x.reshape(1, -1))[0]

    def _run_msaa(self, x0: np.ndarray) -> tuple[np.ndarray, float, bool]:
        x_curr = np.asarray(x0, dtype=float).copy()
        f_curr = self.problem(x_curr)
        self.n_evals += 1

        x_best = x_curr.copy()
        f_best = f_curr
        improved = False

        T = self.T0
        sigma = 0.12 * (self.bounds[:, 1] - self.bounds[:, 0])
        no_improve_local = 0

        for _ in range(self.N_msaa):
            cand = self._handle_constraint(x_curr + self.rng.normal(scale=sigma))
            if cand is None:
                T *= self.beta_temp
                continue

            f_cand = self.problem(cand)
            self.n_evals += 1

            delta_f = f_cand - f_curr
            if delta_f <= 0:
                accept = True
            else:
                denom = max(self.C_boltz * T, 1e-12)
                accept = self.rng.random() < math.exp(-delta_f / denom)

            if accept:
                x_curr = cand
                f_curr = f_cand
                if f_curr < f_best:
                    x_best = x_curr.copy()
                    f_best = f_curr
                    improved = True
                    no_improve_local = 0
                else:
                    no_improve_local += 1

            T *= self.beta_temp
            if no_improve_local >= 40:
                break

        return x_best, f_best, improved

    def _update_basis_one_point(self, x_new: np.ndarray, f_new: float) -> bool:
        if self._is_too_close(x_new):
            return False

        old_X = self.X.copy()
        old_sig = self._basis_signature(old_X)

        X_aug = np.vstack([self.X, x_new.reshape(1, -1)])
        Y_aug = np.concatenate([self.Y, [f_new]])

        keep = np.argsort(Y_aug)[: self.n_basis]
        self.X = X_aug[keep]
        self.Y = Y_aug[keep]

        idx = int(np.argmin(self.Y))
        self.x_best = self.X[idx].copy()
        self.f_best = float(self.Y[idx])
        self.best_history.append(self.f_best)

        added_points = [x_new.copy()] if any(np.allclose(x_new, q, atol=self.duplicate_tol) for q in self.X) else []
        removed_points = [
            p.copy() for p in old_X
            if not any(np.allclose(p, q, atol=self.duplicate_tol) for q in self.X)
        ]
        basis_changed = not np.allclose(old_sig, self._basis_signature(self.X), atol=self.duplicate_tol)

        self.basis_history.append({
            'iter': len(self.basis_history),
            'stage': 'msaa',
            'basis_size': len(self.X),
            'x_best': self.x_best.copy(),
            'f_best': self.f_best,
            'basis_changed': basis_changed,
            'added_points': added_points,
            'removed_points': removed_points,
            'added_count': len(added_points),
            'removed_count': len(removed_points),
            'basis_y_sorted': np.sort(self.Y.copy()),
            'basis_y_min': float(np.min(self.Y)),
            'basis_y_mean': float(np.mean(self.Y)),
            'basis_y_max': float(np.max(self.Y)),
            'x_msaa': x_new.copy(),
            'f_msaa': f_new,
        })

        return basis_changed

    def solve(self, verbose: bool = True) -> SolveResult:
        self._initialize_basis()

        initial_hit_global = int(
            np.linalg.norm(self.x_best - self.problem.x_ref) <= 1e-3 and
            abs(self.f_best - self.problem.f_ref) <= 1e-6
        )
        first_global_iter = 0 if initial_hit_global else -1
        first_global_source = "init" if initial_hit_global else "none"

        msaa_triggered = 0
        msaa_improved_best = 0
        no_improve_counter = 0

        if verbose:
            print(f"Старт: f_best = {self.f_best:.8f}, x_best = {np.round(self.x_best, 6)}")

        for k in range(1, self.max_passes + 1):
            w, beta = self._fit_surrogate()

            avoid = [x.copy() for x in self.X]

            # Минимизируем суррогат (RS)
            x_rs, _ = self._minimize_nd_function(
                lambda x: self._predict(x, w, beta),
                avoid_points=avoid,
                batch_func=lambda Xq: self._predict_batch(Xq, w, beta),
            )
            f_rs = self.problem(x_rs)
            self.n_evals += 1

            # Минимизируем функцию приобретения (Acq)
            x_acq, _ = self._minimize_nd_function(
                lambda x: float(self._acquisition_batch(np.asarray(x, dtype=float).reshape(1, -1), w, beta, k)[0]),
                avoid_points=avoid + [x_rs],
                batch_func=lambda Xq: self._acquisition_batch(Xq, w, beta, k),
            )
            f_acq = self.problem(x_acq)
            self.n_evals += 1

            # Обновляем базис с двумя новыми точками
            improved = self._update_basis_two_points(x_rs, f_rs, x_acq, f_acq)

            if first_global_iter < 0 and (
                    np.linalg.norm(self.x_best - self.problem.x_ref) <= 1e-3 and
                    abs(self.f_best - self.problem.f_ref) <= 1e-6
            ):
                first_global_iter = k
                first_global_source = "RS/Acq"

            no_improve_counter = 0 if improved else no_improve_counter + 1

            if verbose:
                print(
                    f"[GLIS-MSAA | {k:03d}] f_RS={f_rs:.6f}, f_Acq={f_acq:.6f}, "
                    f"f_best={self.f_best:.6f}, no_improve={no_improve_counter}"
                )
                # Показываем, что произошло с базисом
                last_basis = self.basis_history[-1]
                if len(last_basis['added_points']) > 0:
                    print(f"         → Добавлено точек: {len(last_basis['added_points'])}, "
                          f"размер базиса: {last_basis['basis_size']}")

            if no_improve_counter >= self.patience:
                msaa_triggered += 1
                x_msaa, f_msaa, improved_msaa = self._run_msaa(self.x_best)

                if improved_msaa:
                    msaa_improved_best = 1

                changed = self._update_basis_one_point(x_msaa, f_msaa)

                if first_global_iter < 0 and (
                        np.linalg.norm(self.x_best - self.problem.x_ref) <= 1e-3 and
                        abs(self.f_best - self.problem.f_ref) <= 1e-6
                ):
                    first_global_iter = k
                    first_global_source = "MSAA"

                if verbose:
                    print(f"  -> MSAA: f_MSAA={f_msaa:.6f}, basis_changed_after_msaa={changed}")

                no_improve_counter = 0
                if not changed:
                    if verbose:
                        print("Остановка: после MSAA множество базисных решений не изменилось.")
                    break

        return SolveResult(
            x_best=self.x_best.copy(),
            f_best=self.f_best,
            n_evals=self.n_evals,
            n_iters=k,
            msaa_triggered=msaa_triggered,
            initial_hit_global=initial_hit_global,
            first_global_iter=first_global_iter,
            first_global_source=first_global_source,
            msaa_improved_best=msaa_improved_best,
            best_history=self.best_history.copy(),
            basis_history=self.basis_history.copy(),
        )


# ============================================================
# Presets and reporting
# ============================================================
def suggested_defaults(dim: int) -> dict[str, int | float]:
    if dim <= 2:
        return {
            "M": 30,
            "P_max": 40,
            "cands": 2600,
            "restarts": 10,
            "N_msaa": 340,
            "msaa_restarts": 5,
            "T0": 150.0
        }
    if dim <= 3:
        return {
            "M": 24,
            "P_max": 30,
            "cands": 1200,
            "restarts": 5,
            "N_msaa": 180,
            "T0": 150.0
        }
    if dim <= 5:
        return {
            "M": 24,
            "P_max": 28,
            "cands": 1400,
            "restarts": 5,
            "N_msaa": 180,
            "T0": 150.0
        }
    return {
        "M": 20,
        "P_max": 24,
        "cands": 1600,
        "restarts": 4,
        "N_msaa": 150,
        "T0": 100.0
    }


def plot_convergence(best_history: list[float], title: str, out_path: Path | None = None, f_ref: float | None = None) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(best_history)), best_history, marker="o")
    if f_ref is not None:
        plt.axhline(f_ref, linestyle="--", linewidth=1.5)
    plt.xlabel("Итерация")
    plt.ylabel("Лучшее значение f_best")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if out_path is None:
        plt.show()
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()


def plot_basis_value_profile(basis_y_sorted: list[float] | np.ndarray, title: str, out_path: Path) -> None:
    vals = np.asarray(basis_y_sorted, dtype=float).reshape(-1)
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(vals) + 1), vals, marker='o')
    plt.xlabel('Номер точки в отсортированном базисе')
    plt.ylabel('Значение целевой функции')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_iteration_diagnostics(entry: dict, best_history_prefix: list[float], combo_name: str, out_dir: Path, f_ref: float) -> None:
    iter_idx = int(entry.get('iter', 0))

    # 1) convergence up to current iteration
    plot_convergence(
        best_history_prefix,
        f"[{combo_name}] Сходимость до итерации {iter_idx}",
        out_dir / f"iter_{iter_idx:03d}_convergence.png",
        f_ref=f_ref,
    )

    # 2) current basis profile
    plot_basis_value_profile(
        entry.get('basis_y_sorted', []),
        f"[{combo_name}] Итерация {iter_idx}: значения базиса",
        out_dir / f"iter_{iter_idx:03d}_basis_values.png",
    )

    # 3) candidate / update diagnostics
    labels = ['f_best']
    values = [float(entry.get('f_best', np.nan))]
    if 'f_rs' in entry:
        labels.append('f_RS')
        values.append(float(entry['f_rs']))
    if 'f_acq' in entry:
        labels.append('f_Acq')
        values.append(float(entry['f_acq']))
    if 'f_msaa' in entry:
        labels.append('f_MSAA')
        values.append(float(entry['f_msaa']))
    if entry.get('basis_y_min') is not None:
        labels.extend(['basis_min', 'basis_mean', 'basis_max'])
        values.extend([
            float(entry['basis_y_min']),
            float(entry['basis_y_mean']),
            float(entry['basis_y_max']),
        ])

    plt.figure(figsize=(10, 5))
    xpos = np.arange(len(labels))
    plt.bar(xpos, values)
    plt.xticks(xpos, labels, rotation=20)
    plt.ylabel('Значение функции')
    plt.title(f"[{combo_name}] Итерация {iter_idx}: диагностика обновления")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    diag_path = out_dir / f"iter_{iter_idx:03d}_diagnostics.png"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(diag_path, dpi=150)
    plt.close()


def plot_basis_stats_history(basis_history: list[dict], combo_name: str, out_path: Path) -> None:
    iters = [int(e.get('iter', i)) for i, e in enumerate(basis_history)]
    mins = [float(e.get('basis_y_min', np.nan)) for e in basis_history]
    means = [float(e.get('basis_y_mean', np.nan)) for e in basis_history]
    maxs = [float(e.get('basis_y_max', np.nan)) for e in basis_history]

    plt.figure(figsize=(10, 5))
    plt.plot(iters, mins, marker='o', label='min')
    plt.plot(iters, means, marker='s', label='mean')
    plt.plot(iters, maxs, marker='^', label='max')
    plt.xlabel('Итерация')
    plt.ylabel('Значение функции по базису')
    plt.title(f"[{combo_name}] Эволюция статистик базиса")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_nd_run_plots(result: SolveResult, problem: ProblemND, combo_name: str, combo_dir: Path) -> None:
    combo_dir.mkdir(parents=True, exist_ok=True)

    # per-iteration plots
    for idx, entry in enumerate(result.basis_history):
        hist_prefix = result.best_history[: max(1, min(idx + 1, len(result.best_history)))]
        plot_iteration_diagnostics(entry, hist_prefix, combo_name, combo_dir, problem.f_ref)

    # final summary plots
    plot_convergence(
        result.best_history,
        f"[{combo_name}] Итоговая сходимость",
        combo_dir / 'convergence.png',
        f_ref=problem.f_ref,
    )
    if result.basis_history:
        plot_basis_value_profile(
            result.basis_history[-1].get('basis_y_sorted', []),
            f"[{combo_name}] Финальные значения базиса",
            combo_dir / 'final_basis_values.png',
        )
    plot_basis_stats_history(result.basis_history, combo_name, combo_dir / 'basis_stats_history.png')


def save_result_csv(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("=" * 100)
    print("GLIS / GLIS-MSAA — строгая многомерная проверка глобальности (ИСПРАВЛЕННАЯ ВЕРСИЯ)")
    print("=" * 100)

    c1 = get_float("c1", -4.0)
    c2 = get_float("c2", 13.0)
    H = get_float("H", 15.0)
    K2 = get_float("K2", 8.0)
    dim = get_int("n", 2)

    defaults = suggested_defaults(dim)

    solver_name = get_choice("Алгоритм:", ["GLIS-MSAA"], "GLIS-MSAA")

    mode = get_choice(
        "Режим тестовой функции:",
        ["strict_globality", "separable_interior_shifted", "separable_boundary"],
        "strict_globality",
    )

    run_mode = get_choice(
        "Режим перебора для одной размерности:",
        ["одна комбинация", "все комбинации"],
        "все комбинации",
    )

    M = get_int("M (число базисных точек)", int(defaults["M"]))
    P_max = get_int("P_max", int(defaults["P_max"]))
    poly_degree = get_int("Степень полиномиального хвоста (0/1/2)", 1)
    epsilon_rbf = get_float("epsilon_rbf", 1.0)
    lambda_reg = get_float("lambda", 2e-2 if dim <= 2 else (3e-2 if dim <= 5 else 5e-2))
    alpha = get_float("alpha", 6.5 if dim <= 2 else 6.0)
    delta = get_float("delta", 0.20 if dim <= 2 else (0.18 if dim <= 5 else 0.15))
    epsilon_deltaF = get_float("epsilon_deltaF", 1e-8)
    n_restarts = get_int("Число рестартов локальной минимизации", int(defaults["restarts"]))
    n_random_candidates = get_int("Число случайных кандидатов для ND", int(defaults["cands"]))
    base_seed = get_int("base_seed", 42)
    patience = get_int("patience", 9 if dim <= 2 else (8 if dim <= 5 else 7))

    # ИСПРАВЛЕННЫЕ ПАРАМЕТРЫ MSAA (по рекомендации научрука)
    print("\nПАРАМЕТРЫ MSAA (рекомендуемые):")
    T0 = get_float("T0 (начальная температура, 100-150)", float(defaults["T0"]))
    C_boltz = get_float("C (параметр Больцмана, 0.85)", 0.85)
    beta_temp = get_float("beta (охлаждение, 0.99)", 0.99)
    N_msaa = get_int("N (итераций MSAA, 100 или 1000)", int(defaults["N_msaa"]))

    constraint_mode = get_choice("Учёт ограничений в MSAA:", ["clip", "random_reset", "reject"], "clip")

    same_bounds = get_choice("Границы по координатам:", ["одинаковые для всех координат"],
                             "одинаковые для всех координат")
    lo = get_float("  Нижняя граница", -14.0)
    hi = get_float("  Верхняя граница", 9.0)

    bounds = make_bounds_matrix(dim, same_bounds=(lo, hi))

    all_inits = ["lhs", "uniform_random", "uniform_grid", "chaos"]
    all_kernels = ["gaussian", "multiquadric", "inverse_quadratic", "thin_plate", "polyharmonic"]

    if run_mode == "одна комбинация":
        init_methods = [get_choice("Метод инициализации:", all_inits, "lhs")]
        kernels = [get_choice("Тип RBF-ядра:", all_kernels, "gaussian")]
    else:
        init_methods = all_inits
        kernels = all_kernels

    print(
        f"\nБудут запущены все комбинации: {len(init_methods)} метода инициализации × {len(kernels)} ядер = {len(init_methods) * len(kernels)} запусков.")

    problem = ProblemND.create(
        dim=dim,
        bounds=bounds,
        c1=c1,
        c2=c2,
        H=H,
        K2=K2,
        mode=mode,
        seed=base_seed,
    )

    print("\n" + "=" * 100)
    print(f"РАЗМЕРНОСТЬ n = {dim}")
    print("=" * 100)
    print(f"Режим задачи: {mode}")
    print(f"Референс для диагностики: f_ref = {problem.f_ref:.8f}, x_ref = {np.round(problem.x_ref, 6)}")

    out_dir = Path("../GLIS_ND_Results") / f"n{dim}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"summary_n{dim}_all_combos.csv"

    rows = []
    best_row = None
    best_history = None
    combo_idx = 0

    for init_idx, init_method in enumerate(init_methods):
        for ker_idx, rbf_type in enumerate(kernels):
            combo_idx += 1
            seed = base_seed + 10 * combo_idx

            solver = GLISMSAASolverND(
                problem=problem,
                n_basis=M,
                max_passes=P_max,
                rbf_type=rbf_type,
                epsilon_rbf=epsilon_rbf,
                poly_degree=poly_degree,
                init_method=init_method,
                lambda_reg=lambda_reg,
                alpha=alpha,
                delta=delta,
                epsilon_deltaF=epsilon_deltaF,
                n_restarts=n_restarts,
                n_random_candidates=n_random_candidates,
                seed=seed,
                patience=patience,
                T0=T0,
                C_boltz=C_boltz,
                beta_temp=beta_temp,
                N_msaa=N_msaa,
                constraint_mode=constraint_mode,
            )

            print("\n" + "-" * 100)
            print(
                f"Комбинация {combo_idx:02d}/{len(init_methods) * len(kernels)}: "
                f"n={dim}, init={init_method}, kernel={rbf_type}, seed={seed}"
            )

            result = solver.solve(verbose=True)

            err_f = abs(result.f_best - problem.f_ref)
            err_x = float(np.linalg.norm(result.x_best - problem.x_ref))

            row = {
                "n_dim": dim,
                "problem_mode": mode,
                "algorithm": solver_name,
                "init_method": init_method,
                "kernel": rbf_type,
                "seed": seed,
                "f_found": result.f_best,
                "err_f": err_f,
                "err_x_l2": err_x,
                "n_evals": result.n_evals,
                "n_iters": result.n_iters,
                "msaa_triggered": result.msaa_triggered,
                "initial_hit_global": result.initial_hit_global,
                "first_global_iter": result.first_global_iter,
                "first_global_source": result.first_global_source,
                "msaa_improved_best": result.msaa_improved_best,
            }
            rows.append(row)
            save_result_csv(csv_path, row)

            combo_name = f"combo_{combo_idx:02d}_{init_method}_{rbf_type}"
            combo_dir = out_dir / combo_name
            save_nd_run_plots(result, problem, combo_name, combo_dir)

            if best_row is None or (row["err_f"], row["err_x_l2"], row["n_evals"]) < (
                    best_row["err_f"], best_row["err_x_l2"], best_row["n_evals"]
            ):
                best_row = row
                best_history = result.best_history.copy()

            print(
                f"Итог: f_best={result.f_best:.8f}, |err_f|={err_f:.8e}, |err_x|={err_x:.8e}, "
                f"evals={result.n_evals}, iters={result.n_iters}, initial_hit_global={result.initial_hit_global}, "
                f"first_global_iter={result.first_global_iter}, first_global_source={result.first_global_source}, "
                f"msaa_improved_best={result.msaa_improved_best}"
            )

    print("\n" + "=" * 100)
    print("СВОДКА ПО ВСЕМ КОМБИНАЦИЯМ")
    print("=" * 100)

    rows_sorted = sorted(rows, key=lambda r: (r["err_f"], r["err_x_l2"], r["n_evals"]))
    for i, row in enumerate(rows_sorted, start=1):
        print(
            f"{i:02d}. init={row['init_method']:<14} kernel={row['kernel']:<18} "
            f"err_f={row['err_f']:.8e}  err_x={row['err_x_l2']:.8e}  "
            f"evals={row['n_evals']:<5} iters={row['n_iters']:<4} "
            f"msaa_improved={row['msaa_improved_best']}"
        )

    if best_row is not None and best_history is not None:
        print("\nЛучшая комбинация:")
        print(
            f"init={best_row['init_method']}, kernel={best_row['kernel']}, seed={best_row['seed']}, "
            f"err_f={best_row['err_f']:.8e}, err_x={best_row['err_x_l2']:.8e}, "
            f"evals={best_row['n_evals']}, iters={best_row['n_iters']}"
        )

        plot_convergence(
            best_history,
            f"Сходимость лучшей комбинации | n={dim}, init={best_row['init_method']}, kernel={best_row['kernel']}",
            out_dir / 'best_combo_convergence.png',
            f_ref=problem.f_ref,
        )
        print(f"\n📊 Сводная таблица сохранена в: {csv_path.resolve()}")
        print(f"📁 Все многомерные графики сохранены в папку: {out_dir.resolve()}")


if __name__ == "__main__":
    main()