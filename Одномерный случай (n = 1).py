"""
GLIS-1D: Одномерная глобальная оптимизация с МНК-суррогатом (Финальная версия)
Включает: расчёт истинного минимума, улучшенные графики, сводную таблицу, CSV-экспорт,
полный консольный ввод параметров, строгое следование алгоритму GLIS.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import os
import time
import math
import csv

# ============================================================
# 1. КОНСОЛЬНЫЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================
def get_float(prompt, default):
    try:
        val = input(f"{prompt} [{default}]: ").strip()
        return float(val) if val else default
    except Exception:
        return default

def get_int(prompt, default):
    try:
        val = input(f"{prompt} [{default}]: ").strip()
        return int(val) if val else default
    except Exception:
        return default

def get_choice(prompt, options, default):
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    try:
        val = input(f"Выберите [1-{len(options)}] [{default}]: ").strip()
        if not val: return default
        idx = int(val) - 1
        return options[idx] if 0 <= idx < len(options) else default
    except Exception:
        return default

# ============================================================
# 2. ЦЕЛЕВАЯ ФУНКЦИЯ И ИСТИННЫЙ МИНИМУМ
# ============================================================
def make_target_function(c1, c2, H, K2):
    """s(x) = 0.5*(x-c2)^2 + 2 - H*cos(pi*K2/5*(x-c1)) + H"""
    def target(x):
        x = np.asarray(x, dtype=float)
        return 0.5 * (x - c2)**2 + 2.0 - H * np.cos((np.pi * K2 / 5.0) * (x - c1)) + H
    return target

def find_true_minimum(target_func, bounds, n_restarts=50):
    """Находит истинный глобальный минимум с высокой точностью"""
    lo, hi = bounds
    # 1. Плотная сетка
    x_grid = np.linspace(lo, hi, 20000)
    y_grid = target_func(x_grid)
    idx_min = np.argmin(y_grid)
    x_best, f_best = x_grid[idx_min], y_grid[idx_min]

    # 2. Локальная доводка из топ-N точек сетки
    top_idx = np.argsort(y_grid)[:n_restarts]
    for idx in top_idx:
        res = minimize(lambda xx: target_func(xx[0]), x0=[x_grid[idx]], method='L-BFGS-B', bounds=[bounds])
        if res.success and res.fun < f_best:
            f_best = res.fun
            x_best = np.clip(res.x[0], lo, hi)

    return float(x_best), float(f_best)

# ============================================================
# 3. RBF ЯДРА
# ============================================================
class RBFKernel:
    def __init__(self, name="gaussian", epsilon=1.0):
        self.name = name.lower()
        self.epsilon = float(epsilon)

    def __call__(self, r):
        r = np.asarray(r, dtype=float)
        er = self.epsilon * r
        if self.name == "gaussian":
            return np.exp(-(er**2))
        elif self.name == "multiquadric":
            return np.sqrt(1.0 + er**2)
        elif self.name == "inverse_quadratic":
            return 1.0 / (1.0 + er**2)
        elif self.name == "thin_plate":
            out = np.zeros_like(r)
            mask = r > 1e-12
            out[mask] = (r[mask]**2) * np.log(r[mask])
            return out
        elif self.name == "polyharmonic":
            return np.abs(r)**3
        raise ValueError(f"Неизвестное ядро: {self.name}")

# ============================================================
# 4. ИНИЦИАЛИЗАЦИЯ БАЗИСА
# ============================================================
def init_uniform_grid(bounds, n):
    return np.linspace(bounds[0], bounds[1], n)

def init_lhs(bounds, n, rng):
    perm = rng.permutation(n)
    u = rng.random(n)
    return bounds[0] + (perm + u) / n * (bounds[1] - bounds[0])

def init_uniform_random(bounds, n, rng):
    return rng.uniform(bounds[0], bounds[1], n)

def init_chaos(bounds, n, rng):
    z = rng.random()
    if z <= 0.0: z = 0.123456789
    X = np.zeros(n)
    for i in range(n):
        z = 4.0 * z * (1.0 - z)
        X[i] = bounds[0] + z * (bounds[1] - bounds[0])
    return X

# ============================================================
# 5. МНК-АППРОКСИМАЦИЯ (Формула 7 из ТЗ)
# ============================================================
def build_poly_matrix(x, degree):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    cols = [np.ones_like(x)]
    if degree >= 1: cols.append(x)
    if degree >= 2: cols.append(x**2)
    return np.column_stack(cols)

def fit_rbf_mnk(x_data, y_data, kernel, poly_degree, lambda_reg):
    n = len(x_data)
    x = np.asarray(x_data).reshape(-1)
    y = np.asarray(y_data).reshape(-1)

    D = np.abs(x[:, None] - x[None, :])
    Phi = kernel(D)
    P = build_poly_matrix(x, poly_degree)

    A11 = Phi.T @ Phi + lambda_reg * np.eye(n)
    A12 = Phi.T @ P
    A21 = P.T @ Phi
    A22 = P.T @ P
    b1 = Phi.T @ y
    b2 = P.T @ y

    A = np.block([[A11, A12], [A21, A22]])
    b = np.concatenate([b1, b2])

    # Псевдообратная матрица Мур-Пенроуза для устойчивости
    try:
        coeffs = np.linalg.pinv(A) @ b
    except Exception:
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
    return coeffs[:n], coeffs[n:]

def predict_rbf(x_query, x_data, w, beta, kernel, poly_degree):
    x_q = np.asarray(x_query).reshape(-1)
    x_d = np.asarray(x_data).reshape(-1)
    D = np.abs(x_q[:, None] - x_d[None, :])
    Phi_q = kernel(D)
    P_q = build_poly_matrix(x_q, poly_degree)
    return Phi_q @ w + P_q @ beta

# ============================================================
# 6. ФУНКЦИЯ ПРИОБРЕТЕНИЯ GLIS (Формулы 8-12 из ТЗ)
# ============================================================
def compute_idw_raw(x_query, x_data):
    x_q = np.asarray(x_query, dtype=float).reshape(-1)
    x_d = np.asarray(x_data, dtype=float).reshape(-1)
    dist = np.abs(x_q[:, None] - x_d[None, :])
    dist = np.maximum(dist, 1e-12)
    # Альтернативные IDW-веса: плавное затухание вдали от известных точек
    return np.exp(-(dist**2)) / (dist**2)


def compute_idw_weights(x_query, x_data):
    raw = compute_idw_raw(x_query, x_data)
    sums = np.sum(raw, axis=1, keepdims=True)
    return raw / np.maximum(sums, 1e-12)


def novelty_measure(x_query, x_data):
    x_q = np.asarray(x_query, dtype=float).reshape(-1)
    x_d = np.asarray(x_data, dtype=float).reshape(-1)
    raw = compute_idw_raw(x_q, x_d)
    proximity = (2.0 / np.pi) * np.arctan(np.sum(raw, axis=1))
    novelty = 1.0 - proximity

    # Если точка практически совпадает с уже имеющейся, новизна должна быть нулевой
    min_dist = np.min(np.abs(x_q[:, None] - x_d[None, :]), axis=1)
    novelty[min_dist <= 1e-8] = 0.0
    return novelty


def uncertainty_measure(x_query, x_data, y_data, f_hat):
    W = compute_idw_weights(x_query, x_data)
    residuals = (y_data.reshape(1, -1) - f_hat.reshape(-1, 1))**2
    return np.sqrt(np.sum(W * residuals, axis=1))


def acquisition_function(x_query, x_data, y_data, w, beta, kernel, poly_degree, alpha, delta):
    x_q = np.asarray(x_query, dtype=float).reshape(-1)
    f_hat = predict_rbf(x_q, x_data, w, beta, kernel, poly_degree)
    s_x = uncertainty_measure(x_q, x_data, y_data, f_hat)
    z_x = novelty_measure(x_q, x_data)

    y_min = np.min(y_data)
    y_max = np.max(y_data)
    delta_F = y_max - y_min + 1e-8

    f_norm = np.maximum((f_hat - y_min) / delta_F, 0.0)
    s_norm = s_x / delta_F

    # Жёсткий штраф за почти совпадающие точки, чтобы Acq не выбирала дубликаты
    min_dist = np.min(np.abs(x_q[:, None] - np.asarray(x_data, dtype=float)[None, :]), axis=1)
    duplicate_penalty = np.where(min_dist <= 1e-8, 1e6, 0.0)

    # GLIS: минимизируем суррогат, поощряем неопределённость и новизну
    return f_norm - alpha * s_norm - delta * z_x + duplicate_penalty

# ============================================================
# 7. ОПТИМИЗАЦИЯ В 1D
# ============================================================
def minimize_1d(func, bounds, n_restarts=5, eps=1e-5):
    lo, hi = bounds
    search_lo, search_hi = lo + eps, hi - eps
    grid = np.linspace(search_lo, search_hi, 1500)
    vals = np.asarray([func(x) for x in grid])
    best_idx = np.argmin(vals)
    best_x, best_val = grid[best_idx], vals[best_idx]

    starts = np.sort(grid[np.argsort(vals)[:n_restarts]])
    for x0 in starts:
        res = minimize(lambda xx: float(func(xx[0])), x0=[x0], method="L-BFGS-B", bounds=[(search_lo, search_hi)])
        cand = np.clip(res.x[0], lo, hi)
        cand_val = func(cand)
        if cand_val < best_val:
            best_val = cand_val
            best_x = cand

    # Проверка границ
    for edge in [lo, hi]:
        val = func(edge)
        if val < best_val:
            best_val, best_x = val, edge
    return best_x, float(best_val)

# ============================================================
# 8. ГРАФИКИ
# ============================================================
def plot_iteration(x_data, y_data, x_rs, f_rs, x_acq, f_acq, x_best, f_best,
                   x_grid, f_true, f_surrogate, acq_vals, bounds, x_true, f_true_min,
                   kernel_name, poly_deg, out_dir, step, combo_name):
    os.makedirs(out_dir, exist_ok=True)

    # График 1: Функция и суррогат
    plt.figure(figsize=(12, 7))
    plt.plot(x_grid, f_true, 'k-', linewidth=2.5, label='Истинная $s(x)$')
    plt.plot(x_grid, f_surrogate, 'b--', linewidth=1.5, label=f'RBF-суррогат ({kernel_name}, deg={poly_deg})')
    plt.scatter(x_data, y_data, c='gray', s=40, zorder=4, label='Базисные точки', edgecolors='black', linewidth=0.5)
    plt.scatter([x_rs], [f_rs], c='green', s=90, marker='x', zorder=6, label=f'RS-кандидат ({f_rs:.3f})')
    plt.scatter([x_acq], [f_acq], c='orange', s=90, marker='d', zorder=6, label=f'Acq-кандидат ({f_acq:.3f})')
    plt.scatter([x_best], [f_best], c='red', s=110, marker='*', zorder=7, label=f'Лучший найденный ({f_best:.3f})')
    plt.scatter([x_true], [f_true_min], c='purple', s=130, marker='P', zorder=8, label=f'Истинный минимум ({f_true_min:.3f})')
    plt.axvline(bounds[0], color='gray', linestyle=':', alpha=0.6)
    plt.axvline(bounds[1], color='gray', linestyle=':', alpha=0.6)
    plt.title(f"[{combo_name}] Итерация {step}: Суррогатная модель и кандидаты")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"iter_{step:03d}_surrogate.png"), dpi=150)
    plt.close()

    # График 2: Функция приобретения
    plt.figure(figsize=(12, 4))
    plt.plot(x_grid, acq_vals, 'm-', linewidth=2, label='Функция приобретения $I(x)$')
    idx_rs = np.argmin(np.abs(x_grid - x_rs))
    idx_acq = np.argmin(np.abs(x_grid - x_acq))
    plt.scatter([x_rs], [acq_vals[idx_rs]], c='green', s=80, marker='x', zorder=5, label='RS точка')
    plt.scatter([x_acq], [acq_vals[idx_acq]], c='orange', s=80, marker='d', zorder=5, label='Acq точка')
    plt.axvline(bounds[0], color='gray', linestyle=':', alpha=0.6)
    plt.axvline(bounds[1], color='gray', linestyle=':', alpha=0.6)
    plt.title(f"[{combo_name}] Итерация {step}: Функция приобретения")
    plt.xlabel("x")
    plt.ylabel("I(x)")
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"iter_{step:03d}_acquisition.png"), dpi=150)
    plt.close()

def plot_convergence(history, x_true, f_true_min, out_dir, combo_name):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(history)), history, 'b-o', markersize=4, linewidth=1.5, label='f_best по итерациям')
    plt.axhline(f_true_min, color='red', linestyle='--', linewidth=2, label=f'Истинный минимум ({f_true_min:.4f})')
    plt.xlabel("Итерация")
    plt.ylabel("Значение функции")
    plt.title(f"[{combo_name}] Сходимость алгоритма GLIS")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence.png"), dpi=150)
    plt.close()


def plot_basis_values_1d(X, Y, out_dir, combo_name, filename, title_suffix):
    order = np.argsort(X)
    Xs = np.asarray(X)[order]
    Ys = np.asarray(Y)[order]
    plt.figure(figsize=(10, 5))
    plt.plot(Xs, Ys, 'o-', linewidth=1.2, markersize=4)
    plt.xlabel('x базисных точек')
    plt.ylabel('s(x)')
    plt.title(f'[{combo_name}] {title_suffix}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close()

# ============================================================
# 9. ОСНОВНОЙ АЛГОРИТМ (ОДИН ЗАПУСК)
# ============================================================
def run_glis_1d_single(bounds, M, max_iters, patience, poly_deg, lambda_reg,
                       epsilon_rbf, alpha, delta, kernel_name, init_method, seed,
                       out_dir, combo_name, target_func, x_true, f_true_min):
    rng = np.random.default_rng(seed)
    kernel = RBFKernel(kernel_name, epsilon_rbf)

    if init_method == "uniform_grid": x0 = init_uniform_grid(bounds, M)
    elif init_method == "lhs": x0 = init_lhs(bounds, M, rng)
    elif init_method == "uniform_random": x0 = init_uniform_random(bounds, M, rng)
    else: x0 = init_chaos(bounds, M, rng)

    y0 = target_func(x0)
    X = x0.copy()
    Y = y0.copy()

    idx_best = np.argmin(Y)
    x_best, f_best = X[idx_best], Y[idx_best]
    history = [f_best]

    x_grid = np.linspace(bounds[0], bounds[1], 1200)
    f_true_grid = target_func(x_grid)

    # Начальные графики (итерация 0)
    w0, beta0 = fit_rbf_mnk(X, Y, kernel, poly_deg, lambda_reg)
    f_surrogate_grid0 = predict_rbf(x_grid, X, w0, beta0, kernel, poly_deg)
    acq_vals0 = acquisition_function(x_grid, X, Y, w0, beta0, kernel, poly_deg, alpha, delta)
    x_rs0, _ = minimize_1d(lambda x: predict_rbf(x, X, w0, beta0, kernel, poly_deg)[0], bounds)
    f_rs0 = target_func(x_rs0)
    x_acq0, _ = minimize_1d(lambda x: acquisition_function(x, X, Y, w0, beta0, kernel, poly_deg, alpha, delta)[0], bounds)
    f_acq0 = target_func(x_acq0)
    plot_iteration(X, Y, x_rs0, f_rs0, x_acq0, f_acq0, x_best, f_best,
                   x_grid, f_true_grid, f_surrogate_grid0, acq_vals0, bounds, x_true, f_true_min,
                   kernel_name, poly_deg, out_dir, 0, combo_name)
    plot_basis_values_1d(X, Y, out_dir, combo_name, 'initial_basis_values.png', 'Начальные базисные точки')

    no_improve = 0
    start_time = time.time()

    print(f"\n[{combo_name}] Старт: f_best={f_best:.6f}, x_best={x_best:.6f}")

    for step in range(1, max_iters + 1):
        w, beta = fit_rbf_mnk(X, Y, kernel, poly_deg, lambda_reg)
        f_surrogate_grid = predict_rbf(x_grid, X, w, beta, kernel, poly_deg)

        x_rs, _ = minimize_1d(lambda x: predict_rbf(x, X, w, beta, kernel, poly_deg)[0], bounds)
        f_rs = target_func(x_rs)

        acq_vals = acquisition_function(x_grid, X, Y, w, beta, kernel, poly_deg, alpha, delta)
        x_acq, _ = minimize_1d(lambda x: acquisition_function(x, X, Y, w, beta, kernel, poly_deg, alpha, delta)[0], bounds)
        f_acq = target_func(x_acq)

        # Обновление базиса: добавляем 2 кандидата и удаляем 2 наихудшие точки
        prev_best = history[-1]
        X_old = X.copy()
        X_new = np.concatenate([X, [x_rs, x_acq]])
        Y_new = np.concatenate([Y, [f_rs, f_acq]])
        sorted_idx = np.argsort(Y_new)
        keep_idx = sorted_idx[:M]
        removed_idx = sorted_idx[M:]

        X = X_new[keep_idx]
        Y = Y_new[keep_idx]

        idx_best = np.argmin(Y)
        x_best, f_best = X[idx_best], Y[idx_best]
        history.append(f_best)

        removed_pairs = [(float(X_new[i]), float(Y_new[i])) for i in removed_idx]
        improved = f_best < prev_best - 1e-8
        basis_changed = not np.allclose(np.sort(X_old), np.sort(X))

        removed_str = '; '.join([f"{yv:.4f} @ {xv:.4f}" for xv, yv in removed_pairs])
        print(f"[{combo_name}] Итер {step:02d} | RS: {f_rs:.4f} @ {x_rs:.4f} | Acq: {f_acq:.4f} @ {x_acq:.4f} | "
              f"Best: {f_best:.4f} @ {x_best:.4f} | Удалены: {removed_str} | "
              f"Базис {'изменён' if basis_changed else 'не изменён'} | "
              f"{'Улучшение' if improved else 'Без улучшений'}")

        plot_iteration(X, Y, x_rs, f_rs, x_acq, f_acq, x_best, f_best,
                       x_grid, f_true_grid, f_surrogate_grid, acq_vals, bounds, x_true, f_true_min,
                       kernel_name, poly_deg, out_dir, step, combo_name)

        if improved:
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[{combo_name}] Остановка: нет улучшений {patience} итераций подряд.")
            break

    elapsed = time.time() - start_time
    plot_convergence(history, x_true, f_true_min, out_dir, combo_name)
    plot_basis_values_1d(X, Y, out_dir, combo_name, 'final_basis_values.png', 'Финальные базисные точки')

    return {
        'combo': combo_name,
        'init': init_method,
        'kernel': kernel_name,
        'x_found': x_best,
        'f_found': f_best,
        'err_x': abs(x_best - x_true),
        'err_f': abs(f_best - f_true_min),
        'iters': step,
        'time': elapsed
    }

# ============================================================
# 10. ГЛАВНЫЙ КОНТРОЛЛЕР
# ============================================================
def main():
    print("="*80)
    print("GLIS-1D: Полная параметризуемая версия с таблицей сравнения и графиками")
    print("="*80)

    # Параметры целевой функции
    c1 = get_float("Параметр c1 (сдвиг косинуса)", -3.0)
    c2 = get_float("Параметр c2 (сдвиг параболы)", 3.0)
    H = get_float("Параметр H (амплитуда)", 10.0)
    K2 = get_float("Параметр K2 (частота)", 12.0)
    target_func = make_target_function(c1, c2, H, K2)

    bounds = (get_float("Нижняя граница поиска", -10.0), get_float("Верхняя граница поиска", 10.0))

    # Истинный минимум
    print("\n[INFO] Вычисление истинного глобального минимума...")
    x_true, f_true_min = find_true_minimum(target_func, bounds)
    print(f"[INFO] Истинный минимум найден: x* = {x_true:.8f}, f* = {f_true_min:.8f}\n")

    # Режим запуска
    run_mode = get_choice("Режим запуска:", ["одна комбинация", "все комбинации"], "все комбинации")

    # Гиперпараметры алгоритма
    M = get_int("Число базисных точек M", 15)
    max_iters = get_int("Максимум итераций P_max", 30)
    patience = get_int("Терпение (остановка без улучшений)", 10)
    poly_deg = get_int("Степень полиномиального хвоста K (0/1/2)", 1)
    lambda_reg = get_float("Параметр регуляризации lambda", 0.01)
    epsilon_rbf = get_float("Масштаб RBF (epsilon)", 1.0)
    alpha = get_float("Коэффициент alpha (неопределённость)", 4.0)
    delta = get_float("Коэффициент delta (новизна)", 0.15)
    base_seed = get_int("Базовый seed", 42)

    all_inits = ["lhs", "uniform_grid", "uniform_random", "chaos"]
    all_kernels = ["gaussian", "multiquadric", "inverse_quadratic", "thin_plate", "polyharmonic"]

    if run_mode == "одна комбинация":
        init_methods = [get_choice("Метод инициализации:", all_inits, "lhs")]
        kernels = [get_choice("Тип RBF ядра:", all_kernels, "gaussian")]
    else:
        init_methods = all_inits
        kernels = all_kernels
        print(f"\nБудут запущены все комбинации: {len(init_methods)} × {len(kernels)} = {len(init_methods)*len(kernels)} запусков.")

    base_dir = "../GLIS_1D_Results"
    os.makedirs(base_dir, exist_ok=True)

    results = []
    combo_idx = 0

    for init_m in init_methods:
        for ker_n in kernels:
            combo_idx += 1
            seed = base_seed + combo_idx * 17
            combo_name = f"combo_{combo_idx:02d}_{init_m}_{ker_n}"
            out_dir = os.path.join(base_dir, combo_name)

            print(f"\n{'='*80}\nЗАПУСК {combo_idx}/{len(init_methods)*len(kernels)}: {init_m} + {ker_n}\n{'='*80}")
            res = run_glis_1d_single(bounds, M, max_iters, patience, poly_deg, lambda_reg,
                                     epsilon_rbf, alpha, delta, ker_n, init_m, seed,
                                     out_dir, combo_name, target_func, x_true, f_true_min)
            results.append(res)

    # Сводная таблица в консоли
    print("\n" + "="*110)
    print("СВОДНАЯ ТАБЛИЦА СРАВНЕНИЯ МЕТОДОВ")
    print("="*110)
    header = f"{'Combo':<12} | {'Init':<14} | {'Kernel':<18} | {'x_found':<10} | {'f_found':<10} | {'err_x':<10} | {'err_f':<10} | {'Iters':<5} | {'Time(s)':<7}"
    print(header)
    print("-" * 110)
    for r in results:
        print(f"{r['combo']:<12} | {r['init']:<14} | {r['kernel']:<18} | {r['x_found']:<10.6f} | {r['f_found']:<10.6f} | {r['err_x']:<10.6f} | {r['err_f']:<10.6f} | {r['iters']:<5} | {r['time']:<7.2f}")
    print("="*110)

    # Лучший результат
    best = min(results, key=lambda k: k['err_f'])
    print(f"\n🏆 ЛУЧШАЯ КОМБИНАЦИЯ: {best['init']} + {best['kernel']}")
    print(f"   x_found = {best['x_found']:.8f} (ошибка: {best['err_x']:.2e})")
    print(f"   f_found = {best['f_found']:.8f} (ошибка: {best['err_f']:.2e})")
    print(f"   Итераций: {best['iters']}, Время: {best['time']:.2f} сек")
    print(f"   Истинный минимум: x* = {x_true:.8f}, f* = {f_true_min:.8f}")

    # Сохранение CSV
    csv_path = os.path.join(base_dir, "summary_table.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n📊 Сводная таблица сохранена в: {os.path.abspath(csv_path)}")
    print(f"📁 Все графики и отчёты сохранены в папку: {os.path.abspath(base_dir)}")
    print("="*80)

if __name__ == "__main__":
    main()
