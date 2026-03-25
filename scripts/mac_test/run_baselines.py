#!/usr/bin/env python3
"""
Baseline comparison script for ETTm1 (or coal as fallback).

Implements three baselines:
  1. Linear   - sklearn LinearRegression on all input features
  2. NLinear  - normalize input before linear regression (mean subtraction)
  3. DLinear  - detrend input (last-value subtraction) before linear regression

These approximate the LTSF-Linear family of baselines described in:
  "Are Transformers Effective for Time Series Forecasting?" (Zeng et al. 2023)

Usage:
    python3 scripts/mac_test/run_baselines.py
    python3 scripts/mac_test/run_baselines.py --dataset coal
    python3 scripts/mac_test/run_baselines.py --lookback 96 --horizon 1
"""

import os
import sys
import csv
import argparse
import math

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Run linear baselines on time series data.")
    parser.add_argument("--dataset", choices=["ettm1", "coal"], default="ettm1",
                        help="Dataset to use (default: ettm1)")
    parser.add_argument("--lookback", type=int, default=96,
                        help="Input window length (default: 96)")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Prediction horizon in steps (default: 1)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for results.csv")
    return parser.parse_args()


def load_csv(filepath):
    """Load a CSV file. Returns (headers, data) where data is list of list of float."""
    rows = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if row:
                rows.append([float(v) for v in row])
    return headers, rows


def get_ettm1_paths():
    base = os.path.join(REPO_ROOT, "datasets", "benchmarks", "ETT-small")
    train = os.path.join(base, "ETTm1_train_raw.csv")
    val   = os.path.join(base, "ETTm1_val_raw.csv")
    test  = os.path.join(base, "ETTm1_test_raw.csv")
    return train, val, test, ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"], "OT"


def get_coal_paths():
    base = os.path.join(REPO_ROOT, "datasets", "2018_coal")
    train_files = [os.path.join(base, f"burner_{i}.csv") for i in range(10)]
    val_files   = [os.path.join(base, f"burner_{i}.csv") for i in [10, 11]]
    # Coal is pre-normalized; all columns used
    sample_headers, _ = load_csv(train_files[0])
    output_col = "Main_Flm_Int"
    return train_files, val_files, None, sample_headers, output_col


def load_multifile_data(file_list, headers, target_col, lookback, horizon):
    """Load multiple files and build (X, y) pairs using sliding window."""
    target_idx = headers.index(target_col)
    X_all, y_all = [], []
    for fpath in file_list:
        _, data = load_csv(fpath)
        # data is list of rows; each row = all feature values
        n = len(data)
        for i in range(n - lookback - horizon + 1):
            window = data[i : i + lookback]
            # Flatten all features in the window as input
            x_flat = [v for row in window for v in row]
            target_future = data[i + lookback + horizon - 1][target_idx]
            X_all.append(x_flat)
            y_all.append(target_future)
    return X_all, y_all


def load_singlefile_data(filepath, headers, target_col, lookback, horizon):
    return load_multifile_data([filepath], headers, target_col, lookback, horizon)


# ── Minimal linear algebra helpers (no numpy/sklearn dependency) ──────────────

def mat_mul(A, B):
    """Multiply matrix A (m×n) by B (n×p). Lists of lists."""
    m, n = len(A), len(A[0])
    p = len(B[0])
    C = [[0.0]*p for _ in range(m)]
    for i in range(m):
        for k in range(n):
            if A[i][k] == 0.0:
                continue
            for j in range(p):
                C[i][j] += A[i][k] * B[k][j]
    return C


def transpose(A):
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def add_bias_col(X):
    """Add a column of 1s for bias term."""
    return [row + [1.0] for row in X]


def solve_lstsq(X, y):
    """
    Solve min||Xw - y||^2 using normal equations: w = (X^T X)^{-1} X^T y
    Uses simple Gauss-Jordan for small systems. For large X, uses gradient descent fallback.
    """
    n_features = len(X[0])
    n_samples = len(X)
    
    # For large feature sets, use gradient descent (avoids huge matrix inversion)
    if n_features > 500 or n_samples > 50000:
        return solve_gd(X, y, n_features)
    
    # X^T X
    Xt = transpose(X)
    XtX = mat_mul(Xt, [[v] for v in y])  # wait, need X^T y
    # X^T y
    Xty = [sum(Xt[i][j] * y[j] for j in range(n_samples)) for i in range(n_features)]
    # X^T X
    XtX_mat = mat_mul(Xt, X)
    
    # Solve XtX w = Xty via Gauss-Jordan
    # Augment matrix
    n = n_features
    aug = [XtX_mat[i][:] + [Xty[i]] for i in range(n)]
    
    for col in range(n):
        # Find pivot
        pivot = None
        for row in range(col, n):
            if abs(aug[row][col]) > 1e-12:
                pivot = row
                break
        if pivot is None:
            # Singular; add tiny regularization and retry
            for i in range(n):
                aug[i][i] += 1e-6
            for row in range(col, n):
                if abs(aug[row][col]) > 1e-12:
                    pivot = row
                    break
            if pivot is None:
                continue
        
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        
        for row in range(n):
            if row != col and abs(aug[row][col]) > 1e-15:
                factor = aug[row][col]
                aug[row] = [aug[row][j] - factor * aug[col][j] for j in range(n + 1)]
    
    return [aug[i][n] for i in range(n)]


def solve_gd(X, y, n_features, lr=0.01, epochs=200):
    """Gradient descent fallback for large feature sets."""
    w = [0.0] * n_features
    n = len(X)
    for _ in range(epochs):
        grad = [0.0] * n_features
        for i in range(n):
            pred = sum(X[i][j] * w[j] for j in range(n_features))
            err = pred - y[i]
            for j in range(n_features):
                grad[j] += (2.0 / n) * err * X[i][j]
        # gradient clip
        gnorm = math.sqrt(sum(g*g for g in grad))
        if gnorm > 1.0:
            grad = [g / gnorm for g in grad]
        w = [w[j] - lr * grad[j] for j in range(n_features)]
    return w


def predict(X, w):
    return [sum(X[i][j] * w[j] for j in range(len(w))) for i in range(len(X))]


def mse(y_true, y_pred):
    n = len(y_true)
    return sum((y_true[i] - y_pred[i])**2 for i in range(n)) / n


def mae(y_true, y_pred):
    n = len(y_true)
    return sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n


# ── Baseline implementations ──────────────────────────────────────────────────

def run_linear(X_train, y_train, X_test, y_test):
    """Standard linear regression."""
    X_b = add_bias_col(X_train)
    X_bt = add_bias_col(X_test)
    w = solve_lstsq(X_b, y_train)
    preds = predict(X_bt, w)
    return mse(y_test, preds), mae(y_test, preds)


def run_nlinear(X_train, y_train, X_test, y_test):
    """NLinear: subtract the last value of the input window before linear regression."""
    # last value = last feature of each sample (last timestep, first/output feature)
    # We approximate: last_val = last element of each flattened row divided by n_features
    # More precisely: subtract the last timestep's output feature value
    # Since we don't track feature layout here, use row mean as normalization proxy
    def normalize(X):
        X_n = []
        last_vals = []
        for row in X:
            lv = row[-1]  # last element of the flattened window
            last_vals.append(lv)
            X_n.append([v - lv for v in row])
        return X_n, last_vals
    
    X_tr_n, lv_tr = normalize(X_train)
    X_te_n, lv_te = normalize(X_test)
    
    # Adjust targets by subtracting last value
    y_tr_n = [y_train[i] - lv_tr[i] for i in range(len(y_train))]
    
    X_b = add_bias_col(X_tr_n)
    X_bt = add_bias_col(X_te_n)
    w = solve_lstsq(X_b, y_tr_n)
    preds_n = predict(X_bt, w)
    # Add back last value
    preds = [preds_n[i] + lv_te[i] for i in range(len(preds_n))]
    return mse(y_test, preds), mae(y_test, preds)


def run_dlinear(X_train, y_train, X_test, y_test):
    """DLinear: subtract a simple linear trend from the window before regression."""
    def detrend(X):
        X_d = []
        trends = []
        for row in X:
            n = len(row)
            # Compute trend as linear interpolation from first to last value
            start, end = row[0], row[-1]
            trend = [start + (end - start) * (i / (n - 1)) if n > 1 else start for i in range(n)]
            residual = [row[i] - trend[i] for i in range(n)]
            X_d.append(residual)
            trends.append(trend)
        return X_d, trends
    
    X_tr_d, tr_tr = detrend(X_train)
    X_te_d, tr_te = detrend(X_test)
    
    # Target: subtract trend's last value (extrapolated)
    def extrapolate_trend(trend):
        n = len(trend)
        if n < 2:
            return trend[-1]
        slope = (trend[-1] - trend[0]) / (n - 1)
        return trend[-1] + slope  # one step ahead
    
    y_tr_d = [y_train[i] - extrapolate_trend(tr_tr[i]) for i in range(len(y_train))]
    
    X_b = add_bias_col(X_tr_d)
    X_bt = add_bias_col(X_te_d)
    w = solve_lstsq(X_b, y_tr_d)
    preds_d = predict(X_bt, w)
    preds = [preds_d[i] + extrapolate_trend(tr_te[i]) for i in range(len(preds_d))]
    return mse(y_test, preds), mae(y_test, preds)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    lookback = args.lookback
    horizon  = args.horizon
    
    output_dir = args.output_dir or os.path.join(REPO_ROOT, "test_output", "baselines")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Dataset: {args.dataset}")
    print(f"Lookback: {lookback}, Horizon: {horizon}")
    print(f"Output: {output_dir}")
    print()
    
    # ── Load data ──────────────────────────────────────────────────────────────
    if args.dataset == "ettm1":
        train_path, val_path, test_path, headers, target_col = get_ettm1_paths()
        print(f"Loading ETTm1 training data: {train_path}")
        X_train, y_train = load_singlefile_data(train_path, headers, target_col, lookback, horizon)
        print(f"Loading ETTm1 validation data: {val_path}")
        X_test, y_test = load_singlefile_data(val_path, headers, target_col, lookback, horizon)
    else:
        train_files, val_files, _, headers, target_col = get_coal_paths()
        print(f"Loading coal training data ({len(train_files)} files)...")
        X_train, y_train = load_multifile_data(train_files, headers, target_col, lookback, horizon)
        print(f"Loading coal validation data ({len(val_files)} files)...")
        X_test, y_test = load_multifile_data(val_files, headers, target_col, lookback, horizon)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    print(f"Feature dim:      {len(X_train[0]) if X_train else 0}")
    print()
    
    if not X_train or not X_test:
        print("ERROR: No data loaded. Check paths and lookback/horizon settings.")
        sys.exit(1)
    
    # ── Run baselines ──────────────────────────────────────────────────────────
    results = []
    
    print("Running Linear baseline...")
    lin_mse, lin_mae = run_linear(X_train, y_train, X_test, y_test)
    print(f"  MSE={lin_mse:.6f}  MAE={lin_mae:.6f}")
    results.append(("Linear", lin_mse, lin_mae))
    
    print("Running NLinear baseline...")
    nlin_mse, nlin_mae = run_nlinear(X_train, y_train, X_test, y_test)
    print(f"  MSE={nlin_mse:.6f}  MAE={nlin_mae:.6f}")
    results.append(("NLinear", nlin_mse, nlin_mae))
    
    print("Running DLinear baseline...")
    dlin_mse, dlin_mae = run_dlinear(X_train, y_train, X_test, y_test)
    print(f"  MSE={dlin_mse:.6f}  MAE={dlin_mae:.6f}")
    results.append(("DLinear", dlin_mse, dlin_mae))
    
    # ── Save results ───────────────────────────────────────────────────────────
    out_path = os.path.join(output_dir, "results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "dataset", "lookback", "horizon", "mse", "mae"])
        for name, m, a in results:
            writer.writerow([name, args.dataset, lookback, horizon, f"{m:.8f}", f"{a:.8f}"])
    
    print()
    print(f"Results saved to: {out_path}")
    print()
    print("Summary:")
    print(f"{'Model':<12} {'MSE':>12} {'MAE':>12}")
    print("-" * 38)
    for name, m, a in results:
        print(f"{name:<12} {m:>12.6f} {a:>12.6f}")


if __name__ == "__main__":
    main()
