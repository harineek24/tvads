"""
Conformal Prediction — Distribution-Free Uncertainty Quantification
===================================================================
Wraps the MMM and baseline models with conformal prediction intervals
that have GUARANTEED finite-sample coverage — no distributional assumptions.

WHY this matters for recruiters:
- Standard ML gives point estimates with no reliability guarantees
- Bayesian CI depends on correct prior specification
- Conformal prediction provides valid intervals under ONLY exchangeability
- Based on Vovk et al. (2005), popularized by Angelopoulos & Bates (2021)
- Extremely hot topic in ML research — barely anyone applies it in marketing

Methods implemented:
1. Split Conformal — simple calibration on held-out data
2. Conformalized Quantile Regression (CQR) — tighter, adaptive intervals
3. Jackknife+ — leave-one-out with theoretical guarantees
4. Coverage analysis — empirical vs nominal coverage

Outputs:
- data/conformal_intervals.csv
- data/conformal_coverage.csv
- results/conformal_intervals.png
- results/conformal_coverage.png
- results/conformal_width_analysis.png
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from mapie.regression import SplitConformalRegressor, CrossConformalRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

SPEND_COLUMNS = [
    'tv_broadcast_spend', 'tv_cable_spend', 'tv_streaming_spend',
    'paid_search_spend', 'social_spend', 'display_spend'
]


def prepare_features(weekly):
    """Build feature matrix for MMM-like model."""
    features = {}
    for col in SPEND_COLUMNS:
        features[col] = weekly[col].values

    features['month_sin'] = np.sin(2 * np.pi * weekly['month'].values / 12)
    features['month_cos'] = np.cos(2 * np.pi * weekly['month'].values / 12)
    features['trend'] = np.arange(len(weekly)) / len(weekly)

    X = pd.DataFrame(features).values
    y = weekly['total_revenue'].values
    return X, y


def run_split_conformal(X, y, alpha=0.1):
    """
    Split Conformal: Simplest method.
    Split data → train model → calibrate residuals → predict with intervals.
    """
    print(f"\n=== Split Conformal (α={alpha}, target {1-alpha:.0%} coverage) ===")

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )

    # Split: prefit model, then conformalize on calibration data
    n = len(X)
    cal_size = int(0.3 * n)
    idx = np.random.permutation(n)
    train_idx, cal_idx = idx[cal_size:], idx[:cal_size]
    model.fit(X[train_idx], y[train_idx])

    mapie = SplitConformalRegressor(estimator=model, confidence_level=1-alpha, prefit=True)
    mapie.conformalize(X[cal_idx], y[cal_idx])

    predictions, pi = mapie.predict_interval(X)
    lower = pi[:, 0, 0]
    upper = pi[:, 1, 0]

    coverage = np.mean((y >= lower) & (y <= upper))
    avg_width = np.mean(upper - lower)

    print(f"  Coverage: {coverage:.1%} (target: {1-alpha:.0%})")
    print(f"  Avg interval width: ${avg_width:,.0f}")
    print(f"  R²: {r2_score(y, predictions):.4f}")

    intervals = np.stack([lower, upper], axis=1)
    return predictions, intervals, coverage, avg_width


def run_jackknife_plus(X, y, alpha=0.1):
    """
    Jackknife+ (via Cross-Conformal): LOO-style with theoretical coverage guarantees.
    Provides tighter intervals than naive conformal.
    """
    print(f"\n=== Cross-Conformal / Jackknife+ (α={alpha}, target {1-alpha:.0%} coverage) ===")

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )

    mapie = CrossConformalRegressor(estimator=model, confidence_level=1-alpha, cv=5, random_state=42)
    mapie.fit_conformalize(X, y)

    predictions, pi = mapie.predict_interval(X)
    lower = pi[:, 0, 0]
    upper = pi[:, 1, 0]

    coverage = np.mean((y >= lower) & (y <= upper))
    avg_width = np.mean(upper - lower)

    print(f"  Coverage: {coverage:.1%} (target: {1-alpha:.0%})")
    print(f"  Avg interval width: ${avg_width:,.0f}")

    intervals = np.stack([lower, upper], axis=1)
    return predictions, intervals, coverage, avg_width


def run_cqr(X, y, alpha=0.1):
    """
    Conformalized Quantile Regression (CQR):
    Combines quantile regression with conformal calibration.
    Produces ADAPTIVE intervals — wider when uncertain, tighter when confident.
    """
    print(f"\n=== Conformalized Quantile Regression (CQR, α={alpha}) ===")

    # Use GBR with quantile loss for lower/upper bounds
    model_lower = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        loss='quantile', alpha=alpha/2, random_state=42
    )
    model_upper = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        loss='quantile', alpha=1-alpha/2, random_state=42
    )

    # Split for calibration
    n = len(X)
    n_train = int(0.7 * n)
    idx = np.random.permutation(n)
    train_idx, cal_idx = idx[:n_train], idx[n_train:]

    model_lower.fit(X[train_idx], y[train_idx])
    model_upper.fit(X[train_idx], y[train_idx])

    # Calibration scores
    cal_lower = model_lower.predict(X[cal_idx])
    cal_upper = model_upper.predict(X[cal_idx])
    cal_scores = np.maximum(cal_lower - y[cal_idx], y[cal_idx] - cal_upper)

    # Quantile of calibration scores
    q = np.quantile(cal_scores, 1 - alpha, method='higher')

    # Predict on all data
    pred_lower = model_lower.predict(X) - q
    pred_upper = model_upper.predict(X) + q
    pred_median = (model_lower.predict(X) + model_upper.predict(X)) / 2

    coverage = np.mean((y >= pred_lower) & (y <= pred_upper))
    avg_width = np.mean(pred_upper - pred_lower)
    width_std = np.std(pred_upper - pred_lower)

    print(f"  Coverage: {coverage:.1%} (target: {1-alpha:.0%})")
    print(f"  Avg interval width: ${avg_width:,.0f} (±${width_std:,.0f})")
    print(f"  Width adaptivity ratio: {width_std/avg_width:.2f} "
          f"(higher = more adaptive)")

    intervals = np.stack([pred_lower, pred_upper], axis=1)
    return pred_median, intervals, coverage, avg_width


def coverage_analysis(X, y, methods_results):
    """Analyze coverage across different confidence levels."""
    print("\n=== Coverage vs Nominal Level Analysis ===")

    alphas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    coverage_data = []

    for alpha in alphas:
        try:
            model = GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
            )
            mapie = CrossConformalRegressor(estimator=model, confidence_level=1-alpha,
                                            cv=5, random_state=42)
            mapie.fit_conformalize(X, y)
            _, pi = mapie.predict_interval(X)
            lower = pi[:, 0, 0]
            upper = pi[:, 1, 0]

            actual_coverage = np.mean((y >= lower) & (y <= upper))
            avg_width = np.mean(upper - lower)

            coverage_data.append({
                'nominal_coverage': 1 - alpha,
                'actual_coverage': actual_coverage,
                'avg_width': avg_width,
                'gap': actual_coverage - (1 - alpha),
            })
        except Exception:
            pass

    return pd.DataFrame(coverage_data)


def plot_results(weekly, split_results, jp_results, cqr_results, coverage_df):
    """Generate visualization plots."""
    os.makedirs('results', exist_ok=True)

    _, y = prepare_features(weekly)
    weeks = np.arange(1, len(y) + 1)

    # 1. Comparison of interval methods
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    methods = [
        ('Split Conformal', split_results, '#3498db'),
        ('Jackknife+', jp_results, '#27ae60'),
        ('CQR (Adaptive)', cqr_results, '#e74c3c'),
    ]

    for i, (name, (preds, intervals, cov, width), color) in enumerate(methods):
        ax = axes[i]
        ax.plot(weeks, y / 1e6, 'o-', color='#2c3e50', markersize=3,
                linewidth=1, label='Actual')

        if intervals.ndim == 3:
            lower, upper = intervals[:, 0, 0], intervals[:, 1, 0]
        else:
            lower, upper = intervals[:, 0], intervals[:, 1]

        ax.fill_between(weeks, lower / 1e6, upper / 1e6,
                        alpha=0.25, color=color, label=f'90% PI (coverage: {cov:.0%})')
        ax.plot(weeks, preds / 1e6, '--', color=color, linewidth=1, label='Prediction')
        ax.set_ylabel('Revenue ($M)')
        ax.set_title(f'{name} — Coverage: {cov:.0%}, Avg Width: ${width/1e3:,.0f}K')
        ax.legend(fontsize=8, loc='upper left')

    axes[-1].set_xlabel('Week')
    plt.suptitle('Conformal Prediction: Distribution-Free Uncertainty Quantification',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('results/conformal_intervals.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Coverage calibration plot
    if len(coverage_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
        ax.plot(coverage_df['nominal_coverage'], coverage_df['actual_coverage'],
                'o-', color='#8e44ad', markersize=8, linewidth=2, label='Jackknife+')
        ax.fill_between(coverage_df['nominal_coverage'],
                        coverage_df['nominal_coverage'] - 0.05,
                        coverage_df['nominal_coverage'] + 0.05,
                        alpha=0.1, color='gray', label='±5% tolerance')
        ax.set_xlabel('Nominal Coverage')
        ax.set_ylabel('Actual Coverage')
        ax.set_title('Coverage Calibration\n(closer to diagonal = better)')
        ax.legend(fontsize=8)
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.4, 1.0)

        ax = axes[1]
        ax.plot(coverage_df['nominal_coverage'], coverage_df['avg_width'] / 1e3,
                's-', color='#e67e22', markersize=8, linewidth=2)
        ax.set_xlabel('Nominal Coverage')
        ax.set_ylabel('Avg Interval Width ($K)')
        ax.set_title('Interval Width vs Coverage Level\n(trade-off: tighter intervals = lower coverage)')

        plt.tight_layout()
        plt.savefig('results/conformal_coverage.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 3. CQR adaptive width analysis
    fig, ax = plt.subplots(figsize=(12, 4))
    _, cqr_intervals, _, _ = cqr_results
    if cqr_intervals.ndim == 3:
        widths = (cqr_intervals[:, 1, 0] - cqr_intervals[:, 0, 0]) / 1e3
    else:
        widths = (cqr_intervals[:, 1] - cqr_intervals[:, 0]) / 1e3

    colors = plt.cm.RdYlGn_r(widths / widths.max())
    ax.bar(weeks, widths, color=colors, alpha=0.8)
    ax.set_xlabel('Week')
    ax.set_ylabel('Interval Width ($K)')
    ax.set_title('CQR: Adaptive Interval Widths\n'
                 '(Wider when model is uncertain, tighter when confident)')
    ax.axhline(y=np.mean(widths), color='gray', linestyle='--',
               label=f'Mean: ${np.mean(widths):,.0f}K')
    ax.legend()
    plt.tight_layout()
    plt.savefig('results/conformal_width_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Plots saved to results/conformal_*.png")


def main():
    print("=" * 60)
    print("CONFORMAL PREDICTION — Distribution-Free UQ")
    print("=" * 60)

    weekly = pd.read_csv('data/weekly_spend.csv')
    X, y = prepare_features(weekly)

    # 1. Split Conformal
    split_results = run_split_conformal(X, y, alpha=0.1)

    # 2. Jackknife+
    jp_results = run_jackknife_plus(X, y, alpha=0.1)

    # 3. CQR (Adaptive)
    cqr_results = run_cqr(X, y, alpha=0.1)

    # 4. Coverage analysis
    coverage_df = coverage_analysis(X, y, None)

    # Save results
    os.makedirs('data', exist_ok=True)

    # CQR intervals for each week
    _, cqr_intervals, cqr_cov, cqr_width = cqr_results
    if cqr_intervals.ndim == 3:
        lower, upper = cqr_intervals[:, 0, 0], cqr_intervals[:, 1, 0]
    else:
        lower, upper = cqr_intervals[:, 0], cqr_intervals[:, 1]

    intervals_df = pd.DataFrame({
        'week': range(1, len(y) + 1),
        'actual': y,
        'cqr_lower': lower,
        'cqr_upper': upper,
        'cqr_width': upper - lower,
        'covered': ((y >= lower) & (y <= upper)).astype(int),
    })
    intervals_df.to_csv('data/conformal_intervals.csv', index=False)
    coverage_df.to_csv('data/conformal_coverage.csv', index=False)

    # Summary metrics
    summary = {
        'split_conformal': {'coverage': float(split_results[2]),
                            'avg_width': float(split_results[3])},
        'jackknife_plus': {'coverage': float(jp_results[2]),
                           'avg_width': float(jp_results[3])},
        'cqr': {'coverage': float(cqr_results[2]),
                'avg_width': float(cqr_results[3])},
    }
    with open('data/conformal_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot
    plot_results(weekly, split_results, jp_results, cqr_results, coverage_df)

    print("\n✓ Conformal prediction analysis complete")
    return intervals_df, coverage_df, summary


if __name__ == '__main__':
    main()
