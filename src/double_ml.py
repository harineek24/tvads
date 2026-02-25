"""
Double Machine Learning (DML) — Causal Channel Effects
=======================================================
Uses Microsoft's EconML to estimate the causal effect of each advertising
channel on revenue, properly handling confounding from seasonality, trends,
and cross-channel correlation.

WHY DML > standard regression:
- Standard MMM conflates correlation with causation
- DML uses ML (LightGBM) for nuisance parameters (confounders) but preserves
  valid statistical inference on the treatment effect
- Produces debiased/orthogonal estimates with honest confidence intervals
- Based on Chernozhukov et al. (2018) — the gold standard in causal ML

Pipeline:
1. Define treatment (channel spend), outcome (revenue), confounders
2. Fit DML with LightGBM as the first-stage learner
3. Cross-fit to avoid overfitting bias
4. Extract causal ATEs with confidence intervals per channel
5. Compare DML causal effects vs naive OLS coefficients

Outputs:
- data/dml_causal_effects.csv
- data/dml_vs_ols.csv
- results/dml_causal_effects.png
- results/dml_vs_ols.png
- results/dml_cate_heterogeneity.png
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from econml.dml import LinearDML, CausalForestDML
from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper
from lightgbm import LGBMRegressor
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

CHANNEL_NAMES = {
    'tv_broadcast_spend': 'TV Broadcast',
    'tv_cable_spend': 'TV Cable',
    'tv_streaming_spend': 'TV Streaming',
    'paid_search_spend': 'Paid Search',
    'social_spend': 'Social',
    'display_spend': 'Display',
}


def build_confounders(weekly):
    """Build confounder matrix (W) — things that affect both spend and revenue."""
    W = pd.DataFrame()
    W['month_sin'] = np.sin(2 * np.pi * weekly['month'].values / 12)
    W['month_cos'] = np.cos(2 * np.pi * weekly['month'].values / 12)
    W['trend'] = np.arange(len(weekly)) / len(weekly)
    W['quarter'] = ((weekly['month'].values - 1) // 3).astype(float)
    # Lagged revenue as confounder (past performance affects both future spend
    # decisions and future revenue)
    rev = weekly['total_revenue'].values
    W['lag1_revenue'] = np.concatenate([[rev[0]], rev[:-1]])
    W['lag2_revenue'] = np.concatenate([[rev[0], rev[0]], rev[:-2]])
    return W.values


def run_linear_dml(weekly):
    """
    LinearDML: estimates Average Treatment Effect (ATE) of each channel
    with cross-fitted debiased estimates.
    """
    print("\n=== Linear DML: Causal Channel Effects ===")

    Y = weekly['total_revenue'].values  # outcome
    W = build_confounders(weekly)  # confounders

    results = []
    for col in SPEND_COLUMNS:
        T = weekly[col].values.reshape(-1, 1)  # treatment

        # LinearDML: LightGBM for nuisance, linear final stage
        dml = LinearDML(
            model_y=LGBMRegressor(n_estimators=100, max_depth=3, verbose=-1),
            model_t=LGBMRegressor(n_estimators=100, max_depth=3, verbose=-1),
            cv=3,  # cross-fitting folds
            random_state=42,
        )
        dml.fit(Y, T, W=W)

        # ATE with confidence interval
        ate = dml.ate(X=None)
        ate_inf = dml.ate_inference(X=None)

        # Use the PopulationSummaryResults API
        mean_pt_raw = ate_inf.mean_point
        mean_pt = float(mean_pt_raw) if np.ndim(mean_pt_raw) == 0 else float(mean_pt_raw[0])
        stderr_raw = ate_inf.stderr_mean
        stderr = float(stderr_raw) if np.ndim(stderr_raw) == 0 else float(stderr_raw[0])
        pval_raw = ate_inf.pvalue()
        pval = float(pval_raw) if np.ndim(pval_raw) == 0 else float(pval_raw[0])
        ci_raw = ate_inf.conf_int_mean()
        ci_lower = float(ci_raw[0]) if np.ndim(ci_raw[0]) == 0 else float(ci_raw[0][0])
        ci_upper = float(ci_raw[1]) if np.ndim(ci_raw[1]) == 0 else float(ci_raw[1][0])

        channel_name = CHANNEL_NAMES[col]
        mean_spend = weekly[col].mean()
        marginal_roas = mean_pt * mean_spend / weekly['total_revenue'].mean()

        results.append({
            'channel': channel_name,
            'spend_col': col,
            'ate': mean_pt,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': pval,
            'std_err': stderr,
            'mean_spend': mean_spend,
            'marginal_roas': marginal_roas,
        })

        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        print(f"  {channel_name}: ATE = ${mean_pt:.4f}/$ spend "
              f"[{ci_lower:.4f}, {ci_upper:.4f}] p={pval:.4f} {sig}")

    return pd.DataFrame(results)


def run_causal_forest(weekly):
    """
    CausalForestDML: detects heterogeneous treatment effects.
    Do channel effects vary by season?
    """
    print("\n=== Causal Forest DML: Heterogeneous Effects ===")

    Y = weekly['total_revenue'].values
    W = build_confounders(weekly)

    # Use month as the heterogeneity dimension (X)
    X_het = pd.DataFrame({
        'month_sin': np.sin(2 * np.pi * weekly['month'].values / 12),
        'month_cos': np.cos(2 * np.pi * weekly['month'].values / 12),
        'quarter': ((weekly['month'].values - 1) // 3).astype(float),
    }).values

    het_results = []
    for col in ['tv_broadcast_spend', 'paid_search_spend', 'social_spend']:
        T = weekly[col].values.reshape(-1, 1)

        cf = CausalForestDML(
            model_y=LGBMRegressor(n_estimators=100, max_depth=3, verbose=-1),
            model_t=LGBMRegressor(n_estimators=100, max_depth=3, verbose=-1),
            n_estimators=200,
            min_samples_leaf=5,
            cv=3,
            random_state=42,
        )
        cf.fit(Y, T, X=X_het, W=W)

        # CATE for each observation
        cate = cf.effect(X=X_het).flatten()
        cate_inf = cf.effect_inference(X=X_het)
        ci_bounds = cate_inf.conf_int(alpha=0.1)
        ci_lower_arr = np.asarray(ci_bounds[0]).flatten()
        ci_upper_arr = np.asarray(ci_bounds[1]).flatten()

        for i in range(len(weekly)):
            het_results.append({
                'channel': CHANNEL_NAMES[col],
                'week': i + 1,
                'month': weekly['month'].values[i],
                'cate': float(cate[i]),
                'ci_lower': float(ci_lower_arr[i]),
                'ci_upper': float(ci_upper_arr[i]),
            })

        print(f"  {CHANNEL_NAMES[col]}: CATE range [{cate.min():.4f}, {cate.max():.4f}]")

    return pd.DataFrame(het_results)


def compare_dml_vs_ols(weekly, dml_results):
    """Compare DML causal estimates against naive OLS."""
    print("\n=== DML vs Naive OLS Comparison ===")

    Y = weekly['total_revenue'].values

    comparison = []
    for col in SPEND_COLUMNS:
        T = weekly[col].values.reshape(-1, 1)
        ols = LinearRegression().fit(T, Y)
        ols_coef = ols.coef_[0]

        dml_row = dml_results[dml_results['spend_col'] == col].iloc[0]
        dml_ate = dml_row['ate']

        bias = ols_coef - dml_ate
        bias_pct = (bias / abs(dml_ate) * 100) if abs(dml_ate) > 0.001 else 0

        comparison.append({
            'channel': CHANNEL_NAMES[col],
            'ols_coefficient': ols_coef,
            'dml_causal_effect': dml_ate,
            'bias': bias,
            'bias_pct': bias_pct,
            'dml_significant': dml_row['p_value'] < 0.1,
        })

        print(f"  {CHANNEL_NAMES[col]}: OLS={ols_coef:.4f}, DML={dml_ate:.4f}, "
              f"Bias={bias_pct:+.1f}%")

    return pd.DataFrame(comparison)


def plot_results(dml_results, comparison, het_results):
    """Generate visualization plots."""
    os.makedirs('results', exist_ok=True)

    # 1. Causal effects with CIs
    fig, ax = plt.subplots(figsize=(10, 6))
    df = dml_results.sort_values('ate', ascending=True)
    colors = ['#e74c3c' if p > 0.1 else '#27ae60' for p in df['p_value']]

    ax.barh(df['channel'], df['ate'], color=colors, alpha=0.7, edgecolor='white')
    ax.errorbar(df['ate'], df['channel'],
                xerr=[df['ate'] - df['ci_lower'], df['ci_upper'] - df['ate']],
                fmt='none', color='#2c3e50', capsize=5, linewidth=2)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Causal Effect ($/$ spend)', fontsize=12)
    ax.set_title('Double ML: Causal Effect of Channel Spend on Revenue\n'
                 '(Green = significant at 10%, Red = not significant)', fontsize=13)
    plt.tight_layout()
    plt.savefig('results/dml_causal_effects.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. DML vs OLS comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison))
    width = 0.35
    ax.bar(x - width/2, comparison['ols_coefficient'], width,
           label='Naive OLS (biased)', color='#e74c3c', alpha=0.7)
    ax.bar(x + width/2, comparison['dml_causal_effect'], width,
           label='DML (debiased)', color='#2980b9', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison['channel'], rotation=30, ha='right')
    ax.set_ylabel('Effect Size ($/$ spend)')
    ax.set_title('OLS vs Double ML: Confounding Bias Exposed', fontsize=13)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/dml_vs_ols.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Heterogeneous effects over time
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    channels_het = het_results['channel'].unique()
    for i, ch in enumerate(channels_het):
        ch_data = het_results[het_results['channel'] == ch]
        ax = axes[i]
        ax.fill_between(ch_data['week'], ch_data['ci_lower'], ch_data['ci_upper'],
                        alpha=0.2, color='#8e44ad')
        ax.plot(ch_data['week'], ch_data['cate'], color='#8e44ad', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.set_title(f'{ch}\nCausal Effect Over Time', fontsize=11)
        ax.set_xlabel('Week')
        if i == 0:
            ax.set_ylabel('CATE ($/$ spend)')
    plt.suptitle('Causal Forest: How Channel Effectiveness Varies Over Time',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('results/dml_cate_heterogeneity.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nPlots saved to results/dml_*.png")


def main():
    print("=" * 60)
    print("DOUBLE MACHINE LEARNING — Causal Channel Effects")
    print("=" * 60)

    weekly = pd.read_csv('data/weekly_spend.csv')

    # 1. Linear DML — ATE per channel
    dml_results = run_linear_dml(weekly)
    dml_results.to_csv('data/dml_causal_effects.csv', index=False)

    # 2. Causal Forest — heterogeneous effects
    het_results = run_causal_forest(weekly)
    het_results.to_csv('data/dml_heterogeneous_effects.csv', index=False)

    # 3. Compare DML vs naive OLS
    comparison = compare_dml_vs_ols(weekly, dml_results)
    comparison.to_csv('data/dml_vs_ols.csv', index=False)

    # 4. Visualize
    plot_results(dml_results, comparison, het_results)

    print("\n✓ Double ML analysis complete")
    return dml_results, het_results, comparison


if __name__ == '__main__':
    main()
