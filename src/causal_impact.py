"""
CausalImpact — Bayesian Structural Time Series (Manual Implementation)
=======================================================================
Measures the causal impact of a campaign event (TV spend change)
using a BSTS-style counterfactual approach.

WHY this matters:
- Google's standard tool for measuring campaign effectiveness
- Uses structural time series to build a counterfactual
- Provides posterior probability that the effect is real
- Handles seasonality, trends, and regression components

Since the causalimpact package has pandas compatibility issues,
we implement the core methodology directly:
1. Fit a Bayesian regression on pre-period using covariates
2. Project counterfactual into post-period
3. Compute pointwise and cumulative effects with uncertainty bands
4. Posterior probability of a causal effect via bootstrap

Outputs:
- data/causal_impact_results.csv
- data/causal_impact_summary.json
- results/causal_impact_main.png
- results/causal_impact_cumulative.png
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


def find_intervention_point(weekly):
    """Find period where TV spend changes dramatically."""
    tv_spend = (weekly['tv_broadcast_spend'] + weekly['tv_cable_spend'] +
                weekly['tv_streaming_spend'])

    # Look for the biggest sustained change
    rolling_avg = tv_spend.rolling(4, min_periods=1).mean()
    changes = rolling_avg.diff().abs()

    # Find biggest change, but ensure enough pre/post data
    best_idx = 0
    best_change = 0
    for i in range(15, len(weekly) - 10):
        pre_avg = tv_spend.iloc[max(0, i-8):i].mean()
        post_avg = tv_spend.iloc[i:min(len(weekly), i+8)].mean()
        change = abs(post_avg - pre_avg)
        if change > best_change:
            best_change = change
            best_idx = i

    return max(15, min(best_idx, len(weekly) - 10))


def build_covariates(weekly):
    """Build covariate matrix (digital spend + seasonality)."""
    X = pd.DataFrame({
        'paid_search': weekly['paid_search_spend'].values,
        'social': weekly['social_spend'].values,
        'display': weekly['display_spend'].values,
        'month_sin': np.sin(2 * np.pi * weekly['month'].values / 12),
        'month_cos': np.cos(2 * np.pi * weekly['month'].values / 12),
        'trend': np.arange(len(weekly)) / len(weekly),
    })
    return X.values


def run_bsts_causal_impact(weekly, intervention_week, n_bootstrap=1000):
    """
    Manual BSTS-style CausalImpact:
    1. Fit BayesianRidge on pre-period
    2. Project counterfactual into post-period
    3. Bootstrap for uncertainty intervals
    """
    print(f"\n=== BSTS CausalImpact: Revenue Effect of TV Campaign ===")
    print(f"  Intervention at week {intervention_week}")

    y = weekly['total_revenue'].values
    X = build_covariates(weekly)

    # Pre/post split
    X_pre, y_pre = X[:intervention_week], y[:intervention_week]
    X_post, y_post = X[intervention_week:], y[intervention_week:]

    print(f"  Pre-period: {len(X_pre)} weeks, Post-period: {len(X_post)} weeks")

    # Fit Bayesian Ridge on pre-period
    model = BayesianRidge(compute_score=True)
    model.fit(X_pre, y_pre)

    pre_r2 = r2_score(y_pre, model.predict(X_pre))
    print(f"  Pre-period model R²: {pre_r2:.4f}")

    # Predict counterfactual for full series
    y_pred_all, y_pred_std = model.predict(X, return_std=True)

    # Bootstrap uncertainty
    boot_predictions = np.zeros((n_bootstrap, len(X)))
    residuals = y_pre - model.predict(X_pre)
    residual_std = np.std(residuals)

    for b in range(n_bootstrap):
        noise = np.random.normal(0, residual_std, len(X))
        boot_predictions[b] = y_pred_all + noise

    # Compute intervals
    lower_all = np.percentile(boot_predictions, 5, axis=0)
    upper_all = np.percentile(boot_predictions, 95, axis=0)

    # Pointwise effects (actual - counterfactual)
    point_effect = y - y_pred_all
    point_lower = y - upper_all
    point_upper = y - lower_all

    # Cumulative effects (post-period only)
    cum_effect = np.cumsum(point_effect)
    cum_lower = np.cumsum(point_lower)
    cum_upper = np.cumsum(point_upper)

    # Post-period summary
    post_actual = y_post.sum()
    post_predicted = y_pred_all[intervention_week:].sum()
    post_effect = post_actual - post_predicted
    post_effect_pct = post_effect / post_predicted * 100

    # Bootstrap p-value: what fraction of bootstrap counterfactuals exceed actual?
    boot_post_sums = boot_predictions[:, intervention_week:].sum(axis=1)
    p_value = np.mean(boot_post_sums >= post_actual)

    print(f"\n  Post-period actual revenue: ${post_actual:,.0f}")
    print(f"  Counterfactual (no intervention): ${post_predicted:,.0f}")
    print(f"  Estimated causal effect: ${post_effect:,.0f} ({post_effect_pct:+.1f}%)")
    print(f"  Posterior prob of effect: {1 - p_value:.1%}")

    results = {
        'y_actual': y,
        'y_counterfactual': y_pred_all,
        'cf_lower': lower_all,
        'cf_upper': upper_all,
        'point_effect': point_effect,
        'point_lower': point_lower,
        'point_upper': point_upper,
        'cum_effect': cum_effect,
        'cum_lower': cum_lower,
        'cum_upper': cum_upper,
    }

    summary = {
        'intervention_week': int(intervention_week),
        'pre_r2': float(pre_r2),
        'post_actual': float(post_actual),
        'post_counterfactual': float(post_predicted),
        'causal_effect': float(post_effect),
        'causal_effect_pct': float(post_effect_pct),
        'posterior_prob': float(1 - p_value),
        'avg_weekly_effect': float(post_effect / len(X_post)),
    }

    return results, summary


def run_search_impact(weekly, intervention_week, n_bootstrap=1000):
    """CausalImpact on brand search volume."""
    print(f"\n=== BSTS CausalImpact: TV → Brand Search ===")

    if 'brand_search_volume' not in weekly.columns:
        print("  No brand_search_volume column, skipping.")
        return None, None

    y = weekly['brand_search_volume'].values
    X = pd.DataFrame({
        'social': weekly['social_spend'].values,
        'display': weekly['display_spend'].values,
        'month_sin': np.sin(2 * np.pi * weekly['month'].values / 12),
        'month_cos': np.cos(2 * np.pi * weekly['month'].values / 12),
        'trend': np.arange(len(weekly)) / len(weekly),
    }).values

    X_pre, y_pre = X[:intervention_week], y[:intervention_week]

    model = BayesianRidge()
    model.fit(X_pre, y_pre)

    y_pred_all, y_pred_std = model.predict(X, return_std=True)
    residuals = y_pre - model.predict(X_pre)
    residual_std = np.std(residuals)

    post_actual = y[intervention_week:].sum()
    post_predicted = y_pred_all[intervention_week:].sum()
    effect = post_actual - post_predicted
    effect_pct = effect / post_predicted * 100

    # Bootstrap p-value
    boot_sums = np.array([
        (y_pred_all[intervention_week:] +
         np.random.normal(0, residual_std, len(y) - intervention_week)).sum()
        for _ in range(n_bootstrap)
    ])
    p_value = np.mean(boot_sums >= post_actual)

    print(f"  Effect on brand search: {effect_pct:+.1f}%")
    print(f"  Posterior prob: {1 - p_value:.1%}")

    return {
        'y_actual': y,
        'y_counterfactual': y_pred_all,
        'point_effect': y - y_pred_all,
    }, {
        'search_effect': float(effect),
        'search_effect_pct': float(effect_pct),
        'posterior_prob': float(1 - p_value),
    }


def plot_results(results, summary, weekly, intervention_week, search_results=None):
    """Generate visualization plots."""
    os.makedirs('results', exist_ok=True)

    weeks = np.arange(1, len(weekly) + 1)
    y = results['y_actual']

    # 1. Main 3-panel CausalImpact plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Original + Counterfactual
    ax = axes[0]
    ax.plot(weeks, y / 1e6, 'o-', color='#2c3e50', markersize=3,
            linewidth=1.5, label='Actual Revenue')
    ax.plot(weeks, results['y_counterfactual'] / 1e6, '--', color='#3498db',
            linewidth=1.5, label='Counterfactual (no TV change)')
    ax.fill_between(weeks, results['cf_lower'] / 1e6, results['cf_upper'] / 1e6,
                    alpha=0.15, color='#3498db')
    ax.axvline(x=intervention_week, color='red', linestyle='--', alpha=0.7)
    ax.text(intervention_week + 0.5, ax.get_ylim()[0] + 0.7 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            'Intervention →', fontsize=10, color='red')
    ax.set_ylabel('Revenue ($M)')
    ax.set_title(f'CausalImpact: Causal Effect of TV Ramp-Up on Revenue\n'
                 f'Effect: {summary["causal_effect_pct"]:+.1f}%, '
                 f'Posterior Prob: {summary["posterior_prob"]:.0%}',
                 fontsize=13)
    ax.legend(fontsize=9)

    # Pointwise effect
    ax = axes[1]
    ax.plot(weeks, results['point_effect'] / 1e3, '-', color='#27ae60', linewidth=1.5)
    ax.fill_between(weeks, results['point_lower'] / 1e3, results['point_upper'] / 1e3,
                    alpha=0.15, color='#27ae60')
    ax.axvline(x=intervention_week, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_ylabel('Effect ($K)')
    ax.set_title('Pointwise Causal Effect (Actual − Counterfactual)')

    # Cumulative effect
    ax = axes[2]
    ax.plot(weeks, results['cum_effect'] / 1e6, '-', color='#8e44ad', linewidth=2)
    ax.fill_between(weeks, results['cum_lower'] / 1e6, results['cum_upper'] / 1e6,
                    alpha=0.15, color='#8e44ad')
    ax.axvline(x=intervention_week, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_ylabel('Cumulative ($M)')
    ax.set_title('Cumulative Causal Effect')
    ax.set_xlabel('Week')

    plt.tight_layout()
    plt.savefig('results/causal_impact_main.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. TV spend context + search impact
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    tv_spend = (weekly['tv_broadcast_spend'] + weekly['tv_cable_spend'] +
                weekly['tv_streaming_spend']).values

    ax = axes[0]
    ax.bar(weeks, tv_spend / 1e3, color='#2c3e50', alpha=0.7)
    ax.axvline(x=intervention_week, color='red', linestyle='--', alpha=0.7,
               label='Intervention')
    ax.set_ylabel('TV Spend ($K)')
    ax.set_title('Context: TV Spend Around the Intervention Point')
    ax.legend()

    if search_results is not None and search_results[0] is not None:
        ax = axes[1]
        sr = search_results[0]
        ax.plot(weeks, sr['y_actual'] / 1e3, 'o-', color='#2980b9',
                markersize=3, linewidth=1.5, label='Actual Search')
        ax.plot(weeks, sr['y_counterfactual'] / 1e3, '--', color='#e67e22',
                linewidth=1.5, label='Counterfactual')
        ax.axvline(x=intervention_week, color='red', linestyle='--', alpha=0.7)
        ax.set_ylabel('Search Volume (K)')
        ax.set_title(f'TV Impact on Brand Search: {search_results[1]["search_effect_pct"]:+.1f}%')
        ax.legend(fontsize=9)
    else:
        ax = axes[1]
        ax.plot(weeks, y / 1e6, 'o-', color='#27ae60', markersize=3, linewidth=1.5)
        ax.axvline(x=intervention_week, color='red', linestyle='--', alpha=0.7)
        ax.set_ylabel('Revenue ($M)')

    ax.set_xlabel('Week')
    plt.tight_layout()
    plt.savefig('results/causal_impact_cumulative.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Plots saved to results/causal_impact_*.png")


def main():
    print("=" * 60)
    print("CAUSAL IMPACT — Bayesian Structural Time Series")
    print("=" * 60)

    weekly = pd.read_csv('data/weekly_spend.csv')

    # Find intervention point
    intervention_week = find_intervention_point(weekly)
    tv_spend = (weekly['tv_broadcast_spend'] + weekly['tv_cable_spend'] +
                weekly['tv_streaming_spend'])
    print(f"\n  TV spend before intervention: ${tv_spend.iloc[:intervention_week].mean():,.0f}/week")
    print(f"  TV spend after intervention: ${tv_spend.iloc[intervention_week:].mean():,.0f}/week")

    # Run CausalImpact on revenue
    results, summary = run_bsts_causal_impact(weekly, intervention_week)

    # Run on search volume
    search_results = run_search_impact(weekly, intervention_week)

    if search_results[1] is not None:
        summary['search'] = search_results[1]

    # Save results
    os.makedirs('data', exist_ok=True)

    results_df = pd.DataFrame({
        'week': range(1, len(weekly) + 1),
        'actual': results['y_actual'],
        'counterfactual': results['y_counterfactual'],
        'cf_lower': results['cf_lower'],
        'cf_upper': results['cf_upper'],
        'point_effect': results['point_effect'],
        'cum_effect': results['cum_effect'],
    })
    results_df.to_csv('data/causal_impact_results.csv', index=False)

    with open('data/causal_impact_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot
    plot_results(results, summary, weekly, intervention_week, search_results)

    print("\n✓ CausalImpact analysis complete")
    return results, summary


if __name__ == '__main__':
    main()
