"""
TV-Search Cross-Channel Interaction Analysis
=============================================
Models the well-documented phenomenon where TV ads drive branded
search behavior. A TV ad airs → viewers Google the brand → conversions.

This means TV's true ROI is higher than what either attribution or
standard MMM capture, because some of TV's value shows up as
"Paid Search" or "Organic Search" conversions.

Techniques:
1. Granger Causality — Does TV spend statistically "cause" search volume?
2. Cross-correlation — At what lag does TV most affect search?
3. Interaction terms — TV × Search synergy in the MMM
4. Mediation analysis — How much of TV's effect flows through search?

Outputs:
- data/granger_results.csv
- data/cross_channel_effects.csv
- data/tv_search_decomposition.csv
- results/tv_search_lag_correlation.png
- results/granger_causality.png
- results/tv_search_interaction.png
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


def compute_cross_correlations(weekly_spend):
    """
    Compute cross-correlation between TV spend and brand search volume
    at different lags. TV should lead search (positive lags).
    """
    print("\n  Computing cross-correlations...")

    tv_total = (weekly_spend['tv_broadcast_spend'] +
                weekly_spend['tv_cable_spend'] +
                weekly_spend['tv_streaming_spend'])
    search = weekly_spend['brand_search_volume']
    revenue = weekly_spend['total_revenue']

    # Normalize
    tv_norm = (tv_total - tv_total.mean()) / tv_total.std()
    search_norm = (search - search.mean()) / search.std()
    revenue_norm = (revenue - revenue.mean()) / revenue.std()

    results = []
    max_lag = 8

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            # TV leads search by `lag` weeks
            corr_search = np.corrcoef(tv_norm[:-lag], search_norm[lag:])[0, 1]
            corr_rev = np.corrcoef(tv_norm[:-lag], revenue_norm[lag:])[0, 1]
        elif lag < 0:
            corr_search = np.corrcoef(tv_norm[-lag:], search_norm[:lag])[0, 1]
            corr_rev = np.corrcoef(tv_norm[-lag:], revenue_norm[:lag])[0, 1]
        else:
            corr_search = np.corrcoef(tv_norm, search_norm)[0, 1]
            corr_rev = np.corrcoef(tv_norm, revenue_norm)[0, 1]

        results.append({
            'lag': lag,
            'tv_search_correlation': round(float(corr_search), 4)
                if np.isfinite(corr_search) else 0,
            'tv_revenue_correlation': round(float(corr_rev), 4)
                if np.isfinite(corr_rev) else 0,
        })

    results_df = pd.DataFrame(results)

    # Find peak lag
    peak = results_df.loc[results_df['tv_search_correlation'].idxmax()]
    print(f"  Peak TV→Search correlation: {peak['tv_search_correlation']:.3f} "
          f"at lag={int(peak['lag'])} weeks")
    peak_rev = results_df.loc[results_df['tv_revenue_correlation'].idxmax()]
    print(f"  Peak TV→Revenue correlation: {peak_rev['tv_revenue_correlation']:.3f} "
          f"at lag={int(peak_rev['lag'])} weeks")

    return results_df


def run_granger_causality(weekly_spend, max_lag=4):
    """
    Granger causality test: Does past TV spend help predict
    current brand search volume beyond what past search predicts alone?

    H0: TV spend does NOT Granger-cause brand search
    H1: TV spend DOES Granger-cause brand search

    A low p-value means TV spending is predictive of future search volume.
    """
    print("\n  Running Granger causality tests...")

    tv_total = (weekly_spend['tv_broadcast_spend'] +
                weekly_spend['tv_cable_spend'] +
                weekly_spend['tv_streaming_spend']).values
    search = weekly_spend['brand_search_volume'].values
    revenue = weekly_spend['total_revenue'].values
    paid_search = weekly_spend['paid_search_spend'].values

    # Check stationarity (Granger requires stationary series)
    adf_tv = adfuller(tv_total)
    adf_search = adfuller(search)
    print(f"  TV spend ADF p-value: {adf_tv[1]:.4f} "
          f"({'stationary' if adf_tv[1] < 0.05 else 'non-stationary'})")
    print(f"  Search vol ADF p-value: {adf_search[1]:.4f} "
          f"({'stationary' if adf_search[1] < 0.05 else 'non-stationary'})")

    # If non-stationary, difference
    if adf_tv[1] > 0.05:
        tv_total = np.diff(tv_total)
        search = np.diff(search)
        revenue = np.diff(revenue)
        paid_search = np.diff(paid_search)
        print("  Applied first differencing for stationarity")

    # Granger test: TV → Search
    data_tv_search = np.column_stack([search, tv_total])
    results = []

    try:
        gc_results = grangercausalitytests(data_tv_search, maxlag=max_lag, verbose=False)
        for lag in range(1, max_lag + 1):
            f_stat = gc_results[lag][0]['ssr_ftest'][0]
            p_value = gc_results[lag][0]['ssr_ftest'][1]
            significant = p_value < 0.05
            results.append({
                'test': 'TV → Search',
                'lag': lag,
                'f_statistic': round(float(f_stat), 4),
                'p_value': round(float(p_value), 4),
                'significant': significant,
            })
            marker = '***' if p_value < 0.01 else '**' if p_value < 0.05 else ''
            print(f"  TV→Search lag={lag}: F={f_stat:.2f}, p={p_value:.4f} {marker}")
    except Exception as e:
        print(f"  TV→Search Granger test failed: {e}")

    # Granger test: TV → Revenue
    data_tv_rev = np.column_stack([revenue, tv_total])
    try:
        gc_results = grangercausalitytests(data_tv_rev, maxlag=max_lag, verbose=False)
        for lag in range(1, max_lag + 1):
            f_stat = gc_results[lag][0]['ssr_ftest'][0]
            p_value = gc_results[lag][0]['ssr_ftest'][1]
            results.append({
                'test': 'TV → Revenue',
                'lag': lag,
                'f_statistic': round(float(f_stat), 4),
                'p_value': round(float(p_value), 4),
                'significant': p_value < 0.05,
            })
    except Exception as e:
        print(f"  TV→Revenue Granger test failed: {e}")

    # Granger test: TV → Paid Search
    data_tv_ps = np.column_stack([paid_search, tv_total])
    try:
        gc_results = grangercausalitytests(data_tv_ps, maxlag=max_lag, verbose=False)
        for lag in range(1, max_lag + 1):
            f_stat = gc_results[lag][0]['ssr_ftest'][0]
            p_value = gc_results[lag][0]['ssr_ftest'][1]
            results.append({
                'test': 'TV → Paid Search Spend',
                'lag': lag,
                'f_statistic': round(float(f_stat), 4),
                'p_value': round(float(p_value), 4),
                'significant': p_value < 0.05,
            })
    except Exception as e:
        print(f"  TV→Paid Search Granger test failed: {e}")

    return pd.DataFrame(results)


def build_interaction_model(weekly_spend):
    """
    Build an MMM with TV × Search interaction term.

    revenue = base + β_tv * TV + β_search * Search + β_int * (TV × Search)
              + other_channels + seasonality + error

    If β_int > 0, it means TV and Search have synergistic effects:
    the combined impact exceeds the sum of individual impacts.
    """
    print("\n  Building interaction model...")

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge

    tv_total = (weekly_spend['tv_broadcast_spend'] +
                weekly_spend['tv_cable_spend'] +
                weekly_spend['tv_streaming_spend']).values
    paid_search = weekly_spend['paid_search_spend'].values
    social = weekly_spend['social_spend'].values
    display = weekly_spend['display_spend'].values
    revenue = weekly_spend['total_revenue'].values
    month_sin = np.sin(2 * np.pi * weekly_spend['month'].values / 12)
    month_cos = np.cos(2 * np.pi * weekly_spend['month'].values / 12)
    trend = np.arange(len(weekly_spend)) / len(weekly_spend)

    # Model WITHOUT interaction
    X_base = np.column_stack([tv_total, paid_search, social, display,
                               month_sin, month_cos, trend])
    scaler_base = StandardScaler()
    X_base_scaled = scaler_base.fit_transform(X_base)
    model_base = Ridge(alpha=1.0).fit(X_base_scaled, revenue)
    r2_base = model_base.score(X_base_scaled, revenue)

    # Model WITH interaction
    tv_x_search = tv_total * paid_search / (tv_total.std() * paid_search.std())
    X_int = np.column_stack([tv_total, paid_search, social, display,
                              tv_x_search, month_sin, month_cos, trend])
    scaler_int = StandardScaler()
    X_int_scaled = scaler_int.fit_transform(X_int)
    model_int = Ridge(alpha=1.0).fit(X_int_scaled, revenue)
    r2_int = model_int.score(X_int_scaled, revenue)

    feature_names = ['TV', 'Paid Search', 'Social', 'Display',
                     'TV × Search', 'month_sin', 'month_cos', 'trend']
    coefs = model_int.coef_

    print(f"  Model without interaction: R² = {r2_base:.4f}")
    print(f"  Model with interaction:    R² = {r2_int:.4f}")
    print(f"  R² improvement: +{(r2_int - r2_base):.4f}")
    print(f"\n  Coefficients:")
    for name, coef in zip(feature_names, coefs):
        print(f"    {name}: {coef:,.0f}")

    interaction_coef = coefs[4]  # TV × Search
    synergy_pct = abs(interaction_coef) / abs(coefs[0] + coefs[1]) * 100

    results = {
        'r2_base': round(float(r2_base), 4),
        'r2_interaction': round(float(r2_int), 4),
        'r2_improvement': round(float(r2_int - r2_base), 4),
        'interaction_coefficient': round(float(interaction_coef), 2),
        'synergy_pct': round(float(synergy_pct), 1),
        'coefficients': {name: round(float(coef), 2)
                         for name, coef in zip(feature_names, coefs)},
    }

    return results, model_int, X_int_scaled, revenue


def estimate_mediation_effect(weekly_spend):
    """
    Mediation analysis: How much of TV's revenue effect is mediated
    through brand search?

    Path: TV → Brand Search → Revenue (indirect/mediated)
    Path: TV → Revenue directly (direct)

    Total TV effect = Direct + Indirect (through search)
    """
    print("\n  Estimating mediation effects...")

    tv_total = (weekly_spend['tv_broadcast_spend'] +
                weekly_spend['tv_cable_spend'] +
                weekly_spend['tv_streaming_spend']).values
    search = weekly_spend['brand_search_volume'].values
    revenue = weekly_spend['total_revenue'].values
    month_sin = np.sin(2 * np.pi * weekly_spend['month'].values / 12)
    month_cos = np.cos(2 * np.pi * weekly_spend['month'].values / 12)

    # Step 1: Total effect (TV → Revenue)
    X_total = add_constant(np.column_stack([tv_total, month_sin, month_cos]))
    model_total = OLS(revenue, X_total).fit()
    total_effect = model_total.params[1]

    # Step 2: TV → Search (path a)
    X_a = add_constant(np.column_stack([tv_total, month_sin, month_cos]))
    model_a = OLS(search, X_a).fit()
    path_a = model_a.params[1]

    # Step 3: TV + Search → Revenue (paths c' and b)
    X_med = add_constant(np.column_stack([tv_total, search, month_sin, month_cos]))
    model_med = OLS(revenue, X_med).fit()
    direct_effect = model_med.params[1]  # c' (TV → Revenue controlling for search)
    path_b = model_med.params[2]  # b (Search → Revenue controlling for TV)

    # Indirect effect = a × b
    indirect_effect = path_a * path_b
    mediation_pct = indirect_effect / total_effect * 100 if total_effect != 0 else 0

    print(f"  Total TV effect on revenue: {total_effect:,.2f}")
    print(f"  Direct TV effect (controlling for search): {direct_effect:,.2f}")
    print(f"  Indirect effect (TV → Search → Revenue): {indirect_effect:,.2f}")
    print(f"  Mediation percentage: {mediation_pct:.1f}%")
    print(f"  → {mediation_pct:.0f}% of TV's revenue effect flows through brand search")

    return {
        'total_effect': round(float(total_effect), 2),
        'direct_effect': round(float(direct_effect), 2),
        'indirect_effect': round(float(indirect_effect), 2),
        'mediation_pct': round(float(mediation_pct), 1),
        'path_a_tv_to_search': round(float(path_a), 4),
        'path_b_search_to_revenue': round(float(path_b), 2),
    }


def generate_cross_channel_plots(cross_corr_df, granger_df, interaction_results,
                                  mediation_results, weekly_spend):
    """Generate cross-channel visualizations."""
    print("\n  Generating plots...")
    os.makedirs('results', exist_ok=True)

    # --- Plot 1: Cross-correlation lag plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(cross_corr_df['lag'], cross_corr_df['tv_search_correlation'],
           color=['#e74c3c' if l > 0 else '#3498db' for l in cross_corr_df['lag']],
           alpha=0.7, width=0.8)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag (weeks, positive = TV leads)')
    ax.set_ylabel('Correlation')
    ax.set_title('TV Spend → Brand Search Volume: Cross-Correlation\n'
                 '(Red bars = TV leads search, Blue = search leads TV)',
                 fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate peak
    peak_idx = cross_corr_df['tv_search_correlation'].idxmax()
    peak = cross_corr_df.iloc[peak_idx]
    ax.annotate(f"Peak: r={peak['tv_search_correlation']:.3f}\nat lag={int(peak['lag'])}",
                xy=(peak['lag'], peak['tv_search_correlation']),
                xytext=(peak['lag'] + 1.5, peak['tv_search_correlation'] + 0.05),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/tv_search_lag_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/tv_search_lag_correlation.png")

    # --- Plot 2: Granger causality p-values ---
    fig, ax = plt.subplots(figsize=(10, 5))
    tv_search = granger_df[granger_df['test'] == 'TV → Search']
    tv_revenue = granger_df[granger_df['test'] == 'TV → Revenue']

    if len(tv_search) > 0:
        ax.plot(tv_search['lag'], tv_search['p_value'], 'o-',
                color='#e74c3c', linewidth=2, markersize=8, label='TV → Search')
    if len(tv_revenue) > 0:
        ax.plot(tv_revenue['lag'], tv_revenue['p_value'], 's-',
                color='#3498db', linewidth=2, markersize=8, label='TV → Revenue')

    ax.axhline(y=0.05, color='black', linestyle='--', alpha=0.5, label='p=0.05 threshold')
    ax.fill_between([0.5, 4.5], 0, 0.05, alpha=0.1, color='green',
                    label='Significant (p<0.05)')
    ax.set_xlabel('Lag (weeks)')
    ax.set_ylabel('p-value')
    ax.set_title('Granger Causality: Does TV Spend Predict Future Outcomes?\n'
                 '(Below dashed line = statistically significant)',
                 fontweight='bold')
    ax.legend()
    ax.set_ylim(-0.02, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/granger_causality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/granger_causality.png")

    # --- Plot 3: Mediation diagram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Boxes
    for (x, y, label) in [(1, 4, 'TV Spend'), (5, 7, 'Brand Search'), (9, 4, 'Revenue')]:
        bbox = dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', edgecolor='#2c3e50')
        ax.text(x, y, label, fontsize=14, fontweight='bold', ha='center', va='center',
                bbox=bbox)

    # Arrows
    med = mediation_results
    # TV → Search (path a)
    ax.annotate('', xy=(4, 6.8), xytext=(2, 4.5),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
    ax.text(2.5, 5.9, f"a = {med['path_a_tv_to_search']:.4f}",
            fontsize=11, color='#e74c3c', fontweight='bold')

    # Search → Revenue (path b)
    ax.annotate('', xy=(8, 4.5), xytext=(6, 6.8),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
    ax.text(7.2, 5.9, f"b = {med['path_b_search_to_revenue']:,.0f}",
            fontsize=11, color='#e74c3c', fontweight='bold')

    # TV → Revenue (direct, c')
    ax.annotate('', xy=(8, 3.8), xytext=(2, 3.8),
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))
    ax.text(5, 3.2, f"c' (direct) = {med['direct_effect']:,.0f}",
            fontsize=11, color='#3498db', fontweight='bold', ha='center')

    # Summary
    ax.text(5, 1.5,
            f"Indirect effect (a×b): {med['indirect_effect']:,.0f}\n"
            f"Total effect: {med['total_effect']:,.0f}\n"
            f"Mediation: {med['mediation_pct']:.0f}% of TV effect flows through search",
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#fff3cd', edgecolor='#ffc107'))

    ax.set_title('Mediation Analysis: TV → Search → Revenue\n'
                 '(How much of TV\'s impact is mediated through brand search?)',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig('results/tv_search_interaction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/tv_search_interaction.png")


def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("=" * 60)
    print("TV-SEARCH CROSS-CHANNEL ANALYSIS")
    print("=" * 60)

    weekly_spend = pd.read_csv('data/weekly_spend.csv')

    # Cross-correlations
    cross_corr_df = compute_cross_correlations(weekly_spend)

    # Granger causality
    granger_df = run_granger_causality(weekly_spend)

    # Interaction model
    interaction_results, model, X, revenue = build_interaction_model(weekly_spend)

    # Mediation analysis
    mediation_results = estimate_mediation_effect(weekly_spend)

    # Save
    cross_corr_df.to_csv('data/cross_channel_correlations.csv', index=False)
    granger_df.to_csv('data/granger_results.csv', index=False)

    effects = {
        'cross_correlations': cross_corr_df.to_dict(orient='records'),
        'interaction_model': interaction_results,
        'mediation': mediation_results,
    }
    with open('data/cross_channel_effects.json', 'w') as f:
        json.dump(effects, f, indent=2, default=str)
    print("  Saved data/cross_channel_effects.json")

    # Generate plots
    generate_cross_channel_plots(cross_corr_df, granger_df,
                                  interaction_results, mediation_results,
                                  weekly_spend)

    print("\n" + "=" * 60)
    print("CROSS-CHANNEL ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
