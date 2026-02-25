"""
Media Mix Model (MMM)
=====================
Estimates the marginal contribution of each advertising channel
to revenue, accounting for adstock carryover and diminishing returns.

Pipeline:
1. ADSTOCK TRANSFORMATION — Geometric decay carryover
2. SATURATION CURVES — Hill function diminishing returns
3. MMM REGRESSION — Ridge regression with transformed features
4. CHANNEL CONTRIBUTION — Decompose revenue into channel effects
5. BUDGET OPTIMIZER — Constrained optimization for max revenue

Outputs:
- models/mmm_model.joblib
- models/mmm_params.json
- data/channel_contributions.csv
- data/optimal_allocation.csv
- results/actual_vs_predicted.png
- results/channel_contributions.png
- results/saturation_curves.png
- results/adstock_curves.png
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from scipy.optimize import minimize, minimize_scalar
import joblib
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================
# CHANNEL DEFINITIONS
# ============================================

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

CHANNEL_COLORS = {
    'TV Broadcast': '#2c3e50',
    'TV Cable': '#34495e',
    'TV Streaming': '#7f8c8d',
    'Paid Search': '#2980b9',
    'Social': '#8e44ad',
    'Display': '#e67e22',
    'Base + Seasonality': '#95a5a6',
}


# ============================================
# 1. ADSTOCK TRANSFORMATION
# ============================================

def adstock_transform(spend_series, decay_rate):
    """
    Geometric adstock: adstock[t] = spend[t] + decay * adstock[t-1]

    Models carryover effect — a TV ad today still influences behavior
    in subsequent weeks.

    Parameters:
    - spend_series: array of weekly spend
    - decay_rate: 0.0 (no carryover) to 0.99 (very long carryover)
      TV typically: 0.70-0.85
      Digital: 0.30-0.50
    """
    spend = np.asarray(spend_series, dtype=float)
    adstocked = np.zeros_like(spend)
    adstocked[0] = spend[0]
    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay_rate * adstocked[t - 1]
    return adstocked


def find_optimal_decay(spend_series, revenue_series, decay_range=(0.1, 0.95)):
    """
    Grid search for the adstock decay rate that maximizes
    correlation with revenue.
    """
    best_corr = -np.inf
    best_decay = 0.5

    for decay in np.arange(decay_range[0], decay_range[1], 0.05):
        adstocked = adstock_transform(spend_series, decay)
        corr = np.corrcoef(adstocked, revenue_series)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_decay = decay

    return best_decay, best_corr


# ============================================
# 2. SATURATION CURVES (Hill Function)
# ============================================

def hill_saturation(spend, alpha, K):
    """
    Hill function: saturated = spend^alpha / (spend^alpha + K^alpha)

    Models diminishing returns — the 10th million in TV has less impact
    than the 1st million.

    Parameters:
    - alpha: shape parameter (steepness of saturation)
    - K: half-saturation point (spend level at 50% of max effect)
    """
    spend = np.asarray(spend, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.power(spend, alpha) / (np.power(spend, alpha) + np.power(K, alpha))
    return np.nan_to_num(result, 0.0)


def fit_hill_params(adstocked_spend, revenue, initial_alpha=0.7, initial_K=None):
    """
    Fit Hill function parameters using grid search over alpha and K.
    Maximizes correlation between saturated spend and revenue.

    Grid search is more robust than gradient optimization for this
    2-parameter problem with only 52 data points.
    """
    if initial_K is None:
        initial_K = np.median(adstocked_spend[adstocked_spend > 0])
    if initial_K == 0:
        initial_K = 1.0

    best_corr = -np.inf
    best_alpha = 0.7
    best_K = initial_K

    # Grid search: alpha in [0.3, 2.0], K around the median spend
    for alpha in np.arange(0.3, 2.05, 0.1):
        for K_mult in np.arange(0.2, 3.05, 0.2):
            K = initial_K * K_mult
            if K <= 0:
                continue
            saturated = hill_saturation(adstocked_spend, alpha, K)
            if np.std(saturated) < 1e-10:
                continue
            corr = np.corrcoef(saturated, revenue)[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_alpha = alpha
                best_K = K

    return best_alpha, best_K


# ============================================
# 3. MMM REGRESSION
# ============================================

def build_mmm(weekly_spend):
    """
    Full MMM pipeline:
    1. Find optimal adstock decay per channel
    2. Fit Hill saturation per channel
    3. Ridge regression on transformed features
    """
    print("\n[STEP 1] Finding optimal adstock decay rates...")
    revenue = weekly_spend['total_revenue'].values

    adstock_params = {}
    adstocked_data = {}

    for col in SPEND_COLUMNS:
        spend = weekly_spend[col].values
        if spend.sum() == 0:
            adstock_params[col] = 0.5
            adstocked_data[col] = spend
            continue

        decay, corr = find_optimal_decay(spend, revenue)
        adstock_params[col] = round(float(decay), 2)
        adstocked_data[col] = adstock_transform(spend, decay)
        print(f"  {CHANNEL_NAMES[col]}: decay={decay:.2f}, corr={corr:.3f}")

    print("\n[STEP 2] Fitting Hill saturation curves...")
    hill_params = {}
    saturated_data = {}

    for col in SPEND_COLUMNS:
        adstocked = adstocked_data[col]
        if adstocked.sum() == 0:
            hill_params[col] = {'alpha': 0.7, 'K': 1.0}
            saturated_data[col] = adstocked
            continue

        alpha, K = fit_hill_params(adstocked, revenue)
        hill_params[col] = {'alpha': round(float(alpha), 3), 'K': round(float(K), 2)}
        saturated_data[col] = hill_saturation(adstocked, alpha, K)
        print(f"  {CHANNEL_NAMES[col]}: alpha={alpha:.3f}, K={K:.0f}")

    print("\n[STEP 3] Building Ridge regression model...")

    # Build feature matrix
    features = pd.DataFrame(saturated_data)

    # Add seasonality and trend controls
    features['month_sin'] = np.sin(2 * np.pi * weekly_spend['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * weekly_spend['month'] / 12)
    features['trend'] = np.arange(len(weekly_spend)) / len(weekly_spend)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)

    # Ridge regression (handles collinearity between channels)
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, revenue)

    # Evaluate
    predicted = model.predict(X_scaled)
    r2 = r2_score(revenue, predicted)
    mape = mean_absolute_percentage_error(revenue, predicted)
    print(f"  R² = {r2:.4f}")
    print(f"  MAPE = {mape:.4f}")

    return model, scaler, features, adstock_params, hill_params, predicted


# ============================================
# 4. CHANNEL CONTRIBUTION
# ============================================

def compute_contributions(model, scaler, features, weekly_spend):
    """
    Decompose total predicted revenue into channel contributions.

    Uses the proper approach: for each channel, compare prediction
    with actual spend vs. prediction with that channel's spend set to 0
    (in the original/unscaled space, then re-scaled).
    """
    print("\n[STEP 4] Computing channel contributions...")

    X_scaled = scaler.transform(features.values)
    feature_names = list(features.columns)
    total_predicted = model.predict(X_scaled)

    # For each channel, set its spend to zero in ORIGINAL space,
    # then re-scale and predict. The difference is the contribution.
    contributions = {}
    channel_contribs = {}

    for col in SPEND_COLUMNS:
        if col not in feature_names:
            continue
        col_idx = feature_names.index(col)

        # Create features with this channel zeroed out
        features_zero = features.copy()
        features_zero[col] = 0.0
        X_zero_scaled = scaler.transform(features_zero.values)
        pred_without = model.predict(X_zero_scaled)

        contribution_per_week = total_predicted - pred_without
        contributions[col] = contribution_per_week

        name = CHANNEL_NAMES[col]
        channel_contribs[name] = contribution_per_week.sum()

    # Base = prediction when ALL channels are zero
    features_all_zero = features.copy()
    for col in SPEND_COLUMNS:
        if col in feature_names:
            features_all_zero[col] = 0.0
    X_all_zero = scaler.transform(features_all_zero.values)
    pred_base = model.predict(X_all_zero)
    base_contrib = pred_base.sum()

    # Adjust: total must sum to total predicted
    channel_sum = sum(channel_contribs.values())
    residual = total_predicted.sum() - channel_sum - base_contrib
    base_contrib += residual  # absorb interaction effects into base
    channel_contribs['Base + Seasonality'] = base_contrib

    total = total_predicted.sum()
    print("\n  Channel Contributions:")
    results = []
    for name, contrib in sorted(channel_contribs.items(), key=lambda x: -x[1]):
        pct = contrib / total * 100 if total > 0 else 0
        spend = 0
        for col in SPEND_COLUMNS:
            if CHANNEL_NAMES.get(col) == name:
                spend = weekly_spend[col].sum()
        roi = contrib / spend if spend > 0 else 0
        print(f"    {name}: ${contrib:,.0f} ({pct:.1f}%) | Spend: ${spend:,.0f} | ROI: {roi:.2f}x")
        results.append({
            'channel': name,
            'contribution': round(contrib, 2),
            'pct_contribution': round(pct, 2),
            'total_spend': round(spend, 2),
            'roi': round(roi, 3),
        })

    return pd.DataFrame(results), contributions


# ============================================
# 5. BUDGET OPTIMIZER
# ============================================

def optimize_budget(total_budget, model, scaler, features, adstock_params,
                    hill_params, weekly_spend, current_allocation):
    """
    Find the channel allocation that maximizes predicted revenue
    given a total budget constraint.

    Constraints:
    - Sum of all channels = total_budget
    - Each channel: min 2% of total, max 60% of total
    """
    n_channels = len(SPEND_COLUMNS)
    n_weeks = len(weekly_spend)

    # Pre-compute seasonality features
    month_sin = np.sin(2 * np.pi * weekly_spend['month'].values / 12)
    month_cos = np.cos(2 * np.pi * weekly_spend['month'].values / 12)
    trend = np.arange(n_weeks) / n_weeks

    def neg_revenue(allocation):
        """Given annual allocation per channel, predict total annual revenue."""
        # Distribute annual budget equally across weeks (simplified)
        weekly_alloc = allocation / n_weeks

        feature_dict = {}
        for j, col in enumerate(SPEND_COLUMNS):
            weekly_spend_arr = np.full(n_weeks, weekly_alloc[j])
            decay = adstock_params[col]
            adstocked = adstock_transform(weekly_spend_arr, decay)
            alpha = hill_params[col]['alpha']
            K = hill_params[col]['K']
            saturated = hill_saturation(adstocked, alpha, K)
            feature_dict[col] = saturated

        feature_dict['month_sin'] = month_sin
        feature_dict['month_cos'] = month_cos
        feature_dict['trend'] = trend

        feat_df = pd.DataFrame(feature_dict)
        X = scaler.transform(feat_df.values)
        revenue = model.predict(X).sum()
        return -revenue

    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget}]

    # Bounds: 2% to 60% of total per channel
    bounds = [(total_budget * 0.02, total_budget * 0.60)] * n_channels

    # Initial guess: current allocation scaled to target budget
    x0 = current_allocation * (total_budget / current_allocation.sum())

    result = minimize(
        neg_revenue, x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-8}
    )

    return result.x, -result.fun


def compute_optimal_allocations(model, scaler, features, adstock_params,
                                hill_params, weekly_spend):
    """
    Compute optimal allocations for a range of budget levels.
    """
    print("\n[STEP 5] Optimizing budget allocations...")

    current_allocation = np.array([
        weekly_spend[col].sum() for col in SPEND_COLUMNS
    ])
    current_total = current_allocation.sum()

    budget_levels = [
        5_000_000, 8_000_000, 10_000_000, 12_000_000, 15_000_000,
        18_000_000, 20_000_000, 25_000_000, 30_000_000
    ]

    results = []
    for budget in budget_levels:
        opt_alloc, opt_revenue = optimize_budget(
            budget, model, scaler, features, adstock_params,
            hill_params, weekly_spend, current_allocation
        )

        # Also compute current-mix revenue at this budget
        current_at_budget = current_allocation * (budget / current_total)
        # Compute revenue for current mix
        neg_rev_current = 0
        n_weeks = len(weekly_spend)
        feature_dict = {}
        for j, col in enumerate(SPEND_COLUMNS):
            weekly_spend_arr = np.full(n_weeks, current_at_budget[j] / n_weeks)
            decay = adstock_params[col]
            adstocked = adstock_transform(weekly_spend_arr, decay)
            alpha = hill_params[col]['alpha']
            K = hill_params[col]['K']
            feature_dict[col] = hill_saturation(adstocked, alpha, K)
        feature_dict['month_sin'] = np.sin(2 * np.pi * weekly_spend['month'].values / 12)
        feature_dict['month_cos'] = np.cos(2 * np.pi * weekly_spend['month'].values / 12)
        feature_dict['trend'] = np.arange(n_weeks) / n_weeks
        feat_df = pd.DataFrame(feature_dict)
        X = scaler.transform(feat_df.values)
        current_revenue = model.predict(X).sum()

        row = {'budget': budget, 'optimized_revenue': round(opt_revenue, 2),
               'current_mix_revenue': round(current_revenue, 2),
               'lift_pct': round((opt_revenue - current_revenue) / current_revenue * 100, 2)}
        for j, col in enumerate(SPEND_COLUMNS):
            row[f'optimal_{col}'] = round(opt_alloc[j], 2)
            row[f'current_{col}'] = round(current_at_budget[j], 2)
        results.append(row)

        lift_pct = (opt_revenue - current_revenue) / current_revenue * 100
        print(f"  Budget ${budget/1e6:.0f}M: "
              f"Current=${current_revenue/1e6:.1f}M, "
              f"Optimal=${opt_revenue/1e6:.1f}M, "
              f"Lift={lift_pct:.1f}%")

    return pd.DataFrame(results)


# ============================================
# PLOTS
# ============================================

def generate_plots(weekly_spend, predicted, contributions_df,
                   adstock_params, hill_params):
    """Generate all MMM visualizations."""
    print("\n[STEP 6] Generating MMM plots...")
    os.makedirs('results', exist_ok=True)
    revenue = weekly_spend['total_revenue'].values

    # --- Plot 1: Actual vs Predicted Revenue ---
    fig, ax = plt.subplots(figsize=(12, 5))
    weeks = weekly_spend['week'].values
    ax.plot(weeks, revenue / 1e6, 'o-', color='#2c3e50', markersize=4,
            label=f'Actual Revenue')
    ax.plot(weeks, predicted / 1e6, 's--', color='#e74c3c', markersize=4,
            alpha=0.8, label=f'MMM Predicted (R²={r2_score(revenue, predicted):.3f})')
    ax.fill_between(weeks, revenue / 1e6, predicted / 1e6,
                    alpha=0.1, color='#e74c3c')
    ax.set_xlabel('Week')
    ax.set_ylabel('Revenue ($M)')
    ax.set_title('Media Mix Model: Actual vs Predicted Revenue', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/actual_vs_predicted.png")

    # --- Plot 2: Channel Contributions Waterfall ---
    fig, ax = plt.subplots(figsize=(12, 6))
    contribs = contributions_df.sort_values('contribution', ascending=False)
    colors = [CHANNEL_COLORS.get(name, '#bdc3c7') for name in contribs['channel']]
    bars = ax.barh(contribs['channel'], contribs['contribution'] / 1e6, color=colors)
    ax.set_xlabel('Revenue Contribution ($M)')
    ax.set_title('Channel Revenue Contributions (MMM)', fontweight='bold')
    for bar, val in zip(bars, contribs['contribution'] / 1e6):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'${val:.1f}M', va='center', fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('results/channel_contributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/channel_contributions.png")

    # --- Plot 3: Saturation Curves ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, col in enumerate(SPEND_COLUMNS):
        ax = axes[idx]
        name = CHANNEL_NAMES[col]
        alpha = hill_params[col]['alpha']
        K = hill_params[col]['K']

        # Generate curve
        max_spend = weekly_spend[col].max() * 2.5
        spend_range = np.linspace(0, max_spend, 200)
        saturated = hill_saturation(spend_range, alpha, K)

        ax.plot(spend_range / 1e3, saturated, color=CHANNEL_COLORS.get(name, '#3498db'),
                linewidth=2)

        # Mark current spend level
        current_spend = weekly_spend[col].mean()
        current_sat = hill_saturation(np.array([current_spend]), alpha, K)[0]
        ax.scatter([current_spend / 1e3], [current_sat], color='#e74c3c',
                   s=100, zorder=5, label=f'Current ({current_sat:.0%} saturated)')

        # Half-saturation point
        ax.axvline(x=K / 1e3, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

        ax.set_title(f'{name}\n(α={alpha:.2f}, K=${K/1e3:.0f}K)', fontsize=10)
        ax.set_xlabel('Weekly Spend ($K)')
        ax.set_ylabel('Saturation')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Saturation Curves by Channel (Hill Function)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/saturation_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/saturation_curves.png")

    # --- Plot 4: Adstock Decay Curves ---
    fig, ax = plt.subplots(figsize=(10, 6))
    weeks_range = np.arange(12)

    for col in SPEND_COLUMNS:
        name = CHANNEL_NAMES[col]
        decay = adstock_params[col]

        # Impulse response: effect of $1 over time
        impulse = np.zeros(12)
        impulse[0] = 1.0
        response = adstock_transform(impulse, decay)

        ax.plot(weeks_range, response, 'o-',
                color=CHANNEL_COLORS.get(name, '#3498db'),
                linewidth=2, markersize=5,
                label=f'{name} (decay={decay:.2f})')

    ax.set_xlabel('Weeks After Ad Spend')
    ax.set_ylabel('Remaining Effect (Normalized)')
    ax.set_title('Adstock Decay Curves: How Long Does Each Channel\'s Effect Last?',
                 fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(weeks_range)
    plt.tight_layout()
    plt.savefig('results/adstock_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/adstock_curves.png")


# ============================================
# MAIN
# ============================================

def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("=" * 60)
    print("MEDIA MIX MODEL (MMM)")
    print("=" * 60)

    # Load weekly spend data
    print("\nLoading data...")
    weekly_spend = pd.read_csv('data/weekly_spend.csv')
    print(f"  Weeks: {len(weekly_spend)}")
    print(f"  Total spend: ${weekly_spend['total_spend'].sum():,.0f}")
    print(f"  Total revenue: ${weekly_spend['total_revenue'].sum():,.0f}")

    # Build MMM
    model, scaler, features, adstock_params, hill_params, predicted = build_mmm(
        weekly_spend
    )

    # Channel contributions
    contributions_df, contributions_dict = compute_contributions(
        model, scaler, features, weekly_spend
    )

    # Budget optimization
    optimal_df = compute_optimal_allocations(
        model, scaler, features, adstock_params, hill_params, weekly_spend
    )

    # Save model artifacts
    print("\nSaving model artifacts...")
    joblib.dump({'model': model, 'scaler': scaler}, 'models/mmm_model.joblib')

    mmm_params = {
        'adstock_decays': adstock_params,
        'hill_params': hill_params,
        'feature_columns': list(features.columns),
        'spend_columns': SPEND_COLUMNS,
        'channel_names': CHANNEL_NAMES,
        'r2': float(r2_score(weekly_spend['total_revenue'].values, predicted)),
        'mape': float(mean_absolute_percentage_error(
            weekly_spend['total_revenue'].values, predicted)),
    }
    with open('models/mmm_params.json', 'w') as f:
        json.dump(mmm_params, f, indent=2)
    print("  Saved models/mmm_model.joblib")
    print("  Saved models/mmm_params.json")

    # Save data
    contributions_df.to_csv('data/channel_contributions.csv', index=False)
    optimal_df.to_csv('data/optimal_allocation.csv', index=False)
    print("  Saved data/channel_contributions.csv")
    print("  Saved data/optimal_allocation.csv")

    # Generate plots
    generate_plots(weekly_spend, predicted, contributions_df,
                   adstock_params, hill_params)

    # Compute and save marginal ROIs for Streamlit scenario planner
    marginal_rois = {}
    for col in SPEND_COLUMNS:
        name = CHANNEL_NAMES[col]
        current_spend = weekly_spend[col].sum()
        if current_spend > 0:
            contrib = contributions_df[contributions_df['channel'] == name]['contribution'].values
            if len(contrib) > 0:
                marginal_rois[name] = round(float(contrib[0] / current_spend), 3)
            else:
                marginal_rois[name] = 0
        else:
            marginal_rois[name] = 0

    mmm_params['marginal_rois'] = marginal_rois

    # Saturation levels
    saturation_levels = {}
    for col in SPEND_COLUMNS:
        name = CHANNEL_NAMES[col]
        current = weekly_spend[col].mean()
        alpha = hill_params[col]['alpha']
        K = hill_params[col]['K']
        sat = float(hill_saturation(np.array([current]), alpha, K)[0])
        saturation_levels[name] = round(sat * 100, 1)

    mmm_params['saturation_levels'] = saturation_levels

    with open('models/mmm_params.json', 'w') as f:
        json.dump(mmm_params, f, indent=2)

    print("\n" + "=" * 60)
    print("MMM COMPLETE")
    print("=" * 60)

    return model, scaler, contributions_df, optimal_df


if __name__ == '__main__':
    main()
