"""
Kalman Filter — Time-Varying Channel Effectiveness
===================================================
Uses a state-space model with Kalman filtering to estimate how channel
ROAS changes over time — not static coefficients like Ridge/Bayesian MMM.

WHY this matters:
- Real ad effectiveness drifts: audience fatigue, competitive dynamics,
  seasonality of consumer intent, creative wear-out
- Standard MMM assumes FIXED coefficients for the entire year
- Kalman filter tracks the latent state (true ROAS) as it evolves
- Widely used in finance (Kalman 1960) but rare in marketing — impressive

Model:
  State equation:   β_t = β_{t-1} + η_t       (random walk dynamics)
  Observation:      y_t = X_t @ β_t + ε_t      (regression with TV betas)

  Where:
  - β_t = vector of time-varying channel coefficients
  - X_t = channel spend features for week t
  - η_t ~ N(0, Q) = state noise (how fast coefficients drift)
  - ε_t ~ N(0, R) = observation noise

Pipeline:
1. Build feature matrix (same as MMM)
2. Initialize Kalman filter with prior state
3. Forward pass: filter (estimate current state given past)
4. Backward pass: smoother (revise estimates given future data)
5. Extract time-varying ROAS with confidence bands

Outputs:
- data/kalman_tvp_coefficients.csv
- data/kalman_tvp_roas.csv
- results/kalman_tvp_coefficients.png
- results/kalman_tvp_roas_evolution.png
- results/kalman_vs_static.png
"""

import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
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


def prepare_features(weekly):
    """Build feature matrix for time-varying regression."""
    features = {}
    for col in SPEND_COLUMNS:
        features[col] = weekly[col].values

    features['intercept'] = np.ones(len(weekly))
    features['month_sin'] = np.sin(2 * np.pi * weekly['month'].values / 12)
    features['month_cos'] = np.cos(2 * np.pi * weekly['month'].values / 12)

    feature_names = list(CHANNEL_NAMES.values()) + ['Intercept', 'Month Sin', 'Month Cos']

    X = np.column_stack(list(features.values()))
    y = weekly['total_revenue'].values

    return X, y, feature_names


def run_kalman_tvp(X, y, feature_names):
    """
    Fit time-varying parameter model using Kalman filter.
    """
    print("\n=== Kalman Filter: Time-Varying Parameters ===")

    n_obs, n_features = X.shape

    # Normalize features for better conditioning
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_mean, y_std = y.mean(), y.std()
    y_scaled = (y - y_mean) / y_std

    # Initialize with OLS estimates
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y_scaled)
    initial_state = ridge.coef_

    # State dynamics: how fast can coefficients change?
    # Small Q = slow drift, large Q = fast changes
    state_noise = 0.01  # controls drift speed
    obs_noise = 0.1     # observation noise

    # Build observation matrices (one per timestep)
    # y_t = X_t @ beta_t => observation matrix is X_t
    observation_matrices = X_scaled.reshape(n_obs, 1, n_features)

    # Kalman filter
    kf = KalmanFilter(
        n_dim_state=n_features,
        n_dim_obs=1,
        transition_matrices=np.eye(n_features),  # random walk
        observation_matrices=observation_matrices,
        transition_covariance=state_noise * np.eye(n_features),
        observation_covariance=np.array([[obs_noise]]),
        initial_state_mean=initial_state,
        initial_state_covariance=0.1 * np.eye(n_features),
    )

    # EM algorithm to learn noise parameters
    print("  Running EM algorithm to learn noise parameters...")
    kf = kf.em(y_scaled.reshape(-1, 1), n_iter=10)

    # Smooth (forward-backward pass)
    print("  Running Kalman smoother...")
    smoothed_states, smoothed_covariances = kf.smooth(y_scaled.reshape(-1, 1))

    # Filtered states (forward only — causal)
    filtered_states, filtered_covariances = kf.filter(y_scaled.reshape(-1, 1))

    # Compute predictions
    y_pred_scaled = np.array([
        X_scaled[t] @ smoothed_states[t] for t in range(n_obs)
    ])
    y_pred = y_pred_scaled * y_std + y_mean

    r2 = r2_score(y, y_pred)
    print(f"  Smoothed R²: {r2:.4f}")

    # Convert coefficients back to original scale
    # β_orig = β_scaled * (y_std / x_std)
    x_stds = scaler.scale_
    coef_original = smoothed_states * (y_std / x_stds[np.newaxis, :])

    # Confidence bands (±2 std from covariance)
    state_stds = np.sqrt(np.array([
        np.diag(smoothed_covariances[t]) for t in range(n_obs)
    ]))
    coef_upper = (smoothed_states + 2 * state_stds) * (y_std / x_stds[np.newaxis, :])
    coef_lower = (smoothed_states - 2 * state_stds) * (y_std / x_stds[np.newaxis, :])

    return (smoothed_states, coef_original, coef_upper, coef_lower,
            y_pred, r2, scaler, y_mean, y_std)


def compute_time_varying_roas(weekly, coef_original, feature_names):
    """Compute time-varying ROAS from Kalman coefficients."""
    print("\n=== Time-Varying ROAS ===")

    roas_data = []
    for t in range(len(weekly)):
        for i, col in enumerate(SPEND_COLUMNS):
            spend = weekly[col].values[t]
            coef = coef_original[t, i]
            # ROAS = marginal revenue / marginal cost
            # Since coef is d(revenue)/d(spend), ROAS ≈ coef * mean_spend / mean_revenue
            roas = coef  # already in $ revenue per $ spend units

            roas_data.append({
                'week': t + 1,
                'month': weekly['month'].values[t],
                'channel': CHANNEL_NAMES[col],
                'coefficient': coef,
                'spend': spend,
                'roas': roas,
            })

    roas_df = pd.DataFrame(roas_data)

    # Summary
    print("\n  Time-varying ROAS summary (mean ± std across weeks):")
    for ch in CHANNEL_NAMES.values():
        ch_data = roas_df[roas_df['channel'] == ch]['coefficient']
        print(f"    {ch:20s}: {ch_data.mean():8.2f} ± {ch_data.std():.2f}"
              f"  range [{ch_data.min():.2f}, {ch_data.max():.2f}]")

    return roas_df


def compare_static_vs_dynamic(weekly, coef_original, y_pred, feature_names):
    """Compare static (Ridge) vs dynamic (Kalman) coefficients."""
    print("\n=== Static vs Dynamic Comparison ===")

    X, y, _ = prepare_features(weekly)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    static_pred = ridge.predict(X_scaled)

    static_r2 = r2_score(y, static_pred)
    dynamic_r2 = r2_score(y, y_pred)

    print(f"  Static (Ridge) R²:  {static_r2:.4f}")
    print(f"  Dynamic (Kalman) R²: {dynamic_r2:.4f}")
    print(f"  Improvement: +{(dynamic_r2 - static_r2)*100:.2f} pp")

    # Static coefficients (unscaled)
    static_coefs = ridge.coef_ * (y.std() / scaler.scale_)

    return static_coefs, static_r2, dynamic_r2


def plot_results(weekly, coef_original, coef_upper, coef_lower,
                 y_pred, static_coefs, feature_names, roas_df,
                 static_r2, dynamic_r2):
    """Generate visualization plots."""
    os.makedirs('results', exist_ok=True)
    y = weekly['total_revenue'].values
    weeks = np.arange(1, len(weekly) + 1)

    # 1. Time-varying coefficients
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    channel_colors = ['#2c3e50', '#34495e', '#7f8c8d', '#2980b9', '#8e44ad', '#e67e22']

    for i, (col, name) in enumerate(CHANNEL_NAMES.items()):
        ax = axes[i // 3][i % 3]
        color = channel_colors[i]

        ax.plot(weeks, coef_original[:, i], color=color, linewidth=2, label='Kalman (dynamic)')
        ax.fill_between(weeks, coef_lower[:, i], coef_upper[:, i],
                        alpha=0.15, color=color)
        ax.axhline(y=static_coefs[i], color=color, linestyle='--',
                   alpha=0.5, label=f'Ridge (static): {static_coefs[i]:.2f}')
        ax.set_title(name, fontsize=11)
        ax.set_xlabel('Week')
        ax.set_ylabel('Coefficient')
        ax.legend(fontsize=7)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.2)

    plt.suptitle('Kalman Filter: Time-Varying Channel Coefficients\n'
                 '(Shaded = 95% confidence band, dashed = static Ridge estimate)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('results/kalman_tvp_coefficients.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Time-varying ROAS evolution
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (col, name) in enumerate(CHANNEL_NAMES.items()):
        ch_data = roas_df[roas_df['channel'] == name]
        ax.plot(ch_data['week'], ch_data['coefficient'],
                color=channel_colors[i], linewidth=2, label=name, alpha=0.8)

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Marginal Effect ($/$ spend)', fontsize=12)
    ax.set_title('How Channel Effectiveness Evolves Over Time\n'
                 '(Each line shows one channel\'s marginal revenue per dollar spent)',
                 fontsize=13)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig('results/kalman_tvp_roas_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Static vs dynamic fit
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    X, _, _ = prepare_features(weekly)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge = Ridge(alpha=1.0).fit(X_scaled, y)
    static_pred = ridge.predict(X_scaled)

    ax = axes[0]
    ax.plot(weeks, y / 1e6, 'o-', color='#2c3e50', markersize=3,
            linewidth=1, label='Actual')
    ax.plot(weeks, static_pred / 1e6, '--', color='#e74c3c',
            linewidth=1.5, label=f'Static Ridge (R²={static_r2:.3f})')
    ax.plot(weeks, y_pred / 1e6, '-', color='#27ae60',
            linewidth=1.5, label=f'Kalman TVP (R²={dynamic_r2:.3f})')
    ax.set_xlabel('Week')
    ax.set_ylabel('Revenue ($M)')
    ax.set_title('Prediction: Static vs Time-Varying Model')
    ax.legend(fontsize=9)

    ax = axes[1]
    static_resid = (y - static_pred) / 1e3
    dynamic_resid = (y - y_pred) / 1e3
    ax.scatter(weeks, static_resid, color='#e74c3c', alpha=0.5, s=30, label='Static residuals')
    ax.scatter(weeks, dynamic_resid, color='#27ae60', alpha=0.5, s=30, label='Dynamic residuals')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Week')
    ax.set_ylabel('Residual ($K)')
    ax.set_title('Residuals: Dynamic Model Has Less Structure')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('results/kalman_vs_static.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Plots saved to results/kalman_*.png")


def main():
    print("=" * 60)
    print("KALMAN FILTER — Time-Varying Channel Effectiveness")
    print("=" * 60)

    weekly = pd.read_csv('data/weekly_spend.csv')
    X, y, feature_names = prepare_features(weekly)

    # Fit Kalman TVP
    (smoothed_states, coef_original, coef_upper, coef_lower,
     y_pred, dynamic_r2, scaler, y_mean, y_std) = run_kalman_tvp(X, y, feature_names)

    # Time-varying ROAS
    roas_df = compute_time_varying_roas(weekly, coef_original, feature_names)

    # Compare with static
    static_coefs, static_r2, _ = compare_static_vs_dynamic(
        weekly, coef_original, y_pred, feature_names)

    # Save results
    os.makedirs('data', exist_ok=True)

    # Coefficients over time
    coef_df = pd.DataFrame(coef_original[:, :len(SPEND_COLUMNS)],
                           columns=list(CHANNEL_NAMES.values()))
    coef_df.insert(0, 'week', range(1, len(weekly) + 1))
    coef_df.to_csv('data/kalman_tvp_coefficients.csv', index=False)

    roas_df.to_csv('data/kalman_tvp_roas.csv', index=False)

    summary = {
        'static_r2': float(static_r2),
        'dynamic_r2': float(dynamic_r2),
        'improvement_pp': float((dynamic_r2 - static_r2) * 100),
    }
    with open('data/kalman_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot
    plot_results(weekly, coef_original, coef_upper, coef_lower,
                 y_pred, static_coefs, feature_names, roas_df,
                 static_r2, dynamic_r2)

    print("\n✓ Kalman filter analysis complete")
    return coef_df, roas_df, summary


if __name__ == '__main__':
    main()
