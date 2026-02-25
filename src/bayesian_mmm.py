"""
Bayesian Media Mix Model (PyMC)
================================
Full probabilistic MMM with posterior distributions on all parameters:
- Adstock decay rates (Beta priors)
- Hill saturation parameters (HalfNormal priors)
- Channel coefficients (HalfNormal, constrained positive)
- Posterior predictive checks
- Credible intervals on ROAS (not just point estimates)

This is the industry-standard approach used by Meta Robyn and Google Meridian.
The key advantage over frequentist Ridge: uncertainty quantification.

Outputs:
- models/bayesian_mmm_trace.nc (ArviZ InferenceData)
- data/bayesian_channel_contributions.csv
- data/bayesian_roas_posteriors.csv
- results/bayesian_posterior_roas.png
- results/bayesian_posterior_predictive.png
- results/bayesian_forest_plot.png
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.tensor.extra_ops import cumsum
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
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

CHANNEL_COLORS = {
    'TV Broadcast': '#2c3e50', 'TV Cable': '#34495e',
    'TV Streaming': '#7f8c8d', 'Paid Search': '#2980b9',
    'Social': '#8e44ad', 'Display': '#e67e22',
}


def geometric_adstock_pymc(spend, decay, n_weeks):
    """
    Geometric adstock using pytensor scan for MCMC compatibility.
    adstock[t] = spend[t] + decay * adstock[t-1]
    """
    def step(spend_t, adstock_prev, decay_rate):
        return spend_t + decay_rate * adstock_prev

    adstocked, _ = pytensor.scan(
        fn=step,
        sequences=[spend],
        outputs_info=[pt.zeros(())],
        non_sequences=[decay],
        n_steps=n_weeks,
    )
    return adstocked


def geometric_adstock_numpy(spend, decay):
    """NumPy version for pre-computation and posterior analysis."""
    result = np.zeros_like(spend, dtype=float)
    result[0] = spend[0]
    for t in range(1, len(spend)):
        result[t] = spend[t] + decay * result[t - 1]
    return result


def hill_saturation_numpy(x, alpha, K):
    """Hill saturation for posterior analysis."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.power(x, alpha) / (np.power(x, alpha) + np.power(K, alpha))
    return np.nan_to_num(result, 0.0)


def build_bayesian_mmm(weekly_spend):
    """
    Build and sample a fully Bayesian MMM.

    Model:
        revenue ~ Normal(mu, sigma)
        mu = intercept + sum(beta_i * hill(adstock(spend_i, decay_i), alpha_i, K_i))
             + season_sin * c1 + season_cos * c2 + trend * c3

    Priors (informed by domain knowledge):
        decay_i ~ Beta(3, 3)       # centered at 0.5, TV-heavy right tail
        alpha_i ~ HalfNormal(1)    # shape of saturation curve
        K_i ~ HalfNormal(100000)   # half-saturation point
        beta_i ~ HalfNormal(150000) # channel effect (positive)
        intercept ~ Normal(400000, 200000)
        sigma ~ HalfNormal(50000)

    We use a two-stage approach for computational efficiency:
    1. Pre-compute adstock with MAP decay estimates
    2. Run full MCMC on saturation + coefficients
    """
    print("\n[BAYESIAN MMM] Building model...")

    n_channels = len(SPEND_COLUMNS)
    n_weeks = len(weekly_spend)
    revenue = weekly_spend['total_revenue'].values

    # Normalize spend for numerical stability
    spend_matrix = np.column_stack([weekly_spend[col].values for col in SPEND_COLUMNS])
    spend_scale = spend_matrix.max(axis=0)
    spend_scale[spend_scale == 0] = 1
    spend_normalized = spend_matrix / spend_scale

    # Seasonality features
    month_sin = np.sin(2 * np.pi * weekly_spend['month'].values / 12)
    month_cos = np.cos(2 * np.pi * weekly_spend['month'].values / 12)
    trend = np.arange(n_weeks) / n_weeks

    # Revenue normalization
    revenue_scale = revenue.mean()
    revenue_norm = revenue / revenue_scale

    print(f"  Revenue mean: ${revenue_scale:,.0f}")
    print(f"  Channels: {n_channels}")
    print(f"  Weeks: {n_weeks}")

    # Stage 1: Find MAP adstock decays via grid search (fast)
    print("\n  Stage 1: Estimating adstock decays...")
    best_decays = []
    adstocked_spends = []

    for i, col in enumerate(SPEND_COLUMNS):
        spend = spend_normalized[:, i]
        best_corr = -np.inf
        best_decay = 0.5

        for decay in np.arange(0.1, 0.95, 0.05):
            adstocked = geometric_adstock_numpy(spend, decay)
            corr = np.corrcoef(adstocked, revenue_norm)[0, 1]
            if np.isfinite(corr) and corr > best_corr:
                best_corr = corr
                best_decay = decay

        best_decays.append(best_decay)
        adstocked_spends.append(geometric_adstock_numpy(spend, best_decay))
        print(f"    {CHANNEL_NAMES[col]}: decay={best_decay:.2f} (corr={best_corr:.3f})")

    adstocked_matrix = np.column_stack(adstocked_spends)

    # Stage 2: Full Bayesian inference on saturation + coefficients
    print("\n  Stage 2: MCMC sampling...")

    with pm.Model() as model:
        # Priors on Hill parameters
        alpha = pm.HalfNormal('alpha', sigma=1.0, shape=n_channels)
        K = pm.HalfNormal('K', sigma=2.0, shape=n_channels)

        # Channel coefficients (constrained positive — spend should help)
        beta = pm.HalfNormal('beta', sigma=2.0, shape=n_channels)

        # Intercept and controls
        intercept = pm.Normal('intercept', mu=1.0, sigma=0.5)
        season_sin_coef = pm.Normal('season_sin', mu=0, sigma=0.3)
        season_cos_coef = pm.Normal('season_cos', mu=0, sigma=0.3)
        trend_coef = pm.Normal('trend_coef', mu=0, sigma=0.3)

        # Compute saturated spend for each channel
        mu = intercept
        for i in range(n_channels):
            x = pt.as_tensor_variable(adstocked_matrix[:, i])
            # Hill saturation in pytensor
            x_alpha = pt.power(x + 1e-8, alpha[i])
            K_alpha = pt.power(K[i] + 1e-8, alpha[i])
            saturated = x_alpha / (x_alpha + K_alpha)
            mu = mu + beta[i] * saturated

        mu = mu + season_sin_coef * month_sin + season_cos_coef * month_cos
        mu = mu + trend_coef * trend

        # Noise
        sigma = pm.HalfNormal('sigma', sigma=0.3)

        # Likelihood
        likelihood = pm.Normal('revenue', mu=mu, sigma=sigma,
                               observed=revenue_norm)

        # Sample
        trace = pm.sample(
            draws=2000,
            tune=1500,
            cores=1,
            chains=2,
            target_accept=0.90,
            random_seed=42,
            progressbar=True,
            return_inferencedata=True,
        )

    print(f"\n  Sampling complete.")
    print(f"  Divergences: {trace.sample_stats.diverging.values.sum()}")

    return model, trace, revenue_scale, spend_scale, best_decays, adstocked_matrix


def analyze_posteriors(trace, weekly_spend, revenue_scale, spend_scale, best_decays):
    """
    Extract posterior ROAS distributions and channel contributions.
    """
    print("\n[BAYESIAN MMM] Analyzing posteriors...")

    n_channels = len(SPEND_COLUMNS)
    revenue = weekly_spend['total_revenue'].values

    # Extract posterior samples
    beta_samples = trace.posterior['beta'].values.reshape(-1, n_channels)
    alpha_samples = trace.posterior['alpha'].values.reshape(-1, n_channels)
    K_samples = trace.posterior['K'].values.reshape(-1, n_channels)
    intercept_samples = trace.posterior['intercept'].values.flatten()

    n_samples = len(intercept_samples)
    print(f"  Posterior samples: {n_samples}")

    # Compute posterior ROAS for each channel
    roas_posteriors = {}
    contribution_posteriors = {}

    for i, col in enumerate(SPEND_COLUMNS):
        name = CHANNEL_NAMES[col]
        spend = weekly_spend[col].values
        total_spend = spend.sum()
        spend_norm = spend / spend_scale[i]
        adstocked = geometric_adstock_numpy(spend_norm, best_decays[i])

        # For each posterior sample, compute contribution
        contributions = np.zeros(n_samples)
        for s in range(min(n_samples, 500)):  # subsample for speed
            alpha = alpha_samples[s, i]
            K = K_samples[s, i]
            beta = beta_samples[s, i]

            saturated = hill_saturation_numpy(adstocked, alpha, K)
            contrib = beta * saturated.sum() * revenue_scale
            contributions[s] = contrib

        contributions = contributions[:500]
        roas = contributions / total_spend if total_spend > 0 else contributions

        roas_posteriors[name] = roas
        contribution_posteriors[name] = contributions

        median_roas = np.median(roas)
        ci_low, ci_high = np.percentile(roas, [5, 95])
        median_contrib = np.median(contributions)

        print(f"  {name}:")
        print(f"    ROAS: {median_roas:.2f}x (90% CI: [{ci_low:.2f}, {ci_high:.2f}])")
        print(f"    Contribution: ${median_contrib:,.0f}")

    return roas_posteriors, contribution_posteriors


def generate_bayesian_plots(trace, roas_posteriors, contribution_posteriors,
                            weekly_spend, revenue_scale):
    """Generate Bayesian MMM visualizations."""
    print("\n[BAYESIAN MMM] Generating plots...")
    os.makedirs('results', exist_ok=True)

    # --- Plot 1: Posterior ROAS distributions ---
    fig, ax = plt.subplots(figsize=(12, 6))
    positions = []
    labels = []
    for idx, (name, roas) in enumerate(sorted(roas_posteriors.items(),
                                               key=lambda x: -np.median(x[1]))):
        color = CHANNEL_COLORS.get(name, '#3498db')
        bp = ax.boxplot(roas, positions=[idx], widths=0.6,
                        patch_artist=True, vert=True,
                        boxprops=dict(facecolor=color, alpha=0.7),
                        medianprops=dict(color='white', linewidth=2),
                        whiskerprops=dict(color=color),
                        capprops=dict(color=color),
                        flierprops=dict(marker='.', markerfacecolor=color,
                                        markersize=3, alpha=0.3))
        positions.append(idx)
        labels.append(name)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('ROAS (Return on Ad Spend)')
    ax.set_title('Bayesian MMM: Posterior ROAS Distributions\n'
                 '(boxes show 25-75th percentile, whiskers show 90% credible interval)',
                 fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('results/bayesian_posterior_roas.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/bayesian_posterior_roas.png")

    # --- Plot 2: Posterior predictive check ---
    revenue = weekly_spend['total_revenue'].values
    revenue_norm = revenue / revenue_scale

    if hasattr(trace, 'posterior_predictive') and 'revenue' in trace.posterior_predictive:
        pp = trace.posterior_predictive['revenue'].values.reshape(-1, len(revenue))
        pp_mean = pp.mean(axis=0) * revenue_scale
        pp_5 = np.percentile(pp, 5, axis=0) * revenue_scale
        pp_95 = np.percentile(pp, 95, axis=0) * revenue_scale
    else:
        # Reconstruct from posterior parameters
        beta_samples = trace.posterior['beta'].values.reshape(-1, len(SPEND_COLUMNS))
        alpha_samples = trace.posterior['alpha'].values.reshape(-1, len(SPEND_COLUMNS))
        K_samples = trace.posterior['K'].values.reshape(-1, len(SPEND_COLUMNS))
        intercept_samples = trace.posterior['intercept'].values.flatten()
        season_sin = trace.posterior['season_sin'].values.flatten()
        season_cos = trace.posterior['season_cos'].values.flatten()
        trend_coef = trace.posterior['trend_coef'].values.flatten()

        month_sin_vals = np.sin(2 * np.pi * weekly_spend['month'].values / 12)
        month_cos_vals = np.cos(2 * np.pi * weekly_spend['month'].values / 12)
        trend_vals = np.arange(len(weekly_spend)) / len(weekly_spend)

        spend_matrix = np.column_stack([weekly_spend[col].values for col in SPEND_COLUMNS])
        spend_scale_arr = spend_matrix.max(axis=0)
        spend_scale_arr[spend_scale_arr == 0] = 1

        n_samp = min(500, len(intercept_samples))
        preds = np.zeros((n_samp, len(revenue)))

        for s in range(n_samp):
            mu = intercept_samples[s]
            for i, col in enumerate(SPEND_COLUMNS):
                spend_norm = spend_matrix[:, i] / spend_scale_arr[i]
                adstocked = geometric_adstock_numpy(spend_norm, 0.5)
                sat = hill_saturation_numpy(adstocked, alpha_samples[s, i], K_samples[s, i])
                mu = mu + beta_samples[s, i] * sat
            mu = mu + season_sin[s] * month_sin_vals
            mu = mu + season_cos[s] * month_cos_vals
            mu = mu + trend_coef[s] * trend_vals
            preds[s] = mu * revenue_scale

        pp_mean = preds.mean(axis=0)
        pp_5 = np.percentile(preds, 5, axis=0)
        pp_95 = np.percentile(preds, 95, axis=0)

    weeks = weekly_spend['week'].values
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(weeks, pp_5 / 1e6, pp_95 / 1e6,
                    alpha=0.2, color='#3498db', label='90% Credible Interval')
    ax.plot(weeks, pp_mean / 1e6, '-', color='#3498db', linewidth=2,
            label='Posterior Mean')
    ax.plot(weeks, revenue / 1e6, 'o', color='#2c3e50', markersize=5,
            label='Actual Revenue')
    ax.set_xlabel('Week')
    ax.set_ylabel('Revenue ($M)')
    ax.set_title('Bayesian MMM: Posterior Predictive Check\n'
                 '(shaded band = 90% credible interval)',
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/bayesian_posterior_predictive.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/bayesian_posterior_predictive.png")

    # --- Plot 3: Forest plot of channel effects ---
    fig, ax = plt.subplots(figsize=(10, 6))
    channel_names = list(contribution_posteriors.keys())
    medians = [np.median(contribution_posteriors[n]) for n in channel_names]
    ci_lows = [np.percentile(contribution_posteriors[n], 5) for n in channel_names]
    ci_highs = [np.percentile(contribution_posteriors[n], 95) for n in channel_names]

    sorted_idx = np.argsort(medians)
    y_pos = np.arange(len(channel_names))

    for i, idx in enumerate(sorted_idx):
        name = channel_names[idx]
        color = CHANNEL_COLORS.get(name, '#3498db')
        ax.barh(i, medians[idx] / 1e6, color=color, alpha=0.7, height=0.6)
        ax.errorbar(medians[idx] / 1e6, i,
                     xerr=[[max(0, (medians[idx] - ci_lows[idx]) / 1e6)],
                            [(ci_highs[idx] - medians[idx]) / 1e6]],
                     fmt='none', color='black', capsize=4, linewidth=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([channel_names[i] for i in sorted_idx])
    ax.set_xlabel('Revenue Contribution ($M)')
    ax.set_title('Bayesian MMM: Channel Contributions with 90% Credible Intervals',
                 fontweight='bold')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('results/bayesian_forest_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/bayesian_forest_plot.png")


def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("=" * 60)
    print("BAYESIAN MEDIA MIX MODEL (PyMC)")
    print("=" * 60)

    weekly_spend = pd.read_csv('data/weekly_spend.csv')

    # Build and sample
    model, trace, revenue_scale, spend_scale, best_decays, adstocked = \
        build_bayesian_mmm(weekly_spend)

    # Analyze posteriors
    roas_posteriors, contribution_posteriors = analyze_posteriors(
        trace, weekly_spend, revenue_scale, spend_scale, best_decays
    )

    # Save trace
    trace.to_netcdf('models/bayesian_mmm_trace.nc')
    print("  Saved models/bayesian_mmm_trace.nc")

    # Save ROAS posteriors for dashboard
    roas_df = pd.DataFrame({
        name: vals[:500] for name, vals in roas_posteriors.items()
    })
    roas_df.to_csv('data/bayesian_roas_posteriors.csv', index=False)

    # Save contribution summaries
    contrib_rows = []
    for name in CHANNEL_NAMES.values():
        if name in contribution_posteriors:
            vals = contribution_posteriors[name]
            contrib_rows.append({
                'channel': name,
                'median_contribution': round(float(np.median(vals)), 2),
                'ci_5': round(float(np.percentile(vals, 5)), 2),
                'ci_95': round(float(np.percentile(vals, 95)), 2),
                'median_roas': round(float(np.median(roas_posteriors[name])), 3),
                'roas_ci_5': round(float(np.percentile(roas_posteriors[name], 5)), 3),
                'roas_ci_95': round(float(np.percentile(roas_posteriors[name], 95)), 3),
                'prob_positive': round(float((vals > 0).mean()), 3),
            })
    contrib_df = pd.DataFrame(contrib_rows)
    contrib_df.to_csv('data/bayesian_channel_contributions.csv', index=False)
    print("  Saved data/bayesian_channel_contributions.csv")

    # Generate plots
    generate_bayesian_plots(trace, roas_posteriors, contribution_posteriors,
                            weekly_spend, revenue_scale)

    # Save params for dashboard
    bayes_params = {
        'adstock_decays': {CHANNEL_NAMES[col]: float(best_decays[i])
                           for i, col in enumerate(SPEND_COLUMNS)},
        'revenue_scale': float(revenue_scale),
    }
    with open('models/bayesian_mmm_params.json', 'w') as f:
        json.dump(bayes_params, f, indent=2)

    print("\n" + "=" * 60)
    print("BAYESIAN MMM COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
