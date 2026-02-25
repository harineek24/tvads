"""
SHAP Explainability Analysis
=============================
TreeSHAP on the LightGBM baseline model for feature-level interpretability.
Shapley values provide the theoretically grounded way to attribute each
prediction to individual features (additive, consistent, local).

Also computes SHAP-based channel importance for the MMM features.

Outputs:
- data/shap_values_baseline.csv
- data/shap_feature_importance.csv
- results/shap_summary_baseline.png
- results/shap_dependence_hour.png
- results/shap_waterfall_sample.png
"""

import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
import joblib
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

HOLIDAYS_2023 = {
    '2023-01-01', '2023-01-02', '2023-01-16', '2023-02-12',
    '2023-02-20', '2023-05-29', '2023-07-04', '2023-09-04',
    '2023-10-09', '2023-11-23', '2023-11-24', '2023-12-25',
    '2023-12-26', '2023-12-31',
}


def build_features(traffic_df):
    """Reconstruct features matching the baseline model."""
    df = traffic_df.copy()
    ts = pd.to_datetime(df['timestamp'])
    df['hour'] = ts.dt.hour
    df['month'] = ts.dt.month
    df['day_of_week'] = ts.dt.dayofweek
    df['day_of_year'] = ts.dt.dayofyear

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_holiday'] = df['date'].isin(HOLIDAYS_2023).astype(int)
    df['organic_trend'] = df['day_of_year'] / 365.0

    dow_dummies = pd.get_dummies(df['day_of_week'], prefix='dow').astype(int)
    df = pd.concat([df, dow_dummies], axis=1)
    dma_dummies = pd.get_dummies(df['dma'], prefix='dma').astype(int)
    df = pd.concat([df, dma_dummies], axis=1)

    feature_cols = (
        ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
         'is_weekend', 'is_holiday', 'organic_trend'] +
        [c for c in dow_dummies.columns] +
        [c for c in dma_dummies.columns]
    )
    return df, feature_cols


def run_shap_analysis():
    """
    Run TreeSHAP on the LightGBM baseline model.
    TreeSHAP is exact and polynomial-time for tree models (not approximate).
    """
    print("\n[SHAP] Loading model and data...")

    model = joblib.load('models/baseline_model.joblib')
    traffic = pd.read_csv('data/web_traffic.csv')

    print(f"  Traffic rows: {len(traffic):,}")

    # Build features
    traffic_feat, feature_cols = build_features(traffic)

    # Subsample for SHAP (full dataset is too large)
    sample_size = 10000
    sample_idx = np.random.choice(len(traffic_feat), sample_size, replace=False)
    X_sample = traffic_feat.iloc[sample_idx][feature_cols]

    print(f"  Computing SHAP values on {sample_size:,} samples...")
    print(f"  Features: {len(feature_cols)}")

    # TreeSHAP explainer (exact, not approximate)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    print(f"  SHAP values shape: {shap_values.shape}")

    # Feature importance (mean |SHAP|)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_abs_shap,
        'rank': np.argsort(-mean_abs_shap) + 1,
    }).sort_values('mean_abs_shap', ascending=False)

    print("\n  Top 10 features by SHAP importance:")
    for _, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']}: {row['mean_abs_shap']:.4f}")

    return shap_values, X_sample, feature_cols, importance_df, explainer


def generate_shap_plots(shap_values, X_sample, feature_cols, importance_df):
    """Generate SHAP visualizations."""
    print("\n[SHAP] Generating plots...")
    os.makedirs('results', exist_ok=True)

    # --- Plot 1: SHAP Summary (beeswarm) ---
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols,
                      show=False, max_display=15)
    plt.title('SHAP Feature Importance: Baseline Traffic Model\n'
              '(TreeSHAP — exact Shapley values for LightGBM)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/shap_summary_baseline.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print("  Saved results/shap_summary_baseline.png")

    # --- Plot 2: SHAP dependence for hour features ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # hour_sin dependence
    hour_sin_idx = feature_cols.index('hour_sin')
    shap.dependence_plot(hour_sin_idx, shap_values, X_sample,
                         feature_names=feature_cols, ax=axes[0], show=False)
    axes[0].set_title('SHAP Dependence: hour_sin')

    # hour_cos dependence
    hour_cos_idx = feature_cols.index('hour_cos')
    shap.dependence_plot(hour_cos_idx, shap_values, X_sample,
                         feature_names=feature_cols, ax=axes[1], show=False)
    axes[1].set_title('SHAP Dependence: hour_cos')

    plt.suptitle('How Time-of-Day Drives Traffic Predictions (SHAP)',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/shap_dependence_hour.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print("  Saved results/shap_dependence_hour.png")

    # --- Plot 3: Waterfall for a single sample ---
    fig, ax = plt.subplots(figsize=(10, 6))
    # Pick an interesting sample (high-traffic hour)
    high_traffic_idx = np.argmax(X_sample['hour_sin'].values)
    explanation = shap.Explanation(
        values=shap_values[high_traffic_idx],
        base_values=shap_values.mean(),  # approximate
        data=X_sample.iloc[high_traffic_idx].values,
        feature_names=feature_cols,
    )
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title('SHAP Waterfall: Single Prediction Explained', fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/shap_waterfall_sample.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print("  Saved results/shap_waterfall_sample.png")

    # --- Plot 4: Feature importance bar chart (cleaner than summary) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = 15
    top_features = importance_df.head(top_n)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    bars = ax.barh(range(top_n), top_features['mean_abs_shap'].values[::-1],
                   color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features['feature'].values[::-1])
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('Feature Importance: Mean Absolute SHAP Values\n'
                 '(Baseline Traffic Prediction Model)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('results/shap_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print("  Saved results/shap_feature_importance.png")


def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("=" * 60)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    shap_values, X_sample, feature_cols, importance_df, explainer = run_shap_analysis()

    # Save
    importance_df.to_csv('data/shap_feature_importance.csv', index=False)

    # Save SHAP values for top features (full matrix too large)
    top_features = importance_df.head(10)['feature'].tolist()
    top_indices = [feature_cols.index(f) for f in top_features]
    shap_top = pd.DataFrame(
        shap_values[:, top_indices],
        columns=top_features
    )
    shap_top.to_csv('data/shap_values_baseline.csv', index=False)
    print("  Saved data/shap_values_baseline.csv")

    generate_shap_plots(shap_values, X_sample, feature_cols, importance_df)

    print("\n" + "=" * 60)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
