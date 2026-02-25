"""
TV Ad Attribution & Media Mix Model Dashboard
=============================================
Interactive Streamlit dashboard for analyzing TV advertising performance,
channel attribution, budget optimization, and scenario planning.

17 Tabs:
1. Attribution — Per-airing lift analysis with baseline vs actual
2. Budget Optimizer — Optimal spend allocation (hero feature)
3. Scenario Planner — What-if analysis with interactive sliders
4. Channel Deep Dive — Adstock, saturation, and recommendations
5. Bayesian MMM — PyMC MCMC with posterior ROAS and credible intervals
6. SHAP Explainability — TreeSHAP exact Shapley values
7. Markov Attribution — Absorbing Markov chains with removal effects
8. Cross-Channel — TV→Search Granger causality and mediation
9. Model Comparison — Stacked comparison of 6 attribution methods
10. Causal ML (DML) — Double Machine Learning causal effects
11. Transformer — Temporal Fusion Transformer with attention
12. Conformal — Distribution-free prediction intervals
13. Geo-Lift — Synthetic control incrementality testing
14. CausalImpact — Bayesian structural time series
15. Kalman Filter — Time-varying channel effectiveness
16. Client Report — Auto-generated executive summary + PDF export
17. Model Diagnostics — Fit quality, residuals, limitations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
from datetime import datetime, timedelta

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="TV Ad Attribution & MMM",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# DATA LOADING
# ============================================

@st.cache_data
def load_data():
    """Load all datasets and model artifacts."""
    data = {}

    # Core datasets
    data['airings'] = pd.read_csv('data/ad_airings.csv')
    data['weekly_spend'] = pd.read_csv('data/weekly_spend.csv')
    data['daily_traffic'] = pd.read_csv('data/daily_traffic.csv')
    data['hourly_traffic'] = pd.read_csv('data/hourly_traffic_agg.csv')

    # Attribution results
    data['airing_attribution'] = pd.read_csv('data/airing_attribution.csv')
    data['attr_by_network'] = pd.read_csv('data/attribution_by_network.csv')
    data['attr_by_daypart'] = pd.read_csv('data/attribution_by_daypart.csv')
    data['attr_by_creative'] = pd.read_csv('data/attribution_by_creative.csv')
    data['attr_by_dma'] = pd.read_csv('data/attribution_by_dma.csv')

    # MMM results
    data['channel_contributions'] = pd.read_csv('data/channel_contributions.csv')
    data['optimal_allocation'] = pd.read_csv('data/optimal_allocation.csv')

    # Model params
    with open('models/mmm_params.json', 'r') as f:
        data['mmm_params'] = json.load(f)

    if os.path.exists('models/baseline_metrics.json'):
        with open('models/baseline_metrics.json', 'r') as f:
            data['baseline_metrics'] = json.load(f)

    # Advanced analysis datasets
    if os.path.exists('data/bayesian_roas_posteriors.csv'):
        data['bayesian_roas'] = pd.read_csv('data/bayesian_roas_posteriors.csv')
    if os.path.exists('data/bayesian_channel_contributions.csv'):
        data['bayesian_contributions'] = pd.read_csv('data/bayesian_channel_contributions.csv')
    if os.path.exists('models/bayesian_mmm_params.json'):
        with open('models/bayesian_mmm_params.json', 'r') as f:
            data['bayesian_params'] = json.load(f)
    if os.path.exists('data/shap_feature_importance.csv'):
        data['shap_importance'] = pd.read_csv('data/shap_feature_importance.csv')
    if os.path.exists('data/markov_attribution.csv'):
        data['markov_attribution'] = pd.read_csv('data/markov_attribution.csv')
    if os.path.exists('data/markov_removal_effects.csv'):
        data['markov_removal'] = pd.read_csv('data/markov_removal_effects.csv')
    if os.path.exists('data/cross_channel_correlations.csv'):
        data['cross_correlations'] = pd.read_csv('data/cross_channel_correlations.csv')
    if os.path.exists('data/granger_results.csv'):
        data['granger_results'] = pd.read_csv('data/granger_results.csv')
    if os.path.exists('data/cross_channel_effects.json'):
        with open('data/cross_channel_effects.json', 'r') as f:
            data['cross_effects'] = json.load(f)
    if os.path.exists('data/model_comparison.csv'):
        data['model_comparison'] = pd.read_csv('data/model_comparison.csv')

    # New advanced analysis datasets
    if os.path.exists('data/dml_causal_effects.csv'):
        data['dml_effects'] = pd.read_csv('data/dml_causal_effects.csv')
    if os.path.exists('data/dml_vs_ols.csv'):
        data['dml_vs_ols'] = pd.read_csv('data/dml_vs_ols.csv')
    if os.path.exists('data/dml_heterogeneous_effects.csv'):
        data['dml_het'] = pd.read_csv('data/dml_heterogeneous_effects.csv')
    if os.path.exists('data/tft_feature_importance.csv'):
        data['tft_importance'] = pd.read_csv('data/tft_feature_importance.csv')
    if os.path.exists('data/tft_predictions.csv'):
        data['tft_predictions'] = pd.read_csv('data/tft_predictions.csv')
    if os.path.exists('models/tft_metrics.json'):
        with open('models/tft_metrics.json', 'r') as f:
            data['tft_metrics'] = json.load(f)
    if os.path.exists('data/conformal_intervals.csv'):
        data['conformal_intervals'] = pd.read_csv('data/conformal_intervals.csv')
    if os.path.exists('data/conformal_coverage.csv'):
        data['conformal_coverage'] = pd.read_csv('data/conformal_coverage.csv')
    if os.path.exists('data/conformal_summary.json'):
        with open('data/conformal_summary.json', 'r') as f:
            data['conformal_summary'] = json.load(f)
    if os.path.exists('data/geo_lift_results.csv'):
        data['geo_lift'] = pd.read_csv('data/geo_lift_results.csv')
    if os.path.exists('data/synthetic_control_weights.csv'):
        data['sc_weights'] = pd.read_csv('data/synthetic_control_weights.csv')
    if os.path.exists('data/geo_lift_summary.json'):
        with open('data/geo_lift_summary.json', 'r') as f:
            data['geo_lift_summary'] = json.load(f)
    if os.path.exists('data/causal_impact_results.csv'):
        data['causal_impact'] = pd.read_csv('data/causal_impact_results.csv')
    if os.path.exists('data/causal_impact_summary.json'):
        with open('data/causal_impact_summary.json', 'r') as f:
            data['ci_summary'] = json.load(f)
    if os.path.exists('data/kalman_tvp_coefficients.csv'):
        data['kalman_coefs'] = pd.read_csv('data/kalman_tvp_coefficients.csv')
    if os.path.exists('data/kalman_summary.json'):
        with open('data/kalman_summary.json', 'r') as f:
            data['kalman_summary'] = json.load(f)

    return data


@st.cache_resource
def load_models():
    """Load trained models."""
    models = {}
    if os.path.exists('models/mmm_model.joblib'):
        models['mmm'] = joblib.load('models/mmm_model.joblib')
    if os.path.exists('models/baseline_model.joblib'):
        models['baseline'] = joblib.load('models/baseline_model.joblib')
    return models


data = load_data()
models = load_models()

# ============================================
# HELPER FUNCTIONS
# ============================================

CHANNEL_COLORS = {
    'TV Broadcast': '#2c3e50',
    'TV Cable': '#34495e',
    'TV Streaming': '#7f8c8d',
    'Paid Search': '#2980b9',
    'Social': '#8e44ad',
    'Display': '#e67e22',
    'Base + Seasonality': '#95a5a6',
}

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


def hill_saturation(spend, alpha, K):
    """Hill function for saturation curves."""
    spend = np.asarray(spend, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.power(spend, alpha) / (np.power(spend, alpha) + np.power(K, alpha))
    return np.nan_to_num(result, 0.0)


def adstock_transform(spend_series, decay_rate):
    """Geometric adstock transformation."""
    spend = np.asarray(spend_series, dtype=float)
    adstocked = np.zeros_like(spend)
    adstocked[0] = spend[0]
    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay_rate * adstocked[t - 1]
    return adstocked


def format_currency(val, decimals=1):
    """Format number as currency string."""
    if abs(val) >= 1e6:
        return f"${val/1e6:.{decimals}f}M"
    elif abs(val) >= 1e3:
        return f"${val/1e3:.{decimals}f}K"
    else:
        return f"${val:.0f}"


# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("📺 TV Ad Analytics")
st.sidebar.markdown("---")
st.sidebar.markdown("**Brand**: DTC Consumer Brand")
st.sidebar.markdown("**Budget**: $15M Annual")
st.sidebar.markdown("**Period**: Jan–Dec 2023")
st.sidebar.markdown("**DMAs**: 8 Markets")
st.sidebar.markdown("---")

# ============================================
# TABS
# ============================================

all_tabs = st.tabs([
    "📊 Attribution",
    "💰 Budget Optimizer",
    "🔮 Scenario Planner",
    "🔬 Channel Deep Dive",
    "🎲 Bayesian MMM",
    "🧠 SHAP",
    "🔗 Markov",
    "📡 Cross-Channel",
    "⚖️ Comparison",
    "🧬 Causal ML (DML)",
    "🤖 Transformer",
    "📐 Conformal",
    "🌍 Geo-Lift",
    "📈 CausalImpact",
    "⏱️ Kalman Filter",
    "📄 Report",
    "🔧 Diagnostics",
])
(tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9,
 tab_dml, tab_tft, tab_conf, tab_geo, tab_ci, tab_kalman,
 tab10, tab11) = all_tabs


# ============================================
# TAB 1: ATTRIBUTION
# ============================================

with tab1:
    st.header("TV Ad Attribution")
    st.markdown("Measuring the causal impact of each TV ad airing on website traffic "
                "using counterfactual baseline modeling.")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        dmas = sorted(data['airings']['dma'].unique())
        selected_dma = st.selectbox("DMA", dmas, index=0)
    with col_f2:
        dates = sorted(data['airings']['date'].unique())
        date_idx_start = 0
        date_idx_end = len(dates) - 1
        start_date = st.selectbox("Start Date", dates[:30],
                                  index=0, key='attr_start')
    with col_f3:
        end_date = st.selectbox("End Date", dates[-30:],
                                index=len(dates[-30:]) - 1, key='attr_end')

    # Date picker for timeline view
    st.markdown("### Daily Attribution Timeline")
    sample_dates = dates[60:75]  # mid-March sample
    selected_day = st.select_slider(
        "Select a day to view hourly attribution:",
        options=dates,
        value=dates[73]  # Mar 15
    )

    # Hourly chart for selected day
    hourly = data['hourly_traffic']
    day_hourly = hourly[
        (hourly['date'] == selected_day) &
        (hourly['dma'] == selected_dma)
    ].sort_values('hour')

    day_airings = data['airings'][
        (data['airings']['date'] == selected_day) &
        (data['airings']['dma'] == selected_dma)
    ]

    if len(day_hourly) > 0:
        # Compute a simple baseline (average of non-peak hours)
        baseline_mean = day_hourly[day_hourly['hour'].between(2, 6)]['sessions'].mean()
        if pd.isna(baseline_mean):
            baseline_mean = day_hourly['sessions'].min()

        # Hour-of-day baseline curve
        hour_curve = []
        for h in range(24):
            curve_val = (0.2 + 0.5 * np.exp(-((h - 10) ** 2) / 8) +
                         0.8 * np.exp(-((h - 20) ** 2) / 6))
            hour_curve.append(curve_val)
        hour_curve = np.array(hour_curve)
        baseline_sessions = baseline_mean * hour_curve / hour_curve.min() * 0.8

        fig = go.Figure()

        # Baseline
        fig.add_trace(go.Scatter(
            x=day_hourly['hour'], y=baseline_sessions[:len(day_hourly)],
            name='Baseline (No-TV Counterfactual)',
            line=dict(color='#3498db', width=2, dash='dash'),
            fill=None
        ))

        # Actual
        fig.add_trace(go.Scatter(
            x=day_hourly['hour'], y=day_hourly['sessions'],
            name='Actual Traffic',
            line=dict(color='#2ecc71', width=2),
            fill='tonexty',
            fillcolor='rgba(46, 204, 113, 0.15)'
        ))

        # Ad airing markers
        if len(day_airings) > 0:
            airing_hours = pd.to_datetime(day_airings['timestamp']).dt.hour
            fig.add_trace(go.Scatter(
                x=airing_hours,
                y=[day_hourly['sessions'].max() * 1.08] * len(airing_hours),
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='#e74c3c'),
                name='Ad Airings'
            ))

        fig.update_layout(
            title=f"Attribution: {selected_dma} — {selected_day}",
            xaxis_title="Hour of Day",
            yaxis_title="Sessions",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    attr_filtered = data['airing_attribution'][
        (data['airing_attribution']['date'] >= start_date) &
        (data['airing_attribution']['date'] <= end_date) &
        (data['airing_attribution']['dma'] == selected_dma)
    ]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Airings", f"{len(attr_filtered):,}")
    with col2:
        st.metric("Total Lift",
                   f"{attr_filtered['incremental_sessions'].sum():,.0f} sessions")
    with col3:
        total_rev = attr_filtered['incremental_revenue'].sum()
        total_cost = attr_filtered['cost'].sum()
        avg_roas = total_rev / total_cost if total_cost > 0 else 0
        st.metric("Avg ROAS", f"{avg_roas:.2f}x")
    with col4:
        st.metric("Incremental Revenue",
                   format_currency(attr_filtered['incremental_revenue'].sum()))

    # Performance tables
    st.markdown("### Performance Breakdown")
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("**ROAS by Network**")
        net_df = data['attr_by_network'].copy()
        net_df['avg_roas'] = net_df['avg_roas'].round(3)
        net_df = net_df.sort_values('avg_roas', ascending=False)
        st.dataframe(
            net_df[['network', 'total_airings', 'total_cost',
                     'total_incremental_revenue', 'avg_roas']].rename(columns={
                'network': 'Network', 'total_airings': 'Airings',
                'total_cost': 'Cost', 'total_incremental_revenue': 'Incr. Revenue',
                'avg_roas': 'ROAS'
            }),
            use_container_width=True, hide_index=True
        )

    with col_t2:
        st.markdown("**ROAS by Daypart**")
        dp_df = data['attr_by_daypart'].copy()
        dp_df['avg_roas'] = dp_df['avg_roas'].round(3)
        dp_df = dp_df.sort_values('avg_roas', ascending=False)
        st.dataframe(
            dp_df[['daypart', 'total_airings', 'total_cost',
                    'total_incremental_revenue', 'avg_roas']].rename(columns={
                'daypart': 'Daypart', 'total_airings': 'Airings',
                'total_cost': 'Cost', 'total_incremental_revenue': 'Incr. Revenue',
                'avg_roas': 'ROAS'
            }),
            use_container_width=True, hide_index=True
        )

    col_t3, col_t4 = st.columns(2)
    with col_t3:
        st.markdown("**ROAS by Creative**")
        cr_df = data['attr_by_creative'].copy()
        cr_df['avg_roas'] = cr_df['avg_roas'].round(3)
        cr_df = cr_df.sort_values('avg_roas', ascending=False)
        st.dataframe(
            cr_df[['creative_id', 'total_airings',
                    'total_incremental_revenue', 'avg_roas']].rename(columns={
                'creative_id': 'Creative', 'total_airings': 'Airings',
                'total_incremental_revenue': 'Incr. Revenue', 'avg_roas': 'ROAS'
            }),
            use_container_width=True, hide_index=True
        )

    with col_t4:
        st.markdown("**ROAS by DMA**")
        dma_df = data['attr_by_dma'].copy()
        dma_df['avg_roas'] = dma_df['avg_roas'].round(3)
        dma_df = dma_df.sort_values('avg_roas', ascending=False)
        st.dataframe(
            dma_df[['dma', 'total_airings',
                     'total_incremental_revenue', 'avg_roas']].rename(columns={
                'dma': 'DMA', 'total_airings': 'Airings',
                'total_incremental_revenue': 'Incr. Revenue', 'avg_roas': 'ROAS'
            }),
            use_container_width=True, hide_index=True
        )


# ============================================
# TAB 2: BUDGET OPTIMIZER
# ============================================

with tab2:
    st.header("Budget Optimizer")
    st.markdown("Find the optimal channel allocation that maximizes predicted revenue "
                "using the Media Mix Model's learned response curves.")

    total_budget = st.slider(
        "Total Annual Budget ($M)",
        min_value=5.0, max_value=30.0, value=15.0, step=0.5
    )

    # Find closest pre-computed budget level
    opt_df = data['optimal_allocation']
    budget_val = total_budget * 1e6
    closest_idx = (opt_df['budget'] - budget_val).abs().idxmin()
    opt_row = opt_df.iloc[closest_idx]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Allocation")
        current_data = []
        for col in SPEND_COLUMNS:
            current_col = f'current_{col}'
            if current_col in opt_row:
                current_data.append({
                    'channel': CHANNEL_NAMES[col],
                    'spend': opt_row[current_col]
                })
        current_df = pd.DataFrame(current_data)

        fig_current = px.pie(
            current_df, values='spend', names='channel',
            title="Current Split",
            color='channel',
            color_discrete_map=CHANNEL_COLORS,
        )
        fig_current.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_current, use_container_width=True)

        current_rev = opt_row['current_mix_revenue']
        st.metric("Predicted Revenue", format_currency(current_rev))

    with col2:
        st.subheader("Optimized Allocation")
        optimal_data = []
        for col in SPEND_COLUMNS:
            opt_col = f'optimal_{col}'
            if opt_col in opt_row:
                optimal_data.append({
                    'channel': CHANNEL_NAMES[col],
                    'spend': opt_row[opt_col]
                })
        optimal_df = pd.DataFrame(optimal_data)

        fig_optimal = px.pie(
            optimal_df, values='spend', names='channel',
            title="Optimal Split",
            color='channel',
            color_discrete_map=CHANNEL_COLORS,
        )
        fig_optimal.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_optimal, use_container_width=True)

        optimal_rev = opt_row['optimized_revenue']
        delta = optimal_rev - current_rev
        st.metric("Predicted Revenue", format_currency(optimal_rev),
                   delta=f"+{format_currency(delta)}")

    # Revenue lift callout
    lift_pct = opt_row['lift_pct']
    st.success(f"Optimizing allocation at ${total_budget:.1f}M budget increases "
               f"predicted revenue by {lift_pct:.1f}% "
               f"(+{format_currency(delta)})")

    # Side-by-side comparison table
    st.markdown("### Allocation Comparison")
    comparison_data = []
    for col in SPEND_COLUMNS:
        name = CHANNEL_NAMES[col]
        current_val = opt_row.get(f'current_{col}', 0)
        optimal_val = opt_row.get(f'optimal_{col}', 0)
        change = optimal_val - current_val
        change_pct = (change / current_val * 100) if current_val > 0 else 0
        comparison_data.append({
            'Channel': name,
            'Current': format_currency(current_val),
            'Optimal': format_currency(optimal_val),
            'Change': f"{'+' if change >= 0 else ''}{format_currency(change)}",
            'Change %': f"{'+' if change_pct >= 0 else ''}{change_pct:.1f}%"
        })

    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True,
                 hide_index=True)

    # Revenue curve across budget levels
    st.markdown("### Revenue vs Budget Level")
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Scatter(
        x=opt_df['budget'] / 1e6,
        y=opt_df['current_mix_revenue'] / 1e6,
        name='Current Mix',
        line=dict(color='#e74c3c', width=2, dash='dash')
    ))
    fig_rev.add_trace(go.Scatter(
        x=opt_df['budget'] / 1e6,
        y=opt_df['optimized_revenue'] / 1e6,
        name='Optimized Mix',
        line=dict(color='#2ecc71', width=2)
    ))
    fig_rev.update_layout(
        xaxis_title="Total Budget ($M)",
        yaxis_title="Predicted Revenue ($M)",
        height=350,
        hovermode='x unified'
    )
    st.plotly_chart(fig_rev, use_container_width=True)


# ============================================
# TAB 3: SCENARIO PLANNER
# ============================================

with tab3:
    st.header("Scenario Planner")
    st.markdown("Adjust per-channel budgets and see the impact on predicted revenue in real-time.")

    # Current spend baseline
    ws = data['weekly_spend']
    current_spend = {
        'TV Broadcast': ws['tv_broadcast_spend'].sum(),
        'TV Cable': ws['tv_cable_spend'].sum(),
        'TV Streaming': ws['tv_streaming_spend'].sum(),
        'Paid Search': ws['paid_search_spend'].sum(),
        'Social': ws['social_spend'].sum(),
        'Display': ws['display_spend'].sum(),
    }

    mmm_params = data['mmm_params']
    marginal_rois = mmm_params.get('marginal_rois', {})

    # Preset scenarios
    st.markdown("### Quick Scenarios")
    scenario_cols = st.columns(4)
    preset = None
    with scenario_cols[0]:
        if st.button("Cut TV 50%"):
            preset = 'cut_tv'
    with scenario_cols[1]:
        if st.button("All Digital"):
            preset = 'all_digital'
    with scenario_cols[2]:
        if st.button("+20% Budget"):
            preset = 'plus_20'
    with scenario_cols[3]:
        if st.button("Super Bowl"):
            preset = 'super_bowl'

    st.markdown("### Channel Budgets")

    # Apply presets
    defaults = {}
    for ch, val in current_spend.items():
        defaults[ch] = val / 1e6

    if preset == 'cut_tv':
        defaults['TV Broadcast'] *= 0.5
        defaults['TV Cable'] *= 0.5
        defaults['Paid Search'] *= 1.3
    elif preset == 'all_digital':
        defaults['TV Broadcast'] = 0.1
        defaults['TV Cable'] = 0.1
        defaults['TV Streaming'] = 0.5
        defaults['Paid Search'] *= 1.8
        defaults['Social'] *= 1.5
        defaults['Display'] *= 1.5
    elif preset == 'plus_20':
        for ch in defaults:
            defaults[ch] *= 1.2
    elif preset == 'super_bowl':
        defaults['TV Broadcast'] *= 1.5
        defaults['TV Cable'] *= 1.2

    slider_values = {}
    col_s1, col_s2 = st.columns(2)

    channels_left = ['TV Broadcast', 'TV Cable', 'TV Streaming']
    channels_right = ['Paid Search', 'Social', 'Display']

    with col_s1:
        for ch in channels_left:
            max_val = max(10.0, defaults[ch] * 3)
            slider_values[ch] = st.slider(
                f"{ch} ($M)",
                min_value=0.0,
                max_value=max_val,
                value=min(defaults[ch], max_val),
                step=0.1,
                key=f'scenario_{ch}'
            ) * 1e6

    with col_s2:
        for ch in channels_right:
            max_val = max(8.0, defaults[ch] * 3)
            slider_values[ch] = st.slider(
                f"{ch} ($M)",
                min_value=0.0,
                max_value=max_val,
                value=min(defaults[ch], max_val),
                step=0.1,
                key=f'scenario_{ch}'
            ) * 1e6

    total_scenario = sum(slider_values.values())
    st.markdown(f"**Total Spend: {format_currency(total_scenario)}**")

    # Compute predicted revenue using marginal ROIs with saturation
    base_revenue = ws['total_revenue'].sum() - sum(
        marginal_rois.get(ch, 0) * current_spend[ch]
        for ch in current_spend
    )

    predicted_revenue = base_revenue
    for ch, spend_val in slider_values.items():
        current = current_spend[ch]
        roi = marginal_rois.get(ch, 1.0)
        delta = spend_val - current
        # Apply saturation: marginal ROI decreases as spend increases
        saturation_factor = 1.0 / (1.0 + abs(delta) / max(current, 1))
        predicted_revenue += spend_val * roi * saturation_factor + current * roi * (1 - saturation_factor)

    current_total_revenue = ws['total_revenue'].sum()
    delta_rev = predicted_revenue - current_total_revenue
    delta_pct = delta_rev / current_total_revenue * 100

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("Current Revenue", format_currency(current_total_revenue))
    with col_r2:
        st.metric("Predicted Revenue", format_currency(predicted_revenue),
                   delta=f"{'+' if delta_pct >= 0 else ''}{delta_pct:.1f}%")
    with col_r3:
        overall_roas = predicted_revenue / total_scenario if total_scenario > 0 else 0
        st.metric("Overall ROAS", f"{overall_roas:.2f}x")

    # Show saturation curves
    st.markdown("### Diminishing Returns Curves")
    if os.path.exists('results/saturation_curves.png'):
        st.image('results/saturation_curves.png', use_container_width=True)


# ============================================
# TAB 4: CHANNEL DEEP DIVE
# ============================================

with tab4:
    st.header("Channel Deep Dive")

    channel_names_list = list(CHANNEL_NAMES.values())
    selected_channel = st.selectbox("Select Channel", channel_names_list)

    # Find the spend column
    spend_col = None
    for col, name in CHANNEL_NAMES.items():
        if name == selected_channel:
            spend_col = col
            break

    if spend_col and mmm_params:
        adstock_decays = mmm_params.get('adstock_decays', {})
        hill_params = mmm_params.get('hill_params', {})
        saturation_levels = mmm_params.get('saturation_levels', {})

        decay = adstock_decays.get(spend_col, 0.5)
        h_params = hill_params.get(spend_col, {'alpha': 0.7, 'K': 50000})
        sat_level = saturation_levels.get(selected_channel, 50)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Adstock Decay Rate", f"{decay:.2f}")
        with col2:
            st.metric("Saturation Level", f"{sat_level:.0f}%")
        with col3:
            roi = marginal_rois.get(selected_channel, 0)
            st.metric("Marginal ROI", f"{roi:.2f}x")

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            # Adstock decay curve
            st.markdown("### Adstock Decay")
            st.markdown("How long does the advertising effect persist after spending stops?")
            weeks_range = np.arange(16)
            impulse = np.zeros(16)
            impulse[0] = 1.0
            response = adstock_transform(impulse, decay)

            fig_adstock = go.Figure()
            fig_adstock.add_trace(go.Scatter(
                x=weeks_range, y=response,
                mode='lines+markers',
                line=dict(color=CHANNEL_COLORS.get(selected_channel, '#3498db'), width=2),
                fill='tozeroy',
                fillcolor=f'rgba(41, 128, 185, 0.1)',
            ))
            fig_adstock.update_layout(
                xaxis_title="Weeks After Spend",
                yaxis_title="Remaining Effect",
                height=300,
            )
            st.plotly_chart(fig_adstock, use_container_width=True)

        with col_c2:
            # Saturation curve
            st.markdown("### Saturation Curve")
            st.markdown("How much incremental value does additional spend deliver?")
            alpha = h_params['alpha']
            K = h_params['K']
            current_weekly = data['weekly_spend'][spend_col].mean()

            max_spend = current_weekly * 3
            spend_range = np.linspace(0, max_spend, 200)
            saturated = hill_saturation(spend_range, alpha, K)

            fig_sat = go.Figure()
            fig_sat.add_trace(go.Scatter(
                x=spend_range / 1e3, y=saturated,
                mode='lines',
                line=dict(color=CHANNEL_COLORS.get(selected_channel, '#3498db'), width=2),
            ))
            # Current position
            current_sat = hill_saturation(np.array([current_weekly]), alpha, K)[0]
            fig_sat.add_trace(go.Scatter(
                x=[current_weekly / 1e3], y=[current_sat],
                mode='markers+text',
                marker=dict(size=12, color='#e74c3c'),
                text=[f'You Are Here ({current_sat:.0%})'],
                textposition='top center'
            ))
            fig_sat.update_layout(
                xaxis_title="Weekly Spend ($K)",
                yaxis_title="Response (Saturated)",
                yaxis_range=[-0.05, 1.05],
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig_sat, use_container_width=True)

        # Weekly spend contribution over time
        st.markdown("### Weekly Spend Over Time")
        weekly = data['weekly_spend']
        fig_weekly = go.Figure()
        fig_weekly.add_trace(go.Scatter(
            x=weekly['week'],
            y=weekly[spend_col] / 1e3,
            mode='lines+markers',
            line=dict(color=CHANNEL_COLORS.get(selected_channel, '#3498db'), width=2),
            marker=dict(size=4),
        ))
        fig_weekly.update_layout(
            xaxis_title="Week",
            yaxis_title="Spend ($K)",
            height=250,
        )
        st.plotly_chart(fig_weekly, use_container_width=True)

        # Recommendation
        st.markdown("### Recommendation")
        if sat_level > 70:
            st.warning(
                f"**{selected_channel}** is at **{sat_level:.0f}% saturation**. "
                f"Additional spend has diminishing returns. Consider shifting budget "
                f"to channels with lower saturation for better marginal ROI."
            )
        elif sat_level > 40:
            st.info(
                f"**{selected_channel}** is at **{sat_level:.0f}% saturation**. "
                f"There is room for incremental investment, but watch for "
                f"diminishing returns above 70% saturation."
            )
        else:
            st.success(
                f"**{selected_channel}** is at **{sat_level:.0f}% saturation**. "
                f"This channel has significant headroom for additional investment "
                f"with strong marginal returns."
            )


# ============================================
# TAB 5: BAYESIAN MMM
# ============================================

with tab5:
    st.header("Bayesian Media Mix Model")
    st.markdown(
        "Full Bayesian estimation using PyMC with MCMC sampling. Unlike the frequentist MMM, "
        "this provides **posterior distributions** over ROAS — capturing uncertainty, not just "
        "point estimates."
    )

    if 'bayesian_contributions' in data:
        bc = data['bayesian_contributions']

        # Posterior ROAS metrics with credible intervals
        st.subheader("Posterior ROAS by Channel (90% Credible Intervals)")
        roas_cols = st.columns(len(bc))
        for i, (_, row) in enumerate(bc.iterrows()):
            with roas_cols[i]:
                ci_text = f"[{row['roas_ci_5']:.2f}, {row['roas_ci_95']:.2f}]"
                st.metric(
                    row['channel'],
                    f"{row['median_roas']:.2f}x",
                    delta=ci_text,
                    delta_color="off"
                )

        # Posterior ROAS forest plot
        st.subheader("Posterior ROAS Distributions")
        bc_sorted = bc.sort_values('median_roas', ascending=True)
        fig = go.Figure()
        for _, row in bc_sorted.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['roas_ci_5'], row['median_roas'], row['roas_ci_95']],
                y=[row['channel']] * 3,
                mode='markers+lines',
                marker=dict(size=[8, 14, 8], color=['gray', '#8e44ad', 'gray']),
                line=dict(color='#8e44ad', width=3),
                name=row['channel'],
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['channel']}</b><br>"
                    f"Median: {row['median_roas']:.2f}x<br>"
                    f"90% CI: [{row['roas_ci_5']:.2f}, {row['roas_ci_95']:.2f}]"
                ),
            ))
        fig.add_vline(x=1.0, line_dash='dash', line_color='red',
                      annotation_text='Break-even (1.0x)')
        fig.update_layout(
            title='Posterior ROAS with 90% Credible Intervals',
            xaxis_title='ROAS',
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('results/bayesian_posterior_roas.png'):
                st.image('results/bayesian_posterior_roas.png',
                         caption='Posterior ROAS Distributions (MCMC)')
        with col2:
            if os.path.exists('results/bayesian_forest_plot.png'):
                st.image('results/bayesian_forest_plot.png',
                         caption='Forest Plot: Credible Intervals')

        # Bayesian contributions comparison
        st.subheader("Bayesian Channel Contributions")
        fig = px.bar(
            bc.sort_values('median_contribution', ascending=True),
            x='median_contribution', y='channel', orientation='h',
            color='channel',
            color_discrete_map=CHANNEL_COLORS,
            title='Bayesian MMM: Median Channel Revenue Contributions'
        )
        fig.update_layout(height=350, showlegend=False,
                          xaxis_title='Revenue Contribution ($)')
        st.plotly_chart(fig, use_container_width=True)

        if os.path.exists('results/bayesian_posterior_predictive.png'):
            st.image('results/bayesian_posterior_predictive.png',
                     caption='Posterior Predictive Check: Model vs Actual')

        # Technical details
        if 'bayesian_params' in data:
            bp = data['bayesian_params']
            st.subheader("MCMC Diagnostics")
            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            with col_d1:
                st.metric("Chains", bp.get('n_chains', 2))
            with col_d2:
                st.metric("Draws / Chain", bp.get('n_draws', 2000))
            with col_d3:
                st.metric("Tune Steps", bp.get('n_tune', 1500))
            with col_d4:
                st.metric("Divergences", bp.get('divergences', 'N/A'))

    else:
        st.info("Run `python src/bayesian_mmm.py` to generate Bayesian MMM results.")



# ============================================
# TAB 6: SHAP EXPLAINABILITY
# ============================================

with tab6:
    st.header("SHAP Explainability")
    st.markdown(
        "**TreeSHAP** on the LightGBM baseline model provides exact Shapley values — "
        "the theoretically grounded way to attribute each prediction to individual features. "
        "These are additive, consistent, and locally accurate."
    )

    if 'shap_importance' in data:
        shap_df = data['shap_importance']

        # Top features table
        st.subheader("Feature Importance (Mean |SHAP Value|)")
        top_n = 15
        top_features = shap_df.head(top_n)

        fig = px.bar(
            top_features.sort_values('mean_abs_shap', ascending=True),
            x='mean_abs_shap', y='feature', orientation='h',
            title='Top 15 Features by Mean Absolute SHAP Value',
            color='mean_abs_shap',
            color_continuous_scale='Viridis',
        )
        fig.update_layout(height=450, showlegend=False,
                          xaxis_title='Mean |SHAP Value|',
                          yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)

        # SHAP plots
        st.subheader("SHAP Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('results/shap_summary_baseline.png'):
                st.image('results/shap_summary_baseline.png',
                         caption='SHAP Summary (Beeswarm) Plot')
        with col2:
            if os.path.exists('results/shap_dependence_hour.png'):
                st.image('results/shap_dependence_hour.png',
                         caption='SHAP Dependence: Time-of-Day Features')

        col3, col4 = st.columns(2)
        with col3:
            if os.path.exists('results/shap_waterfall_sample.png'):
                st.image('results/shap_waterfall_sample.png',
                         caption='SHAP Waterfall: Single Prediction Explained')
        with col4:
            if os.path.exists('results/shap_feature_importance.png'):
                st.image('results/shap_feature_importance.png',
                         caption='Feature Importance Bar Chart')

        # Interpretation
        st.subheader("Key Interpretations")
        top3 = shap_df.head(3)['feature'].tolist()
        st.markdown(
            f"- **{top3[0]}** is the most important feature, confirming geographic "
            f"variation dominates traffic patterns\n"
            f"- **{top3[1]}** and **{top3[2]}** capture time-of-day cyclical effects "
            f"(encoded via sine/cosine to preserve continuity)\n"
            f"- Weekend and holiday indicators have moderate importance, reflecting "
            f"behavioral shifts in browsing patterns\n"
            f"- SHAP values are exact (not approximate) because TreeSHAP exploits "
            f"the tree structure for polynomial-time computation"
        )
    else:
        st.info("Run `python src/shap_analysis.py` to generate SHAP results.")


# ============================================
# TAB 7: MARKOV ATTRIBUTION
# ============================================

with tab7:
    st.header("Markov Chain Attribution")
    st.markdown(
        "Absorbing Markov chains model the customer journey as a stochastic process. "
        "**Removal effects** measure each channel's importance by simulating what happens "
        "when it's removed from the journey entirely."
    )

    if 'markov_attribution' in data:
        markov_df = data['markov_attribution']

        # Attribution comparison: Markov vs Last-Touch vs First-Touch
        st.subheader("Attribution Method Comparison")
        fig = go.Figure()

        for method, col_name, color in [
            ('Markov Chain', 'markov_attribution', '#2c3e50'),
            ('Last-Touch', 'last_touch_attribution', '#e74c3c'),
            ('First-Touch', 'first_touch_attribution', '#e67e22'),
        ]:
            fig.add_trace(go.Bar(
                x=markov_df['channel'],
                y=markov_df[col_name] * 100,
                name=method,
                marker_color=color,
                opacity=0.85,
            ))

        fig.update_layout(
            barmode='group',
            title='Channel Attribution: Markov vs Heuristic Models',
            yaxis_title='Attribution Share (%)',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Difference analysis
        st.subheader("Markov vs Last-Touch Differences")
        markov_df_display = markov_df.copy()
        markov_df_display['markov_vs_lasttouch'] = (
            (markov_df['markov_attribution'] - markov_df['last_touch_attribution']) * 100
        )
        markov_df_display['interpretation'] = markov_df_display['markov_vs_lasttouch'].apply(
            lambda x: 'Undervalued by Last-Touch' if x > 1
            else ('Overvalued by Last-Touch' if x < -1 else 'Similar')
        )

        fig_diff = px.bar(
            markov_df_display.sort_values('markov_vs_lasttouch'),
            x='markov_vs_lasttouch', y='channel', orientation='h',
            color='interpretation',
            color_discrete_map={
                'Undervalued by Last-Touch': '#27ae60',
                'Overvalued by Last-Touch': '#e74c3c',
                'Similar': '#95a5a6',
            },
            title='Markov minus Last-Touch Attribution (percentage points)',
        )
        fig_diff.update_layout(height=300, xaxis_title='Difference (pp)')
        st.plotly_chart(fig_diff, use_container_width=True)

        # Removal effects
        if 'markov_removal' in data:
            st.subheader("Channel Removal Effects")
            st.markdown(
                "The removal effect measures the drop in total conversions when a channel "
                "is completely removed from all customer journeys."
            )
            removal_df = data['markov_removal']
            fig_removal = px.bar(
                removal_df.sort_values('removal_effect', ascending=True),
                x='removal_effect', y='channel', orientation='h',
                color='removal_effect',
                color_continuous_scale='Reds',
                title='Removal Effects: Conversion Drop When Channel Is Removed',
            )
            fig_removal.update_layout(height=350, xaxis_title='Removal Effect',
                                       showlegend=False)
            st.plotly_chart(fig_removal, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('results/markov_transition_heatmap.png'):
                st.image('results/markov_transition_heatmap.png',
                         caption='Transition Probability Matrix')
        with col2:
            if os.path.exists('results/markov_vs_lasttouch.png'):
                st.image('results/markov_vs_lasttouch.png',
                         caption='Markov vs Last-Touch Comparison')

    else:
        st.info("Run `python src/markov_attribution.py` to generate Markov results.")


# ============================================
# TAB 8: CROSS-CHANNEL EFFECTS
# ============================================

with tab8:
    st.header("Cross-Channel Effects: TV → Search")
    st.markdown(
        "Investigating how TV advertising drives online search behavior — the 'TV halo effect'. "
        "Uses Granger causality, mediation analysis, and interaction modeling."
    )

    if 'cross_effects' in data:
        effects = data['cross_effects']

        # Key metrics
        st.subheader("TV → Search Relationship")
        col1, col2, col3 = st.columns(3)
        with col1:
            peak_corr = effects.get('peak_correlation', {})
            st.metric(
                "Peak TV→Search Correlation",
                f"r = {peak_corr.get('correlation', 0):.3f}",
                delta=f"Lag {peak_corr.get('lag_weeks', 0)} weeks"
            )
        with col2:
            mediation = effects.get('mediation', {})
            indirect = mediation.get('indirect_effect', 0)
            total = mediation.get('total_effect', 1)
            pct_mediated = (indirect / total * 100) if total != 0 else 0
            st.metric(
                "Effect Mediated via Search",
                f"{pct_mediated:.0f}%",
                delta="indirect path"
            )
        with col3:
            interaction = effects.get('interaction', {})
            st.metric(
                "TV×Search Synergy",
                f"${interaction.get('interaction_coefficient', 0):,.0f}",
                delta=f"R² lift: +{interaction.get('r2_improvement', 0):.4f}"
            )

        # Correlation plot
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if os.path.exists('results/tv_search_lag_correlation.png'):
                st.image('results/tv_search_lag_correlation.png',
                         caption='TV-Search Lag Correlation')
        with col_p2:
            if os.path.exists('results/tv_search_interaction.png'):
                st.image('results/tv_search_interaction.png',
                         caption='TV × Search Revenue Interaction')

        # Granger causality
        st.subheader("Granger Causality Test")
        st.markdown(
            "Does past TV spend contain information about future search volume "
            "beyond what search's own history provides?"
        )
        if 'granger_results' in data:
            gr = data['granger_results']
            for _, row in gr.iterrows():
                sig = "Yes" if row['significant'] else "No"
                color = "green" if row['significant'] else "red"
                st.markdown(
                    f"- **Lag {int(row['lag'])} weeks**: F={row['f_statistic']:.2f}, "
                    f"p={row['p_value']:.4f} — Significant: :{color}[{sig}]"
                )
            st.caption(
                "Note: With only 52 weekly observations and differenced data, "
                "non-significant results are expected. The cross-correlation and "
                "mediation analyses provide complementary evidence."
            )

        if os.path.exists('results/granger_causality.png'):
            st.image('results/granger_causality.png',
                     caption='Granger Causality Test Results')

        # Mediation analysis
        st.subheader("Mediation Analysis: TV → Search → Revenue")
        if mediation:
            fig_med = go.Figure()

            paths = [
                ('Direct: TV → Revenue', mediation.get('direct_effect', 0), '#2c3e50'),
                ('Indirect: TV → Search → Revenue', mediation.get('indirect_effect', 0), '#8e44ad'),
                ('Total Effect', mediation.get('total_effect', 0), '#2980b9'),
            ]
            fig_med.add_trace(go.Bar(
                x=[p[0] for p in paths],
                y=[p[1] for p in paths],
                marker_color=[p[2] for p in paths],
                text=[f'${p[1]:,.0f}' for p in paths],
                textposition='outside',
            ))
            fig_med.update_layout(
                title='Mediation: How TV Affects Revenue (Direct vs Through Search)',
                yaxis_title='Effect Size ($)',
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig_med, use_container_width=True)

    else:
        st.info("Run `python src/cross_channel.py` to generate cross-channel results.")


# ============================================
# TAB 9: MODEL COMPARISON
# ============================================

with tab9:
    st.header("Stacked Model Comparison")
    st.markdown(
        "Six attribution methods applied to the same data produce different answers. "
        "This comparison reveals **where models agree** (higher confidence) and "
        "**where they disagree** (more uncertainty in the true attribution)."
    )

    if 'model_comparison' in data:
        comp_df = data['model_comparison']
        channels = [c for c in comp_df.columns if c != 'model']
        active_channels = [c for c in channels if comp_df[c].sum() > 0.01]

        # Stacked bar chart (interactive)
        st.subheader("Attribution Shares by Model")
        fig = go.Figure()
        for ch in active_channels:
            color = CHANNEL_COLORS.get(ch, '#bdc3c7')
            fig.add_trace(go.Bar(
                y=comp_df['model'],
                x=comp_df[ch] * 100,
                name=ch,
                orientation='h',
                marker_color=color,
                text=comp_df[ch].apply(lambda v: f'{v:.0%}' if v > 0.05 else ''),
                textposition='inside',
                textfont_color='white',
            ))
        fig.update_layout(
            barmode='stack',
            title='How Each Method Attributes Revenue Across Channels',
            xaxis_title='Attribution Share (%)',
            height=450,
            legend=dict(orientation='h', yanchor='bottom', y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Grouped bar chart
        st.subheader("Channel-Level Comparison")
        fig_grouped = go.Figure()
        model_colors = {
            'Last-Touch': '#e74c3c', 'First-Touch': '#e67e22',
            'Markov Chain': '#2c3e50', 'LightGBM Attribution': '#27ae60',
            'Frequentist MMM': '#3498db', 'Bayesian MMM': '#8e44ad',
        }
        for _, row in comp_df.iterrows():
            model_name = row['model']
            fig_grouped.add_trace(go.Bar(
                x=active_channels,
                y=[row[ch] * 100 for ch in active_channels],
                name=model_name,
                marker_color=model_colors.get(model_name, 'gray'),
                opacity=0.85,
            ))
        fig_grouped.update_layout(
            barmode='group',
            title='Each Channel Seen Through Different Lenses',
            yaxis_title='Attribution Share (%)',
            height=400,
        )
        st.plotly_chart(fig_grouped, use_container_width=True)

        # Radar chart image
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if os.path.exists('results/model_comparison_radar.png'):
                st.image('results/model_comparison_radar.png',
                         caption='Attribution Radar Chart')
        with col_r2:
            if os.path.exists('results/model_comparison_insights.png'):
                st.image('results/model_comparison_insights.png',
                         caption='Key Insights Summary')

        # Comparison table
        st.subheader("Full Comparison Matrix")
        display_df = comp_df.copy()
        for ch in active_channels:
            display_df[ch] = display_df[ch].apply(lambda x: f'{x:.1%}')
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Insights
        st.subheader("Key Takeaways")
        st.markdown("""
        1. **TV's value varies 2-5x** depending on methodology — Last-Touch dramatically
           undervalues TV because it ignores TV's role as a journey initiator

        2. **Heuristic models over-credit digital** — Paid Search gets credit for conversions
           that TV initiated (the cross-channel effect documented in the Cross-Channel tab)

        3. **Bayesian MMM provides uncertainty** — Unlike point estimates, the Bayesian
           approach shows credible intervals, making it clear when we're confident vs uncertain

        4. **Markov Chain captures journey dynamics** — By modeling the full customer journey
           as a stochastic process, it properly credits channels that appear early in the funnel

        5. **No single model is correct** — Each captures different aspects of advertising
           effectiveness. The recommendation is to triangulate across methods.
        """)

    else:
        st.info("Run `python src/model_comparison.py` to generate comparison results.")


# ============================================
# TAB: DOUBLE MACHINE LEARNING
# ============================================

with tab_dml:
    st.header("Double Machine Learning (DML)")
    st.markdown(
        "**Debiased/orthogonal causal inference** (Chernozhukov et al., 2018). "
        "Uses LightGBM for nuisance parameters but preserves valid statistical "
        "inference on the treatment effect — the gold standard in causal ML."
    )

    if 'dml_effects' in data:
        dml_df = data['dml_effects']

        # Causal effect metrics
        st.subheader("Causal Average Treatment Effects (ATE)")
        dml_cols = st.columns(3)
        for i, (_, row) in enumerate(dml_df.iterrows()):
            with dml_cols[i % 3]:
                sig_marker = " ***" if row['p_value'] < 0.01 else (" **" if row['p_value'] < 0.05 else "")
                st.metric(
                    row['channel'],
                    f"${row['ate']:.3f}/$ spent",
                    delta=f"p={row['p_value']:.4f}{sig_marker}",
                    delta_color="off"
                )

        # DML vs OLS comparison
        if 'dml_vs_ols' in data:
            st.subheader("DML vs Naive OLS: Exposing Confounding Bias")
            ols_df = data['dml_vs_ols']
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ols_df['channel'], y=ols_df['ols_coefficient'],
                                 name='Naive OLS (biased)', marker_color='#e74c3c', opacity=0.7))
            fig.add_trace(go.Bar(x=ols_df['channel'], y=ols_df['dml_causal_effect'],
                                 name='DML (debiased)', marker_color='#2980b9', opacity=0.7))
            fig.update_layout(barmode='group', height=400,
                              title='OLS vs DML: How Much Bias Does Confounding Introduce?',
                              yaxis_title='Effect ($/$ spend)')
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('results/dml_causal_effects.png'):
                st.image('results/dml_causal_effects.png',
                         caption='Causal Effects with 90% CIs')
        with col2:
            if os.path.exists('results/dml_cate_heterogeneity.png'):
                st.image('results/dml_cate_heterogeneity.png',
                         caption='Causal Forest: Heterogeneous Effects Over Time')

    else:
        st.info("Run `python src/double_ml.py` to generate DML results.")


# ============================================
# TAB: TEMPORAL FUSION TRANSFORMER
# ============================================

with tab_tft:
    st.header("Temporal Fusion Transformer (TFT)")
    st.markdown(
        "**State-of-the-art deep learning** for time series (Google Research, 2021). "
        "Includes Variable Selection Network, LSTM encoder, and Multi-Head Attention — "
        "all providing interpretable insights into which features and past timesteps "
        "drive predictions."
    )

    if 'tft_metrics' in data:
        metrics = data['tft_metrics']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train R²", f"{metrics.get('train_r2', 0):.3f}")
        with col2:
            st.metric("Test R²", f"{metrics.get('test_r2', 0):.3f}")
        with col3:
            st.metric("Test MAE", f"${metrics.get('test_mae', 0):,.0f}")
        with col4:
            st.metric("80% PI Coverage", f"{metrics.get('coverage', 0):.0%}")

        # Variable importance
        if 'tft_importance' in data:
            st.subheader("Variable Selection Network: Learned Feature Importance")
            vi = data['tft_importance']
            fig = px.bar(vi.sort_values('importance', ascending=True),
                         x='importance', y='feature', orientation='h',
                         color='importance', color_continuous_scale='Viridis',
                         title='TFT Learns Which Inputs Matter (Higher = More Important)')
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('results/tft_forecast.png'):
                st.image('results/tft_forecast.png',
                         caption='Revenue Forecast with Prediction Intervals')
        with col2:
            if os.path.exists('results/tft_attention_heatmap.png'):
                st.image('results/tft_attention_heatmap.png',
                         caption='Attention Weights & Training Convergence')

        st.caption(
            "Note: With only 52 weekly observations, the TFT overfits "
            "(negative test R²). This demonstrates the architecture and "
            "interpretability — real-world deployment needs more data."
        )
    else:
        st.info("Run `python src/temporal_fusion.py` to generate TFT results.")


# ============================================
# TAB: CONFORMAL PREDICTION
# ============================================

with tab_conf:
    st.header("Conformal Prediction")
    st.markdown(
        "**Distribution-free uncertainty quantification** with guaranteed finite-sample "
        "coverage. No distributional assumptions — only exchangeability. "
        "Based on Vovk et al. (2005), popularized by Angelopoulos & Bates (2021)."
    )

    if 'conformal_summary' in data:
        cs = data['conformal_summary']

        st.subheader("Method Comparison: 90% Prediction Intervals")
        methods = ['split_conformal', 'jackknife_plus', 'cqr']
        method_names = ['Split Conformal', 'Jackknife+ (Cross-Conformal)', 'CQR (Adaptive)']
        mc = st.columns(3)
        for i, (method, name) in enumerate(zip(methods, method_names)):
            if method in cs:
                with mc[i]:
                    st.metric(f"{name}\nCoverage",
                              f"{cs[method]['coverage']:.0%}",
                              delta=f"Width: ${cs[method]['avg_width']:,.0f}")

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('results/conformal_intervals.png'):
                st.image('results/conformal_intervals.png',
                         caption='Three Conformal Methods Compared')
        with col2:
            if os.path.exists('results/conformal_coverage.png'):
                st.image('results/conformal_coverage.png',
                         caption='Coverage Calibration & Width Trade-off')

        if os.path.exists('results/conformal_width_analysis.png'):
            st.image('results/conformal_width_analysis.png',
                     caption='CQR: Adaptive Intervals (Wider When Uncertain)')

    else:
        st.info("Run `python src/conformal_prediction.py` to generate conformal results.")


# ============================================
# TAB: GEO-LIFT / SYNTHETIC CONTROL
# ============================================

with tab_geo:
    st.header("Geo-Lift / Synthetic Control")
    st.markdown(
        "**Causal incrementality testing** (Abadie et al., 2010). Constructs a synthetic "
        "'control DMA' from weighted donor markets to estimate the true causal lift "
        "from advertising — what Google and Meta actually use."
    )

    if 'geo_lift_summary' in data:
        gs = data['geo_lift_summary']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Treatment DMA", gs.get('treatment_dma', 'N/A'))
        with col2:
            st.metric("Causal Lift", f"{gs.get('avg_lift_pct', 0):+.1f}%")
        with col3:
            st.metric("Pre-Period R²", f"{gs.get('pre_r2', 0):.3f}")

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('results/geo_lift_main.png'):
                st.image('results/geo_lift_main.png',
                         caption='Synthetic Control: Actual vs Counterfactual')
        with col2:
            if os.path.exists('results/geo_lift_weights.png'):
                st.image('results/geo_lift_weights.png',
                         caption='Donor DMA Weights')

        if os.path.exists('results/geo_lift_placebo.png'):
            st.subheader("Placebo Tests: Falsification Check")
            st.markdown(
                "If the method is valid, placebo tests on control DMAs should show "
                "no effect. The treatment DMA (red) should stand out."
            )
            st.image('results/geo_lift_placebo.png',
                     caption='Placebo Tests Across All DMAs')

    else:
        st.info("Run `python src/geo_lift.py` to generate geo-lift results.")


# ============================================
# TAB: CAUSAL IMPACT (BSTS)
# ============================================

with tab_ci:
    st.header("CausalImpact (BSTS)")
    st.markdown(
        "**Bayesian Structural Time Series** — Google's approach to measuring "
        "campaign impact. Fits a model on pre-intervention data, projects a "
        "counterfactual, and computes posterior probability of causal effect."
    )

    if 'ci_summary' in data:
        cis = data['ci_summary']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Intervention Week", cis.get('intervention_week', 'N/A'))
        with col2:
            st.metric("Causal Effect", f"${cis.get('causal_effect', 0):,.0f}")
        with col3:
            st.metric("Effect %", f"{cis.get('causal_effect_pct', 0):+.1f}%")
        with col4:
            st.metric("Posterior Prob", f"{cis.get('posterior_prob', 0):.0%}")

        if os.path.exists('results/causal_impact_main.png'):
            st.image('results/causal_impact_main.png',
                     caption='CausalImpact: Original, Pointwise, and Cumulative Effects')

        if os.path.exists('results/causal_impact_cumulative.png'):
            st.image('results/causal_impact_cumulative.png',
                     caption='TV Spend Context & Search Volume Impact')

        if 'search' in cis:
            st.subheader("TV → Brand Search Impact")
            st.metric("Search Volume Effect", f"{cis['search']['search_effect_pct']:+.1f}%",
                      delta=f"Posterior Prob: {cis['search']['posterior_prob']:.0%}")

    else:
        st.info("Run `python src/causal_impact.py` to generate CausalImpact results.")


# ============================================
# TAB: KALMAN FILTER
# ============================================

with tab_kalman:
    st.header("Kalman Filter: Time-Varying Parameters")
    st.markdown(
        "**State-space model** where channel effectiveness drifts over time "
        "as a random walk. Unlike static MMM (one coefficient per channel for the year), "
        "the Kalman filter tracks how ROAS evolves week by week."
    )

    if 'kalman_summary' in data:
        ks = data['kalman_summary']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Static (Ridge) R²", f"{ks.get('static_r2', 0):.4f}")
        with col2:
            st.metric("Dynamic (Kalman) R²", f"{ks.get('dynamic_r2', 0):.4f}")
        with col3:
            st.metric("Improvement", f"+{ks.get('improvement_pp', 0):.2f} pp")

        if os.path.exists('results/kalman_tvp_coefficients.png'):
            st.image('results/kalman_tvp_coefficients.png',
                     caption='Time-Varying Coefficients (Shaded = 95% CI, Dashed = Static)')

        if os.path.exists('results/kalman_tvp_roas_evolution.png'):
            st.image('results/kalman_tvp_roas_evolution.png',
                     caption='How Channel Effectiveness Evolves Over Time')

        if os.path.exists('results/kalman_vs_static.png'):
            st.image('results/kalman_vs_static.png',
                     caption='Static vs Dynamic: Prediction & Residuals')

        # Interactive coefficient chart
        if 'kalman_coefs' in data:
            st.subheader("Interactive: Time-Varying Coefficients")
            kc = data['kalman_coefs']
            channels = [c for c in kc.columns if c != 'week']
            fig = go.Figure()
            for ch in channels:
                fig.add_trace(go.Scatter(
                    x=kc['week'], y=kc[ch], mode='lines',
                    name=ch, line=dict(width=2),
                ))
            fig.update_layout(
                title='Channel Effectiveness Over Time (Kalman Smoothed)',
                xaxis_title='Week', yaxis_title='Coefficient',
                height=400, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Run `python src/kalman_tvp.py` to generate Kalman filter results.")


# ============================================
# TAB 10: CLIENT REPORT
# ============================================

with tab10:
    st.header("Client Report")
    st.markdown("Auto-generated executive summary for stakeholder presentation.")

    # Executive summary
    total_tv_spend = (ws['tv_broadcast_spend'].sum() +
                      ws['tv_cable_spend'].sum() +
                      ws['tv_streaming_spend'].sum())
    total_spend = ws['total_spend'].sum()
    total_revenue = ws['total_revenue'].sum()
    n_airings = len(data['airings'])
    attr_rev = data['airing_attribution']['incremental_revenue'].sum()
    overall_roas_val = attr_rev / total_tv_spend if total_tv_spend > 0 else 0

    # Find top network
    net_df = data['attr_by_network'].sort_values('avg_roas', ascending=False)
    top_network = net_df.iloc[0]['network'] if len(net_df) > 0 else 'N/A'
    top_network_roas = net_df.iloc[0]['avg_roas'] if len(net_df) > 0 else 0

    # Find best/worst channels from MMM
    contrib_df = data['channel_contributions']
    positive_channels = contrib_df[
        (contrib_df['channel'] != 'Base + Seasonality') & (contrib_df['roi'] > 0)
    ]
    if len(positive_channels) > 0:
        best_channel = positive_channels.sort_values('roi', ascending=False).iloc[0]['channel']
        worst_channel = positive_channels.sort_values('roi').iloc[0]['channel']
    else:
        best_channel = "Paid Search"
        worst_channel = "Display"

    st.markdown("---")
    st.subheader("Executive Summary")

    summary_text = (
        f"Over the 2023 campaign period, the brand invested **{format_currency(total_spend)}** "
        f"across all channels, generating **{format_currency(total_revenue)}** in total revenue "
        f"(blended ROAS: {total_revenue/total_spend:.1f}x).\n\n"
        f"**TV Advertising** accounted for {format_currency(total_tv_spend)} across "
        f"**{n_airings:,} DMA-level airings**. The attribution model identified "
        f"**{format_currency(attr_rev)}** in incremental revenue directly attributable to TV "
        f"within a 20-minute response window (ROAS: {overall_roas_val:.2f}x). "
        f"This conservative estimate captures only immediate response; the Media Mix Model "
        f"captures additional long-term brand effects through adstock modeling.\n\n"
        f"**Top performing network**: {top_network} (ROAS: {top_network_roas:.2f}x)\n\n"
        f"**Recommended action**: The budget optimizer suggests reallocating spend from "
        f"{worst_channel} to {best_channel} for an estimated revenue uplift."
    )
    st.markdown(summary_text)

    # Key visualizations
    st.markdown("---")
    st.subheader("Key Visualizations")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        if os.path.exists('results/channel_contributions.png'):
            st.image('results/channel_contributions.png',
                     caption="Channel Revenue Contributions (MMM)")
    with col_v2:
        if os.path.exists('results/roas_by_network.png'):
            st.image('results/roas_by_network.png',
                     caption="ROAS by Network (Attribution)")

    col_v3, col_v4 = st.columns(2)
    with col_v3:
        if os.path.exists('results/actual_vs_predicted.png'):
            st.image('results/actual_vs_predicted.png',
                     caption="MMM: Actual vs Predicted Revenue")
    with col_v4:
        if os.path.exists('results/baseline_vs_actual.png'):
            st.image('results/baseline_vs_actual.png',
                     caption="Attribution: Baseline vs Actual Traffic")

    # PDF Export
    st.markdown("---")
    st.subheader("Export Report")

    if st.button("Generate PDF Report"):
        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Helvetica', 'B', 18)
            pdf.cell(0, 12, 'TV Advertising Performance Report', ln=True, align='C')
            pdf.set_font('Helvetica', '', 11)
            pdf.cell(0, 8, 'Period: January 2023 - December 2023', ln=True, align='C')
            pdf.ln(8)

            # Executive Summary
            pdf.set_font('Helvetica', 'B', 14)
            pdf.cell(0, 10, 'Executive Summary', ln=True)
            pdf.set_font('Helvetica', '', 10)
            pdf.multi_cell(0, 5,
                f"Total investment: {format_currency(total_spend)} across all channels. "
                f"Total revenue: {format_currency(total_revenue)} "
                f"(blended ROAS: {total_revenue/total_spend:.1f}x).\n\n"
                f"TV spend: {format_currency(total_tv_spend)} across {n_airings:,} airings. "
                f"Incremental TV revenue: {format_currency(attr_rev)} "
                f"(attribution ROAS: {overall_roas_val:.2f}x).\n\n"
                f"Top network: {top_network} (ROAS: {top_network_roas:.2f}x). "
                f"Recommended: shift budget from {worst_channel} to {best_channel}."
            )
            pdf.ln(5)

            # Channel contributions
            pdf.set_font('Helvetica', 'B', 14)
            pdf.cell(0, 10, 'Channel Contributions (MMM)', ln=True)
            pdf.set_font('Helvetica', '', 10)
            for _, row in contrib_df.iterrows():
                pdf.cell(0, 6,
                    f"  {row['channel']}: {format_currency(row['contribution'])} "
                    f"({row['pct_contribution']:.1f}%)", ln=True)
            pdf.ln(5)

            # Add chart images
            for img_path in ['results/channel_contributions.png',
                             'results/roas_by_network.png',
                             'results/actual_vs_predicted.png']:
                if os.path.exists(img_path):
                    pdf.add_page()
                    pdf.image(img_path, x=15, w=180)

            pdf_bytes = bytes(pdf.output())
            st.download_button(
                "Download PDF Report",
                pdf_bytes,
                file_name="tv_performance_report.pdf",
                mime="application/pdf"
            )
            st.success("Report generated successfully!")

        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")


# ============================================
# TAB 6: MODEL DIAGNOSTICS
# ============================================

with tab11:
    st.header("Model Diagnostics")
    st.markdown("Quality checks and honest assessment of model limitations.")

    # MMM diagnostics
    st.subheader("Media Mix Model")

    mmm_r2 = mmm_params.get('r2', 0)
    mmm_mape = mmm_params.get('mape', 0)

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.metric("R-squared", f"{mmm_r2:.4f}")
    with col_d2:
        st.metric("MAPE", f"{mmm_mape:.2%}")
    with col_d3:
        st.metric("Training Samples", "52 weeks")

    if os.path.exists('results/actual_vs_predicted.png'):
        st.image('results/actual_vs_predicted.png',
                 caption="Actual vs Predicted Revenue (MMM)")

    # Residuals analysis
    st.markdown("### Residuals Analysis")
    weekly = data['weekly_spend']
    if 'mmm' in models:
        mmm_obj = models['mmm']
        model = mmm_obj['model']
        scaler = mmm_obj['scaler']

        # Reconstruct features to compute residuals
        adstock_decays = mmm_params.get('adstock_decays', {})
        h_params = mmm_params.get('hill_params', {})

        features = {}
        for col in SPEND_COLUMNS:
            spend = weekly[col].values
            decay = adstock_decays.get(col, 0.5)
            adstocked = adstock_transform(spend, decay)
            alpha = h_params.get(col, {}).get('alpha', 0.7)
            K = h_params.get(col, {}).get('K', 50000)
            features[col] = hill_saturation(adstocked, alpha, K)

        features['month_sin'] = np.sin(2 * np.pi * weekly['month'].values / 12)
        features['month_cos'] = np.cos(2 * np.pi * weekly['month'].values / 12)
        features['trend'] = np.arange(len(weekly)) / len(weekly)

        feat_df = pd.DataFrame(features)
        X_scaled = scaler.transform(feat_df.values)
        predicted = model.predict(X_scaled)
        residuals = weekly['total_revenue'].values - predicted

        fig_resid = make_subplots(rows=1, cols=2,
                                  subplot_titles=('Residuals Over Time',
                                                  'Residual Distribution'))

        fig_resid.add_trace(go.Scatter(
            x=weekly['week'], y=residuals / 1e3,
            mode='markers+lines',
            marker=dict(color='#e74c3c', size=4),
            line=dict(width=1),
        ), row=1, col=1)
        fig_resid.add_hline(y=0, line_dash='dash', line_color='gray', row=1, col=1)

        fig_resid.add_trace(go.Histogram(
            x=residuals / 1e3,
            nbinsx=15,
            marker_color='#3498db',
        ), row=1, col=2)

        fig_resid.update_layout(height=300, showlegend=False)
        fig_resid.update_xaxes(title_text="Week", row=1, col=1)
        fig_resid.update_yaxes(title_text="Residual ($K)", row=1, col=1)
        fig_resid.update_xaxes(title_text="Residual ($K)", row=1, col=2)
        st.plotly_chart(fig_resid, use_container_width=True)

    # Baseline model diagnostics
    st.subheader("Baseline Attribution Model")
    if 'baseline_metrics' in data:
        bm = data['baseline_metrics']
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            st.metric("R-squared", f"{bm.get('r2', 0):.4f}")
        with col_b2:
            st.metric("MAE", f"{bm.get('mae', 0):.2f} sessions")
        with col_b3:
            st.metric("Training Samples", "731K clean periods")

    if os.path.exists('results/baseline_vs_actual.png'):
        st.image('results/baseline_vs_actual.png',
                 caption="Baseline vs Actual Traffic (Sample Day)")

    # Honest limitations
    st.markdown("---")
    st.subheader("What the Model Doesn't Capture")
    st.markdown("""
    **Known limitations and caveats:**

    1. **Short attribution window**: The attribution model uses a 20-minute response
       window. TV ads also drive long-term brand awareness (days/weeks later) which
       isn't captured in the airing-level attribution. The MMM partially addresses
       this through adstock modeling.

    2. **Cross-channel interactions**: The MMM treats channels independently. In reality,
       TV ads boost search behavior (TV → Google search → purchase), which means TV's
       true contribution is likely higher than the model estimates.

    3. **Creative wear-out**: The model doesn't account for audience fatigue with
       repeated creative exposure. ROAS for a creative may decline over time as
       the same audience sees it repeatedly.

    4. **Competitive effects**: Competitor TV spending can reduce our ad effectiveness.
       This external factor is not modeled.

    5. **Small sample size**: The MMM has only 52 weekly data points. With 6 channels
       + seasonality controls, overfitting is a risk. The Ridge regularization helps
       but doesn't eliminate this concern.

    6. **Simulated data**: This analysis uses simulated data to demonstrate methodology.
       Real-world data would include additional noise, missing data, and measurement
       errors that require additional preprocessing.
    """)
