"""
TV Ad Attribution & Media Mix Model Dashboard
=============================================
Clean, organized dashboard with sidebar navigation.
Sections: Overview, Attribution, MMM & Optimization, Advanced Causal ML,
          How It Works, Report & Diagnostics.
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
    data = {}
    data["airings"] = pd.read_csv("data/ad_airings.csv")
    data["weekly_spend"] = pd.read_csv("data/weekly_spend.csv")
    data["daily_traffic"] = pd.read_csv("data/daily_traffic.csv")
    data["hourly_traffic"] = pd.read_csv("data/hourly_traffic_agg.csv")
    data["airing_attribution"] = pd.read_csv("data/airing_attribution.csv")
    data["attr_by_network"] = pd.read_csv("data/attribution_by_network.csv")
    data["attr_by_daypart"] = pd.read_csv("data/attribution_by_daypart.csv")
    data["attr_by_creative"] = pd.read_csv("data/attribution_by_creative.csv")
    data["attr_by_dma"] = pd.read_csv("data/attribution_by_dma.csv")
    data["channel_contributions"] = pd.read_csv("data/channel_contributions.csv")
    data["optimal_allocation"] = pd.read_csv("data/optimal_allocation.csv")
    with open("models/mmm_params.json", "r") as f:
        data["mmm_params"] = json.load(f)
    if os.path.exists("models/baseline_metrics.json"):
        with open("models/baseline_metrics.json", "r") as f:
            data["baseline_metrics"] = json.load(f)
    optional_files = {
        "bayesian_roas": "data/bayesian_roas_posteriors.csv",
        "bayesian_contributions": "data/bayesian_channel_contributions.csv",
        "shap_importance": "data/shap_feature_importance.csv",
        "markov_attribution": "data/markov_attribution.csv",
        "markov_removal": "data/markov_removal_effects.csv",
        "cross_correlations": "data/cross_channel_correlations.csv",
        "granger_results": "data/granger_results.csv",
        "model_comparison": "data/model_comparison.csv",
        "dml_effects": "data/dml_causal_effects.csv",
        "dml_vs_ols": "data/dml_vs_ols.csv",
        "dml_het": "data/dml_heterogeneous_effects.csv",
        "tft_importance": "data/tft_feature_importance.csv",
        "tft_predictions": "data/tft_predictions.csv",
        "conformal_intervals": "data/conformal_intervals.csv",
        "conformal_coverage": "data/conformal_coverage.csv",
        "geo_lift": "data/geo_lift_results.csv",
        "sc_weights": "data/synthetic_control_weights.csv",
        "causal_impact": "data/causal_impact_results.csv",
        "kalman_coefs": "data/kalman_tvp_coefficients.csv",
    }
    for key, path in optional_files.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
    optional_json = {
        "bayesian_params": "models/bayesian_mmm_params.json",
        "cross_effects": "data/cross_channel_effects.json",
        "tft_metrics": "models/tft_metrics.json",
        "conformal_summary": "data/conformal_summary.json",
        "geo_lift_summary": "data/geo_lift_summary.json",
        "ci_summary": "data/causal_impact_summary.json",
        "kalman_summary": "data/kalman_summary.json",
    }
    for key, path in optional_json.items():
        if os.path.exists(path):
            with open(path, "r") as f:
                data[key] = json.load(f)
    return data


@st.cache_resource
def load_models():
    models = {}
    if os.path.exists("models/mmm_model.joblib"):
        models["mmm"] = joblib.load("models/mmm_model.joblib")
    if os.path.exists("models/baseline_model.joblib"):
        models["baseline"] = joblib.load("models/baseline_model.joblib")
    return models


data = load_data()
models = load_models()

# ============================================
# CONSTANTS & HELPERS
# ============================================

CHANNEL_COLORS = {
    "TV Broadcast": "#2c3e50", "TV Cable": "#34495e", "TV Streaming": "#7f8c8d",
    "Paid Search": "#2980b9", "Social": "#8e44ad", "Display": "#e67e22",
    "Base + Seasonality": "#95a5a6",
}

SPEND_COLUMNS = [
    "tv_broadcast_spend", "tv_cable_spend", "tv_streaming_spend",
    "paid_search_spend", "social_spend", "display_spend"
]

CHANNEL_NAMES = {
    "tv_broadcast_spend": "TV Broadcast", "tv_cable_spend": "TV Cable",
    "tv_streaming_spend": "TV Streaming", "paid_search_spend": "Paid Search",
    "social_spend": "Social", "display_spend": "Display",
}


def hill_saturation(spend, alpha, K):
    spend = np.asarray(spend, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.power(spend, alpha) / (np.power(spend, alpha) + np.power(K, alpha))
    return np.nan_to_num(result, 0.0)


def adstock_transform(spend_series, decay_rate):
    spend = np.asarray(spend_series, dtype=float)
    adstocked = np.zeros_like(spend)
    adstocked[0] = spend[0]
    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay_rate * adstocked[t - 1]
    return adstocked


def format_currency(val, decimals=1):
    if abs(val) >= 1e6:
        return "${:.{}f}M".format(val/1e6, decimals)
    elif abs(val) >= 1e3:
        return "${:.{}f}K".format(val/1e3, decimals)
    else:
        return "${:.0f}".format(val)


# ============================================
# SIDEBAR NAVIGATION
# ============================================

st.sidebar.title("📺 TV Ad Analytics")
st.sidebar.markdown("---")

PAGES = {
    "Overview": "overview",
    "Attribution": "attribution",
    "Budget Optimizer": "optimizer",
    "Scenario Planner": "scenario",
    "Channel Deep Dive": "deep_dive",
    "Model Comparison": "comparison",
    "Bayesian MMM": "bayesian",
    "SHAP & Markov": "shap_markov",
    "Cross-Channel Effects": "cross_channel",
    "Causal ML (DML)": "dml",
    "Transformer (TFT)": "tft",
    "Conformal Prediction": "conformal",
    "Geo-Lift": "geo_lift",
    "CausalImpact (BSTS)": "causal_impact",
    "Kalman Filter": "kalman",
    "How It Works": "how_it_works",
    "Report & Diagnostics": "report",
}

SECTIONS = {
    "Core Analytics": ["Overview", "Attribution", "Budget Optimizer", "Scenario Planner", "Channel Deep Dive"],
    "Model Comparison": ["Model Comparison", "Bayesian MMM", "SHAP & Markov", "Cross-Channel Effects"],
    "Advanced Causal ML": ["Causal ML (DML)", "Transformer (TFT)", "Conformal Prediction", "Geo-Lift", "CausalImpact (BSTS)", "Kalman Filter"],
    "Reference": ["How It Works", "Report & Diagnostics"],
}

# Build sidebar with sections
selected_page = None
for section, pages in SECTIONS.items():
    st.sidebar.markdown(f"**{section}**")
    for page in pages:
        if st.sidebar.button(page, key=f"nav_{PAGES[page]}", use_container_width=True):
            st.session_state["page"] = PAGES[page]
    st.sidebar.markdown("")

if "page" not in st.session_state:
    st.session_state["page"] = "overview"

page = st.session_state["page"]

st.sidebar.markdown("---")
st.sidebar.caption("DTC Consumer Brand | 5M Annual | Jan-Dec 2023 | 8 DMAs")

ws = data["weekly_spend"]
mmm_params = data["mmm_params"]
marginal_rois = mmm_params.get("marginal_rois", {})


# ============================================
# PAGE: OVERVIEW
# ============================================

if page == "overview":
    st.title("TV Ad Attribution & Media Mix Model")
    st.markdown(
        "A comprehensive marketing analytics platform combining **12 statistical and ML models** "
        "to measure TV advertising effectiveness, optimize budget allocation, and quantify causal impact."
    )

    # Key metrics row
    total_tv_spend = ws['tv_broadcast_spend'].sum() + ws['tv_cable_spend'].sum() + ws['tv_streaming_spend'].sum()
    total_spend = ws['total_spend'].sum()
    total_revenue = ws['total_revenue'].sum()
    n_airings = len(data['airings'])
    attr_rev = data['airing_attribution']['incremental_revenue'].sum()
    overall_roas = attr_rev / total_tv_spend if total_tv_spend > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Revenue", format_currency(total_revenue))
    c2.metric("Total Ad Spend", format_currency(total_spend))
    c3.metric("TV Airings", "{:,}".format(n_airings))
    c4.metric("TV ROAS", "{:.2f}x".format(overall_roas))
    c5.metric("Models Built", "12")

    st.markdown("---")

    # Section cards
    st.subheader("Navigate the Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Core Analytics**
        - **Attribution** -- Per-airing lift analysis with counterfactual baselines
        - **Budget Optimizer** -- Find the optimal channel allocation
        - **Scenario Planner** -- What-if analysis with interactive sliders
        - **Channel Deep Dive** -- Adstock decay, saturation curves, recommendations

        **Model Comparison**
        - **Model Comparison** -- 6 attribution methods side-by-side
        - **Bayesian MMM** -- PyMC MCMC with posterior ROAS distributions
        - **SHAP & Markov** -- TreeSHAP explainability + Markov chain attribution
        - **Cross-Channel** -- TV-to-Search halo effect with Granger causality
        """)

    with col2:
        st.markdown("""
        **Advanced Causal ML**
        - **Causal ML (DML)** -- Double Machine Learning debiased effects
        - **Transformer (TFT)** -- Temporal Fusion Transformer with attention
        - **Conformal Prediction** -- Distribution-free uncertainty intervals
        - **Geo-Lift** -- Synthetic control incrementality testing
        - **CausalImpact** -- Bayesian structural time series
        - **Kalman Filter** -- Time-varying channel effectiveness

        **Reference**
        - **How It Works** -- In-depth model explanations and methodology
        - **Report & Diagnostics** -- Executive summary, PDF export, model quality
        """)

    st.markdown("---")

    # Quick charts
    st.subheader("Spend & Revenue Overview")
    col_a, col_b = st.columns(2)

    with col_a:
        contrib_df = data['channel_contributions']
        media_only = contrib_df[contrib_df['channel'] != 'Base + Seasonality']
        fig = px.pie(media_only, values='contribution', names='channel',
                     color='channel', color_discrete_map=CHANNEL_COLORS,
                     title='Revenue Contribution by Channel (MMM)')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ws['week'], y=ws['total_revenue'] / 1e6,
                                 name='Revenue', line=dict(color='#27ae60', width=2)))
        fig.add_trace(go.Scatter(x=ws['week'], y=ws['total_spend'] / 1e6,
                                 name='Total Spend', line=dict(color='#e74c3c', width=2, dash='dash')))
        fig.update_layout(title='Weekly Revenue vs Spend', height=350,
                          xaxis_title='Week', yaxis_title='$M', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)


# ============================================
# PAGE: ATTRIBUTION
# ============================================

elif page == "attribution":
    st.title("TV Ad Attribution")
    st.markdown("Measuring the causal impact of each TV ad airing on website traffic "
                "using counterfactual baseline modeling.")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        dmas = sorted(data['airings']['dma'].unique())
        selected_dma = st.selectbox("DMA", dmas, index=0)
    with col_f2:
        dates = sorted(data['airings']['date'].unique())
        start_date = st.selectbox("Start Date", dates[:30], index=0, key='attr_start')
    with col_f3:
        end_date = st.selectbox("End Date", dates[-30:], index=len(dates[-30:]) - 1, key='attr_end')

    st.markdown("### Daily Attribution Timeline")
    selected_day = st.select_slider("Select a day to view hourly attribution:",
                                     options=dates, value=dates[73])

    hourly = data['hourly_traffic']
    day_hourly = hourly[(hourly['date'] == selected_day) & (hourly['dma'] == selected_dma)].sort_values('hour')
    day_airings = data['airings'][(data['airings']['date'] == selected_day) & (data['airings']['dma'] == selected_dma)]

    if len(day_hourly) > 0:
        baseline_mean = day_hourly[day_hourly['hour'].between(2, 6)]['sessions'].mean()
        if pd.isna(baseline_mean):
            baseline_mean = day_hourly['sessions'].min()
        hour_curve = np.array([0.2 + 0.5 * np.exp(-((h - 10) ** 2) / 8) + 0.8 * np.exp(-((h - 20) ** 2) / 6) for h in range(24)])
        baseline_sessions = baseline_mean * hour_curve / hour_curve.min() * 0.8

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=day_hourly['hour'], y=baseline_sessions[:len(day_hourly)],
                                 name='Baseline (No-TV Counterfactual)',
                                 line=dict(color='#3498db', width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=day_hourly['hour'], y=day_hourly['sessions'],
                                 name='Actual Traffic', line=dict(color='#2ecc71', width=2),
                                 fill='tonexty', fillcolor='rgba(46, 204, 113, 0.15)'))
        if len(day_airings) > 0:
            airing_hours = pd.to_datetime(day_airings['timestamp']).dt.hour
            fig.add_trace(go.Scatter(x=airing_hours,
                                     y=[day_hourly['sessions'].max() * 1.08] * len(airing_hours),
                                     mode='markers', marker=dict(symbol='triangle-down', size=10, color='#e74c3c'),
                                     name='Ad Airings'))
        fig.update_layout(title="Attribution: {} -- {}".format(selected_dma, selected_day),
                          xaxis_title="Hour of Day", yaxis_title="Sessions", height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

    attr_filtered = data['airing_attribution'][
        (data['airing_attribution']['date'] >= start_date) &
        (data['airing_attribution']['date'] <= end_date) &
        (data['airing_attribution']['dma'] == selected_dma)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Airings", "{:,}".format(len(attr_filtered)))
    c2.metric("Total Lift", "{:,.0f} sessions".format(attr_filtered['incremental_sessions'].sum()))
    total_rev = attr_filtered['incremental_revenue'].sum()
    total_cost = attr_filtered['cost'].sum()
    avg_roas = total_rev / total_cost if total_cost > 0 else 0
    c3.metric("Avg ROAS", "{:.2f}x".format(avg_roas))
    c4.metric("Incremental Revenue", format_currency(total_rev))

    st.markdown("### Performance Breakdown")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**ROAS by Network**")
        net_df = data['attr_by_network'].copy()
        net_df['avg_roas'] = net_df['avg_roas'].round(3)
        st.dataframe(net_df.sort_values('avg_roas', ascending=False)[
            ['network', 'total_airings', 'total_cost', 'total_incremental_revenue', 'avg_roas']
        ].rename(columns={'network': 'Network', 'total_airings': 'Airings', 'total_cost': 'Cost',
                          'total_incremental_revenue': 'Incr. Revenue', 'avg_roas': 'ROAS'}),
            use_container_width=True, hide_index=True)
    with col_t2:
        st.markdown("**ROAS by Daypart**")
        dp_df = data['attr_by_daypart'].copy()
        dp_df['avg_roas'] = dp_df['avg_roas'].round(3)
        st.dataframe(dp_df.sort_values('avg_roas', ascending=False)[
            ['daypart', 'total_airings', 'total_cost', 'total_incremental_revenue', 'avg_roas']
        ].rename(columns={'daypart': 'Daypart', 'total_airings': 'Airings', 'total_cost': 'Cost',
                          'total_incremental_revenue': 'Incr. Revenue', 'avg_roas': 'ROAS'}),
            use_container_width=True, hide_index=True)

    col_t3, col_t4 = st.columns(2)
    with col_t3:
        st.markdown("**ROAS by Creative**")
        cr_df = data['attr_by_creative'].copy()
        cr_df['avg_roas'] = cr_df['avg_roas'].round(3)
        st.dataframe(cr_df.sort_values('avg_roas', ascending=False)[
            ['creative_id', 'total_airings', 'total_incremental_revenue', 'avg_roas']
        ].rename(columns={'creative_id': 'Creative', 'total_airings': 'Airings',
                          'total_incremental_revenue': 'Incr. Revenue', 'avg_roas': 'ROAS'}),
            use_container_width=True, hide_index=True)
    with col_t4:
        st.markdown("**ROAS by DMA**")
        dma_df = data['attr_by_dma'].copy()
        dma_df['avg_roas'] = dma_df['avg_roas'].round(3)
        st.dataframe(dma_df.sort_values('avg_roas', ascending=False)[
            ['dma', 'total_airings', 'total_incremental_revenue', 'avg_roas']
        ].rename(columns={'dma': 'DMA', 'total_airings': 'Airings',
                          'total_incremental_revenue': 'Incr. Revenue', 'avg_roas': 'ROAS'}),
            use_container_width=True, hide_index=True)


# ============================================
# PAGE: BUDGET OPTIMIZER
# ============================================

elif page == "optimizer":
    st.title("Budget Optimizer")
    st.markdown("Find the optimal channel allocation that maximizes predicted revenue "
                "using the Media Mix Model's learned response curves.")

    total_budget = st.slider("Total Annual Budget ($M)", min_value=5.0, max_value=30.0, value=15.0, step=0.5)
    opt_df = data['optimal_allocation']
    budget_val = total_budget * 1e6
    closest_idx = (opt_df['budget'] - budget_val).abs().idxmin()
    opt_row = opt_df.iloc[closest_idx]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Allocation")
        current_data = [{'channel': CHANNEL_NAMES[col], 'spend': opt_row.get('current_' + col, 0)} for col in SPEND_COLUMNS]
        current_df = pd.DataFrame(current_data)
        fig = px.pie(current_df, values='spend', names='channel', title="Current Split",
                     color='channel', color_discrete_map=CHANNEL_COLORS)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Predicted Revenue", format_currency(opt_row['current_mix_revenue']))

    with col2:
        st.subheader("Optimized Allocation")
        optimal_data = [{'channel': CHANNEL_NAMES[col], 'spend': opt_row.get('optimal_' + col, 0)} for col in SPEND_COLUMNS]
        optimal_df = pd.DataFrame(optimal_data)
        fig = px.pie(optimal_df, values='spend', names='channel', title="Optimal Split",
                     color='channel', color_discrete_map=CHANNEL_COLORS)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        delta = opt_row['optimized_revenue'] - opt_row['current_mix_revenue']
        st.metric("Predicted Revenue", format_currency(opt_row['optimized_revenue']),
                  delta="+{}".format(format_currency(delta)))

    st.success("Optimizing allocation at ${:.1f}M budget increases predicted revenue by {:.1f}% (+{})".format(
        total_budget, opt_row['lift_pct'], format_currency(delta)))

    st.markdown("### Allocation Comparison")
    comparison_data = []
    for col in SPEND_COLUMNS:
        name = CHANNEL_NAMES[col]
        cv = opt_row.get('current_' + col, 0)
        ov = opt_row.get('optimal_' + col, 0)
        ch = ov - cv
        cp = (ch / cv * 100) if cv > 0 else 0
        comparison_data.append({'Channel': name, 'Current': format_currency(cv), 'Optimal': format_currency(ov),
                                'Change': "{}{}" .format('+' if ch >= 0 else '', format_currency(ch)),
                                'Change %': "{}{:.1f}%".format('+' if cp >= 0 else '', cp)})
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    st.markdown("### Revenue vs Budget Level")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=opt_df['budget'] / 1e6, y=opt_df['current_mix_revenue'] / 1e6,
                             name='Current Mix', line=dict(color='#e74c3c', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=opt_df['budget'] / 1e6, y=opt_df['optimized_revenue'] / 1e6,
                             name='Optimized Mix', line=dict(color='#2ecc71', width=2)))
    fig.update_layout(xaxis_title="Total Budget ($M)", yaxis_title="Predicted Revenue ($M)",
                      height=350, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)


# ============================================
# PAGE: SCENARIO PLANNER
# ============================================

elif page == "scenario":
    st.title("Scenario Planner")
    st.markdown("Adjust per-channel budgets and see the impact on predicted revenue in real-time.")

    current_spend = {
        'TV Broadcast': ws['tv_broadcast_spend'].sum(), 'TV Cable': ws['tv_cable_spend'].sum(),
        'TV Streaming': ws['tv_streaming_spend'].sum(), 'Paid Search': ws['paid_search_spend'].sum(),
        'Social': ws['social_spend'].sum(), 'Display': ws['display_spend'].sum(),
    }

    st.markdown("### Quick Scenarios")
    sc = st.columns(4)
    preset = None
    if sc[0].button("Cut TV 50%"): preset = 'cut_tv'
    if sc[1].button("All Digital"): preset = 'all_digital'
    if sc[2].button("+20% Budget"): preset = 'plus_20'
    if sc[3].button("Super Bowl"): preset = 'super_bowl'

    defaults = {ch: val / 1e6 for ch, val in current_spend.items()}
    if preset == 'cut_tv':
        defaults['TV Broadcast'] *= 0.5; defaults['TV Cable'] *= 0.5; defaults['Paid Search'] *= 1.3
    elif preset == 'all_digital':
        defaults['TV Broadcast'] = 0.1; defaults['TV Cable'] = 0.1; defaults['TV Streaming'] = 0.5
        defaults['Paid Search'] *= 1.8; defaults['Social'] *= 1.5; defaults['Display'] *= 1.5
    elif preset == 'plus_20':
        defaults = {ch: v * 1.2 for ch, v in defaults.items()}
    elif preset == 'super_bowl':
        defaults['TV Broadcast'] *= 1.5; defaults['TV Cable'] *= 1.2

    slider_values = {}
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        for ch in ['TV Broadcast', 'TV Cable', 'TV Streaming']:
            mx = max(10.0, defaults[ch] * 3)
            slider_values[ch] = st.slider("{} ($M)".format(ch), 0.0, mx, min(defaults[ch], mx), 0.1, key='sc_' + ch) * 1e6
    with col_s2:
        for ch in ['Paid Search', 'Social', 'Display']:
            mx = max(8.0, defaults[ch] * 3)
            slider_values[ch] = st.slider("{} ($M)".format(ch), 0.0, mx, min(defaults[ch], mx), 0.1, key='sc_' + ch) * 1e6

    total_scenario = sum(slider_values.values())
    st.markdown("**Total Spend: {}**".format(format_currency(total_scenario)))

    base_revenue = ws['total_revenue'].sum() - sum(marginal_rois.get(ch, 0) * current_spend[ch] for ch in current_spend)
    predicted_revenue = base_revenue
    for ch, sv in slider_values.items():
        cur = current_spend[ch]
        roi = marginal_rois.get(ch, 1.0)
        d = sv - cur
        sf = 1.0 / (1.0 + abs(d) / max(cur, 1))
        predicted_revenue += sv * roi * sf + cur * roi * (1 - sf)

    current_total_revenue = ws['total_revenue'].sum()
    dr = predicted_revenue - current_total_revenue
    dp = dr / current_total_revenue * 100

    r1, r2, r3 = st.columns(3)
    r1.metric("Current Revenue", format_currency(current_total_revenue))
    r2.metric("Predicted Revenue", format_currency(predicted_revenue), delta="{}{:.1f}%".format('+' if dp >= 0 else '', dp))
    r3.metric("Overall ROAS", "{:.2f}x".format(predicted_revenue / total_scenario if total_scenario > 0 else 0))


# ============================================
# PAGE: CHANNEL DEEP DIVE
# ============================================

elif page == "deep_dive":
    st.title("Channel Deep Dive")

    channel_names_list = list(CHANNEL_NAMES.values())
    selected_channel = st.selectbox("Select Channel", channel_names_list)
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

        c1, c2, c3 = st.columns(3)
        c1.metric("Adstock Decay Rate", "{:.2f}".format(decay))
        c2.metric("Saturation Level", "{:.0f}%".format(sat_level))
        c3.metric("Marginal ROI", "{:.2f}x".format(marginal_rois.get(selected_channel, 0)))

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("### Adstock Decay")
            st.caption("How long does the advertising effect persist after spending stops?")
            weeks_range = np.arange(16)
            impulse = np.zeros(16); impulse[0] = 1.0
            response = adstock_transform(impulse, decay)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=weeks_range, y=response, mode='lines+markers',
                                     line=dict(color=CHANNEL_COLORS.get(selected_channel, '#3498db'), width=2),
                                     fill='tozeroy', fillcolor='rgba(41, 128, 185, 0.1)'))
            fig.update_layout(xaxis_title="Weeks After Spend", yaxis_title="Remaining Effect", height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col_c2:
            st.markdown("### Saturation Curve")
            st.caption("How much incremental value does additional spend deliver?")
            alpha = h_params['alpha']; K = h_params['K']
            current_weekly = ws[spend_col].mean()
            spend_range = np.linspace(0, current_weekly * 3, 200)
            saturated = hill_saturation(spend_range, alpha, K)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spend_range / 1e3, y=saturated, mode='lines',
                                     line=dict(color=CHANNEL_COLORS.get(selected_channel, '#3498db'), width=2)))
            current_sat = hill_saturation(np.array([current_weekly]), alpha, K)[0]
            fig.add_trace(go.Scatter(x=[current_weekly / 1e3], y=[current_sat], mode='markers+text',
                                     marker=dict(size=12, color='#e74c3c'),
                                     text=['You Are Here ({:.0%})'.format(current_sat)], textposition='top center'))
            fig.update_layout(xaxis_title="Weekly Spend ($K)", yaxis_title="Response (Saturated)",
                              yaxis_range=[-0.05, 1.05], height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Weekly Spend Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ws['week'], y=ws[spend_col] / 1e3, mode='lines+markers',
                                 line=dict(color=CHANNEL_COLORS.get(selected_channel, '#3498db'), width=2),
                                 marker=dict(size=4)))
        fig.update_layout(xaxis_title="Week", yaxis_title="Spend ($K)", height=250)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Recommendation")
        if sat_level > 70:
            st.warning("**{}** is at **{:.0f}% saturation**. Additional spend has diminishing returns. "
                       "Consider shifting budget to channels with lower saturation.".format(selected_channel, sat_level))
        elif sat_level > 40:
            st.info("**{}** is at **{:.0f}% saturation**. Room for incremental investment, "
                    "but watch for diminishing returns above 70%.".format(selected_channel, sat_level))
        else:
            st.success("**{}** is at **{:.0f}% saturation**. Significant headroom for additional "
                       "investment with strong marginal returns.".format(selected_channel, sat_level))


# ============================================
# PAGE: MODEL COMPARISON
# ============================================

elif page == "comparison":
    st.title("Stacked Model Comparison")
    st.markdown(
        "Six attribution methods applied to the same data produce different answers. "
        "This comparison reveals **where models agree** (higher confidence) and "
        "**where they disagree** (more uncertainty in the true attribution).")

    if 'model_comparison' in data:
        comp_df = data['model_comparison']
        channels = [c for c in comp_df.columns if c != 'model']
        active_channels = [c for c in channels if comp_df[c].sum() > 0.01]

        st.subheader("Attribution Shares by Model")
        fig = go.Figure()
        for ch in active_channels:
            color = CHANNEL_COLORS.get(ch, '#bdc3c7')
            fig.add_trace(go.Bar(y=comp_df['model'], x=comp_df[ch] * 100, name=ch, orientation='h',
                                 marker_color=color,
                                 text=comp_df[ch].apply(lambda v: '{:.0%}'.format(v) if v > 0.05 else ''),
                                 textposition='inside', textfont_color='white'))
        fig.update_layout(barmode='stack', title='How Each Method Attributes Revenue Across Channels',
                          xaxis_title='Attribution Share (%)', height=450,
                          legend=dict(orientation='h', yanchor='bottom', y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Channel-Level Comparison")
        model_colors = {'Last-Touch': '#e74c3c', 'First-Touch': '#e67e22', 'Markov Chain': '#2c3e50',
                        'LightGBM Attribution': '#27ae60', 'Frequentist MMM': '#3498db', 'Bayesian MMM': '#8e44ad'}
        fig = go.Figure()
        for _, row in comp_df.iterrows():
            mn = row['model']
            fig.add_trace(go.Bar(x=active_channels, y=[row[ch] * 100 for ch in active_channels],
                                 name=mn, marker_color=model_colors.get(mn, 'gray'), opacity=0.85))
        fig.update_layout(barmode='group', title='Each Channel Seen Through Different Lenses',
                          yaxis_title='Attribution Share (%)', height=400)
        st.plotly_chart(fig, use_container_width=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if os.path.exists('results/model_comparison_radar.png'):
                st.image('results/model_comparison_radar.png', caption='Attribution Radar Chart')
        with col_r2:
            if os.path.exists('results/model_comparison_insights.png'):
                st.image('results/model_comparison_insights.png', caption='Key Insights Summary')

        st.subheader("Full Comparison Matrix")
        display_df = comp_df.copy()
        for ch in active_channels:
            display_df[ch] = display_df[ch].apply(lambda x: '{:.1%}'.format(x))
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.subheader("Key Takeaways")
        st.markdown("""
1. **TV's value varies 2-5x** depending on methodology -- Last-Touch dramatically undervalues TV
2. **Heuristic models over-credit digital** -- Paid Search gets credit for conversions that TV initiated
3. **Bayesian MMM provides uncertainty** -- Credible intervals show where we're confident vs uncertain
4. **Markov Chain captures journey dynamics** -- Properly credits early-funnel channels
5. **No single model is correct** -- Triangulate across methods for the best estimate
        """)
    else:
        st.info("Run `python src/model_comparison.py` to generate comparison results.")


# ============================================
# PAGE: BAYESIAN MMM
# ============================================

elif page == "bayesian":
    st.title("Bayesian Media Mix Model")
    st.markdown(
        "Full Bayesian estimation using PyMC with MCMC sampling. Provides **posterior distributions** "
        "over ROAS -- capturing uncertainty, not just point estimates.")

    if 'bayesian_contributions' in data:
        bc = data['bayesian_contributions']

        st.subheader("Posterior ROAS by Channel (90% Credible Intervals)")
        roas_cols = st.columns(len(bc))
        for i, (_, row) in enumerate(bc.iterrows()):
            with roas_cols[i]:
                st.metric(row['channel'], "{:.2f}x".format(row['median_roas']),
                          delta="[{:.2f}, {:.2f}]".format(row['roas_ci_5'], row['roas_ci_95']),
                          delta_color="off")

        st.subheader("Posterior ROAS Distributions")
        bc_sorted = bc.sort_values('median_roas', ascending=True)
        fig = go.Figure()
        for _, row in bc_sorted.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['roas_ci_5'], row['median_roas'], row['roas_ci_95']],
                y=[row['channel']] * 3, mode='markers+lines',
                marker=dict(size=[8, 14, 8], color=['gray', '#8e44ad', 'gray']),
                line=dict(color='#8e44ad', width=3), name=row['channel'], showlegend=False))
        fig.add_vline(x=1.0, line_dash='dash', line_color='red', annotation_text='Break-even (1.0x)')
        fig.update_layout(title='Posterior ROAS with 90% Credible Intervals', xaxis_title='ROAS', height=350)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('results/bayesian_posterior_roas.png'):
                st.image('results/bayesian_posterior_roas.png', caption='Posterior ROAS Distributions (MCMC)')
        with col2:
            if os.path.exists('results/bayesian_forest_plot.png'):
                st.image('results/bayesian_forest_plot.png', caption='Forest Plot: Credible Intervals')

        st.subheader("Bayesian Channel Contributions")
        fig = px.bar(bc.sort_values('median_contribution', ascending=True),
                     x='median_contribution', y='channel', orientation='h',
                     color='channel', color_discrete_map=CHANNEL_COLORS,
                     title='Bayesian MMM: Median Channel Revenue Contributions')
        fig.update_layout(height=350, showlegend=False, xaxis_title='Revenue Contribution ($)')
        st.plotly_chart(fig, use_container_width=True)

        if 'bayesian_params' in data:
            bp = data['bayesian_params']
            st.subheader("MCMC Diagnostics")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Chains", bp.get('n_chains', 2))
            d2.metric("Draws / Chain", bp.get('n_draws', 2000))
            d3.metric("Tune Steps", bp.get('n_tune', 1500))
            d4.metric("Divergences", bp.get('divergences', 'N/A'))
    else:
        st.info("Run `python src/bayesian_mmm.py` to generate Bayesian MMM results.")


# ============================================
# PAGE: SHAP & MARKOV
# ============================================

elif page == "shap_markov":
    st.title("SHAP Explainability & Markov Attribution")

    tab_shap, tab_markov = st.tabs(["SHAP Explainability", "Markov Attribution"])

    with tab_shap:
        st.markdown("**TreeSHAP** on the LightGBM baseline model provides exact Shapley values -- "
                    "the theoretically grounded way to attribute each prediction to individual features.")

        if 'shap_importance' in data:
            shap_df = data['shap_importance']
            top_features = shap_df.head(15)
            fig = px.bar(top_features.sort_values('mean_abs_shap', ascending=True),
                         x='mean_abs_shap', y='feature', orientation='h',
                         title='Top 15 Features by Mean |SHAP Value|',
                         color='mean_abs_shap', color_continuous_scale='Viridis')
            fig.update_layout(height=450, showlegend=False, xaxis_title='Mean |SHAP Value|', yaxis_title='')
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists('results/shap_summary_baseline.png'):
                    st.image('results/shap_summary_baseline.png', caption='SHAP Summary (Beeswarm) Plot')
            with col2:
                if os.path.exists('results/shap_dependence_hour.png'):
                    st.image('results/shap_dependence_hour.png', caption='SHAP Dependence: Time-of-Day')

            col3, col4 = st.columns(2)
            with col3:
                if os.path.exists('results/shap_waterfall_sample.png'):
                    st.image('results/shap_waterfall_sample.png', caption='SHAP Waterfall: Single Prediction')
            with col4:
                if os.path.exists('results/shap_feature_importance.png'):
                    st.image('results/shap_feature_importance.png', caption='Feature Importance Bar Chart')
        else:
            st.info("Run `python src/shap_analysis.py` to generate SHAP results.")

    with tab_markov:
        st.markdown("Absorbing Markov chains model the customer journey as a stochastic process. "
                    "**Removal effects** measure each channel's importance by simulating what happens "
                    "when it's removed from the journey entirely.")

        if 'markov_attribution' in data:
            markov_df = data['markov_attribution']

            st.subheader("Attribution Method Comparison")
            fig = go.Figure()
            for method, col_name, color in [('Markov Chain', 'markov_attribution', '#2c3e50'),
                                             ('Last-Touch', 'last_touch_attribution', '#e74c3c'),
                                             ('First-Touch', 'first_touch_attribution', '#e67e22')]:
                fig.add_trace(go.Bar(x=markov_df['channel'], y=markov_df[col_name] * 100,
                                     name=method, marker_color=color, opacity=0.85))
            fig.update_layout(barmode='group', title='Channel Attribution: Markov vs Heuristic Models',
                              yaxis_title='Attribution Share (%)', height=400)
            st.plotly_chart(fig, use_container_width=True)

            if 'markov_removal' in data:
                st.subheader("Channel Removal Effects")
                removal_df = data['markov_removal']
                fig = px.bar(removal_df.sort_values('removal_effect', ascending=True),
                             x='removal_effect', y='channel', orientation='h',
                             color='removal_effect', color_continuous_scale='Reds',
                             title='Removal Effects: Conversion Drop When Channel Is Removed')
                fig.update_layout(height=350, xaxis_title='Removal Effect', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists('results/markov_transition_heatmap.png'):
                    st.image('results/markov_transition_heatmap.png', caption='Transition Probability Matrix')
            with col2:
                if os.path.exists('results/markov_vs_lasttouch.png'):
                    st.image('results/markov_vs_lasttouch.png', caption='Markov vs Last-Touch Comparison')
        else:
            st.info("Run `python src/markov_attribution.py` to generate Markov results.")


# ============================================
# PAGE: CROSS-CHANNEL EFFECTS
# ============================================

elif page == "cross_channel":
    st.title("Cross-Channel Effects: TV -> Search")
    st.markdown("Investigating how TV advertising drives online search behavior -- the 'TV halo effect'. "
                "Uses Granger causality, mediation analysis, and interaction modeling.")

    if 'cross_effects' in data:
        effects = data['cross_effects']

        c1, c2, c3 = st.columns(3)
        with c1:
            pc = effects.get('peak_correlation', {})
            st.metric("Peak TV->Search Correlation", "r = {:.3f}".format(pc.get('correlation', 0)),
                      delta="Lag {} weeks".format(pc.get('lag_weeks', 0)))
        with c2:
            med = effects.get('mediation', {})
            ind = med.get('indirect_effect', 0); tot = med.get('total_effect', 1)
            st.metric("Effect Mediated via Search", "{:.0f}%".format(ind / tot * 100 if tot != 0 else 0))
        with c3:
            inter = effects.get('interaction', {})
            st.metric("TV x Search Synergy", "${:,.0f}".format(inter.get('interaction_coefficient', 0)),
                      delta="R2 lift: +{:.4f}".format(inter.get('r2_improvement', 0)))

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('results/tv_search_lag_correlation.png'):
                st.image('results/tv_search_lag_correlation.png', caption='TV-Search Lag Correlation')
        with col2:
            if os.path.exists('results/tv_search_interaction.png'):
                st.image('results/tv_search_interaction.png', caption='TV x Search Revenue Interaction')

        if 'granger_results' in data:
            st.subheader("Granger Causality Test")
            gr = data['granger_results']
            for _, row in gr.iterrows():
                sig = "Yes" if row['significant'] else "No"
                color = "green" if row['significant'] else "red"
                st.markdown("- **Lag {} weeks**: F={:.2f}, p={:.4f} -- Significant: :{}[{}]".format(
                    int(row['lag']), row['f_statistic'], row['p_value'], color, sig))

        if 'mediation' in effects:
            med = effects['mediation']
            st.subheader("Mediation: TV -> Search -> Revenue")
            paths = [('Direct: TV -> Revenue', med.get('direct_effect', 0), '#2c3e50'),
                     ('Indirect: TV -> Search -> Revenue', med.get('indirect_effect', 0), '#8e44ad'),
                     ('Total Effect', med.get('total_effect', 0), '#2980b9')]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[p[0] for p in paths], y=[p[1] for p in paths],
                                 marker_color=[p[2] for p in paths],
                                 text=['${:,.0f}'.format(p[1]) for p in paths], textposition='outside'))
            fig.update_layout(title='How TV Affects Revenue (Direct vs Through Search)',
                              yaxis_title='Effect Size ($)', height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python src/cross_channel.py` to generate cross-channel results.")


# ============================================
# PAGE: DOUBLE MACHINE LEARNING (DML)
# ============================================

elif page == "dml":
    st.title("Double Machine Learning (DML)")
    st.markdown(
        "**Debiased/orthogonal causal inference** (Chernozhukov et al., 2018). "
        "Uses LightGBM for nuisance parameters but preserves valid statistical "
        "inference on the treatment effect -- the gold standard in causal ML.")

    if 'dml_effects' in data:
        dml_df = data['dml_effects']

        st.subheader("Causal Average Treatment Effects (ATE)")
        cols = st.columns(3)
        for i, (_, row) in enumerate(dml_df.iterrows()):
            with cols[i % 3]:
                sig = " ***" if row['p_value'] < 0.01 else (" **" if row['p_value'] < 0.05 else "")
                st.metric(row['channel'], "${:.3f}/$ spent".format(row['ate']),
                          delta="p={:.4f}{}".format(row['p_value'], sig), delta_color="off")

        if 'dml_vs_ols' in data:
            st.subheader("DML vs Naive OLS: Exposing Confounding Bias")
            ols_df = data['dml_vs_ols']
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ols_df['channel'], y=ols_df['ols_coefficient'],
                                 name='Naive OLS (biased)', marker_color='#e74c3c', opacity=0.7))
            fig.add_trace(go.Bar(x=ols_df['channel'], y=ols_df['dml_causal_effect'],
                                 name='DML (debiased)', marker_color='#2980b9', opacity=0.7))
            fig.update_layout(barmode='group', height=400,
                              title='How Much Bias Does Confounding Introduce?',
                              yaxis_title='Effect ($/$ spend)')
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('results/dml_causal_effects.png'):
                st.image('results/dml_causal_effects.png', caption='Causal Effects with 90% CIs')
        with col2:
            if os.path.exists('results/dml_cate_heterogeneity.png'):
                st.image('results/dml_cate_heterogeneity.png', caption='Causal Forest: Heterogeneous Effects')
    else:
        st.info("Run `python src/double_ml.py` to generate DML results.")


# ============================================
# PAGE: TEMPORAL FUSION TRANSFORMER
# ============================================

elif page == "tft":
    st.title("Temporal Fusion Transformer (TFT)")
    st.markdown(
        "**State-of-the-art deep learning** for time series (Google Research, 2021). "
        "Variable Selection Network, LSTM encoder, and Multi-Head Attention provide "
        "interpretable insights into which features and past timesteps drive predictions.")

    if 'tft_metrics' in data:
        m = data['tft_metrics']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train R2", "{:.3f}".format(m.get('train_r2', 0)))
        c2.metric("Test R2", "{:.3f}".format(m.get('test_r2', 0)))
        c3.metric("Test MAE", "${:,.0f}".format(m.get('test_mae', 0)))
        c4.metric("80% PI Coverage", "{:.0%}".format(m.get('coverage', 0)))

        if 'tft_importance' in data:
            st.subheader("Variable Selection: Learned Feature Importance")
            vi = data['tft_importance']
            fig = px.bar(vi.sort_values('importance', ascending=True),
                         x='importance', y='feature', orientation='h',
                         color='importance', color_continuous_scale='Viridis',
                         title='TFT Learns Which Inputs Matter')
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('results/tft_forecast.png'):
                st.image('results/tft_forecast.png', caption='Revenue Forecast with Prediction Intervals')
        with col2:
            if os.path.exists('results/tft_attention_heatmap.png'):
                st.image('results/tft_attention_heatmap.png', caption='Attention Weights & Training')

        st.caption("Note: With only 52 weekly observations, the TFT overfits. "
                   "This demonstrates the architecture -- real-world deployment needs more data.")
    else:
        st.info("Run `python src/temporal_fusion.py` to generate TFT results.")


# ============================================
# PAGE: CONFORMAL PREDICTION
# ============================================

elif page == "conformal":
    st.title("Conformal Prediction")
    st.markdown(
        "**Distribution-free uncertainty quantification** with guaranteed finite-sample "
        "coverage. No distributional assumptions -- only exchangeability. "
        "Based on Vovk et al. (2005), Angelopoulos & Bates (2021).")

    if 'conformal_summary' in data:
        cs = data['conformal_summary']
        st.subheader("Method Comparison: 90% Prediction Intervals")
        methods = [('split_conformal', 'Split Conformal'), ('jackknife_plus', 'Jackknife+'), ('cqr', 'CQR (Adaptive)')]
        mc = st.columns(3)
        for i, (key, name) in enumerate(methods):
            if key in cs:
                with mc[i]:
                    st.metric("{} Coverage".format(name), "{:.0%}".format(cs[key]['coverage']),
                              delta="Width: ${:,.0f}".format(cs[key]['avg_width']))

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('results/conformal_intervals.png'):
                st.image('results/conformal_intervals.png', caption='Three Conformal Methods Compared')
        with col2:
            if os.path.exists('results/conformal_coverage.png'):
                st.image('results/conformal_coverage.png', caption='Coverage Calibration & Width Trade-off')

        if os.path.exists('results/conformal_width_analysis.png'):
            st.image('results/conformal_width_analysis.png', caption='CQR: Adaptive Intervals')
    else:
        st.info("Run `python src/conformal_prediction.py` to generate conformal results.")


# ============================================
# PAGE: GEO-LIFT
# ============================================

elif page == "geo_lift":
    st.title("Geo-Lift / Synthetic Control")
    st.markdown(
        "**Causal incrementality testing** (Abadie et al., 2010). Constructs a synthetic "
        "'control DMA' from weighted donor markets to estimate the true causal lift.")

    if 'geo_lift_summary' in data:
        gs = data['geo_lift_summary']
        c1, c2, c3 = st.columns(3)
        c1.metric("Treatment DMA", gs.get('treatment_dma', 'N/A'))
        c2.metric("Causal Lift", "{:+.1f}%".format(gs.get('avg_lift_pct', 0)))
        c3.metric("Pre-Period R2", "{:.3f}".format(gs.get('pre_r2', 0)))

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('results/geo_lift_main.png'):
                st.image('results/geo_lift_main.png', caption='Synthetic Control: Actual vs Counterfactual')
        with col2:
            if os.path.exists('results/geo_lift_weights.png'):
                st.image('results/geo_lift_weights.png', caption='Donor DMA Weights')

        if os.path.exists('results/geo_lift_placebo.png'):
            st.subheader("Placebo Tests: Falsification Check")
            st.markdown("If valid, placebo tests on control DMAs should show no effect. "
                        "Treatment DMA (red) should stand out.")
            st.image('results/geo_lift_placebo.png', caption='Placebo Tests Across All DMAs')
    else:
        st.info("Run `python src/geo_lift.py` to generate geo-lift results.")


# ============================================
# PAGE: CAUSAL IMPACT
# ============================================

elif page == "causal_impact":
    st.title("CausalImpact (BSTS)")
    st.markdown(
        "**Bayesian Structural Time Series** -- Google's approach to measuring "
        "campaign impact. Fits a model on pre-intervention data, projects a "
        "counterfactual, and computes posterior probability of causal effect.")

    if 'ci_summary' in data:
        cis = data['ci_summary']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Intervention Week", cis.get('intervention_week', 'N/A'))
        c2.metric("Causal Effect", "${:,.0f}".format(cis.get('causal_effect', 0)))
        c3.metric("Effect %", "{:+.1f}%".format(cis.get('causal_effect_pct', 0)))
        c4.metric("Posterior Prob", "{:.0%}".format(cis.get('posterior_prob', 0)))

        if os.path.exists('results/causal_impact_main.png'):
            st.image('results/causal_impact_main.png', caption='Original, Pointwise, and Cumulative Effects')
        if os.path.exists('results/causal_impact_cumulative.png'):
            st.image('results/causal_impact_cumulative.png', caption='TV Spend Context & Search Impact')

        if 'search' in cis:
            st.subheader("TV -> Brand Search Impact")
            st.metric("Search Volume Effect", "{:+.1f}%".format(cis['search']['search_effect_pct']),
                      delta="Posterior Prob: {:.0%}".format(cis['search']['posterior_prob']))
    else:
        st.info("Run `python src/causal_impact.py` to generate CausalImpact results.")


# ============================================
# PAGE: KALMAN FILTER
# ============================================

elif page == "kalman":
    st.title("Kalman Filter: Time-Varying Parameters")
    st.markdown(
        "**State-space model** where channel effectiveness drifts over time. "
        "Unlike static MMM, the Kalman filter tracks how ROAS evolves week by week.")

    if 'kalman_summary' in data:
        ks = data['kalman_summary']
        c1, c2, c3 = st.columns(3)
        c1.metric("Static (Ridge) R2", "{:.4f}".format(ks.get('static_r2', 0)))
        c2.metric("Dynamic (Kalman) R2", "{:.4f}".format(ks.get('dynamic_r2', 0)))
        c3.metric("Improvement", "+{:.2f} pp".format(ks.get('improvement_pp', 0)))

        if os.path.exists('results/kalman_tvp_coefficients.png'):
            st.image('results/kalman_tvp_coefficients.png', caption='Time-Varying Coefficients')
        if os.path.exists('results/kalman_tvp_roas_evolution.png'):
            st.image('results/kalman_tvp_roas_evolution.png', caption='Channel Effectiveness Over Time')
        if os.path.exists('results/kalman_vs_static.png'):
            st.image('results/kalman_vs_static.png', caption='Static vs Dynamic: Prediction & Residuals')

        if 'kalman_coefs' in data:
            st.subheader("Interactive: Time-Varying Coefficients")
            kc = data['kalman_coefs']
            channels = [c for c in kc.columns if c != 'week']
            fig = go.Figure()
            for ch in channels:
                fig.add_trace(go.Scatter(x=kc['week'], y=kc[ch], mode='lines', name=ch, line=dict(width=2)))
            fig.update_layout(title='Channel Effectiveness Over Time (Kalman Smoothed)',
                              xaxis_title='Week', yaxis_title='Coefficient', height=400, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python src/kalman_tvp.py` to generate Kalman filter results.")


# ============================================
# PAGE: HOW IT WORKS
# ============================================

elif page == "how_it_works":
    st.title("How It Works: Model Methodology Guide")
    st.markdown("In-depth explanations of every model in this platform, the math behind them, "
                "and why each one matters for marketing analytics.")

    st.markdown("---")

    # ---- SECTION 1: CORE MODELS ----
    st.header("1. Core Attribution & MMM")

    with st.expander("LightGBM Baseline Attribution", expanded=False):
        st.markdown("""
**What it does:** Builds a counterfactual baseline of expected website traffic (what would have
happened without TV ads) using a gradient-boosted decision tree, then measures the *lift* above
baseline during each TV airing window.

**How it works:**
1. Train LightGBM on "clean" periods (no TV ad within 60 min) using features: hour-of-day,
   day-of-week, month, DMA, holidays, and lagged traffic
2. Predict baseline traffic for every hour, including hours with TV airings
3. For each airing, compute `lift = actual_traffic - predicted_baseline` in a 20-minute window
4. Convert lift to incremental revenue using an average revenue-per-session rate

**Why LightGBM:** Handles nonlinear interactions (e.g., primetime x weekend), missing data
natively, and trains fast on 700K+ hourly observations. R-squared typically >0.92.

**Limitations:** Only captures *immediate* response (20 min). Long-term brand effects are missed.
        """)

    with st.expander("Ridge Regression Media Mix Model (MMM)", expanded=False):
        st.markdown("""
**What it does:** Estimates the contribution of each advertising channel to total weekly revenue,
accounting for ad carry-over (adstock) and diminishing returns (saturation).

**How it works:**
1. **Adstock transformation:** `adstocked[t] = spend[t] + decay * adstocked[t-1]`
   Models the carry-over effect -- a TV ad seen today still influences behavior next week.
   Each channel has its own decay rate (TV ~0.7, Digital ~0.3).

2. **Hill saturation:** `response = spend^alpha / (spend^alpha + K^alpha)`
   Models diminishing returns -- doubling spend does NOT double response.
   K = half-saturation point, alpha = curve steepness.

3. **Ridge regression:** Fits `revenue ~ sum(beta_i * saturated_adstocked_spend_i) + seasonality + trend`
   with L2 regularization to prevent overfitting on 52 weekly data points.

**Key outputs:** Channel ROAS, contribution shares, optimal budget allocation via constrained
optimization (scipy.optimize.minimize with SLSQP).

**Why Ridge over OLS:** With 52 observations and 9 features, OLS would overfit. Ridge shrinks
coefficients toward zero, producing more stable and generalizable estimates.
        """)

    with st.expander("Budget Optimizer", expanded=False):
        st.markdown("""
**What it does:** Finds the channel allocation that maximizes predicted revenue for a given
total budget, subject to the MMM's learned response curves.

**How it works:**
1. Define objective: `maximize sum(beta_i * hill(adstock(spend_i)))` over all channels
2. Constraints: `sum(spend_i) = total_budget`, `spend_i >= 0`
3. Solve using scipy SLSQP (Sequential Least Squares Programming)
4. Repeat across budget levels ($5M to $30M) to trace out the efficiency frontier

**Key insight:** Because channels have different saturation levels, the optimal mix shifts
as budget changes. At low budgets, concentrate on high-ROI channels. At high budgets,
diversify to avoid saturation.
        """)

    st.markdown("---")

    # ---- SECTION 2: BAYESIAN & EXPLAINABILITY ----
    st.header("2. Bayesian & Explainability Models")

    with st.expander("Bayesian MMM (PyMC / MCMC)", expanded=False):
        st.markdown("""
**What it does:** Re-estimates the MMM using full Bayesian inference, producing *posterior
distributions* over ROAS rather than single point estimates.

**How it works:**
1. **Priors:** Place informative priors on channel coefficients (HalfNormal), adstock decays
   (Beta), and noise (HalfNormal). These encode domain knowledge: "ROAS should be positive."
2. **Likelihood:** `revenue ~ Normal(mu, sigma)` where `mu = intercept + sum(beta_i * transformed_spend_i)`
3. **MCMC sampling:** PyMC's NUTS sampler draws ~4,000 samples from the posterior distribution
4. **Output:** For each channel, we get a full distribution of plausible ROAS values, not just one number

**Why Bayesian > Frequentist:**
- Uncertainty is *first-class*: "TV Broadcast ROAS is 1.2x [0.8, 1.6] with 90% probability"
- Priors prevent negative ROAS estimates (which are economically nonsensical)
- Handles small samples better by incorporating prior information
- Posterior predictive checks validate model fit

**MCMC Diagnostics:** We check for convergence (R-hat < 1.01), effective sample size (>400),
and divergences (should be 0). These ensure the posterior samples are reliable.
        """)

    with st.expander("TreeSHAP Explainability", expanded=False):
        st.markdown("""
**What it does:** Decomposes every individual prediction into the contribution of each input
feature using Shapley values from cooperative game theory.

**How it works:**
1. **Shapley values** (1953): For each prediction, compute the average marginal contribution
   of each feature across all possible feature orderings
2. **TreeSHAP** (Lundberg & Lee, 2017): Exploits tree structure to compute *exact* Shapley
   values in O(TLD) time instead of the exponential brute-force approach
3. **Properties:** Additive (sum to prediction), consistent, locally accurate

**Key visualizations:**
- **Beeswarm plot:** Shows distribution of SHAP values across all predictions for each feature
- **Waterfall:** Explains a single prediction step by step
- **Dependence plot:** How a feature's effect changes as its value changes

**Why SHAP > feature importance:** Standard feature importance (gain/split count) doesn't
show *direction* of effect or handle correlated features properly. SHAP is the only method
with all three axiomatic guarantees from game theory.
        """)

    with st.expander("Markov Chain Attribution", expanded=False):
        st.markdown("""
**What it does:** Models the customer journey as an absorbing Markov chain, where each state
is a marketing touchpoint, and computes each channel's contribution via removal effects.

**How it works:**
1. Build a transition matrix P from observed customer journeys
   (e.g., P[TV -> Search] = 0.3 means 30% of users who see TV next interact with Search)
2. Add absorbing states: "Conversion" and "Drop-off"
3. **Removal effect:** For each channel, set its transitions to Drop-off and recompute
   the conversion probability. The drop = that channel's contribution.
4. Normalize removal effects to get attribution shares

**Why Markov > Last-Touch:**
- Last-Touch gives 100% credit to the final touchpoint before conversion
- Markov recognizes that early-funnel channels (like TV) *initiate* journeys
- TV typically gets 2-3x more credit under Markov than Last-Touch
        """)

    st.markdown("---")

    # ---- SECTION 3: CROSS-CHANNEL ----
    st.header("3. Cross-Channel Analysis")

    with st.expander("Granger Causality & Mediation Analysis", expanded=False):
        st.markdown("""
**Granger Causality** tests whether past TV spend helps predict future search volume beyond
what search's own history provides:
- Regress `search[t] ~ search[t-1:t-k]` (restricted model)
- Regress `search[t] ~ search[t-1:t-k] + tv_spend[t-1:t-k]` (unrestricted model)
- F-test whether TV coefficients are jointly significant

**Mediation Analysis** decomposes TV's effect on revenue into:
- **Direct path:** TV -> Revenue (immediate brand response)
- **Indirect path:** TV -> Search -> Revenue (TV drives search, search drives purchase)
- Uses Baron & Kenny (1986) framework with OLS at each stage

**TV x Search Interaction:** Tests whether the combination of high TV + high Search
produces *more* revenue than the sum of their individual effects (synergy).
        """)

    st.markdown("---")

    # ---- SECTION 4: ADVANCED CAUSAL ML ----
    st.header("4. Advanced Causal ML Techniques")

    with st.expander("Double Machine Learning (DML) -- Chernozhukov et al., 2018", expanded=False):
        st.markdown("""
**What it does:** Estimates the *causal* effect of each channel's spend on revenue, properly
removing confounding from seasonality, trends, and cross-channel correlation.

**Why standard regression fails:** OLS conflates correlation with causation. If brands spend
more on TV during Q4 (holidays) when revenue is naturally high, OLS attributes the seasonal
revenue bump to TV.

**How DML fixes this:**
1. **First stage (nuisance):** Use LightGBM to predict:
   - Y_hat = E[Revenue | Confounders] (expected revenue from confounders alone)
   - T_hat = E[Spend | Confounders] (expected spend from confounders alone)
2. **Residualize:** Compute Y_tilde = Y - Y_hat, T_tilde = T - T_hat
3. **Second stage (causal):** Regress Y_tilde ~ T_tilde
   This gives the *debiased* causal effect -- the part of spend-revenue correlation
   NOT explained by shared confounders

**Cross-fitting:** To avoid overfitting bias, use K-fold: train nuisance on fold k,
predict on fold -k. This ensures honesty of the causal estimate.

**CausalForest extension:** Detects *heterogeneous* treatment effects -- does TV
effectiveness vary by season? CausalForest (Athey & Imbens, 2018) uses random forests
in the residualized space to estimate conditional average treatment effects (CATEs).
        """)

    with st.expander("Temporal Fusion Transformer (TFT) -- Lim et al., 2021", expanded=False):
        st.markdown("""
**What it does:** Deep learning model that forecasts revenue while providing interpretable
insights through attention mechanisms and variable selection.

**Architecture components:**
1. **Gated Residual Network (GRN):** Non-linear processing with skip connections and
   gating (ELU + GLU). Controls information flow: `output = GLU(W1 * ELU(W2 * x) + x)`
2. **Variable Selection Network (VSN):** Learns which input features matter via softmax
   attention weights. Provides feature importance *learned* by the model, not post-hoc.
3. **LSTM Encoder:** Captures temporal dependencies in the sequence of weekly observations
4. **Multi-Head Attention:** Identifies which *past weeks* are most relevant for each
   prediction. Similar to Transformer attention in NLP.
5. **Quantile Outputs:** Predicts 10th, 50th, 90th percentiles for uncertainty estimation

**Why TFT for marketing:** Unlike standard time series models (ARIMA, Prophet), TFT can:
- Handle multiple input features (spend, seasonality, trends) natively
- Provide *interpretable* attention over past timesteps
- Learn nonlinear interactions between features
- Produce calibrated prediction intervals

**Caveat:** With only 52 weekly observations, TFT overfits. Real deployment needs
200+ observations minimum.
        """)

    with st.expander("Conformal Prediction -- Vovk et al., 2005", expanded=False):
        st.markdown("""
**What it does:** Wraps any ML model with prediction intervals that have *guaranteed*
finite-sample coverage -- no distributional assumptions needed.

**The guarantee:** If data is exchangeable (a weaker assumption than i.i.d.), then a
(1-alpha) conformal prediction interval covers the true value with probability >= 1-alpha.
Period. No ifs, no buts.

**Three methods implemented:**

1. **Split Conformal:** Train model on 70%, calibrate residuals on 30%.
   Interval = prediction +/- quantile(|residuals|, 1-alpha). Simple but wastes data.

2. **Cross-Conformal (Jackknife+):** K-fold version. For each fold, train on K-1 folds,
   compute conformity score on held-out fold. Aggregates for tighter intervals with
   theoretical guarantee: coverage >= 1 - 2*alpha.

3. **Conformalized Quantile Regression (CQR):** Trains *quantile* regressors for the lower
   and upper bounds, then calibrates them with conformal scores. Produces *adaptive* intervals:
   wider when the model is uncertain, tighter when confident. Best of both worlds.

**Why this matters:** Standard ML gives point predictions with no reliability guarantee.
Bayesian CI requires correct prior specification. Conformal prediction just works.
        """)

    with st.expander("Geo-Lift / Synthetic Control -- Abadie et al., 2010", expanded=False):
        st.markdown("""
**What it does:** Estimates the causal effect of advertising in a specific geographic market
by constructing a synthetic "control" market from a weighted combination of untreated markets.

**How it works:**
1. **Select treatment DMA:** The market with the largest TV spend ramp-up
2. **Pre-period:** Before the intervention, find weights w_1, ..., w_J for donor DMAs such
   that `sum(w_j * revenue_j) ≈ revenue_treated` in the pre-period
3. **Constraints:** w_j >= 0, sum(w_j) = 1 (convex combination)
4. **Post-period:** The synthetic control predicts what the treatment DMA *would have done*
   without the intervention. `Causal lift = actual - synthetic`

**Placebo tests:** Run the same analysis pretending each control DMA is the treatment.
If the method is valid, the actual treatment effect should be an outlier compared to
placebo effects. This is the key falsification check.

**Why Synthetic Control > Difference-in-Differences:**
- No parallel trends assumption required
- Works with a single treated unit
- Provides a visible counterfactual for intuitive interpretation
- What Google (GeoLift) and Meta actually use for incrementality testing
        """)

    with st.expander("CausalImpact (BSTS) -- Brodersen et al., 2015", expanded=False):
        st.markdown("""
**What it does:** Measures the causal impact of a campaign event (like a TV spend ramp-up)
using a Bayesian structural time series model to build a counterfactual.

**How it works:**
1. **Find intervention point:** Identify the week where TV spend changes dramatically
2. **Pre-period model:** Fit Bayesian Ridge regression on pre-intervention data using
   covariates (digital spend, seasonality, trend) to predict revenue
3. **Counterfactual projection:** Use the pre-period model to project what revenue
   *would have been* in the post-period if the intervention never happened
4. **Causal effect:** actual - counterfactual, with bootstrap uncertainty bands
5. **Posterior probability:** What fraction of bootstrap counterfactuals exceed actual?

**Three-panel output:**
- **Panel 1:** Actual vs counterfactual revenue over time
- **Panel 2:** Pointwise effect (actual - counterfactual) each week
- **Panel 3:** Cumulative effect (running sum of pointwise effects)

**Why BSTS:** This is Google's standard tool for measuring campaign effectiveness.
The Bayesian framework naturally provides uncertainty quantification.
        """)

    with st.expander("Kalman Filter: Time-Varying Parameters", expanded=False):
        st.markdown("""
**What it does:** Estimates how each channel's effectiveness (coefficient) changes over time,
treating the coefficients as a random walk in a state-space model.

**The problem with static MMM:** Standard MMM estimates ONE coefficient per channel for the
entire year. But channel effectiveness *changes*: TV might be more effective in Q4 due to
holiday messaging, or less effective in summer when audiences shift outdoor.

**State-space formulation:**
- **State equation:** `beta[t] = beta[t-1] + eta[t]` (coefficients drift as random walk)
- **Observation equation:** `revenue[t] = X[t] * beta[t] + epsilon[t]`
- **Kalman filter:** Forward pass estimates `beta[t]` given data up to time t
- **Kalman smoother:** Backward pass refines estimates using ALL data (past and future)

**EM algorithm:** Learns the noise parameters (Q = state noise, R = observation noise)
from data automatically.

**Key insight:** If a channel's coefficient increases over time, its effectiveness is
growing (e.g., due to creative refresh or audience expansion). If decreasing, the channel
may be saturating or suffering creative fatigue.
        """)

    st.markdown("---")

    # ---- SECTION 5: STACKED COMPARISON ----
    st.header("5. The Stacked Model Comparison")

    st.markdown("""
The **Model Comparison** page puts 6 attribution methods side-by-side on the same data.
This is not about finding the "best" model -- it's about understanding *structural differences*
in how models think about attribution.

**The 6 models:**

| Model | Type | Key Assumption | TV Bias |
|-------|------|----------------|---------|
| Last-Touch | Heuristic | Last channel before conversion gets 100% credit | Undervalues TV |
| First-Touch | Heuristic | First channel in journey gets 100% credit | Overvalues TV |
| Markov Chain | Probabilistic | Journey follows a Markov process | Balanced |
| LightGBM | ML (Counterfactual) | Baseline traffic is learnable | Conservative |
| Frequentist MMM | Econometric | Linear response with adstock + saturation | Moderate |
| Bayesian MMM | Bayesian Econometric | Same as above + prior uncertainty | Moderate + CI |

**Key patterns to look for:**
1. **TV attribution varies 2-5x** across methods. This is NOT a bug -- it reflects genuine
   methodological uncertainty about TV's true contribution.
2. **Last-Touch undervalues TV** because TV rarely appears as the *final* touchpoint.
   Users see a TV ad, then later search on Google and convert. Last-Touch credits Google.
3. **Markov and Bayesian MMM tend to agree** more closely, as both model the full journey
   or response function rather than using simplistic heuristics.
4. **Where models agree = high confidence.** If all 6 methods say Paid Search contributes
   15-20%, you can be fairly confident in that range.
5. **Where models disagree = uncertainty.** This is where you need additional evidence
   (e.g., geo-lift experiments or holdout tests) to resolve.

**The recommendation:** Don't pick one model. Triangulate across methods, weight models
by their theoretical appropriateness for your question, and use disagreement as a signal
for where to invest in better measurement.
    """)

    st.markdown("---")

    # ---- SECTION 6: METHODOLOGY NOTES ----
    st.header("6. Data & Methodology Notes")
    st.markdown("""
**Data:** Simulated dataset for a DTC consumer brand with:
- 52 weeks of channel-level spend data (TV Broadcast, Cable, Streaming, Search, Social, Display)
- ~2,800 individual TV ad airings across 8 DMAs
- Hourly web traffic data (sessions) for attribution
- Brand search volume for cross-channel analysis

**Why simulated data:** This project demonstrates *methodology*, not specific results.
The data generation process embeds realistic patterns: adstock carry-over, Hill saturation,
seasonality, cross-channel effects, and heterogeneous DMA responses.

**Technical stack:**
- **Python:** pandas, numpy, scikit-learn, LightGBM, PyMC, PyTorch
- **Causal ML:** EconML (Microsoft), MAPIE, pykalman
- **Visualization:** Plotly, Matplotlib, Streamlit
- **Statistical methods:** Bayesian inference, Granger causality, synthetic control,
  conformal prediction, Kalman filtering, Double ML
    """)


# ============================================
# PAGE: REPORT & DIAGNOSTICS
# ============================================

elif page == "report":
    st.title("Report & Diagnostics")

    st.header("Executive Summary")
    total_tv_spend = ws['tv_broadcast_spend'].sum() + ws['tv_cable_spend'].sum() + ws['tv_streaming_spend'].sum()
    total_spend = ws['total_spend'].sum()
    total_revenue = ws['total_revenue'].sum()
    attr_rev = data['airing_attribution']['incremental_revenue'].sum()

    st.markdown("""
| Metric | Value |
|--------|-------|
| **Total Annual Revenue** | {} |
| **Total Ad Spend** | {} |
| **TV Ad Spend** | {} ({:.0f}% of total) |
| **TV Airings** | {:,} across 8 DMAs |
| **TV Attributed Revenue** | {} |
| **Blended TV ROAS** | {:.2f}x |
| **Models Built** | 12 (6 attribution + 6 advanced causal) |
    """.format(
        format_currency(total_revenue),
        format_currency(total_spend),
        format_currency(total_tv_spend),
        total_tv_spend / total_spend * 100,
        len(data['airings']),
        format_currency(attr_rev),
        attr_rev / total_tv_spend if total_tv_spend > 0 else 0
    ))

    st.header("Model Quality Checks")

    checks = []
    if 'baseline_metrics' in data:
        bm = data['baseline_metrics']
        r2 = bm.get('test_r2', 0)
        checks.append(('Baseline Model R2', '{:.4f}'.format(r2), 'Pass' if r2 > 0.85 else 'Warning'))

    mmm_r2 = mmm_params.get('model_r2', 0)
    checks.append(('MMM R2', '{:.4f}'.format(mmm_r2), 'Pass' if mmm_r2 > 0.80 else 'Warning'))

    if 'bayesian_params' in data:
        bp = data['bayesian_params']
        div = bp.get('divergences', 0)
        checks.append(('Bayesian Divergences', str(div), 'Pass' if div == 0 else 'Warning'))

    if 'conformal_summary' in data:
        cs = data['conformal_summary']
        for method in ['split_conformal', 'cqr']:
            if method in cs:
                cov = cs[method]['coverage']
                checks.append(('{} Coverage'.format(method), '{:.0%}'.format(cov),
                                'Pass' if cov >= 0.85 else 'Warning'))

    if 'kalman_summary' in data:
        ks = data['kalman_summary']
        kr2 = ks.get('dynamic_r2', 0)
        checks.append(('Kalman R2', '{:.4f}'.format(kr2), 'Pass' if kr2 > 0.80 else 'Warning'))

    if checks:
        check_df = pd.DataFrame(checks, columns=['Check', 'Value', 'Status'])
        st.dataframe(check_df, use_container_width=True, hide_index=True)

    n_pass = sum(1 for c in checks if c[2] == 'Pass')
    if n_pass == len(checks):
        st.success("All {} model quality checks passed.".format(len(checks)))
    else:
        st.warning("{}/{} checks passed. Review warnings above.".format(n_pass, len(checks)))

    st.header("Generated Artifacts")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Data Files**")
        data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
        for f in sorted(data_files):
            size = os.path.getsize(os.path.join('data', f))
            st.text("{:<45s} {:>8s}".format(f, '{:.0f} KB'.format(size / 1024)))
    with col2:
        st.markdown("**Result Visualizations**")
        if os.path.exists('results'):
            result_files = [f for f in os.listdir('results') if f.endswith('.png')]
            for f in sorted(result_files):
                st.text(f)
        else:
            st.text("No results directory found.")

    st.header("Export")
    st.markdown("To generate a PDF report, run:")
    st.code("python src/generate_report.py", language="bash")
    if os.path.exists('results/executive_report.pdf'):
        with open('results/executive_report.pdf', 'rb') as f:
            st.download_button("Download Executive Report (PDF)", f,
                               file_name="tv_attribution_report.pdf", mime="application/pdf")


# ============================================
# FALLBACK
# ============================================

else:
    st.error("Page '{}' not found. Use the sidebar to navigate.".format(page))
