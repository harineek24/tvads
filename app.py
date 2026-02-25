"""
TV Ad Attribution & Media Mix Model Dashboard
=============================================
Interactive Streamlit dashboard for analyzing TV advertising performance,
channel attribution, budget optimization, and scenario planning.

6 Tabs:
1. Attribution — Per-airing lift analysis with baseline vs actual
2. Budget Optimizer — Optimal spend allocation (hero feature)
3. Scenario Planner — What-if analysis with interactive sliders
4. Channel Deep Dive — Adstock, saturation, and recommendations
5. Client Report — Auto-generated executive summary + PDF export
6. Model Diagnostics — Fit quality, residuals, limitations
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Attribution",
    "💰 Budget Optimizer",
    "🔮 Scenario Planner",
    "🔬 Channel Deep Dive",
    "📄 Client Report",
    "🔧 Model Diagnostics",
])


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
# TAB 5: CLIENT REPORT
# ============================================

with tab5:
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

with tab6:
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
