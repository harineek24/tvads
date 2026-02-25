"""
TV Ad Attribution & Advanced Causal ML Dashboard
=================================================
7 cutting-edge models for measuring TV advertising effectiveness.
Sidebar navigation with grouped sections.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="TV Ad Analytics | 7 Advanced Models",
    page_icon="\U0001f4fa",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# DATA LOADING
# ============================================

@st.cache_data
def load_data():
    data = {}
    data["weekly_spend"] = pd.read_csv("data/weekly_spend.csv")
    data["airings"] = pd.read_csv("data/ad_airings.csv")

    optional_csv = {
        "bayesian_contributions": "data/bayesian_channel_contributions.csv",
        "bayesian_roas": "data/bayesian_roas_posteriors.csv",
        "bayesian_dag": "data/bayesian_causal_dag.csv",
        "bayesian_calibration": "data/bayesian_calibration.csv",
        "shapley_values": "data/shapley_values.csv",
        "shapley_interactions": "data/shapley_interactions.csv",
        "shapley_asymmetric": "data/shapley_asymmetric.csv",
        "markov_attribution": "data/markov_attribution.csv",
        "markov_removal": "data/markov_removal_effects.csv",
        "deep_causal_dag": "data/deep_causal_dag.csv",
        "deep_causal_adstock": "data/deep_causal_adstock.csv",
        "deep_causal_contributions": "data/deep_causal_contributions.csv",
        "dml_effects": "data/dml_causal_effects.csv",
        "dml_vs_ols": "data/dml_vs_ols.csv",
        "dml_het": "data/dml_heterogeneous_effects.csv",
        "tft_importance": "data/tft_feature_importance.csv",
        "tft_predictions": "data/tft_predictions.csv",
        "conformal_intervals": "data/conformal_intervals.csv",
        "conformal_coverage": "data/conformal_coverage.csv",
        "geo_lift": "data/geo_lift_results.csv",
        "sc_weights": "data/synthetic_control_weights.csv",
        "model_comparison": "data/model_comparison.csv",
    }
    for key, path in optional_csv.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path)

    optional_json = {
        "bayesian_params": "models/bayesian_mmm_params.json",
        "shapley_summary": "data/shapley_summary.json",
        "deep_causal_summary": "data/deep_causal_summary.json",
        "tft_metrics": "models/tft_metrics.json",
        "conformal_summary": "data/conformal_summary.json",
        "geo_lift_summary": "data/geo_lift_summary.json",
    }
    for key, path in optional_json.items():
        if os.path.exists(path):
            with open(path, "r") as f:
                data[key] = json.load(f)
    return data


data = load_data()
ws = data["weekly_spend"]

# ============================================
# CONSTANTS
# ============================================

CHANNEL_COLORS = {
    "TV Broadcast": "#2c3e50", "TV Cable": "#34495e", "TV Streaming": "#7f8c8d",
    "Paid Search": "#2980b9", "Social": "#8e44ad", "Display": "#e67e22",
}


def format_currency(val, decimals=1):
    if abs(val) >= 1e6:
        return "${:.{}f}M".format(val / 1e6, decimals)
    elif abs(val) >= 1e3:
        return "${:.{}f}K".format(val / 1e3, decimals)
    return "${:.0f}".format(val)


# ============================================
# SIDEBAR NAVIGATION
# ============================================

st.sidebar.title("\U0001f4fa TV Ad Analytics")
st.sidebar.markdown("---")

PAGES = {
    "Overview": "overview",
    "Bayesian MMM": "bayesian",
    "Markov Chains": "markov",
    "Shapley Attribution": "shapley",
    "DeepCausalMMM": "deep_causal",
    "Double ML (DML)": "dml",
    "Transformer (TFT)": "tft",
    "Conformal Prediction": "conformal",
    "Geo-Lift": "geo_lift",
    "Model Comparison": "comparison",
    "How It Works": "how_it_works",
}

SECTIONS = {
    "Core Models": ["Overview", "Bayesian MMM", "Markov Chains", "Shapley Attribution", "DeepCausalMMM"],
    "Causal ML": ["Double ML (DML)", "Transformer (TFT)", "Conformal Prediction", "Geo-Lift"],
    "Synthesis": ["Model Comparison", "How It Works"],
}

for section, pages in SECTIONS.items():
    st.sidebar.markdown("**{}**".format(section))
    for page in pages:
        if st.sidebar.button(page, key="nav_" + PAGES[page], use_container_width=True):
            st.session_state["page"] = PAGES[page]
    st.sidebar.markdown("")

if "page" not in st.session_state:
    st.session_state["page"] = "overview"

page = st.session_state["page"]

st.sidebar.markdown("---")
st.sidebar.caption("DTC Brand | $15M Annual | 52 Weeks | 8 DMAs | 8 Models")


# ============================================
# PAGE: OVERVIEW
# ============================================

if page == "overview":
    st.title("TV Ad Attribution & Advanced Causal ML")
    st.markdown(
        "A marketing analytics platform with **8 statistical and ML models** "
        "measuring TV advertising effectiveness -- from intuitive journey models "
        "to cutting-edge causal ML and deep learning."
    )

    total_tv = ws['tv_broadcast_spend'].sum() + ws['tv_cable_spend'].sum() + ws['tv_streaming_spend'].sum()
    total_spend = ws['total_spend'].sum()
    total_rev = ws['total_revenue'].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", format_currency(total_rev))
    c2.metric("Total Ad Spend", format_currency(total_spend))
    c3.metric("TV Share of Spend", "{:.0f}%".format(total_tv / total_spend * 100))
    c4.metric("Models Built", "8")

    st.markdown("---")

    st.subheader("The 8 Models")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Core Models**
- **Bayesian MMM** -- How much revenue does each channel drive? (with uncertainty)
- **Markov Chains** -- How do customers move through the ad journey?
- **Shapley Attribution** -- What's the fairest way to split credit?
- **DeepCausalMMM** -- Can a neural network learn advertising effects automatically?

**Causal ML**
- **Double ML** -- What's the TRUE causal effect, removing confounding bias?
        """)
    with col2:
        st.markdown("""
**Causal ML (continued)**
- **Transformer (TFT)** -- Can deep learning forecast revenue from ad spend?
- **Conformal Prediction** -- How confident should we be in our predictions?
- **Geo-Lift** -- If we turned off TV ads in one city, what would happen?

**Synthesis**
- **Model Comparison** -- All 8 methods side-by-side on the same data
        """)

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        if "bayesian_contributions" in data:
            bc = data["bayesian_contributions"]
            fig = px.pie(bc, values="median_contribution", names="channel",
                         color="channel", color_discrete_map=CHANNEL_COLORS,
                         title="Revenue Attribution (Bayesian MMM)")
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ws["week"], y=ws["total_revenue"] / 1e6,
                                 name="Revenue", line=dict(color="#27ae60", width=2)))
        fig.add_trace(go.Scatter(x=ws["week"], y=ws["total_spend"] / 1e6,
                                 name="Total Spend", line=dict(color="#e74c3c", width=2, dash="dash")))
        fig.update_layout(title="Weekly Revenue vs Spend", height=350,
                          xaxis_title="Week", yaxis_title="$M", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)


# ============================================
# PAGE: BAYESIAN MMM
# ============================================

elif page == "bayesian":
    st.title("Bayesian MMM with Causal DAG & Experiment Calibration")
    st.markdown(
        "Full Bayesian estimation via PyMC with MCMC sampling. Upgraded with "
        "**causal DAG identification** (TV \u2192 Search mediation) and "
        "**experiment calibration** (lift test anchoring)."
    )

    tab_core, tab_dag, tab_cal = st.tabs([
        "Posterior ROAS", "Causal DAG", "Experiment Calibration"
    ])

    with tab_core:
        if "bayesian_contributions" in data:
            bc = data["bayesian_contributions"]

            st.subheader("Posterior ROAS by Channel (90% Credible Intervals)")
            cols = st.columns(len(bc))
            for i, (_, row) in enumerate(bc.iterrows()):
                with cols[i]:
                    st.metric(row["channel"], "{:.2f}x".format(row["median_roas"]),
                              delta="[{:.2f}, {:.2f}]".format(row["roas_ci_5"], row["roas_ci_95"]),
                              delta_color="off")

            bc_sorted = bc.sort_values("median_roas", ascending=True)
            fig = go.Figure()
            for _, row in bc_sorted.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row["roas_ci_5"], row["median_roas"], row["roas_ci_95"]],
                    y=[row["channel"]] * 3, mode="markers+lines",
                    marker=dict(size=[8, 14, 8], color=["gray", "#8e44ad", "gray"]),
                    line=dict(color="#8e44ad", width=3), name=row["channel"], showlegend=False))
            fig.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="Break-even")
            fig.update_layout(title="Posterior ROAS with 90% Credible Intervals",
                              xaxis_title="ROAS", height=350)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists("results/bayesian_posterior_roas.png"):
                    st.image("results/bayesian_posterior_roas.png", caption="ROAS Posterior Distributions")
            with col2:
                if os.path.exists("results/bayesian_posterior_predictive.png"):
                    st.image("results/bayesian_posterior_predictive.png", caption="Posterior Predictive Check")

            if os.path.exists("results/bayesian_forest_plot.png"):
                st.image("results/bayesian_forest_plot.png", caption="Channel Contributions Forest Plot")

            if "bayesian_params" in data:
                bp = data["bayesian_params"]
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Chains", bp.get("n_chains", 2))
                d2.metric("Draws / Chain", bp.get("n_draws", 2000))
                d3.metric("Tune Steps", bp.get("n_tune", 1500))
                d4.metric("Divergences", bp.get("divergences", "N/A"))
        else:
            st.info("Run `python src/bayesian_mmm.py` to generate results.")

    with tab_dag:
        st.subheader("Causal DAG: TV \u2192 Search Mediation")
        st.markdown(
            "When TV causally drives brand search, a naive model over-credits search. "
            "The causal DAG explicitly models this pathway (PyMC Labs, 2024)."
        )

        if os.path.exists("results/bayesian_causal_dag.png"):
            st.image("results/bayesian_causal_dag.png", caption="Learned Causal Structure")

        if "bayesian_dag" in data:
            dag_df = data["bayesian_dag"]
            st.dataframe(dag_df, use_container_width=True, hide_index=True)

        if "bayesian_params" in data:
            bp = data["bayesian_params"]
            dag_info = bp.get("causal_dag", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Mediation %",
                       "{:.0f}%".format(dag_info.get("mediation_pct", 0)),
                       delta="of TV effect flows through search")
            c2.metric("TV\u2192Search Synergy",
                       "{:.4f}".format(dag_info.get("tv_search_synergy_median", 0)))
            c3.metric("Significant Edges",
                       dag_info.get("n_significant_edges", 0))

    with tab_cal:
        st.subheader("Experiment Calibration")
        st.markdown(
            "MMM estimates are calibrated against simulated geo-holdout lift tests "
            "(Zhang et al., 2024). In practice, use real incrementality experiments."
        )

        if os.path.exists("results/bayesian_calibration.png"):
            st.image("results/bayesian_calibration.png",
                     caption="MMM vs Experiment ROAS + Calibration Factors")

        if "bayesian_calibration" in data:
            cal_df = data["bayesian_calibration"]
            st.dataframe(cal_df.round(3), use_container_width=True, hide_index=True)

            if "bayesian_params" in data:
                bp = data["bayesian_params"]
                cal_info = bp.get("calibration", {})
                st.metric("Channels Within CI",
                          "{}/{}".format(cal_info.get("n_within_ci", 0),
                                         cal_info.get("n_channels", 0)))


# ============================================
# PAGE: MARKOV CHAINS
# ============================================

elif page == "markov":
    st.title("Markov Chain Attribution")
    st.markdown(
        "Models the customer journey as a **step-by-step path** through marketing touchpoints. "
        "Instead of giving all credit to the last ad someone saw, Markov chains ask: "
        '"if we removed this channel entirely, how many fewer conversions would we get?"'
    )

    if "markov_attribution" in data:
        markov_df = data["markov_attribution"]

        st.subheader("Attribution: Markov vs Last-Touch vs First-Touch")
        fig = go.Figure()
        for method, col_name, color in [("Markov Chain", "markov_attribution", "#2c3e50"),
                                         ("Last-Touch", "last_touch_attribution", "#e74c3c"),
                                         ("First-Touch", "first_touch_attribution", "#e67e22")]:
            fig.add_trace(go.Bar(x=markov_df["channel"], y=markov_df[col_name] * 100,
                                 name=method, marker_color=color, opacity=0.85))
        fig.update_layout(barmode="group",
                          title="Who Gets Credit? Three Different Answers",
                          yaxis_title="Attribution Share (%)", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Why does this matter?**
        - **Last-Touch** says: "Google Search gets all the credit" (because that's the last click)
        - **Markov Chain** says: "Wait -- the customer only searched because they saw a TV ad first"
        - TV typically gets **2-3x more credit** under Markov than Last-Touch
        """)

        if "markov_removal" in data:
            st.subheader("Removal Effect: What Happens If We Remove Each Channel?")
            removal_df = data["markov_removal"]
            fig = px.bar(removal_df.sort_values("removal_effect", ascending=True),
                         x="removal_effect", y="channel", orientation="h",
                         color="removal_effect", color_continuous_scale="Reds",
                         title="Conversion Drop When Channel Is Removed")
            fig.update_layout(height=350, xaxis_title="Removal Effect", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("results/markov_transition_heatmap.png"):
                st.image("results/markov_transition_heatmap.png",
                         caption="Transition Matrix: Probability of Moving Between Channels")
        with col2:
            if os.path.exists("results/markov_vs_lasttouch.png"):
                st.image("results/markov_vs_lasttouch.png",
                         caption="Markov vs Last-Touch: The Difference")

        if os.path.exists("results/markov_removal_effects.png"):
            st.image("results/markov_removal_effects.png",
                     caption="Channel Removal Effects")
    else:
        st.info("Run `python src/markov_attribution.py` to generate results.")


# ============================================
# PAGE: SHAPLEY ATTRIBUTION
# ============================================

elif page == "shapley":
    st.title("Shapley Value Attribution")
    st.markdown(
        "**Game-theoretic fair attribution** using exact Shapley values from cooperative game theory. "
        "The only method satisfying all 4 fairness axioms: efficiency, symmetry, linearity, null player."
    )

    tab_exact, tab_inter, tab_asym = st.tabs([
        "Exact Shapley Values", "Interaction Index", "Asymmetric (Causal Order)"
    ])

    with tab_exact:
        if "shapley_values" in data:
            sv = data["shapley_values"]
            ch_col = "channel_name" if "channel_name" in sv.columns else "channel"

            st.subheader("Channel Attribution (64 Coalitions Evaluated)")
            if "shapley_pct" in sv.columns:
                cols = st.columns(len(sv))
                for i, (_, row) in enumerate(sv.iterrows()):
                    with cols[i]:
                        st.metric(row[ch_col], "{:.1f}%".format(row["shapley_pct"]))

                fig = px.bar(sv.sort_values("shapley_pct", ascending=True),
                             x="shapley_pct", y=ch_col, orientation="h",
                             color="channel", color_discrete_map=CHANNEL_COLORS,
                             title="Exact Shapley Attribution (%)")
                fig.update_layout(height=350, showlegend=False, xaxis_title="Attribution Share (%)")
                st.plotly_chart(fig, use_container_width=True)

            if os.path.exists("results/shapley_attribution.png"):
                st.image("results/shapley_attribution.png",
                         caption="Shapley vs Asymmetric vs Proportional Attribution")

            if "shapley_summary" in data:
                ss = data["shapley_summary"]
                st.subheader("Key Findings")
                findings = ss.get("key_findings", [])
                for f in findings:
                    st.markdown("- {}".format(f))
        else:
            st.info("Run `python src/shapley_attribution.py` to generate results.")

    with tab_inter:
        st.subheader("Shapley-Owen Interaction Index")
        st.markdown(
            "Measures pairwise synergies. Positive = super-additive (channels amplify each other). "
            "Tests the TV \u2192 Search halo hypothesis."
        )

        if "shapley_interactions" in data:
            inter_df = data["shapley_interactions"]
            if os.path.exists("results/shapley_interactions.png"):
                st.image("results/shapley_interactions.png",
                         caption="Pairwise Interaction Heatmap (red=synergy, blue=redundancy)")
            st.dataframe(inter_df.round(4), use_container_width=True, hide_index=True)
        else:
            st.info("Run Shapley analysis to generate interaction indices.")

    with tab_asym:
        st.subheader("Asymmetric Shapley (Causal Ordering)")
        st.markdown(
            "Standard Shapley treats all orderings equally. Asymmetric Shapley respects "
            "the causal ordering: **TV \u2192 Social \u2192 Paid Search \u2192 Display**."
        )

        if "shapley_asymmetric" in data:
            asym_df = data["shapley_asymmetric"]
            st.dataframe(asym_df.round(3), use_container_width=True, hide_index=True)

        if os.path.exists("results/shapley_marginal.png"):
            st.image("results/shapley_marginal.png",
                     caption="Marginal Contribution Curves Across Budget Levels")


# ============================================
# PAGE: DEEP CAUSAL MMM
# ============================================

elif page == "deep_causal":
    st.title("DeepCausalMMM: Neural Adstock + DAG Structure Learning")
    st.markdown(
        "**The cutting edge** (2024/2025): Instead of manually specifying adstock decay rates "
        "and saturation curves, a **GRU neural network learns them from data**. Combined with "
        "**NOTEARS DAG structure learning** (Zheng et al., 2018) to automatically discover "
        "causal relationships between marketing channels."
    )

    if "deep_causal_summary" in data:
        ds = data["deep_causal_summary"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train R\u00b2", "{:.4f}".format(ds.get("train_r2", 0)))
        c2.metric("Test R\u00b2", "{:.4f}".format(ds.get("test_r2", 0)))
        c3.metric("DAG Penalty", "{:.6f}".format(ds.get("dag_penalty", 0)))
        c4.metric("Causal Edges Discovered", ds.get("n_causal_edges", 0))

        tab_dag, tab_adstock, tab_contrib = st.tabs([
            "Learned Causal DAG", "Neural Adstock", "Channel Contributions"
        ])

        with tab_dag:
            st.subheader("Automatically Discovered Causal Structure")
            st.markdown(
                "The NOTEARS algorithm learns which channels causally influence which, "
                "subject to an acyclicity constraint (trace(e^{W*W}) = d)."
            )

            if os.path.exists("results/deep_causal_dag.png"):
                st.image("results/deep_causal_dag.png",
                         caption="Learned DAG: Edge Thickness = Causal Strength")

            if "deep_causal_dag" in data:
                dag_df = data["deep_causal_dag"]
                st.subheader("Adjacency Matrix (W)")
                st.dataframe(dag_df.round(4), use_container_width=True, hide_index=True)

            if "discovered_edges" in ds:
                st.subheader("Discovered Causal Edges")
                for edge in ds["discovered_edges"]:
                    st.markdown("- **{}** (strength: {:.3f})".format(
                        edge.get("edge", ""), edge.get("weight", 0)))

        with tab_adstock:
            st.subheader("GRU-Learned vs Geometric Adstock")
            st.markdown(
                "Instead of assuming geometric decay, the GRU cell learns each channel's "
                "temporal carry-over pattern directly from data."
            )

            if os.path.exists("results/deep_causal_adstock.png"):
                st.image("results/deep_causal_adstock.png",
                         caption="Neural vs Traditional Adstock Curves")

            if "deep_causal_adstock" in data:
                adstock_df = data["deep_causal_adstock"]
                st.dataframe(adstock_df.round(4), use_container_width=True, hide_index=True)

        with tab_contrib:
            st.subheader("Channel ROAS & Contributions")

            if os.path.exists("results/deep_causal_contributions.png"):
                st.image("results/deep_causal_contributions.png",
                         caption="DeepCausalMMM Channel Contributions")

            if "deep_causal_contributions" in data:
                contrib_df = data["deep_causal_contributions"]
                st.dataframe(contrib_df.round(3), use_container_width=True, hide_index=True)
    else:
        st.info("Run `python src/deep_causal_mmm.py` to generate results.")


# ============================================
# PAGE: DOUBLE ML
# ============================================

elif page == "dml":
    st.title("Double Machine Learning (DML)")
    st.markdown(
        "**Debiased causal inference** (Chernozhukov et al., 2018). Uses LightGBM for "
        "nuisance parameters but preserves valid statistical inference on the treatment "
        "effect. The gold standard for observational causal estimation in marketing."
    )

    if "dml_effects" in data:
        dml_df = data["dml_effects"]

        st.subheader("Causal Average Treatment Effects (ATE)")
        cols = st.columns(3)
        for i, (_, row) in enumerate(dml_df.iterrows()):
            with cols[i % 3]:
                sig = " ***" if row["p_value"] < 0.01 else (" **" if row["p_value"] < 0.05 else "")
                st.metric(row["channel"], "${:.3f}/$ spent".format(row["ate"]),
                          delta="p={:.4f}{}".format(row["p_value"], sig), delta_color="off")

        if "dml_vs_ols" in data:
            st.subheader("DML vs Naive OLS: Exposing Confounding Bias")
            ols_df = data["dml_vs_ols"]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ols_df["channel"], y=ols_df["ols_coefficient"],
                                 name="Naive OLS (biased)", marker_color="#e74c3c", opacity=0.7))
            fig.add_trace(go.Bar(x=ols_df["channel"], y=ols_df["dml_causal_effect"],
                                 name="DML (debiased)", marker_color="#2980b9", opacity=0.7))
            fig.update_layout(barmode="group", height=400,
                              title="How Much Bias Does Confounding Introduce?",
                              yaxis_title="Effect ($/$ spend)")
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("results/dml_causal_effects.png"):
                st.image("results/dml_causal_effects.png", caption="ATEs with 95% CIs")
        with col2:
            if os.path.exists("results/dml_cate_heterogeneity.png"):
                st.image("results/dml_cate_heterogeneity.png", caption="CausalForest: Heterogeneous Effects")
    else:
        st.info("Run `python src/double_ml.py` to generate results.")


# ============================================
# PAGE: TFT
# ============================================

elif page == "tft":
    st.title("Temporal Fusion Transformer (TFT)")
    st.markdown(
        "**State-of-the-art deep learning** for time series (Google Research, 2021). "
        "Variable Selection Network, LSTM encoder, and Multi-Head Attention provide "
        "interpretable insights into which features and past timesteps drive predictions."
    )

    if "tft_metrics" in data:
        m = data["tft_metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train R\u00b2", "{:.3f}".format(m.get("train_r2", 0)))
        c2.metric("Test R\u00b2", "{:.3f}".format(m.get("test_r2", 0)))
        c3.metric("Test MAE", "${:,.0f}".format(m.get("test_mae", 0)))
        c4.metric("PI Coverage", "{:.0%}".format(m.get("coverage", 0)))

        if "tft_importance" in data:
            vi = data["tft_importance"]
            fig = px.bar(vi.sort_values("importance", ascending=True),
                         x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale="Viridis",
                         title="TFT Variable Selection: Learned Feature Importance")
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("results/tft_forecast.png"):
                st.image("results/tft_forecast.png", caption="Revenue Forecast + Prediction Intervals")
        with col2:
            if os.path.exists("results/tft_attention_heatmap.png"):
                st.image("results/tft_attention_heatmap.png", caption="Attention Weights & Training")

        st.caption("Note: With 52 weekly observations, TFT overfits. "
                   "Real deployment needs 200+ observations minimum.")
    else:
        st.info("Run `python src/temporal_fusion.py` to generate results.")


# ============================================
# PAGE: CONFORMAL PREDICTION
# ============================================

elif page == "conformal":
    st.title("Conformal Prediction")
    st.markdown(
        "**Distribution-free uncertainty** with guaranteed finite-sample coverage. "
        "No distributional assumptions -- only exchangeability. "
        "Based on Vovk et al. (2005), Angelopoulos & Bates (2021)."
    )

    if "conformal_summary" in data:
        cs = data["conformal_summary"]
        methods = [("split_conformal", "Split Conformal"),
                   ("jackknife_plus", "Jackknife+"),
                   ("cqr", "CQR (Adaptive)")]
        mc = st.columns(3)
        for i, (key, name) in enumerate(methods):
            if key in cs:
                with mc[i]:
                    st.metric("{} Coverage".format(name),
                              "{:.0%}".format(cs[key]["coverage"]),
                              delta="Width: ${:,.0f}".format(cs[key]["avg_width"]))

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("results/conformal_intervals.png"):
                st.image("results/conformal_intervals.png", caption="Three Methods Compared")
        with col2:
            if os.path.exists("results/conformal_coverage.png"):
                st.image("results/conformal_coverage.png", caption="Coverage Calibration")

        if os.path.exists("results/conformal_width_analysis.png"):
            st.image("results/conformal_width_analysis.png", caption="CQR: Adaptive Intervals")
    else:
        st.info("Run `python src/conformal_prediction.py` to generate results.")


# ============================================
# PAGE: GEO-LIFT
# ============================================

elif page == "geo_lift":
    st.title("Geo-Lift / Synthetic Control")
    st.markdown(
        "**Causal incrementality testing** (Abadie et al., 2010). The 'ground truth' layer. "
        "Constructs a synthetic control DMA from weighted donors to estimate true causal lift. "
        "This is what Google GeoLift and Meta actually use."
    )

    if "geo_lift_summary" in data:
        gs = data["geo_lift_summary"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Treatment DMA", gs.get("treatment_dma", "N/A"))
        c2.metric("Causal Lift", "{:+.1f}%".format(gs.get("avg_lift_pct", 0)))
        c3.metric("Pre-Period R\u00b2", "{:.3f}".format(gs.get("pre_r2", 0)))

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("results/geo_lift_main.png"):
                st.image("results/geo_lift_main.png", caption="Actual vs Synthetic Counterfactual")
        with col2:
            if os.path.exists("results/geo_lift_weights.png"):
                st.image("results/geo_lift_weights.png", caption="Donor DMA Weights")

        if os.path.exists("results/geo_lift_placebo.png"):
            st.subheader("Placebo Tests: Falsification Check")
            st.markdown("Treatment DMA should stand out vs placebos.")
            st.image("results/geo_lift_placebo.png", caption="Placebo Tests Across All DMAs")
    else:
        st.info("Run `python src/geo_lift.py` to generate results.")


# ============================================
# PAGE: MODEL COMPARISON
# ============================================

elif page == "comparison":
    st.title("8-Model Comparison")
    st.markdown(
        "All 8 methods applied to the same data. Where models **agree** = higher "
        "confidence. Where they **disagree** = genuine methodological uncertainty."
    )

    if "model_comparison" in data:
        comp_df = data["model_comparison"]
        channels = [c for c in comp_df.columns if c != "model"]
        active_channels = [c for c in channels if comp_df[c].sum() > 0.01]

        st.subheader("Attribution Shares by Model")
        fig = go.Figure()
        for ch in active_channels:
            color = CHANNEL_COLORS.get(ch, "#bdc3c7")
            fig.add_trace(go.Bar(
                y=comp_df["model"], x=comp_df[ch] * 100, name=ch, orientation="h",
                marker_color=color,
                text=comp_df[ch].apply(lambda v: "{:.0%}".format(v) if v > 0.05 else ""),
                textposition="inside", textfont_color="white"))
        fig.update_layout(barmode="stack",
                          title="How Each Method Attributes Revenue Across Channels",
                          xaxis_title="Attribution Share (%)", height=500,
                          legend=dict(orientation="h", yanchor="bottom", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Channel-Level Comparison")
        model_colors = {
            "Bayesian MMM": "#8e44ad", "Markov Chain": "#2c3e50",
            "Shapley Values": "#1abc9c", "DeepCausalMMM": "#e74c3c",
            "Double ML": "#2980b9", "TFT Attention": "#27ae60",
            "Geo-Lift": "#e67e22", "Conformal": "#34495e",
        }
        fig = go.Figure()
        for _, row in comp_df.iterrows():
            mn = row["model"]
            fig.add_trace(go.Bar(
                x=active_channels,
                y=[row[ch] * 100 for ch in active_channels],
                name=mn, marker_color=model_colors.get(mn, "gray"), opacity=0.85))
        fig.update_layout(barmode="group",
                          title="Each Channel Through Different Methodological Lenses",
                          yaxis_title="Attribution Share (%)", height=400)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("results/model_comparison_radar.png"):
                st.image("results/model_comparison_radar.png", caption="Attribution Radar")
        with col2:
            if os.path.exists("results/model_comparison_insights.png"):
                st.image("results/model_comparison_insights.png", caption="Key Insights")

        st.subheader("Full Comparison Matrix")
        display_df = comp_df.copy()
        for ch in active_channels:
            display_df[ch] = display_df[ch].apply(lambda x: "{:.1%}".format(x))
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.subheader("Key Takeaways")
        st.markdown("""
1. **Causal models give TV more credit** -- DML and DeepCausalMMM properly debias confounding
2. **Shapley values are the fairest** -- 4 game-theoretic axioms guarantee fair allocation
3. **Bayesian MMM quantifies uncertainty** -- credible intervals show where we're confident
4. **Geo-Lift is the ground truth** -- incrementality experiments measure true causal lift
5. **Triangulate across methods** -- no single model is correct; use disagreement as a signal
        """)
    else:
        st.info("Run `python src/model_comparison.py` to generate comparison results.")


# ============================================
# PAGE: HOW IT WORKS
# ============================================

elif page == "how_it_works":
    st.title("How It Works")
    st.markdown(
        "Every model explained in **plain English first**, then the technical details. "
        "If you're new to ML, start with the analogies -- they're designed to build intuition."
    )

    st.markdown("---")

    # ---- MODEL 1 ----
    st.header("1. Bayesian MMM")
    st.markdown("*Question it answers: How much revenue does each ad channel drive?*")

    with st.expander("Plain English", expanded=True):
        st.markdown("""
Imagine you run a lemonade stand and you advertise in three places: a sign on the
street, flyers at school, and a radio ad. At the end of the month, you made $1,000.
**How much of that $1,000 came from each type of advertising?**

That's what Media Mix Modeling does. It looks at how much you spent on each channel
each week, and how much revenue came in, and figures out the relationship.

The **"Bayesian"** part means: instead of giving you ONE answer ("TV drives $300"),
it gives you a RANGE: "TV drives somewhere between $200 and $400, most likely around $300."
That range is called a **credible interval** and it honestly tells you how uncertain
the estimate is.

The **"Causal DAG"** upgrade recognizes that TV ads make people Google your brand.
If you don't account for this, you'll accidentally give Google credit for sales that
TV actually started.
        """)

    with st.expander("The Technical Bit"):
        st.markdown("""
- **Adstock:** TV ad effect doesn't disappear instantly. `effect[t] = spend[t] + decay * effect[t-1]`
- **Saturation:** Doubling your budget doesn't double results (diminishing returns). Modeled with Hill function.
- **MCMC:** Markov Chain Monte Carlo sampling (PyMC/NUTS) draws thousands of plausible parameter values from the posterior distribution
- **Causal DAG:** Explicitly models TV -> Brand Search mediation pathway to avoid over-crediting search
- **Experiment calibration:** Anchors MMM estimates to real lift test results
        """)

    st.markdown("---")

    # ---- MODEL 2 ----
    st.header("2. Markov Chain Attribution")
    st.markdown("*Question it answers: How do customers move through the ad journey?*")

    with st.expander("Plain English", expanded=True):
        st.markdown("""
Think of a customer's path to buying as a series of steps:

**See TV ad** -> **Google the brand** -> **Click a social ad** -> **Buy**

Last-touch attribution would give ALL the credit to social (the last step).
But that's unfair -- the customer only Googled because they saw the TV ad first!

Markov chains model this as a **flow chart**. At each step, there's a probability of
moving to the next channel, dropping off, or converting. The key insight is the
**removal effect**: "If we completely removed TV from the flow chart, how many fewer
people would reach 'Buy'?" That drop is TV's true contribution.

It's like asking: "What would happen if we removed a bridge from a road network?
How much more traffic would get stuck?"
        """)

    with st.expander("The Technical Bit"):
        st.markdown("""
- **Transition matrix:** P[i,j] = probability of moving from channel i to channel j
- **Absorbing states:** "Conversion" and "Drop-off" are terminal states
- **Removal effect:** Set all transitions from channel i to "Drop-off", recompute conversion probability
- The drop in conversion probability = that channel's contribution
- Normalize removal effects across channels to get attribution shares
        """)

    st.markdown("---")

    # ---- MODEL 3 ----
    st.header("3. Shapley Value Attribution")
    st.markdown("*Question it answers: What's the fairest way to split credit across channels?*")

    with st.expander("Plain English", expanded=True):
        st.markdown("""
Imagine 6 friends work on a group project and get a grade. How do you fairly decide
who contributed how much? You could look at what each person added when they joined
different team combinations:

- Alice working alone: B grade
- Alice + Bob: B+ grade
- Alice + Bob + Carol: A- grade

The **Shapley value** considers EVERY possible team combination and averages out
each person's marginal contribution. It's the only method mathematically proven to be
"fair" -- it satisfies 4 axioms from game theory that no other method does.

We do this with 6 marketing channels: try every possible combination (2^6 = 64 total),
measure how well each combo predicts revenue, and compute each channel's average
contribution across all combos.
        """)

    with st.expander("The Technical Bit"):
        st.markdown("""
- **Value function:** v(S) = R-squared of LightGBM model trained on only channels in coalition S
- **Exact computation:** All 64 coalitions evaluated (tractable with 6 channels)
- **4 axioms:** Efficiency (shares sum to total), Symmetry, Linearity, Null Player
- **Interaction index:** Measures synergy -- do TV + Search together produce MORE than separately?
- **Asymmetric Shapley:** Respects causal ordering (TV comes before Search in the journey)
        """)

    st.markdown("---")

    # ---- MODEL 4 ----
    st.header("4. DeepCausalMMM")
    st.markdown("*Question it answers: Can a neural network learn advertising effects that humans have to guess?*")

    with st.expander("Plain English", expanded=True):
        st.markdown("""
In traditional MMM, a human analyst has to decide things like:
- "How quickly does a TV ad's effect fade?" (the decay rate)
- "At what point does more spending stop helping?" (the saturation curve)

These are just guesses. What if we let a **neural network figure them out from the data**?

That's DeepCausalMMM. It uses a type of neural network called a **GRU** (Gated Recurrent
Unit) that's designed for time-series data. Instead of us telling it "TV ads decay by
30% per week," the GRU watches the pattern of spend-and-revenue over time and learns
the decay pattern itself.

The **DAG structure learning** part is even cooler: the model automatically discovers
which channels influence which. It might learn that "TV spending causes an increase in
search volume" without us telling it to look for that.

Think of it as the difference between hand-drawing a map vs letting a drone survey the
terrain and draw the map automatically.
        """)

    with st.expander("The Technical Bit"):
        st.markdown("""
- **GRU Adstock:** Replaces geometric decay with a learned recurrent transformation per channel
- **Learned Saturation:** 2-layer MLP with sigmoid learns diminishing returns curve
- **NOTEARS** (Zheng et al., 2018): Learns adjacency matrix W with DAG constraint `trace(e^(W*W)) = d`
- **Loss:** MSE + DAG penalty + L1 sparsity on W
- **Reference:** DeepCausalMMM (arXiv 2024), tested on 190 DMAs, 109 weeks, 13 channels
        """)

    st.markdown("---")

    # ---- MODEL 5 ----
    st.header("5. Double Machine Learning (DML)")
    st.markdown("*Question it answers: What's the TRUE causal effect of ad spend, removing all the noise?*")

    with st.expander("Plain English", expanded=True):
        st.markdown("""
Here's a trap that catches most analysts: TV ad spend goes up in December.
Revenue also goes up in December (because of Christmas shopping). A simple
model would say "TV ads caused the revenue increase!" -- but maybe people
were going to buy anyway because of the holidays.

This is called **confounding** -- when a hidden factor (seasonality) affects
both the cause (ad spend) and the effect (revenue), creating a fake correlation.

**Double ML fixes this in two steps:**

1. **Predict away the confounders:** Use a powerful ML model (LightGBM) to
   predict what revenue WOULD have been based on seasonality alone, and what
   ad spend WOULD have been based on seasonality alone.

2. **Look at the leftovers:** The difference between actual and predicted is
   the "surprise" part. If surprise-high-TV-spend weeks also have surprise-high-
   revenue, THAT's the causal effect -- because the seasonal pattern was already
   removed.

It's like measuring if studying helps test scores, but first removing the effect
of "some students are naturally smarter" -- you isolate the pure effect of studying.
        """)

    with st.expander("The Technical Bit"):
        st.markdown("""
- **Orthogonalization:** Y_tilde = Y - E[Y|X], T_tilde = T - E[T|X] (remove confounders)
- **Cross-fitting:** K-fold to avoid overfitting bias
- **ATE:** Average Treatment Effect = regression of Y_tilde on T_tilde
- **CausalForest** (Athey & Imbens, 2018): Finds heterogeneous effects (does TV work differently by season?)
- **Key advantage:** Valid p-values and confidence intervals even though ML was used
        """)

    st.markdown("---")

    # ---- MODEL 6 ----
    st.header("6. Temporal Fusion Transformer (TFT)")
    st.markdown("*Question it answers: Can deep learning forecast revenue from ad spend patterns?*")

    with st.expander("Plain English", expanded=True):
        st.markdown("""
You know how ChatGPT reads your whole message and pays more attention to the important
words? **Transformers** use an "attention mechanism" that decides which parts of the
input matter most.

TFT applies this to time series data. Given 52 weeks of ad spend and revenue, it:

1. **Picks which inputs matter** -- maybe Social spend is important but Display isn't.
   It learns this automatically (Variable Selection Network).

2. **Looks at the past** -- similar to how you might look at "what happened last
   Christmas" to predict this Christmas. The attention mechanism highlights which
   past weeks are most relevant.

3. **Gives you uncertainty** -- instead of saying "revenue will be $800K," it says
   "revenue will be between $700K and $900K, most likely $800K."

**Honest caveat:** With only 52 weeks of data, this model overfits (memorizes the data
rather than learning general patterns). It's here to demonstrate the architecture --
real deployment needs 200+ weeks.
        """)

    with st.expander("The Technical Bit"):
        st.markdown("""
- **Variable Selection Network:** Softmax attention weights over inputs -> learned feature importance
- **LSTM Encoder:** Captures sequential temporal dependencies
- **Multi-Head Attention:** Identifies which past timesteps matter for each prediction
- **Quantile outputs:** Predicts 10th, 50th, 90th percentiles (prediction intervals)
- **Architecture:** Based on Google Research (Lim et al., 2021)
        """)

    st.markdown("---")

    # ---- MODEL 7 ----
    st.header("7. Conformal Prediction")
    st.markdown("*Question it answers: How confident should we actually be in our predictions?*")

    with st.expander("Plain English", expanded=True):
        st.markdown("""
Most ML models give you a prediction but don't tell you how reliable it is.
Is "$800K predicted revenue" actually reliable? Could it be $600K or $1M?

Conformal prediction wraps ANY model with a **guarantee**: "If I say the true value
is between $700K and $900K, I'm right at least 90% of the time." No ifs, no buts.

How? It looks at how wrong the model was on past data, and uses those errors to
build prediction intervals. If the model was often wrong by +/- $100K, the interval
will be about $200K wide.

**Three flavors:**
- **Split Conformal:** Simple. Train on 70%, measure errors on 30%, build intervals.
- **Jackknife+:** Cleverer. Uses all the data through cross-validation.
- **CQR:** Smartest. Makes wider intervals when the model is less sure, narrower
  when it's confident. (Like how a weather forecast might say "definitely sunny"
  for tomorrow but "50-80F, could rain" for next week.)
        """)

    with st.expander("The Technical Bit"):
        st.markdown("""
- **Guarantee:** Coverage >= 1-alpha under exchangeability (weaker than i.i.d.)
- **Split Conformal:** interval = prediction +/- quantile(|residuals|, 1-alpha)
- **Jackknife+:** K-fold cross-conformal, coverage >= 1-2*alpha
- **CQR** (Conformalized Quantile Regression): adaptive intervals via calibrated quantile models
- **Based on:** Vovk et al. (2005), Angelopoulos & Bates (2021)
        """)

    st.markdown("---")

    # ---- MODEL 8 ----
    st.header("8. Geo-Lift / Synthetic Control")
    st.markdown("*Question it answers: If we turned off TV ads in one city, what would actually happen?*")

    with st.expander("Plain English", expanded=True):
        st.markdown("""
All the other models use math to ESTIMATE the effect of ads. This one actually
TESTS it -- or at least simulates a test.

Imagine you have 8 cities. In City A, you run a big TV campaign. In the other 7 cities,
you don't. At the end, City A's revenue went up 15%. But was that because of the TV ads,
or would it have happened anyway?

To find out, you build a **"synthetic City A"** -- a weighted mix of the other 7 cities
that, before the campaign, tracked City A's revenue almost perfectly. If the synthetic
version predicted $1M and City A actually made $1.15M, that extra $150K is the
**causal lift** from TV ads.

The beauty is the **placebo test**: run the same analysis pretending each control city
is the treatment city. None of them should show an effect (because they didn't run
the campaign). If City A's effect stands out from the placebos, you can be confident
it's real.

This is what Google and Meta actually use to measure ad effectiveness.
        """)

    with st.expander("The Technical Bit"):
        st.markdown("""
- **Synthetic control** (Abadie et al., 2010): weighted convex combination of donor units
- **Constraints:** w_j >= 0, sum(w_j) = 1
- **Causal lift:** actual - synthetic counterfactual in post-intervention period
- **Placebo tests:** run on all control units for falsification
- **2025 benchmark:** 225 geo-based tests found median iROAS of 2.31x, CTV at 3.30x
        """)

    st.markdown("---")

    # ---- WHY 8 MODELS ----
    st.header("Why Use 8 Models Instead of 1?")
    st.markdown("""
Each model answers a slightly different question and makes different assumptions.
When they **agree**, you can be confident. When they **disagree**, that's where
you need to investigate further.

| Model | Best For | Think of it as... |
|-------|----------|-------------------|
| **Bayesian MMM** | Budget planning | "How should we allocate next quarter's budget?" |
| **Markov Chains** | Understanding the journey | "How do customers flow from TV to purchase?" |
| **Shapley Values** | Fair credit | "Who deserves the bonus?" (mathematically fair) |
| **DeepCausalMMM** | Letting data speak | "What if we let AI figure out the ad effects?" |
| **Double ML** | Removing bias | "What's the REAL effect after removing seasonal noise?" |
| **TFT** | Forecasting | "What will revenue be next month given this spend plan?" |
| **Conformal** | Honest uncertainty | "How wide should our error bars actually be?" |
| **Geo-Lift** | Ground truth | "Let's actually run an experiment and measure it" |

The industry calls this **triangulation** -- using multiple independent methods to
converge on the truth, rather than trusting any single model.
    """)

    st.markdown("---")
    st.header("Technical Stack")
    st.markdown("""
- **Python:** pandas, numpy, scikit-learn, LightGBM, PyMC, PyTorch
- **Causal ML:** EconML (Microsoft), Shapley game theory, NOTEARS DAG learning
- **Uncertainty:** MAPIE (conformal), PyMC (Bayesian)
- **Visualization:** Plotly, Matplotlib, Streamlit
    """)


# ============================================
# FALLBACK
# ============================================

else:
    st.error("Page '{}' not found. Use the sidebar to navigate.".format(page))
