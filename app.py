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
    "Core Models": ["Overview", "Bayesian MMM", "Shapley Attribution", "DeepCausalMMM"],
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
st.sidebar.caption("DTC Brand | $15M Annual | 52 Weeks | 8 DMAs | 7 Models")


# ============================================
# PAGE: OVERVIEW
# ============================================

if page == "overview":
    st.title("TV Ad Attribution & Advanced Causal ML")
    st.markdown(
        "A cutting-edge marketing analytics platform with **7 advanced statistical and ML models** "
        "measuring TV advertising effectiveness. No simple heuristics -- every model here uses "
        "**causal inference, Bayesian uncertainty, or deep learning**."
    )

    total_tv = ws['tv_broadcast_spend'].sum() + ws['tv_cable_spend'].sum() + ws['tv_streaming_spend'].sum()
    total_spend = ws['total_spend'].sum()
    total_rev = ws['total_revenue'].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", format_currency(total_rev))
    c2.metric("Total Ad Spend", format_currency(total_spend))
    c3.metric("TV Share of Spend", "{:.0f}%".format(total_tv / total_spend * 100))
    c4.metric("Advanced Models", "7")

    st.markdown("---")

    st.subheader("The 7 Models")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Core Models**
- **Bayesian MMM** -- PyMC MCMC with causal DAG identification + experiment calibration (Google Meridian / PyMC-Marketing style)
- **Shapley Attribution** -- Exact game-theoretic fair attribution across 64 coalitions with interaction indices
- **DeepCausalMMM** -- Neural network (GRU) learns adstock patterns + DAG structure learning discovers channel causality

**Causal ML**
- **Double ML** -- Debiased causal effects via Chernozhukov et al. (2018) with CausalForest heterogeneity
        """)
    with col2:
        st.markdown("""
**Causal ML (continued)**
- **Transformer (TFT)** -- Temporal Fusion Transformer with learned variable selection and multi-head attention
- **Conformal Prediction** -- Distribution-free uncertainty with guaranteed coverage (Split, Jackknife+, CQR)
- **Geo-Lift** -- Synthetic control incrementality testing with placebo falsification

**Synthesis**
- **Model Comparison** -- All 7 methods side-by-side: where they agree = confidence, where they disagree = uncertainty
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
    st.title("7-Model Comparison")
    st.markdown(
        "All 7 advanced methods applied to the same data. Where models **agree** = higher "
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
            "Bayesian MMM": "#8e44ad", "Shapley Values": "#2c3e50",
            "DeepCausalMMM": "#e74c3c", "Double ML": "#2980b9",
            "TFT Attention": "#27ae60", "Geo-Lift": "#e67e22", "Conformal": "#34495e",
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
    st.title("How It Works: Model Methodology Guide")
    st.markdown("In-depth explanations of all 7 models, the math behind them, "
                "and why each one matters for 2025 marketing analytics.")

    st.markdown("---")
    st.header("1. Bayesian MMM with Causal DAG & Experiment Calibration")

    with st.expander("Bayesian MMM (PyMC-Marketing / Google Meridian)", expanded=False):
        st.markdown("""
**What it does:** Full Bayesian estimation of channel ROAS with posterior distributions,
causal DAG identification, and experiment calibration.

**The 3 innovations over standard MMM:**

1. **Causal DAG Identification** (PyMC Labs, 2024):
   When TV causally drives brand search, a naive model that estimates both simultaneously
   produces biased results. The fix: specify a proper causal graph
   `TV -> Search Volume -> Conversions` and model the TV->Search pathway explicitly.

2. **Experiment Calibration** (Zhang et al., 2024):
   Reparametrize MMM in terms of ROAS rather than regression coefficients. Calibrate
   through saturation curves using geo-holdout lift test results. PyMC-Marketing implements
   this for continuous refinement as more experiments run.

3. **Full Bayesian Inference** (MCMC):
   Posterior distributions over ROAS: "TV Broadcast ROAS is 1.2x [0.8, 1.6] with 90% probability"
   -- not just a point estimate.

**Model:**
```
revenue ~ Normal(mu, sigma)
mu = intercept + sum(beta_i * hill(adstock(spend_i))) + TV*Search_synergy + seasonality
```

**Priors:** HalfNormal on betas (positive), Beta on decays, Normal on synergy terms.
MCMC: 2 chains x 2000 draws, NUTS sampler, target_accept=0.90.
        """)

    st.markdown("---")
    st.header("2. Shapley Value Attribution")

    with st.expander("Game-Theoretic Attribution (Exact Shapley + Interactions)", expanded=False):
        st.markdown("""
**What it does:** Distributes total revenue across channels using Shapley values from
cooperative game theory -- the ONLY method satisfying all 4 fairness axioms.

**The 4 Axioms:**
- **Efficiency:** Attribution sums to total value (no leakage)
- **Symmetry:** Equal contributors get equal credit
- **Linearity:** Attribution of a sum = sum of attributions
- **Null Player:** Channels that contribute nothing get zero credit

**How it works:**
1. For all 2^6 = 64 coalitions of 6 channels, train LightGBM to predict revenue
2. Value function v(S) = R-squared of model using only channels in S
3. Shapley value: phi_i = weighted average of marginal contributions across all orderings

**Shapley-Owen Interaction Index:**
Measures pairwise synergies. For channels i,j:
phi_ij = weighted average of [v(S+{i,j}) - v(S+{i}) - v(S+{j}) + v(S)]
Positive = super-additive (TV + Search together > TV alone + Search alone)

**Asymmetric Shapley:** Respects causal ordering (TV fires first, then Search converts).
Only considers orderings consistent with the causal graph.
        """)

    st.markdown("---")
    st.header("3. DeepCausalMMM")

    with st.expander("Neural Adstock + DAG Structure Learning (2024/2025 Frontier)", expanded=False):
        st.markdown("""
**What it does:** Replaces hand-crafted adstock decay and saturation curves with
neural networks that LEARN them from data. Simultaneously discovers the causal DAG
between channels.

**Architecture:**
1. **GRU Adstock:** One GRU cell per channel processes spend time series sequentially,
   learning carry-over patterns. Output = adstocked representation.
   (Replaces: geometric decay `x[t] = spend[t] + decay * x[t-1]`)

2. **Learned Saturation:** 2-layer MLP per channel with sigmoid output learns the
   diminishing-returns curve shape. (Replaces: Hill function with fixed alpha, K)

3. **NOTEARS DAG Learning** (Zheng et al., 2018): Learns 6x6 adjacency matrix W.
   DAG constraint: `h(W) = trace(exp(W * W)) - d = 0` ensures acyclicity.
   Discovered edges reveal which channels causally influence which.

4. **Output:** `revenue = sum(beta_i * saturated_i * dag_effective_i) + seasonality + trend`

**Training Loss:** MSE + lambda_dag * h(W) + lambda_sparse * L1(W)
Three-phase schedule: focus on fit -> ramp up DAG penalty -> full regularization.

**Reference:** DeepCausalMMM package (arXiv 2024), tested on 190 DMAs, 109 weeks, 13 channels.
        """)

    st.markdown("---")
    st.header("4. Double Machine Learning (DML)")

    with st.expander("Debiased Causal Effects (Chernozhukov et al., 2018)", expanded=False):
        st.markdown("""
**The problem:** OLS conflates correlation with causation. If brands spend more on TV
during Q4 when revenue is naturally high, OLS attributes seasonal revenue to TV.

**How DML fixes this:**
1. **First stage (nuisance):** Use LightGBM to predict Y_hat = E[Revenue | Confounders]
   and T_hat = E[Spend | Confounders]
2. **Residualize:** Y_tilde = Y - Y_hat, T_tilde = T - T_hat
3. **Second stage (causal):** Regress Y_tilde ~ T_tilde -> debiased ATE

**Cross-fitting:** K-fold to avoid overfitting bias. Train nuisance on fold k,
predict on fold -k.

**CausalForest extension** (Athey & Imbens, 2018): Detects heterogeneous treatment
effects by season. Does TV work differently in Q4 vs Q2?

**Key output:** "Each $1 of TV Broadcast spend causally generates $X in revenue,
controlling for all confounders, with p-value Y."
        """)

    st.markdown("---")
    st.header("5. Temporal Fusion Transformer (TFT)")

    with st.expander("Deep Learning Forecasting with Interpretability (Lim et al., 2021)", expanded=False):
        st.markdown("""
**Architecture:**
1. **Variable Selection Network (VSN):** Learns which input features matter via softmax
   attention weights. Provides feature importance *learned* by the model.
2. **LSTM Encoder:** Captures temporal dependencies in weekly observations
3. **Multi-Head Attention:** Identifies which past weeks are most relevant
4. **Quantile Outputs:** Predicts 10th, 50th, 90th percentiles for uncertainty

**Why TFT for marketing:**
- Handles multiple input features natively
- Interpretable attention over past timesteps
- Learns nonlinear feature interactions
- Calibrated prediction intervals

**Caveat:** 52 weekly observations is insufficient. Real deployment needs 200+.
        """)

    st.markdown("---")
    st.header("6. Conformal Prediction")

    with st.expander("Distribution-Free Uncertainty (Vovk et al., 2005)", expanded=False):
        st.markdown("""
**The guarantee:** If data is exchangeable, a (1-alpha) conformal prediction interval
covers the true value with probability >= 1-alpha. Period. No distributional assumptions.

**Three methods:**
1. **Split Conformal:** Train on 70%, calibrate residuals on 30%.
   Interval = prediction +/- quantile(|residuals|). Simple but wastes data.

2. **Jackknife+:** K-fold cross-conformal. Coverage >= 1-2*alpha.
   Uses all data for both training and calibration.

3. **CQR (Conformalized Quantile Regression):** Trains quantile regressors for bounds,
   then calibrates with conformal scores. *Adaptive* intervals: wider when uncertain.

**Why this matters:** Standard ML gives no reliability guarantee. Bayesian CI requires
correct priors. Conformal just works.
        """)

    st.markdown("---")
    st.header("7. Geo-Lift / Synthetic Control")

    with st.expander("Incrementality Testing (Abadie et al., 2010)", expanded=False):
        st.markdown("""
**What it does:** The 'ground truth' for measuring ad effectiveness. Constructs a
synthetic control from donor DMAs to estimate what would have happened without ads.

**How it works:**
1. Select treatment DMA (highest spend ramp-up)
2. Pre-period: find weights w_j for donor DMAs so synthetic matches treatment
3. Post-period: causal lift = actual - synthetic counterfactual
4. Placebo tests: run on control DMAs for falsification

**Why this is the gold standard in 2025:**
- A 2025 benchmarking study of 225 geo-based tests found median incremental ROAS
  of 2.31x across channels, with CTV leading at 3.30x
- Google GeoLift and Meta use this exact methodology
- No parallel trends assumption needed (unlike diff-in-diff)
- Haus builds "Causal MMM" -- MMM founded on incrementality experiments
        """)

    st.markdown("---")
    st.header("The Triangulation Approach (2025 Industry Standard)")
    st.markdown("""
The industry has converged on using **multiple methods together**:

| Layer | Method | Purpose |
|-------|--------|---------|
| **Strategic Planning** | Bayesian MMM | Long-term budget allocation with uncertainty |
| **Causal Ground Truth** | Geo-Lift / Incrementality | Validate MMM estimates with experiments |
| **Fair Attribution** | Shapley Values | Game-theoretically fair credit allocation |
| **Frontier** | DeepCausalMMM | Let neural nets learn what we currently hand-specify |
| **Debiased Effects** | Double ML | Control for confounding in observational data |
| **Uncertainty** | Conformal Prediction | Guaranteed prediction intervals |
| **Forecasting** | TFT | Deep learning with interpretable attention |

**No single model is trusted alone.** Use disagreement between methods as a signal
for where to invest in better measurement.
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
