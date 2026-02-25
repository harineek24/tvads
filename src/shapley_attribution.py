"""
Shapley Value Attribution for Marketing Channels
==================================================
Game-theoretic approach to fairly distribute total revenue credit across
marketing channels using cooperative game theory (Shapley values).

This is the 2024/2025 industry-standard approach that supersedes last-touch,
first-touch, and even Markov chain attribution. Shapley values are the UNIQUE
allocation satisfying four fairness axioms: efficiency, symmetry, linearity,
and null-player.

Three methods implemented:

1. **Exact Shapley Values**: For each of the 2^6 = 64 coalitions of channels,
   train a LightGBM model and use R^2 as the value function v(S). Then apply
   the classical Shapley formula to compute each channel's fair share.

2. **Shapley-Owen Interaction Index**: Pairwise synergy measurement. Captures
   whether two channels together create super-additive value (e.g., the
   TV -> Search halo effect).

3. **Asymmetric Shapley Values**: Incorporates causal ordering
   (TV airs first -> users search -> then convert) so that only orderings
   consistent with the causal graph are considered.

Outputs:
- data/shapley_values.csv
- data/shapley_interactions.csv
- data/shapley_asymmetric.csv
- data/shapley_summary.json
- results/shapley_attribution.png
- results/shapley_interactions.png
- results/shapley_marginal.png
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import math
import itertools
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPEND_COLUMNS = [
    'tv_broadcast_spend',
    'tv_cable_spend',
    'tv_streaming_spend',
    'paid_search_spend',
    'social_spend',
    'display_spend',
]

CHANNEL_NAMES = {
    'tv_broadcast_spend': 'TV Broadcast',
    'tv_cable_spend': 'TV Cable',
    'tv_streaming_spend': 'TV Streaming',
    'paid_search_spend': 'Paid Search',
    'social_spend': 'Social',
    'display_spend': 'Display',
}

# Causal ordering for asymmetric Shapley (index = priority, lower = earlier)
CAUSAL_ORDER = [
    'tv_broadcast_spend',   # 0 - airs first
    'tv_cable_spend',       # 1
    'tv_streaming_spend',   # 2
    'social_spend',         # 3
    'paid_search_spend',    # 4 - users search after seeing TV/social
    'display_spend',        # 5 - retargeting / last mile
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


# ===========================================================================
# 1.  Data loading and feature engineering
# ===========================================================================
def load_data():
    """Load weekly spend data and engineer seasonality features."""
    path = os.path.join(DATA_DIR, 'weekly_spend.csv')
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} weekly observations from {path}")

    # Seasonality features (always present in every coalition model)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['trend'] = np.arange(len(df), dtype=float)

    print(f"  Engineered seasonality features: month_sin, month_cos, trend")
    print(f"  Target variable: total_revenue  (mean = ${df['total_revenue'].mean():,.0f})")
    return df


# ===========================================================================
# 2.  Coalition value function  v(S) = R^2 of LightGBM on coalition S
# ===========================================================================
SEASONALITY_COLS = ['month_sin', 'month_cos', 'trend']


def coalition_value(df, coalition_columns):
    """
    Train a LightGBM regressor using only the specified channel spend columns
    (plus always-present seasonality features) and return in-sample R^2 as
    the value function v(S).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with all columns.
    coalition_columns : list[str]
        Subset of SPEND_COLUMNS that belong to this coalition.

    Returns
    -------
    float
        R^2 score (can be negative for very poor models; floored at 0).
    """
    feature_cols = SEASONALITY_COLS + list(coalition_columns)
    X = df[feature_cols].values
    y = df['total_revenue'].values

    # If the coalition is empty, only seasonality features are used
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'num_leaves': 15,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': 1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    r2 = model.score(X, y)
    return max(r2, 0.0)


def compute_all_coalition_values(df):
    """
    Enumerate all 2^n coalitions and compute v(S) for each.

    Returns
    -------
    dict
        Mapping from frozenset of channel indices -> R^2 value.
    """
    n = len(SPEND_COLUMNS)
    total = 2 ** n
    print(f"\n  Computing value function for all {total} coalitions ...")

    values = {}
    for i in range(total):
        # Bitmask -> coalition
        coalition = []
        for j in range(n):
            if i & (1 << j):
                coalition.append(SPEND_COLUMNS[j])
        key = frozenset(coalition)
        values[key] = coalition_value(df, coalition)

        if (i + 1) % 16 == 0 or i + 1 == total:
            print(f"    [{i + 1:3d}/{total}] coalitions evaluated")

    # Report a few notable values
    print(f"\n  v(empty)     = {values[frozenset()]:8.4f}")
    print(f"  v(all)       = {values[frozenset(SPEND_COLUMNS)]:8.4f}")
    for col in SPEND_COLUMNS:
        print(f"  v({{{CHANNEL_NAMES[col]:12s}}}) = {values[frozenset([col])]:8.4f}")

    return values


# ===========================================================================
# 3a.  Exact Shapley Values
# ===========================================================================
def exact_shapley_values(values):
    """
    Compute exact Shapley values using the combinatorial formula:

        phi_i = SUM over S not containing i:
                  [ |S|! * (n - |S| - 1)! / n! ] * [ v(S + {i}) - v(S) ]

    With n=6, this is 2^5 = 32 terms per channel.
    """
    n = len(SPEND_COLUMNS)
    factorial = math.factorial
    n_fact = factorial(n)

    shapley = {}
    for i, channel in enumerate(SPEND_COLUMNS):
        phi = 0.0
        # Iterate over all subsets of SPEND_COLUMNS \ {channel}
        others = [c for c in SPEND_COLUMNS if c != channel]
        for r in range(len(others) + 1):
            for combo in itertools.combinations(others, r):
                S = frozenset(combo)
                S_plus_i = S | {channel}
                s_size = len(S)
                weight = factorial(s_size) * factorial(n - s_size - 1) / n_fact
                marginal = values[S_plus_i] - values[S]
                phi += weight * marginal
        shapley[channel] = phi

    return shapley


# ===========================================================================
# 3b.  Shapley-Owen Interaction Index
# ===========================================================================
def shapley_owen_interaction(values):
    """
    Compute pairwise interaction indices (Shapley-Owen):

        phi_ij = SUM over S not containing i or j:
                   [ |S|! * (n - |S| - 2)! / (n - 1)! ] *
                   [ v(S+{i,j}) - v(S+{i}) - v(S+{j}) + v(S) ]

    Positive = super-additive synergy; Negative = redundancy.
    """
    n = len(SPEND_COLUMNS)
    factorial = math.factorial
    nm1_fact = factorial(n - 1)

    interactions = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            ch_i = SPEND_COLUMNS[i]
            ch_j = SPEND_COLUMNS[j]
            others = [c for c in SPEND_COLUMNS if c != ch_i and c != ch_j]

            phi_ij = 0.0
            for r in range(len(others) + 1):
                for combo in itertools.combinations(others, r):
                    S = frozenset(combo)
                    s_size = len(S)
                    weight = factorial(s_size) * factorial(n - s_size - 2) / nm1_fact

                    v_ij = values[S | {ch_i, ch_j}]
                    v_i = values[S | {ch_i}]
                    v_j = values[S | {ch_j}]
                    v_s = values[S]

                    phi_ij += weight * (v_ij - v_i - v_j + v_s)

            interactions[i, j] = phi_ij
            interactions[j, i] = phi_ij  # symmetric

    return interactions


# ===========================================================================
# 3c.  Asymmetric Shapley Values (with causal ordering)
# ===========================================================================
def asymmetric_shapley_values(values):
    """
    Compute asymmetric Shapley values that respect a causal ordering.

    With causal ordering  ch_0 > ch_1 > ... > ch_{n-1}  (earlier = higher
    priority), we only consider permutations consistent with the partial
    order. For a total order, there is exactly ONE consistent permutation,
    so the asymmetric Shapley value of channel i is simply its marginal
    contribution when added at its position in the causal order.

    For a richer analysis we use a relaxed version: for each channel i, we
    consider all permutations where the *relative order among channels with
    a direct causal link to i* is preserved. In practice, with our total
    order, we average over permutations where each channel can only appear
    after all channels that causally precede it.

    Implementation: We generate all topological sorts of the causal DAG
    (here the DAG is a total order, so there is exactly 1 topological sort).
    For the total-order case, the asymmetric Shapley value degenerates to
    the sequential marginal contribution along the causal chain. To make
    the analysis richer, we also compute a *relaxed* version where we group
    channels into tiers and allow permutations within tiers:

        Tier 0: TV Broadcast, TV Cable, TV Streaming  (awareness)
        Tier 1: Social                                 (engagement)
        Tier 2: Paid Search                            (intent capture)
        Tier 3: Display                                (retargeting)
    """
    n = len(CAUSAL_ORDER)

    # --- Strict causal ordering (single permutation) ---
    strict_values = {}
    running_coalition = frozenset()
    for channel in CAUSAL_ORDER:
        v_before = values[running_coalition]
        running_coalition = running_coalition | {channel}
        v_after = values[running_coalition]
        strict_values[channel] = v_after - v_before

    # --- Tiered ordering (average over within-tier permutations) ---
    tiers = [
        ['tv_broadcast_spend', 'tv_cable_spend', 'tv_streaming_spend'],
        ['social_spend'],
        ['paid_search_spend'],
        ['display_spend'],
    ]

    # Generate all orderings: permute within each tier, keep tier order
    tier_perms = []
    for tier in tiers:
        tier_perms.append(list(itertools.permutations(tier)))

    all_orderings = list(itertools.product(*tier_perms))
    print(f"\n  Asymmetric Shapley: {len(all_orderings)} causal-consistent orderings")

    tiered_values = {ch: 0.0 for ch in CAUSAL_ORDER}
    for ordering_tuple in all_orderings:
        perm = []
        for tier_perm in ordering_tuple:
            perm.extend(tier_perm)
        running = frozenset()
        for channel in perm:
            v_before = values[running]
            running = running | {channel}
            v_after = values[running]
            tiered_values[channel] += (v_after - v_before)

    n_orderings = len(all_orderings)
    for ch in CAUSAL_ORDER:
        tiered_values[ch] /= n_orderings

    return strict_values, tiered_values


# ===========================================================================
# 4.  Marginal contribution curves (for budget scaling plot)
# ===========================================================================
def marginal_contribution_curves(df, values_cache=None):
    """
    For each channel, compute Shapley value at different budget scale factors
    (0.25x, 0.5x, 0.75x, 1.0x, 1.25x, 1.5x, 2.0x) to show how the
    channel's importance changes with budget.
    """
    scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    results = {ch: [] for ch in SPEND_COLUMNS}

    print(f"\n  Computing marginal contribution curves at {len(scales)} budget scales ...")

    for s_idx, scale in enumerate(scales):
        df_scaled = df.copy()
        for col in SPEND_COLUMNS:
            df_scaled[col] = df[col] * scale

        # Recompute all coalition values at this scale
        n = len(SPEND_COLUMNS)
        vals = {}
        for i in range(2 ** n):
            coalition = []
            for j in range(n):
                if i & (1 << j):
                    coalition.append(SPEND_COLUMNS[j])
            key = frozenset(coalition)
            vals[key] = coalition_value(df_scaled, coalition)

        # Compute Shapley values at this scale
        shapley = exact_shapley_values(vals)
        for ch in SPEND_COLUMNS:
            results[ch].append(shapley[ch])

        print(f"    Scale {scale:.2f}x done ({s_idx + 1}/{len(scales)})")

    return scales, results


# ===========================================================================
# 5.  Proportional (naive) attribution baseline
# ===========================================================================
def proportional_attribution(df):
    """Simple proportional attribution: credit = share of total spend."""
    total_by_channel = df[SPEND_COLUMNS].sum()
    grand_total = total_by_channel.sum()
    proportions = total_by_channel / grand_total
    return proportions.to_dict()


# ===========================================================================
# 6.  Output Generation
# ===========================================================================
def save_shapley_values(shapley, output_path):
    """Save exact Shapley values to CSV."""
    total_phi = sum(shapley.values())
    rows = []
    for col in SPEND_COLUMNS:
        rows.append({
            'channel_column': col,
            'channel_name': CHANNEL_NAMES[col],
            'shapley_value': shapley[col],
            'shapley_pct': shapley[col] / total_phi * 100 if total_phi > 0 else 0,
        })
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)
    print(f"\n  Saved exact Shapley values -> {output_path}")
    return result_df


def save_interaction_matrix(interactions, output_path):
    """Save Shapley-Owen interaction indices as a 6x6 matrix CSV."""
    names = [CHANNEL_NAMES[c] for c in SPEND_COLUMNS]
    idf = pd.DataFrame(interactions, index=names, columns=names)
    idf.to_csv(output_path)
    print(f"  Saved interaction matrix     -> {output_path}")
    return idf


def save_asymmetric_values(strict, tiered, output_path):
    """Save asymmetric Shapley values to CSV."""
    total_strict = sum(strict.values())
    total_tiered = sum(tiered.values())
    rows = []
    for col in CAUSAL_ORDER:
        rows.append({
            'channel_column': col,
            'channel_name': CHANNEL_NAMES[col],
            'causal_position': CAUSAL_ORDER.index(col),
            'strict_shapley': strict[col],
            'strict_pct': strict[col] / total_strict * 100 if total_strict > 0 else 0,
            'tiered_shapley': tiered[col],
            'tiered_pct': tiered[col] / total_tiered * 100 if total_tiered > 0 else 0,
        })
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)
    print(f"  Saved asymmetric values      -> {output_path}")
    return result_df


def save_summary_json(shapley, interactions, strict, tiered, proportional,
                      coalition_values, output_path):
    """Save summary metrics and key findings to JSON."""
    n = len(SPEND_COLUMNS)

    # Find largest interaction (TV-Search synergy)
    max_inter_val = -np.inf
    max_inter_pair = ("", "")
    for i in range(n):
        for j in range(i + 1, n):
            if interactions[i, j] > max_inter_val:
                max_inter_val = interactions[i, j]
                max_inter_pair = (CHANNEL_NAMES[SPEND_COLUMNS[i]],
                                  CHANNEL_NAMES[SPEND_COLUMNS[j]])

    # TV-Search synergy specifically
    tv_idx_broadcast = SPEND_COLUMNS.index('tv_broadcast_spend')
    search_idx = SPEND_COLUMNS.index('paid_search_spend')
    tv_search_synergy = interactions[tv_idx_broadcast, search_idx]

    # Total TV Shapley share
    tv_cols = ['tv_broadcast_spend', 'tv_cable_spend', 'tv_streaming_spend']
    total_phi = sum(shapley.values())
    tv_total_share = sum(shapley[c] for c in tv_cols) / total_phi * 100 if total_phi > 0 else 0

    summary = {
        'method': 'Shapley Value Attribution (Game Theory)',
        'total_coalitions_evaluated': 2 ** n,
        'channels': n,
        'value_function': 'LightGBM R-squared (100 trees, max_depth=4)',
        'v_empty_coalition': coalition_values[frozenset()],
        'v_grand_coalition': coalition_values[frozenset(SPEND_COLUMNS)],
        'exact_shapley_values': {
            CHANNEL_NAMES[c]: round(shapley[c], 6) for c in SPEND_COLUMNS
        },
        'exact_shapley_pct': {
            CHANNEL_NAMES[c]: round(shapley[c] / total_phi * 100, 2)
            for c in SPEND_COLUMNS
        } if total_phi > 0 else {},
        'proportional_attribution_pct': {
            CHANNEL_NAMES[c]: round(proportional[c] * 100, 2)
            for c in SPEND_COLUMNS
        },
        'key_findings': {
            'strongest_pairwise_synergy': {
                'pair': list(max_inter_pair),
                'interaction_index': round(float(max_inter_val), 6),
            },
            'tv_search_synergy': round(float(tv_search_synergy), 6),
            'total_tv_shapley_share_pct': round(tv_total_share, 2),
            'causal_ordering_used': [CHANNEL_NAMES[c] for c in CAUSAL_ORDER],
        },
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary JSON           -> {output_path}")
    return summary


# ===========================================================================
# 7.  Visualizations
# ===========================================================================
def plot_attribution_comparison(shapley, asymmetric_tiered, proportional,
                                output_path):
    """
    Stacked bar chart comparing Exact Shapley vs Asymmetric Shapley vs
    simple proportional attribution.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    methods = ['Exact Shapley', 'Asymmetric\nShapley (Tiered)', 'Proportional\n(Spend Share)']
    channel_names = [CHANNEL_NAMES[c] for c in SPEND_COLUMNS]
    colors = ['#2196F3', '#1565C0', '#0D47A1', '#FF9800', '#4CAF50', '#9C27B0']

    # Normalize each to percentages
    exact_total = sum(shapley[c] for c in SPEND_COLUMNS)
    asym_total = sum(asymmetric_tiered[c] for c in SPEND_COLUMNS)

    exact_pcts = [shapley[c] / exact_total * 100 if exact_total > 0 else 0
                  for c in SPEND_COLUMNS]
    asym_pcts = [asymmetric_tiered[c] / asym_total * 100 if asym_total > 0 else 0
                 for c in SPEND_COLUMNS]
    prop_pcts = [proportional[c] * 100 for c in SPEND_COLUMNS]

    data = np.array([exact_pcts, asym_pcts, prop_pcts])
    x = np.arange(len(methods))
    bar_width = 0.5

    bottom = np.zeros(len(methods))
    for ch_idx in range(len(SPEND_COLUMNS)):
        bars = ax.bar(x, data[:, ch_idx], bar_width, bottom=bottom,
                      label=channel_names[ch_idx], color=colors[ch_idx],
                      edgecolor='white', linewidth=0.5)
        # Add percentage labels for segments > 5%
        for m_idx in range(len(methods)):
            if data[m_idx, ch_idx] > 5:
                ax.text(x[m_idx], bottom[m_idx] + data[m_idx, ch_idx] / 2,
                        f'{data[m_idx, ch_idx]:.1f}%',
                        ha='center', va='center', fontsize=8,
                        fontweight='bold', color='white')
        bottom += data[:, ch_idx]

    ax.set_ylabel('Attribution Share (%)', fontsize=12)
    ax.set_title('Shapley Value Attribution vs Proportional Spend Share',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', bbox_to_anchor=(1.22, 1.0), fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=100, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved attribution chart      -> {output_path}")


def plot_interaction_heatmap(interactions, output_path):
    """Heatmap of pairwise Shapley-Owen interaction indices."""
    fig, ax = plt.subplots(figsize=(9, 8))

    names = [CHANNEL_NAMES[c] for c in SPEND_COLUMNS]

    # Use diverging colormap: blue = redundancy, red = synergy
    max_abs = max(abs(interactions.min()), abs(interactions.max()))
    if max_abs == 0:
        max_abs = 1.0
    im = ax.imshow(interactions, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs,
                   aspect='equal')

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(names, fontsize=10)

    # Annotate cells
    for i in range(len(names)):
        for j in range(len(names)):
            val = interactions[i, j]
            color = 'white' if abs(val) > max_abs * 0.6 else 'black'
            if i == j:
                text = '-'
            else:
                text = f'{val:.4f}'
            ax.text(j, i, text, ha='center', va='center', fontsize=8,
                    color=color, fontweight='bold' if i != j else 'normal')

    # Highlight TV Broadcast <-> Paid Search cell
    tv_b_idx = SPEND_COLUMNS.index('tv_broadcast_spend')
    search_idx = SPEND_COLUMNS.index('paid_search_spend')
    for (ri, ci) in [(tv_b_idx, search_idx), (search_idx, tv_b_idx)]:
        rect = plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1, linewidth=3,
                              edgecolor='gold', facecolor='none')
        ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Interaction Index (+ synergy / - redundancy)', fontsize=10)

    ax.set_title('Shapley-Owen Pairwise Interaction Indices\n'
                 '(Gold border = TV Broadcast <-> Paid Search)',
                 fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved interaction heatmap    -> {output_path}")


def plot_marginal_curves(scales, marginal_results, output_path):
    """
    Line chart: for each channel, show how its Shapley value changes as
    total budget scales from 0.25x to 2.0x.
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    colors = ['#2196F3', '#1565C0', '#0D47A1', '#FF9800', '#4CAF50', '#9C27B0']
    markers = ['o', 's', '^', 'D', 'v', 'P']

    for idx, col in enumerate(SPEND_COLUMNS):
        vals = marginal_results[col]
        ax.plot(scales, vals, color=colors[idx], marker=markers[idx],
                linewidth=2, markersize=7, label=CHANNEL_NAMES[col])

    ax.set_xlabel('Budget Scale Factor', fontsize=12)
    ax.set_ylabel('Shapley Value (R^2 contribution)', fontsize=12)
    ax.set_title('Marginal Contribution Curves: Shapley Value vs Budget Scale',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=9)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Current budget')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved marginal curves        -> {output_path}")


# ===========================================================================
# 8.  Main Execution
# ===========================================================================
def main():
    print("=" * 70)
    print("  SHAPLEY VALUE ATTRIBUTION FOR MARKETING CHANNELS")
    print("  Game-Theoretic Fair Credit Allocation")
    print("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Load Data ---
    print("\n[1/8] Loading data ...")
    df = load_data()

    # --- Compute all coalition values ---
    print("\n[2/8] Computing coalition value function v(S) for all 2^6 = 64 coalitions ...")
    coalition_values = compute_all_coalition_values(df)

    # --- Exact Shapley Values ---
    print("\n[3/8] Computing exact Shapley values ...")
    shapley = exact_shapley_values(coalition_values)
    total_phi = sum(shapley.values())
    print("\n  Exact Shapley Values:")
    print("  " + "-" * 45)
    for col in SPEND_COLUMNS:
        pct = shapley[col] / total_phi * 100 if total_phi > 0 else 0
        print(f"    {CHANNEL_NAMES[col]:15s}:  {shapley[col]:.6f}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':15s}:  {total_phi:.6f}  (100.0%)")

    # --- Shapley-Owen Interaction Index ---
    print("\n[4/8] Computing Shapley-Owen pairwise interaction indices ...")
    interactions = shapley_owen_interaction(coalition_values)
    print("\n  Top synergies (positive = super-additive):")
    pairs = []
    n = len(SPEND_COLUMNS)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((interactions[i, j], SPEND_COLUMNS[i], SPEND_COLUMNS[j]))
    pairs.sort(key=lambda x: abs(x[0]), reverse=True)
    for val, c1, c2 in pairs[:5]:
        label = "SYNERGY" if val > 0 else "REDUNDANCY"
        print(f"    {CHANNEL_NAMES[c1]:15s} x {CHANNEL_NAMES[c2]:15s}: "
              f"{val:+.6f}  [{label}]")

    # --- Asymmetric Shapley Values ---
    print("\n[5/8] Computing asymmetric Shapley values with causal ordering ...")
    strict_asym, tiered_asym = asymmetric_shapley_values(coalition_values)
    total_tiered = sum(tiered_asym.values())
    print("\n  Asymmetric Shapley Values (tiered):")
    print("  " + "-" * 45)
    for col in CAUSAL_ORDER:
        pct = tiered_asym[col] / total_tiered * 100 if total_tiered > 0 else 0
        print(f"    {CHANNEL_NAMES[col]:15s}:  {tiered_asym[col]:.6f}  ({pct:5.1f}%)")

    # --- Proportional baseline ---
    print("\n[6/8] Computing proportional (spend-share) attribution baseline ...")
    proportional = proportional_attribution(df)
    print("  Proportional Attribution:")
    for col in SPEND_COLUMNS:
        print(f"    {CHANNEL_NAMES[col]:15s}:  {proportional[col] * 100:5.1f}%")

    # --- Save data outputs ---
    print("\n[7/8] Saving output files ...")
    save_shapley_values(shapley, os.path.join(DATA_DIR, 'shapley_values.csv'))
    save_interaction_matrix(interactions,
                            os.path.join(DATA_DIR, 'shapley_interactions.csv'))
    save_asymmetric_values(strict_asym, tiered_asym,
                           os.path.join(DATA_DIR, 'shapley_asymmetric.csv'))
    summary = save_summary_json(shapley, interactions, strict_asym, tiered_asym,
                                proportional, coalition_values,
                                os.path.join(DATA_DIR, 'shapley_summary.json'))

    # --- Visualizations ---
    print("\n[8/8] Generating visualizations ...")
    plot_attribution_comparison(shapley, tiered_asym, proportional,
                                os.path.join(RESULTS_DIR, 'shapley_attribution.png'))
    plot_interaction_heatmap(interactions,
                            os.path.join(RESULTS_DIR, 'shapley_interactions.png'))

    # Marginal contribution curves (requires recomputing at multiple scales)
    scales, marginal_results = marginal_contribution_curves(df)
    plot_marginal_curves(scales, marginal_results,
                         os.path.join(RESULTS_DIR, 'shapley_marginal.png'))

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("  SHAPLEY ATTRIBUTION COMPLETE")
    print("=" * 70)
    print(f"\n  Key Findings:")
    kf = summary['key_findings']
    pair = kf['strongest_pairwise_synergy']['pair']
    print(f"    - Strongest synergy: {pair[0]} x {pair[1]} "
          f"(interaction = {kf['strongest_pairwise_synergy']['interaction_index']:.4f})")
    print(f"    - TV-Search synergy index: {kf['tv_search_synergy']:.4f}")
    print(f"    - Total TV Shapley share: {kf['total_tv_shapley_share_pct']:.1f}%")
    print(f"\n  Output Files:")
    print(f"    data/shapley_values.csv        - Per-channel exact Shapley values")
    print(f"    data/shapley_interactions.csv   - 6x6 interaction index matrix")
    print(f"    data/shapley_asymmetric.csv     - Asymmetric Shapley (causal order)")
    print(f"    data/shapley_summary.json       - Summary metrics & key findings")
    print(f"    results/shapley_attribution.png - Comparison chart")
    print(f"    results/shapley_interactions.png- Interaction heatmap")
    print(f"    results/shapley_marginal.png    - Marginal contribution curves")
    print()


if __name__ == '__main__':
    main()
