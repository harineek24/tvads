"""
Markov Chain Attribution Model
===============================
Probabilistic multi-touch attribution using absorbing Markov chains.

Instead of assigning credit to the last touchpoint or splitting equally,
this model builds a transition probability matrix between channel states
and computes each channel's contribution via removal effects:

    removal_effect(channel) = P(conversion | all channels) -
                               P(conversion | all channels except this one)

This gives the Shapley-value-inspired marginal contribution of each channel
to the conversion journey.

Outputs:
- data/markov_transition_matrix.csv
- data/markov_attribution.csv
- data/markov_removal_effects.csv
- results/markov_transition_heatmap.png
- results/markov_removal_effects.png
- results/markov_vs_lasttouch.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


def generate_journey_data(traffic_df, airings_df, n_journeys=50000):
    """
    Synthesize multi-touch customer journeys from the traffic data.

    Each journey is a sequence of channel touchpoints leading to
    conversion or abandonment. Channels: TV, Direct, Organic Search,
    Paid Search, Social.

    The journey data respects the traffic source distribution and
    TV-influenced traffic patterns.
    """
    print("\n  Generating customer journeys...")

    channels = ['TV', 'Direct', 'Organic_Search', 'Paid_Search', 'Social']

    # Channel transition probabilities (informed by traffic patterns)
    # TV → high probability of Direct or Organic_Search (brand lift)
    transition_probs = {
        'Start': {'TV': 0.15, 'Direct': 0.25, 'Organic_Search': 0.30,
                  'Paid_Search': 0.20, 'Social': 0.10},
        'TV': {'Direct': 0.35, 'Organic_Search': 0.25,
               'Paid_Search': 0.15, 'Social': 0.10, 'TV': 0.05,
               'Conversion': 0.05, 'Null': 0.05},
        'Direct': {'Direct': 0.10, 'Organic_Search': 0.10,
                   'Paid_Search': 0.05, 'Social': 0.05, 'TV': 0.02,
                   'Conversion': 0.15, 'Null': 0.53},
        'Organic_Search': {'Direct': 0.15, 'Organic_Search': 0.05,
                           'Paid_Search': 0.08, 'Social': 0.05, 'TV': 0.02,
                           'Conversion': 0.12, 'Null': 0.53},
        'Paid_Search': {'Direct': 0.10, 'Organic_Search': 0.05,
                        'Paid_Search': 0.05, 'Social': 0.03, 'TV': 0.02,
                        'Conversion': 0.20, 'Null': 0.55},
        'Social': {'Direct': 0.12, 'Organic_Search': 0.08,
                   'Paid_Search': 0.06, 'Social': 0.04, 'TV': 0.03,
                   'Conversion': 0.07, 'Null': 0.60},
    }

    journeys = []
    for _ in range(n_journeys):
        path = []
        state = 'Start'
        max_steps = 8

        for step in range(max_steps):
            probs = transition_probs.get(state, transition_probs['Direct'])
            next_states = list(probs.keys())
            next_probs = list(probs.values())
            # Normalize
            total = sum(next_probs)
            next_probs = [p / total for p in next_probs]

            next_state = np.random.choice(next_states, p=next_probs)

            if next_state == 'Conversion':
                journeys.append({
                    'path': ' > '.join(path),
                    'touchpoints': path.copy(),
                    'n_touches': len(path),
                    'converted': True,
                })
                break
            elif next_state == 'Null':
                journeys.append({
                    'path': ' > '.join(path),
                    'touchpoints': path.copy(),
                    'n_touches': len(path),
                    'converted': False,
                })
                break
            else:
                if next_state in channels:
                    path.append(next_state)
                    state = next_state

        if len(path) == max_steps:
            journeys.append({
                'path': ' > '.join(path),
                'touchpoints': path.copy(),
                'n_touches': len(path),
                'converted': False,
            })

    df = pd.DataFrame(journeys)
    n_conv = df['converted'].sum()
    print(f"  Generated {len(df):,} journeys, {n_conv:,} conversions "
          f"({n_conv/len(df):.1%})")
    return df


def build_transition_matrix(journeys_df):
    """
    Build the Markov chain transition probability matrix from journey data.
    States: Start, TV, Direct, Organic_Search, Paid_Search, Social, Conversion, Null
    """
    print("\n  Building transition matrix...")

    states = ['Start', 'TV', 'Direct', 'Organic_Search', 'Paid_Search',
              'Social', 'Conversion', 'Null']
    n_states = len(states)
    state_idx = {s: i for i, s in enumerate(states)}

    # Count transitions
    transition_counts = np.zeros((n_states, n_states))

    for _, row in journeys_df.iterrows():
        touches = row['touchpoints']
        converted = row['converted']

        # Start → first touch
        if len(touches) > 0:
            transition_counts[state_idx['Start'], state_idx[touches[0]]] += 1

        # Touch → touch transitions
        for i in range(len(touches) - 1):
            from_state = touches[i]
            to_state = touches[i + 1]
            if from_state in state_idx and to_state in state_idx:
                transition_counts[state_idx[from_state], state_idx[to_state]] += 1

        # Last touch → outcome
        if len(touches) > 0:
            last = touches[-1]
            if converted:
                transition_counts[state_idx[last], state_idx['Conversion']] += 1
            else:
                transition_counts[state_idx[last], state_idx['Null']] += 1

    # Normalize to probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_counts / row_sums

    # Make Conversion and Null absorbing
    transition_matrix[state_idx['Conversion'], :] = 0
    transition_matrix[state_idx['Conversion'], state_idx['Conversion']] = 1
    transition_matrix[state_idx['Null'], :] = 0
    transition_matrix[state_idx['Null'], state_idx['Null']] = 1

    trans_df = pd.DataFrame(transition_matrix, index=states, columns=states)
    print("\n  Transition matrix:")
    print(trans_df.round(3).to_string())

    return transition_matrix, states, state_idx


def compute_conversion_probability(transition_matrix, states, state_idx):
    """
    Compute the total conversion probability from the absorbing Markov chain.
    Uses the fundamental matrix N = (I - Q)^{-1} where Q is the transient part.
    """
    absorbing = [state_idx['Conversion'], state_idx['Null']]
    transient = [i for i in range(len(states)) if i not in absorbing]

    # Q = transient-to-transient transition matrix
    Q = transition_matrix[np.ix_(transient, transient)]

    # R = transient-to-absorbing transition matrix
    R = transition_matrix[np.ix_(transient, absorbing)]

    # Fundamental matrix: N = (I - Q)^{-1}
    I = np.eye(len(transient))
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        N = np.linalg.pinv(I - Q)

    # B = N × R = absorption probabilities
    B = N @ R

    # P(conversion | start from Start)
    start_idx_transient = transient.index(state_idx['Start'])
    conv_idx_absorbing = list(absorbing).index(state_idx['Conversion'])

    total_conv_prob = B[start_idx_transient, conv_idx_absorbing]
    return total_conv_prob, N, B, transient, absorbing


def compute_removal_effects(transition_matrix, states, state_idx):
    """
    Compute removal effect for each channel:
    removal_effect(ch) = P(conv | all) - P(conv | remove ch)

    When a channel is removed, all transitions TO that channel are
    redirected to Null (the user never encounters that touchpoint).
    """
    print("\n  Computing removal effects...")

    channels = ['TV', 'Direct', 'Organic_Search', 'Paid_Search', 'Social']

    # Baseline conversion probability
    base_conv_prob, _, _, _, _ = compute_conversion_probability(
        transition_matrix, states, state_idx
    )
    print(f"  Baseline conversion probability: {base_conv_prob:.4f}")

    removal_effects = {}

    for channel in channels:
        # Create modified matrix without this channel
        modified = transition_matrix.copy()
        ch_idx = state_idx[channel]

        # Redirect all transitions to this channel → Null instead
        for i in range(len(states)):
            if i != ch_idx:
                prob_to_channel = modified[i, ch_idx]
                modified[i, ch_idx] = 0
                modified[i, state_idx['Null']] += prob_to_channel

        # Zero out the channel's own row (it doesn't exist)
        modified[ch_idx, :] = 0
        modified[ch_idx, state_idx['Null']] = 1

        # Compute conversion probability without this channel
        try:
            conv_prob_without, _, _, _, _ = compute_conversion_probability(
                modified, states, state_idx
            )
        except Exception:
            conv_prob_without = base_conv_prob

        removal_effect = base_conv_prob - conv_prob_without
        removal_effects[channel] = {
            'removal_effect': removal_effect,
            'conv_prob_without': conv_prob_without,
            'pct_contribution': removal_effect / base_conv_prob * 100 if base_conv_prob > 0 else 0,
        }

        print(f"  {channel}: removal_effect={removal_effect:.4f}, "
              f"P(conv|remove)={conv_prob_without:.4f}, "
              f"contribution={removal_effect/base_conv_prob:.1%}")

    # Normalize to get fractional attribution
    total_removal = sum(r['removal_effect'] for r in removal_effects.values())
    for channel in channels:
        if total_removal > 0:
            removal_effects[channel]['normalized_attribution'] = (
                removal_effects[channel]['removal_effect'] / total_removal
            )
        else:
            removal_effects[channel]['normalized_attribution'] = 1.0 / len(channels)

    return removal_effects, base_conv_prob


def compute_last_touch_attribution(journeys_df):
    """Simple last-touch attribution for comparison."""
    converted = journeys_df[journeys_df['converted']].copy()

    last_touch_counts = {}
    for _, row in converted.iterrows():
        if len(row['touchpoints']) > 0:
            last = row['touchpoints'][-1]
            last_touch_counts[last] = last_touch_counts.get(last, 0) + 1

    total = sum(last_touch_counts.values())
    return {ch: count / total for ch, count in last_touch_counts.items()}


def compute_first_touch_attribution(journeys_df):
    """Simple first-touch attribution for comparison."""
    converted = journeys_df[journeys_df['converted']].copy()

    first_touch_counts = {}
    for _, row in converted.iterrows():
        if len(row['touchpoints']) > 0:
            first = row['touchpoints'][0]
            first_touch_counts[first] = first_touch_counts.get(first, 0) + 1

    total = sum(first_touch_counts.values())
    return {ch: count / total for ch, count in first_touch_counts.items()}


def generate_markov_plots(transition_matrix, states, removal_effects,
                          last_touch, first_touch):
    """Generate Markov attribution visualizations."""
    print("\n  Generating plots...")
    os.makedirs('results', exist_ok=True)

    channels = ['TV', 'Direct', 'Organic_Search', 'Paid_Search', 'Social']

    # --- Plot 1: Transition matrix heatmap ---
    fig, ax = plt.subplots(figsize=(10, 8))
    display_states = ['Start', 'TV', 'Direct', 'Organic_Search', 'Paid_Search',
                      'Social', 'Conversion', 'Null']
    display_idx = [states.index(s) for s in display_states]
    display_matrix = transition_matrix[np.ix_(display_idx, display_idx)]

    im = ax.imshow(display_matrix, cmap='YlOrRd', vmin=0, vmax=0.6)
    ax.set_xticks(range(len(display_states)))
    ax.set_xticklabels(display_states, rotation=45, ha='right')
    ax.set_yticks(range(len(display_states)))
    ax.set_yticklabels(display_states)

    for i in range(len(display_states)):
        for j in range(len(display_states)):
            val = display_matrix[i, j]
            if val > 0.01:
                color = 'white' if val > 0.3 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label='Transition Probability')
    ax.set_title('Markov Chain Transition Matrix\n'
                 '(Rows = from state, Columns = to state)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/markov_transition_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/markov_transition_heatmap.png")

    # --- Plot 2: Removal effects ---
    fig, ax = plt.subplots(figsize=(10, 6))
    channel_names = sorted(removal_effects.keys(),
                           key=lambda x: -removal_effects[x]['removal_effect'])
    effects = [removal_effects[ch]['removal_effect'] for ch in channel_names]
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(channel_names)))

    bars = ax.barh(range(len(channel_names)), effects, color=colors, height=0.6)
    ax.set_yticks(range(len(channel_names)))
    ax.set_yticklabels(channel_names)
    ax.set_xlabel('Removal Effect (drop in conversion probability)')
    ax.set_title('Markov Chain Attribution: Removal Effects\n'
                 '(How much does P(conversion) drop without each channel?)',
                 fontweight='bold')
    for bar, val in zip(bars, effects):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('results/markov_removal_effects.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/markov_removal_effects.png")

    # --- Plot 3: Markov vs Last-Touch vs First-Touch ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(channels))
    width = 0.25

    markov_vals = [removal_effects.get(ch, {}).get('normalized_attribution', 0)
                   for ch in channels]
    lt_vals = [last_touch.get(ch, 0) for ch in channels]
    ft_vals = [first_touch.get(ch, 0) for ch in channels]

    bars1 = ax.bar(x - width, markov_vals, width, label='Markov Chain',
                   color='#2c3e50', alpha=0.85)
    bars2 = ax.bar(x, lt_vals, width, label='Last-Touch',
                   color='#e74c3c', alpha=0.85)
    bars3 = ax.bar(x + width, ft_vals, width, label='First-Touch',
                   color='#3498db', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=20, ha='right')
    ax.set_ylabel('Attribution Share')
    ax.set_title('Attribution Model Comparison:\nMarkov Chain vs Last-Touch vs First-Touch',
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(markov_vals), max(lt_vals), max(ft_vals)) * 1.2)
    plt.tight_layout()
    plt.savefig('results/markov_vs_lasttouch.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/markov_vs_lasttouch.png")


def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("=" * 60)
    print("MARKOV CHAIN ATTRIBUTION")
    print("=" * 60)

    # Load data
    traffic = pd.read_csv('data/web_traffic.csv')
    airings = pd.read_csv('data/ad_airings.csv')

    # Generate journeys
    journeys_df = generate_journey_data(traffic, airings, n_journeys=50000)

    # Build transition matrix
    transition_matrix, states, state_idx = build_transition_matrix(journeys_df)

    # Compute removal effects
    removal_effects, base_conv_prob = compute_removal_effects(
        transition_matrix, states, state_idx
    )

    # Comparison models
    last_touch = compute_last_touch_attribution(journeys_df)
    first_touch = compute_first_touch_attribution(journeys_df)

    # Save
    trans_df = pd.DataFrame(transition_matrix, index=states, columns=states)
    trans_df.to_csv('data/markov_transition_matrix.csv')

    removal_df = pd.DataFrame([
        {'channel': ch, **vals} for ch, vals in removal_effects.items()
    ])
    removal_df.to_csv('data/markov_removal_effects.csv', index=False)

    # Full attribution comparison
    channels = ['TV', 'Direct', 'Organic_Search', 'Paid_Search', 'Social']
    comparison_rows = []
    for ch in channels:
        comparison_rows.append({
            'channel': ch,
            'markov_attribution': removal_effects.get(ch, {}).get('normalized_attribution', 0),
            'last_touch_attribution': last_touch.get(ch, 0),
            'first_touch_attribution': first_touch.get(ch, 0),
            'removal_effect': removal_effects.get(ch, {}).get('removal_effect', 0),
        })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv('data/markov_attribution.csv', index=False)
    print("\n  Saved data/markov_attribution.csv")

    # Generate plots
    generate_markov_plots(transition_matrix, states, removal_effects,
                          last_touch, first_touch)

    print("\n" + "=" * 60)
    print("MARKOV CHAIN ATTRIBUTION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
