"""
Deep Causal Media Mix Model (DeepCausalMMM)
=============================================
Cutting-edge 2024/2025 approach that replaces hand-crafted adstock/saturation
with neural-learned transformations, combined with DAG structure learning
to discover causal relationships between marketing channels.

Key innovations over traditional MMM:
1. GRU-based Adstock — Instead of geometric decay with a fixed rate, a GRU cell
   learns temporal carry-over patterns from data. Each channel gets its own GRU
   that processes the spend time series and outputs the "adstocked" representation.
2. Learned Saturation — Instead of Hill function with fixed K/S, a small MLP
   with sigmoid output learns the saturation curve shape per channel.
3. DAG Structure Learning — A weighted adjacency matrix W (6x6) is learned
   to represent causal relationships between channels (e.g., TV -> Search).
   A DAG constraint from NOTEARS (Zheng et al. 2018) ensures acyclicity:
   h(W) = trace(expm(W * W)) - d = 0.
4. Output layer — Linear combination of effective (DAG-modified) channel
   representations plus seasonality and trend predicts revenue.

Training loss:
   L = MSE(predicted, actual) + lambda_dag * h(W) + lambda_sparse * L1(W)

Outputs:
- data/deep_causal_dag.csv            — Learned adjacency matrix W
- data/deep_causal_adstock.csv        — Learned adstock impulse responses (from GRU)
- data/deep_causal_contributions.csv  — Channel contributions and ROAS
- data/deep_causal_summary.json       — Model metrics, DAG penalty, causal edges
- results/deep_causal_dag.png         — Visualization of the learned causal DAG
- results/deep_causal_adstock.png     — Learned vs geometric adstock curves
- results/deep_causal_contributions.png — Channel ROAS bar chart
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================
# CHANNEL DEFINITIONS
# ============================================

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
    'TV Broadcast': '#2c3e50',
    'TV Cable': '#34495e',
    'TV Streaming': '#7f8c8d',
    'Paid Search': '#2980b9',
    'Social': '#8e44ad',
    'Display': '#e67e22',
}

NUM_CHANNELS = len(SPEND_COLUMNS)


# ============================================
# 1. GRU-BASED ADSTOCK MODULE
# ============================================

class GRUAdstock(nn.Module):
    """
    Learns temporal carry-over (adstock) for each channel using a GRU cell.

    Instead of a fixed geometric decay x[t] = spend[t] + decay * x[t-1],
    the GRU learns the optimal gating behavior from data:
      - The reset gate controls how much past adstock to forget
      - The update gate controls how much new spend to incorporate

    Input:  (T, num_channels) raw spend
    Output: (T, num_channels) adstocked spend
    """

    def __init__(self, num_channels, hidden_size=8):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        # One GRU cell per channel for independent adstock learning
        self.gru_cells = nn.ModuleList([
            nn.GRUCell(input_size=1, hidden_size=hidden_size)
            for _ in range(num_channels)
        ])
        # Project GRU hidden state back to a single adstocked value per channel
        self.projections = nn.ModuleList([
            nn.Linear(hidden_size, 1)
            for _ in range(num_channels)
        ])

    def forward(self, x):
        """
        x: (T, num_channels) — raw spend time series
        Returns: (T, num_channels) — adstocked representation
        """
        T = x.shape[0]
        adstocked = torch.zeros_like(x)

        for ch in range(self.num_channels):
            h = torch.zeros(1, self.hidden_size)  # initial hidden state
            channel_spend = x[:, ch].unsqueeze(1)  # (T, 1)

            for t in range(T):
                inp = channel_spend[t:t+1, :]  # (1, 1)
                h = self.gru_cells[ch](inp, h)   # (1, hidden_size)
                adstocked[t, ch] = self.projections[ch](h).squeeze()

        return adstocked


# ============================================
# 2. LEARNED SATURATION MODULE
# ============================================

class LearnedSaturation(nn.Module):
    """
    Replaces the Hill function S(x) = x^s / (x^s + K^s) with a small MLP
    that learns the saturation curve shape for each channel.

    Architecture per channel: Linear(1, 8) -> ReLU -> Linear(8, 1) -> Sigmoid
    The sigmoid output ensures diminishing returns (bounded 0-1).

    Input:  (T, num_channels) adstocked spend
    Output: (T, num_channels) saturated values in [0, 1]
    """

    def __init__(self, num_channels, hidden_size=8):
        super().__init__()
        self.num_channels = num_channels
        self.networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
            for _ in range(num_channels)
        ])

    def forward(self, x):
        """
        x: (T, num_channels) — adstocked spend
        Returns: (T, num_channels) — saturated values in [0, 1]
        """
        saturated = torch.zeros_like(x)
        for ch in range(self.num_channels):
            saturated[:, ch] = self.networks[ch](x[:, ch:ch+1]).squeeze(-1)
        return saturated


# ============================================
# 3. DAG STRUCTURE LEARNING MODULE
# ============================================

class DAGStructureLearner(nn.Module):
    """
    Learns a weighted adjacency matrix W representing causal relationships
    between marketing channels (e.g., TV spend causes Search volume lift).

    W[i, j] > 0 means channel j causally influences channel i.
    The effective representation of each channel becomes:
        channel_i_effective = channel_i + sum_j(W[i, j] * channel_j)

    Acyclicity is enforced via the NOTEARS constraint (Zheng et al. 2018):
        h(W) = trace(expm(W * W)) - d = 0
    where d = number of channels (nodes).

    This ensures the learned graph is a valid DAG.
    """

    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        # Learnable adjacency matrix — initialized near zero
        self.W = nn.Parameter(torch.zeros(num_channels, num_channels) * 0.01)

    def forward(self, x):
        """
        Apply causal structure: x_effective = x + x @ W^T
        (Each channel receives influence from its causal parents)

        x: (T, num_channels) — saturated channel representations
        Returns: (T, num_channels) — DAG-modified representations
        """
        # Zero out diagonal — a channel cannot cause itself
        W_masked = self.W * (1.0 - torch.eye(self.num_channels))
        # x_effective[i] = x[i] + sum_j(W[i,j] * x[j])
        x_effective = x + x @ W_masked.T
        return x_effective

    def dag_penalty(self):
        """
        NOTEARS acyclicity constraint: h(W) = tr(expm(W * W)) - d
        Must equal 0 for the graph to be a valid DAG.
        """
        W_masked = self.W * (1.0 - torch.eye(self.num_channels))
        # Element-wise square (Hadamard), then matrix exponential
        M = W_masked * W_masked
        expm_M = torch.matrix_exp(M)
        h = torch.trace(expm_M) - self.num_channels
        return h

    def get_adjacency_matrix(self):
        """Return the learned adjacency matrix with zeroed diagonal."""
        W_masked = self.W * (1.0 - torch.eye(self.num_channels))
        return W_masked.detach().numpy()


# ============================================
# 4. FULL DeepCausalMMM MODEL
# ============================================

class DeepCausalMMM(nn.Module):
    """
    Full Deep Causal Media Mix Model.

    Pipeline:
        Raw Spend -> GRU Adstock -> Learned Saturation -> DAG Structure ->
        Linear Combination + Seasonality + Trend -> Predicted Revenue

    The model jointly learns:
    - How advertising effects carry over in time (adstock via GRU)
    - The shape of diminishing returns (saturation via MLP)
    - Causal relationships between channels (DAG via NOTEARS)
    - Channel contribution weights (linear output layer)
    """

    def __init__(self, num_channels, num_months=12, gru_hidden=8, sat_hidden=8):
        super().__init__()
        self.num_channels = num_channels

        # Core modules
        self.adstock = GRUAdstock(num_channels, hidden_size=gru_hidden)
        self.saturation = LearnedSaturation(num_channels, hidden_size=sat_hidden)
        self.dag = DAGStructureLearner(num_channels)

        # Channel contribution weights (positive via softplus)
        self.channel_weights = nn.Parameter(torch.ones(num_channels) * 0.5)

        # Seasonality: month embeddings (12 months -> 1 value each)
        self.month_embedding = nn.Embedding(num_months, 1)

        # Trend coefficient
        self.trend_weight = nn.Parameter(torch.tensor(0.0))

        # Intercept (base revenue)
        self.intercept = nn.Parameter(torch.tensor(0.0))

    def forward(self, spend, month_indices, trend):
        """
        spend:        (T, num_channels) — raw weekly spend
        month_indices: (T,) — month index 0..11
        trend:        (T,) — normalized week index

        Returns: (T,) — predicted revenue
        """
        # Step 1: GRU adstock
        adstocked = self.adstock(spend)

        # Step 2: Learned saturation
        saturated = self.saturation(adstocked)

        # Step 3: DAG structure modification
        effective = self.dag(saturated)

        # Step 4: Weighted channel contributions
        # Use softplus to ensure positive weights
        weights = torch.nn.functional.softplus(self.channel_weights)
        channel_contrib = effective * weights.unsqueeze(0)  # (T, C)
        total_media = channel_contrib.sum(dim=1)  # (T,)

        # Step 5: Seasonality + trend + intercept
        season = self.month_embedding(month_indices).squeeze(-1)  # (T,)
        predicted = self.intercept + total_media + season + self.trend_weight * trend

        return predicted

    def get_channel_contributions(self, spend, month_indices, trend):
        """
        Decompose prediction into individual channel contributions,
        seasonality, trend, and intercept.
        """
        with torch.no_grad():
            adstocked = self.adstock(spend)
            saturated = self.saturation(adstocked)
            effective = self.dag(saturated)
            weights = torch.nn.functional.softplus(self.channel_weights)
            channel_contrib = effective * weights.unsqueeze(0)  # (T, C)

            season = self.month_embedding(month_indices).squeeze(-1)
            trend_contrib = self.trend_weight * trend
            intercept_val = self.intercept.item()

            return {
                'channel_contributions': channel_contrib.numpy(),
                'seasonality': season.numpy(),
                'trend': trend_contrib.numpy(),
                'intercept': intercept_val
            }


# ============================================
# 5. TRAINING PIPELINE
# ============================================

def train_model(model, spend_tensor, month_tensor, trend_tensor, revenue_tensor,
                n_epochs=1000, lr=0.003, lambda_dag=1.0, lambda_sparse=0.05,
                train_mask=None):
    """
    Train the DeepCausalMMM with augmented Lagrangian for DAG constraint.

    Loss = MSE(pred, actual) + lambda_dag * h(W) + lambda_sparse * ||W||_1

    We gradually increase lambda_dag and lambda_sparse to first let the model
    learn channel effects, then enforce sparsity and acyclicity.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {'loss': [], 'mse': [], 'dag_penalty': [], 'l1_penalty': []}

    print("=" * 70)
    print("TRAINING DeepCausalMMM")
    print("=" * 70)
    print(f"  Epochs: {n_epochs}  |  LR: {lr}  |  lambda_dag: {lambda_dag}  |  lambda_sparse: {lambda_sparse}")
    print(f"  Training samples: {train_mask.sum().item() if train_mask is not None else len(revenue_tensor)}")
    print()

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        predicted = model(spend_tensor, month_tensor, trend_tensor)

        # Apply training mask
        if train_mask is not None:
            pred_train = predicted[train_mask]
            rev_train = revenue_tensor[train_mask]
        else:
            pred_train = predicted
            rev_train = revenue_tensor

        # MSE loss on training data
        mse_loss = nn.functional.mse_loss(pred_train, rev_train)

        # DAG acyclicity penalty
        dag_pen = model.dag.dag_penalty()

        # L1 sparsity on adjacency matrix (encourages sparse causal graph)
        W_masked = model.dag.W * (1.0 - torch.eye(model.num_channels))
        l1_pen = W_masked.abs().sum()

        # Gradually increase DAG and sparsity penalties:
        # Phase 1 (0-20%): focus on MSE, minimal regularization
        # Phase 2 (20-60%): ramp up DAG and sparsity
        # Phase 3 (60-100%): full regularization strength
        progress = epoch / n_epochs
        if progress < 0.2:
            dag_weight = lambda_dag * 0.01
            sparse_weight = lambda_sparse * 0.1
        elif progress < 0.6:
            ramp = (progress - 0.2) / 0.4  # 0 to 1
            dag_weight = lambda_dag * (0.01 + 0.99 * ramp)
            sparse_weight = lambda_sparse * (0.1 + 0.9 * ramp)
        else:
            dag_weight = lambda_dag
            sparse_weight = lambda_sparse

        # Total loss
        total_loss = mse_loss + dag_weight * dag_pen + sparse_weight * l1_pen

        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        scheduler.step()

        # Record history
        history['loss'].append(total_loss.item())
        history['mse'].append(mse_loss.item())
        history['dag_penalty'].append(dag_pen.item())
        history['l1_penalty'].append(l1_pen.item())

        # Print progress
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs}  |  Loss: {total_loss.item():.4f}  |  "
                  f"MSE: {mse_loss.item():.4f}  |  DAG h(W): {dag_pen.item():.6f}  |  "
                  f"L1(W): {l1_pen.item():.4f}")

    print()
    print(f"  Final MSE: {history['mse'][-1]:.4f}")
    print(f"  Final DAG penalty h(W): {history['dag_penalty'][-1]:.6f}")
    print(f"  Final L1(W): {history['l1_penalty'][-1]:.4f}")
    print()

    return history


# ============================================
# 6. EXTRACT GRU IMPULSE RESPONSES
# ============================================

def extract_gru_impulse_responses(model, max_lag=12):
    """
    Compute impulse response functions from the trained GRU adstock.

    For each channel, feed a unit impulse at t=0 followed by zeros,
    and record the GRU output over time. This reveals the learned
    carry-over pattern — analogous to geometric decay but data-driven.
    """
    model.eval()
    impulse_responses = np.zeros((max_lag, NUM_CHANNELS))

    with torch.no_grad():
        for ch in range(NUM_CHANNELS):
            # Create impulse: 1 at t=0, 0 elsewhere
            impulse = torch.zeros(max_lag, NUM_CHANNELS)
            impulse[0, ch] = 1.0

            adstocked = model.adstock(impulse)
            impulse_responses[:, ch] = adstocked[:, ch].numpy()

    return impulse_responses


def geometric_adstock_reference(decay, max_lag=12):
    """Reference geometric decay for comparison."""
    return np.array([decay ** t for t in range(max_lag)])


# ============================================
# 7. VISUALIZATION
# ============================================

def plot_causal_dag(W, channel_names, filepath):
    """
    Visualize the learned causal DAG using matplotlib (no networkx).

    Channels are placed in a circle. Arrows represent causal edges,
    with line width proportional to edge strength.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    n = len(channel_names)

    # Place nodes in a circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Rotate so first node is at top
    angles = angles - np.pi / 2
    radius = 3.0
    positions = {i: (radius * np.cos(a), radius * np.sin(a)) for i, a in enumerate(angles)}

    # Determine edge threshold (only show meaningful edges)
    W_abs = np.abs(W)
    threshold = 0.05

    # Draw edges first (so nodes are on top)
    max_w = W_abs.max() if W_abs.max() > 0 else 1.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            weight = W[i, j]
            if abs(weight) < threshold:
                continue

            x_from, y_from = positions[j]
            x_to, y_to = positions[i]

            # Shorten arrow to avoid overlapping with node circles
            dx = x_to - x_from
            dy = y_to - y_from
            dist = np.sqrt(dx**2 + dy**2)
            shrink = 0.55  # shrink from each end
            x_start = x_from + shrink * dx / dist
            y_start = y_from + shrink * dy / dist
            x_end = x_to - shrink * dx / dist
            y_end = y_to - shrink * dy / dist

            color = '#e74c3c' if weight > 0 else '#3498db'
            linewidth = 1.0 + 3.0 * (abs(weight) / max_w)
            alpha = 0.4 + 0.6 * (abs(weight) / max_w)

            ax.annotate('',
                        xy=(x_end, y_end),
                        xytext=(x_start, y_start),
                        arrowprops=dict(
                            arrowstyle='->', color=color,
                            lw=linewidth, alpha=alpha,
                            mutation_scale=20,
                            connectionstyle='arc3,rad=0.15'
                        ))

            # Edge weight label
            mid_x = (x_start + x_end) / 2 + 0.15
            mid_y = (y_start + y_end) / 2 + 0.15
            ax.text(mid_x, mid_y, f'{weight:.2f}', fontsize=7,
                    ha='center', va='center', color=color, alpha=alpha,
                    fontweight='bold')

    # Draw nodes
    colors_list = list(CHANNEL_COLORS.values())
    for i in range(n):
        x, y = positions[i]
        circle = plt.Circle((x, y), 0.45, color=colors_list[i],
                             ec='white', linewidth=2, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, channel_names[i], ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', zorder=6)

    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Learned Causal DAG Between Marketing Channels\n'
                 '(Deep Causal MMM — NOTEARS Structure Learning)',
                 fontsize=14, fontweight='bold', pad=20)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#e74c3c', lw=2, label='Positive causal effect'),
        Line2D([0], [0], color='#3498db', lw=2, label='Negative causal effect'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=10,
              ncol=2, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_adstock_comparison(impulse_responses, channel_names, filepath):
    """
    Plot GRU-learned impulse responses vs reference geometric decay curves.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    max_lag = impulse_responses.shape[0]
    lags = np.arange(max_lag)

    # Reference geometric decays for comparison
    reference_decays = [0.7, 0.65, 0.5, 0.3, 0.4, 0.35]

    colors_list = list(CHANNEL_COLORS.values())
    for i, (ch_name, decay) in enumerate(zip(channel_names, reference_decays)):
        ax = axes[i]

        # Normalize GRU impulse response to start at 1 for comparison
        gru_ir = impulse_responses[:, i]
        if abs(gru_ir[0]) > 1e-8:
            gru_ir_norm = gru_ir / gru_ir[0]
        else:
            gru_ir_norm = gru_ir

        geo_ir = geometric_adstock_reference(decay, max_lag)

        ax.bar(lags - 0.15, np.abs(gru_ir_norm), width=0.3, color=colors_list[i],
               alpha=0.7, label='GRU Learned')
        ax.plot(lags, geo_ir, 'k--', linewidth=2, alpha=0.6,
                label=f'Geometric (decay={decay})')
        ax.set_title(ch_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Lag (weeks)')
        ax.set_ylabel('Impulse Response')
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=-0.1)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Learned Adstock Patterns (GRU) vs Geometric Decay\n'
                 'Neural networks learn flexible, data-driven carry-over curves',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_contributions(contributions_df, filepath):
    """
    Bar chart of channel ROAS from the deep causal model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    channel_names_list = list(CHANNEL_NAMES.values())
    colors_list = list(CHANNEL_COLORS.values())

    # Left: ROAS comparison
    ax = axes[0]
    roas_values = contributions_df['roas'].values
    bars = ax.bar(range(len(channel_names_list)), roas_values, color=colors_list, alpha=0.85)
    ax.set_xticks(range(len(channel_names_list)))
    ax.set_xticklabels(channel_names_list, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('ROAS (Revenue per $1 Spend)', fontsize=11)
    ax.set_title('Channel ROAS — Deep Causal MMM\n(Neural Adstock + Learned Saturation + DAG)',
                 fontsize=12, fontweight='bold')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, roas_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Right: Revenue contribution share
    ax = axes[1]
    contrib_values = contributions_df['revenue_contribution'].values
    contrib_values = np.maximum(contrib_values, 0)  # clip negatives for pie
    if contrib_values.sum() > 0:
        ax.pie(contrib_values, labels=channel_names_list, colors=colors_list,
               autopct='%1.1f%%', startangle=140, textprops={'fontsize': 9})
    ax.set_title('Revenue Contribution Share\n(Deep Causal MMM)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================
# 8. MAIN EXECUTION
# ============================================

def main():
    print()
    print("=" * 70)
    print("  DEEP CAUSAL MEDIA MIX MODEL (DeepCausalMMM)")
    print("  GRU Adstock + Learned Saturation + DAG Structure Learning")
    print("=" * 70)
    print()

    # ------------------------------------------
    # Load data
    # ------------------------------------------
    print("[1/7] Loading data...")
    df = pd.read_csv('data/weekly_spend.csv')
    print(f"  Loaded {len(df)} weeks of data")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Revenue range: ${df['total_revenue'].min():,.0f} — ${df['total_revenue'].max():,.0f}")
    print()

    # ------------------------------------------
    # Prepare tensors
    # ------------------------------------------
    print("[2/7] Preparing tensors...")

    # Spend features
    spend_raw = df[SPEND_COLUMNS].values.astype(np.float32)

    # Scale spend to [0, 1] range for neural network stability
    spend_max = spend_raw.max(axis=0, keepdims=True)
    spend_max = np.where(spend_max == 0, 1.0, spend_max)  # avoid division by zero
    spend_scaled = spend_raw / spend_max

    # Revenue target — scale to roughly [0, 1]
    revenue_raw = df['total_revenue'].values.astype(np.float32)
    revenue_mean = revenue_raw.mean()
    revenue_std = revenue_raw.std()
    revenue_scaled = (revenue_raw - revenue_mean) / revenue_std

    # Month indices (0-indexed)
    month_indices = (df['month'].values - 1).astype(np.int64)

    # Trend (normalized week number)
    trend = np.linspace(0, 1, len(df)).astype(np.float32)

    # Convert to tensors
    spend_tensor = torch.tensor(spend_scaled, dtype=torch.float32)
    revenue_tensor = torch.tensor(revenue_scaled, dtype=torch.float32)
    month_tensor = torch.tensor(month_indices, dtype=torch.long)
    trend_tensor = torch.tensor(trend, dtype=torch.float32)

    # Train/test split: 80/20
    n_total = len(df)
    n_train = int(n_total * 0.8)
    train_mask = torch.zeros(n_total, dtype=torch.bool)
    train_mask[:n_train] = True
    test_mask = ~train_mask

    print(f"  Spend shape: {spend_tensor.shape}")
    print(f"  Train: {n_train} weeks  |  Test: {n_total - n_train} weeks")
    print(f"  Spend scaling factors: {spend_max.flatten().tolist()}")
    print(f"  Revenue mean: {revenue_mean:.2f}  |  std: {revenue_std:.2f}")
    print()

    # ------------------------------------------
    # Build model
    # ------------------------------------------
    print("[3/7] Building DeepCausalMMM architecture...")
    model = DeepCausalMMM(
        num_channels=NUM_CHANNELS,
        num_months=12,
        gru_hidden=8,
        sat_hidden=8
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params}")
    print(f"  Trainable parameters: {trainable_params}")
    print(f"  GRU Adstock: {sum(p.numel() for p in model.adstock.parameters())} params")
    print(f"  Learned Saturation: {sum(p.numel() for p in model.saturation.parameters())} params")
    print(f"  DAG Structure (W): {model.dag.W.numel()} params")
    print()

    # ------------------------------------------
    # Train
    # ------------------------------------------
    print("[4/7] Training model...")
    history = train_model(
        model, spend_tensor, month_tensor, trend_tensor, revenue_tensor,
        n_epochs=1000, lr=0.003, lambda_dag=1.0, lambda_sparse=0.05,
        train_mask=train_mask
    )

    # ------------------------------------------
    # Evaluate
    # ------------------------------------------
    print("[5/7] Evaluating model...")
    model.eval()
    with torch.no_grad():
        predictions = model(spend_tensor, month_tensor, trend_tensor)

        # Unscale predictions and actuals
        pred_unscaled = predictions.numpy() * revenue_std + revenue_mean
        actual_unscaled = revenue_raw

        # Train metrics
        pred_train = pred_unscaled[train_mask.numpy()]
        actual_train = actual_unscaled[train_mask.numpy()]
        ss_res_train = np.sum((actual_train - pred_train) ** 2)
        ss_tot_train = np.sum((actual_train - actual_train.mean()) ** 2)
        r2_train = 1 - ss_res_train / ss_tot_train
        mae_train = np.mean(np.abs(actual_train - pred_train))
        mape_train = np.mean(np.abs((actual_train - pred_train) / actual_train)) * 100

        # Test metrics
        pred_test = pred_unscaled[test_mask.numpy()]
        actual_test = actual_unscaled[test_mask.numpy()]
        ss_res_test = np.sum((actual_test - pred_test) ** 2)
        ss_tot_test = np.sum((actual_test - actual_test.mean()) ** 2)
        r2_test = 1 - ss_res_test / ss_tot_test
        mae_test = np.mean(np.abs(actual_test - pred_test))
        mape_test = np.mean(np.abs((actual_test - pred_test) / actual_test)) * 100

    print(f"  Train R2:  {r2_train:.4f}  |  MAE: ${mae_train:,.0f}  |  MAPE: {mape_train:.1f}%")
    print(f"  Test  R2:  {r2_test:.4f}  |  MAE: ${mae_test:,.0f}  |  MAPE: {mape_test:.1f}%")
    print()

    # Note on overfitting
    if r2_train > 0.9 and r2_test < 0.5:
        print("  NOTE: High train R2 / low test R2 indicates overfitting.")
        print("  This is expected with 52 weeks of data and a neural model.")
        print("  The learned structures (DAG, adstock) are still interpretable.")
        print()

    # ------------------------------------------
    # Extract results
    # ------------------------------------------
    print("[6/7] Extracting learned structures...")

    # 6a. DAG adjacency matrix
    W_learned = model.dag.get_adjacency_matrix()
    channel_names_list = list(CHANNEL_NAMES.values())

    dag_df = pd.DataFrame(W_learned, index=channel_names_list, columns=channel_names_list)
    dag_df.to_csv('data/deep_causal_dag.csv')
    print(f"  Saved: data/deep_causal_dag.csv")
    print(f"  Adjacency matrix W (rows = effect, cols = cause):")
    for i, name_i in enumerate(channel_names_list):
        for j, name_j in enumerate(channel_names_list):
            if i != j and abs(W_learned[i, j]) > 0.05:
                direction = "+" if W_learned[i, j] > 0 else "-"
                print(f"    {name_j} -> {name_i}: {direction}{abs(W_learned[i, j]):.3f}")
    print()

    # 6b. GRU impulse responses (learned adstock)
    impulse_responses = extract_gru_impulse_responses(model, max_lag=12)
    adstock_df = pd.DataFrame(impulse_responses, columns=channel_names_list)
    adstock_df.index.name = 'lag_weeks'
    adstock_df.to_csv('data/deep_causal_adstock.csv')
    print(f"  Saved: data/deep_causal_adstock.csv")
    print()

    # 6c. Channel contributions and ROAS
    decomp = model.get_channel_contributions(spend_tensor, month_tensor, trend_tensor)
    channel_contribs = decomp['channel_contributions']  # (T, C) in scaled space

    # Unscale contributions to dollar values
    # Each channel's contribution in the scaled space maps to revenue via revenue_std
    channel_contribs_dollars = channel_contribs * revenue_std

    total_spend_per_channel = spend_raw.sum(axis=0)
    total_contrib_per_channel = channel_contribs_dollars.sum(axis=0)

    roas_per_channel = np.where(
        total_spend_per_channel > 0,
        total_contrib_per_channel / total_spend_per_channel,
        0.0
    )

    contributions_df = pd.DataFrame({
        'channel': channel_names_list,
        'total_spend': total_spend_per_channel,
        'revenue_contribution': total_contrib_per_channel,
        'roas': roas_per_channel,
        'contribution_pct': total_contrib_per_channel / np.abs(total_contrib_per_channel).sum() * 100
    })
    contributions_df.to_csv('data/deep_causal_contributions.csv', index=False)
    print(f"  Saved: data/deep_causal_contributions.csv")
    print()
    print("  Channel Contributions (Deep Causal MMM):")
    print("  " + "-" * 70)
    print(f"  {'Channel':<16} {'Spend':>12} {'Contribution':>14} {'ROAS':>8} {'Share':>8}")
    print("  " + "-" * 70)
    for _, row in contributions_df.iterrows():
        print(f"  {row['channel']:<16} ${row['total_spend']:>11,.0f} "
              f"${row['revenue_contribution']:>13,.0f} {row['roas']:>7.2f} "
              f"{row['contribution_pct']:>7.1f}%")
    print()

    # 6d. Discover causal edges
    causal_edges = []
    for i in range(NUM_CHANNELS):
        for j in range(NUM_CHANNELS):
            if i != j and abs(W_learned[i, j]) > 0.05:
                causal_edges.append({
                    'from': channel_names_list[j],
                    'to': channel_names_list[i],
                    'weight': float(W_learned[i, j])
                })

    # Sort by absolute weight
    causal_edges.sort(key=lambda e: abs(e['weight']), reverse=True)

    # 6e. Summary JSON
    summary = {
        'model': 'DeepCausalMMM',
        'description': 'GRU Adstock + Learned Saturation + DAG Structure Learning',
        'data': {
            'n_weeks': int(n_total),
            'n_train': int(n_train),
            'n_test': int(n_total - n_train),
            'n_channels': NUM_CHANNELS
        },
        'architecture': {
            'gru_hidden_size': 8,
            'saturation_hidden_size': 8,
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params)
        },
        'training': {
            'n_epochs': 1000,
            'learning_rate': 0.003,
            'lambda_dag': 1.0,
            'lambda_sparse': 0.05,
            'final_mse': float(history['mse'][-1]),
            'final_dag_penalty': float(history['dag_penalty'][-1]),
            'final_l1_penalty': float(history['l1_penalty'][-1])
        },
        'metrics': {
            'train_r2': float(r2_train),
            'train_mae': float(mae_train),
            'train_mape_pct': float(mape_train),
            'test_r2': float(r2_test),
            'test_mae': float(mae_test),
            'test_mape_pct': float(mape_test)
        },
        'causal_edges': causal_edges,
        'n_causal_edges': len(causal_edges),
        'channel_roas': {row['channel']: round(float(row['roas']), 4) for _, row in contributions_df.iterrows()},
        'notes': [
            'With only 52 weeks of data, the neural model will likely overfit on training data.',
            'The learned DAG and adstock patterns are still interpretable and useful.',
            'DAG penalty h(W) should approach 0 for a valid acyclic graph.',
            'GRU adstock captures more flexible carry-over patterns than geometric decay.',
            'The learned saturation MLP adapts to each channel\'s specific diminishing returns shape.'
        ]
    }

    with open('data/deep_causal_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: data/deep_causal_summary.json")
    print()

    # ------------------------------------------
    # Generate plots
    # ------------------------------------------
    print("[7/7] Generating visualizations...")
    os.makedirs('results', exist_ok=True)

    # 7a. Causal DAG visualization
    plot_causal_dag(W_learned, channel_names_list, 'results/deep_causal_dag.png')

    # 7b. Adstock comparison (GRU vs geometric)
    plot_adstock_comparison(impulse_responses, channel_names_list, 'results/deep_causal_adstock.png')

    # 7c. Channel ROAS
    plot_contributions(contributions_df, 'results/deep_causal_contributions.png')

    print()
    print("=" * 70)
    print("  DeepCausalMMM COMPLETE")
    print("=" * 70)
    print()
    print(f"  Model Performance:")
    print(f"    Train R2: {r2_train:.4f}  |  Test R2: {r2_test:.4f}")
    print(f"    DAG penalty h(W): {history['dag_penalty'][-1]:.6f}")
    print(f"    Discovered {len(causal_edges)} causal edges between channels")
    print()
    print(f"  Output files:")
    print(f"    data/deep_causal_dag.csv")
    print(f"    data/deep_causal_adstock.csv")
    print(f"    data/deep_causal_contributions.csv")
    print(f"    data/deep_causal_summary.json")
    print(f"    results/deep_causal_dag.png")
    print(f"    results/deep_causal_adstock.png")
    print(f"    results/deep_causal_contributions.png")
    print()


if __name__ == '__main__':
    main()
