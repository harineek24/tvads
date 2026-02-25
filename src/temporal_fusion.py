"""
Temporal Fusion Transformer (TFT) — Revenue Forecasting
========================================================
State-of-the-art deep learning for time series forecasting with
interpretable multi-head attention.

WHY TFT matters:
- Outperforms LightGBM and ARIMA on temporal data (Google Research, 2021)
- Provides interpretable attention weights showing WHICH past weeks
  and WHICH features matter for each prediction
- Variable selection network automatically identifies important inputs
- Handles static (DMA), known future (holidays), and observed (spend) inputs

Architecture:
1. Variable Selection Network — learnable feature importance
2. LSTM Encoder-Decoder — sequential processing
3. Multi-Head Attention — which past timesteps matter
4. Gated Residual Network — non-linear transformations
5. Quantile outputs — built-in uncertainty via prediction intervals

Outputs:
- data/tft_predictions.csv
- data/tft_attention_weights.csv
- data/tft_feature_importance.csv
- models/tft_model.pt
- results/tft_forecast.png
- results/tft_attention_heatmap.png
- results/tft_variable_importance.png
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

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


# ============================================
# SIMPLIFIED TFT ARCHITECTURE
# ============================================

class GatedResidualNetwork(nn.Module):
    """GRN: Core building block of TFT."""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        # Skip connection
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        h = self.elu(self.fc1(x))
        h = self.dropout(h)
        output = self.fc2(h)
        gate = self.sigmoid(self.gate(h))
        return self.layer_norm(gate * output + (1 - gate) * residual)


class VariableSelectionNetwork(nn.Module):
    """VSN: Learns which input variables matter."""
    def __init__(self, n_vars, input_dim, hidden_dim):
        super().__init__()
        self.n_vars = n_vars
        self.grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim)
            for _ in range(n_vars)
        ])
        self.softmax_layer = nn.Linear(n_vars * hidden_dim, n_vars)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # inputs: (batch, n_vars, input_dim)
        processed = []
        for i in range(self.n_vars):
            processed.append(self.grns[i](inputs[:, i, :]))

        stacked = torch.stack(processed, dim=1)  # (batch, n_vars, hidden)
        flat = stacked.reshape(stacked.size(0), -1)
        weights = self.softmax(self.softmax_layer(flat))  # (batch, n_vars)

        # Weighted sum
        weighted = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return weighted, weights


class TemporalFusionTransformer(nn.Module):
    """
    Simplified TFT for weekly revenue forecasting.
    Includes: VSN, LSTM, Multi-Head Attention, Quantile outputs.
    """
    def __init__(self, n_features, hidden_dim=32, n_heads=4,
                 n_quantiles=3, seq_len=8, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.n_quantiles = n_quantiles

        # Variable Selection
        self.vsn = VariableSelectionNetwork(n_features, 1, hidden_dim)

        # Temporal processing (LSTM encoder)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True,
                            num_layers=2, dropout=dropout)

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads,
                                                dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Output GRN + quantile heads
        self.output_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_quantiles)
        ])

    def forward(self, x):
        """
        x: (batch, seq_len, n_features)
        Returns: quantile predictions (batch, n_quantiles)
        """
        batch_size = x.size(0)

        # Variable Selection per timestep
        vsn_outputs = []
        vsn_weights_all = []
        for t in range(self.seq_len):
            step_input = x[:, t, :].unsqueeze(-1)  # (batch, n_features, 1)
            vsn_out, vsn_w = self.vsn(step_input)
            vsn_outputs.append(vsn_out)
            vsn_weights_all.append(vsn_w)

        vsn_sequence = torch.stack(vsn_outputs, dim=1)  # (batch, seq_len, hidden)

        # LSTM
        lstm_out, _ = self.lstm(vsn_sequence)  # (batch, seq_len, hidden)

        # Multi-Head Attention
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out,
            need_weights=True
        )
        attn_out = self.attn_norm(attn_out + lstm_out)  # residual

        # Use last timestep for prediction
        final = attn_out[:, -1, :]
        final = self.output_grn(final)

        # Quantile predictions
        quantiles = [head(final) for head in self.quantile_heads]
        quantiles = torch.cat(quantiles, dim=-1)

        # Average VSN weights across timesteps
        avg_vsn_weights = torch.stack(vsn_weights_all, dim=1).mean(dim=1)

        return quantiles, avg_vsn_weights, attn_weights


class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression."""
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            error = target - preds[:, i]
            losses.append(torch.max(q * error, (q - 1) * error))
        return torch.stack(losses).mean()


def prepare_sequences(weekly, seq_len=8):
    """Create supervised sequences from weekly data."""
    features = []
    for col in SPEND_COLUMNS:
        features.append(weekly[col].values)

    # Add seasonality features
    features.append(np.sin(2 * np.pi * weekly['month'].values / 12))
    features.append(np.cos(2 * np.pi * weekly['month'].values / 12))
    features.append(np.arange(len(weekly)) / len(weekly))  # trend

    feature_names = list(CHANNEL_NAMES.values()) + ['Month Sin', 'Month Cos', 'Trend']

    X = np.column_stack(features)
    y = weekly['total_revenue'].values

    # Normalize
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Create sequences
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i:i + seq_len])
        y_seq.append(y_scaled[i + seq_len])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    return X_seq, y_seq, scaler_x, scaler_y, feature_names


def train_tft(weekly, seq_len=8, epochs=300, lr=0.001):
    """Train the TFT model."""
    print("\n=== Training Temporal Fusion Transformer ===")

    X_seq, y_seq, scaler_x, scaler_y, feature_names = prepare_sequences(weekly, seq_len)
    n_features = X_seq.shape[2]

    # Train/test split (last 8 weeks for validation)
    split = len(X_seq) - 8
    X_train = torch.FloatTensor(X_seq[:split])
    y_train = torch.FloatTensor(y_seq[:split])
    X_test = torch.FloatTensor(X_seq[split:])
    y_test = torch.FloatTensor(y_seq[split:])

    print(f"  Features: {n_features}, Seq length: {seq_len}")
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Model
    model = TemporalFusionTransformer(
        n_features=n_features,
        hidden_dim=32,
        n_heads=4,
        n_quantiles=3,
        seq_len=seq_len,
        dropout=0.1
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)
    loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds, _, _ = model(X_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss.item())

        train_losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= 50:
            print(f"  Early stopping at epoch {epoch}")
            break

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.6f}")

    model.load_state_dict(best_state)
    print(f"  Best training loss: {best_loss:.6f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_preds, vsn_weights, attn_weights = model(X_test)
        train_preds, train_vsn, _ = model(X_train)

    # Unscale predictions
    test_median = scaler_y.inverse_transform(
        test_preds[:, 1].numpy().reshape(-1, 1)).flatten()
    test_lower = scaler_y.inverse_transform(
        test_preds[:, 0].numpy().reshape(-1, 1)).flatten()
    test_upper = scaler_y.inverse_transform(
        test_preds[:, 2].numpy().reshape(-1, 1)).flatten()
    test_actual = scaler_y.inverse_transform(
        y_test.numpy().reshape(-1, 1)).flatten()

    train_median = scaler_y.inverse_transform(
        train_preds[:, 1].numpy().reshape(-1, 1)).flatten()
    train_actual = scaler_y.inverse_transform(
        y_train.numpy().reshape(-1, 1)).flatten()

    # Metrics
    test_r2 = r2_score(test_actual, test_median)
    test_mae = mean_absolute_error(test_actual, test_median)
    train_r2 = r2_score(train_actual, train_median)

    print(f"\n  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  Test MAE: ${test_mae:,.0f}")

    # Coverage of prediction intervals
    coverage = np.mean((test_actual >= test_lower) & (test_actual <= test_upper))
    print(f"  80% PI Coverage: {coverage:.0%}")

    return (model, scaler_x, scaler_y, feature_names,
            test_actual, test_median, test_lower, test_upper,
            train_actual, train_median,
            vsn_weights, attn_weights, train_losses,
            {'test_r2': test_r2, 'test_mae': test_mae, 'train_r2': train_r2,
             'coverage': coverage, 'seq_len': seq_len})


def extract_interpretations(vsn_weights, attn_weights, feature_names):
    """Extract variable importance and attention patterns."""
    # Variable importance from VSN
    vi = vsn_weights.numpy().mean(axis=0)
    var_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': vi,
    }).sort_values('importance', ascending=False)

    print("\n=== TFT Variable Importance (VSN Weights) ===")
    for _, row in var_importance.iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"  {row['feature']:20s} {row['importance']:.3f} {bar}")

    # Attention weights (which past weeks matter)
    attn = attn_weights.numpy().mean(axis=(0, 1))  # average across batch and heads
    attn_df = pd.DataFrame({
        'past_week': [f'w-{i}' for i in range(len(attn), 0, -1)],
        'attention_weight': attn,
    })

    return var_importance, attn_df


def plot_results(test_actual, test_median, test_lower, test_upper,
                 train_actual, train_median, var_importance, attn_df,
                 train_losses, metrics):
    """Generate TFT visualization plots."""
    os.makedirs('results', exist_ok=True)

    # 1. Forecast with prediction intervals
    fig, ax = plt.subplots(figsize=(12, 5))
    n_train = len(train_actual)
    n_test = len(test_actual)
    weeks_train = np.arange(1, n_train + 1)
    weeks_test = np.arange(n_train + 1, n_train + n_test + 1)

    ax.plot(weeks_train, train_actual / 1e6, 'o-', color='#2c3e50',
            markersize=3, linewidth=1, label='Actual (Train)', alpha=0.6)
    ax.plot(weeks_train, train_median / 1e6, '-', color='#3498db',
            linewidth=1, label='TFT Prediction (Train)', alpha=0.6)
    ax.plot(weeks_test, test_actual / 1e6, 'o-', color='#2c3e50',
            markersize=5, linewidth=2, label='Actual (Test)')
    ax.plot(weeks_test, test_median / 1e6, 's-', color='#e74c3c',
            markersize=5, linewidth=2, label='TFT Prediction (Test)')
    ax.fill_between(weeks_test, test_lower / 1e6, test_upper / 1e6,
                    alpha=0.2, color='#e74c3c', label='80% Prediction Interval')
    ax.axvline(x=n_train + 0.5, color='gray', linestyle='--', alpha=0.5,
               label='Train/Test Split')
    ax.set_xlabel('Week')
    ax.set_ylabel('Revenue ($M)')
    ax.set_title(f'Temporal Fusion Transformer: Revenue Forecasting\n'
                 f'Test R² = {metrics["test_r2"]:.3f}, '
                 f'80% PI Coverage = {metrics["coverage"]:.0%}',
                 fontsize=13)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('results/tft_forecast.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Variable importance from VSN
    fig, ax = plt.subplots(figsize=(10, 5))
    vi_sorted = var_importance.sort_values('importance', ascending=True)
    colors = ['#8e44ad' if 'TV' in f else '#2980b9' if f in ['Paid Search', 'Social', 'Display']
              else '#27ae60' for f in vi_sorted['feature']]
    ax.barh(vi_sorted['feature'], vi_sorted['importance'], color=colors, alpha=0.8)
    ax.set_xlabel('VSN Selection Weight', fontsize=12)
    ax.set_title('TFT Variable Selection Network: Learned Feature Importance\n'
                 '(Higher = model relies more on this feature)', fontsize=13)
    plt.tight_layout()
    plt.savefig('results/tft_variable_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Attention heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Attention over past weeks
    ax = axes[0]
    attn_vals = attn_df['attention_weight'].values
    ax.bar(range(len(attn_vals)), attn_vals, color='#e67e22', alpha=0.8)
    ax.set_xticks(range(len(attn_vals)))
    ax.set_xticklabels(attn_df['past_week'], fontsize=8)
    ax.set_xlabel('Past Week')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Multi-Head Attention: Which Past Weeks Matter?')

    # Training loss curve
    ax = axes[1]
    ax.plot(train_losses, color='#2c3e50', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Quantile Loss')
    ax.set_title('Training Convergence')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('results/tft_attention_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Plots saved to results/tft_*.png")


def main():
    print("=" * 60)
    print("TEMPORAL FUSION TRANSFORMER — Revenue Forecasting")
    print("=" * 60)

    weekly = pd.read_csv('data/weekly_spend.csv')

    # Train TFT
    (model, scaler_x, scaler_y, feature_names,
     test_actual, test_median, test_lower, test_upper,
     train_actual, train_median,
     vsn_weights, attn_weights, train_losses, metrics) = train_tft(weekly)

    # Extract interpretations
    var_importance, attn_df = extract_interpretations(
        vsn_weights, attn_weights, feature_names)

    # Save results
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    var_importance.to_csv('data/tft_feature_importance.csv', index=False)
    attn_df.to_csv('data/tft_attention_weights.csv', index=False)

    predictions = pd.DataFrame({
        'week': range(len(train_actual) + 1, len(train_actual) + len(test_actual) + 1),
        'actual': test_actual,
        'predicted_median': test_median,
        'predicted_lower': test_lower,
        'predicted_upper': test_upper,
    })
    predictions.to_csv('data/tft_predictions.csv', index=False)

    with open('models/tft_metrics.json', 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    torch.save(model.state_dict(), 'models/tft_model.pt')

    # Plot
    plot_results(test_actual, test_median, test_lower, test_upper,
                 train_actual, train_median, var_importance, attn_df,
                 train_losses, metrics)

    print("\n✓ TFT analysis complete")
    return model, var_importance, attn_df, predictions, metrics


if __name__ == '__main__':
    main()
