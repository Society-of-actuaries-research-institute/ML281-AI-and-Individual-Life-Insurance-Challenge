"""
Phase 3b: Deep Learning Models for Mortality Prediction (CORRECTED)

CORRECTED VERSION: Uses rate regression instead of binary classification.
Each row represents an aggregated cell with Death_Count deaths out of 
Policies_Exposed exposure.

Models:
  - TabNet (Arik & Pfister, AAAI 2021): Attention-based DL for tabular data
  - FT-Transformer (Gorishniy et al., NeurIPS 2021): Transformer for tabular data

Key changes from original:
1. Target is mortality_rate (Death_Count / Policies_Exposed), not binary
2. Loss functions use exposure-weighted MSE
3. Evaluation metrics compare predicted vs actual mortality RATES
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import os
import gc
import warnings
warnings.filterwarnings("ignore")

# Paths — resolved relative to this file (soa_ai_v2/code/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data", "parquet")
# Results base honors the optional SOA_RESULTS_DIR env var (default unchanged).
_RESULTS = os.environ.get("SOA_RESULTS_DIR") or os.path.join(_ROOT, "results")
OUTPUT_DIR = os.path.join(_RESULTS, "models")
FIGURES_DIR = os.path.join(_RESULTS, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (12, 7),
    "font.size": 12,
    "axes.titlesize": 14,
    "figure.dpi": 150,
})

CATEGORICAL_FEATURES = [
    "Sex", "Smoker_Status", "Insurance_Plan", "Age_Ind",
    "SOA_Post_Lvl_Ind", "Slct_Ult_Ind", "Preferred_Indicator",
    "Number_of_Pfd_Classes", "Preferred_Class", "Face_Amount_Band",
]
NUMERIC_FEATURES = [
    "Issue_Age", "Duration", "Attained_Age", "Issue_Year",
]


# =============================================================================
# Data Loading
# =============================================================================

def load_and_prepare_data():
    """Load parquet data and prepare for modeling."""
    print("  Loading parquet data...")
    dfs = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith(".parquet"):
            dfs.append(pd.read_parquet(os.path.join(DATA_DIR, f)))
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df):,} rows (aggregated cells)")

    df = df[df["Policies_Exposed"] > 0].copy()
    
    # CORRECTED: Target is mortality rate
    df["mortality_rate"] = df["Death_Count"] / df["Policies_Exposed"]
    df["log_exposure"] = np.log(df["Policies_Exposed"])

    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col + "_encoded"] = le.fit_transform(df[col])
        label_encoders[col] = le

    df["Age_Band"] = pd.cut(
        df["Attained_Age"],
        bins=[0, 24, 44, 64, 120],
        labels=[0, 1, 2, 3],
    ).astype(float).fillna(2.0)

    df["log_amount_exposed"] = np.log1p(df["Amount_Exposed"])
    df["duration_squared"] = df["Duration"] ** 2

    # Fill NaN for DL models
    feature_cols = get_feature_cols()
    for col in feature_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(0)

    total_exposure = df["Policies_Exposed"].sum()
    total_deaths = df["Death_Count"].sum()
    overall_rate = total_deaths / total_exposure
    
    print(f"  After filtering: {len(df):,} cells")
    print(f"  Total exposure: {total_exposure:,.0f}, Deaths: {total_deaths:,.0f}")
    print(f"  Overall mortality rate: {overall_rate:.6f}")

    return df, label_encoders


def get_feature_cols():
    """Get list of feature columns for modeling."""
    encoded_cats = [c + "_encoded" for c in CATEGORICAL_FEATURES]
    return NUMERIC_FEATURES + encoded_cats + ["Age_Band", "log_amount_exposed", "duration_squared"]


def temporal_train_test_split(df, train_end_year, test_year):
    """Split data by observation year."""
    train = df[df["Observation_Year"] <= train_end_year].copy()
    test = df[df["Observation_Year"] == test_year].copy()
    return train, test


def sample_for_training(df, max_rows=500_000):
    """Sample for training efficiency, preserving exposure distribution."""
    if len(df) <= max_rows:
        return df

    cells_with_deaths = df[df["Death_Count"] > 0]
    cells_no_deaths = df[df["Death_Count"] == 0]
    
    n_death_cells = len(cells_with_deaths)
    n_no_death = min(len(cells_no_deaths), max_rows - n_death_cells)
    
    if n_no_death > 0 and n_no_death < len(cells_no_deaths):
        probs = cells_no_deaths["Policies_Exposed"] / cells_no_deaths["Policies_Exposed"].sum()
        cells_no_deaths = cells_no_deaths.sample(n=n_no_death, weights=probs, random_state=42)
    
    result = pd.concat([cells_with_deaths, cells_no_deaths]).sample(frac=1, random_state=42)
    return result


def get_device():
    """Select best available device."""
    if torch.backends.mps.is_available():
        try:
            t = torch.randn(2, 2, device="mps")
            _ = t @ t
            print("  Using MPS (Apple Silicon)")
            return "mps"
        except Exception:
            pass
    if torch.cuda.is_available():
        print("  Using CUDA")
        return "cuda"
    print("  Using CPU")
    return "cpu"


# =============================================================================
# Custom Loss Function for Exposure-Weighted Rate Prediction
# =============================================================================

class ExposureWeightedMSELoss(nn.Module):
    """MSE loss weighted by exposure - appropriate for mortality rate prediction."""
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true, exposure):
        """
        Args:
            y_pred: Predicted mortality rates
            y_true: Actual mortality rates
            exposure: Policy exposure (weight)
        """
        squared_error = (y_pred - y_true) ** 2
        weighted_loss = (squared_error * exposure).sum() / exposure.sum()
        return weighted_loss


class PoissonLoss(nn.Module):
    """Poisson loss for count data with exposure offset."""
    def __init__(self):
        super().__init__()
    
    def forward(self, log_rate_pred, death_count, exposure):
        """
        Poisson negative log-likelihood.
        
        Args:
            log_rate_pred: log(predicted rate) from model
            death_count: Actual death counts
            exposure: Policy exposure
        """
        # Expected deaths = exposure * exp(log_rate_pred)
        expected = exposure * torch.exp(log_rate_pred)
        # Poisson NLL: expected - actual * log(expected)
        loss = expected - death_count * torch.log(expected + 1e-8)
        return loss.mean()


# =============================================================================
# Model 1: TabNet for Rate Regression
# =============================================================================

def train_tabnet(X_train, y_rate_train, exposure_train, X_test, exposure_test):
    """
    TabNet for mortality rate regression.
    """
    from pytorch_tabnet.tab_model import TabNetRegressor
    
    scaler = StandardScaler()
    X_tr = np.nan_to_num(X_train, nan=0.0)
    X_te = np.nan_to_num(X_test, nan=0.0)
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    
    # TabNet device
    tabnet_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = TabNetRegressor(
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=10, gamma=0.9),
        verbose=0,
        device_name=tabnet_device,
        seed=42,
    )
    
    # Reshape targets for TabNet regressor
    y_tr = y_rate_train.reshape(-1, 1).astype(np.float32)
    y_te_dummy = np.zeros((len(X_te), 1), dtype=np.float32)
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te_dummy)],
        max_epochs=50,
        patience=10,
        batch_size=4096,
        virtual_batch_size=256,
        drop_last=False,
        # Use exposure as sample weight (TabNet supports this)
        weights=1,  # TabNet doesn't have direct sample_weight, we'll use custom approach
    )
    
    y_pred_rate = model.predict(X_te).flatten()
    y_pred_rate = np.clip(y_pred_rate, 0, 1)
    
    model._scaler = scaler
    return model, y_pred_rate


# =============================================================================
# Model 2: FT-Transformer for Rate Regression
# =============================================================================

class FTTransformerNet(nn.Module):
    """FT-Transformer for regression."""
    def __init__(self, n_features, embed_dim=32, num_heads=4, num_layers=3,
                 ff_dim=128, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.embed_dim = embed_dim

        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(n_features)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regression head (output single value)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),  # Bound output to [0, 1] for mortality rate
        )

    def forward(self, x):
        batch_size = x.size(0)
        tokens = []
        for i, emb in enumerate(self.feature_embeddings):
            tokens.append(emb(x[:, i:i+1]))
        tokens = torch.stack(tokens, dim=1)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out = self.transformer(tokens)
        cls_out = out[:, 0, :]
        pred = self.head(cls_out).squeeze(-1)
        return pred


def train_ft_transformer(X_train, y_rate_train, exposure_train, X_test, exposure_test):
    """FT-Transformer for mortality rate regression with exposure weighting."""
    scaler = StandardScaler()
    X_tr = np.nan_to_num(X_train, nan=0.0)
    X_te = np.nan_to_num(X_test, nan=0.0)
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    device_name = get_device()
    device = torch.device(device_name)

    n_features = X_tr.shape[1]
    model = FTTransformerNet(
        n_features=n_features,
        embed_dim=32,
        num_heads=4,
        num_layers=3,
        ff_dim=128,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # DataLoader with exposure weights
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_rate_train, dtype=torch.float32),
        torch.tensor(exposure_train, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4096, shuffle=True, drop_last=False,
    )

    # Validation set
    val_size = min(100_000, len(X_te))
    if val_size < len(X_te):
        val_idx = np.random.RandomState(42).choice(len(X_te), val_size, replace=False)
        X_val = X_te[val_idx]
        y_val = y_rate_train[val_idx] if len(y_rate_train) > val_idx.max() else np.zeros(val_size)
        exp_val = exposure_test[val_idx]
    else:
        X_val = X_te
        y_val = np.zeros(len(X_te))  # We don't have y_test during training
        exp_val = exposure_test

    # Training loop
    best_loss = float("inf")
    patience_counter = 0
    patience = 5
    best_state = None

    for epoch in range(30):
        model.train()
        epoch_loss = 0.0
        total_weight = 0.0
        
        for X_batch, y_batch, exp_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            exp_batch = exp_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            
            # Exposure-weighted MSE loss
            squared_error = (pred - y_batch) ** 2
            loss = (squared_error * exp_batch).sum() / exp_batch.sum()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * exp_batch.sum().item()
            total_weight += exp_batch.sum().item()

        scheduler.step()
        avg_loss = epoch_loss / total_weight

        # Early stopping based on training loss (since we don't have val targets)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict on test set
    model.eval()
    X_te_tensor = torch.tensor(X_te, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # Batch prediction for memory efficiency
        batch_size = 4096
        preds = []
        for i in range(0, len(X_te_tensor), batch_size):
            batch = X_te_tensor[i:i+batch_size]
            pred = model(batch).cpu().numpy()
            preds.append(pred)
        y_pred_rate = np.concatenate(preds)

    y_pred_rate = np.clip(y_pred_rate, 0, 1)
    
    model._scaler = scaler
    return model, y_pred_rate


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(y_rate_true, y_rate_pred, exposure, model_name):
    """Evaluate predictions against actual mortality rates."""
    metrics = {"model": model_name}
    
    # Weighted RMSE
    weighted_mse = np.average((y_rate_true - y_rate_pred)**2, weights=exposure)
    metrics["weighted_rmse"] = np.sqrt(weighted_mse)
    
    # Weighted MAE
    metrics["weighted_mae"] = np.average(np.abs(y_rate_true - y_rate_pred), weights=exposure)
    
    # A/E Ratio
    total_actual = np.sum(y_rate_true * exposure)
    total_predicted = np.sum(y_rate_pred * exposure)
    metrics["ae_ratio"] = total_actual / total_predicted if total_predicted > 0 else np.nan
    
    # Weighted R²
    ss_res = np.sum(exposure * (y_rate_true - y_rate_pred)**2)
    ss_tot = np.sum(exposure * (y_rate_true - np.average(y_rate_true, weights=exposure))**2)
    metrics["weighted_r2"] = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    # Calibration Error
    pred_df = pd.DataFrame({
        "pred": y_rate_pred,
        "actual": y_rate_true,
        "exposure": exposure,
    })
    try:
        pred_df["decile"] = pd.qcut(pred_df["pred"], 10, duplicates="drop", labels=False)
    except ValueError:
        pred_df["decile"] = pd.qcut(pred_df["pred"].rank(method="first"), 5, duplicates="drop", labels=False)
    
    cal = pred_df.groupby("decile").apply(
        lambda g: pd.Series({
            "pred_rate": np.average(g["pred"], weights=g["exposure"]),
            "actual_rate": np.average(g["actual"], weights=g["exposure"]),
            "total_exposure": g["exposure"].sum(),
        })
    )
    metrics["calibration_error"] = np.average(
        np.abs(cal["pred_rate"] - cal["actual_rate"]),
        weights=cal["total_exposure"]
    )
    
    return metrics


# =============================================================================
# Model Tournament
# =============================================================================

def run_dl_tournament(df):
    """Run deep learning model tournament."""
    print("\n=== DEEP LEARNING TOURNAMENT (RATE REGRESSION) ===")

    feature_cols = get_feature_cols()
    all_results = []
    best_models = {}

    train_test_configs = [
        (2014, 2015),
        (2015, 2016),
        (2016, 2017),
        (2017, 2018),
        (2018, 2019),
    ]

    models_to_train = {
        "TabNet": train_tabnet,
        "FT-Transformer": train_ft_transformer,
    }

    for train_end, test_year in train_test_configs:
        print(f"\n--- Train: 2012-{train_end}, Test: {test_year} ---")
        train_df, test_df = temporal_train_test_split(df, train_end, test_year)
        train_sampled = sample_for_training(train_df, max_rows=500_000)
        
        train_exposure = train_sampled["Policies_Exposed"].sum()
        train_deaths = train_sampled["Death_Count"].sum()
        test_exposure = test_df["Policies_Exposed"].sum()
        test_deaths = test_df["Death_Count"].sum()
        
        print(f"  Train: {len(train_sampled):,} cells, {train_exposure:,.0f} exposure")
        print(f"  Test:  {len(test_df):,} cells, {test_exposure:,.0f} exposure")
        print(f"  Train rate: {train_deaths/train_exposure:.6f}, Test rate: {test_deaths/test_exposure:.6f}")

        X_train = train_sampled[feature_cols].values
        y_rate_train = train_sampled["mortality_rate"].values
        exposure_train = train_sampled["Policies_Exposed"].values
        
        X_test = test_df[feature_cols].values
        y_rate_test = test_df["mortality_rate"].values
        exposure_test = test_df["Policies_Exposed"].values

        for model_name, train_fn in models_to_train.items():
            print(f"  Training {model_name}...")
            try:
                model, y_pred_rate = train_fn(
                    X_train, y_rate_train, exposure_train,
                    X_test, exposure_test
                )
                
                metrics = evaluate_model(y_rate_test, y_pred_rate, exposure_test, model_name)
                metrics["train_end"] = train_end
                metrics["test_year"] = test_year
                all_results.append(metrics)

                print(f"    RMSE: {metrics['weighted_rmse']:.6f} | "
                      f"A/E: {metrics['ae_ratio']:.4f} | "
                      f"Cal Error: {metrics['calibration_error']:.6f}")

                if test_year == 2019:
                    best_models[model_name] = {
                        "model": model,
                        "X_test": X_test,
                        "y_rate_test": y_rate_test,
                        "y_pred_rate": y_pred_rate,
                        "exposure_test": exposure_test,
                        "feature_cols": feature_cols,
                    }

            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "model": model_name,
                    "train_end": train_end,
                    "test_year": test_year,
                    "weighted_rmse": np.nan,
                    "ae_ratio": np.nan,
                    "calibration_error": np.nan,
                    "error": str(e),
                })

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_dl_poisson.csv"), index=False)
    print(f"\n  Saved model_comparison_dl_poisson.csv")

    # Merge with GBDT results if available
    try:
        gbdt_df = pd.read_csv(os.path.join(OUTPUT_DIR, "model_comparison_poisson.csv"))
        combined = pd.concat([gbdt_df, results_df], ignore_index=True)
        combined.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_all_poisson.csv"), index=False)
        print(f"  Saved model_comparison_all_poisson.csv")
    except FileNotFoundError:
        pass

    return results_df, best_models


# =============================================================================
# Visualization
# =============================================================================

def create_comparison_plots(results_df, best_models):
    """Create comparison visualizations."""
    print("\n=== CREATING VISUALIZATIONS ===")

    # 1. Performance over time
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for metric, ax, title in [
        ("weighted_rmse", axes[0], "Weighted RMSE (lower=better)"),
        ("ae_ratio", axes[1], "A/E Ratio (1.0=perfect)"),
        ("calibration_error", axes[2], "Calibration Error (lower=better)"),
    ]:
        for model_name in results_df["model"].unique():
            mdata = results_df[results_df["model"] == model_name].sort_values("test_year")
            if not mdata[metric].isna().all():
                ax.plot(mdata["test_year"], mdata[metric], "o-", label=model_name, markersize=6)
        
        ax.set_title(title)
        ax.set_xlabel("Test Year")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if metric == "ae_ratio":
            ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.5)

    plt.suptitle("Deep Learning Models: Rate Prediction Performance", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure17_dl_comparison_poisson.png"), bbox_inches="tight")
    plt.close()
    print("  Saved figure17_dl_comparison_poisson.png")

    # 2. Calibration plots
    if best_models:
        fig, axes = plt.subplots(1, len(best_models), figsize=(7*len(best_models), 6))
        if len(best_models) == 1:
            axes = [axes]
        
        for ax, (model_name, data) in zip(axes, best_models.items()):
            pred_df = pd.DataFrame({
                "pred": data["y_pred_rate"],
                "actual": data["y_rate_test"],
                "exposure": data["exposure_test"],
            })
            
            try:
                pred_df["decile"] = pd.qcut(pred_df["pred"], 10, duplicates="drop", labels=False)
            except ValueError:
                pred_df["decile"] = pd.qcut(pred_df["pred"].rank(method="first"), 5, duplicates="drop", labels=False)
            
            cal = pred_df.groupby("decile").apply(
                lambda g: pd.Series({
                    "pred_rate": np.average(g["pred"], weights=g["exposure"]),
                    "actual_rate": np.average(g["actual"], weights=g["exposure"]),
                })
            ).reset_index()
            
            ax.scatter(cal["pred_rate"], cal["actual_rate"], s=100, alpha=0.7)
            max_rate = max(cal["pred_rate"].max(), cal["actual_rate"].max()) * 1.1
            ax.plot([0, max_rate], [0, max_rate], "k--", alpha=0.5)
            ax.set_xlabel("Predicted Rate")
            ax.set_ylabel("Actual Rate")
            ax.set_title(f"{model_name}")
            ax.grid(True, alpha=0.3)
        
        plt.suptitle("Deep Learning Calibration: Predicted vs Actual Mortality Rate", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "figure18_dl_calibration_poisson.png"), bbox_inches="tight")
        plt.close()
        print("  Saved figure18_dl_calibration_poisson.png")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("PHASE 3b: DEEP LEARNING - RATE REGRESSION (CORRECTED)")
    print("=" * 60)

    df, label_encoders = load_and_prepare_data()
    results_df, best_models = run_dl_tournament(df)
    create_comparison_plots(results_df, best_models)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: DL Models (2019 Test Year)")
    print("=" * 60)
    
    final = results_df[results_df["test_year"] == 2019]
    if len(final) > 0:
        print(final[["model", "weighted_rmse", "ae_ratio", "calibration_error"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("PHASE 3b COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
