# SOA AI Challenge 2026

## Project Overview

Mortality prediction modeling using neural networks for the SOA (Society of Actuaries) Individual Life Experience Committee (ILEC) dataset. The goal is to predict `death_claim_amount` weighted by `amount_exposed`.

## Key Files

```
files/
├── dataset/
│   └── mortality_grouped.parquet    # Main modeling dataset (~944K rows)
├── models/
│   └── neural.joblib                # Pre-trained neural network model
├── partition/                        # Raw partitioned data
notebooks/
└── soa_neural.ipynb                  # Main neural network workflow
```

## Loading Pre-trained Models

Models are saved in `files/models/` and should be loaded for review (not retrained):

```python
import joblib

# Load the pre-trained neural network model
model = joblib.load("files/models/neural.joblib")

# Generate predictions
predictions = model.predict(X)
```

**Important:** Do not retrain models unless explicitly requested. Use the saved models for analysis and review.

## Data Schema

**Target:** `death_claim_amount`
**Weight:** `amount_exposed`
**Benchmark:** `qx_vbt15` (VBT 2015 mortality rates)

### Key Features
| Feature | Type | Notes |
|---------|------|-------|
| `attained_age` | int | 50-95, use splines |
| `duration` | int | Policy year, select effect in early durations |
| `sex` | category | M/F |
| `smoker_status` | category | NS/S |
| `insurance_plan` | category | Term, Perm, UL, ULSG, VL, VLSG |
| `face_amount_band` | category | 11 bands, use embeddings |
| `preferred_class` | category | Underwriting tier |
| `class_enh` | category | Combined: number_of_pfd_classes + preferred_class |

### Derived Features
- `qx_raw` = death_claim_amount / amount_exposed
- `capped_duration` = duration clipped at 26
- `binned_face` = grouped face_amount_band (5 bins)

## Libraries

- **morai**: Custom actuarial library for preprocessing, charting, neural networks
  - `morai.forecast.preprocessors` - data preprocessing
  - `morai.models.neural` - Neural network with embeddings
  - `morai.experience.charters` - PDP, rate comparison, A/E charts
  - `morai.experience.tables` - VBT rate mapping
- **polars**: Fast data loading and manipulation
- **pandas**: Data analysis
- **torch**: Neural network backend

## Coding Conventions

- Column names: lowercase with underscores
- Use `category` dtype for categorical columns
- Always use `amount_exposed` as weight
- A/E = Actual / Expected (target ratio is 1.00)
- Seed: 42 for reproducibility

## Model Preferences

### Preprocessing
```python
feature_dict = {
    "target": ["qx_raw"],
    "weight": ["amount_exposed"],
    "passthrough": ["duration", "face_amount_band"],
    "ordinal": ["sex", "smoker_status", "observation_year"],
    "ohe": ["class_enh", "insurance_plan"],
    "spline": {"attained_age": {"n_knots": 8, "degree": 3, "knots": "quantile"}},
}
```

### Neural Network
- Architecture: Wide + Deep
- Loss: Poisson (for mortality rates)
- Embeddings: face_amount_band
- Early stopping on validation loss

### Analysis Workflow
1. A/E ratios by dimension (train vs test)
2. Rate comparison charts (qx_raw vs qx_vbt15 vs qx_model)
3. PDP plots with `weight="amount_exposed"`, `secondary="death_count"`
4. SHAP analysis (100 background samples)
5. Embedding similarity heatmaps

## Known Issues

- Early durations (1-5) often have low A/E due to select mortality
- Young ages (41-50) may have sparse data
- VBT15 benchmark may not match modern experience

## Commands

```bash
# Load partitioned data (polars)
pl.scan_parquet("files/partition/*/*.parquet", hive_partitioning=True)
```
