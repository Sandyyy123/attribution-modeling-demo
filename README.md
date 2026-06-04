# Attribution Modeling - Shapley Value Framework

Game-theoretic contribution modeling that fairly distributes credit for an outcome
across N contributing factors. Works for marketing attribution, feature importance,
sales attribution, and any multi-factor contribution problem.

## Core Approach

Uses **Shapley values** from cooperative game theory: each contributor's credit is
the average marginal contribution across all possible orderings of contributors.
This is the only attribution method satisfying efficiency, symmetry, and fairness axioms simultaneously.

## Modules

| File | Description |
|------|-------------|
| `attribution.py` | Shapley engine, characteristic functions, preprocessing helpers |
| `main.py` | End-to-end demo with synthetic marketing data |

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Key Functions

```python
from attribution import shapley_values, normalize_attributions, clean_dataset

# Define your characteristic function v(S) -> outcome_value
def my_fn(coalition):
    ...

# Compute Shapley values
raw = shapley_values(players=["Email", "Search", "Social"], characteristic_fn=my_fn)
pct = normalize_attributions(raw)   # -> {"Email": 45.2, "Search": 31.1, "Social": 23.7}

# Clean your dataset first
df_clean = clean_dataset(df_raw, outlier_method="iqr", impute_strategy="median")
```

## Preprocessing

- **Outliers**: IQR fencing (default) or Z-score capping; threshold configurable
- **Missing values**: Column-level missingness threshold drop + KNN / median / mean imputation
- **Correlation**: Removes highly correlated features to avoid attribution dilution

---

*Dr. Sandeep Grover - PhD Data Science | Algorithm Design | Statistical Modeling*
