"""
main.py - End-to-end attribution modeling demo.
Run: python main.py
"""

import numpy as np
import pandas as pd
from attribution import (shapley_values, normalize_attributions,
                          linear_model_fn, clean_dataset)


def generate_synthetic_data(n: int = 1000, seed: int = 42) -> tuple:
    """Synthetic marketing touchpoint dataset with known ground-truth attribution."""
    rng = np.random.default_rng(seed)
    channels = ["Email", "Paid_Search", "Social", "Display", "Direct"]

    # True weights (ground truth we want to recover)
    true_weights = {"Email": 0.35, "Paid_Search": 0.25, "Social": 0.20,
                    "Display": 0.10, "Direct": 0.10}

    X = rng.standard_normal((n, len(channels)))
    X[:, 1] += 0.8 * X[:, 0]   # Paid_Search correlated with Email
    X[:, 4] += 0.3 * X[:, 2]   # Direct correlated with Social

    # Add some missing values and outliers for demo
    miss_mask = rng.random((n, len(channels))) < 0.08
    X[miss_mask] = np.nan
    outlier_idx = rng.choice(n, size=int(n * 0.03), replace=False)
    X[outlier_idx, 0] = rng.uniform(15, 25, size=len(outlier_idx))

    w = np.array(list(true_weights.values()))
    X_clean = np.nan_to_num(X, nan=0.0)
    y = X_clean @ w + rng.normal(0, 0.5, n)

    df = pd.DataFrame(X, columns=channels)
    return df, y, channels, true_weights


def main():
    print("=== Attribution Modeling Demo ===\n")

    # 1. Generate data
    df_raw, y, channels, true_weights = generate_synthetic_data()
    print(f"Raw data: {df_raw.shape} | Missing: {df_raw.isnull().mean().mean():.1%}\n")

    # 2. Clean dataset
    print("[1/3] Cleaning dataset...")
    df_clean = clean_dataset(df_raw, outlier_method="iqr", impute_strategy="median",
                              drop_missing_threshold=0.5, drop_correlated_threshold=0.92)
    X_clean = df_clean.values
    remaining_channels = list(df_clean.columns)

    # 3. Shapley attribution
    print("\n[2/3] Computing Shapley attribution values...")
    char_fn = linear_model_fn(X_clean, y)
    raw_shapley = shapley_values(remaining_channels, char_fn, n_samples=256)
    attribution_pct = normalize_attributions(raw_shapley)

    # 4. Results
    print("\n[3/3] Attribution results:")
    print("-" * 45)
    print(f"{'Channel':<18} {'Shapley %':>12} {'True %':>10}")
    print("-" * 45)
    for ch in sorted(attribution_pct, key=attribution_pct.get, reverse=True):
        true = true_weights.get(ch, 0.0) * 100
        print(f"{ch:<18} {attribution_pct[ch]:>11.1f}% {true:>9.1f}%")
    print("-" * 45)
    print("\nDone. Extend with your own characteristic_fn for domain-specific attribution.")


if __name__ == "__main__":
    main()
