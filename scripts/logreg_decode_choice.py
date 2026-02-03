#%% Logistic regression decoding of choices from firing rates 
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate

# Whether to balance classes by downsampling the larger class
balance_by_downsample = False
class_weight = None if balance_by_downsample else "balanced"

# Load firing rates and choices from Excel
df = pd.read_excel(r'data/firing_rates.xlsx', header=[0,1])  # MultiIndex columns

# Drop rows with all NaN values (if any)
df = df.dropna(how="all").reset_index(drop=True)

# Convert choices to binary: left (-1) -> 0, right (+1) -> 1
y = (df["choice"].values == 1).astype(int)
y = y.ravel()  # Ensure y is 1d

# Meta columns to keep
meta_lvl0 = {"choice", "trial_id", 'uuid'} 
meta_cols = [c for c in df.columns if c[0] in meta_lvl0]

# Only keep neurons from specific brain areas (optional)
keep_acronyms = {"MOp5", "MOp6a", "MOp6b"} # motor areas
neuron_cols_keep = [c for c in df.columns if (c[0] not in meta_lvl0) and (c[1] in keep_acronyms)]

#df = df[meta_cols + neuron_cols_keep].copy()

# Use only the neurons from the specified brain areas
X = df[neuron_cols_keep].to_numpy(dtype=float)

print(f"Data shape: X={X.shape}, y={y.shape}"
      )
# Balance the classes in the dataset 
# X, y already built
if balance_by_downsample == True:
    rng = np.random.default_rng(seed=1)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n = min(len(idx0), len(idx1))
    keep = np.concatenate([
        rng.choice(idx0, n, replace=False),
        rng.choice(idx1, n, replace=False),
    ])

    X_bal, y_bal = X[keep], y[keep]
    X, y = X_bal, y_bal

# Pipeline: scale within each CV fold (prevents leakage) + logistic regression
model = Pipeline(
    steps=[
        ("normalise", StandardScaler(with_mean=True, with_std=True)),
        ("model", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=5000,
            class_weight=class_weight,  # set to 'balanced' if classes are skewed

        )),
    ]
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

scoring = {
    "acc": "accuracy",
}

cv_out = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
    n_jobs=-1, # use all available cores
)

# Summarize results
def mean_and_sd(arr):
    return float(np.mean(arr)), float(np.std(arr, ddof=1))
acc_m, acc_s = mean_and_sd(cv_out["test_acc"])
print("\n5-fold cross-validation performance (test):")
print(f"  Accuracy:          {acc_m:.3f} ± {acc_s:.3f}")

# Fold-by-fold accuracies:
print("\nFold accuracies:", np.round(cv_out["test_acc"], 3))

# Save accuracies to a excel file (under a column logistic regression motor cortex)
df_acc = pd.DataFrame({
    "logistic_regression_motor_cortex": cv_out["test_acc"]
})
df_acc.to_excel(r"results/choice_decoding_accuracies.xlsx", index=False)

