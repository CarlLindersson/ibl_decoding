#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from matplotlib import pyplot as plt
from src.functions import leaf_descendants
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from iblatlas.atlas import AllenAtlas
from iblatlas.plots import plot_scalar_on_slice

# Load firing rates and choices from Excel
df = pd.read_excel(r'data/firing_rates.xlsx', header=[0,1])  # MultiIndex columns

# Drop rows with all NaN values (if any)
df = df.dropna(how="all").reset_index(drop=True)

# Meta columns to keep
meta_lvl0 = {"choice", "trial_id", 'uuid'} 

# Convert choices to binary: left (-1) -> 0, right (+1) -> 1
y = (df["choice"].values == 1).astype(int)
y = y.ravel()  # Ensure y is 1d

# Get present areas in the data
present = df.columns.get_level_values(1).unique().tolist()
print(present)
area_subdivisions = {
    "motor cortex": leaf_descendants(["MOp", "MOs"],
                                   only_present=present,
                                   pattern=r"^(MOp|MOs)(5|6a|6b)$"),
    "thalamus": leaf_descendants(["TH"], only_present=present),
  #  "hypothalamus": leaf_descendants(["HY"], only_present=present),
}
#area_subdivisions['hypothalamus'].append('ZI')  # add zona incerta


# Get lowest class count across all trials
min_class_count = np.bincount(y).min()

# Random number generator
rng = np.random.default_rng(seed=1)

# Balance classes once (same trials for every area)
idx0 = np.where(y == 0)[0]
idx1 = np.where(y == 1)[0]
keep_trials = np.concatenate([
    rng.choice(idx0, min_class_count, replace=False),
    rng.choice(idx1, min_class_count, replace=False),
])
y_bal = y[keep_trials]

# Neuron columns per area
area_neuron_cols = {}
for area_name, acronyms in area_subdivisions.items():
    cols = [c for c in df.columns if (c[0] not in meta_lvl0) and (c[1] in acronyms)]
    if len(cols) > 0:
        area_neuron_cols[area_name] = cols

# Balance neuron count across areas (min #neurons)
min_neurons = min(len(cols) for cols in area_neuron_cols.values())

# Run cross-validation for each area
area_accuracies = {}
for area_name, acronyms in area_subdivisions.items():
    print(f"\nDecoding from area: {area_name} ({acronyms})")

    cols = area_neuron_cols.get(area_name, [])
    if len(cols) == 0:
        print(f"  WARNING: No neurons found for area {area_name}, skipping")
        continue

    # subsample neurons for this area
    neuron_idx = rng.choice(len(cols), size=min_neurons, replace=False)
    cols_sub = [cols[i] for i in neuron_idx]   # list of tuples (good for MultiIndex)
    print("neurons used:", len(cols_sub))

    # Use the SAME balanced trials for every area
    X_area = df.loc[keep_trials, cols_sub].to_numpy(dtype=float)
    y_area = y_bal

    # Pipeline: scale within each CV fold (prevents leakage) + logistic regression
    model = Pipeline(
        steps=[
            ("normalise", StandardScaler(with_mean=True, with_std=True)),
            ("model", LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=5000,
            )),
        ]
    )

    # scoring
    scoring = { "acc": "accuracy"}

    # Cross-validate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv_out_area = cross_validate(
        model,
        X_area,
        y_area,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1, # use all available cores
    )

    # Summarize results
    def mean_and_sd(arr):
        return float(np.mean(arr)), float(np.std(arr, ddof=1))
    acc_m, acc_s = mean_and_sd(cv_out_area["test_acc"])
    area_accuracies[area_name] = (acc_m, acc_s)
    print(f"  Accuracy:          {acc_m:.3f} ± {acc_s:.3f}")
    print("  Fold accuracies:", np.round(cv_out_area["test_acc"], 3))

#%% plot area accuracies
area_names = list(area_accuracies.keys())
area_means = [area_accuracies[name][0] for name in area_names]
area_sds = [area_accuracies[name][1] for name in area_names]
fig, [ax, ax2] = plt.subplots(1,2, figsize=(15,6))
ax.bar(area_names, area_means, yerr=area_sds, capsize=5, 
       color=['tab:blue', 'tab:orange', 'tab:green'])
ax.set_ylabel("Decoding Accuracy")
ax.set_title("Choice Decoding Accuracy by Brain Area")
ax.set_ylim(0, 1)
ax.spines[['top', 'right']].set_visible(False)


# Vizualize brain areas in brain slice
ba = AllenAtlas()  # IBL/Allen atlas    
groups = {
    "motor cortex": ["MOp5", "MOp6a", "MOp6b"],
    "thalamus": ["AD", "AMd", "AMv", "AV", "IAD", "PR"],
    #"hypothalamus": ["ZI"],
}
group_code = {"motor cortex": 1, "thalamus": 2, "hypothalamus": 3}

# Flatten into regions + values
regions = np.array([r for g in groups for r in groups[g]], dtype=object)
values  = np.array([group_code[g] for g in groups for _ in groups[g]], dtype=float)

# Discrete 3-color map (one color per group)
cmap = ListedColormap(["tab:blue", "tab:orange", "tab:green"])
norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)

# One plot: sagittal slice (coord is ML in microns)
# If you don’t see all regions, tweak coord (e.g., 500, 1000, 1500, 2000, 2500)
brain_slice = plot_scalar_on_slice(
    regions=regions,
    values=values,
    coord=-1000,              # ML (µm). Try a few values if needed.
    slice="sagittal",
    hemisphere="left",       # try "right" if you prefer the other side
    mapping="Allen",
    vector=True,
    background="boundary",
    empty_color="none",
    brain_atlas=ba,
    cmap=cmap,
    #norm=norm,
    show_cbar=False,         # we’ll add a clean legend instead
    ax=ax2
)
ax2.set_aspect('equal')  # Keep aspect ratio
# Legend (explicit group->color)
legend_handles = [
    Patch(facecolor=cmap(0), edgecolor="none", label="motor cortex"),
    Patch(facecolor=cmap(1), edgecolor="none", label="thalamus"),
    Patch(facecolor=cmap(2), edgecolor="none", label="hypothalamus (ZI)"),
]
ax2.legend(handles=legend_handles, loc="lower right", frameon=True)
ax2.set_title("Highlighted regions (sagittal slice)")
ax2.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
plt.show()
