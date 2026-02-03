
#%%

# Get Labels
uuids = np.array([c[0] for c in neuron_cols_keep], dtype=object)
acronyms = np.array([c[1] for c in neuron_cols_keep], dtype=object)

# 
betas = []  # list of (n_neurons,) arrays
for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
    model.fit(X[tr], y[tr])

    # get coefficients from the logistic regression inside the pipeline
    beta = model.named_steps["model"].coef_.ravel()  # shape: (n_neurons,)
    betas.append(beta)

betas = np.vstack(betas)  # shape: (n_folds, n_neurons)

# --- Summaries per neuron ---
beta_mean = betas.mean(axis=0)
beta_abs_mean = np.abs(betas).mean(axis=0)   # typical “importance” proxy

neur_summary = pd.DataFrame({
    "uuid": uuids,
    "acronym": acronyms,
    "beta_mean": beta_mean,
    "beta_abs_mean": beta_abs_mean,
})

area_summary = (
    neur_summary
    .groupby("acronym")
    .agg(
        n_neurons=("beta_abs_mean", "size"),
        mean_abs_beta=("beta_abs_mean", "mean"),
        sum_abs_beta=("beta_abs_mean", "sum"),
    )
    .sort_values("mean_abs_beta", ascending=False)
)

print(area_summary.head(15))

# --- Plot: choose metric ---
metric = "mean_abs_beta"  # or "sum_abs_beta"
top_k = 20
plot_df = area_summary.head(top_k)

plt.figure()
plt.bar(plot_df.index.astype(str), plot_df[metric].to_numpy())
plt.xticks(rotation=60, ha="right")
plt.ylabel(metric)
plt.title(f"Top areas by {metric} (logreg β from 5-fold CV)")
plt.tight_layout()
plt.show()
# %%


#area_subdivisions = { "motor areas": ["MOp5", "MOp6a", "MOp6b", "MOs5", "MOs6a", "MOs6b"],
#  "hypothalamus": ["ZI", "LHA", "DMH", "VMH", "AHN", "PH", "PVH", "MPO", "SHy", "STN", "TU", "TM", "PMv", "PMd"],
#  "thalamus": ["LGd", "LP", "PO", "RT", "VAL", "VM", "VPM", "VPL", "MG", "POL", "PT", "SPFm", "SPFll", "SPFvl"],
#} 





print(area_subdivisions)