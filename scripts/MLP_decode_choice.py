#%% MLP decoding of choices from firing rates
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

# Settings
balance_by_downsample = True
keep_acronyms = {"MOp5", "MOp6a", "MOp6b"}  # motor areas

seed = 0
n_splits = 5
batch_size = 64
epochs = 100 # number of training epochs, you can change this and see how it affects performance.
lr = 1e-3 # learning rate, change and see how it affects performance.
weight_decay = 0.0 # set to e.g., 1e-4 for L2 regularization
hidden_dim = 64
dropout = 0.0  # set e.g. 0.2 if you want regularisation

in_path = Path("data/firing_rates.xlsx")
out_path = Path("results/choice_decoding_accuracies.xlsx")
out_path.parent.mkdir(parents=True, exist_ok=True)


# Reproducibility
rng = np.random.default_rng(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_excel(in_path, header=[0, 1])  # MultiIndex columns
df = df.dropna(how="all").reset_index(drop=True)

# Convert choices to binary: left (-1) -> 0, right (+1) -> 1
y = (df["choice"].values == 1).astype(np.int64).ravel()

# Meta columns to keep (not used further, but kept for parity with your script)
meta_lvl0 = {"choice", "trial_id", "uuid"}
meta_cols = [c for c in df.columns if c[0] in meta_lvl0]

# Keep only neurons from specific brain areas
neuron_cols_keep = [
    c for c in df.columns
    if (c[0] not in meta_lvl0) and (c[1] in keep_acronyms)
]

X = df[neuron_cols_keep].to_numpy(dtype=float)

# Balance classes by downsampling
if balance_by_downsample:
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n = min(len(idx0), len(idx1))
    keep = np.concatenate([
        rng.choice(idx0, n, replace=False),
        rng.choice(idx1, n, replace=False),
    ])
    X, y = X[keep], y[keep]

# Helper: standardize (no leakage)
def standardize_with_train_stats(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    std = np.where(std == 0, 1.0, std)  # avoid divide-by-zero
    return (X_train - mean) / std, (X_test - mean) / std

# Model
class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # 2 classes (0/1)
        )

    def forward(self, x):
        return self.net(x)

# Cross-validation
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
fold_accs = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # fold-wise standardize
    X_train, X_test = standardize_with_train_stats(X_train, X_test)

    # tensors
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    # model
    model = SimpleMLP(in_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)

    # Loss funcrion
    criterion = nn.CrossEntropyLoss()

    # Optimiser (Adam with weight decay for L2 regularization)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # train
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()

    # test
    model.eval()
    with torch.no_grad():
        logits = model(Xte.to(device))
        pred = torch.argmax(logits, dim=1).cpu().numpy()
    acc = float((pred == y_test).mean())
    fold_accs.append(acc)

    print(f"Fold {fold}/{n_splits} accuracy: {acc:.3f}")

fold_accs = np.array(fold_accs, dtype=float)
acc_m = float(fold_accs.mean())
acc_s = float(fold_accs.std(ddof=1))

print("\n5-fold cross-validation performance (test):")
print(f"  Accuracy: {acc_m:.3f} ± {acc_s:.3f}")
print("Fold accuracies:", np.round(fold_accs, 3))

# save
df_acc = pd.DataFrame({"mlp_motor_cortex": fold_accs})
df_acc.to_excel(out_path, index=False)
print(f"\nSaved fold accuracies to: {out_path}")
