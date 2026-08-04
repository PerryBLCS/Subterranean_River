"""Benchmark LowRankCrossFusion gate styles on the real Taiwan Credit data.

Usage:
    python run_gate_benchmark.py --epochs 30 --seeds 0 1 2
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import essay_tcn_reference as m


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def build_taiwan_features(df):
    """Build static features and 5-channel x 6-month temporal features."""
    eps = 1e-6
    n = len(df)
    pay = df[["PAY_6", "PAY_5", "PAY_4", "PAY_3", "PAY_2", "PAY_0"]].to_numpy(dtype=np.float32)
    bill = df[[f"BILL_AMT{i}" for i in range(6, 0, -1)]].to_numpy(dtype=np.float32)
    pay_amt = df[[f"PAY_AMT{i}" for i in range(6, 0, -1)]].to_numpy(dtype=np.float32)
    limit = df["LIMIT_BAL"].to_numpy(dtype=np.float32)[:, None]

    temporal = np.stack(
        [
            pay,
            signed_log1p(bill),
            signed_log1p(pay_amt),
            bill / np.maximum(limit, eps),
            pay_amt / (bill + eps),
        ],
        axis=-1,
    ).astype(np.float32)

    edu = df["EDUCATION"].map(lambda v: v if v in (1, 2, 3, 4) else 4).to_numpy()
    mar = df["MARRIAGE"].map(lambda v: v if v in (1, 2, 3) else 3).to_numpy()
    static_parts = [
        np.log1p(limit),
        df[["AGE"]].to_numpy(dtype=np.float32),
        pd.get_dummies(df["SEX"], prefix="sex").to_numpy(dtype=np.float32),
        pd.get_dummies(edu, prefix="edu").to_numpy(dtype=np.float32),
        pd.get_dummies(mar, prefix="mar").to_numpy(dtype=np.float32),
        bill.mean(axis=1, keepdims=True) / np.maximum(limit, eps),
        signed_log1p(bill).mean(axis=1, keepdims=True),
        signed_log1p(pay_amt).mean(axis=1, keepdims=True),
        (pay_amt / (bill + eps)).mean(axis=1, keepdims=True),
        (pay >= 3).sum(axis=1, keepdims=True).astype(np.float32),
        pay.max(axis=1, keepdims=True).astype(np.float32),
        pay.min(axis=1, keepdims=True).astype(np.float32),
        (bill[:, -1:] - bill[:, :1]) / np.maximum(limit, eps),
        (pay_amt[:, -1:] - pay_amt[:, :1]) / np.maximum(limit, eps),
    ]
    static = np.concatenate(static_parts, axis=1).astype(np.float32)
    y = df["default payment next month"].to_numpy(dtype=np.float32)
    return static, temporal, y


class FusionHead(nn.Module):
    def __init__(self, fusion):
        super().__init__()
        self.fusion = fusion
        self.clf = nn.Linear(fusion.static_proj.out_features, 1)

    def forward(self, static_feat, temporal_feat):
        fused = self.fusion(static_feat, temporal_feat)
        return self.clf(fused.mean(dim=1)).squeeze(-1)


class NoFusionHead(nn.Module):
    def __init__(self, static_dim, temporal_dim, hidden_dim):
        super().__init__()
        self.static_proj = nn.Linear(static_dim, hidden_dim)
        self.temporal_proj = nn.Linear(temporal_dim, hidden_dim)
        self.clf = nn.Linear(hidden_dim, 1)

    def forward(self, static_feat, temporal_feat):
        fused = self.static_proj(static_feat).unsqueeze(1) + self.temporal_proj(temporal_feat)
        return self.clf(fused.mean(dim=1)).squeeze(-1)


def train_and_evaluate(model, loader, static_val, temporal_val, y_val,
                       static_test, temporal_test, y_test, epochs, seed):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_val, best_state, wait = -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        for static_b, temporal_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(static_b, temporal_b), y_b)
            if hasattr(model, "fusion"):
                loss = loss + 0.02 * model.fusion.fusion_budget_loss()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            p_val = torch.sigmoid(model(static_val, temporal_val))
        auc_val = roc_auc_score(y_val.numpy(), p_val.numpy())
        if auc_val > best_val:
            best_val = auc_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 6:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_train = torch.sigmoid(model(loader.dataset.tensors[0], loader.dataset.tensors[1]))
        p_test = torch.sigmoid(model(static_test, temporal_test))
        gate = model.fusion.last_gate if hasattr(model, "fusion") else None

    result = {
        "val_auc": best_val,
        "test_auc": roc_auc_score(y_test.numpy(), p_test.numpy()),
        "test_auc_pr": average_precision_score(y_test.numpy(), p_test.numpy()),
        "train_auc": roc_auc_score(loader.dataset.tensors[2].numpy(), p_train.numpy()),
    }
    if gate is not None:
        g = gate.flatten()
        result["gate_mean"] = float(g.mean())
        result["gate_std"] = float(g.std())
        result["gate_lt01"] = float((g < 0.1).float().mean())
        result["gate_gt09"] = float((g > 0.9).float().mean())
    else:
        result.update(gate_mean=float("nan"), gate_std=float("nan"),
                      gate_lt01=float("nan"), gate_gt09=float("nan"))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    df = pd.read_excel("data/taiwan_credit.xls", header=1)
    static, temporal, y = build_taiwan_features(df)

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.15, stratify=y, random_state=42)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.15 / 0.85, stratify=y[train_idx], random_state=42
    )

    scaler_s = StandardScaler().fit(static[train_idx])
    scaler_t = StandardScaler().fit(temporal[train_idx].reshape(-1, temporal.shape[-1]))
    static = scaler_s.transform(static).astype(np.float32)
    temporal = scaler_t.transform(
        temporal.reshape(-1, temporal.shape[-1])
    ).reshape(temporal.shape).astype(np.float32)

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(static[train_idx]),
            torch.from_numpy(temporal[train_idx]),
            torch.from_numpy(y[train_idx]),
        ),
        batch_size=256,
        shuffle=True,
    )
    static_val = torch.from_numpy(static[val_idx])
    temporal_val = torch.from_numpy(temporal[val_idx])
    y_val = torch.from_numpy(y[val_idx])
    static_test = torch.from_numpy(static[test_idx])
    temporal_test = torch.from_numpy(temporal[test_idx])
    y_test = torch.from_numpy(y[test_idx])

    hidden = 128
    styles = ["none", "score", "mlp", "full"]
    print("gate_style | val_auc | test_auc | test_auc_pr | train_auc | gate_mean | gate_std | <0.1 | >0.9 | sec")
    for style in styles:
        rows = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            if style == "none":
                model = NoFusionHead(static.shape[1], temporal.shape[-1], hidden)
            else:
                fusion = m.LowRankCrossFusion(
                    static.shape[1], temporal.shape[-1], hidden,
                    rank=4, gate_style=style, gate_hidden=8,
                )
                model = FusionHead(fusion)
            start = time.time()
            rows.append(train_and_evaluate(
                model, loader, static_val, temporal_val, y_val,
                static_test, temporal_test, y_test, args.epochs, seed,
            ))
            rows[-1]["sec"] = time.time() - start

        def mean_std(key):
            values = [r[key] for r in rows]
            return np.mean(values), np.std(values)

        vals = {key: mean_std(key) for key in
                ["val_auc", "test_auc", "test_auc_pr", "train_auc",
                 "gate_mean", "gate_std", "gate_lt01", "gate_gt09", "sec"]}
        fmt = (
            f"{style:9s} | {vals['val_auc'][0]:.4f}+-{vals['val_auc'][1]:.4f} | "
            f"{vals['test_auc'][0]:.4f}+-{vals['test_auc'][1]:.4f} | "
            f"{vals['test_auc_pr'][0]:.4f}+-{vals['test_auc_pr'][1]:.4f} | "
            f"{vals['train_auc'][0]:.4f}+-{vals['train_auc'][1]:.4f} | "
            f"{vals['gate_mean'][0]:.3f}+-{vals['gate_mean'][1]:.3f} | "
            f"{vals['gate_std'][0]:.3f}+-{vals['gate_std'][1]:.3f} | "
            f"{vals['gate_lt01'][0]:.2f} | {vals['gate_gt09'][0]:.2f} | "
            f"{vals['sec'][0]:.0f}"
        )
        print(fmt)


if __name__ == "__main__":
    main()
