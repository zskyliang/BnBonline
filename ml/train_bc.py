#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ACTION_DIM = 5
CHANNELS = 5
ROWS = 13
COLS = 15


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_state_map(sample: dict) -> np.ndarray:
    state = sample.get("state", {})
    state_map = state.get("state_map", None)
    if state_map is None:
        state_map = sample.get("state_map", None)
    arr = np.asarray(state_map, dtype=np.float32)
    if arr.shape != (ROWS, COLS, CHANNELS):
        raise ValueError(f"invalid state_map shape {arr.shape}, expected {(ROWS, COLS, CHANNELS)}")
    return arr


def to_state_vec(sample: dict) -> np.ndarray:
    state = sample.get("state", {})
    vec = state.get("state_vector", None)
    if vec is None:
        vec = sample.get("state_vector", None)
    arr = np.asarray(vec, dtype=np.float32)
    if arr.shape != (2,):
        raise ValueError(f"invalid state_vector shape {arr.shape}, expected (2,)")
    return arr


def macro_f1_from_confusion(conf: np.ndarray) -> float:
    f1s = []
    for i in range(conf.shape[0]):
        tp = float(conf[i, i])
        fp = float(conf[:, i].sum() - tp)
        fn = float(conf[i, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall <= 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1s.append(f1)
    return float(np.mean(f1s))


@dataclass
class PreparedDataset:
    maps: np.ndarray
    vecs: np.ndarray
    actions: np.ndarray
    pre_death_ratio: float
    action_hist: Dict[str, int]


def prepare_dataset(rows: List[dict], drop_pre_death: bool) -> PreparedDataset:
    maps = []
    vecs = []
    actions = []
    pre_death_count = 0
    action_hist: Dict[str, int] = {}

    for item in rows:
        pre_death = bool(item.get("pre_death", False))
        if pre_death:
            pre_death_count += 1
        if drop_pre_death and pre_death:
            continue

        action = int(item.get("action", 0))
        if action < 0 or action >= ACTION_DIM:
            continue

        m = to_state_map(item)  # HWC
        v = to_state_vec(item)
        maps.append(np.transpose(m, (2, 0, 1)))  # CHW
        vecs.append(v)
        actions.append(action)
        action_hist[str(action)] = action_hist.get(str(action), 0) + 1

    if not maps:
        raise ValueError("dataset is empty after filtering")

    return PreparedDataset(
        maps=np.asarray(maps, dtype=np.float32),
        vecs=np.asarray(vecs, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        pre_death_ratio=(pre_death_count / max(1, len(rows))),
        action_hist=action_hist,
    )


class BCDataset(Dataset):
    def __init__(self, maps: np.ndarray, vecs: np.ndarray, actions: np.ndarray):
        self.maps = maps
        self.vecs = vecs
        self.actions = actions

    def __len__(self) -> int:
        return self.actions.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.maps[idx]),
            torch.from_numpy(self.vecs[idx]),
            torch.tensor(self.actions[idx], dtype=torch.long),
        )


class DodgeBCNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNELS, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * ROWS * COLS + 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, ACTION_DIM)

    def forward(self, state_map: torch.Tensor, state_vector: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(state_map))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.cat([x, state_vector], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, np.ndarray]:
    model.eval()
    correct = 0
    total = 0
    confusion = np.zeros((ACTION_DIM, ACTION_DIM), dtype=np.int64)
    with torch.no_grad():
        for maps, vecs, labels in loader:
            maps = maps.to(device)
            vecs = vecs.to(device)
            labels = labels.to(device)
            logits = model(maps, vecs)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.shape[0])
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[int(t), int(p)] += 1
    acc = correct / max(1, total)
    f1 = macro_f1_from_confusion(confusion)
    return acc, f1, confusion


def split_indices(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_n = max(1, int(math.floor(n * val_ratio)))
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    if train_idx.shape[0] == 0:
        train_idx = val_idx
    return train_idx, val_idx


def rebalance_wait_ratio(
    actions: np.ndarray, idx: np.ndarray, max_wait_ratio: float, seed: int
) -> np.ndarray:
    if max_wait_ratio >= 0.999:
        return idx
    if max_wait_ratio <= 0:
        return idx[actions[idx] != 0]
    train_actions = actions[idx]
    wait_mask = train_actions == 0
    wait_idx = idx[wait_mask]
    move_idx = idx[~wait_mask]
    if move_idx.shape[0] == 0:
        return idx
    max_wait_allowed = int((max_wait_ratio / max(1e-8, 1 - max_wait_ratio)) * move_idx.shape[0])
    if max_wait_allowed < 1:
        max_wait_allowed = 1
    if wait_idx.shape[0] <= max_wait_allowed:
        return idx
    rng = np.random.default_rng(seed + 7)
    rng.shuffle(wait_idx)
    keep_wait = wait_idx[:max_wait_allowed]
    merged = np.concatenate([move_idx, keep_wait], axis=0)
    rng.shuffle(merged)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BC dodge model and export ONNX.")
    parser.add_argument("--dataset", default="output/ml/datasets/dodge_bc_v1.jsonl")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler", choices=["balanced", "uniform"], default="balanced")
    parser.add_argument("--sampler-power", type=float, default=0.75)
    parser.add_argument("--max-wait-ratio-train", type=float, default=0.6)
    parser.add_argument("--drop-pre-death", action="store_true", default=True)
    parser.add_argument("--keep-pre-death", action="store_true", help="Keep pre_death samples in BC dataset.")
    parser.add_argument("--out-pt", default="output/ml/models/best.pt")
    parser.add_argument("--out-onnx", default="output/ml/models/dodge_bc_v1.onnx")
    parser.add_argument("--out-metrics", default="output/ml/reports/bc_v1_metrics.json")
    args = parser.parse_args()

    if args.keep_pre_death:
        args.drop_pre_death = False

    set_seed(args.seed)

    raw_rows = read_jsonl(args.dataset)
    data = prepare_dataset(raw_rows, drop_pre_death=args.drop_pre_death)
    train_idx, val_idx = split_indices(data.actions.shape[0], args.val_ratio, args.seed)
    train_idx = rebalance_wait_ratio(
        data.actions, train_idx, max_wait_ratio=args.max_wait_ratio_train, seed=args.seed
    )

    train_ds = BCDataset(data.maps[train_idx], data.vecs[train_idx], data.actions[train_idx])
    val_ds = BCDataset(data.maps[val_idx], data.vecs[val_idx], data.actions[val_idx])
    train_counts = np.bincount(data.actions[train_idx], minlength=ACTION_DIM).astype(np.float32)
    if args.sampler == "balanced":
        sample_weights = np.power(1.0 / np.maximum(train_counts[data.actions[train_idx]], 1.0), args.sampler_power)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=int(train_idx.shape[0]),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=False, drop_last=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DodgeBCNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    class_weights = np.zeros((ACTION_DIM,), dtype=np.float32)
    nonzero_mask = train_counts > 0
    class_weights[nonzero_mask] = 1.0 / train_counts[nonzero_mask]
    class_weights = class_weights / max(1e-8, float(class_weights.sum())) * ACTION_DIM
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))

    best_val_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for maps, vecs, labels in train_loader:
            maps = maps.to(device)
            vecs = vecs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(maps, vecs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_n = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_n
            total_count += batch_n

        train_loss = total_loss / max(1, total_count)
        val_acc, val_f1, _ = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
            }
        )
        print(
            f"[EPOCH {epoch}] train_loss={train_loss:.6f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}",
            flush=True,
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("failed to get best model state")

    model.load_state_dict(best_state)
    model.eval()

    val_acc, val_f1, confusion = evaluate(model, val_loader, device)
    recall_by_action = {}
    for i in range(ACTION_DIM):
        denom = int(confusion[i, :].sum())
        recall_by_action[str(i)] = float(confusion[i, i] / denom) if denom > 0 else 0.0

    ensure_parent(args.out_pt)
    ensure_parent(args.out_onnx)
    ensure_parent(args.out_metrics)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "rows": ROWS,
                "cols": COLS,
                "channels": CHANNELS,
                "action_dim": ACTION_DIM,
            },
        },
        args.out_pt,
    )

    dummy_map = torch.zeros((1, CHANNELS, ROWS, COLS), dtype=torch.float32).to(device)
    dummy_vec = torch.zeros((1, 2), dtype=torch.float32).to(device)
    torch.onnx.export(
        model,
        (dummy_map, dummy_vec),
        args.out_onnx,
        input_names=["state_map", "state_vector"],
        output_names=["logits"],
        opset_version=18,
    )
    # Keep ONNX as a single-file artifact for browser runtime loading.
    merged = onnx.load(args.out_onnx, load_external_data=True)
    onnx.save_model(merged, args.out_onnx, save_as_external_data=False)
    sidecar = args.out_onnx + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)

    metrics = {
        "dataset_path": args.dataset,
        "dataset_size_raw": len(raw_rows),
        "dataset_size_used": int(data.actions.shape[0]),
        "drop_pre_death": bool(args.drop_pre_death),
        "sampler": args.sampler,
        "sampler_power": args.sampler_power,
        "max_wait_ratio_train": args.max_wait_ratio_train,
        "pre_death_ratio_raw": data.pre_death_ratio,
        "train_size": int(train_idx.shape[0]),
        "val_size": int(val_idx.shape[0]),
        "action_hist": data.action_hist,
        "train_action_hist": {str(i): int(train_counts[i]) for i in range(ACTION_DIM)},
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
        "confusion_matrix": confusion.tolist(),
        "recall_by_action": recall_by_action,
        "history": history,
        "out_pt": args.out_pt,
        "out_onnx": args.out_onnx,
    }
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("[DONE] metrics:", args.out_metrics)


if __name__ == "__main__":
    main()
