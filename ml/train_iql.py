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
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler


ACTION_DIM = 5
ROWS = 13
COLS = 15
TARGET_CHANNELS = 8
TARGET_VEC_DIM = 9


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


def normalize_state_map(sample: dict) -> np.ndarray:
    state = sample.get("state", {})
    state_map = state.get("state_map", sample.get("state_map"))
    arr = np.asarray(state_map, dtype=np.float32)
    if arr.shape[0:2] != (ROWS, COLS):
        raise ValueError(f"invalid state_map spatial shape: {arr.shape}")
    if arr.shape[2] == TARGET_CHANNELS:
        return arr
    if arr.shape[2] < 5:
        raise ValueError(f"state_map channels too small: {arr.shape}")

    out = np.zeros((ROWS, COLS, TARGET_CHANNELS), dtype=np.float32)
    copy_ch = min(arr.shape[2], TARGET_CHANNELS)
    out[:, :, :copy_ch] = arr[:, :, :copy_ch]
    # Backward-compatible derivations for old 5-channel data.
    if arr.shape[2] < 6:
        out[:, :, 5] = out[:, :, 2]  # blast surrogate
    if arr.shape[2] < 7:
        out[:, :, 6] = (out[:, :, 4] > 0.5).astype(np.float32)  # reachable surrogate
    if arr.shape[2] < 8:
        left = np.roll(out[:, :, 2], 1, axis=1)
        right = np.roll(out[:, :, 2], -1, axis=1)
        out[:, :, 7] = np.clip(np.abs(left - right), 0, 1)  # half-body boundary surrogate
    return out


def normalize_state_vec(sample: dict) -> np.ndarray:
    state = sample.get("state", {})
    state_vec = state.get("state_vector", sample.get("state_vector"))
    vec = np.asarray(state_vec, dtype=np.float32).reshape(-1)
    if vec.shape[0] >= TARGET_VEC_DIM:
        return vec[:TARGET_VEC_DIM]
    out = np.zeros((TARGET_VEC_DIM,), dtype=np.float32)
    out[: vec.shape[0]] = vec
    meta = sample.get("meta", {})
    if vec.shape[0] <= 7:
        out[7] = float(meta.get("safeNeighbors", 0)) / 4.0
    if vec.shape[0] <= 8:
        out[8] = min(1.0, float(meta.get("activeBombs", 0)) / 10.0)
    return np.clip(out, -1.0, 1.0)


def normalize_next_state_map(sample: dict) -> np.ndarray:
    nxt = sample.get("next_state")
    if not nxt:
        return normalize_state_map(sample)
    fake = {"state": nxt, "meta": sample.get("meta", {})}
    return normalize_state_map(fake)


def normalize_next_state_vec(sample: dict) -> np.ndarray:
    nxt = sample.get("next_state")
    if not nxt:
        return normalize_state_vec(sample)
    fake = {"state": nxt, "meta": sample.get("meta", {})}
    return normalize_state_vec(fake)


def infer_risk_label(sample: dict) -> int:
    if "risk_label" in sample:
        return 1 if int(sample.get("risk_label", 0)) > 0 else 0
    if sample.get("pre_death", False):
        return 1
    if sample.get("done", False):
        return 1
    if float(sample.get("reward", 0.0)) < -0.5:
        return 1
    meta = sample.get("meta", {})
    if int(meta.get("activeBombs", 0)) > 0 and int(meta.get("safeNeighbors", 0)) <= 1:
        return 1
    return 0


@dataclass
class PreparedDataset:
    state_maps: np.ndarray
    state_vecs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    next_maps: np.ndarray
    next_vecs: np.ndarray
    risk_labels: np.ndarray
    policy_tags: np.ndarray
    action_hist: Dict[str, int]
    policy_tag_hist: Dict[str, int]


def prepare_dataset(rows: List[dict]) -> PreparedDataset:
    s_maps = []
    s_vecs = []
    actions = []
    rewards = []
    dones = []
    n_maps = []
    n_vecs = []
    risk_labels = []
    policy_tags = []
    action_hist: Dict[str, int] = {}
    policy_tag_hist: Dict[str, int] = {}

    for item in rows:
        action = int(item.get("action", 0))
        if action < 0 or action >= ACTION_DIM:
            continue
        s_map = normalize_state_map(item)
        s_vec = normalize_state_vec(item)
        n_map = normalize_next_state_map(item)
        n_vec = normalize_next_state_vec(item)

        reward = float(item.get("reward", 0.0))
        done = 1.0 if bool(item.get("done", False)) else 0.0
        risk = float(infer_risk_label(item))

        s_maps.append(np.transpose(s_map, (2, 0, 1)))
        s_vecs.append(s_vec)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        n_maps.append(np.transpose(n_map, (2, 0, 1)))
        n_vecs.append(n_vec)
        risk_labels.append(risk)
        policy_tag = str(item.get("policy_tag", "expert"))
        if policy_tag == "random":
            policy_id = 1
        elif policy_tag == "epsilon":
            policy_id = 2
        else:
            policy_id = 0
            policy_tag = "expert"
        policy_tags.append(policy_id)

        action_hist[str(action)] = action_hist.get(str(action), 0) + 1
        policy_tag_hist[policy_tag] = policy_tag_hist.get(policy_tag, 0) + 1

    if not actions:
        raise ValueError("dataset is empty after filtering")

    return PreparedDataset(
        state_maps=np.asarray(s_maps, dtype=np.float32),
        state_vecs=np.asarray(s_vecs, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        next_maps=np.asarray(n_maps, dtype=np.float32),
        next_vecs=np.asarray(n_vecs, dtype=np.float32),
        risk_labels=np.asarray(risk_labels, dtype=np.float32),
        policy_tags=np.asarray(policy_tags, dtype=np.int64),
        action_hist=action_hist,
        policy_tag_hist=policy_tag_hist,
    )


class IQLDataset(Dataset):
    def __init__(
        self,
        s_maps: np.ndarray,
        s_vecs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        n_maps: np.ndarray,
        n_vecs: np.ndarray,
        risk_labels: np.ndarray,
        policy_tags: np.ndarray,
    ):
        self.s_maps = s_maps
        self.s_vecs = s_vecs
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.n_maps = n_maps
        self.n_vecs = n_vecs
        self.risk_labels = risk_labels
        self.policy_tags = policy_tags

    def __len__(self) -> int:
        return self.actions.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.s_maps[idx]),
            torch.from_numpy(self.s_vecs[idx]),
            torch.tensor(self.actions[idx], dtype=torch.long),
            torch.tensor(self.rewards[idx], dtype=torch.float32),
            torch.tensor(self.dones[idx], dtype=torch.float32),
            torch.from_numpy(self.n_maps[idx]),
            torch.from_numpy(self.n_vecs[idx]),
            torch.tensor(self.risk_labels[idx], dtype=torch.float32),
            torch.tensor(self.policy_tags[idx], dtype=torch.long),
        )


class IQLNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(TARGET_CHANNELS, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * ROWS * COLS + TARGET_VEC_DIM, 256)
        self.q1 = nn.Linear(256, ACTION_DIM)
        self.q2 = nn.Linear(256, ACTION_DIM)
        self.v = nn.Linear(256, 1)
        self.policy = nn.Linear(256, ACTION_DIM)
        self.risk = nn.Linear(256, 1)

    def encode(self, state_map: torch.Tensor, state_vec: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(state_map))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.cat([x, state_vec], dim=1)
        return F.relu(self.fc(x))

    def forward(self, state_map: torch.Tensor, state_vec: torch.Tensor):
        z = self.encode(state_map, state_vec)
        return self.q1(z), self.q2(z), self.v(z), self.policy(z), self.risk(z)


class IQLPolicyExport(nn.Module):
    def __init__(self, net: IQLNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, state_map: torch.Tensor, state_vec: torch.Tensor):
        _, _, _, policy_logits, risk_logit = self.net(state_map, state_vec)
        return policy_logits, risk_logit


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


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, torch.full_like(diff, expectile), torch.full_like(diff, 1.0 - expectile))
    return (weight * diff.pow(2)).mean()


def evaluate_policy(model: IQLNet, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    total = 0
    correct = 0
    risk_total = 0
    risk_correct = 0
    conf = np.zeros((ACTION_DIM, ACTION_DIM), dtype=np.int64)
    with torch.no_grad():
        for s_map, s_vec, actions, _, _, _, _, risk_labels, _ in loader:
            s_map = s_map.to(device)
            s_vec = s_vec.to(device)
            actions = actions.to(device)
            risk_labels = risk_labels.to(device)
            _, _, _, policy_logits, risk_logit = model(s_map, s_vec)
            pred_actions = torch.argmax(policy_logits, dim=1)
            correct += int((pred_actions == actions).sum().item())
            total += int(actions.shape[0])
            for t, p in zip(actions.cpu().numpy(), pred_actions.cpu().numpy()):
                conf[int(t), int(p)] += 1
            pred_risk = (torch.sigmoid(risk_logit.squeeze(1)) >= 0.5).float()
            risk_correct += int((pred_risk == risk_labels).sum().item())
            risk_total += int(risk_labels.shape[0])
    acc = correct / max(1, total)
    f1 = macro_f1_from_confusion(conf)
    risk_acc = risk_correct / max(1, risk_total)
    return acc, f1, risk_acc


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * sp.data)


def build_balanced_sampler(
    actions: np.ndarray,
    policy_tags: np.ndarray,
    action_power: float,
    policy_power: float,
) -> WeightedRandomSampler:
    action_counts = np.bincount(actions, minlength=ACTION_DIM).astype(np.float64)
    policy_counts = np.bincount(policy_tags, minlength=3).astype(np.float64)
    action_counts[action_counts <= 0] = 1.0
    policy_counts[policy_counts <= 0] = 1.0
    action_weights = np.power(action_counts, -max(0.0, action_power))
    policy_weights = np.power(policy_counts, -max(0.0, policy_power))
    sample_weights = action_weights[actions] * policy_weights[policy_tags]
    sample_weights = np.maximum(sample_weights, 1e-8)
    weights_tensor = torch.from_numpy(sample_weights.astype(np.float64))
    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(actions),
        replacement=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train discrete IQL dodge model with auxiliary risk head.")
    parser.add_argument("--dataset", default="output/ml/datasets/dodge_iql_v1_mixed_200k.jsonl")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--adv-max-weight", type=float, default=80.0)
    parser.add_argument("--risk-loss-weight", type=float, default=0.35)
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument("--sampler", choices=["balanced", "uniform"], default="balanced")
    parser.add_argument("--action-balance-power", type=float, default=0.7)
    parser.add_argument("--policy-balance-power", type=float, default=0.5)
    parser.add_argument("--out-pt", default="output/ml/models/dodge_iql_v1.pt")
    parser.add_argument("--out-onnx", default="output/ml/models/dodge_iql_v1.onnx")
    parser.add_argument("--out-metrics", default="output/ml/reports/iql_v1_metrics.json")
    args = parser.parse_args()

    set_seed(args.seed)

    rows = read_jsonl(args.dataset)
    data = prepare_dataset(rows)
    n = data.actions.shape[0]
    train_idx, val_idx = split_indices(n, args.val_ratio, args.seed)

    train_ds = IQLDataset(
        data.state_maps[train_idx],
        data.state_vecs[train_idx],
        data.actions[train_idx],
        data.rewards[train_idx],
        data.dones[train_idx],
        data.next_maps[train_idx],
        data.next_vecs[train_idx],
        data.risk_labels[train_idx],
        data.policy_tags[train_idx],
    )
    val_ds = IQLDataset(
        data.state_maps[val_idx],
        data.state_vecs[val_idx],
        data.actions[val_idx],
        data.rewards[val_idx],
        data.dones[val_idx],
        data.next_maps[val_idx],
        data.next_vecs[val_idx],
        data.risk_labels[val_idx],
        data.policy_tags[val_idx],
    )
    if args.sampler == "balanced":
        train_sampler = build_balanced_sampler(
            data.actions[train_idx],
            data.policy_tags[train_idx],
            args.action_balance_power,
            args.policy_balance_power,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, shuffle=False, drop_last=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IQLNet().to(device)
    target_model = IQLNet().to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: List[dict] = []
    best_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for s_map, s_vec, actions, rewards, dones, n_map, n_vec, risk_labels, _ in train_loader:
            s_map = s_map.to(device)
            s_vec = s_vec.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)
            n_map = n_map.to(device)
            n_vec = n_vec.to(device)
            risk_labels = risk_labels.to(device)

            q1, q2, v, policy_logits, risk_logit = model(s_map, s_vec)
            with torch.no_grad():
                _, _, next_v, _, _ = target_model(n_map, n_vec)
                next_v = next_v.squeeze(1)
                q_target = rewards + args.gamma * (1.0 - dones) * next_v

            q1_a = q1.gather(1, actions.unsqueeze(1)).squeeze(1)
            q2_a = q2.gather(1, actions.unsqueeze(1)).squeeze(1)
            q_loss = F.mse_loss(q1_a, q_target) + F.mse_loss(q2_a, q_target)

            q_min_a = torch.min(q1_a, q2_a).detach()
            v_pred = v.squeeze(1)
            v_loss = expectile_loss(q_min_a - v_pred, args.expectile)

            adv = (q_min_a - v_pred.detach())
            exp_adv = torch.exp(args.beta * adv).clamp(max=args.adv_max_weight)
            logp = F.log_softmax(policy_logits, dim=1).gather(1, actions.unsqueeze(1)).squeeze(1)
            actor_loss = -(exp_adv * logp).mean()

            risk_loss = F.binary_cross_entropy_with_logits(risk_logit.squeeze(1), risk_labels)

            loss = q_loss + v_loss + actor_loss + args.risk_loss_weight * risk_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            soft_update(target_model, model, args.target_tau)

            batch_n = int(actions.shape[0])
            total_loss += float(loss.item()) * batch_n
            total_n += batch_n

        train_loss = total_loss / max(1, total_n)
        val_acc, val_f1, val_risk_acc = evaluate_policy(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_policy_acc": val_acc,
                "val_policy_f1": val_f1,
                "val_risk_acc": val_risk_acc,
            }
        )
        print(
            f"[EPOCH {epoch}] train_loss={train_loss:.6f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_risk_acc={val_risk_acc:.4f}",
            flush=True,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("failed to get best model")

    model.load_state_dict(best_state)

    val_acc, val_f1, val_risk_acc = evaluate_policy(model, val_loader, device)

    ensure_parent(args.out_pt)
    ensure_parent(args.out_onnx)
    ensure_parent(args.out_metrics)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "rows": ROWS,
                "cols": COLS,
                "channels": TARGET_CHANNELS,
                "vec_dim": TARGET_VEC_DIM,
                "action_dim": ACTION_DIM,
            },
        },
        args.out_pt,
    )

    export_model = IQLPolicyExport(model).to(device)
    export_model.eval()
    dummy_map = torch.zeros((1, TARGET_CHANNELS, ROWS, COLS), dtype=torch.float32).to(device)
    dummy_vec = torch.zeros((1, TARGET_VEC_DIM), dtype=torch.float32).to(device)

    torch.onnx.export(
        export_model,
        (dummy_map, dummy_vec),
        args.out_onnx,
        input_names=["state_map", "state_vector"],
        output_names=["policy_logits", "risk_logit"],
        opset_version=18,
    )
    merged = onnx.load(args.out_onnx, load_external_data=True)
    onnx.save_model(merged, args.out_onnx, save_as_external_data=False)
    sidecar = args.out_onnx + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)

    risk_ratio = float(np.mean(data.risk_labels)) if data.risk_labels.size > 0 else 0.0
    metrics = {
        "dataset_path": args.dataset,
        "dataset_size": int(n),
        "train_size": int(train_idx.shape[0]),
        "val_size": int(val_idx.shape[0]),
        "action_hist": data.action_hist,
        "policy_tag_hist": data.policy_tag_hist,
        "risk_label_ratio": risk_ratio,
        "val_policy_acc": val_acc,
        "val_policy_f1": val_f1,
        "val_risk_acc": val_risk_acc,
        "history": history,
        "config": {
            "gamma": args.gamma,
            "expectile": args.expectile,
            "beta": args.beta,
            "adv_max_weight": args.adv_max_weight,
            "risk_loss_weight": args.risk_loss_weight,
            "target_tau": args.target_tau,
            "sampler": args.sampler,
            "action_balance_power": args.action_balance_power,
            "policy_balance_power": args.policy_balance_power,
            "channels": TARGET_CHANNELS,
            "vec_dim": TARGET_VEC_DIM,
        },
        "out_pt": args.out_pt,
        "out_onnx": args.out_onnx,
    }
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("[DONE] metrics:", args.out_metrics)


if __name__ == "__main__":
    main()
