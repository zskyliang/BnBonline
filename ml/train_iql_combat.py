#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler


ACTION_DIM = 6
ROWS = 13
COLS = 15
TARGET_CHANNELS = 10
TARGET_VEC_DIM = 16

OUTCOME_TO_ID = {
    "ongoing": 0,
    "win": 1,
    "loss": 2,
    "draw": 3,
    "self_kill": 4,
}
ID_TO_OUTCOME = {v: k for k, v in OUTCOME_TO_ID.items()}

COMBAT_AUX_NAMES = [
    "post_bomb_escape_le2",
    "post_bomb_escape_le4",
    "post_bomb_escape_le6",
    "min_escape_eta_after_bomb",
    "corridor_deadend_depth",
    "enemy_recent_bomb_cd",
    "enemy_heading_delta",
    "item_race_delta",
    "local_choke_occupancy",
]
COMBAT_AUX_DIM = len(COMBAT_AUX_NAMES)


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

    out = np.zeros((ROWS, COLS, TARGET_CHANNELS), dtype=np.float32)
    copy_ch = min(arr.shape[2], TARGET_CHANNELS)
    out[:, :, :copy_ch] = arr[:, :, :copy_ch]

    # Backward compatibility for older dodge features.
    if arr.shape[2] < 6:
        out[:, :, 5] = out[:, :, 2]
    if arr.shape[2] < 7:
        out[:, :, 6] = (out[:, :, 4] > 0.5).astype(np.float32)
    if arr.shape[2] < 8:
        left = np.roll(out[:, :, 2], 1, axis=1)
        right = np.roll(out[:, :, 2], -1, axis=1)
        out[:, :, 7] = np.clip(np.abs(left - right), 0, 1)
    if arr.shape[2] < 9:
        out[:, :, 8] = 0
    if arr.shape[2] < 10:
        out[:, :, 9] = 0

    return np.clip(out, -1.0, 1.0)


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


def clamp01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def normalize_outcome_tag(sample: dict) -> str:
    raw = str(sample.get("outcome_tag", "ongoing")).strip().lower()
    done = bool(sample.get("done", False))
    pre_death = bool(sample.get("pre_death", False))

    if raw == "death":
        raw = "self_kill" if pre_death else "loss"
    if raw == "done":
        raw = "draw" if done else "ongoing"

    if raw in OUTCOME_TO_ID:
        return raw
    if done and pre_death:
        return "self_kill"
    if done:
        return "draw"
    return "ongoing"


def infer_risk_label(sample: dict, outcome_tag: str) -> int:
    if "risk_label" in sample:
        return 1 if int(sample.get("risk_label", 0)) > 0 else 0
    if sample.get("pre_death", False):
        return 1
    if outcome_tag in ("loss", "self_kill"):
        return 1
    if sample.get("done", False):
        return 1
    if float(sample.get("reward", 0.0)) < -0.5:
        return 1
    meta = sample.get("meta", {})
    if int(meta.get("activeBombs", 0)) > 0 and int(meta.get("safeNeighbors", 0)) <= 1:
        return 1
    return 0


def normalize_action_mask(sample: dict, action: int) -> np.ndarray:
    mask = np.ones((ACTION_DIM,), dtype=np.float32)
    raw = sample.get("action_mask")
    if isinstance(raw, list) and len(raw) >= ACTION_DIM:
        mask = np.asarray(raw[:ACTION_DIM], dtype=np.float32)
        mask = np.where(mask > 0.5, 1.0, 0.0).astype(np.float32)
    if mask.sum() <= 0:
        mask[:] = 1.0
    if action < 0 or action >= ACTION_DIM:
        return mask
    if mask[action] <= 0:
        # Keep BC/IQL target action valid for log-prob computation.
        mask[action] = 1.0
    return mask


def find_peak_pos(channel: np.ndarray, threshold: float = 0.5) -> Tuple[int, int]:
    if channel.ndim != 2:
        return -1, -1
    idx = np.unravel_index(np.argmax(channel), channel.shape)
    if float(channel[idx]) < threshold:
        return -1, -1
    return int(idx[0]), int(idx[1])


def collect_walkable_mask(state_map: np.ndarray) -> np.ndarray:
    obstacle = state_map[:, :, 0]
    bomb = state_map[:, :, 1]
    walkable = (obstacle < 0.85) & (bomb < 0.95)
    return walkable


def bfs_distances(walkable: np.ndarray, start_y: int, start_x: int, max_depth: int) -> np.ndarray:
    dist = np.full((ROWS, COLS), fill_value=999, dtype=np.int32)
    if start_y < 0 or start_x < 0 or start_y >= ROWS or start_x >= COLS:
        return dist
    if not bool(walkable[start_y, start_x]):
        return dist
    qy = [start_y]
    qx = [start_x]
    head = 0
    dist[start_y, start_x] = 0
    while head < len(qy):
        y = qy[head]
        x = qx[head]
        head += 1
        d = int(dist[y, x])
        if d >= max_depth:
            continue
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ny < 0 or ny >= ROWS or nx < 0 or nx >= COLS:
                continue
            if not bool(walkable[ny, nx]):
                continue
            if dist[ny, nx] <= d + 1:
                continue
            dist[ny, nx] = d + 1
            qy.append(ny)
            qx.append(nx)
    return dist


def build_hypothetical_blast_mask(state_map: np.ndarray, y0: int, x0: int, power: int) -> np.ndarray:
    out = np.zeros((ROWS, COLS), dtype=np.bool_)
    if y0 < 0 or x0 < 0:
        return out
    out[y0, x0] = True
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        y = y0
        x = x0
        for _ in range(max(1, int(power))):
            y += dy
            x += dx
            if y < 0 or y >= ROWS or x < 0 or x >= COLS:
                break
            out[y, x] = True
            if state_map[y, x, 0] >= 0.95:
                break
    return out


def count_walkable_neighbors(walkable: np.ndarray, y: int, x: int) -> int:
    c = 0
    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
        if ny < 0 or ny >= ROWS or nx < 0 or nx >= COLS:
            continue
        if bool(walkable[ny, nx]):
            c += 1
    return c


def compute_corridor_deadend_depth(walkable: np.ndarray, y0: int, x0: int, max_depth: int = 8) -> float:
    if y0 < 0 or x0 < 0 or not bool(walkable[y0, x0]):
        return 0.0
    if count_walkable_neighbors(walkable, y0, x0) >= 3:
        return 0.0

    best = 0
    visited = {(y0, x0)}
    frontier = [(y0, x0, 0)]
    while frontier:
        y, x, d = frontier.pop(0)
        best = max(best, d)
        if d >= max_depth:
            continue
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ny < 0 or ny >= ROWS or nx < 0 or nx >= COLS:
                continue
            if not bool(walkable[ny, nx]):
                continue
            if (ny, nx) in visited:
                continue
            deg = count_walkable_neighbors(walkable, ny, nx)
            if deg >= 3:
                continue
            visited.add((ny, nx))
            frontier.append((ny, nx, d + 1))
    return clamp01(best / float(max_depth))


def compute_item_race_delta(state_map: np.ndarray, self_pos: Tuple[int, int], enemy_pos: Tuple[int, int]) -> float:
    sy, sx = self_pos
    ey, ex = enemy_pos
    if sy < 0 or ey < 0:
        return 0.5

    walkable = collect_walkable_mask(state_map)
    self_dist = bfs_distances(walkable, sy, sx, max_depth=24)
    enemy_dist = bfs_distances(walkable, ey, ex, max_depth=24)

    item_mask = state_map[:, :, 9] > 0.1
    item_idx = np.argwhere(item_mask)
    if item_idx.shape[0] == 0:
        return 0.5

    best_delta = 0.0
    for y, x in item_idx:
        ds = int(self_dist[y, x])
        de = int(enemy_dist[y, x])
        if ds >= 999 and de >= 999:
            continue
        if ds >= 999:
            delta = -1.0
        elif de >= 999:
            delta = 1.0
        else:
            delta = (de - ds) / 10.0
        best_delta = max(best_delta, max(-1.0, min(1.0, float(delta))))

    return clamp01((best_delta + 1.0) * 0.5)


def compute_local_choke_occupancy(state_map: np.ndarray, self_pos: Tuple[int, int], enemy_pos: Tuple[int, int]) -> float:
    sy, sx = self_pos
    ey, ex = enemy_pos
    if sy < 0 or ey < 0:
        return 0.0

    walkable = collect_walkable_mask(state_map)
    self_dist = bfs_distances(walkable, sy, sx, max_depth=10)
    enemy_dist = bfs_distances(walkable, ey, ex, max_depth=10)

    total = 0
    enemy_dom = 0
    for y in range(max(0, sy - 4), min(ROWS, sy + 5)):
        for x in range(max(0, sx - 4), min(COLS, sx + 5)):
            if not bool(walkable[y, x]):
                continue
            deg = count_walkable_neighbors(walkable, y, x)
            if deg > 2:
                continue
            ds = int(self_dist[y, x])
            de = int(enemy_dist[y, x])
            if ds >= 999 and de >= 999:
                continue
            total += 1
            if de < ds:
                enemy_dom += 1
    if total <= 0:
        return 0.0
    return clamp01(enemy_dom / float(total))


def compute_combat_aux_features(
    sample: dict,
    state_map: np.ndarray,
    state_vec: np.ndarray,
    next_map: np.ndarray,
) -> np.ndarray:
    out = np.zeros((COMBAT_AUX_DIM,), dtype=np.float32)

    sy, sx = find_peak_pos(state_map[:, :, 3], threshold=0.5)
    ey, ex = find_peak_pos(state_map[:, :, 8], threshold=0.2)
    n_ey, n_ex = find_peak_pos(next_map[:, :, 8], threshold=0.2)

    walkable = collect_walkable_mask(state_map)

    # post_bomb_escape_steps + min_escape_eta_after_bomb
    if sy >= 0 and sx >= 0:
        dist = bfs_distances(walkable, sy, sx, max_depth=6)
        power_norm = float(state_vec[11]) if state_vec.shape[0] > 11 else 0.3
        power = max(1, int(round(power_norm * 6)))
        blast = build_hypothetical_blast_mask(state_map, sy, sx, power)

        safe_steps = 999
        for y in range(ROWS):
            for x in range(COLS):
                d = int(dist[y, x])
                if d > 6:
                    continue
                if blast[y, x]:
                    continue
                if state_map[y, x, 2] > 0.75:
                    continue
                safe_steps = min(safe_steps, d)

        if safe_steps <= 2:
            out[0] = 1.0
        if safe_steps <= 4:
            out[1] = 1.0
        if safe_steps <= 6:
            out[2] = 1.0

        if safe_steps <= 6:
            out[3] = clamp01((6.0 - float(safe_steps)) / 6.0)
        else:
            out[3] = 0.0

        out[4] = compute_corridor_deadend_depth(walkable, sy, sx)

    # enemy_recent_bomb_cd (fallback from enemy_can_drop_bomb + active_bombs)
    enemy_can_drop = float(state_vec[14]) if state_vec.shape[0] > 14 else 0.0
    active_bombs = float(state_vec[8]) if state_vec.shape[0] > 8 else 0.0
    out[5] = clamp01(0.65 * enemy_can_drop + 0.35 * (1.0 - active_bombs))

    # enemy_heading_delta
    if ey >= 0 and ex >= 0 and n_ey >= 0 and n_ex >= 0:
        md = abs(n_ey - ey) + abs(n_ex - ex)
        out[6] = clamp01(min(2.0, float(md)) / 2.0)

    out[7] = compute_item_race_delta(state_map, (sy, sx), (ey, ex))
    out[8] = compute_local_choke_occupancy(state_map, (sy, sx), (ey, ex))

    return out


@dataclass
class PreparedDataset:
    state_maps: np.ndarray
    state_vecs: np.ndarray
    actions: np.ndarray
    action_masks: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    next_maps: np.ndarray
    next_vecs: np.ndarray
    risk_labels: np.ndarray
    policy_tags: np.ndarray
    outcome_ids: np.ndarray
    aux_features: np.ndarray
    sample_weights: np.ndarray
    action_hist: Dict[str, int]
    policy_tag_hist: Dict[str, int]
    outcome_hist: Dict[str, int]


def prepare_dataset(rows: List[dict]) -> PreparedDataset:
    s_maps = []
    s_vecs = []
    actions = []
    action_masks = []
    rewards = []
    dones = []
    n_maps = []
    n_vecs = []
    risk_labels = []
    policy_tags = []
    outcome_ids = []
    aux_features = []
    action_hist: Dict[str, int] = {}
    policy_tag_hist: Dict[str, int] = {}
    outcome_hist: Dict[str, int] = {k: 0 for k in OUTCOME_TO_ID.keys()}

    for item in rows:
        action = int(item.get("action", 0))
        if action < 0 or action >= ACTION_DIM:
            continue

        s_map = normalize_state_map(item)
        s_vec = normalize_state_vec(item)
        n_map = normalize_next_state_map(item)
        n_vec = normalize_next_state_vec(item)
        mask = normalize_action_mask(item, action)

        outcome_tag = normalize_outcome_tag(item)
        outcome_id = OUTCOME_TO_ID[outcome_tag]

        reward = float(item.get("reward", 0.0))
        done = 1.0 if bool(item.get("done", False)) else 0.0
        risk = float(infer_risk_label(item, outcome_tag))
        aux = compute_combat_aux_features(item, s_map, s_vec, n_map)

        s_maps.append(np.transpose(s_map, (2, 0, 1)))
        s_vecs.append(s_vec)
        actions.append(action)
        action_masks.append(mask)
        rewards.append(reward)
        dones.append(done)
        n_maps.append(np.transpose(n_map, (2, 0, 1)))
        n_vecs.append(n_vec)
        risk_labels.append(risk)
        outcome_ids.append(outcome_id)
        aux_features.append(aux)

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
        outcome_hist[outcome_tag] = outcome_hist.get(outcome_tag, 0) + 1

    if not actions:
        raise ValueError("dataset is empty after filtering")

    return PreparedDataset(
        state_maps=np.asarray(s_maps, dtype=np.float32),
        state_vecs=np.asarray(s_vecs, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        action_masks=np.asarray(action_masks, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        next_maps=np.asarray(n_maps, dtype=np.float32),
        next_vecs=np.asarray(n_vecs, dtype=np.float32),
        risk_labels=np.asarray(risk_labels, dtype=np.float32),
        policy_tags=np.asarray(policy_tags, dtype=np.int64),
        outcome_ids=np.asarray(outcome_ids, dtype=np.int64),
        aux_features=np.asarray(aux_features, dtype=np.float32),
        sample_weights=np.ones((len(actions),), dtype=np.float32),
        action_hist=action_hist,
        policy_tag_hist=policy_tag_hist,
        outcome_hist=outcome_hist,
    )


def build_sample_weights(
    actions: np.ndarray,
    dones: np.ndarray,
    outcome_ids: np.ndarray,
    aux_features: np.ndarray,
    terminal_base_weight: float,
    self_kill_weight: float,
    timeout_draw_weight: float,
    successful_trap_weight: float,
) -> np.ndarray:
    w = np.ones((actions.shape[0],), dtype=np.float32)

    done_mask = dones > 0.5
    w[done_mask] *= max(1.0, float(terminal_base_weight))

    self_kill_mask = outcome_ids == OUTCOME_TO_ID["self_kill"]
    draw_mask = outcome_ids == OUTCOME_TO_ID["draw"]
    win_mask = outcome_ids == OUTCOME_TO_ID["win"]

    w[self_kill_mask] *= max(1.0, float(self_kill_weight))
    w[draw_mask & done_mask] *= max(1.0, float(timeout_draw_weight))

    # successful_trap proxy: terminal win and drop_bomb action.
    trap_mask = win_mask & done_mask & (actions == 5)
    w[trap_mask] *= max(1.0, float(successful_trap_weight))

    # Emphasize situations where post-bomb escape is poor.
    poor_escape = aux_features[:, 3] < 0.25
    w[poor_escape & (actions == 5)] *= 1.15

    return np.maximum(w, 1e-6)


class IQLDataset(Dataset):
    def __init__(
        self,
        s_maps: np.ndarray,
        s_vecs: np.ndarray,
        actions: np.ndarray,
        action_masks: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        n_maps: np.ndarray,
        n_vecs: np.ndarray,
        risk_labels: np.ndarray,
        policy_tags: np.ndarray,
        outcome_ids: np.ndarray,
        aux_features: np.ndarray,
        sample_weights: np.ndarray,
    ):
        self.s_maps = s_maps
        self.s_vecs = s_vecs
        self.actions = actions
        self.action_masks = action_masks
        self.rewards = rewards
        self.dones = dones
        self.n_maps = n_maps
        self.n_vecs = n_vecs
        self.risk_labels = risk_labels
        self.policy_tags = policy_tags
        self.outcome_ids = outcome_ids
        self.aux_features = aux_features
        self.sample_weights = sample_weights

    def __len__(self) -> int:
        return self.actions.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.s_maps[idx]),
            torch.from_numpy(self.s_vecs[idx]),
            torch.tensor(self.actions[idx], dtype=torch.long),
            torch.from_numpy(self.action_masks[idx]),
            torch.tensor(self.rewards[idx], dtype=torch.float32),
            torch.tensor(self.dones[idx], dtype=torch.float32),
            torch.from_numpy(self.n_maps[idx]),
            torch.from_numpy(self.n_vecs[idx]),
            torch.tensor(self.risk_labels[idx], dtype=torch.float32),
            torch.tensor(self.policy_tags[idx], dtype=torch.long),
            torch.tensor(self.outcome_ids[idx], dtype=torch.long),
            torch.from_numpy(self.aux_features[idx]),
            torch.tensor(self.sample_weights[idx], dtype=torch.float32),
        )


class IQLCombatNet(nn.Module):
    def __init__(self, gru_hidden: int = 256, use_gru: bool = True) -> None:
        super().__init__()
        self.use_gru = use_gru
        self.gru_hidden = int(gru_hidden)

        self.conv1 = nn.Conv2d(TARGET_CHANNELS, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc_in = nn.Linear(64 * ROWS * COLS + TARGET_VEC_DIM, 512)

        if self.use_gru:
            self.gru = nn.GRU(input_size=512, hidden_size=self.gru_hidden, num_layers=1, batch_first=True)
            self.fc = nn.Linear(self.gru_hidden, 256)
        else:
            self.fc = nn.Linear(512, 256)

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
        x = F.relu(self.fc_in(x))
        if self.use_gru:
            seq = x.unsqueeze(1)
            seq_out, _ = self.gru(seq)
            x = seq_out[:, -1, :]
        return F.relu(self.fc(x))

    def forward(self, state_map: torch.Tensor, state_vec: torch.Tensor):
        z = self.encode(state_map, state_vec)
        return self.q1(z), self.q2(z), self.v(z), self.policy(z), self.risk(z)


class IQLPolicyExport(nn.Module):
    def __init__(self, net: IQLCombatNet) -> None:
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


def expectile_loss_weighted(diff: torch.Tensor, expectile: float, sample_w: torch.Tensor) -> torch.Tensor:
    w = torch.where(diff > 0, torch.full_like(diff, expectile), torch.full_like(diff, 1.0 - expectile))
    return (w * diff.pow(2) * sample_w).mean()


def mask_logits(logits: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
    return torch.where(action_masks > 0.5, logits, neg_inf)


def evaluate_policy(
    model: IQLCombatNet,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float, float, float]:
    model.eval()
    total = 0
    correct = 0
    risk_total = 0
    risk_correct = 0
    illegal_pred = 0
    action5_total = 0
    action5_hit = 0
    conf = np.zeros((ACTION_DIM, ACTION_DIM), dtype=np.int64)

    with torch.no_grad():
        for s_map, s_vec, actions, action_masks, _, _, _, _, risk_labels, _, _, _, _ in loader:
            s_map = s_map.to(device)
            s_vec = s_vec.to(device)
            actions = actions.to(device)
            action_masks = action_masks.to(device)
            risk_labels = risk_labels.to(device)

            _, _, _, policy_logits, risk_logit = model(s_map, s_vec)
            masked_logits = mask_logits(policy_logits, action_masks)
            pred_actions = torch.argmax(masked_logits, dim=1)

            correct += int((pred_actions == actions).sum().item())
            total += int(actions.shape[0])
            for t, p in zip(actions.cpu().numpy(), pred_actions.cpu().numpy()):
                conf[int(t), int(p)] += 1
                if int(t) == 5:
                    action5_total += 1
                    if int(p) == 5:
                        action5_hit += 1

            illegal = action_masks.gather(1, pred_actions.unsqueeze(1)).squeeze(1) <= 0.5
            illegal_pred += int(illegal.sum().item())

            pred_risk = (torch.sigmoid(risk_logit.squeeze(1)) >= 0.5).float()
            risk_correct += int((pred_risk == risk_labels).sum().item())
            risk_total += int(risk_labels.shape[0])

    acc = correct / max(1, total)
    f1 = macro_f1_from_confusion(conf)
    risk_acc = risk_correct / max(1, risk_total)
    illegal_rate = illegal_pred / max(1, total)
    action5_recall = action5_hit / max(1, action5_total)
    return acc, f1, risk_acc, illegal_rate, action5_recall


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * sp.data)


def build_balanced_sampler(
    actions: np.ndarray,
    policy_tags: np.ndarray,
    outcome_ids: np.ndarray,
    base_weights: np.ndarray,
    action_power: float,
    policy_power: float,
    outcome_power: float,
) -> WeightedRandomSampler:
    action_counts = np.bincount(actions, minlength=ACTION_DIM).astype(np.float64)
    policy_counts = np.bincount(policy_tags, minlength=3).astype(np.float64)
    outcome_counts = np.bincount(outcome_ids, minlength=len(OUTCOME_TO_ID)).astype(np.float64)
    action_counts[action_counts <= 0] = 1.0
    policy_counts[policy_counts <= 0] = 1.0
    outcome_counts[outcome_counts <= 0] = 1.0

    action_weights = np.power(action_counts, -max(0.0, action_power))
    policy_weights = np.power(policy_counts, -max(0.0, policy_power))
    outcome_weights = np.power(outcome_counts, -max(0.0, outcome_power))

    sample_weights = (
        action_weights[actions]
        * policy_weights[policy_tags]
        * outcome_weights[outcome_ids]
        * np.maximum(base_weights.astype(np.float64), 1e-8)
    )
    sample_weights = np.maximum(sample_weights, 1e-8)
    weights_tensor = torch.from_numpy(sample_weights.astype(np.float64))
    return WeightedRandomSampler(weights=weights_tensor, num_samples=len(actions), replacement=True)


def load_conv_weights_from_init(model: IQLCombatNet, init_pt: str) -> List[str]:
    if not init_pt or not os.path.exists(init_pt):
        return []
    ckpt = torch.load(init_pt, map_location="cpu")
    src_state = ckpt.get("model_state_dict", ckpt)
    dst_state = model.state_dict()
    copied = []
    for key in ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"]:
        if key in src_state and key in dst_state and tuple(src_state[key].shape) == tuple(dst_state[key].shape):
            dst_state[key] = src_state[key].detach().clone()
            copied.append(key)
    if copied:
        model.load_state_dict(dst_state)
    return copied


def set_conv_trainable(model: IQLCombatNet, trainable: bool) -> None:
    for p in model.conv1.parameters():
        p.requires_grad = trainable
    for p in model.conv2.parameters():
        p.requires_grad = trainable


def default_metrics_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"output/ml/reports/combat_phase0_train_{ts}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train discrete combat IQL model (Phase 0).")
    parser.add_argument("--dataset", default="output/ml/datasets/combat_phase0_v1.jsonl")
    parser.add_argument("--epochs", type=int, default=60)
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
    parser.add_argument("--outcome-balance-power", type=float, default=0.8)
    parser.add_argument("--terminal-base-weight", type=float, default=1.2)
    parser.add_argument("--self-kill-weight", type=float, default=2.4)
    parser.add_argument("--timeout-draw-weight", type=float, default=1.6)
    parser.add_argument("--successful-trap-weight", type=float, default=1.5)
    parser.add_argument("--init-pt", default="")
    parser.add_argument("--freeze-conv-epochs", type=int, default=50)
    parser.add_argument("--gru-hidden", type=int, default=256)
    parser.add_argument("--disable-gru", action="store_true")
    parser.add_argument("--out-pt", default="output/ml/models/combat_phase0_iql_v1.pt")
    parser.add_argument("--out-onnx", default="output/ml/models/combat_phase0_iql_v1.onnx")
    parser.add_argument("--out-metrics", default=default_metrics_path())
    args = parser.parse_args()

    set_seed(args.seed)

    rows = read_jsonl(args.dataset)
    data = prepare_dataset(rows)
    data.sample_weights = build_sample_weights(
        data.actions,
        data.dones,
        data.outcome_ids,
        data.aux_features,
        terminal_base_weight=args.terminal_base_weight,
        self_kill_weight=args.self_kill_weight,
        timeout_draw_weight=args.timeout_draw_weight,
        successful_trap_weight=args.successful_trap_weight,
    )

    n = data.actions.shape[0]
    train_idx, val_idx = split_indices(n, args.val_ratio, args.seed)

    train_ds = IQLDataset(
        data.state_maps[train_idx],
        data.state_vecs[train_idx],
        data.actions[train_idx],
        data.action_masks[train_idx],
        data.rewards[train_idx],
        data.dones[train_idx],
        data.next_maps[train_idx],
        data.next_vecs[train_idx],
        data.risk_labels[train_idx],
        data.policy_tags[train_idx],
        data.outcome_ids[train_idx],
        data.aux_features[train_idx],
        data.sample_weights[train_idx],
    )
    val_ds = IQLDataset(
        data.state_maps[val_idx],
        data.state_vecs[val_idx],
        data.actions[val_idx],
        data.action_masks[val_idx],
        data.rewards[val_idx],
        data.dones[val_idx],
        data.next_maps[val_idx],
        data.next_vecs[val_idx],
        data.risk_labels[val_idx],
        data.policy_tags[val_idx],
        data.outcome_ids[val_idx],
        data.aux_features[val_idx],
        data.sample_weights[val_idx],
    )

    if args.sampler == "balanced":
        train_sampler = build_balanced_sampler(
            data.actions[train_idx],
            data.policy_tags[train_idx],
            data.outcome_ids[train_idx],
            data.sample_weights[train_idx],
            args.action_balance_power,
            args.policy_balance_power,
            args.outcome_balance_power,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, shuffle=False, drop_last=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IQLCombatNet(gru_hidden=args.gru_hidden, use_gru=not args.disable_gru).to(device)
    target_model = IQLCombatNet(gru_hidden=args.gru_hidden, use_gru=not args.disable_gru).to(device)

    copied = load_conv_weights_from_init(model, args.init_pt)
    target_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    freeze_conv_epochs = max(0, int(args.freeze_conv_epochs))
    conv_frozen = False
    if freeze_conv_epochs > 0:
        set_conv_trainable(model, False)
        conv_frozen = True

    history: List[dict] = []
    best_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        if conv_frozen and epoch > freeze_conv_epochs:
            set_conv_trainable(model, True)
            conv_frozen = False

        model.train()
        total_loss = 0.0
        total_n = 0

        for s_map, s_vec, actions, action_masks, rewards, dones, n_map, n_vec, risk_labels, _, _, _, sample_w in train_loader:
            s_map = s_map.to(device)
            s_vec = s_vec.to(device)
            actions = actions.to(device)
            action_masks = action_masks.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)
            n_map = n_map.to(device)
            n_vec = n_vec.to(device)
            risk_labels = risk_labels.to(device)
            sample_w = sample_w.to(device)
            sample_w = sample_w / torch.clamp(sample_w.mean(), min=1e-6)

            q1, q2, v, policy_logits, risk_logit = model(s_map, s_vec)
            with torch.no_grad():
                _, _, next_v, _, _ = target_model(n_map, n_vec)
                next_v = next_v.squeeze(1)
                q_target = rewards + args.gamma * (1.0 - dones) * next_v

            q1_a = q1.gather(1, actions.unsqueeze(1)).squeeze(1)
            q2_a = q2.gather(1, actions.unsqueeze(1)).squeeze(1)
            q_loss = ((q1_a - q_target).pow(2) * sample_w).mean() + ((q2_a - q_target).pow(2) * sample_w).mean()

            q_min_a = torch.min(q1_a, q2_a).detach()
            v_pred = v.squeeze(1)
            v_loss = expectile_loss_weighted(q_min_a - v_pred, args.expectile, sample_w)

            adv = q_min_a - v_pred.detach()
            exp_adv = torch.exp(args.beta * adv).clamp(max=args.adv_max_weight)
            masked_logits = mask_logits(policy_logits, action_masks)
            logp = F.log_softmax(masked_logits, dim=1).gather(1, actions.unsqueeze(1)).squeeze(1)
            actor_loss = -(exp_adv * logp * sample_w).mean()

            risk_per = F.binary_cross_entropy_with_logits(risk_logit.squeeze(1), risk_labels, reduction="none")
            risk_loss = (risk_per * sample_w).mean()

            loss = q_loss + v_loss + actor_loss + args.risk_loss_weight * risk_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            soft_update(target_model, model, args.target_tau)

            batch_n = int(actions.shape[0])
            total_loss += float(loss.item()) * batch_n
            total_n += batch_n

        train_loss = total_loss / max(1, total_n)
        val_acc, val_f1, val_risk_acc, illegal_rate, action5_recall = evaluate_policy(model, val_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_policy_acc": val_acc,
                "val_policy_f1": val_f1,
                "val_risk_acc": val_risk_acc,
                "illegal_action_pred_rate": illegal_rate,
                "action5_recall": action5_recall,
                "conv_frozen": epoch <= freeze_conv_epochs,
            }
        )

        print(
            "[EPOCH {}] train_loss={:.6f} val_acc={:.4f} val_f1={:.4f} val_risk_acc={:.4f} "
            "illegal_rate={:.5f} action5_recall={:.4f} conv_frozen={}".format(
                epoch,
                train_loss,
                val_acc,
                val_f1,
                val_risk_acc,
                illegal_rate,
                action5_recall,
                epoch <= freeze_conv_epochs,
            ),
            flush=True,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("failed to get best model")

    model.load_state_dict(best_state)

    val_acc, val_f1, val_risk_acc, illegal_rate, action5_recall = evaluate_policy(model, val_loader, device)

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
                "gru_hidden": args.gru_hidden,
                "use_gru": not args.disable_gru,
            },
            "init_conv_copied": copied,
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
    aux_mean = data.aux_features.mean(axis=0).tolist() if data.aux_features.size > 0 else [0.0] * COMBAT_AUX_DIM

    metrics = {
        "dataset_path": args.dataset,
        "dataset_size": int(n),
        "train_size": int(train_idx.shape[0]),
        "val_size": int(val_idx.shape[0]),
        "action_hist": data.action_hist,
        "policy_tag_hist": data.policy_tag_hist,
        "outcome_hist": data.outcome_hist,
        "risk_label_ratio": risk_ratio,
        "sample_weight_mean": float(data.sample_weights.mean()),
        "sample_weight_max": float(data.sample_weights.max()),
        "aux_feature_names": COMBAT_AUX_NAMES,
        "aux_feature_mean": aux_mean,
        "val_policy_acc": val_acc,
        "val_policy_f1": val_f1,
        "val_risk_acc": val_risk_acc,
        "illegal_action_pred_rate": illegal_rate,
        "action5_recall": action5_recall,
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
            "outcome_balance_power": args.outcome_balance_power,
            "terminal_base_weight": args.terminal_base_weight,
            "self_kill_weight": args.self_kill_weight,
            "timeout_draw_weight": args.timeout_draw_weight,
            "successful_trap_weight": args.successful_trap_weight,
            "channels": TARGET_CHANNELS,
            "vec_dim": TARGET_VEC_DIM,
            "action_dim": ACTION_DIM,
            "gru_hidden": args.gru_hidden,
            "use_gru": not args.disable_gru,
            "freeze_conv_epochs": freeze_conv_epochs,
            "init_pt": args.init_pt,
            "init_conv_copied": copied,
        },
        "out_pt": args.out_pt,
        "out_onnx": args.out_onnx,
    }

    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("[DONE] metrics:", args.out_metrics)


if __name__ == "__main__":
    main()
