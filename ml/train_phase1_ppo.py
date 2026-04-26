#!/usr/bin/env python3
import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import onnx
import torch
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.data import DataLoader, TensorDataset

from combat_phase1_env import ACTION_DIM, MAP_CH, MAP_COLS, MAP_ROWS, VEC_DIM, CombatPhase1ReplayEnv, load_rollout_episodes


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Phase1 PPO (fixed-opponent rollout replay).")
    p.add_argument("--dataset", required=True)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=6)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=384)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.15)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=20260425)
    p.add_argument("--device", default="cuda")
    p.add_argument("--invalid-action-penalty", type=float, default=-0.10)
    p.add_argument("--mismatch-penalty", type=float, default=-0.02)
    p.add_argument("--match-bonus", type=float, default=0.02)
    p.add_argument("--distill-epochs", type=int, default=8)
    p.add_argument("--distill-batch-size", type=int, default=1024)
    p.add_argument("--distill-lr", type=float, default=3e-4)
    p.add_argument("--distill-samples", type=int, default=180_000)
    p.add_argument("--out-zip", required=True)
    p.add_argument("--out-onnx", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


class DistillCombatNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(MAP_CH, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * MAP_ROWS * MAP_COLS + VEC_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, ACTION_DIM)
        self.risk_head = nn.Linear(256, 1)

    def forward(self, state_map: torch.Tensor, state_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(state_map)
        x = x.reshape(x.shape[0], -1)
        x = torch.cat([x, state_vector], dim=1)
        x = self.fc(x)
        logits = self.policy_head(x)
        risk_logit = self.risk_head(x)
        return logits, risk_logit


@dataclass
class DistillData:
    state_maps: np.ndarray
    state_vecs: np.ndarray
    labels: np.ndarray


def _iter_samples_from_episodes(episodes, max_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    state_maps: List[np.ndarray] = []
    state_vecs: List[np.ndarray] = []
    for ep in episodes:
        for row in ep:
            state_maps.append(np.transpose(row.state_map, (2, 0, 1)).astype(np.float32, copy=False))
            state_vecs.append(row.state_vector.astype(np.float32, copy=False))
            if max_samples > 0 and len(state_maps) >= max_samples:
                return np.stack(state_maps, axis=0), np.stack(state_vecs, axis=0)
    return np.stack(state_maps, axis=0), np.stack(state_vecs, axis=0)


def collect_distill_labels(
    model: RecurrentPPO,
    episodes,
    max_samples: int,
) -> DistillData:
    maps, vecs = _iter_samples_from_episodes(episodes, max_samples=max_samples)
    labels = np.zeros((maps.shape[0],), dtype=np.int64)
    for i in range(maps.shape[0]):
        obs = {
            "state_map": maps[i],
            "state_vector": vecs[i],
            "action_mask": np.ones((ACTION_DIM,), dtype=np.float32),
        }
        act, _ = model.predict(obs, deterministic=True)
        labels[i] = int(act)
    return DistillData(state_maps=maps, state_vecs=vecs, labels=labels)


def train_distill(
    data: DistillData,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> Tuple[DistillCombatNet, List[Dict]]:
    net = DistillCombatNet().to(device)
    ds = TensorDataset(
        torch.from_numpy(data.state_maps),
        torch.from_numpy(data.state_vecs),
        torch.from_numpy(data.labels),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        net.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for sm, sv, y in dl:
            sm = sm.to(device)
            sv = sv.to(device)
            y = y.to(device)
            logits, _ = net(sm, sv)
            loss = ce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * int(y.shape[0])
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.shape[0])
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / max(1, total),
                "acc": correct / max(1, total),
            }
        )
    return net, history


def export_onnx(net: DistillCombatNet, out_onnx: str, device: torch.device) -> None:
    ensure_parent(out_onnx)
    net.eval()
    dummy_map = torch.zeros((1, MAP_CH, MAP_ROWS, MAP_COLS), dtype=torch.float32, device=device)
    dummy_vec = torch.zeros((1, VEC_DIM), dtype=torch.float32, device=device)
    torch.onnx.export(
        net,
        (dummy_map, dummy_vec),
        out_onnx,
        input_names=["state_map", "state_vector"],
        output_names=["policy_logits", "risk_logit"],
        opset_version=18,
        dynamo=False,
    )
    merged = onnx.load(out_onnx, load_external_data=True)
    onnx.save_model(merged, out_onnx, save_as_external_data=False)
    sidecar = out_onnx + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_parent(args.out_zip)
    ensure_parent(args.out_onnx)
    ensure_parent(args.out_report)

    cuda_available = torch.cuda.is_available()
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else ""
    requested_device = str(args.device).strip().lower()
    if requested_device.startswith("cuda") and not cuda_available:
        raise RuntimeError("CUDA device is required for this run, but torch.cuda.is_available() is False.")
    print(
        "[ENV]",
        json.dumps(
            {
                "requested_device": args.device,
                "cuda_available": cuda_available,
                "cuda_device_name": cuda_device_name or None,
            }
        ),
    )

    started = time.time()
    episodes = load_rollout_episodes(args.dataset, max_rows=max(0, int(args.max_rows)))

    def make_env() -> CombatPhase1ReplayEnv:
        return CombatPhase1ReplayEnv(
            episodes,
            invalid_action_penalty=args.invalid_action_penalty,
            mismatch_penalty=args.mismatch_penalty,
            match_bonus=args.match_bonus,
        )

    vec_env = DummyVecEnv([make_env for _ in range(max(1, int(args.n_envs)))])

    model = RecurrentPPO(
        policy="MultiInputLstmPolicy",
        env=vec_env,
        n_steps=max(16, int(args.n_steps)),
        batch_size=max(16, int(args.batch_size)),
        n_epochs=max(1, int(args.n_epochs)),
        learning_rate=float(args.learning_rate),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_range=float(args.clip_range),
        ent_coef=float(args.ent_coef),
        vf_coef=float(args.vf_coef),
        max_grad_norm=float(args.max_grad_norm),
        seed=int(args.seed),
        device=args.device,
        verbose=1,
        policy_kwargs={"net_arch": [256, 256], "lstm_hidden_size": 256},
    )
    model.learn(total_timesteps=max(1, int(args.total_timesteps)), progress_bar=False)
    model.save(args.out_zip)

    distill_data = collect_distill_labels(model, episodes, max_samples=max(1, int(args.distill_samples)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distill_net, distill_hist = train_distill(
        distill_data,
        epochs=max(1, int(args.distill_epochs)),
        batch_size=max(64, int(args.distill_batch_size)),
        lr=float(args.distill_lr),
        device=device,
    )
    export_onnx(distill_net, args.out_onnx, device=device)

    action_hist: Dict[str, int] = {str(i): 0 for i in range(ACTION_DIM)}
    for a in distill_data.labels.tolist():
        action_hist[str(int(a))] = action_hist.get(str(int(a)), 0) + 1

    report = {
        "ts": int(time.time() * 1000),
        "duration_sec": time.time() - started,
        "dataset": args.dataset,
        "episode_count": len(episodes),
        "total_rows": int(sum(len(ep) for ep in episodes)),
        "ppo": {
            "total_timesteps": int(args.total_timesteps),
            "n_envs": int(args.n_envs),
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "n_epochs": int(args.n_epochs),
            "learning_rate": float(args.learning_rate),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "clip_range": float(args.clip_range),
            "ent_coef": float(args.ent_coef),
            "vf_coef": float(args.vf_coef),
            "max_grad_norm": float(args.max_grad_norm),
            "seed": int(args.seed),
            "device": args.device,
            "cuda_available": bool(cuda_available),
            "cuda_device_name": cuda_device_name or None,
        },
        "distill": {
            "samples": int(distill_data.labels.shape[0]),
            "epochs": int(args.distill_epochs),
            "batch_size": int(args.distill_batch_size),
            "lr": float(args.distill_lr),
            "history": distill_hist,
            "teacher_action_hist": action_hist,
        },
        "out_zip": args.out_zip,
        "out_onnx": args.out_onnx,
    }
    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("[DONE]", json.dumps({"out_report": args.out_report, "out_zip": args.out_zip, "out_onnx": args.out_onnx}))


if __name__ == "__main__":
    main()
