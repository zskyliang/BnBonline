#!/usr/bin/env python3
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ACTION_DIM = 6
MAP_ROWS = 13
MAP_COLS = 15
MAP_CH = 10
VEC_DIM = 32


@dataclass
class RolloutStep:
    state_map: np.ndarray
    state_vector: np.ndarray
    action_mask: np.ndarray
    action: int
    reward: float
    done: bool
    outcome_tag: str
    terminal_reason: str
    episode_id: str


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _normalize_state_map(raw_map: object) -> np.ndarray:
    arr = np.asarray(raw_map, dtype=np.float32)
    if arr.ndim != 3:
        out = np.zeros((MAP_ROWS, MAP_COLS, MAP_CH), dtype=np.float32)
        return out
    if arr.shape[0] != MAP_ROWS or arr.shape[1] != MAP_COLS:
        out = np.zeros((MAP_ROWS, MAP_COLS, MAP_CH), dtype=np.float32)
        rows = min(MAP_ROWS, arr.shape[0])
        cols = min(MAP_COLS, arr.shape[1])
        ch = min(MAP_CH, arr.shape[2] if arr.ndim == 3 else 0)
        if rows > 0 and cols > 0 and ch > 0:
            out[:rows, :cols, :ch] = arr[:rows, :cols, :ch]
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(out, -1.0, 1.0)
    if arr.shape[2] < MAP_CH:
        out = np.zeros((MAP_ROWS, MAP_COLS, MAP_CH), dtype=np.float32)
        out[:, :, : arr.shape[2]] = arr
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(out, -1.0, 1.0)
    out = np.nan_to_num(arr[:, :, :MAP_CH], nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(out, -1.0, 1.0)


def _normalize_state_vector(raw_vec: object) -> np.ndarray:
    vec = np.asarray(raw_vec, dtype=np.float32).reshape(-1)
    out = np.zeros((VEC_DIM,), dtype=np.float32)
    if vec.size > 0:
        out[: min(VEC_DIM, vec.size)] = vec[:VEC_DIM]
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(out, -1.0, 1.0)


def _normalize_action_mask(raw_mask: object, action: int) -> np.ndarray:
    mask = np.ones((ACTION_DIM,), dtype=np.float32)
    if isinstance(raw_mask, list) and len(raw_mask) >= ACTION_DIM:
        vals = np.asarray(raw_mask[:ACTION_DIM], dtype=np.float32)
        mask = np.where(vals > 0.5, 1.0, 0.0).astype(np.float32)
    if mask.sum() <= 0:
        mask[:] = 1.0
    if 0 <= action < ACTION_DIM and mask[action] <= 0:
        mask[action] = 1.0
    return np.nan_to_num(mask, nan=1.0, posinf=1.0, neginf=0.0)


def _terminal_bonus(outcome_tag: str, terminal_reason: str) -> float:
    tag = str(outcome_tag or "ongoing").strip().lower()
    reason = str(terminal_reason or "").strip().lower()
    if tag == "win" or reason == "caught_enemy":
        return 1.0
    if tag == "self_kill":
        return -1.2
    if tag == "loss" or reason == "caught_self":
        return -1.0
    if tag == "draw" or reason == "stall_abort":
        return -0.2
    return 0.0


def _to_rollout_step(row: Dict) -> Optional[RolloutStep]:
    state = row.get("state", {})
    action = int(row.get("action", 0))
    if action < 0 or action >= ACTION_DIM:
        return None
    state_map = _normalize_state_map(state.get("state_map"))
    state_vector = _normalize_state_vector(state.get("state_vector"))
    action_mask = _normalize_action_mask(row.get("action_mask"), action)
    outcome_tag = str(row.get("outcome_tag", "ongoing"))
    terminal_reason = str(row.get("terminal_reason", row.get("meta", {}).get("terminalReason", "")))
    reward = float(row.get("reward", 0.0) or 0.0)
    if not np.isfinite(reward):
        reward = 0.0
    if bool(row.get("done", False)):
        reward += _terminal_bonus(outcome_tag, terminal_reason)
    return RolloutStep(
        state_map=state_map,
        state_vector=state_vector,
        action_mask=action_mask,
        action=action,
        reward=reward,
        done=bool(row.get("done", False)),
        outcome_tag=outcome_tag,
        terminal_reason=terminal_reason,
        episode_id=str(row.get("episode_id", "runtime")),
    )


def load_rollout_episodes(dataset_path: str, max_rows: int = 0) -> List[List[RolloutStep]]:
    episodes: Dict[str, List[RolloutStep]] = {}
    count = 0
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            step = _to_rollout_step(raw)
            if step is None:
                continue
            episodes.setdefault(step.episode_id, []).append(step)
            count += 1
            if max_rows > 0 and count >= max_rows:
                break
    out = [steps for steps in episodes.values() if len(steps) > 1]
    out.sort(key=lambda arr: arr[0].episode_id)
    if not out:
        raise ValueError(f"no usable rollout episodes in {dataset_path}")
    return out


class CombatPhase1ReplayEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        episodes: List[List[RolloutStep]],
        *,
        invalid_action_penalty: float = -0.10,
        mismatch_penalty: float = -0.02,
        match_bonus: float = 0.02,
    ) -> None:
        super().__init__()
        self.episodes = episodes
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.mismatch_penalty = float(mismatch_penalty)
        self.match_bonus = float(match_bonus)

        self.action_space = spaces.Discrete(ACTION_DIM)
        self.observation_space = spaces.Dict(
            {
                "state_map": spaces.Box(low=-1.0, high=1.0, shape=(MAP_CH, MAP_ROWS, MAP_COLS), dtype=np.float32),
                "state_vector": spaces.Box(low=-1.0, high=1.0, shape=(VEC_DIM,), dtype=np.float32),
                "action_mask": spaces.Box(low=0.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32),
            }
        )
        self._episode_idx = 0
        self._step_idx = 0
        self._np_random = np.random.default_rng(42)

    def _obs_from_step(self, step: RolloutStep) -> Dict[str, np.ndarray]:
        return {
            "state_map": np.nan_to_num(
                np.transpose(step.state_map, (2, 0, 1)).astype(np.float32, copy=False),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            ),
            "state_vector": np.nan_to_num(
                step.state_vector.astype(np.float32, copy=False),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            ),
            "action_mask": np.nan_to_num(
                step.action_mask.astype(np.float32, copy=False),
                nan=1.0,
                posinf=1.0,
                neginf=0.0,
            ),
        }

    def _pick_episode_index(self) -> int:
        return int(self._np_random.integers(0, len(self.episodes)))

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self._episode_idx = self._pick_episode_index()
        self._step_idx = 0
        step = self.episodes[self._episode_idx][self._step_idx]
        return self._obs_from_step(step), {"episode_id": step.episode_id}

    def step(self, action: int):
        ep = self.episodes[self._episode_idx]
        cur = ep[self._step_idx]
        act = int(action)

        reward = float(cur.reward)
        if act == cur.action:
            reward += self.match_bonus
        else:
            reward += self.mismatch_penalty
        if act < 0 or act >= ACTION_DIM or cur.action_mask[act] <= 0.5:
            reward += self.invalid_action_penalty

        terminated = bool(cur.done or self._step_idx >= (len(ep) - 2))
        truncated = False

        info = {
            "episode_id": cur.episode_id,
            "expert_action": int(cur.action),
            "outcome_tag": cur.outcome_tag,
            "terminal_reason": cur.terminal_reason,
        }

        if terminated:
            next_obs = self._obs_from_step(ep[-1])
        else:
            self._step_idx += 1
            nxt = ep[self._step_idx]
            next_obs = self._obs_from_step(nxt)

        if not np.isfinite(reward):
            reward = 0.0
        return next_obs, float(reward), terminated, truncated, info
