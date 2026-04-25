#!/usr/bin/env node
const fs = require("fs");
const path = require("path");

const ROWS = 13;
const COLS = 15;
const ACTIONS = { wait: 0, up: 1, down: 2, left: 3, right: 4, bomb: 5 };
const SCENARIOS = ["open_random", "escape_after_bomb", "enemy_choke", "item_race", "deadend_chase"];
const BEHAVIOR_KEYS = [
  "item_control",
  "bomb_space_reduction",
  "post_bomb_escape",
  "obstacle_escape",
  "move_to_safe",
  "close_chase",
  "choke_control",
  "deny_space",
  "power_advantage_attack",
  "tactical_wait",
  "terminal_reference",
  "survival_failure_reference",
];

function getArg(name, fallback) {
  const prefix = `--${name}=`;
  for (const arg of process.argv.slice(2)) {
    if (arg.startsWith(prefix)) return arg.slice(prefix.length);
  }
  return fallback;
}

function asInt(v, fallback) {
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? n : fallback;
}

function asFloat(v, fallback) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(v, lo, hi) {
  const n = Number(v);
  if (!Number.isFinite(n)) return lo;
  return Math.max(lo, Math.min(hi, n));
}

function compact(n) {
  return Math.round(clamp(n, -10, 10) * 1000) / 1000;
}

function makeRng(seed) {
  let s = (seed >>> 0) || 1;
  return function rng() {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function choice(rng, arr) {
  return arr[Math.floor(rng() * arr.length)] || arr[0];
}

function ensureParent(file) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
}

function baseWindmillMap() {
  const map = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
  const heart = [
    "...............",
    "..#...###...#..",
    ".....#####.....",
    "....#.....#....",
    "..##.......##..",
    ".###.......###.",
    ".###.......###.",
    "..##.......##..",
    "..###.....###..",
    "....##...##....",
    ".....#####.....",
    "......###......",
    "...............",
  ];
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      if (heart[y][x] === "#") map[y][x] = 3;
    }
  }
  for (let x = 0; x < COLS; x++) {
    map[0][x] = 8;
    map[ROWS - 1][x] = 8;
  }
  for (let y = 0; y < ROWS; y++) {
    map[y][0] = 8;
    map[y][COLS - 1] = 8;
  }
  map[1][1] = 0;
  map[1][2] = 0;
  map[2][1] = 0;
  for (let x = 6; x < 9; x++) map[6][x] = 9;
  return map;
}

function isRigid(no) {
  return no > 0 && no < 100 && no !== 3 && no !== 8;
}

function isWalkable(map, x, y) {
  return x >= 0 && y >= 0 && x < COLS && y < ROWS && (map[y][x] === 0 || map[y][x] > 100);
}

function softCells(map) {
  const out = [];
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      const no = map[y][x];
      if (no > 0 && no < 100 && !isRigid(no) && no !== 8) out.push({ x, y, no });
    }
  }
  return out;
}

function shuffleInPlace(rng, arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function randomizeMap(rng, opts) {
  const map = baseWindmillMap();
  const cells = shuffleInPlace(rng, softCells(map));
  const keepMin = Math.max(0, asInt(opts.softObstacleKeepMin, 0));
  const keepMax = Math.max(keepMin, asInt(opts.softObstacleKeepMax, 20));
  const keep = Math.min(cells.length, keepMin + Math.floor(rng() * (keepMax - keepMin + 1)));
  const keepSet = new Set(cells.slice(0, keep).map((c) => `${c.x},${c.y}`));
  let cleared = 0;
  let items = 0;
  for (const c of cells) {
    if (keepSet.has(`${c.x},${c.y}`)) {
      map[c.y][c.x] = c.no;
    } else {
      map[c.y][c.x] = 0;
      cleared++;
      if (rng() < opts.itemDensity) {
        map[c.y][c.x] = choice(rng, [101, 102, 103]);
        items++;
      }
    }
  }
  return { map, softKept: keep, softCleared: cleared, itemsFromCleared: items };
}

function bfs(map, start, goal) {
  const q = [{ x: start.x, y: start.y, d: 0 }];
  const seen = new Set([`${start.x},${start.y}`]);
  for (let head = 0; head < q.length; head++) {
    const cur = q[head];
    if (goal && cur.x === goal.x && cur.y === goal.y) return cur.d;
    for (const [dx, dy] of [[0, -1], [0, 1], [-1, 0], [1, 0]]) {
      const nx = cur.x + dx;
      const ny = cur.y + dy;
      const key = `${nx},${ny}`;
      if (seen.has(key) || !isWalkable(map, nx, ny)) continue;
      seen.add(key);
      q.push({ x: nx, y: ny, d: cur.d + 1 });
    }
  }
  return goal ? 999 : q.map((c) => ({ x: c.x, y: c.y, d: c.d }));
}

function walkableCells(map) {
  const out = [];
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) if (isWalkable(map, x, y)) out.push({ x, y });
  }
  return out;
}

function pickSpawnPair(rng, map, minDist, maxDist, bucketIndex) {
  const cells = walkableCells(map);
  const buckets = [[1, 3], [4, 6], [7, 10]];
  const want = buckets[bucketIndex % buckets.length] || buckets[0];
  let best = null;
  let bestScore = -1e9;
  for (let i = 0; i < Math.max(160, cells.length * 3); i++) {
    const a = choice(rng, cells);
    const b = choice(rng, cells);
    if (!a || !b || (a.x === b.x && a.y === b.y)) continue;
    const d = bfs(map, a, b);
    if (d >= 999) continue;
    const inRange = d >= minDist && d <= maxDist;
    const inBucket = d >= Math.max(want[0], minDist) && d <= Math.min(want[1], maxDist);
    const score = (inBucket ? 100 : inRange ? 60 : -Math.abs(d - maxDist)) + rng();
    if (score > bestScore) {
      bestScore = score;
      best = { self: a, enemy: b, dist: d };
    }
  }
  return best || { self: cells[0], enemy: cells[Math.min(1, cells.length - 1)], dist: cells.length > 1 ? bfs(map, cells[0], cells[1]) : 999 };
}

function countSafeNeighbors(map, pos, danger) {
  let n = 0;
  for (const [dx, dy] of [[0, -1], [0, 1], [-1, 0], [1, 0]]) {
    const x = pos.x + dx, y = pos.y + dy;
    if (isWalkable(map, x, y) && !danger.has(`${x},${y}`)) n++;
  }
  return n;
}

function dangerLine(map, origin, power) {
  const danger = new Set([`${origin.x},${origin.y}`]);
  for (const [dx, dy] of [[0, -1], [0, 1], [-1, 0], [1, 0]]) {
    for (let i = 1; i <= power; i++) {
      const x = origin.x + dx * i, y = origin.y + dy * i;
      if (x < 0 || y < 0 || x >= COLS || y >= ROWS) break;
      const no = map[y][x];
      if (no > 0 && no < 100) {
        danger.add(`${x},${y}`);
        break;
      }
      danger.add(`${x},${y}`);
    }
  }
  return danger;
}

function nearestItemDistance(map, pos) {
  const q = bfs(map, pos, null);
  let best = 999;
  for (const c of q) {
    if (map[c.y] && map[c.y][c.x] > 100) best = Math.min(best, c.d);
  }
  return best;
}

function obstacleValue(no) {
  if (no === 0 || no > 100) return 0;
  if (no === 3 || no === 8 || no === 100) return 0.5;
  return 1;
}

function itemValue(no) {
  if (no === 101) return 1.0;
  if (no === 102) return 0.9;
  if (no === 103) return 0.95;
  return 0;
}

function buildStateMap(map, self, enemy, bomb, danger) {
  const sm = Array.from({ length: ROWS }, () => Array.from({ length: COLS }, () => Array(10).fill(0)));
  const reachable = new Set(bfs(map, self, null).map((c) => `${c.x},${c.y}`));
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      const key = `${x},${y}`;
      sm[y][x][0] = obstacleValue(map[y][x]);
      sm[y][x][2] = danger.has(key) ? 0.65 : 0;
      sm[y][x][4] = countSafeNeighbors(map, { x, y }, danger) >= 3 ? 1 : 0;
      sm[y][x][5] = danger.has(key) ? 0.55 : 0;
      sm[y][x][6] = reachable.has(key) ? 1 : 0;
      sm[y][x][9] = itemValue(map[y][x]);
    }
  }
  if (bomb) sm[bomb.y][bomb.x][1] = 0.45;
  sm[self.y][self.x][3] = 1;
  sm[enemy.y][enemy.x][8] = 1;
  return sm;
}

function shortestSafeEta(map, self, danger) {
  const q = bfs(map, self, null);
  let best = 999;
  for (const c of q) {
    if (!danger.has(`${c.x},${c.y}`)) {
      best = c.d;
      break;
    }
  }
  return best >= 999 ? 1 : clamp(best / 10, 0, 1);
}

function deadendDepth(map, pos, maxDepth) {
  let best = 0;
  for (const [dx, dy] of [[0, -1], [0, 1], [-1, 0], [1, 0]]) {
    let depth = 0;
    for (let i = 1; i <= maxDepth; i++) {
      const p = { x: pos.x + dx * i, y: pos.y + dy * i };
      if (!isWalkable(map, p.x, p.y)) break;
      depth = i;
      let exits = 0;
      for (const [ex, ey] of [[0, -1], [0, 1], [-1, 0], [1, 0]]) {
        if (isWalkable(map, p.x + ex, p.y + ey)) exits++;
      }
      if (exits >= 3) break;
    }
    best = Math.max(best, depth);
  }
  return clamp(best / maxDepth, 0, 1);
}

function makeProps(rng) {
  const cap = 1 + Math.floor(rng() * 5);
  return {
    cap,
    active: Math.floor(rng() * Math.min(3, cap + 1)),
    power: 1 + Math.floor(rng() * 6),
    speed: 3 + Math.floor(rng() * 6),
  };
}

function buildVector(opts) {
  const { map, self, enemy, selfProps, enemyProps, spawnDist, danger, bomb, scenario } = opts;
  const manhattan = Math.abs(self.x - enemy.x) + Math.abs(self.y - enemy.y);
  const pathDist = bfs(map, self, enemy);
  const safeN = countSafeNeighbors(map, self, danger);
  const enemyDanger = bomb ? dangerLine(map, self, selfProps.power) : new Set();
  const enemyEsc = countSafeNeighbors(map, enemy, enemyDanger) / 4;
  const trapClosure = clamp(1 - enemyEsc + (scenario === "enemy_choke" || scenario === "deadend_chase" ? 0.2 : 0), 0, 1);
  const selfItem = nearestItemDistance(map, self);
  const enemyItem = nearestItemDistance(map, enemy);
  const itemRace = clamp((enemyItem - selfItem + 10) / 20, 0, 1);
  const powerGap = clamp((selfProps.cap + selfProps.power + selfProps.speed - enemyProps.cap - enemyProps.power - enemyProps.speed + 17) / 34, 0, 1);
  const postEscape = shortestSafeEta(map, self, danger);
  const blastOverlap = danger.has(`${self.x},${self.y}`) ? 0.8 : (safeN <= 1 ? 0.35 : 0.05);
  return [
    0, 0,
    danger.has(`${self.x},${self.y}`) ? 1 : 0,
    danger.has(`${self.x},${self.y}`) ? 1 : 0,
    blastOverlap,
    blastOverlap,
    blastOverlap,
    safeN / 4,
    bomb ? 0.1 : 0,
    selfProps.active < selfProps.cap ? 1 : 0,
    clamp((selfProps.cap - selfProps.active) / 5, 0, 1),
    clamp(selfProps.power / 8, 0, 1),
    clamp(selfProps.speed / 10, 0, 1),
    clamp(manhattan / (ROWS + COLS), 0, 1),
    enemyProps.active < enemyProps.cap ? 1 : 0,
    clamp(1 / Math.max(1, manhattan), 0, 1),
    clamp((4 - safeN) / 4, 0, 1),
    postEscape,
    deadendDepth(map, self, 8),
    blastOverlap,
    enemyEsc,
    trapClosure,
    itemRace,
    powerGap,
    clamp(selfProps.cap / 5, 0, 1),
    clamp(selfProps.active / 5, 0, 1),
    clamp(enemyProps.cap / 5, 0, 1),
    clamp(enemyProps.active / 5, 0, 1),
    clamp(enemyProps.power / 8, 0, 1),
    clamp(enemyProps.speed / 10, 0, 1),
    clamp(pathDist / (ROWS + COLS), 0, 1),
    clamp(spawnDist / (ROWS + COLS), 0, 1),
  ].map(compact);
}

function legalMask(map, self, selfProps, danger) {
  const mask = [1, 0, 0, 0, 0, 0];
  const dirs = [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]];
  for (let a = 1; a <= 4; a++) {
    const nx = self.x + dirs[a][0], ny = self.y + dirs[a][1];
    if (isWalkable(map, nx, ny)) mask[a] = 1;
  }
  if (selfProps.active < selfProps.cap && shortestSafeEta(map, self, dangerLine(map, self, selfProps.power)) < 0.95) mask[5] = 1;
  return mask;
}

function bestMoveToward(map, from, to, danger) {
  let best = 0;
  let bestD = bfs(map, from, to);
  const dirs = [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]];
  for (let a = 1; a <= 4; a++) {
    const p = { x: from.x + dirs[a][0], y: from.y + dirs[a][1] };
    if (!isWalkable(map, p.x, p.y) || danger.has(`${p.x},${p.y}`)) continue;
    const d = bfs(map, p, to);
    if (d < bestD) {
      bestD = d;
      best = a;
    }
  }
  return best;
}

function bestMoveToSafety(map, from, danger) {
  const dirs = [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]];
  let best = 0;
  let bestScore = danger.has(`${from.x},${from.y}`) ? -10 : 0;
  for (let a = 1; a <= 4; a++) {
    const p = { x: from.x + dirs[a][0], y: from.y + dirs[a][1] };
    if (!isWalkable(map, p.x, p.y)) continue;
    const score = (danger.has(`${p.x},${p.y}`) ? -4 : 4) + countSafeNeighbors(map, p, danger);
    if (score > bestScore) {
      bestScore = score;
      best = a;
    }
  }
  return best;
}

function movePos(map, pos, action) {
  const dirs = [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]];
  const d = dirs[action] || dirs[0];
  const next = { x: pos.x + d[0], y: pos.y + d[1] };
  return isWalkable(map, next.x, next.y) ? next : pos;
}

function behaviorBand(score) {
  if (score >= 0.70) return "very_high";
  if (score >= 0.45) return "high";
  if (score >= 0.18) return "mid";
  if (score > 0) return "low";
  return "zero";
}

function makeRow(rng, index, opts) {
  let env;
  let pair;
  for (let tries = 0; tries < 20; tries++) {
    env = randomizeMap(rng, opts);
    pair = pickSpawnPair(rng, env.map, opts.spawnMin, opts.spawnMax, index);
    if (pair && pair.dist < 999) break;
  }
  const scenario = choice(rng, opts.scenarios);
  let self = { ...pair.self };
  let enemy = { ...pair.enemy };
  const selfProps = makeProps(rng);
  const enemyProps = makeProps(rng);
  let bomb = null;
  let danger = new Set();
  if (scenario === "escape_after_bomb") {
    bomb = { ...self };
    danger = dangerLine(env.map, bomb, selfProps.power);
  } else if (scenario === "enemy_choke" || scenario === "deadend_chase") {
    if (bfs(env.map, self, enemy) <= 5 && rng() < 0.75) {
      bomb = { ...self };
      danger = dangerLine(env.map, bomb, selfProps.power);
    }
  } else if (rng() < 0.25) {
    bomb = { ...enemy };
    danger = dangerLine(env.map, bomb, enemyProps.power);
  }

  const spawnDist = pair.dist;
  let action = 0;
  const mask = legalMask(env.map, self, selfProps, danger);
  const enemyEscIfBomb = countSafeNeighbors(env.map, enemy, dangerLine(env.map, self, selfProps.power));
  if (mask[5] && (scenario === "enemy_choke" || scenario === "deadend_chase") && enemyEscIfBomb <= 2 && rng() < 0.82) {
    action = 5;
  } else if (scenario === "escape_after_bomb" || danger.has(`${self.x},${self.y}`)) {
    action = bestMoveToSafety(env.map, self, danger);
  } else if (scenario === "item_race") {
    const cells = walkableCells(env.map).filter((c) => env.map[c.y][c.x] > 100);
    action = cells.length ? bestMoveToward(env.map, self, choice(rng, cells), danger) : bestMoveToward(env.map, self, enemy, danger);
  } else {
    action = bestMoveToward(env.map, self, enemy, danger);
    if (mask[5] && rng() < 0.22) action = 5;
  }
  if (!mask[action]) action = mask.findIndex((v) => v === 1);
  if (action < 0) action = 0;

  const close = clamp(1 - spawnDist / 10, 0, 1);
  const bombThreat = action === 5 ? clamp(0.25 + (4 - enemyEscIfBomb) * 0.16 + close * 0.2, 0, 1) : (bomb ? 0.18 : 0.08);
  let selfTrapRisk = action === 5 && shortestSafeEta(env.map, self, dangerLine(env.map, self, selfProps.power)) > 0.65 ? 0.75 : 0.15;
  // Keep an explicit slice of bad-bomb pressure samples; natural dead ends are too sparse in the mostly-cleared map.
  if (action === 5 && (scenario === "deadend_chase" || scenario === "enemy_choke") && rng() < 0.18) {
    selfTrapRisk = Math.max(selfTrapRisk, 0.72);
  }
  if (action === 5 && countSafeNeighbors(env.map, self, dangerLine(env.map, self, selfProps.power)) <= 1 && rng() < 0.35) {
    selfTrapRisk = Math.max(selfTrapRisk, 0.78);
  }
  const doneRoll = rng();
  const done = doneRoll < 0.08;
  const selfKill = done && selfTrapRisk > 0.55 && rng() < 0.78;
  const win = done && !selfKill && (action === 5 || bombThreat >= 0.35);
  const outcome = done ? (selfKill ? "self_kill" : (win ? "win" : "loss")) : "ongoing";
  const terminalReason = done ? (selfKill ? "caught_self" : (win ? "caught_enemy" : "caught_self")) : "";
  const breakdown = {};
  if (scenario === "item_race") breakdown.item_control = compact(0.18 + rng() * 0.15);
  if (action === 5) {
    breakdown.bomb_space_reduction = compact(clamp(0.2 + bombThreat * 0.35, 0.2, 0.55));
    if (selfTrapRisk < 0.55) breakdown.post_bomb_escape = compact(0.2 + rng() * 0.18);
  }
  if (danger.size && action !== 5) breakdown.obstacle_escape = compact(0.16 + rng() * 0.16);
  if (action >= 1 && action <= 4 && !danger.has(`${movePos(env.map, self, action).x},${movePos(env.map, self, action).y}`)) breakdown.move_to_safe = compact(0.12 + rng() * 0.15);
  if (scenario === "deadend_chase" || (action !== 0 && close > 0.45)) breakdown.close_chase = compact(0.12 + close * 0.16);
  if (scenario === "enemy_choke" || enemyEscIfBomb <= 1) breakdown.choke_control = compact(0.22 + (4 - enemyEscIfBomb) * 0.06);
  if (bombThreat >= 0.25) breakdown.deny_space = compact(0.12 + bombThreat * 0.12);
  if (selfProps.cap + selfProps.power + selfProps.speed > enemyProps.cap + enemyProps.power + enemyProps.speed + 2) breakdown.power_advantage_attack = compact(0.12 + rng() * 0.1);
  if (action === 0 && !danger.has(`${self.x},${self.y}`)) breakdown.tactical_wait = compact(0.05 + rng() * 0.08);
  if (done) breakdown.terminal_reference = win ? 0.25 : 0.18;
  if (selfKill || selfTrapRisk > 0.55) breakdown.survival_failure_reference = 0.22;

  let score = Object.values(breakdown).reduce((a, b) => a + Number(b || 0), 0);
  score = clamp(score, 0, 1);
  const state = { state_map: buildStateMap(env.map, self, enemy, bomb, danger), state_vector: buildVector({ map: env.map, self, enemy, selfProps, enemyProps, spawnDist, danger, bomb, scenario }) };
  const nextSelf = action === 5 ? self : movePos(env.map, self, action);
  const nextDanger = action === 5 ? dangerLine(env.map, self, selfProps.power) : danger;
  const nextState = { state_map: buildStateMap(env.map, nextSelf, enemy, action === 5 ? self : bomb, nextDanger), state_vector: buildVector({ map: env.map, self: nextSelf, enemy, selfProps: { ...selfProps, active: action === 5 ? Math.min(selfProps.cap, selfProps.active + 1) : selfProps.active }, enemyProps, spawnDist, danger: nextDanger, bomb: action === 5 ? self : bomb, scenario }) };
  const sampleBucket = done ? "terminal" : (action === 5 ? (selfTrapRisk >= 0.55 ? "drop_bomb_bad" : "drop_bomb_safe") : (selfTrapRisk >= 0.55 ? "pre_death" : "ongoing"));
  const auxLabels = {
    terminal_reason: terminalReason,
    my_bomb_threat_score: compact(bombThreat),
    close_range_duel_score: compact(close),
    enemy_self_kill_episode: 0,
    stall_abort_episode: 0,
    winning_bomb_source_recent: win && action === 5 ? 1 : 0,
    bomb_escape_success_label: action === 5 && selfTrapRisk < 0.55 ? 1 : 0,
    bomb_self_trap_risk: compact(selfTrapRisk),
    enemy_trap_after_bomb: compact(clamp(1 - enemyEscIfBomb / 4, 0, 1)),
    nearest_safe_tile_eta: compact(shortestSafeEta(env.map, self, danger)),
    commitment_depth: compact(deadendDepth(env.map, self, 8)),
    terminal_credit_action: done ? 1 : 0,
    behavior_score: compact(score),
    behavior_score_band: behaviorBand(score),
    behavior_high_value: score >= 0.35 ? 1 : 0,
    behavior_failure_reference: selfKill || selfTrapRisk >= 0.55 ? 1 : 0,
    behavior_score_breakdown: breakdown,
  };
  return {
    id: `O${opts.seed}_${index}_${Math.floor(rng() * 1e9)}`,
    ts: Date.now(),
    state,
    action,
    action_mask: mask,
    reward: done ? (win ? 1.2 : selfKill ? -1.5 : -1.0) : compact(0.02 + score * 0.25),
    done,
    next_state: nextState,
    pre_death: selfKill || sampleBucket === "pre_death",
    risk_label: selfKill || selfTrapRisk >= 0.55 ? 1 : 0,
    policy_tag: `offline_micro_${scenario}`,
    episode_id: `offline_${opts.seed}_${Math.floor(index / 16)}`,
    agent_id: "offline_agent",
    opponent_id: "offline_opponent",
    outcome_tag: outcome,
    terminal_reason: terminalReason,
    sample_bucket: sampleBucket,
    aux_labels: auxLabels,
    meta: {
      action_source: "offline_micro_scenario",
      offline_micro_sample: 1,
      scenarioName: scenario,
      perspective: "agent",
      spawnShortestPathDist: spawnDist,
      spawnShortestPathDistNorm: compact(spawnDist / (ROWS + COLS)),
      closeRangeDuelScore: compact(close),
      myBombThreatScore: compact(bombThreat),
      property_bucket_score: compact((selfProps.cap + selfProps.power + selfProps.speed) / 19),
      soft_obstacles_kept: env.softKept,
      soft_obstacles_cleared: env.softCleared,
      items_from_cleared: env.itemsFromCleared,
      self_total_bubble_cap: selfProps.cap,
      self_active_bubble_count: selfProps.active,
      self_power: selfProps.power,
      self_speed: selfProps.speed,
      enemy_total_bubble_cap: enemyProps.cap,
      enemy_active_bubble_count: enemyProps.active,
      enemy_power: enemyProps.power,
      enemy_speed: enemyProps.speed,
    },
  };
}

function updateStats(stats, row) {
  stats.rows_written++;
  stats.action_hist_written[row.action] = (stats.action_hist_written[row.action] || 0) + 1;
  stats.done_hist_written[row.done ? 1 : 0] = (stats.done_hist_written[row.done ? 1 : 0] || 0) + 1;
  stats.pre_death_hist_written[row.pre_death ? 1 : 0] = (stats.pre_death_hist_written[row.pre_death ? 1 : 0] || 0) + 1;
  stats.outcome_hist_written[row.outcome_tag] = (stats.outcome_hist_written[row.outcome_tag] || 0) + 1;
  stats.bucket_hist_written[row.sample_bucket] = (stats.bucket_hist_written[row.sample_bucket] || 0) + 1;
  if (row.terminal_reason) stats.terminal_reason_hist_written[row.terminal_reason] = (stats.terminal_reason_hist_written[row.terminal_reason] || 0) + 1;
  const d = Number(row.meta.spawnShortestPathDist);
  if (d >= 1 && d <= 3) stats.spawn_dist_hist_written["1_3"]++;
  else if (d >= 4 && d <= 6) stats.spawn_dist_hist_written["4_6"]++;
  else if (d >= 7 && d <= 10) stats.spawn_dist_hist_written["7_10"]++;
  else stats.spawn_dist_hist_written.other++;
  const p = Number(row.meta.property_bucket_score || 0);
  if (p < 0.34) stats.property_bucket_hist_written.low++;
  else if (p < 0.67) stats.property_bucket_hist_written.medium++;
  else stats.property_bucket_hist_written.high++;
  const aux = row.aux_labels || {};
  if (aux.bomb_escape_success_label > 0.5) stats.aux_label_hist_written.bomb_escape_success_label++;
  if (aux.bomb_self_trap_risk >= 0.55) stats.aux_label_hist_written.bomb_self_trap_risk_high++;
  if (aux.enemy_trap_after_bomb >= 0.55) stats.aux_label_hist_written.enemy_trap_after_bomb_high++;
  if (aux.terminal_credit_action > 0.5) stats.aux_label_hist_written.terminal_credit_action++;
  const score = Number(aux.behavior_score || 0);
  stats.behavior_score_sum += score;
  stats.behavior_score_hist_written[behaviorBand(score)] = (stats.behavior_score_hist_written[behaviorBand(score)] || 0) + 1;
  if (score >= 0.35) stats.high_value_behavior_rows++;
  const breakdown = aux.behavior_score_breakdown || {};
  for (const key of BEHAVIOR_KEYS) if (Number(breakdown[key] || 0) > 0) stats.behavior_score_breakdown_hist_written[key]++;
}

function makeEmptyStats() {
  return {
    rows_written: 0,
    action_hist_written: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 },
    done_hist_written: { 0: 0, 1: 0 },
    pre_death_hist_written: { 0: 0, 1: 0 },
    outcome_hist_written: { ongoing: 0, win: 0, loss: 0, draw: 0, self_kill: 0 },
    bucket_hist_written: { ongoing: 0, pre_death: 0, drop_bomb_safe: 0, drop_bomb_bad: 0, terminal: 0 },
    terminal_reason_hist_written: { caught_enemy: 0, caught_self: 0, enemy_self_kill_discard: 0, stall_abort: 0 },
    spawn_dist_hist_written: { "1_3": 0, "4_6": 0, "7_10": 0, other: 0 },
    property_bucket_hist_written: { low: 0, medium: 0, high: 0 },
    aux_label_hist_written: { bomb_escape_success_label: 0, bomb_self_trap_risk_high: 0, enemy_trap_after_bomb_high: 0, terminal_credit_action: 0 },
    behavior_score_hist_written: { zero: 0, low: 0, mid: 0, high: 0, very_high: 0 },
    behavior_score_breakdown_hist_written: BEHAVIOR_KEYS.reduce((acc, k) => { acc[k] = 0; return acc; }, {}),
    high_value_behavior_rows: 0,
    behavior_score_sum: 0,
  };
}

function main() {
  const started = Date.now();
  const targetFrames = Math.max(1, asInt(getArg("target-frames", "30000"), 30000));
  const datasetPath = path.resolve(getArg("dataset-path", `output/ml/datasets/combat_phase0_offline_micro_${Date.now()}.jsonl`));
  const reportPath = path.resolve(getArg("report-path", `output/ml/reports/combat_phase0_offline_micro_${Date.now()}.json`));
  const seed = asInt(getArg("seed-base", "20260423"), 20260423) >>> 0;
  const rng = makeRng(seed);
  const opts = {
    seed,
    scenarios: String(getArg("scenario-buckets", SCENARIOS.join(","))).split(",").map((s) => s.trim()).filter(Boolean),
    softObstacleKeepMin: asInt(getArg("soft-obstacle-keep-min", "0"), 0),
    softObstacleKeepMax: asInt(getArg("soft-obstacle-keep-max", "20"), 20),
    itemDensity: clamp(asFloat(getArg("random-item-density", "0.18"), 0.18), 0, 0.8),
    spawnMin: Math.max(1, asInt(getArg("spawn-shortest-path-min", "1"), 1)),
    spawnMax: Math.max(1, asInt(getArg("spawn-shortest-path-max", "10"), 10)),
  };
  ensureParent(datasetPath);
  ensureParent(reportPath);
  const stream = fs.createWriteStream(datasetPath, { flags: "w" });
  const stats = makeEmptyStats();
  for (let i = 0; i < targetFrames; i++) {
    const row = makeRow(rng, i, opts);
    stream.write(JSON.stringify(row) + "\n");
    updateStats(stats, row);
  }
  stream.end();
  stream.on("finish", () => {
    const durationSec = (Date.now() - started) / 1000;
    const report = Object.assign({
      ts: Date.now(),
      mode: "offline_micro_collect",
      target_frames: targetFrames,
      dataset_path: datasetPath,
      report_path: reportPath,
      duration_sec: durationSec,
      rows_per_sec: durationSec > 0 ? stats.rows_written / durationSec : 0,
      soft_obstacle_keep_min: opts.softObstacleKeepMin,
      soft_obstacle_keep_max: opts.softObstacleKeepMax,
      scenario_buckets: opts.scenarios,
      high_value_ratio: stats.rows_written ? stats.high_value_behavior_rows / stats.rows_written : 0,
      behavior_score_mean: stats.rows_written ? stats.behavior_score_sum / stats.rows_written : 0,
    }, stats);
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log("[OFFLINE-MICRO]", JSON.stringify({ rows: stats.rows_written, rows_per_sec: report.rows_per_sec, report_path: reportPath }));
  });
}

main();
