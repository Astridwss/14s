"""验证 reward 去重改动前后奖励值完全一致。

构造 env，用固定随机种子跑一段，逐步对比：
  - 新 compute_reward_detailed（去重后，env.step 实际使用）
  - 旧参考实现（去重前：_r_ld / _r_wx 各自对同一 (雷达,目标) 对重算边缘）
total 与 breakdown 必须逐项完全一致，否则 FAIL。

用法（仓库根目录）::

    .venv/Scripts/python.exe scripts/verify_reward_equivalence.py [n_steps]
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("PROJECT_ROOT", _ROOT)

import numpy as np

from use_cases.config.config_assembler import ConfigAssembler
from services.scene.env_wrapper import GroupedEnvWrapper
from services.scene.reward.reward import (
    _RewardStream,
    _is_edge,
    _nearest_radar_gap,
    _split_agents,
)


# ============================================================
# 旧参考实现（去重前，逐行照搬改造前的 _r_ld / _r_wx 循环）
# ============================================================

def old_compute_reward_detailed(calc, raw_obs, actions_onehot, prev_onehot):
    radar_ids, sat_ids = _split_agents(calc.agent_keys, raw_obs.dict_equip_state)
    total = 0.0
    breakdown = {}
    streams = [old_r_ld(calc, raw_obs, actions_onehot, radar_ids)]
    if calc.wx_enabled:
        streams.append(old_r_wx(calc, raw_obs, actions_onehot, radar_ids, sat_ids))
    streams.append(calc._r_switch(raw_obs, actions_onehot, prev_onehot))
    for _val, _bd in streams:
        total += _val
        breakdown.update(_bd)
    return float(total), breakdown


def old_r_ld(calc, raw_obs, actions_onehot, radar_ids):
    detection = raw_obs.dict_detection_result
    equip = raw_obs.dict_equip_state
    track = raw_obs.dict_system_track
    radar_set = set(radar_ids)

    visible = {}
    for r in radar_ids:
        st = equip[r]
        for t, res in detection.get(r, {}).items():
            if not res.detectable_flag:
                continue
            is_edge = False
            trk = track.get(t)
            if trk is not None:
                is_edge = _is_edge(st, trk, calc.EDGE_RATIO)
            visible.setdefault(t, {})[r] = is_edge

    stream = _RewardStream("ld")
    locks = {}
    actions = np.asarray(actions_onehot)
    n = min(actions.shape[0], len(calc.agent_keys))
    for i in range(n):
        aid = calc.agent_keys[i]
        if aid not in radar_set:
            continue
        for b in np.nonzero(actions[i])[0]:
            b = int(b)
            if b < 1 or b > len(calc.target_keys):
                continue
            t = calc.target_keys[b - 1]
            if t in visible and aid in visible[t]:
                locks.setdefault(t, []).append(aid)
            else:
                stream.add("valid", -calc.VALID_PENALTY)

    for t, able in visible.items():
        locked = locks.get(t, [])
        n_lock = len(locked)
        if n_lock == 0:
            stream.add("miss", -calc.MISS_PENALTY)
            continue
        stream.add("cover", calc.COVER_REWARD)
        edge = sum(1 for r in locked if able.get(r, False))
        stream.add("edge", -calc.EDGE_PENALTY, edge)
        if n_lock > calc.K:
            stream.add("redundant", -calc.REDUNDANT_PENALTY, n_lock - calc.K)

    return stream.result()


def old_r_wx(calc, raw_obs, actions_onehot, radar_ids, sat_ids):
    detection = raw_obs.dict_detection_result
    equip = raw_obs.dict_equip_state
    track = raw_obs.dict_system_track
    sat_set = set(sat_ids)

    radar_visible = set()
    radar_edge = set()
    for r in radar_ids:
        st = equip[r]
        for t, res in detection.get(r, {}).items():
            if not res.detectable_flag:
                continue
            radar_visible.add(t)
            trk = track.get(t)
            if trk is not None and _is_edge(st, trk, calc.EDGE_RATIO):
                radar_edge.add(t)

    stream = _RewardStream("wx")
    actions = np.asarray(actions_onehot)
    n = min(actions.shape[0], len(calc.agent_keys))
    for i in range(n):
        aid = calc.agent_keys[i]
        if aid not in sat_set:
            continue
        t = None
        for b in np.nonzero(actions[i])[0]:
            b = int(b)
            if 1 <= b <= len(calc.target_keys):
                t = calc.target_keys[b - 1]
                break
        seen = detection.get(aid, {})
        useful = {x for x in seen
                  if seen[x].detectable_flag
                  and (x not in radar_visible or x in radar_edge)}
        if t is not None and (t not in seen or not seen[t].detectable_flag):
            stream.add("valid", -calc.WX_VALID_PENALTY)
            continue
        if t not in useful:
            if useful:
                stream.add("idle", -calc.WX_IDLE_PENALTY)
            continue
        if t not in radar_visible:
            gap = _nearest_radar_gap(raw_obs, radar_ids, t)
            stream.add("blind", calc.WX_BLIND_BASE + calc.WX_GAP_WEIGHT * (1.0 - gap))
        else:
            stream.add("handoff", calc.WX_HANDOFF_REWARD)

    return stream.result()


# ============================================================
# 主校验
# ============================================================

def build_env():
    scene_path = os.path.join(_ROOT, "test", "mock_scene_200r_50s.json")
    request = {
        "plan_id": 9999,
        "scene_url": scene_path,
        "algorithm": "qmix",
        "group_size": 0,
        "max_episode_steps": 600,
    }
    conf = ConfigAssembler("verify_reward", request, mode="train").build()
    return GroupedEnvWrapper(conf, env_config=conf.env), conf


def main(n_steps=200):
    env, conf = build_env()
    calc = env._reward_calc
    n_agents = conf.env.n_agents
    n_actions = conf.env.n_actions
    rng = np.random.default_rng(12345)

    env.reset()
    prev = None
    max_total_diff = 0.0
    mismatches = 0
    for step in range(n_steps):
        actions = (rng.random((n_agents, n_actions)) < 0.05).astype(np.float32)
        _obs, new_total, _term, _trunc, info = env.step(actions)
        raw_obs = info["raw_obs"]
        new_bd = info["reward_breakdown"]

        old_total, old_bd = old_compute_reward_detailed(calc, raw_obs, actions, prev)

        if new_bd != old_bd:
            mismatches += 1
            print(f"[MISMATCH] step {step} breakdown differs")
            print(f"  new keys: {set(new_bd) - set(old_bd)} / old keys: {set(old_bd) - set(new_bd)}")
            for k in set(new_bd) | set(old_bd):
                if new_bd.get(k) != old_bd.get(k):
                    print(f"  {k}: new={new_bd.get(k)} old={old_bd.get(k)}")

        max_total_diff = max(max_total_diff, abs(new_total - old_total))
        prev = np.asarray(actions).copy()

    print(f"[verify] {n_steps} steps, breakdown mismatches = {mismatches}, "
          f"max |new-old| total diff = {max_total_diff:.12f}")
    print("PASS" if (mismatches == 0 and max_total_diff == 0.0) else "FAIL")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
