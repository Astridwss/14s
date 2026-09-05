"""加载收敛权重 + 贪心一局，打印奖励分项表。

针对 rl_test_826_1（group_size=10 H-QMIX，纯 phase1 LD 标定）训练出的收敛权重，
加载后跑一局贪心（epsilon=0），打印分项，回答「收敛后奖励为何为负」。

用法::

    python test/diagnose_converged_breakdown.py
"""

import os
import sys

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from use_cases.config.config_assembler import ConfigAssembler
from services.scene.env_wrapper import GroupedEnvWrapper
from services.algorithms.qmix.agent import Agents
from services.scene.grouping import RadarGrouper
from utils.reward_diagnostics import diagnose_episode

LOAD_DIR = os.path.join(_PROJ_ROOT, "models", "rl_test_826_1")
SCENE = os.path.join(_PROJ_ROOT, "scenarios", "rl_test_817_1", "scene.json")


def main():
    request = {
        "plan_id": 867,
        "scene_url": SCENE,
        "algorithm": "qmix",
        "group_size": 10,          # 匹配权重前缀 LD100_WX25_TARGET21_groupsize10
        "phase1_episodes": -1,     # 与训练一致：纯 phase1
    }
    conf = ConfigAssembler("diag_converged", request, mode="train").build()

    env = GroupedEnvWrapper(conf, env_config=conf.env)
    group_assignments = RadarGrouper.try_build(
        group_size=conf.algo.group_size,
        radar_keys=conf.env.radar_keys,
        radar_info_dict=env.dict_radar_info,
    )
    agents = Agents(conf, algo_config=conf.algo, group_assignments=group_assignments)

    agents.policy.load_state(LOAD_DIR)
    print(f"[诊断] 已加载收敛权重: {LOAD_DIR}")

    # 与训练一致：phase1（WX 待机，R_wx 不计算）
    agents.set_phase(1)
    env.set_phase(1)

    tracker = diagnose_episode(env, agents, epsilon=0.0, max_steps=600)

    acc = tracker._acc
    sum_breakdown = sum(v["value"] for v in acc.values())
    diff = abs(sum_breakdown - tracker.total)
    print(f"\n[校验] 分项 value 之和 = {sum_breakdown:.4f}  |  累计 total = {tracker.total:.4f}")
    print(f"[校验] 差值 = {diff:.6f}  {'PASS' if diff < 1e-3 else 'FAIL'}")


if __name__ == "__main__":
    main()
