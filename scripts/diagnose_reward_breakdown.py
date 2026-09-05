"""单局奖励分项诊断 —— 验证 reward.py 原生分项 + RewardBreakdownTracker 输出。

用法（项目根目录）::

    python test/diagnose_reward_breakdown.py

用一个真实场景（100 雷达 + 25 卫星 + 21 目标）跑 30 步，打印分项表，
并校验「分项 value 之和 == 累计总奖励」，证明原生分项无漂移。
"""

import os
import sys

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from use_cases.config.config_assembler import ConfigAssembler
from services.scene.env_wrapper import GroupedEnvWrapper
from services.algorithms.qmix.agent import Agents
from utils.reward_diagnostics import diagnose_episode


def main():
    scene_path = os.path.abspath(
        os.path.join(_PROJ_ROOT, "scenarios", "rl_test_817_1", "scene.json")
    )
    request = {
        "plan_id": 867,
        "scene_url": scene_path,
        "algorithm": "qmix",
        "group_size": 0,          # 标准 QMIX（验证分项与 agent 类型无关）
        "phase1_episodes": 0,     # 直接联合（WX 也参与，分项更全）
        "max_episode_steps": 600,
    }
    conf = ConfigAssembler("diag_reward", request, mode="train").build()

    env = GroupedEnvWrapper(conf, env_config=conf.env)
    agents = Agents(conf, algo_config=conf.algo, group_assignments=None)

    tracker = diagnose_episode(env, agents, epsilon=0.0, max_steps=30)

    # 校验：分项 value 之和应等于累计总奖励（原生分项无漂移）
    acc = tracker._acc
    sum_breakdown = sum(v["value"] for v in acc.values())
    diff = abs(sum_breakdown - tracker.total)
    print(f"\n[校验] 分项 value 之和 = {sum_breakdown:.4f}  |  累计 total = {tracker.total:.4f}")
    print(f"[校验] 差值 = {diff:.6f}  {'PASS' if diff < 1e-3 else 'FAIL'}")


if __name__ == "__main__":
    main()
