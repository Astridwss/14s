"""Profile env.step() wall-clock + cProfile attribution for the 200r/50s/21t scene.

用法（仓库根目录）::

    .venv/Scripts/python.exe scripts/profile_env_step.py [n_steps]

用一个真实规模场景（200 雷达 + 50 卫星 + 21 目标）跑 N 步，输出:
  - full env.step() 的 ms/step（墙钟）
  - cProfile 累计耗时 Top 函数（用于归因 step_forward / reward / 几何换算）
"""
import io
import os
import sys
import time
import cProfile
import pstats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("PROJECT_ROOT", _ROOT)

import numpy as np

from use_cases.config.config_assembler import ConfigAssembler
from services.scene.env_wrapper import GroupedEnvWrapper


def build_env():
    scene_path = os.path.join(_ROOT, "test", "mock_scene_200r_50s.json")
    request = {
        "plan_id": 9999,
        "scene_url": scene_path,
        "algorithm": "qmix",
        "group_size": 0,
        "max_episode_steps": 600,
    }
    conf = ConfigAssembler("profile_env", request, mode="train").build()
    env = GroupedEnvWrapper(conf, env_config=conf.env)
    return env, conf


def main(n_steps=30):
    env, conf = build_env()
    n_agents = conf.env.n_agents
    n_actions = conf.env.n_actions
    rng = np.random.default_rng(0)

    def step():
        # 多标签随机动作：每个智能体以 5% 概率锁定每个目标，走完整 R_ld / R_wx 结算路径
        actions = (rng.random((n_agents, n_actions)) < 0.05).astype(np.float32)
        return env.step(actions)

    # 预热 + 复位
    env.reset()
    for _ in range(3):
        step()
    env.reset()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        step()
    dt = time.perf_counter() - t0
    print(f"[wall] full env.step() = {dt / n_steps * 1000.0:.2f} ms/step"
          f"  (n={n_steps}, total={dt:.3f}s)")

    # cProfile 归因（15 步，避免 cProfile 开销淹没统计）
    env.reset()
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(15):
        step()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
