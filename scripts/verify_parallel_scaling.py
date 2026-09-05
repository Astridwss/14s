"""验证并行 rollout 的真实扩展比：1 vs N worker 的 env.step() 吞吐。

回答「并行是不是真 N× / 为什么 8 会不稳定」：
  - 用 multiprocessing spawn（与 services/sample/parallel_rollout.py 同一机制，每 worker 独立进程+GIL）
  - 每 worker 独立建 env，跑 K 步随机动作，只计 step 时间（不含建 env 冷启动）
  - 扩展比 = N * t_serial / max(worker_step_time)（pool.map 是同步屏障，整轮 = 最慢 worker）
  - 理论上限 = 物理核数；异构核（P/E 核）+ 同步屏障会让扩展比打折，worker 耗时 spread 直接暴露拖尾

用法（仓库根目录）::

    .venv/Scripts/python.exe scripts/verify_parallel_scaling.py [K] [N]
"""
import multiprocessing as mp
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("PROJECT_ROOT", _ROOT)

import numpy as np

from utils.cpu_cores import logical_cpu_count, physical_cpu_count

_SCENE = os.path.join(_ROOT, "test", "mock_scene_200r_50s.json")


def _build_env():
    from use_cases.config.config_assembler import ConfigAssembler
    from services.scene.env_wrapper import GroupedEnvWrapper
    request = {
        "plan_id": 9999, "scene_url": _SCENE, "algorithm": "qmix",
        "group_size": 0, "max_episode_steps": 600,
    }
    conf = ConfigAssembler("scaling", request, mode="train").build()
    return GroupedEnvWrapper(conf, env_config=conf.env), conf


def _run_steps(args):
    """worker：独立建 env，跑 K 步，返回 (steps, step_only_elapsed_seconds)。"""
    _scene, K, seed = args
    env, conf = _build_env()
    n_agents, n_actions = conf.env.n_agents, conf.env.n_actions
    rng = np.random.default_rng(seed)
    env.reset()
    # 预热（不进计时，避开首个 step 的懒初始化）
    for _ in range(3):
        env.step((rng.random((n_agents, n_actions)) < 0.05).astype(np.float32))
    env.reset()
    t0 = time.perf_counter()
    for _ in range(K):
        env.step((rng.random((n_agents, n_actions)) < 0.05).astype(np.float32))
    return K, time.perf_counter() - t0


def main(K=15, N=8):
    print(f"机器: 逻辑 {logical_cpu_count()} / 物理 {physical_cpu_count()} 核 | "
          f"场景 200r+50s+21t | 每 worker {K} 步")

    # 串行基准（1 worker）
    _, t1 = _run_steps((_SCENE, K, 0))
    print(f"[serial]   1 worker: {t1:.3f}s -> {K / t1:.2f} step/s")

    # 并行 N worker（与 parallel_rollout 同 spawn 机制）
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(N)
    t0 = time.perf_counter()
    res = pool.map(_run_steps, [(_SCENE, K, i) for i in range(N)])
    wall = time.perf_counter() - t0
    pool.close(); pool.join()

    per = [r[1] for r in res]
    slowest = max(per)
    total_steps = sum(r[0] for r in res)
    speedup = (N * t1) / slowest          # 稳态扩展比（剔除建 env 冷启动）
    print(f"[parallel] {N} workers: 稳态步进 = {slowest:.3f}s(最慢) -> {total_steps / slowest:.2f} step/s")
    print(f"[parallel] {N} workers: pool.map 墙钟 = {wall:.3f}s（含 spawn + 建 env 冷启动）")
    print(f"[speedup]  实测扩展比 = {speedup:.2f}x  (理论 N={N})")
    print(f"[spread]   worker 步进 min={min(per):.3f}s max={slowest:.3f}s "
          f"max/min={slowest / min(per):.2f}x  (拖尾越明显, 扩展比越低)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15,
         int(sys.argv[2]) if len(sys.argv) > 2 else 8)
