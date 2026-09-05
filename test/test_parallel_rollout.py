"""并行 rollout 验证 —— 正确性（串行==并行）+ 加速比。

运行（必须在 __main__ 下，Windows spawn 需要）::

    .venv/Scripts/python.exe test/test_parallel_rollout.py

验证点:
    1. 正确性: 相同权重 + epsilon=0（确定性 env + 贪心策略）下，
       串行 RolloutWorker 与 并行 ParallelRollout 产出的 (reward, steps) 完全一致。
    2. 加速比: N 个 worker 并行跑 N 局 明显快于 串行跑 N 局。

不改 sim/client.py（冻结内核），只实例化多份 TrainingEnv 副本。
"""

import os
import sys
import tempfile
import time
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from types import SimpleNamespace

from scripts.generate_mock_scene import generate_mock_scene
from use_cases.config.config_assembler import ConfigAssembler
from services.sample.parallel_rollout import ParallelRollout, _conf_to_dict
from use_cases.config.config_types import EnvConfig, AlgorithmConfig
from services.scene.env_wrapper import GroupedEnvWrapper
from services.algorithms.qmix.agent import Agents
from services.sample.rollout import RolloutWorker


# 小场景：40 雷达 / 8 卫星 / 10 目标，跑得快又能体现 env 步进成本
N_RADARS, N_SATELLITES, N_TARGETS = 40, 8, 10
DURATION_S = 100
MAX_EPISODE_STEPS = 50
PLAN_ID = 9876


def build_conf(scene_path):
    assembler = ConfigAssembler(
        task_id="parallel_rollout_test",
        request_data={
            "scene_url": scene_path,
            "plan_id": PLAN_ID,
            "hyperparameters": {
                "max_episode_steps": MAX_EPISODE_STEPS,
                "device": "cpu",
                "group_size": 0,   # 标准 QMIX（并行管线与 HQMIX 相同）
            },
        },
        mode="train",
    )
    return assembler.build()


def build_serial_stack(conf_dict):
    """与 parallel_rollout._init_worker 完全相同的重建路径。"""
    conf = SimpleNamespace(**conf_dict)
    ec = EnvConfig.from_config(conf)
    ac = AlgorithmConfig.from_config(conf)
    env = GroupedEnvWrapper(conf, env_config=ec)
    agents = Agents(conf, algo_config=ac, group_assignments=None)
    rollout = RolloutWorker(env, agents,
                            radar_keys=ec.agent_keys, target_keys=ec.target_keys)
    return env, agents, rollout


def test_correctness(conf, conf_dict):
    print("\n" + "=" * 64)
    print("[用例1] 正确性：串行 == 并行（固定权重 + epsilon=0）")
    print("=" * 64)

    # 串行参考栈
    env, agents, rollout = build_serial_stack(conf_dict)
    weights = agents.get_inference_weights()

    n_ep = 3
    serial = []
    for _ in range(n_ep):
        r, s, _ = rollout.generate_train_episode(epsilon=0.0)
        serial.append((float(r), int(s)))
    print(f"  串行: {serial}")

    # 并行（3 worker 各跑 1 局）
    mgr = ParallelRollout(conf, None, n_workers=n_ep)
    try:
        parallel = mgr.generate_episodes(weights, phase=1, epsilon=0.0, count=n_ep)
    finally:
        mgr.close()
    parallel = [(float(r), int(s)) for r, s, _ in parallel]
    print(f"  并行: {parallel}")

    for i, (a, b) in enumerate(zip(serial, parallel)):
        assert a == b, f"第 {i} 局 reward 不一致: 串行 {a} vs 并行 {b}"
    print(f"  [OK] {n_ep} 局 reward/steps 串并行完全一致（确定性重放 + 贪心策略）")


def test_speedup(conf):
    print("\n" + "=" * 64)
    print("[用例2] 加速比：N worker 并行 vs 串行")
    print("=" * 64)

    n_workers = 4

    # 串行基准
    _, agents, rollout = build_serial_stack(_conf_to_dict(conf))
    weights = agents.get_inference_weights()

    t0 = time.time()
    for _ in range(n_workers):
        rollout.generate_train_episode(epsilon=0.0)
    serial_t = time.time() - t0
    print(f"  串行 {n_workers} 局: {serial_t:.2f}s")

    # 并行
    mgr = ParallelRollout(conf, None, n_workers=n_workers)
    try:
        # 预热一轮（摊掉 spawn + 环境/网络构建固定开销）
        mgr.generate_episodes(weights, phase=1, epsilon=0.0)
        t0 = time.time()
        mgr.generate_episodes(weights, phase=1, epsilon=0.0)
        parallel_t = time.time() - t0
    finally:
        mgr.close()
    print(f"  并行 {n_workers} 局: {parallel_t:.2f}s")

    speedup = serial_t / max(parallel_t, 1e-9)
    print(f"  [OK] 加速比 ≈ {speedup:.1f}×（{n_workers} workers，稳定态单轮）")
    assert speedup > 1.5, f"并行未体现出加速: {speedup:.1f}×"
    print("  [OK] 加速比 > 1.5×，并行环境采样有效")


def main():
    tmp = tempfile.mkdtemp(prefix="parallel_rollout_")
    scene_path = os.path.join(tmp, "mock_scene.json")
    try:
        print("生成 mock 场景 ...")
        generate_mock_scene(
            n_radars=N_RADARS, n_satellites=N_SATELLITES,
            n_targets=N_TARGETS, duration_s=DURATION_S,
            plan_id=PLAN_ID, output_path=scene_path,
        )

        conf = build_conf(scene_path)
        conf_dict = _conf_to_dict(conf)

        test_correctness(conf, conf_dict)
        test_speedup(conf)

        print("\n" + "=" * 64)
        print("[结论] 并行 rollout 验证通过：结果与串行一致，且加速显著")
        print("=" * 64)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
