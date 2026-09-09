"""
并行环境采样 —— 多进程 worker 各自持有一份冻结 sim 环境副本，独立跑局。

为什么可行（不改 sim/client.py）:
    TrainingEnv.load_battle_scene 只读 JSON（read_battle_scene_from_json 无副作用），
    step_forward 只改 self 实例属性，无模块级全局 / 锁 / 随机数 / 文件写入
    → 完全实例隔离，天然可多进程并行。

架构（同步轮次，Ape-X 简化版）:
    主进程 learner 每轮广播 (weights, phase, epsilon)
    → N 个 worker 各跑一局 → 回收 N 份轨迹回主进程 buffer。

Windows 下 multiprocessing 默认 spawn：worker 入口必须是模块级函数
（可被 pickle 导入），故 _init_worker / _run_one_episode 都定义在模块顶层。

用法:
    from services.sample.parallel_rollout import ParallelRollout
    mgr = ParallelRollout(conf, group_assignments, n_workers=8)
    results = mgr.generate_episodes(weights, phase, epsilon, episode_indices)
    #   results = [(reward, steps, ep_data, frames), ...]，frames 为整局态势轨迹字节列表
    mgr.close()
"""

import multiprocessing as mp
from types import SimpleNamespace

import torch


# ============================================================
# worker 全局状态（每个进程一份）
# ============================================================
_WS: dict = {}


def _conf_to_dict(conf) -> dict:
    """把 RuntimeConfig / SimpleNamespace 转成可 pickle 的纯 dict。

    - 跳过聚焦子配置 env/algo/train/infra（worker 侧用 from_config 重建）
    - torch.device 转字符串（worker 侧按字符串解析，避免跨进程依赖）
    """
    d = {}
    for k, v in vars(conf).items():
        if k in ("env", "algo", "train", "infra"):
            continue
        if isinstance(v, torch.device):
            v = str(v)
        d[k] = v
    return d


def _build_worker_devices(conf, n_workers):
    """按 learner 设备与可见 NPU 卡数生成 worker 轮询 device 表；无多卡时返回 None。

    - learner 设备非 npu（cpu/cuda 等）：返回 None，worker 沿用 conf.device。
    - learner 设备为 npu 且可见卡数 > 1：worker i → npu:(i % 卡数)。
      N 个 worker 各持一个 torch_npu context，若全挤在 npu:0 会 OOM，故分摊到多卡。
    """
    dev = str(getattr(conf, 'device', 'cpu'))
    if not dev.startswith('npu'):
        return None
    try:
        import torch_npu  # noqa: F401
        n_cards = torch_npu.npu.device_count()
    except Exception:
        return None
    if n_cards <= 1:
        return None
    return [f"npu:{i % n_cards}" for i in range(n_workers)]


def _init_worker(conf_dict, group_assignments, worker_devices, idx_counter):
    """spawn 后每个 worker 进程调用一次：重建 env + policy 副本 + rollout。"""
    # 离线 NPU 机器需注册 npu 后端（本机无 torch_npu 时静默跳过）
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass

    from use_cases.config.config_types import EnvConfig, AlgorithmConfig
    from services.scene.env_wrapper import GroupedEnvWrapper
    from services.algorithms.qmix.agent import Agents
    from services.sample.rollout import RolloutWorker

    conf = SimpleNamespace(**conf_dict)

    # 多卡分摊：worker 按启动序号认领一张卡（轮询），覆盖 learner 的 conf.device，
    # 避免 N 个 worker 各建一个 torch_npu context 全挤在 npu:0 触发 OOM。
    if worker_devices:
        with idx_counter.get_lock():
            idx = idx_counter.value
            idx_counter.value += 1
        conf.device = worker_devices[idx % len(worker_devices)]

    ec = EnvConfig.from_config(conf)
    ac = AlgorithmConfig.from_config(conf)

    env = GroupedEnvWrapper(conf, env_config=ec)
    agents = Agents(conf, algo_config=ac, group_assignments=group_assignments)
    rollout = RolloutWorker(env, agents,
                            radar_keys=ec.agent_keys, target_keys=ec.target_keys)

    # 并行模式不再在 worker 侧直接 ZMQ push：worker 只负责「采集」整局态势轨迹
    # （逐帧序列化、烘焙 EpisodeIdx/StepIdx，不发送），轨迹回主进程入池后由
    # 独立发送线程按前端倍速推送（见 services/zmq/trajectory_pool.py）。
    from utils.situation_logger import SituationLogHook
    push_enabled = int(getattr(conf, 'push_interval', 0) or 0) > 0

    _WS["env"] = env
    _WS["agents"] = agents
    _WS["rollout"] = rollout
    _WS["push_enabled"] = push_enabled
    _WS["situation_hook"] = SituationLogHook(
        radar_keys=ec.agent_keys, target_keys=ec.target_keys, log_interval=50)


def _run_one_episode(task):
    """单个 worker 执行一局。task = (weights, phase, epsilon, episode_idx)。

    返回 (ep_reward, step_count, ep_data, frames)，frames 为整局态势轨迹字节列表
    （push 关闭时为空列表）。
    """
    weights, phase, epsilon, episode_idx = task
    env = _WS["env"]
    agents = _WS["agents"]
    rollout = _WS["rollout"]
    situation_hook = _WS.get("situation_hook")

    agents.set_inference_weights(weights)
    agents.set_phase(phase)
    env.set_phase(phase)

    collector = None
    hooks = []
    if _WS.get("push_enabled"):
        from services.zmq.zmq_push import SituationCollector
        collector = SituationCollector(episode_idx)
        hooks.append(collector)
    if situation_hook is not None:
        hooks.append(situation_hook)

    ep_reward, step_count, ep_data = rollout.generate_train_episode(
        epsilon=epsilon, step_hooks=hooks)
    frames = collector.frames if collector is not None else []
    return ep_reward, step_count, ep_data, frames


# ============================================================
# 对外管理器
# ============================================================
class ParallelRollout:
    """并行 rollout 管理器：N 个 worker 进程，每轮同步跑 N 局。

    Args:
        conf:               RuntimeConfig（或带属性访问的配置对象）
        group_assignments:  RadarGrouper 分组结果（None = 标准 QMIX）
        n_workers:          环境副本 / worker 进程数
    """

    def __init__(self, conf, group_assignments, n_workers: int):
        self.n_workers = int(n_workers)
        if self.n_workers < 2:
            raise ValueError(f"n_workers 需 >= 2，收到 {self.n_workers}")
        self._ctx = mp.get_context("spawn")
        worker_devices = _build_worker_devices(conf, self.n_workers)
        self._idx_counter = self._ctx.Value('i', 0)
        if worker_devices:
            print(f"[ParallelRollout] 多卡分摊: {self.n_workers} workers → "
                  f"{worker_devices}")
        self._pool = self._ctx.Pool(
            self.n_workers,
            initializer=_init_worker,
            initargs=(_conf_to_dict(conf), group_assignments,
                      worker_devices or [], self._idx_counter),
        )

    def generate_episodes(self, weights, phase, epsilon, episode_indices):
        """并行跑一局，每局用 episode_indices 中对应下标打标。

        Returns:
            [(ep_reward, step_count, ep_data, frames), ...]，顺序与 episode_indices 一致。
            frames 为整局态势轨迹字节列表（push 关闭时为空列表）。
        """
        tasks = [(weights, phase, epsilon, int(eidx)) for eidx in episode_indices]
        return self._pool.map(_run_one_episode, tasks)

    def close(self):
        self._pool.close()
        self._pool.join()
