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
    results = mgr.generate_episodes(weights, phase, epsilon)   # [(reward, steps, ep_data), ...]
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


def _init_worker(conf_dict, group_assignments):
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
    ec = EnvConfig.from_config(conf)
    ac = AlgorithmConfig.from_config(conf)

    env = GroupedEnvWrapper(conf, env_config=ec)
    agents = Agents(conf, algo_config=ac, group_assignments=group_assignments)
    rollout = RolloutWorker(env, agents,
                            radar_keys=ec.agent_keys, target_keys=ec.target_keys)

    # 并行模式下每个 worker 自建 ZMQ 推送 + 态势日志：独立进程独立 socket，
    # 逐帧打 episode_idx/step_idx，前端按 (episode_idx, step_idx) 重排消费。
    from services.zmq.zmq_push import TrainingSituationPushService
    from utils.situation_logger import SituationLogHook
    push_service = TrainingSituationPushService.from_conf(conf)

    _WS["env"] = env
    _WS["agents"] = agents
    _WS["rollout"] = rollout
    _WS["push_service"] = push_service
    _WS["situation_hook"] = SituationLogHook(
        radar_keys=ec.agent_keys, target_keys=ec.target_keys, log_interval=50)


def _run_one_episode(task):
    """单个 worker 执行一局。task = (weights, phase, epsilon, episode_idx)。"""
    weights, phase, epsilon, episode_idx = task
    env = _WS["env"]
    agents = _WS["agents"]
    rollout = _WS["rollout"]
    push_service = _WS.get("push_service")
    situation_hook = _WS.get("situation_hook")

    agents.set_inference_weights(weights)
    agents.set_phase(phase)
    env.set_phase(phase)

    hooks = []
    if push_service is not None:
        hooks.extend(push_service.get_step_hooks(episode_idx))
    if situation_hook is not None:
        hooks.append(situation_hook)

    ep_reward, step_count, ep_data = rollout.generate_train_episode(
        epsilon=epsilon, step_hooks=hooks)
    return ep_reward, step_count, ep_data


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
        self._pool = self._ctx.Pool(
            self.n_workers,
            initializer=_init_worker,
            initargs=(_conf_to_dict(conf), group_assignments),
        )

    def generate_episodes(self, weights, phase, epsilon, episode_indices):
        """并行跑一局，每局用 episode_indices 中对应下标打标。

        Returns:
            [(ep_reward, step_count, ep_data), ...]，顺序与 episode_indices 一致。
        """
        tasks = [(weights, phase, epsilon, int(eidx)) for eidx in episode_indices]
        return self._pool.map(_run_one_episode, tasks)

    def close(self):
        self._pool.close()
        self._pool.join()
