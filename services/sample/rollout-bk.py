"""
轨迹采样服务 —— 执行单局环境步进，收集并返回轨迹数据。

自包含积木：构造时接收 env、agents、实体 keys。不依赖 Config 或 Adapter。

Recorder 模式：每种 episode 类型对应一个 EpisodeRecorder 子类，
_run_episode 只负责循环控制，不持有数据存储逻辑。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch


# ============================================================
# StepContext —— 不可变数据载体
# ============================================================

@dataclass(frozen=True)
class StepContext:
    """单步环境交互后产生的完整上下文，一次构造，所有消费者共享。

    消除原来 11 参数的 _record 闭包签名和 info_callback 的单独传参。
    """

    step_idx: int
    obs: np.ndarray
    next_obs: np.ndarray
    state: np.ndarray
    next_state: np.ndarray
    actions: np.ndarray          # (n_agents, 1)
    actions_onehot: np.ndarray   # (n_agents, n_actions)
    reward: float
    terminated: bool
    truncated: bool
    avail_actions: np.ndarray
    next_avail_actions: np.ndarray
    next_info: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# EpisodeRecorder —— 抽象基类
# ============================================================

class EpisodeRecorder(ABC):
    """每步记录策略的抽象。

    子类负责：存储数据、最终组装产出。
    """

    @abstractmethod
    def record_step(self, ctx: StepContext) -> None:
        """处理单步上下文。"""
        ...

    @abstractmethod
    def finalize(self, actual_steps: int) -> None:
        """episode 结束时调用，用于切片等收尾操作。"""
        ...

    @abstractmethod
    def result(self) -> Any:
        """返回 episode 的最终产出，类型由子类决定。"""
        ...



# ============================================================
# TrainRecorder —— 训练用预分配记录器
# ============================================================

class TrainRecorder(EpisodeRecorder):
    """预分配 numpy 数组，零运行时 append 开销。

    产出: (ep_reward, actual_steps, episode_batch_dict)
    """

    def __init__(self, n_agents: int, n_actions: int,
                 max_steps: int, obs_dim: int, state_dim: int):
        self._n_agents = n_agents
        self._max_steps = max_steps
        self._edata = self._alloc(max_steps, n_agents, n_actions,
                                  obs_dim, state_dim)
        self._actual_steps = 0

    # ---- 公开 API ----

    def record_step(self, ctx: StepContext) -> None:
        idx = ctx.step_idx
        if idx >= self._max_steps:
            return
        self._edata['obs'][idx] = ctx.obs
        self._edata['next_obs'][idx] = ctx.next_obs
        self._edata['state'][idx] = ctx.state
        self._edata['next_state'][idx] = ctx.next_state
        self._edata['actions'][idx] = ctx.actions
        self._edata['actions_onehot'][idx] = ctx.actions_onehot
        self._edata['rewards'][idx] = ctx.reward
        self._edata['terminated'][idx] = 1.0 if ctx.terminated else 0.0
        self._edata['avail_actions'][idx] = ctx.avail_actions
        self._edata['next_avail_actions'][idx] = ctx.next_avail_actions

    def finalize(self, actual_steps: int) -> None:
        self._actual_steps = min(actual_steps, self._max_steps)

    def result(self):
        n = self._actual_steps
        episode_batch = {k: v[:n] for k, v in self._edata.items()}
        ep_reward = float(self._edata['rewards'][:n].sum())
        return ep_reward, n, episode_batch


    # ---- 内部 ----

    @staticmethod
    def _alloc(size: int, n_agents: int, n_actions: int,
               obs_dim: int, state_dim: int) -> Dict[str, np.ndarray]:
        return {
            'obs': np.zeros((size, n_agents, obs_dim), dtype=np.float32),
            'next_obs': np.zeros((size, n_agents, obs_dim), dtype=np.float32),
            'state': np.zeros((size, state_dim), dtype=np.float32),
            'next_state': np.zeros((size, state_dim), dtype=np.float32),
            'actions': np.zeros((size, n_agents, 1), dtype=np.int64),
            'actions_onehot': np.zeros((size, n_agents, n_actions), dtype=np.float32),
            'rewards': np.zeros((size, 1), dtype=np.float32),
            'terminated': np.zeros((size, 1), dtype=np.float32),
            'avail_actions': np.zeros((size, n_agents, n_actions), dtype=np.float32),
            'next_avail_actions': np.zeros((size, n_agents, n_actions), dtype=np.float32),
        }


# ============================================================
# EvalRecorder —— 推演用累加记录器
# ============================================================

class EvalRecorder(EpisodeRecorder):
    """累加 reward + 收集每步有效指令。

    产出: (ep_reward, step_count, eval_records_dict)
    """

    def __init__(self):
        self._ep_reward = 0.0
        self._eval_records: Dict[int, list] = {}
        self._step_count = 0

    # ---- 公开 API ----

    def record_step(self, ctx: StepContext) -> None:
        self._ep_reward += ctx.reward

        action_time = int(ctx.next_info.get('action_time', 0))
        valid_cmds = [
            cmd for cmd in ctx.next_info.get('agent_actions_list', [])
            if getattr(cmd, 'str_target_id', '') not in ('', '0')
        ]
        if action_time not in self._eval_records:
            self._eval_records[action_time] = valid_cmds

    def finalize(self, actual_steps: int) -> None:
        self._step_count = actual_steps

    def result(self):
        return self._ep_reward, self._step_count, self._eval_records


# ============================================================
# RolloutWorker
# ============================================================

class RolloutWorker:
    """单局轨迹采样器。

    train / eval 共用 ``_run_episode`` 循环，通过不同的 EpisodeRecorder
    子类实现数据存储策略的多态。
    """

    def __init__(self, env, agents, radar_keys, target_keys):
        self.env = env
        self.agents = agents
        self.radar_keys = radar_keys
        self.target_keys = target_keys

        self._max_steps = getattr(env, '_max_episode_steps', 600)

    # ============================================================
    # 公开 API
    # ============================================================

    def generate_train_episode(self, epsilon: float, episode_num: int = 0,
                                render_callback: Optional[Callable] = None,
                                step_hooks: Optional[List[Callable]] = None):
        """跑完完整一局，专用于训练。

        Returns:
            (ep_reward, actual_steps, episode_batch_dict)
        """
        del episode_num  # 保留参数兼容性，当前通过 step_hooks 注入态势日志
        recorder = TrainRecorder(
            n_agents=self.agents.n_agents,
            n_actions=self.agents.n_actions,
            max_steps=self._max_steps,
            obs_dim=self._infer_obs_dim(),
            state_dim=self._infer_state_dim(),
        )
        self._run_episode(epsilon, recorder,
                          step_hooks=step_hooks,
                          render_callback=render_callback)
        return recorder.result()

    def generate_eval_episode(self, episode_num: int = 0, step_hooks: Optional[List[Callable]] = None):
        """跑完完整一局，专用于评估推演。

        Returns:
            (ep_reward, step_count, eval_records_dict)
        """
        del episode_num
        recorder = EvalRecorder()
        self._run_episode(0.0, recorder, step_hooks=step_hooks, render_callback=None)
        return recorder.result()

    # ============================================================
    # 统一 episode 循环
    # ============================================================

    def _run_episode(
        self,
        epsilon: float,
        recorder: EpisodeRecorder,
        step_hooks: Optional[List[Callable]] = None,
        render_callback: Optional[Callable] = None,
    ) -> None:
        """统一的 episode 生成循环 —— train / eval 共用。

        Args:
            epsilon:         探索率
            recorder:        EpisodeRecorder 子类实例（策略模式）
            step_hooks:      每步后执行的回调列表（ZMQ 推送、态势日志等）
            render_callback: 渲染回调
        """
        setup = self._setup_episode()
        obs = setup['obs']
        avail_actions = setup['avail_actions']
        state = setup['info']['state']
        last_actions = setup['last_actions']

        terminated, truncated, step_idx = False, False, 0

        while not (terminated or truncated):
            #模型决策阶段：根据当前观测和历史动作，输出 LD+WX 动作
            actions, _, actions_onehot = self.agents.perform_inference(
                obs=obs, last_actions=last_actions,
                avail_actions=avail_actions, epsilon=epsilon,
            )

            actions_np, actions_onehot_np, env_actions = self._to_env_actions(
                actions, actions_onehot,
            )

            next_obs, reward, terminated, truncated, next_info = self.env.step(env_actions)

            # 将 reward 附加到 next_info，供 hooks 使用
            next_info['_last_reward'] = reward

            # 构建不可变上下文 → Recorder 消费
            ctx = StepContext(
                step_idx=step_idx,
                obs=obs,
                next_obs=next_obs,
                state=state,
                next_state=next_info['state'],
                actions=actions_np.reshape(-1, 1),
                actions_onehot=actions_onehot_np,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                avail_actions=avail_actions,
                next_avail_actions=next_info['avail_actions'],
                next_info=next_info,
            )
            recorder.record_step(ctx)

            # 渲染
            if render_callback is not None:
                render_callback(next_info)

            # 外部 hooks（ZMQ 推送、态势日志等） SituationCollector.__call__
            if step_hooks is not None:
                for hook in step_hooks:
                    hook(step_idx, terminated, truncated, next_info)

            # 状态转移
            obs = next_obs
            state = next_info['state']
            avail_actions = next_info['avail_actions']
            last_actions = actions_onehot_np
            step_idx += 1

        recorder.finalize(step_idx)

    # ============================================================
    # 内部工具方法
    # ============================================================

    def _setup_episode(self):
        """训练 / 推演共用的 episode 初始化。"""
        n_agents = self.agents.n_agents
        n_actions = self.agents.n_actions
        obs, info = self.env.reset()
        last_actions = np.zeros((n_agents, n_actions), dtype=np.float32)
        self.agents.init_episode()
        return {
            'obs': obs,
            'info': info,
            'avail_actions': info['avail_actions'],
            'last_actions': last_actions,
        }

    @staticmethod
    def _to_env_actions(actions, actions_onehot):
        """将 Agents 输出的动作张量转为引擎可用的格式。

        Returns:
            actions_np:        (n_agents,) 或 (n_agents, 1)  argmax 代表动作（buffer 兼容）
            actions_onehot_np: (n_agents, n_actions) 多标签 one-hot
            env_actions:       (n_agents, n_actions) 多标签 one-hot，直接交给 env.step
        """
        actions_np = (actions.cpu().detach().numpy()
                      if isinstance(actions, torch.Tensor) else np.array(actions))
        actions_onehot_np = (actions_onehot.cpu().detach().numpy()
                             if isinstance(actions_onehot, torch.Tensor)
                             else np.array(actions_onehot))
        env_actions = actions_onehot_np  # 多标签 one-hot 直接交给 env.step
        return actions_np, actions_onehot_np, env_actions

    def _infer_obs_dim(self) -> int:
        """从 ObservationBuilder 推断单智能体观测维度。"""
        obs_builder = getattr(self.env, '_obs_builder', None)
        if obs_builder is not None:
            rd = getattr(obs_builder, 'radar_obs_dim', None)
            if rd is not None:
                return int(rd)
        return 25  # 默认雷达观测维度

    def _infer_state_dim(self) -> int:
        """从 ObservationBuilder 推断全局状态维度。

        状态布局与 ObservationBuilder.build_global_state() 保持一致:
          [targets (n_targets×8) | agents (n_agents×5) | time (1)]
        """
        from services.scene.scene_constants import TARGET_STATE_FEATURES, RADAR_STATE_FEATURES
        n_agents = self.agents.n_agents
        n_targets = len(self.target_keys)
        return n_targets * TARGET_STATE_FEATURES + n_agents * RADAR_STATE_FEATURES + 1
