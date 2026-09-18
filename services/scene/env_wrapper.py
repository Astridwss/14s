"""
Gym 风格环境包装器 —— 对接仿真引擎。

依赖：services.scene（观测构建、动作映射），不直接依赖 ScenarioAdapter。
"""

from typing import Optional

import numpy as np

import utils.pymap3d_cache  # noqa: F401  缓存 WGS84 椭球常量，消除 pymap3d 每次重建开销

from sim import TrainingEnv
from services.scene.action.mapper import ActionMapper
from services.scene.state.builder import ObservationBuilder
from services.scene.state.satellite_draw import compute_satellite_fov_targets
from services.scene.reward import RewardCalculator
from use_cases.config.config_types import EnvConfig


class GroupedEnvWrapper:
    """Gym 风格环境包装器。

    可通过两种方式构造:
      - GroupedEnvWrapper(conf)  —— 兼容旧版扁平 RuntimeConfig
      - GroupedEnvWrapper.from_env_config(env_cfg) —— 新版聚焦 EnvConfig
    """

    def __init__(self, conf, env_config: Optional[EnvConfig] = None):
        self.conf = conf
        ec = env_config or EnvConfig.from_config(conf)

        # 1. 引擎初始化
        self._sim = TrainingEnv()
        self._sim.load_battle_scene(ec.plan_id, ec.local_scene_path, time_step=ec.time_step)

        self._max_episode_steps = ec.max_episode_steps
        self._step_count = 0
        self._current_time = 0
        self._prev_actions = None  # 上一步动作列表（切换惩罚用）

        # 卫星视场补充信息：静态卫星信息（全程不变，缓存一次，仅绘制用）
        self._satellite_info = self._sim._dict_satellite_info

        # 2. 组装场景积木
        agent_keys = ec.agent_keys
        satellite_keys = getattr(ec, 'satellites_keys', [])
        self._obs_builder = ObservationBuilder(
            agent_keys=agent_keys,
            target_keys=ec.target_keys,
            n_agents=ec.n_agents,
            n_targets=ec.n_targets,
            n_actions=ec.n_actions,
            radar_obs_dim=ec.radar_obs_dim,
            satellite_ids=list(satellite_keys),
        )

        self._action_mapper = ActionMapper(
            agent_keys=agent_keys,
            target_keys=ec.target_keys,
            n_agents=ec.n_agents,
            n_actions=ec.n_actions,
            satellite_keys=list(satellite_keys),
        )

        self._reward_calc = RewardCalculator(
            agent_keys=agent_keys,
            target_keys=ec.target_keys,
            satellite_keys=list(satellite_keys),
        )

    @property
    def dict_radar_info(self):
        """雷达信息字典，供 RadarGrouper 聚类使用。"""
        return self._sim.dict_radar_info
    
    # 20260912 TaoXL add
    @property
    def dict_satellite_info(self):
        return self._satellite_info

    @property
    def dict_target_info(self):
        return self._sim._dict_target_info
    # 20260912 TaoXL add end

    def set_phase(self, phase: int) -> None:
        """同步训练阶段：phase1 屏蔽 R_wx（纯 LD 标定），phase2 解冻 WX 奖励。"""
        self._reward_calc.wx_enabled = (phase >= 2)

    # ============================================================
    # Gym API
    # ============================================================

    def reset(self):
        self._step_count = 0
        self._prev_actions = None
        raw_obs = self._sim.reset()
        self._current_time = raw_obs.current_time

        return self._obs_builder.build_observations(raw_obs), {
            "avail_actions": self._action_mapper.build_action_mask(raw_obs),
            "state": self._obs_builder.build_global_state(raw_obs),
            "raw_obs": raw_obs,
        }

    def step(self, actions):
        self._step_count += 1
        action_time = self._current_time

        agent_actions = self._action_mapper.to_engine_commands(actions, action_time)
        raw_obs, sim_terminated = self._sim.step_forward(agent_actions)
        self._current_time = raw_obs.current_time

        # 新奖励：R_ld + R_wx + R_switch（替换 sim.generate_reward + _switch_penalty）
        reward, reward_breakdown = self._reward_calc.compute_reward_detailed(
            raw_obs=raw_obs,
            actions_onehot=actions,
            prev_onehot=self._prev_actions,
        )
        self._prev_actions = np.asarray(actions).copy()
        terminated = bool(sim_terminated)  # 自然结束：current_time > _end_tim
        truncated = bool(self._step_count >= self._max_episode_steps)

        # TXL卫星视场覆盖（仅绘制补充，不参与训练；旁路异常不影响主流程）
        try:
            satellite_fov = compute_satellite_fov_targets(
                raw_obs, self._satellite_info, agent_actions,
            )
        except Exception:
            satellite_fov = []

        return (
            self._obs_builder.build_observations(raw_obs),
            reward,
            terminated,
            truncated,
            {
                "avail_actions": self._action_mapper.build_action_mask(raw_obs),
                "state": self._obs_builder.build_global_state(raw_obs),
                "raw_obs": raw_obs,
                "agent_actions_list": agent_actions,
                "action_time": action_time,
                "reward_breakdown": reward_breakdown,
                "satellite_fov": satellite_fov,
            },
        )
