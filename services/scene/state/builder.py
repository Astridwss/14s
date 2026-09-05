"""
观测构建 —— AgentObservation → 神经网络输入张量。

自包含积木：构造时接收维度和实体 ID，每步调用时返回 numpy 数组。
"""
from typing import List, Optional

import numpy as np
import pymap3d as pm
from sim import AgentObservation
from services.scene.scene_constants import (
    RADAR_SELF_FEATURES, TARGET_FEATURES, SATELLITE_BROADCAST_FEATURES,
    RADAR_STATE_FEATURES, TARGET_STATE_FEATURES,
    MAX_RANGE_KM, AZI_RANGE, ELE_RANGE,
    ECF_POS_SCALE, ECF_VEL_SCALE, EPISODE_DURATION,
    AGENT_TYPE_RADAR, AGENT_TYPE_SATELLITE,
)
from services.scene.state.satellite_broadcast import (
    get_satellite_visible_targets,
    compute_boundary_gap,
    build_satellite_confidence_map,
)


class ObservationBuilder:
    """将仿真引擎的 AgentObservation 转为规整 numpy 张量。"""

    def __init__(self, agent_keys, target_keys,
                 n_agents, n_targets, n_actions, radar_obs_dim,
                 satellite_ids: Optional[List[str]] = None):
        self.agent_keys = agent_keys
        self.target_keys = target_keys
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.n_actions = n_actions
        self.satellite_ids = satellite_ids or []

        # 观测维度 = 自身 + 目标特征(含卫星广播) × 目标数
        self.radar_obs_dim = (
            RADAR_SELF_FEATURES
            + n_targets * (TARGET_FEATURES + SATELLITE_BROADCAST_FEATURES)
        )

        # 每目标特征步长（含广播）
        self._target_stride = TARGET_FEATURES + SATELLITE_BROADCAST_FEATURES

        # 启动时构建查找表，热路径 O(1)
        self._target_idx = {tid: i for i, tid in enumerate(target_keys)}

    # ============================================================
    # 局部观测
    # ============================================================

    def build_observations(self, agent_obs: AgentObservation) -> np.ndarray:
        """每步调用：构建 (n_agents, obs_dim) 观测矩阵。

        观测布局（每个目标）:
          [原有 5 维: detectable, distance, azi, ele, lock_rate]
          [广播 3 维: sat_visible, boundary_gap, sat_conf]
        """
        obs = np.zeros((self.n_agents, self.radar_obs_dim), dtype=np.float32)
        sensor_by_id = agent_obs.dict_equip_state
        track_dict = agent_obs.dict_system_track

        # ---- 卫星广播特征预计算（同一步内所有地面雷达共享） ----
        sat_visible_set = get_satellite_visible_targets(
            agent_obs, self.satellite_ids,
        )
        sat_conf_map = build_satellite_confidence_map(
            agent_obs, self.satellite_ids,
        )

        # 全局锁定计数（所有智能体共享）
        target_lock_counts = {}
        for s in sensor_by_id.values():
            for tid in s.lst_track_no:
                target_lock_counts[tid] = target_lock_counts.get(tid, 0) + 1

        for i, real_sid in enumerate(self.agent_keys):
            s = sensor_by_id.get(real_sid)
            if s is None:
                continue

            # —— 自身特征 ——
            max_cap = max(1, s.track_num_max)
            obs[i, 0] = s.residual_track_num / max_cap

            if s.lst_track_no:
                locked_tid = s.lst_track_no[0]
                ti = self._target_idx.get(locked_tid)
                if ti is not None:
                    obs[i, 1] = (ti + 1) / self.n_actions

            obs[i, 2] = s.range_min / MAX_RANGE_KM
            obs[i, 3] = s.range_max / MAX_RANGE_KM
            obs[i, 4] = s.azi_min / AZI_RANGE
            obs[i, 5] = s.azi_max / AZI_RANGE
            obs[i, 6] = s.ele_min / ELE_RANGE
            obs[i, 7] = s.ele_max / ELE_RANGE

            # 类型标签
            if s.type == 1:
                obs[i, 8] = AGENT_TYPE_RADAR
            elif s.type == 2:
                obs[i, 8] = AGENT_TYPE_SATELLITE

            # —— 目标特征（原有 5 维 + 广播 3 维） ——
            for ti, real_tid in enumerate(self.target_keys):
                res = agent_obs.dict_detection_result.get(real_sid, {}).get(real_tid)
                radar_detectable = res is not None and res.detectable_flag
                sat_detectable = real_tid in sat_visible_set

                # 既不可见也无卫星广播 → 跳过
                if not radar_detectable and not sat_detectable:
                    continue

                base = RADAR_SELF_FEATURES + ti * self._target_stride

                # -- 原有 5 维目标特征 --
                if radar_detectable:
                    obs[i, base + 0] = 1.0

                    t = track_dict.get(real_tid)
                    if t:
                        a, e, r = pm.geodetic2aer(
                            t.latitude, t.longitude, t.altitude,
                            s.latitude, s.longitude, s.altitude
                        )
                        obs[i, base + 1] = (r / 1000.0) / MAX_RANGE_KM
                        obs[i, base + 2] = a / AZI_RANGE
                        obs[i, base + 3] = e / ELE_RANGE

                    obs[i, base + 4] = target_lock_counts.get(real_tid, 0) / max(1, self.n_agents)
                # else: 不可见 → 原有 5 维保持 0

                # -- 广播 3 维 --
                bcast_base = base + TARGET_FEATURES
                obs[i, bcast_base + 0] = 1.0 if sat_detectable else 0.0

                if sat_detectable and not radar_detectable:
                    obs[i, bcast_base + 1] = compute_boundary_gap(
                        agent_obs, real_sid, real_tid,
                    )
                # 雷达已可见 → gap=0（已在范围内，无边界距离）

                obs[i, bcast_base + 2] = sat_conf_map.get(real_tid, 0.0)

        return obs

    # ============================================================
    # 全局状态
    # ============================================================

    def build_global_state(self, agent_obs: AgentObservation) -> np.ndarray:
        """每步调用：构建 (state_dim,) 全局状态向量。"""
        parts = []
        sensor_by_id = agent_obs.dict_equip_state
        track_dict = agent_obs.dict_system_track

        # 目标锁定计数
        target_lock_counts = {}
        for s in sensor_by_id.values():
            for tid in s.lst_track_no:
                target_lock_counts[tid] = target_lock_counts.get(tid, 0) + 1

        # —— 目标状态矩阵 (n_targets × 8) ——
        target_states = np.zeros((self.n_targets, TARGET_STATE_FEATURES), dtype=np.float32)
        for i, real_tid in enumerate(self.target_keys):
            t = track_dict.get(real_tid)
            if t is None:
                continue
            lock_rate = target_lock_counts.get(real_tid, 0) / max(1, self.n_agents)
            target_states[i] = [
                t.ecf_x / ECF_POS_SCALE, t.ecf_y / ECF_POS_SCALE, t.ecf_z / ECF_POS_SCALE,
                t.ecf_vx / ECF_VEL_SCALE, t.ecf_vy / ECF_VEL_SCALE, t.ecf_vz / ECF_VEL_SCALE,
                lock_rate, 1.0,
            ]
        parts.append(target_states.flatten())

        # —— 智能体状态矩阵 (n_agents × 5) ——
        agent_status = np.zeros((self.n_agents, RADAR_STATE_FEATURES), dtype=np.float32)
        for i, real_sid in enumerate(self.agent_keys):
            s = sensor_by_id.get(real_sid)
            if s is None:
                agent_status[i] = [0, 0, 0, 0, -1.0]
                continue

            assigned_idx = -1.0
            if s.lst_track_no:
                locked_tid = s.lst_track_no[0]
                ti = self._target_idx.get(locked_tid)
                if ti is not None:
                    assigned_idx = float(ti) / max(1, self.n_targets)

            max_cap = max(1, s.track_num_max)
            rx, ry, rz = pm.geodetic2ecef(s.latitude, s.longitude, s.altitude)
            rx, ry, rz = rx / 1000.0, ry / 1000.0, rz / 1000.0

            agent_status[i] = [
                rx / ECF_POS_SCALE, ry / ECF_POS_SCALE, rz / ECF_POS_SCALE,
                len(s.lst_track_no) / max_cap,
                assigned_idx,
            ]
        parts.append(agent_status.flatten())

        # —— 时间 ——
        parts.append(np.array([agent_obs.current_time / EPISODE_DURATION], dtype=np.float32))

        return np.concatenate(parts)
