"""
观测构建 —— AgentObservation → 神经网络输入张量。

固定槽位 + dense-then-scatter
- 槽位映射：雷达 → [0, 200)，卫星 → [200, 283)（与 mapper.py 一致）
- 内部先用真实数 n_real 做向量化 dense 计算，再 scatter 到 (283, 177) 的 MAX 矩阵；dummy 槽位补零
- 关键回归点：lock_rate 分母用 _n_agents_real（250）而非 MAX_AGENTS（283），保证满载时 obs 数值零回归（此时 dense==slot，逐位一致）
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
    MAX_TARGETS, MAX_AGENTS, MAX_ACTIONS,
    build_agent_slot_map,
)
from services.scene.state.satellite_broadcast import (
    get_satellite_visible_targets,
    build_satellite_confidence_map,
)


class ObservationBuilder:
    """将仿真引擎的 AgentObservation 转为规整 numpy 张量。"""

    def __init__(self, agent_keys, target_keys,
                 n_agents, n_targets, n_actions, radar_obs_dim,
                 satellite_ids: Optional[List[str]] = None):
        self.agent_keys = list(agent_keys)
        self.target_keys = list(target_keys)
        self.satellite_ids = satellite_ids or []

        # 可变实体数泛化：输出维度恒按 MAX 固定槽位（与 ActionMapper / 网络权重一致），
        # 真实实体数不足时用 padding 占满槽位。传入的 n_agents/n_targets/n_actions 仅保留
        # 兼容旧调用方，维度一律以 MAX_* 常量为准。
        self.n_agents = MAX_AGENTS      # 283
        self.n_targets = MAX_TARGETS    # 21
        self.n_actions = MAX_ACTIONS    # 22

        # 观测维度 = 自身 + 目标特征(含卫星广播) × MAX_TARGETS
        self.radar_obs_dim = (
            RADAR_SELF_FEATURES
            + MAX_TARGETS * (TARGET_FEATURES + SATELLITE_BROADCAST_FEATURES)
        )

        # 每目标特征步长（含广播）
        self._target_stride = TARGET_FEATURES + SATELLITE_BROADCAST_FEATURES

        # 真实实体数（归一化分母 / 真实遍历用，避免把 dummy 槽位计入）
        self._n_agents_real = len(self.agent_keys)
        self._n_targets_real = len(self.target_keys)

        # 固定槽位映射（与 ActionMapper / RewardCalculator 同一约定）：
        #   雷达 → agent 槽位 [0, MAX_RADARS)，卫星 → [MAX_RADARS, MAX_AGENTS)
        self._agent_slot = build_agent_slot_map(self.agent_keys, self.satellite_ids)

        # 密集索引（真实顺序 0..n_real-1，用于向量化紧凑计算后再 scatter 到槽位）
        self._agent_dense = {sid: i for i, sid in enumerate(self.agent_keys)}
        self._target_idx = {tid: i for i, tid in enumerate(self.target_keys)}

    # ============================================================
    # 局部观测
    # ============================================================

    def build_observations(self, agent_obs: AgentObservation) -> np.ndarray:
        """每步调用：构建 (MAX_AGENTS, obs_dim) 观测矩阵。

        观测布局（每个目标）:
          [原有 5 维: detectable, distance, azi, ele, lock_rate]
          [广播 3 维: sat_visible, boundary_gap, sat_conf]

        可变实体数泛化：真实实体在密集空间（0..n_real-1）向量化计算，再 scatter 到
        固定槽位（雷达 [0, MAX_RADARS)，卫星 [MAX_RADARS, MAX_AGENTS)）；dummy 槽位补零。
        """
        n_real = self._n_agents_real
        n_t_real = self._n_targets_real
        obs_compact = np.zeros((n_real, self.radar_obs_dim), dtype=np.float32)
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

        # —— 向量化预备：一次批量算所有 (智能体, 目标) 的距离/方位/俯仰，替代逐对标量 geodetic2aer ——
        _ag = [sensor_by_id.get(sid) for sid in self.agent_keys]
        ag_lats = np.array([s.latitude if s is not None else 0.0 for s in _ag], dtype=np.float64)
        ag_lons = np.array([s.longitude if s is not None else 0.0 for s in _ag], dtype=np.float64)
        ag_alts = np.array([s.altitude if s is not None else 0.0 for s in _ag], dtype=np.float64)

        _trk = [track_dict.get(tid) for tid in self.target_keys]
        tg_lats = np.array([t.latitude if t is not None else 0.0 for t in _trk], dtype=np.float64)
        tg_lons = np.array([t.longitude if t is not None else 0.0 for t in _trk], dtype=np.float64)
        tg_alts = np.array([t.altitude if t is not None else 0.0 for t in _trk], dtype=np.float64)

        # 每个目标一次向量化 geodetic2aer，得到 (n_real, n_t_real) 的距离(km)/方位/俯仰
        dist_km = np.zeros((n_real, n_t_real), dtype=np.float64)
        az_deg = np.zeros((n_real, n_t_real), dtype=np.float64)
        el_deg = np.zeros((n_real, n_t_real), dtype=np.float64)
        for ti in range(n_t_real):
            a, e, r = pm.geodetic2aer(
                tg_lats[ti], tg_lons[ti], tg_alts[ti],
                ag_lats, ag_lons, ag_alts,
            )
            az_deg[:, ti] = a
            el_deg[:, ti] = e
            dist_km[:, ti] = r / 1000.0

        # —— 探测矩阵 (n_real, n_t_real)：从 dict_detection_result 一次性展开 ——
        radar_detect = np.zeros((n_real, n_t_real), dtype=bool)
        for sid, tdict in agent_obs.dict_detection_result.items():
            ai = self._agent_dense.get(sid)
            if ai is None or not tdict:
                continue
            idxs = []
            flags = []
            for tid, res in tdict.items():
                ti = self._target_idx.get(tid)
                if ti is not None:
                    idxs.append(ti)
                    flags.append(res.detectable_flag)
            if idxs:
                radar_detect[ai, np.asarray(idxs, dtype=np.intp)] = np.asarray(flags, dtype=bool)

        # —— 自身特征（逐智能体标量 → 向量化赋值到 obs_compact[:, 0..8]） ——
        max_cap = np.array([max(1, s.track_num_max) if s is not None else 1 for s in _ag], dtype=np.float64)
        residual = np.array([s.residual_track_num if s is not None else 0.0 for s in _ag], dtype=np.float64)
        rmin = np.array([s.range_min if s is not None else 0.0 for s in _ag], dtype=np.float64)
        rmax = np.array([s.range_max if s is not None else MAX_RANGE_KM for s in _ag], dtype=np.float64)
        amin = np.array([s.azi_min if s is not None else 0.0 for s in _ag], dtype=np.float64)
        amax = np.array([s.azi_max if s is not None else 0.0 for s in _ag], dtype=np.float64)
        emin = np.array([s.ele_min if s is not None else 0.0 for s in _ag], dtype=np.float64)
        emax = np.array([s.ele_max if s is not None else 0.0 for s in _ag], dtype=np.float64)

        first_locked = np.zeros(n_real, dtype=np.float64)
        typ = np.zeros(n_real, dtype=np.float64)
        for i, s in enumerate(_ag):
            if s is None:
                continue
            if s.lst_track_no:
                ti = self._target_idx.get(s.lst_track_no[0])
                if ti is not None:
                    first_locked[i] = (ti + 1) / self.n_actions
            if s.type == 1:
                typ[i] = AGENT_TYPE_RADAR
            elif s.type == 2:
                typ[i] = AGENT_TYPE_SATELLITE

        obs_compact[:, 0] = residual / max_cap
        obs_compact[:, 1] = first_locked
        obs_compact[:, 2] = rmin / MAX_RANGE_KM
        obs_compact[:, 3] = rmax / MAX_RANGE_KM
        obs_compact[:, 4] = amin / AZI_RANGE
        obs_compact[:, 5] = amax / AZI_RANGE
        obs_compact[:, 6] = emin / ELE_RANGE
        obs_compact[:, 7] = emax / ELE_RANGE
        obs_compact[:, 8] = typ

        # —— 目标特征（原有 5 维 + 广播 3 维，向量化填充） ——
        base_idx = RADAR_SELF_FEATURES + np.arange(n_t_real) * self._target_stride
        bcast_idx = base_idx + TARGET_FEATURES

        trk_exists = np.array([tid in track_dict for tid in self.target_keys], dtype=bool)
        sat_vis = np.array([tid in sat_visible_set for tid in self.target_keys], dtype=np.float32)
        sat_conf = np.array([sat_conf_map.get(tid, 0.0) for tid in self.target_keys], dtype=np.float32)
        lock_rate = np.array(
            [target_lock_counts.get(tid, 0) / max(1, self._n_agents_real) for tid in self.target_keys],
            dtype=np.float32,
        )

        # 距离/方位/俯仰仅在「雷达可见且航迹存在」时写；锁定率在「雷达可见」即写（对齐原逻辑）
        fill_123 = radar_detect & trk_exists[None, :]
        rd_f32 = radar_detect.astype(np.float32)

        obs_compact[:, base_idx + 0] = rd_f32
        obs_compact[:, base_idx + 1] = np.where(fill_123, dist_km / MAX_RANGE_KM, 0.0)
        obs_compact[:, base_idx + 2] = np.where(fill_123, az_deg / AZI_RANGE, 0.0)
        obs_compact[:, base_idx + 3] = np.where(fill_123, el_deg / ELE_RANGE, 0.0)
        obs_compact[:, base_idx + 4] = np.where(radar_detect, lock_rate[None, :], 0.0)

        obs_compact[:, bcast_idx + 0] = sat_vis[None, :]
        # boundary_gap：仅在「卫星可见且雷达不可见」时写 clamp(dist/range_max - 1, 0, 2)。
        # 直接复用已算好的 dist_km，替代 compute_boundary_gap 的逐对标量 geodetic2aer（原开销源）。
        rmax_eff = np.where(rmax == 0.0, MAX_RANGE_KM, rmax)  # 对齐原 `range_max or MAX_RANGE_KM`
        with np.errstate(divide='ignore', invalid='ignore'):
            gap = dist_km / rmax_eff[:, None] - 1.0
        gap = np.clip(gap, 0.0, 2.0)
        gap = np.where((rmax_eff > 0.0)[:, None] & trk_exists[None, :], gap, 2.0)
        cond = sat_vis[None, :].astype(bool) & (~radar_detect)
        obs_compact[:, bcast_idx + 1] = np.where(cond, gap, 0.0)
        obs_compact[:, bcast_idx + 2] = sat_conf[None, :]

        # —— scatter：真实实体从密集空间搬到固定槽位，dummy 槽位补零 ——
        obs = np.zeros((self.n_agents, self.radar_obs_dim), dtype=np.float32)
        for dense_i, sid in enumerate(self.agent_keys):
            obs[self._agent_slot[sid]] = obs_compact[dense_i]
        return obs

    # ============================================================
    # 全局状态
    # ============================================================

    def build_global_state(self, agent_obs: AgentObservation) -> np.ndarray:
        """每步调用：构建 (state_dim,) 全局状态向量。

        可变实体数泛化：目标段恒 MAX_TARGETS×8、智能体段恒 MAX_AGENTS×5；
        真实目标写到 0..n_t_real-1，真实智能体写到固定槽位，dummy 补零。
        """
        parts = []
        sensor_by_id = agent_obs.dict_equip_state
        track_dict = agent_obs.dict_system_track

        # 目标锁定计数
        target_lock_counts = {}
        for s in sensor_by_id.values():
            for tid in s.lst_track_no:
                target_lock_counts[tid] = target_lock_counts.get(tid, 0) + 1

        # —— 目标状态矩阵 (MAX_TARGETS × 8) ——
        target_states = np.zeros((self.n_targets, TARGET_STATE_FEATURES), dtype=np.float32)
        for i, real_tid in enumerate(self.target_keys):
            t = track_dict.get(real_tid)
            if t is None:
                continue
            lock_rate = target_lock_counts.get(real_tid, 0) / max(1, self._n_agents_real)
            target_states[i] = [
                t.ecf_x / ECF_POS_SCALE, t.ecf_y / ECF_POS_SCALE, t.ecf_z / ECF_POS_SCALE,
                t.ecf_vx / ECF_VEL_SCALE, t.ecf_vy / ECF_VEL_SCALE, t.ecf_vz / ECF_VEL_SCALE,
                lock_rate, 1.0,
            ]
        parts.append(target_states.flatten())

        # —— 智能体状态矩阵 (MAX_AGENTS × 5) ——
        agent_status = np.zeros((self.n_agents, RADAR_STATE_FEATURES), dtype=np.float32)

        # 向量化：批量 geodetic2ecef 替代逐智能体标量调用
        _ag = [sensor_by_id.get(sid) for sid in self.agent_keys]
        ag_lats = np.array([s.latitude if s is not None else 0.0 for s in _ag], dtype=np.float64)
        ag_lons = np.array([s.longitude if s is not None else 0.0 for s in _ag], dtype=np.float64)
        ag_alts = np.array([s.altitude if s is not None else 0.0 for s in _ag], dtype=np.float64)
        _rx, _ry, _rz = pm.geodetic2ecef(ag_lats, ag_lons, ag_alts)
        _rx = np.asarray(_rx) / 1000.0
        _ry = np.asarray(_ry) / 1000.0
        _rz = np.asarray(_rz) / 1000.0

        for i, real_sid in enumerate(self.agent_keys):
            s = sensor_by_id.get(real_sid)
            slot = self._agent_slot[real_sid]
            if s is None:
                agent_status[slot] = [0, 0, 0, 0, -1.0]
                continue

            assigned_idx = -1.0
            if s.lst_track_no:
                locked_tid = s.lst_track_no[0]
                ti = self._target_idx.get(locked_tid)
                if ti is not None:
                    assigned_idx = float(ti) / max(1, self.n_targets)

            max_cap = max(1, s.track_num_max)
            agent_status[slot] = [
                _rx[i] / ECF_POS_SCALE, _ry[i] / ECF_POS_SCALE, _rz[i] / ECF_POS_SCALE,
                len(s.lst_track_no) / max_cap,
                assigned_idx,
            ]
        parts.append(agent_status.flatten())

        # —— 时间 ——
        parts.append(np.array([agent_obs.current_time / EPISODE_DURATION], dtype=np.float32))

        return np.concatenate(parts)
