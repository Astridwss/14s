"""卫星广播特征计算 —— 三个独立纯函数，每个只做一件事。

卫星探测结果不直接扩大地面雷达的动作掩码，而是作为观测增强，
让地面雷达提前感知"即将进入范围的目标"，学会预留容量和预判交接时机。

用法（在 ObservationBuilder.build_observations 中调用）:
    sat_visible_set = get_satellite_visible_targets(agent_obs, satellite_ids)
    sat_conf_map   = build_satellite_confidence_map(agent_obs, satellite_ids)
    gap = compute_boundary_gap(agent_obs, radar_id, target_id)
"""

from typing import Dict, List, Set

import pymap3d as pm

from services.scene.scene_constants import MAX_RANGE_KM


# ============================================================
# 方法 1: 卫星目标可见性 —— "这目标存在吗"
# ============================================================

def get_satellite_visible_targets(
    agent_obs, satellite_ids: List[str],
) -> Set[str]:
    """返回至少被一颗卫星探测到的目标 ID 集合。

    Args:
        agent_obs: AgentObservation，含 dict_detection_result
        satellite_ids: 卫星装备 ID 列表

    Returns:
        被任意卫星可见的目标 ID 集合（可能为空）
    """
    visible: Set[str] = set()
    detection = agent_obs.dict_detection_result

    for sat_id in satellite_ids:
        target_dict = detection.get(sat_id, {})
        for tid, res in target_dict.items():
            if res.detectable_flag:
                visible.add(tid)
    return visible


# ============================================================
# 方法 2: 目标与雷达的边界距离 —— "离我多远"
# ============================================================

def compute_boundary_gap(agent_obs, radar_id: str, target_id: str) -> float:
    """计算目标与雷达探测范围边界的归一化距离。

    仅对"卫星可见但雷达不可见"的目标有意义。
    调用方应在 sat_visible=1 且 radar_detectable=0 时才调用。

    公式:
        gap = clamp(distance / range_max - 1, 0.0, 2.0)
        0.0 = 目标刚好在雷达探测范围边界或以内（即将进入）
        1.0 = 目标在 2 倍最大探测距离处
        2.0 = 目标在 3 倍及以上距离（clamp 上限）

    Args:
        agent_obs: AgentObservation
        radar_id: 雷达装备 ID
        target_id: 目标 ID

    Returns:
        归一化边界距离 [0.0, 2.0]
    """
    radar_state = agent_obs.dict_equip_state.get(radar_id)
    track = agent_obs.dict_system_track.get(target_id)

    if radar_state is None or track is None:
        return 2.0

    # 用 pymap3d 计算大地距离，与仿真引擎探测判定一致
    _a, _e, r_meters = pm.geodetic2aer(
        track.latitude, track.longitude, track.altitude,
        radar_state.latitude, radar_state.longitude, radar_state.altitude,
    )
    distance_km = r_meters / 1000.0
    range_max = getattr(radar_state, 'range_max', MAX_RANGE_KM) or MAX_RANGE_KM

    if range_max <= 0:
        return 2.0

    gap = distance_km / range_max - 1.0
    return float(max(0.0, min(2.0, gap)))


# ============================================================
# 方法 3: 卫星置信度 —— "消息有多可靠"
# ============================================================

def build_satellite_confidence_map(
    agent_obs, satellite_ids: List[str],
) -> Dict[str, float]:
    """计算每个目标被多少颗卫星看到，归一化到 [0, 1]。

    多颗卫星确认同一目标 → 置信度高 → DRQN 更倾向于信任此信息。

    Args:
        agent_obs: AgentObservation
        satellite_ids: 卫星装备 ID 列表

    Returns:
        {target_id: confidence_ratio}，未出现的 key 视为 0.0
    """
    n_sats = max(1, len(satellite_ids))
    count: Dict[str, int] = {}
    detection = agent_obs.dict_detection_result

    for sat_id in satellite_ids:
        target_dict = detection.get(sat_id, {})
        for tid, res in target_dict.items():
            if res.detectable_flag:
                count[tid] = count.get(tid, 0) + 1

    return {tid: cnt / n_sats for tid, cnt in count.items()}
