"""卫星视场覆盖计算 —— 推演态势绘制补充信息（不参与训练）。

纯函数：从当前帧 raw_obs + 静态卫星信息 + 当前锁定关系，算出每颗卫星
视场内覆盖到的目标（排除已锁定的），供态势推送复用 Detection 字段画覆盖连线。

复用 sim/satellite_fov_calculation.SatelliteFovCalculation 的几何计算，
与离线后处理 update_model_inference_satellite_result 同源，但这里是"单帧实时"版本，
不做整局 deepcopy、不污染 agent_actions_list、不改 proto。
"""

from typing import Dict, List, Tuple

import numpy as np

from sim.satellite_fov_calculation import SatelliteFovCalculation


def compute_satellite_fov_targets(
    raw_obs,
    satellite_info: Dict,
    agent_actions: List,
) -> List[Tuple[str, str]]:
    """计算当前帧每颗卫星视场内覆盖的目标。

    返回 [(sat_id, target_id), ...]，仅包含"视场内且非当前锁定"的目标，
    供 build_situation_frame 追加进 Detection（复用现有 proto 字段，不改 proto）。

    Args:
        raw_obs: AgentObservation，含 dict_equip_state / dict_system_track / current_time
        satellite_info: {sat_id: SatelliteInfo} 静态卫星信息（视场角、轨迹、最大指向角）
        agent_actions: 当前 step 真实动作列表（List[AgentActionCommand]）

    说明：卫星只有在"锁定某目标"时才有相机指向与视场；待机卫星不产生覆盖连线。
    """
    time = int(raw_obs.current_time)

    # 当前帧所有有位置的目标（target_ids 与 targets 索引对齐）
    target_ids = list(raw_obs.dict_system_track.keys())
    targets = [
        np.array([trk.longitude, trk.latitude, trk.altitude], dtype=np.float64)
        for trk in raw_obs.dict_system_track.values()
    ]
    if not targets:
        return []

    calc = SatelliteFovCalculation()
    fov_pairs: List[Tuple[str, str]] = []

    for cmd in agent_actions:
        sat_id = getattr(cmd, 'str_equip_id', '')
        locked_tid = getattr(cmd, 'str_target_id', '')
        if not sat_id or sat_id not in satellite_info or locked_tid in ("", "0"):
            continue  # 非卫星，或卫星待机无指向

        sat_info = satellite_info[sat_id]
        equip = raw_obs.dict_equip_state.get(sat_id)
        center_trk = raw_obs.dict_system_track.get(locked_tid)
        if equip is None or center_trk is None:
            continue

        # 卫星上一时刻位置（建立相机参考系用；time==0 时无上一时刻）
        sat_last = None
        if time > 0 and (time - 1) in sat_info.dict_satellite_traj_pt_info:
            pt = sat_info.dict_satellite_traj_pt_info[time - 1]
            sat_last = np.array(
                [pt.longitude, pt.latitude, pt.altitude], dtype=np.float64,
            )

        try:
            in_fov = calc.find_targets_in_fov(
                sat_current_geo_pos=np.array(
                    [equip.longitude, equip.latitude, equip.altitude],
                    dtype=np.float64,
                ),
                fov_az=sat_info.azi_max - sat_info.azi_min,
                fov_el=sat_info.ele_max - sat_info.ele_min,
                center_target_geo_pos=np.array(
                    [center_trk.longitude, center_trk.latitude, center_trk.altitude],
                    dtype=np.float64,
                ),
                max_pointing_angle=sat_info.camera_pointing_max,
                targets=targets,
                sat_last_geo_pos=sat_last,
            )
        except (ValueError, ZeroDivisionError):
            # 几何退化（位置零向量 / 重合），单颗卫星算失败不影响整体
            continue

        for idx in in_fov:
            tid = target_ids[idx]
            if tid != locked_tid:
                fov_pairs.append((sat_id, tid))

    return fov_pairs
