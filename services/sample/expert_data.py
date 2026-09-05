"""
专家数据生成 —— 接收外部构建好的积木，按时间步生成 RL 张量并保存为 CSV。

不调用同层 services/scene/ —— 依赖由调用方显式注入。
"""
import os
import csv
import numpy as np
import pymap3d as pm

from sim.datastruct import (
    AgentObservation, SystemTrackBase, EquipmentState, EquipmentToTargetDetectionResult,
)


def generate_expert_csv(conf, dest_csv_path: str,
                        obs_builder, action_mapper,
                        plan_id: int, scene_file_path: str,
                        radar_keys, target_keys, sat_keys):
    """数据管道：读取预案 → 抽取 RL 张量 → 保存 CSV。

    所有场景相关的积木（obs_builder, action_mapper, keys）由调用方提供，
    本函数只负责时间步遍历和 CSV 写入。

    覆盖雷达(LD)与卫星(WX)两类智能体：
      - 雷达探测用 range/azi/ele 判定（与 sim 内核一致）；
      - 卫星探测用 ECEF 视场夹角判定（复刻 sim/client.py step_forward）；
      - 专家动作从 plan_result 按装备 ID 提取，单目标索引（LD 多标签简化成单目标，
        WX 单选正好吻合）；有调度的一方正常生成，无调度的一方退化为 0（待机）。
    """
    if not scene_file_path or not os.path.exists(scene_file_path):
        raise FileNotFoundError(f"[ExpertData] 致命错误：找不到场景文件 {scene_file_path}")

    print(f"[ExpertData] 开始处理场景预案: {scene_file_path}")

    # ── 读取预案 ──
    from sim.plan_file_process import PlanFileProcess
    processor = PlanFileProcess()
    plan_file_info = processor.read_plan_file_info_from_json(plan_id=plan_id, file_path=scene_file_path)
    battle_scene = plan_file_info.battle_scene
    plan_result = plan_file_info.plan_result

    # ── 专家规划结果非空校验：LD 与 WX 均无调度时拒绝，避免训练出「永远待机」 ──
    radar_set = set(radar_keys)
    sat_set = set(sat_keys or [])
    ld_has = any(eid in radar_set for eid in plan_result.dict_equip_id_target_id_detection_time)
    wx_has = any(eid in sat_set for eid in plan_result.dict_equip_id_target_id_detection_time)
    if not ld_has and not wx_has:
        raise ValueError(
            "[ExpertData] 预案无专家规划结果(splitQuduanResult)：LD 与 WX 均无调度，"
            "IL 数据将全为待机，拒绝训练。"
        )
    print(f"[ExpertData] 专家规划结果覆盖: LD={'有' if ld_has else '无'} / WX={'有' if wx_has else '无'}")

    all_times = []
    for target in battle_scene.dict_target_id_info.values():
        all_times.extend(target.dict_target_traj_pt_info.keys())
    if not all_times:
        print("[ExpertData]: 未能从 JSON 预案中提取出目标的轨迹时间")
        return

    start_time, end_time = min(all_times), max(all_times)

    # ── 4. 按时间步生成数据 ──
    os.makedirs(os.path.dirname(dest_csv_path), exist_ok=True)
    with open(dest_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['time_step', 'agent_id', 'expert_action', 'obs', 'state', 'avail_actions'])

        for t in range(int(start_time), int(end_time) + 1):
            raw_obs = AgentObservation()
            raw_obs.current_time = float(t)
            alive_targets = []

            # A. 填充目标航迹（每目标新建 trk，避免共享引用污染）
            for tgt_id, target_info in battle_scene.dict_target_id_info.items():
                if t in target_info.dict_target_traj_pt_info:
                    pt = target_info.dict_target_traj_pt_info[t]
                    trk = SystemTrackBase(
                        detect_time=int(t),
                        str_system_track_no=tgt_id,
                        longitude=pt.longitude,
                        latitude=pt.latitude,
                        altitude=pt.altitude,
                        ecf_x=pt.ecf_x,
                        ecf_y=pt.ecf_y,
                        ecf_z=pt.ecf_z,
                        ecf_vx=pt.ecf_vx,
                        ecf_vy=pt.ecf_vy,
                        ecf_vz=pt.ecf_vz,
                        rcs=pt.rcs,
                        type=pt.type,
                    )
                    raw_obs.dict_system_track[tgt_id] = trk
                    alive_targets.append(tgt_id)

            # B. 填充雷达状态与可视性（每雷达新建 s，避免共享引用污染）
            for radar_id in radar_keys:
                radar_info = battle_scene.dict_radar_id_info[radar_id]
                s = EquipmentState()
                s.time = int(t)
                s.str_equip_id = radar_id
                s.latitude = radar_info.latitude
                s.longitude = radar_info.longitude
                s.altitude = radar_info.altitude
                s.range_min, s.range_max = radar_info.range_min, radar_info.range_max
                s.azi_min, s.azi_max = radar_info.azi_min, radar_info.azi_max
                s.ele_min, s.ele_max = radar_info.ele_min, radar_info.ele_max
                s.azi_pointing = (radar_info.azi_min + radar_info.azi_max) / 2
                s.ele_pointing = (radar_info.ele_min + radar_info.ele_max) / 2
                s.type = 1
                s.track_num_max = radar_info.track_num_max
                s.lst_track_no = []

                if radar_id in plan_result.dict_equip_id_target_id_detection_time:
                    for tgt_id, time_ranges in \
                            plan_result.dict_equip_id_target_id_detection_time[radar_id].items():
                        for tr in time_ranges:
                            if tr.time_range.value_min <= t <= tr.time_range.value_max:
                                if tgt_id in target_keys:
                                    s.lst_track_no.append(tgt_id)
                                break

                s.residual_track_num = max(0, s.track_num_max - len(s.lst_track_no))
                raw_obs.dict_equip_state[radar_id] = s

                raw_obs.dict_detection_result[radar_id] = {}
                for tgt_id in alive_targets:
                    dr = EquipmentToTargetDetectionResult()
                    dr.str_equip_id = radar_id
                    dr.str_target_id = tgt_id
                    pt = battle_scene.dict_target_id_info[tgt_id].dict_target_traj_pt_info[t]
                    a, e, r = pm.geodetic2aer(
                        pt.latitude, pt.longitude, pt.altitude,
                        radar_info.latitude, radar_info.longitude, radar_info.altitude,
                    )
                    if (radar_info.azi_min <= a <= radar_info.azi_max and
                            radar_info.ele_min <= e <= radar_info.ele_max and
                            radar_info.range_min <= (r / 1000.0) <= radar_info.range_max):
                        dr.detectable_flag = True
                    else:
                        dr.detectable_flag = False
                    raw_obs.dict_detection_result[radar_id][tgt_id] = dr

            # C. 填充卫星状态与可视性（复刻 sim/client.py step_forward 的卫星逻辑）
            for sat_id in sat_keys:
                sat_info = battle_scene.dict_satellite_id_info.get(sat_id)
                if sat_info is None:
                    continue

                s = EquipmentState()
                s.time = int(t)
                s.str_equip_id = sat_id

                if t in sat_info.dict_satellite_traj_pt_info:
                    sat_pt = sat_info.dict_satellite_traj_pt_info[t]
                    s.longitude = sat_pt.longitude
                    s.latitude = sat_pt.latitude
                    s.altitude = sat_pt.altitude

                s.range_min = 0.0
                s.range_max = 36000.0
                s.azi_min = sat_info.azi_min
                s.azi_max = sat_info.azi_max
                s.ele_min = sat_info.ele_min
                s.ele_max = sat_info.ele_max
                s.azi_pointing = (sat_info.azi_min + sat_info.azi_max) / 2
                s.ele_pointing = (sat_info.ele_min + sat_info.ele_max) / 2
                s.type = 2
                s.track_num_max = sat_info.track_num_max
                s.lst_track_no = []

                if sat_id in plan_result.dict_equip_id_target_id_detection_time:
                    for tgt_id, time_ranges in \
                            plan_result.dict_equip_id_target_id_detection_time[sat_id].items():
                        for tr in time_ranges:
                            if tr.time_range.value_min <= t <= tr.time_range.value_max:
                                if tgt_id in target_keys:
                                    s.lst_track_no.append(tgt_id)
                                break

                s.residual_track_num = max(0, s.track_num_max - len(s.lst_track_no))
                raw_obs.dict_equip_state[sat_id] = s

                raw_obs.dict_detection_result[sat_id] = {}
                for tgt_id in alive_targets:
                    dr = EquipmentToTargetDetectionResult()
                    dr.str_equip_id = sat_id
                    dr.str_target_id = tgt_id
                    pt = battle_scene.dict_target_id_info[tgt_id].dict_target_traj_pt_info[t]

                    sat_xyz = pm.geodetic2ecef(s.latitude, s.longitude, s.altitude)
                    target_xyz = pm.geodetic2ecef(pt.latitude, pt.longitude, pt.altitude)
                    earth_center = np.array([0.0, 0.0, 0.0])
                    vec_cam = earth_center - sat_xyz
                    vec_target = np.array([target_xyz[0], target_xyz[1], target_xyz[2]]) - np.array([sat_xyz[0], sat_xyz[1], sat_xyz[2]])

                    norm_cam = np.linalg.norm(vec_cam)
                    norm_target = np.linalg.norm(vec_target)
                    if norm_cam == 0 or norm_target == 0:
                        dr.detectable_flag = False
                    else:
                        cos_theta = np.dot(vec_cam, vec_target) / (norm_cam * norm_target)
                        cos_theta = np.clip(cos_theta, -1.0, 1.0)
                        theta_degrees = np.degrees(np.arccos(cos_theta))
                        dr.detectable_flag = theta_degrees < s.azi_max
                    raw_obs.dict_detection_result[sat_id][tgt_id] = dr

            # D. 产出张量
            obs_tensors = obs_builder.build_observations(raw_obs)
            avail_actions = action_mapper.build_action_mask(raw_obs)
            state_tensor = obs_builder.build_global_state(raw_obs)
            state_str = "|".join(map(str, state_tensor.tolist()))

            # E. 提取专家动作 & 掩码校正（雷达 LD + 卫星 WX，统一按 agent_keys 顺序）
            for agent_idx, equip_id in enumerate(list(radar_keys) + list(sat_keys)):
                expert_action = 0
                if equip_id in plan_result.dict_equip_id_target_id_detection_time:
                    for tgt_id, time_ranges in \
                            plan_result.dict_equip_id_target_id_detection_time[equip_id].items():
                        for tr in time_ranges:
                            if tr.time_range.value_min <= t <= tr.time_range.value_max:
                                if tgt_id in target_keys:
                                    expert_action = target_keys.index(tgt_id) + 1
                                break
                        if expert_action != 0:
                            break

                if expert_action > 0 and avail_actions[agent_idx, expert_action] == 0.0:
                    expert_action = 0  # 目标在盲区，纠正为待机

                obs_arr_str = "|".join(map(str, obs_tensors[agent_idx].tolist()))
                avail_arr_str = "|".join(map(str, avail_actions[agent_idx].tolist()))
                writer.writerow([t, agent_idx, expert_action, obs_arr_str, state_str, avail_arr_str])

    print(f"[ExpertData] CSV 数据集生成完毕，已保存至 {dest_csv_path}")


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    _project_root = Path(__file__).resolve().parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    print("================  启动专家数据生成流水线 ================")

    test_task_id = "ILtest867"
    local_scene_path = os.path.join(str(_project_root), "scenarios", test_task_id, "scene.json")
    dest_csv_path = os.path.join(str(_project_root), "data", test_task_id, "expert_data.csv")

    if not os.path.exists(local_scene_path):
        print(f"找不到场景文件 {local_scene_path}")
        sys.exit(1)

    class DummyConf:
        pass

    mock_conf = DummyConf()
    mock_conf.plan_id = 867
    mock_conf.local_scene_path = local_scene_path

    try:
        generate_expert_csv(mock_conf, dest_csv_path)
    except Exception as e:
        print(f"生成 CSV 失败: {e}")
