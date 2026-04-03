import os
import sys
import csv
import pymap3d as pm
from pathlib import Path

# 确保能找到根目录下的 sim 和 env
_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from env.adapter import ScenarioAdapter
from sim.plan_file_process import PlanFileProcess
from sim.datastruct import (
    AgentObservation, SystemTrackBase, EquipmentState, EquipmentToTargetDetectionResult
)


def generate_expert_csv(conf, dest_csv_path: str):
    """
    数据管道：读取预案 -> 抽取 RL 张量 -> 保存 CSV
    """
    plan_id = getattr(conf, 'plan_id', 867)
    scene_file_path = getattr(conf, 'local_scene_path', '')
    
    if not scene_file_path or not os.path.exists(scene_file_path):
        raise FileNotFoundError(f"[DataPipeline] 致命错误：找不到场景文件 {scene_file_path}，无法生成专家数据")
        
    print(f"[DataPipeline] 开始处理场景预案: {scene_file_path}")
    
    ScenarioAdapter.parse_and_inject(conf)
    
    radar_keys = conf.radar_keys
    target_keys = conf.target_keys
    
    adapter = ScenarioAdapter(conf)
    
    processor = PlanFileProcess()
    plan_file_info = processor.read_plan_file_info_from_json(plan_id=plan_id, file_path=scene_file_path)
    battle_scene = plan_file_info.battle_scene
    plan_result = plan_file_info.plan_result
    
    # 提取时间轴
    all_times = []
    for target in battle_scene.dict_target_id_info.values():
        all_times.extend(target.dict_target_traj_pt_info.keys())
        
    if not all_times: 
        print("[DataPipeline]: 未能从 JSON 预案中提取出目标的轨迹时间")
        return
        
    start_time, end_time = min(all_times), max(all_times)

    # 3. 开始按时间步生成数据
    os.makedirs(os.path.dirname(dest_csv_path), exist_ok=True)
    with open(dest_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['time_step', 'agent_id', 'expert_action', 'obs', 'state', 'avail_actions'])
        
        for t in range(int(start_time), int(end_time) + 1):
            raw_obs = AgentObservation()
            raw_obs.current_time = float(t)
            
            alive_targets = []
            
            # ======================A. 填充真实目标航迹===================================
            for tgt_id, target_info in battle_scene.dict_target_id_info.items():
                if t in target_info.dict_target_traj_pt_info:
                    pt = target_info.dict_target_traj_pt_info[t]
                    
                    trk = SystemTrackBase()
                    trk.detect_time = int(t)
                    trk.str_system_track_no = tgt_id 
                    trk.latitude = pt.latitude
                    trk.longitude = pt.longitude
                    trk.altitude = pt.altitude
                    
                    ecf_x_m, ecf_y_m, ecf_z_m = pm.geodetic2ecef(pt.latitude, pt.longitude, pt.altitude)
                    trk.ecf_x = ecf_x_m / 1000.0
                    trk.ecf_y = ecf_y_m / 1000.0
                    trk.ecf_z = ecf_z_m / 1000.0
                    trk.ecf_vx, trk.ecf_vy, trk.ecf_vz = 0.0, 0.0, 0.0
                    trk.rcs = getattr(pt, 'rcs', 1.0)
                    trk.type = getattr(pt, 'type', 1)
                    
                    raw_obs.dict_system_track[tgt_id] = trk
                    alive_targets.append(tgt_id)
                    
            # =====================B. 填充雷达状态与可视性矩阵================================
            for agent_idx, radar_id in enumerate(radar_keys):
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
                s.track_num_max = radar_info.track_num_max
                s.lst_track_no = [] 
                
                if radar_id in plan_result.dict_equip_id_target_id_detection_time:
                    for tgt_id, time_ranges in plan_result.dict_equip_id_target_id_detection_time[radar_id].items():
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
                        radar_info.latitude, radar_info.longitude, radar_info.altitude
                    )
                    
                    if (radar_info.azi_min <= a <= radar_info.azi_max and 
                        radar_info.ele_min <= e <= radar_info.ele_max and 
                        radar_info.range_min <= (r / 1000.0) <= radar_info.range_max):
                        dr.detectable_flag = True
                    else:
                        dr.detectable_flag = False
                        
                    raw_obs.dict_detection_result[radar_id][tgt_id] = dr

            # =====================C. 产出张量====================================
            obs_tensors = adapter.extract_observations(raw_obs)
            avail_actions = adapter.extract_action_masks(raw_obs)
            state_tensor = adapter.extract_global_state(raw_obs)
            state_str = "|".join(map(str, state_tensor.tolist()))

            # =====================D. 提取专家动作并写入 CSV=====================
            for agent_idx, radar_id in enumerate(radar_keys):
                expert_action = 0 
                if radar_id in plan_result.dict_equip_id_target_id_detection_time:
                    for tgt_id, time_ranges in plan_result.dict_equip_id_target_id_detection_time[radar_id].items():
                        for tr in time_ranges:
                            if tr.time_range.value_min <= t <= tr.time_range.value_max:
                                if tgt_id in target_keys:
                                    expert_action = target_keys.index(tgt_id) + 1
                                break
                        if expert_action != 0: 
                            break
                
                # ================== 【核心新增逻辑：用物理掩码纠正专家】 ==================
                # 如果专家要求跟踪某个目标 (expert_action > 0)
                if expert_action > 0:
                    # 去 avail_actions 掩码里查一下，这个目标现在到底能不能看见？
                    # avail_actions 形状是 (n_radars, n_actions)，掩码 0.0 代表不可见
                    if avail_actions[agent_idx, expert_action] == 0.0:
                        # 专家在瞎指挥！目标已经在物理盲区，强行将标签纠正为 0 (待机)
                        expert_action = 0
                # =========================================================================

                obs_arr_str = "|".join(map(str, obs_tensors[agent_idx].tolist()))
                avail_arr_str = "|".join(map(str, avail_actions[agent_idx].tolist()))
                writer.writerow([t, agent_idx, expert_action, obs_arr_str, state_str, avail_arr_str])

    print(f"[DataPipeline]CSV 数据集生成完毕，已保存至 {dest_csv_path}")

if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path
    
    _project_root = Path(__file__).resolve().parent.parent
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