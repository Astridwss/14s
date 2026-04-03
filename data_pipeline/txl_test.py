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
from sim.datastruct import (
    PlanFileInfo, BattleScene, EquipmentDetectionTime, PlanResult, 
    AgentActionCommand, TargetTrajPtInfo, TargetInfo, SensorInfo, SatelliteInfo,
)


def write_plan_file_info_to_csv(plan_file_info: PlanFileInfo, dest_path: str) -> None:
    # csv文件字段定义：[时间，[当前装备信息]，[目标信息]×21，[预案中跟踪的目标编号]]
    lst_field_name = ['Time', 'EquipID', 'RestResourcePer', 'CurrentTrackID', 'Rmin', 'Rmax', 'Amin', 'Amax', 'Emin', 'Emax']
    for i in range(1, 22):
        lst_field_name.append('TargetID' + str(i))
        lst_field_name.append('R' + str(i))
        lst_field_name.append('A' + str(i))
        lst_field_name.append('E' + str(i))
        lst_field_name.append('Detectable' + str(i))

    lst_field_name.append('PlanTrackID')

    lst_csv_data = []
    start_time = 0
    end_time = int((plan_file_info.battle_scene.end_time - plan_file_info.battle_scene.start_time) / 1000)
    time_step = 1

    dict_equip_id_rest_resource: Dict[str, float] = {}
    dict_equip_id_current_track_id: Dict[str, str] = {}
    for radar_info in plan_file_info.battle_scene.dict_radar_id_info.values():
        dict_equip_id_rest_resource[radar_info.str_sensor_id] = 1.0
        dict_equip_id_current_track_id[radar_info.str_sensor_id] = '0'

    for satellite_info in plan_file_info.battle_scene.dict_satellite_id_info.values():
        dict_equip_id_rest_resource[satellite_info.str_satellite_id] = 1.0
        dict_equip_id_current_track_id[satellite_info.str_satellite_id] = '0'

    for t in range(start_time, end_time + 1, time_step):
        # 判断t时刻是否所有目标均无轨迹点，若是则不生成样本
        write_csv_flag = False
        for target_info in plan_file_info.battle_scene.dict_target_id_info.values():
            if t in target_info.dict_target_traj_pt_info:
                write_csv_flag = True
                break

        if not write_csv_flag:
            continue

        # 遍历雷达
        for radar_info in plan_file_info.battle_scene.dict_radar_id_info.values():
            dict_current_data = {}
            dict_current_data['Time'] = t
            dict_current_data['EquipID'] = radar_info.str_sensor_id
            dict_current_data['RestResourcePer'] = dict_equip_id_rest_resource.get(radar_info.str_sensor_id)
            dict_current_data['CurrentTrackID'] = dict_equip_id_current_track_id.get(radar_info.str_sensor_id)
            dict_current_data['Rmin'] = radar_info.range_min
            dict_current_data['Rmax'] = radar_info.range_max
            dict_current_data['Amin'] = radar_info.azi_min
            dict_current_data['Amax'] = radar_info.azi_max
            dict_current_data['Emin'] = radar_info.ele_min
            dict_current_data['Emax'] = radar_info.ele_max

            # 遍历目标
            target_no = 1
            for target_info in plan_file_info.battle_scene.dict_target_id_info.values():
                dict_current_data['TargetID' + str(target_no)] = target_info.str_target_id

                if t in target_info.dict_target_traj_pt_info:
                    traj_pt_info = target_info.dict_target_traj_pt_info.get(t)

                    a, e, r = pm.geodetic2aer(traj_pt_info.latitude, traj_pt_info.longitude, traj_pt_info.altitude,
                                                radar_info.latitude, radar_info.longitude, radar_info.altitude)
                    dict_current_data['R' + str(target_no)] = r
                    dict_current_data['A' + str(target_no)] = a
                    dict_current_data['E' + str(target_no)] = e

                    if radar_info.azi_min <= a <= radar_info.azi_max and radar_info.ele_min <= e <= radar_info.ele_max \
                            and radar_info.range_min <= r / 1000.0 <= radar_info.range_max:
                        dict_current_data['Detectable' + str(target_no)] = 1
                    else:
                        dict_current_data['Detectable' + str(target_no)] = 0

                else:
                    dict_current_data['R' + str(target_no)] = 0.0
                    dict_current_data['A' + str(target_no)] = 0.0
                    dict_current_data['E' + str(target_no)] = 0.0
                    dict_current_data['Detectable' + str(target_no)] = 0

                target_no += 1

            # 补齐目标数据
            if target_no < 22:
                for j in range(target_no, 22):
                    dict_current_data['TargetID' + str(j)] = '0'
                    dict_current_data['R' + str(j)] = 0.0
                    dict_current_data['A' + str(j)] = 0.0
                    dict_current_data['E' + str(j)] = 0.0
                    dict_current_data['Detectable' + str(j)] = 0

            # 获取预案中跟踪的目标编号
            dict_current_data['PlanTrackID'] = '0'
            track_flag = False
            if radar_info.str_sensor_id in plan_file_info.plan_result.dict_equip_id_target_id_detection_time:
                dict_target_id_detection_time = plan_file_info.plan_result.dict_equip_id_target_id_detection_time.get(radar_info.str_sensor_id)

                # 20260320当前只考虑装备容量为1，因此只选择目标列表中第一个能看到的目标作为跟踪目标
                for lst_detection_time in dict_target_id_detection_time.values():
                    for detection_time in lst_detection_time:
                        if detection_time.time_range.whether_contains_value(t):
                            dict_current_data['PlanTrackID'] = detection_time.str_target_id
                            track_flag = True
                            break

                    if track_flag:
                        break

            if track_flag:
                dict_equip_id_rest_resource[radar_info.str_sensor_id] = 0.0
                dict_equip_id_current_track_id[radar_info.str_sensor_id] = dict_current_data.get('PlanTrackID')
            else:
                dict_equip_id_rest_resource[radar_info.str_sensor_id] = 1.0
                dict_equip_id_current_track_id[radar_info.str_sensor_id] = '0'

            lst_csv_data.append(dict_current_data)

        # 遍历卫星
        for satellite_info in plan_file_info.battle_scene.dict_satellite_id_info.values():
            dict_current_data = {}

            lst_csv_data.append(dict_current_data)

    with open(dest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=lst_field_name)
        writer.writeheader()
        writer.writerows(lst_csv_data)

    return