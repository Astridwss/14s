# 预案文件处理类
# sim/plan_file_process.py 的头部导入区域
import json
import copy
from typing import List, Dict
import numpy as np
from .datastruct import (
    PlanFileInfo, BattleScene, EquipmentDetectionTime, PlanResult,
    AgentActionCommand, TargetTrajPtInfo, TargetInfo, SensorInfo, SatelliteInfo,
    SatelliteCameraInfo
)
from .two_dim_coordinate import TwoDimensionMinMax
from .ganttchart import GanttChart, OverlappingNumberRequirement
from .satellite_fov_calculation import SatelliteFovCalculation


class PlanFileProcess:
    # =======================================================
    # 1. 从json格式的预案文件中，读取作战场景
    # =======================================================
    def read_battle_scene_from_json(self, plan_id: int, file_path: str) -> BattleScene:
        battle_scene = BattleScene()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict):
                plan_info_list = data.get('planInfoList', [])
                for dict_plan_info in plan_info_list:
                    # 🚨 核心修复 1：兼容多种 ID
                    current_id = dict_plan_info.get('planId') or dict_plan_info.get(
                        'associatedTaskId') or dict_plan_info.get('id')

                    if current_id == plan_id or len(plan_info_list) == 1:
                        battle_scene.start_time = dict_plan_info.get('startTime', 0)
                        battle_scene.end_time = dict_plan_info.get('endTime', 0)

                        str_plan_scene = dict_plan_info.get('planScene')
                        plan_scene_data = json.loads(str_plan_scene)

                        if isinstance(plan_scene_data, dict):
                            dict_analyse_param = plan_scene_data.get('analyseParam')

                            # 雷达信息
                            lst_radar_infos = dict_analyse_param.get('radarInfos')
                            lst_radar = dict_plan_info.get('radarList', [])
                            for dict_radar in lst_radar:
                                if dict_radar.get('id') not in lst_radar_infos:
                                    continue

                                radar_info = SensorInfo()
                                radar_info.str_sensor_id = str(dict_radar.get('id'))
                                radar_info.str_sensor_name = dict_radar.get('radarMC')
                                radar_info.longitude = float(dict_radar.get('radarSZWZJD', 0))
                                radar_info.latitude = float(dict_radar.get('radarSZWZWD', 0))
                                radar_info.altitude = float(dict_radar.get('radarSZWZGD', 0))
                                radar_info.range_min = 0.0

                                max_detection_rcs = float(dict_radar.get('maxDetectionRCS', 1))
                                range_max = float(dict_radar.get('maxDetectionRange', 4000))
                                radar_info.range_max = range_max * pow(1.0 / max_detection_rcs, 0.25)

                                # radar_info.azi_min = 0.0
                                # radar_info.azi_max = 360.0

                                radar_info.azi_min = float(
                                    dict_radar.get('radarZMCX', 0) - dict_radar.get('radarDSFW', 0) / 2.0)
                                radar_info.azi_max = float(
                                    dict_radar.get('radarZMCX', 360) + dict_radar.get('radarDSFW', 0) / 2.0)

                                #print(f"left：{radar_info.azi_min}, right:{radar_info.azi_max}")

                                # radar_info.ele_min = float(dict_radar.get('minElePower', 0))
                                # radar_info.ele_max = float(dict_radar.get('maxElePower', 90))

                                radar_info.ele_min = float(dict_radar.get('radarFYXJ', 0))
                                radar_info.ele_max = float(dict_radar.get('radarFYSJ', 90))

                                radar_info.track_num_max = 20
                                battle_scene.dict_radar_id_info[radar_info.str_sensor_id] = radar_info

                            # 卫星信息
                            lst_sate_ids = dict_analyse_param.get('sateIds')
                            lst_satellite = dict_plan_info.get('satellifeList', [])

                            for dict_satellite in lst_satellite:
                                if dict_satellite.get('id') not in lst_sate_ids:
                                    continue

                                satellite_info = SatelliteInfo()
                                satellite_info.str_satellite_id = str(dict_satellite.get('id', ''))
                                satellite_info.str_satellite_name = dict_satellite.get('satellifeName', '')

                                str_sensor_info = dict_satellite.get('sensorInfo')
                                sensor_info_data = json.loads(str_sensor_info)

                                if isinstance(sensor_info_data, dict):
                                    lst_sensor_info = sensor_info_data.get('sensorInfo')

                                    # first_count_flag = True
                                    for dict_sensor_info in lst_sensor_info:
                                        str_sensor_type = dict_sensor_info.get('sensor', 'infrared')
                                        if str_sensor_type == 'infrared':
                                            camera_type = 1
                                        elif str_sensor_type == 'duidi':
                                            # camera_type = 2   20260924 前台填错了，临时调整后台
                                            camera_type = 3
                                        elif str_sensor_type == 'duikong':
                                            # camera_type = 3   20260924 前台填错了，临时调整后台
                                            camera_type = 2
                                        else:
                                            camera_type = 0

                                        lst_work_mode = dict_sensor_info.get('workMode', [])
                                        for dict_work_mode in lst_work_mode:
                                            str_work_mode_name = dict_work_mode.get('name', '')

                                            satellite_camera_info = SatelliteCameraInfo()
                                            satellite_camera_info.camera_type = camera_type

                                            if '搜索' in str_work_mode_name:
                                                satellite_camera_info.work_mode = 1
                                            elif '凝视' in str_work_mode_name:
                                                satellite_camera_info.work_mode = 2
                                            else:
                                                satellite_camera_info.work_mode = 0

                                            dict_work_mode_parameters = dict_work_mode.get('workModeParameters', {})
                                            satellite_camera_info.azi_min = dict_work_mode_parameters.get('startazimuth', -2.0)
                                            satellite_camera_info.azi_max = dict_work_mode_parameters.get('endazimuth', 2.0)
                                            satellite_camera_info.ele_min = dict_work_mode_parameters.get('startelevation', -2.0)
                                            satellite_camera_info.ele_max = dict_work_mode_parameters.get('endelevation', 2.0)
                                            satellite_camera_info.camera_pointing_max = dict_work_mode_parameters.get('pointing', 10.0)
                                            satellite_camera_info.detection_range = dict_work_mode_parameters.get('distance', 4000.0)

                                            satellite_info.lst_camera_info.append(satellite_camera_info)

                                            # 20260921 TaoXL modify
                                            # sensor_azi_min = -dict_work_mode.get('workModeParameters').get(
                                            #     'azimuth') / 2.0
                                            # sensor_azi_max = dict_work_mode.get('workModeParameters').get(
                                            #     'azimuth') / 2.0
                                            # sensor_ele_min = -dict_work_mode.get('workModeParameters').get(
                                            #     'elevation') / 2.0
                                            # sensor_ele_max = dict_work_mode.get('workModeParameters').get(
                                            #     'elevation') / 2.0
                                            # sensor_pointing_max = dict_work_mode.get('workModeParameters').get(
                                            #     'pointing', 10.0)

                                            # 20260923 TaoXL modify
                                            # if '-' in satellite_info.str_satellite_name:
                                            #     sensor_azi_min = -5.0
                                            #     sensor_azi_max = 5.0
                                            #     sensor_ele_min = -5.0
                                            #     sensor_ele_max = 5.0
                                            #     sensor_pointing_max = 30.0
                                            # else:
                                            #     sensor_azi_min = -2.0
                                            #     sensor_azi_max = 2.0
                                            #     sensor_ele_min = -2.0
                                            #     sensor_ele_max = 2.0
                                            #     sensor_pointing_max = 10.0
                                            #
                                            # if first_count_flag:
                                            #     satellite_info.azi_min = sensor_azi_min
                                            #     satellite_info.azi_max = sensor_azi_max
                                            #     satellite_info.ele_min = sensor_ele_min
                                            #     satellite_info.ele_max = sensor_ele_max
                                            #     satellite_info.camera_pointing_max = sensor_pointing_max
                                            #     first_count_flag = False
                                            # else:
                                            #     if sensor_azi_min < satellite_info.azi_min:
                                            #         satellite_info.azi_min = sensor_azi_min
                                            #     if sensor_azi_max > satellite_info.azi_max:
                                            #         satellite_info.azi_max = sensor_azi_max
                                            #     if sensor_ele_min < satellite_info.ele_min:
                                            #         satellite_info.ele_min = sensor_ele_min
                                            #     if sensor_ele_max > satellite_info.ele_max:
                                            #         satellite_info.ele_max = sensor_ele_max
                                            #     if sensor_pointing_max > satellite_info.camera_pointing_max:
                                            #         satellite_info.camera_pointing_max = sensor_pointing_max

                                satellite_info.track_num_max = 1

                                # 20260408
                                str_target_calc = dict_satellite.get('targetCalc')
                                target_calc_data = json.loads(str_target_calc)

                                if isinstance(target_calc_data, dict):
                                    lst_vpt = target_calc_data.get('vPtList')

                                    for dict_vpt in lst_vpt:
                                        traj_pt_info = TargetTrajPtInfo()
                                        traj_pt_info.time = round(dict_vpt.get('dTime'))
                                        traj_pt_info.longitude = dict_vpt.get('geoPos').get('x')
                                        traj_pt_info.latitude = dict_vpt.get('geoPos').get('y')
                                        traj_pt_info.altitude = dict_vpt.get('geoPos').get('z')
                                        traj_pt_info.rcs = 1.0
                                        traj_pt_info.type = 5

                                        satellite_info.dict_satellite_traj_pt_info[traj_pt_info.time] = traj_pt_info

                                battle_scene.dict_satellite_id_info[satellite_info.str_satellite_id] = satellite_info

                            # 目标信息
                            lst_air_target_ids = dict_analyse_param.get('airTargetIds')
                            lst_missile_target = dict_plan_info.get('missileTargetList', [])
                            for dict_missile_target in lst_missile_target:
                                if dict_missile_target.get('id') not in lst_air_target_ids:
                                    continue
                                target_info = TargetInfo()
                                target_info.str_target_id = str(dict_missile_target.get('id'))
                                target_info.str_target_name = dict_missile_target.get('targetName')

                                str_missile_calc = dict_missile_target.get('missileCalc')
                                if not str_missile_calc: continue
                                missile_calc_data = json.loads(str_missile_calc)

                                if isinstance(missile_calc_data, dict):
                                    dict_cur_tar_info = missile_calc_data.get('curTarInfo', {})

                                    # 🚨 核心修复 2：进入 vPtList 层级提取轨迹点！
                                    # lst_vbf = dict_cur_tar_info.get('vBfList', [])
                                    lst_vpt = dict_cur_tar_info.get('vPtList', [])
                                    # for dict_vbf in lst_vbf:
                                    #     lst_vpt = dict_vbf.get('vPtList', [])
                                    for dict_vpt in lst_vpt:
                                        traj_pt_info = TargetTrajPtInfo()
                                        traj_pt_info.time = round(dict_vpt.get('dTime', 0))
                                        geo_pos = dict_vpt.get('geoPos', {})
                                        traj_pt_info.longitude = float(geo_pos.get('x', 0.0))
                                        traj_pt_info.latitude = float(geo_pos.get('y', 0.0))
                                        traj_pt_info.altitude = float(geo_pos.get('z', 0.0)) * 1000.0
                                        traj_pt_info.rcs = 1.0
                                        traj_pt_info.type = 1

                                        target_info.dict_target_traj_pt_info[traj_pt_info.time] = traj_pt_info

                                battle_scene.dict_target_id_info[target_info.str_target_id] = target_info

        except Exception as e:
            print(f"解析作战场景失败：{e}")
        return battle_scene

    # =======================================================
    # 2. 从json格式的预案文件中，读取规划结果
    # =======================================================
    def read_plan_result_from_json(self, plan_id: int, file_path: str) -> PlanResult:
        plan_result = PlanResult()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict):
                plan_info_list = data.get('planInfoList', [])
                for dict_plan_info in plan_info_list:
                    # 🚨 同步修改：兼容真实的 ID 字段
                    current_id = dict_plan_info.get('planId') or dict_plan_info.get(
                        'associatedTaskId') or dict_plan_info.get('id')

                    if current_id == plan_id or len(plan_info_list) == 1:
                        str_split_quduan_result = dict_plan_info.get('splitQuduanResult')
                        if not str_split_quduan_result: continue

                        split_quduan_result_data = json.loads(str_split_quduan_result)

                        if isinstance(split_quduan_result_data, dict):
                            lst_task_result = split_quduan_result_data.get('vtaskResult', [])
                            for dict_task_result in lst_task_result:
                                if dict_task_result.get('strTaskName') == 'ALL':
                                    str_equip_id = dict_task_result.get('strRadarId')
                                    str_target_id = dict_task_result.get('strTarId')
                                    lst_quduan = dict_task_result.get('vquDuanList', [])

                                    for dict_quduan in lst_quduan:
                                        detection_time = EquipmentDetectionTime()
                                        detection_time.str_equip_id = str_equip_id
                                        detection_time.str_target_id = str_target_id
                                        detection_time.time_range = TwoDimensionMinMax(dict_quduan.get('x'),
                                                                                       dict_quduan.get('y'))

                                        if str_equip_id not in plan_result.dict_equip_id_target_id_detection_time:
                                            plan_result.dict_equip_id_target_id_detection_time[str_equip_id] = {}
                                        if str_target_id not in plan_result.dict_equip_id_target_id_detection_time[
                                            str_equip_id]:
                                            plan_result.dict_equip_id_target_id_detection_time[str_equip_id][
                                                str_target_id] = []

                                        plan_result.dict_equip_id_target_id_detection_time[str_equip_id][
                                            str_target_id].append(detection_time)
        except Exception as e:
            print(f"解析规划结果失败：{e}")
        return plan_result

    # =======================================================
    # 3. 读取聚合体
    # =======================================================
    def read_plan_file_info_from_json(self, plan_id: int, file_path: str) -> PlanFileInfo:
        plan_file_info = PlanFileInfo()
        plan_file_info.battle_scene = self.read_battle_scene_from_json(plan_id=plan_id, file_path=file_path)
        plan_file_info.plan_result = self.read_plan_result_from_json(plan_id=plan_id, file_path=file_path)
        return plan_file_info


    # 更新模型推理结果，对处于卫星视场内的目标创建AgentActionCommand
    def update_model_inference_satellite_result(self,
                                                dict_model_inference_result: Dict[int, List[AgentActionCommand]],
                                                battle_scene: BattleScene) -> Dict[int, List[AgentActionCommand]]:
        # 将卫星视场角内的目标，均添加临时的动作指令，以在下一步生成相应的探测弧段
        dict_model_inference_result_update = copy.deepcopy(dict_model_inference_result)
        lst_target_id = list(battle_scene.dict_target_id_info.keys())

        for time, lst_model_inference_result in dict_model_inference_result_update.items():
            lst_add_command: List[AgentActionCommand] = []

            targets: List[np.ndarray] = []
            for tgt in battle_scene.dict_target_id_info.values():
                if time in tgt.dict_target_traj_pt_info:
                    targets.append(np.array([tgt.dict_target_traj_pt_info.get(time).longitude,
                                             tgt.dict_target_traj_pt_info.get(time).latitude,
                                             tgt.dict_target_traj_pt_info.get(time).altitude]))

            for model_inference_result in lst_model_inference_result:
                str_equip_id = model_inference_result.str_equip_id
                str_target_id = model_inference_result.str_target_id

                if (str_equip_id not in battle_scene.dict_satellite_id_info) or (
                        str_target_id not in battle_scene.dict_target_id_info):
                    continue

                sat_current_geo_pos: np.ndarray = None
                satellite_info: SatelliteInfo = battle_scene.dict_satellite_id_info.get(str_equip_id)
                if time in satellite_info.dict_satellite_traj_pt_info:
                    sat_current_geo_pos = np.array([satellite_info.dict_satellite_traj_pt_info.get(time).longitude,
                                                    satellite_info.dict_satellite_traj_pt_info.get(time).latitude,
                                                    satellite_info.dict_satellite_traj_pt_info.get(time).altitude])

                sat_last_geo_pos: np.ndarray = None
                if time > 0 and (time - 1) in satellite_info.dict_satellite_traj_pt_info:
                    sat_last_geo_pos = np.array([satellite_info.dict_satellite_traj_pt_info.get(time - 1).longitude,
                                                 satellite_info.dict_satellite_traj_pt_info.get(time - 1).latitude,
                                                 satellite_info.dict_satellite_traj_pt_info.get(time - 1).altitude])

                # 20260923 TaoXL modify
                # fov_az = satellite_info.azi_max - satellite_info.azi_min
                # fov_el = satellite_info.ele_max - satellite_info.ele_min
                # max_pointing_angle = satellite_info.camera_pointing_max

                center_target_geo_pos: np.ndarray = None
                target_info: TargetInfo = battle_scene.dict_target_id_info.get(str_target_id)
                if time in target_info.dict_target_traj_pt_info:
                    center_target_geo_pos = np.array([target_info.dict_target_traj_pt_info.get(time).longitude,
                                                      target_info.dict_target_traj_pt_info.get(time).latitude,
                                                      target_info.dict_target_traj_pt_info.get(time).altitude])

                fov_calc = SatelliteFovCalculation()
                in_fov_indices: List[int] = []
                # 20260923 TaoXL add
                for camera_info in satellite_info.lst_camera_info:
                    if ((camera_info.camera_type == 1 and camera_info.work_mode == 1) or
                            (camera_info.camera_type == 2 and camera_info.work_mode == 1)):
                        for i, target_geo_pos in enumerate(targets):
                            in_flag: bool = fov_calc.judge_target_can_be_detected_by_satellite_ground_camera(sat_current_geo_pos=sat_current_geo_pos,
                                                                                                             azi_min=camera_info.azi_min,
                                                                                                             azi_max=camera_info.azi_max,
                                                                                                             ele_min=camera_info.ele_min,
                                                                                                             ele_max=camera_info.ele_max,
                                                                                                             target_geo_pos=target_geo_pos,
                                                                                                             sat_last_geo_pos=sat_last_geo_pos)
                            if in_flag and (i not in in_fov_indices):
                                in_fov_indices.append(i)

                    elif camera_info.camera_type == 3 and camera_info.work_mode == 1:
                        for i, target_geo_pos in enumerate(targets):
                            in_flag: bool = fov_calc.judge_target_can_be_detected_by_low_orbit_satellite_air_camera(sat_current_geo_pos=sat_current_geo_pos,
                                                                                                                    ele_min=camera_info.ele_min,
                                                                                                                    ele_max=camera_info.ele_max,
                                                                                                                    target_geo_pos=target_geo_pos,
                                                                                                                    sat_last_geo_pos=sat_last_geo_pos)
                            if in_flag and (i not in in_fov_indices):
                                in_fov_indices.append(i)

                    elif camera_info.camera_type == 1 and camera_info.work_mode == 2:
                        in_flag_indices: List[int] = fov_calc.find_targets_in_fov(sat_current_geo_pos=sat_current_geo_pos,
                                                                                  fov_az=camera_info.azi_max - camera_info.azi_min,
                                                                                  fov_el=camera_info.ele_max - camera_info.ele_min,
                                                                                  center_target_geo_pos=center_target_geo_pos,
                                                                                  max_pointing_angle=camera_info.camera_pointing_max,
                                                                                  targets=targets,
                                                                                  sat_last_geo_pos=sat_last_geo_pos)
                        for i in in_flag_indices:
                            if i not in in_fov_indices:
                                in_fov_indices.append(i)

                for index in in_fov_indices:
                    str_add_target_id = lst_target_id[index]
                    if str_add_target_id != str_target_id:
                        add_command = AgentActionCommand(time=time, str_equip_id=str_equip_id,
                                                         str_target_id=str_add_target_id)
                        lst_add_command.append(add_command)

            if lst_add_command:
                lst_model_inference_result.extend(lst_add_command)

        return dict_model_inference_result_update


    # =======================================================
    # 5. 推演结果生成与保存相关 (保持空壳和占位不变)
    # =======================================================
    # 将模型推理结果，转换为规划结果
    def write_model_inference_result_to_plan_result(self,
                                                    dict_model_inference_result: Dict[int, List[AgentActionCommand]],
                                                    battle_scene: BattleScene,
                                                    time_cut: int = 20) -> PlanResult:
        # 更新模型推理结果，对处于卫星视场内的目标创建AgentActionCommand
        dict_model_inference_result_update = self.update_model_inference_satellite_result(dict_model_inference_result=dict_model_inference_result,
                                                                                          battle_scene=battle_scene)

        plan_result = PlanResult()

        # print(dict_model_inference_result.keys())

        # dict_model_inference_result_sorted = dict(sorted(dict_model_inference_result.items()))   20260409 暂不需要再排序啦
        for lst_model_inference_result in dict_model_inference_result_update.values():
            for model_inference_result in lst_model_inference_result:
                if model_inference_result.str_equip_id not in plan_result.dict_equip_id_target_id_detection_time:
                    plan_result.dict_equip_id_target_id_detection_time[model_inference_result.str_equip_id] = {}

                if model_inference_result.str_target_id not in plan_result.dict_equip_id_target_id_detection_time[
                    model_inference_result.str_equip_id]:
                    plan_result.dict_equip_id_target_id_detection_time[model_inference_result.str_equip_id][
                        model_inference_result.str_target_id] = []

                if len(plan_result.dict_equip_id_target_id_detection_time[model_inference_result.str_equip_id][
                           model_inference_result.str_target_id]) == 0:
                    detection_time = EquipmentDetectionTime()
                    detection_time.str_equip_id = model_inference_result.str_equip_id
                    detection_time.str_target_id = model_inference_result.str_target_id
                    detection_time.time_range.value_min = model_inference_result.time
                    detection_time.time_range.value_max = model_inference_result.time

                    plan_result.dict_equip_id_target_id_detection_time[model_inference_result.str_equip_id][
                        model_inference_result.str_target_id].append(detection_time)

                else:
                    last_detection_time = \
                    plan_result.dict_equip_id_target_id_detection_time[model_inference_result.str_equip_id][
                        model_inference_result.str_target_id][-1]

                    if model_inference_result.time - last_detection_time.time_range.value_max < time_cut:
                        last_detection_time.time_range.value_max = model_inference_result.time

                    else:
                        detection_time = EquipmentDetectionTime()
                        detection_time.str_equip_id = model_inference_result.str_equip_id
                        detection_time.str_target_id = model_inference_result.str_target_id
                        detection_time.time_range.value_min = model_inference_result.time
                        detection_time.time_range.value_max = model_inference_result.time

                        plan_result.dict_equip_id_target_id_detection_time[model_inference_result.str_equip_id][
                            model_inference_result.str_target_id].append(detection_time)

        print(plan_result.dict_equip_id_target_id_detection_time.keys())

        return plan_result

    # 将规划结果，转换为json格式的预案文件
    def write_plan_result_to_json(self, plan_result: PlanResult, dest_path: str) -> None:
        data = {}
        data['splitQuduanResult'] = {}
        data['splitQuduanResult']['vtaskResult'] = []

        for dict_target_id_detection_time in plan_result.dict_equip_id_target_id_detection_time.values():
            for lst_detection_time in dict_target_id_detection_time.values():
                dict_task_result = {}

                for detection_time in lst_detection_time:
                    dict_task_result['strRadarId'] = detection_time.str_equip_id
                    dict_task_result['strTarId'] = detection_time.str_target_id

                    if 'vquDuanList' not in dict_task_result:
                        dict_task_result['vquDuanList'] = []

                    dict_task_result['vquDuanList'].append(
                        {'x': detection_time.time_range.value_min, 'y': detection_time.time_range.value_max})

                dict_task_result['strTaskName'] = 'ALL'
                data['splitQuduanResult']['vtaskResult'].append(dict_task_result)

        data['splitQuduanResult'] = json.dumps(data['splitQuduanResult'])
        with open(dest_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        return

    def write_plan_metric_to_json(self, plan_result: PlanResult, battle_scene: BattleScene, dest_path: str) -> None:
        dict_target_id_equip_id_detection_time: Dict[str, Dict[str, List[EquipmentDetectionTime]]] = {}

        for str_equip_id, dict_target_id_detection_time in plan_result.dict_equip_id_target_id_detection_time.items():
            for str_target_id, lst_detection_time in dict_target_id_detection_time.items():
                if str_target_id not in dict_target_id_equip_id_detection_time:
                    dict_target_id_equip_id_detection_time[str_target_id] = {}

                dict_target_id_equip_id_detection_time[str_target_id][str_equip_id] = lst_detection_time

        data = {}
        data['schemeEvaluteResult'] = {}
        data['schemeEvaluteResult']['schemeEvaResult'] = []

        dict_eva_result = {}
        dict_eva_result['vecTargetEvaResult'] = []

        for str_target_id, dict_equip_id_detection_time in dict_target_id_equip_id_detection_time.items():
            # for循环，每个目标一个
            dict_target_eva_result = {}
            dict_target_eva_result['strTargetID'] = str_target_id

            gantt_chart = GanttChart()
            gantt_chart.str_body_name = str_target_id

            for str_equip_id, lst_detection_time in dict_equip_id_detection_time.items():
                for detection_time in lst_detection_time:
                    if str_equip_id not in gantt_chart.dict_activity:
                        gantt_chart.dict_activity[str_equip_id] = []

                    gantt_chart.dict_activity[str_equip_id].append(detection_time.time_range)

            lst_result_over_one_cover: List[
                Tuple[TwoDimensionMinMax, List[str]]] = gantt_chart.overlapping_number_analysis_in_total(0,
                                                                                                         OverlappingNumberRequirement.OVERLAPPING_NUMBER_REQUIREMENT_LARGER)
            lst_result_one_cover: List[
                Tuple[TwoDimensionMinMax, List[str]]] = gantt_chart.overlapping_number_analysis_in_total(1,
                                                                                                         OverlappingNumberRequirement.OVERLAPPING_NUMBER_REQUIREMENT_EQUAL)
            lst_result_two_cover: List[
                Tuple[TwoDimensionMinMax, List[str]]] = gantt_chart.overlapping_number_analysis_in_total(2,
                                                                                                         OverlappingNumberRequirement.OVERLAPPING_NUMBER_REQUIREMENT_EQUAL)
            lst_result_over_three_cover: List[
                Tuple[TwoDimensionMinMax, List[str]]] = gantt_chart.overlapping_number_analysis_in_total(2,
                                                                                                         OverlappingNumberRequirement.OVERLAPPING_NUMBER_REQUIREMENT_LARGER)

            target_traj_total_time = 2600
            if str_target_id in battle_scene.dict_target_id_info:
                target_traj_total_time = len(
                    battle_scene.dict_target_id_info.get(str_target_id).dict_target_traj_pt_info)

            if target_traj_total_time > 0:
                dict_target_eva_result['vecContinuityResult'] = []
                dict_target_eva_result['vecReliabilityResult'] = []

                over_one_cover_time = 0
                interrupt_num = 0
                for i, (time_range, lst_equip_id) in enumerate(lst_result_over_one_cover):
                    over_one_cover_time = over_one_cover_time + time_range.range()

                    if i < len(lst_result_over_one_cover) - 1:
                        if abs(lst_result_over_one_cover[i + 1][0].value_min - time_range.value_max) > 1e-6: \
                                interrupt_num += 1

                dict_continuity_result = {'dDetectCoverAge': over_one_cover_time / target_traj_total_time * 100.0,
                                          'regionType': 'ENUM_COMPREHENSIVE', 'uiInterruputNum': interrupt_num}
                dict_target_eva_result['vecContinuityResult'].append(dict_continuity_result)

                one_cover_time = 0
                for time_range, lst_equip_id in lst_result_one_cover:
                    one_cover_time = one_cover_time + time_range.range()

                two_cover_time = 0
                for time_range, lst_equip_id in lst_result_two_cover:
                    two_cover_time = two_cover_time + time_range.range()

                over_three_cover_time = 0
                for time_range, lst_equip_id in lst_result_over_three_cover:
                    over_three_cover_time = over_three_cover_time + time_range.range()

                dict_reliability_result = {}
                dict_reliability_result['regionType'] = 'ENUM_COMPREHENSIVE'
                dict_reliability_result['vecCovernumCoverage'] = [
                    {'dVal': (target_traj_total_time - over_one_cover_time) / target_traj_total_time * 100.0,
                     'uiCoverNum': 0},
                    {'dVal': one_cover_time / target_traj_total_time * 100.0, 'uiCoverNum': 1},
                    {'dVal': two_cover_time / target_traj_total_time * 100.0, 'uiCoverNum': 2},
                    {'dVal': over_three_cover_time / target_traj_total_time * 100.0, 'uiCoverNum': 3}
                ]

                dict_target_eva_result['vecReliabilityResult'].append(dict_reliability_result)

                dict_eva_result['vecTargetEvaResult'].append(dict_target_eva_result)

        data['schemeEvaluteResult']['schemeEvaResult'].append(dict_eva_result)

        data['schemeEvaluteResult'] = json.dumps(data['schemeEvaluteResult'])
        with open(dest_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        return

    # 将模型推理结果，转换为json格式的预案文件
    def write_model_inference_result_to_json(self, dict_model_inference_result: Dict[int, List[AgentActionCommand]],
                                             battle_scene: BattleScene, dest_path: str) -> None:
        plan_result = self.write_model_inference_result_to_plan_result(
            dict_model_inference_result=dict_model_inference_result, battle_scene=battle_scene)
        self.write_plan_result_to_json(plan_result=plan_result, dest_path=dest_path)

        return

    # 将模型推理结果，转换为json格式的预案评估文件
    def write_model_inference_metric_to_json(self, dict_model_inference_result: Dict[int, List[AgentActionCommand]],
                                             battle_scene: BattleScene, dest_path: str) -> None:
        plan_file_process = PlanFileProcess()
        plan_result = self.write_model_inference_result_to_plan_result(
            dict_model_inference_result=dict_model_inference_result, battle_scene=battle_scene)
        self.write_plan_metric_to_json(plan_result=plan_result, battle_scene=battle_scene, dest_path=dest_path)

        return
