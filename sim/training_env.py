# 训练环境类

import random
import pymap3d as pm
from typing import Dict, List, Tuple
from datastruct import TargetTrajPtInfo, TargetInfo, SensorInfo, SatelliteInfo, BattleScene, EquipmentDetectionTime, PlanFileInfo, SystemTrackBase, EquipmentState, EquipmentToTargetDetectionResult, AgentObservation, AgentActionCommand
from plan_file_process import PlanFileProcess


class TrainingEnv:
    def __init__(self) -> None:
        self.__dict_radar_info: Dict[str, SensorInfo] = {}              # 雷达信息，[雷达站号，雷达信息]
        self.__dict_satellite_info: Dict[str, SatelliteInfo] = {}       # 卫星信息，[卫星站号，卫星信息]
        self.__dict_target_info: Dict[str, TargetInfo] = {}             # 目标信息，[目标编号，目标信息]
        self.__start_time: int = 0                                      # 场景起始时间，单位：s
        self.__end_time: int = 0                                        # 场景结束时间，单位：s
        self.__time_step: int = 1                                       # 仿真时间步长，单位：s
        self.__current_time: int = 0                                    # 场景当前时间，单位：s

    # 加载作战场景
    def load_battle_scene(self, plan_id: int, file_path: str, time_step: int = 1) -> None:
        plan_file_process = PlanFileProcess()
        battle_scene = plan_file_process.read_battle_scene_from_json(plan_id=plan_id, file_path=file_path)

        self.__dict_radar_info = battle_scene.dict_radar_id_info
        self.__dict_satellite_info = battle_scene.dict_satellite_id_info
        self.__dict_target_info = battle_scene.dict_target_id_info
        self.__start_time = 0
        self.__end_time = int((battle_scene.end_time - battle_scene.start_time) / 1000)
        self.__time_step = time_step

        return

    # 步进控制
    def step_forward(self, agent_action: AgentActionCommand) -> Tuple[AgentObservation, bool]:
        # Step 0：返回结果初始化
        agent_observation = AgentObservation()
        terminate_flag = False

        # Step 1：更新场景当前时间
        self.__current_time = self.__current_time + self.__time_step
        agent_observation.current_time = self.__current_time
        if agent_observation.current_time > self.__end_time:
            terminate_flag = True
            return agent_observation, terminate_flag

        # Step 2：生成智能体观测信息中的系统航迹
        for target_info in self.__dict_target_info.values():
            if agent_observation.current_time in target_info.dict_target_traj_pt_info:
                traj_pt_info = target_info.dict_target_traj_pt_info.get(agent_observation.current_time)

                system_track = SystemTrackBase()
                system_track.detect_time = agent_observation.current_time
                system_track.str_system_track_no = target_info.str_target_id
                system_track.longitude = traj_pt_info.longitude
                system_track.latitude = traj_pt_info.latitude
                system_track.altitude = traj_pt_info.altitude
                system_track.ecf_x = traj_pt_info.ecf_x
                system_track.ecf_y = traj_pt_info.ecf_y
                system_track.ecf_z = traj_pt_info.ecf_z
                system_track.ecf_vx = traj_pt_info.ecf_vx
                system_track.ecf_vy = traj_pt_info.ecf_vy
                system_track.ecf_vz = traj_pt_info.ecf_vz
                system_track.rcs = traj_pt_info.rcs
                system_track.type = traj_pt_info.type

                agent_observation.dict_system_track[system_track.str_system_track_no] = system_track

        if len(agent_observation.dict_system_track) == 0:
            terminate_flag = False
            return agent_observation, terminate_flag

        # Step 3：生成智能体观测信息中的装备状态
        for radar_info in self.__dict_radar_info.values():
            equip_state = EquipmentState()
            equip_state.time = agent_observation.current_time
            equip_state.str_equip_id = radar_info.str_sensor_id
            equip_state.longitude = radar_info.longitude
            equip_state.latitude = radar_info.latitude
            equip_state.altitude = radar_info.altitude
            equip_state.range_min = radar_info.range_min
            equip_state.range_max = radar_info.range_max
            equip_state.azi_min = radar_info.azi_min
            equip_state.azi_max = radar_info.azi_max
            equip_state.ele_min = radar_info.ele_min
            equip_state.ele_max = radar_info.ele_max
            equip_state.azi_pointing = (radar_info.azi_min + radar_info.azi_max) / 2
            equip_state.ele_pointing = (radar_info.ele_min + radar_info.ele_max) / 2
            equip_state.type = 1
            equip_state.lst_track_no.append(agent_action.str_target_id)
            equip_state.track_num_max = radar_info.track_num_max
            equip_state.residual_track_num = equip_state.track_num_max - len(equip_state.lst_track_no)

            agent_observation.dict_equip_state[equip_state.str_equip_id] = equip_state

        for satellite_info in self.__dict_satellite_info.values():
            equip_state = EquipmentState()

            agent_observation.dict_equip_state[equip_state.str_equip_id] = equip_state

        # Step 4：生成智能体观测信息中的装备对目标的可探测性结果
        for equip_info in agent_observation.dict_equip_state.values():
            for system_track in agent_observation.dict_system_track.values():
                detection_result = EquipmentToTargetDetectionResult()
                detection_result.str_equip_id = equip_info.str_equip_id
                detection_result.str_target_id = system_track.str_system_track_no

                a, e, r = pm.geodetic2aer(system_track.latitude, system_track.longitude, system_track.altitude,
                                          equip_info.latitude, equip_info.longitude, equip_info.altitude)

                if equip_info.azi_min <= a <= equip_info.azi_max and\
                        equip_info.ele_min <= e <= equip_info.ele_max and\
                        equip_info.range_min <= r / 1000.0 <= equip_info.range_max:
                    detection_result.detectable_flag = True
                else:
                    detection_result.detectable_flag = False

                if detection_result.str_equip_id not in agent_observation.dict_detection_result:
                    agent_observation.dict_detection_result[detection_result.str_equip_id] = {}

                agent_observation.dict_detection_result[detection_result.str_equip_id][detection_result.str_target_id] = detection_result

        # Step 5：函数退出
        return agent_observation, terminate_flag

    # 重置控制
    def reset(self) -> AgentObservation:
        self.__current_time = self.__start_time - self.__time_step

        agent_action = AgentActionCommand()
        agent_observation, terminate_flag = self.step_forward(agent_action=agent_action)
        return agent_observation

    def generate_reward(self) -> float:
        pass