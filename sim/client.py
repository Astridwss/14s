# sim/client.py (原 TrainingEnv)
import pymap3d as pm
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict  #20260821 

from .datastruct import (
    SystemTrackBase, EquipmentState, EquipmentToTargetDetectionResult, 
    AgentObservation, AgentActionCommand
)
from .plan_file_process import PlanFileProcess
from .satellite_fov_calculation import SatelliteFovCalculation

class TrainingEnv:
    """
    职责：加载预案、时间步进、推演物理状态。
    (注：奖励计算逻辑已清空，留待后续基于观测字典重构)
    """
    def __init__(self) -> None:
        # 物理真值数据池
        self._dict_radar_info = {}       
        self._dict_satellite_info = {}   
        self._dict_target_info = {}      
        
        # 时间控制
        self._start_time: int = 0        
        self._end_time: int = 0          
        self._time_step: int = 1         
        self._current_time: int = 0      
        
        # 状态暂存
        self._current_obs: AgentObservation = AgentObservation()
        self._last_actions: List[AgentActionCommand] = []

    def load_battle_scene(self, plan_id: int, file_path: str, time_step: int = 1) -> None:
        """统一的数据加载入口"""
        plan_file_process = PlanFileProcess()
        battle_scene = plan_file_process.read_battle_scene_from_json(plan_id=plan_id, file_path=file_path)

        self._dict_radar_info = battle_scene.dict_radar_id_info
        self._dict_satellite_info = battle_scene.dict_satellite_id_info
        self._dict_target_info = battle_scene.dict_target_id_info
        self._start_time = 0
        self._end_time = int((battle_scene.end_time - battle_scene.start_time) / 1000)
        self._time_step = time_step

    @property
    def dict_radar_info(self):
        """雷达信息字典，供 RadarGrouper 聚类使用。"""
        return self._dict_radar_info
    
    # 20260912 TaoXL add
    @property
    def dict_satellite_info(self):
        return self._dict_satellite_info
    
    @property
    def dict_target_info(self):
        return self._dict_target_info
    # 20260912 TaoXL add end

    def reset(self) -> AgentObservation:
        """环境重置"""
        self._current_time = self._start_time - self._time_step
        self._last_actions = []
        obs, _ = self.step_forward(agent_actions=[])
        return obs

    def step_forward(self, agent_actions: List[AgentActionCommand]) -> Tuple[AgentObservation, bool]:
        """
        物理世界步进：接收多智能体动作列表，输出下一帧的局部观测
        """
        self._last_actions = agent_actions  # 记录动作
        agent_observation = AgentObservation()
        terminate_flag = False

        # 1. 更新场景当前时间
        self._current_time += self._time_step
        agent_observation.current_time = self._current_time
        
        if agent_observation.current_time > self._end_time:
            terminate_flag = True
            return agent_observation, terminate_flag

        # 2. 生成系统航迹 (天上真有什么)
        for target_info in self._dict_target_info.values():
            if self._current_time in target_info.dict_target_traj_pt_info:
                pt = target_info.dict_target_traj_pt_info.get(self._current_time)
                trk = SystemTrackBase(
                    detect_time=self._current_time,
                    str_system_track_no=target_info.str_target_id,
                    longitude=pt.longitude, latitude=pt.latitude, altitude=pt.altitude,
                    ecf_x=pt.ecf_x, ecf_y=pt.ecf_y, ecf_z=pt.ecf_z,
                    ecf_vx=pt.ecf_vx, ecf_vy=pt.ecf_vy, ecf_vz=pt.ecf_vz,
                    rcs=pt.rcs, type=pt.type
                )
                agent_observation.dict_system_track[trk.str_system_track_no] = trk

        # 3. 解析动作：把智能体的动作映射到对应的雷达上
        #2026822 txl
        #action_map = {cmd.str_equip_id: cmd.str_target_id for cmd in agent_actions}
        action_map = defaultdict(list)
        for cmd in agent_actions:
            action_map[cmd.str_equip_id].append(cmd.str_target_id)

        # 4. 生成雷达状态
        for radar_info in self._dict_radar_info.values():
            equip_state = EquipmentState(
                time=self._current_time, str_equip_id=radar_info.str_sensor_id,
                longitude=radar_info.longitude, latitude=radar_info.latitude, altitude=radar_info.altitude,
                range_min=radar_info.range_min, range_max=radar_info.range_max,
                azi_min=radar_info.azi_min, azi_max=radar_info.azi_max,
                ele_min=radar_info.ele_min, ele_max=radar_info.ele_max,
                azi_pointing=(radar_info.azi_min + radar_info.azi_max) / 2,
                ele_pointing=(radar_info.ele_min + radar_info.ele_max) / 2,
                type=1, track_num_max=radar_info.track_num_max
            )
            # 执行动作锁定
            target_to_lock = action_map.get(equip_state.str_equip_id)
            if target_to_lock:
                #20260822 txl
                equip_state.lst_track_no = target_to_lock[:equip_state.track_num_max]
            else:
                equip_state.lst_track_no = []
            
            equip_state.residual_track_num = max(0, equip_state.track_num_max - len(equip_state.lst_track_no))
            agent_observation.dict_equip_state[equip_state.str_equip_id] = equip_state

        # 20260408
        for satellite_info in self._dict_satellite_info.values():
            equip_state = EquipmentState()
            equip_state.time = agent_observation.current_time
            equip_state.str_equip_id = satellite_info.str_satellite_id

            if agent_observation.current_time in satellite_info.dict_satellite_traj_pt_info:
                equip_state.longitude = satellite_info.dict_satellite_traj_pt_info.get(
                    agent_observation.current_time).longitude
                equip_state.latitude = satellite_info.dict_satellite_traj_pt_info.get(
                    agent_observation.current_time).latitude
                equip_state.altitude = satellite_info.dict_satellite_traj_pt_info.get(
                    agent_observation.current_time).altitude

            equip_state.range_min = 0.0
            equip_state.range_max = 36000.0
            equip_state.azi_min = satellite_info.azi_min
            equip_state.azi_max = satellite_info.azi_max
            equip_state.ele_min = satellite_info.ele_min
            equip_state.ele_max = satellite_info.ele_max
            equip_state.azi_pointing = (satellite_info.azi_min + satellite_info.azi_max) / 2
            equip_state.ele_pointing = (satellite_info.ele_min + satellite_info.ele_max) / 2
            equip_state.type = 2
            equip_state.track_num_max = satellite_info.track_num_max

            target_to_lock = action_map.get(equip_state.str_equip_id)
            if target_to_lock:
                #20260822 txl
                #equip_state.lst_track_no.append(target_to_lock)
                equip_state.lst_track_no = target_to_lock[:equip_state.track_num_max]
            else:
                equip_state.lst_track_no = []

            equip_state.residual_track_num = equip_state.track_num_max - len(equip_state.lst_track_no)
            agent_observation.dict_equip_state[equip_state.str_equip_id] = equip_state

        # 5. 计算可视性矩阵
        for equip_id, equip_info in agent_observation.dict_equip_state.items():
            agent_observation.dict_detection_result[equip_id] = {}
            for target_id, system_track in agent_observation.dict_system_track.items():
                det_res = EquipmentToTargetDetectionResult(
                    str_equip_id=equip_id, str_target_id=target_id, detectable_flag=False
                )

                # 20260408
                if equip_info.type == 1:
                    a, e, r = pm.geodetic2aer(
                        system_track.latitude, system_track.longitude, system_track.altitude,
                        equip_info.latitude, equip_info.longitude, equip_info.altitude
                    )
                    if (equip_info.azi_min <= a <= equip_info.azi_max and
                        equip_info.ele_min <= e <= equip_info.ele_max and
                        equip_info.range_min <= r / 1000.0 <= equip_info.range_max):
                        det_res.detectable_flag = True
                elif equip_info.type == 2:
                    camera_pointing_max = 0.0
                    if equip_id in self._dict_satellite_info:
                        camera_pointing_max = self._dict_satellite_info.get(equip_id).camera_pointing_max

                    sat_xyz = pm.geodetic2ecef(equip_info.latitude, equip_info.longitude, equip_info.altitude)
                    target_xyz = pm.geodetic2ecef(system_track.latitude, system_track.longitude, system_track.altitude)
                    earth_center = np.array([0.0, 0.0, 0.0])
                    vec_cam = earth_center - sat_xyz
                    vec_target = np.array([target_xyz[0], target_xyz[1], target_xyz[2]]) - np.array([sat_xyz[0], sat_xyz[1], sat_xyz[2]])

                    fov_calc = SatelliteFovCalculation()
                    earth_occluded_flag = fov_calc.is_earth_occluded(np.array([sat_xyz[0], sat_xyz[1], sat_xyz[2]]),
                                                                     np.array(
                                                                         [target_xyz[0], target_xyz[1], target_xyz[2]]))
                    if earth_occluded_flag:
                        det_res.detectable_flag = False
                    else:
                        norm_cam = np.linalg.norm(vec_cam)
                        norm_target = np.linalg.norm(vec_target)

                        if norm_cam == 0 or norm_target == 0:
                            det_res.detectable_flag = False
                        else:
                            dot_product = np.dot(vec_cam, vec_target)
                            cos_theta = dot_product / (norm_cam * norm_target)
                            cos_theta = np.clip(cos_theta, -1.0, 1.0)
                            theta_radians = np.arccos(cos_theta)
                            theta_degrees = np.degrees(theta_radians)

                            if abs(theta_degrees) < equip_info.azi_max + camera_pointing_max:
                                det_res.detectable_flag = True
                            else:
                                det_res.detectable_flag = False
                else:
                    continue
                
                agent_observation.dict_detection_result[equip_id][target_id] = det_res

        self._current_obs = agent_observation # 暂存当前帧，留给以后的算分器使用
        return agent_observation, terminate_flag


    def generate_reward(self) -> float:
        """
        200-50-21 规模（200 雷达 + 50 卫星 + 21 目标）协同接力跟踪奖励。

        针对「250 传感器追 21 目标」的大冗余场景，教会智能体：
          - 覆盖：每个可见目标至少被 1 部传感器锁定；
          - 效能：不要扎堆，同一目标超过 K 部锁定即为冗余；
          - 接力：目标走到某传感器探测边缘时，提前由友军接住。

        奖励项（按目标/动作结算）：
          R_miss      可见目标 0 锁定         -> -20（丢失目标，最高优先级）
          R_cover     可见目标 >=1 锁定        -> +2
          R_valid     锁定不可见目标            -> -5（瞎指）
          R_redundant 同目标 >K 部锁定          -> -5 / 超出部（K=4）
          R_danger    仅 1 部锁定且处于边缘     -> -3（危险，无友军接力）
          R_relay     边缘 + 区 双机共轨        -> +8（完美接力）

        注：单智能体每步最多锁定 1 个目标，track_num_max 在当前动作空间下
        不会触顶，故不设「容量超载」项（旧版 bk1 的该项实为死代码）。
        """
        reward = 0.0
        obs = self._current_obs
        if not obs or not self._last_actions:
            return reward

        K = 4  # 冗余容忍上限：同一目标最多 K 部雷达锁定不罚（接力区 2~4 部重叠不扣分）

        # ---- 1. 可见性矩阵 + 边缘判定：{target_id: {sensor_id: is_at_edge}} ----
        visible = {}  # t_id -> {s_id: is_edge}
        for s_id, targets in obs.dict_detection_result.items():
            equip = obs.dict_equip_state.get(s_id)
            if not equip:
                continue
            if equip.type != 1:  # 卫星广播-only，不参与锁定/边缘判定 → 不计入 visible
                continue
            for t_id, res in targets.items():
                if not res.detectable_flag:
                    continue
                is_edge = False
                trk = obs.dict_system_track.get(t_id)
                if trk:
                    _, _, r = pm.geodetic2aer(
                        trk.latitude, trk.longitude, trk.altitude,
                        equip.latitude, equip.longitude, equip.altitude,
                    )
                    is_edge = (r / 1000.0) > (equip.range_max * 0.85)
                visible.setdefault(t_id, {})[s_id] = is_edge

        # ---- 2. 动作结算：非法动作惩罚 + 每目标锁定者列表 ----
        locks = {}  # t_id -> [s_id, ...]（仅合法锁定）
        for cmd in self._last_actions:
            if not cmd.str_target_id:
                continue
            s_id, t_id = cmd.str_equip_id, cmd.str_target_id
            if t_id not in visible or s_id not in visible[t_id]:
                reward -= 5.0  # R_valid
            else:
                locks.setdefault(t_id, []).append(s_id)

        # ---- 3. 覆盖 / 冗余 / 接力（按目标结算） ----
        for t_id, able in visible.items():
            locked = locks.get(t_id, [])
            n = len(locked)

            if n == 0:
                reward -= 20.0  # R_miss
                continue

            reward += 2.0  # R_cover

            if n > K:
                reward -= 5.0 * (n - K)  # R_redundant

            edge = sum(1 for s in locked if able.get(s, False))
            comfort = n - edge
            if n == 1 and edge == 1:
                reward -= 3.0  # R_danger：边缘孤机，无友军接力
            elif edge > 0 and comfort > 0:
                reward += 8.0  # R_relay：边缘 + 区 完美接力

        return float(reward)

    def generate_reward_bk6(self) -> float:
        """[备份 2026-08-18] 改动前版本：K=2 + 卫星参与动作/重数统计。

        作为"卫星广播-only + K=4"改造的回退点，逻辑与改造前 generate_reward 完全一致。
        """
        reward = 0.0
        obs = self._current_obs
        if not obs or not self._last_actions:
            return reward

        K = 2  # 冗余容忍上限：同一目标最多 K 部传感器锁定不罚

        # ---- 1. 可见性矩阵 + 边缘判定：{target_id: {sensor_id: is_at_edge}} ----
        visible = {}  # t_id -> {s_id: is_edge}
        for s_id, targets in obs.dict_detection_result.items():
            equip = obs.dict_equip_state.get(s_id)
            if not equip:
                continue
            for t_id, res in targets.items():
                if not res.detectable_flag:
                    continue
                is_edge = False
                if equip.type == 1:  # 仅雷达做边缘判定，卫星视作"舒适区"
                    trk = obs.dict_system_track.get(t_id)
                    if trk:
                        _, _, r = pm.geodetic2aer(
                            trk.latitude, trk.longitude, trk.altitude,
                            equip.latitude, equip.longitude, equip.altitude,
                        )
                        is_edge = (r / 1000.0) > (equip.range_max * 0.85)
                visible.setdefault(t_id, {})[s_id] = is_edge

        # ---- 2. 动作结算：非法动作惩罚 + 每目标锁定者列表 ----
        locks = {}  # t_id -> [s_id, ...]（仅合法锁定）
        for cmd in self._last_actions:
            if not cmd.str_target_id:
                continue
            s_id, t_id = cmd.str_equip_id, cmd.str_target_id
            if t_id not in visible or s_id not in visible[t_id]:
                reward -= 5.0  # R_valid
            else:
                locks.setdefault(t_id, []).append(s_id)

        # ---- 3. 覆盖 / 冗余 / 接力（按目标结算） ----
        for t_id, able in visible.items():
            locked = locks.get(t_id, [])
            n = len(locked)

            if n == 0:
                reward -= 20.0  # R_miss
                continue

            reward += 2.0  # R_cover

            if n > K:
                reward -= 5.0 * (n - K)  # R_redundant

            edge = sum(1 for s in locked if able.get(s, False))
            comfort = n - edge
            if n == 1 and edge == 1:
                reward -= 3.0  # R_danger：边缘孤机，无友军接力
            elif edge > 0 and comfort > 0:
                reward += 8.0  # R_relay：边缘 + 区 完美接力

        return float(reward)

    def generate_reward_bk5(self) -> float:
        """旧版覆盖式奖励（雷达+卫星），重设计前备份。"""
        reward = 0.0
        if not self._current_obs or not self._last_actions:
            return reward

        radar_can_see = {s_id : set() for s_id in self._dict_radar_info.keys()}
        sat_can_see = {s_id: set() for s_id in self._dict_satellite_info.keys()}

        radar_visible_targets = set()
        sat_visible_targets = set()

        for s_id, targets_dict in self._current_obs.dict_detection_result.items():
            is_radar = s_id in radar_can_see
            is_sat = s_id in sat_can_see

            for t_id, res in targets_dict.items():
                if res.detectable_flag:
                    if is_radar:
                        radar_can_see[s_id].add(t_id)
                        radar_visible_targets.add(t_id)
                    elif is_sat:
                        sat_can_see[s_id].add(t_id)
                        sat_visible_targets.add(t_id)

        radar_tracked_targets = set()
        sat_tracked_targets = set()

        for cmd in self._last_actions:
            if cmd.str_target_id:
                s_id = cmd.str_equip_id
                t_id = cmd.str_target_id

                is_radar = s_id in radar_can_see
                is_sat = s_id in sat_can_see

                if is_radar:
                    if t_id not in radar_can_see.get(s_id, set()):
                        reward -= 5.0
                    else:
                        radar_tracked_targets.add(t_id)
                elif is_sat:
                    if t_id not in sat_can_see.get(s_id, set()):
                        reward -= 5.0
                    else:
                        sat_tracked_targets.add(t_id)

        reward += (len(radar_tracked_targets) + len(sat_tracked_targets)) * 10.0

        global_tracked = radar_tracked_targets | sat_tracked_targets

        radar_missed = radar_tracked_targets - global_tracked
        sat_missed = sat_tracked_targets - global_tracked

        radar_wasted_capacity = len(radar_can_see) - len(radar_tracked_targets)
        sat_wasted_capacity = len(sat_can_see) - len(sat_tracked_targets)

        radar_faulty = min(radar_wasted_capacity, len(radar_missed))
        sat_faulty = min(sat_wasted_capacity, len(sat_missed))

        reward -= (radar_faulty + sat_faulty) * 15.0

        return float(reward) / 10.0


    def generate_reward_BK4(self) -> float:
        """20260408 容量只有7，必然有14个目标看不了，这是物理极限，不扣分"""
        reward = 0.0
        if not self._current_obs or not self._last_actions:
            return reward

        # 20260409
        # radar_can_see = {s_id: set() for s_id in self._dict_radar_info.keys()}
        radar_can_see = {s_id: set() for s_id in self._current_obs.dict_detection_result.keys()}
        global_visible_targets =  set()

        for s_id, targets_dict in self._current_obs.dict_detection_result.items():
            for t_id, res in targets_dict.items():
                if res.detectable_flag:
                    radar_can_see[s_id].add(t_id)
                    global_visible_targets.add(t_id)

        success_track_targets = set()

        for cmd in self._last_actions:
            if cmd.str_target_id:
                s_id = cmd.str_equip_id
                t_id = cmd.str_target_id

                if t_id not in radar_can_see.get(s_id, set()):
                    reward -= 5.0
                else:
                    success_track_targets.add(t_id)

        reward += len(success_track_targets) * 10.0

        missed_visible_targets = global_visible_targets - success_track_targets

        # 20260409
        # total_radar_capacity = len(self._dict_radar_info)
        total_radar_capacity = len(self._dict_radar_info) + len(self._dict_satellite_info)

        wastes_capacity = total_radar_capacity - len(success_track_targets)

        fautly_misses = min(wastes_capacity, len(missed_visible_targets))

        reward -= fautly_misses * 15.0

        return reward


    def generate_reward_bk3(self):
        """20260407"""
        reward = 0.0
        if not self._current_obs or not self._last_actions:
            return reward

        radar_can_see = {s_id: set() for s_id in self._dict_radar_info.keys()}

        for s_id, targets_dict in self._current_obs.dict_detection_result.items():
            for t_id, res in targets_dict.items():
                if res.detectable_flag:
                    radar_can_see[s_id].add(t_id)

        success_track_targets = set()

        for cmd in self._last_actions:
            if cmd.str_target_id:
                s_id = cmd.str_equip_id
                t_id = cmd.str_target_id

                if t_id not in radar_can_see.get(s_id, set()):
                    reward -= 5.0
                else:
                    success_track_targets.add(t_id)

        reward += len(success_track_targets) * 10.0

        return float(reward)

    def generate_reward_bk2(self):
        """出所测试用"""
        reward = 0.0
        if not self._current_obs:
            return reward

        radar_can_see = {s_id: set() for s_id in self._dict_radar_info.keys()}

        for s_id, targets_dict in self._current_obs.dict_detection_result.items():
            for t_id, res in targets_dict.items():
                if res.detectable_flag:
                    radar_can_see[s_id].add(t_id)


        target_locked_by = {t_id:[] for t_id in self._dict_target_info.keys()}
        radar_tracking = {s_id:[] for s_id in self._dict_radar_info.keys()}

        for cmd in self._last_actions:
            if cmd.str_target_id:
                s_id, t_id = cmd.str_equip_id, cmd.str_target_id

                if t_id not in radar_can_see.get(s_id, set()):
                    reward -= 5.0
                else:
                    radar_tracking[s_id].append(t_id)
                    target_locked_by[t_id].append(s_id)

        for s_id, tracking_list in radar_tracking.items():
            load = len(tracking_list)
            capacity = self._dict_radar_info[s_id].track_num_max
            visible_targets = radar_can_see.get(s_id, set())

            if load > capacity:
                reward -= 10.0 * (load - capacity)

            if load == 0 and len(visible_targets) > 0:
                reward -= 15.0

        for t_id, locked_by_radars in target_locked_by.items():
            lock_count = len(locked_by_radars)

            if lock_count == 1:
                reward += 10

            elif lock_count > 1:
                reward += 10.0
                reward -= 8.0 * (lock_count - 1)

        return reward


    def generate_reward_bk1(self) -> float:
        """
        面向接力跟踪与效能节约的多智能体协同奖励函数
        """
        reward = 0.0
        if not self._current_obs: 
            return reward

        # ==========================================
        # 1. 物理态势感知与边缘检测
        # ==========================================
        # 结构: {target_id: {radar_id: is_at_edge (bool)}}
        visible_matrix = {} 
        
        for s_id, targets_dict in self._current_obs.dict_detection_result.items():
            equip_state = self._current_obs.dict_equip_state.get(s_id)
            if not equip_state: continue
                
            for t_id, res in targets_dict.items():
                if res.detectable_flag:
                    if t_id not in visible_matrix:
                        visible_matrix[t_id] = {}
                        
                    # 计算目标是否处于该雷达的“交接警戒区”（探测边缘）
                    is_at_edge = False
                    trk = self._current_obs.dict_system_track.get(t_id)
                    if trk:
                        # 简化距离计算，判断是否处于最大射程的 85% 以外
                        a, e, r = pm.geodetic2aer(
                            trk.latitude, trk.longitude, trk.altitude,
                            equip_state.latitude, equip_state.longitude, equip_state.altitude
                        )
                        if (r / 1000.0) > (equip_state.range_max * 0.85):
                            is_at_edge = True
                            
                    visible_matrix[t_id][s_id] = is_at_edge

        # ==========================================
        # 2. 动作结果统计
        # ==========================================
        target_locks = {t_id: [] for t_id in self._dict_target_info.keys()}
        sensor_loads = {s_id: 0 for s_id in self._dict_radar_info.keys()}

        for cmd in self._last_actions:
            if cmd.str_target_id:
                s_id, t_id = cmd.str_equip_id, cmd.str_target_id
                
                # 容量统计
                if s_id in sensor_loads: 
                    sensor_loads[s_id] += 1
                
                # 非法动作惩罚：锁定不可见目标
                if t_id not in visible_matrix or s_id not in visible_matrix[t_id]:
                    reward -= 5.0  
                else:
                    target_locks[t_id].append(s_id)

        # ==========================================
        # 3. 业务逻辑：覆盖、效能与接力结算 (K=4 版本)
        # ==========================================
        for t_id, able_sensors in visible_matrix.items():
            locked_by = target_locks.get(t_id, [])
            lock_count = len(locked_by)

            # --- R_cov: 绝对覆盖奖惩 ---
            if lock_count == 0:
                reward -= 20.0  # 致命错误：目标在眼皮底下丢失
                continue

            # --- R_relay & R_eff: 接力与效能结算 (K=4) ---
            if lock_count == 1:
                # 单机跟踪
                s_id = locked_by[0]
                is_edge = able_sensors.get(s_id, False)
                
                if is_edge:
                    # 【危险预警】：处于边缘且无友军接力，随时会丢
                    reward += 1.0  
                else:
                    # 【单机区】：最省效能的跟踪
                    reward += 8.0  

            elif lock_count == 2:
                # 2 部雷达跟踪
                s1, s2 = locked_by[0], locked_by[1]
                edge1 = able_sensors.get(s1, False)
                edge2 = able_sensors.get(s2, False)
                
                if (edge1 and not edge2) or (edge2 and not edge1):
                    # 【完美接力奖赏】：一部雷达即将脱锁，另一部雷达区稳稳接住
                    reward += 15.0  
                else:
                    # 【双机稳健重叠】：都在区或边缘
                    reward += 8.0   

            elif lock_count == 3:
                # 【三机融合】：高冗余跟踪，容忍度内，但收益开始递减
                reward += 4.0       

            elif lock_count == 4:
                # 【四机饱和】：达到 K=4 的容忍上限，勉强不扣分
                reward += 1.0       

            elif lock_count >= 5:
                # --- R_eff: 惩罚无效冗余 ---
                # 超过 4 部雷达就是纯粹的浪费，多一部扣 5 分
                reward -= 5.0 * (lock_count - 4)

        # ==========================================
        # 4. 物理约束：超载惩罚
        # ==========================================
        for s_id, load_num in sensor_loads.items():
            sensor = self._dict_radar_info.get(s_id)
            if sensor and load_num > sensor.track_num_max:
                reward -= 20.0 * (load_num - sensor.track_num_max) # 严惩超载

        return reward


"""
1.绝对覆盖奖惩 ($R_{cov}$)：目标只要在任何雷达的视野内，就必须被至少 1 部雷达锁定。若目标“裸奔”，给予极高惩罚。
2.效能节约惩罚 ($R_{eff}$)：打破“越多越好”的贪婪策略。定义 $K=2$ 为最佳接力冗余度。超过 2 部雷达同时跟踪同一个目标，即视为浪费，给予线性惩罚。
3.边缘接力奖赏 ($R_{relay}$ - 秘籍)：如何让模型学会交接？当目标处于雷达 A 的探测边缘（例如距离 $> 0.85 \times R_{max}$），此时雷达 A 极易脱锁。
    如果此时只有雷达 A 跟踪，给予危险惩罚，逼迫网络呼叫支援。如果此时雷达 B（目标在其区）提前介入，形成“A边缘 + B”的双机共轨状态，给予最高额的接力配合奖赏。
4.物理约束惩罚 ($R_{phy}$)：严惩违背规律的动作（瞎指、超载）。
"""