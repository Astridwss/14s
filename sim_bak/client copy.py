# sim/client.py (原 TrainingEnv)
import pymap3d as pm
from typing import Dict, List, Tuple
from .datastruct import (
    SystemTrackBase, EquipmentState, EquipmentToTargetDetectionResult, 
    AgentObservation, AgentActionCommand
)
from .plan_file_process import PlanFileProcess

class TrainingEnv:
    """
    加载预案、时间步进、推演物理状态。
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

        # 2. 生成系统航迹
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
        action_map = {cmd.str_equip_id: cmd.str_target_id for cmd in agent_actions}

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
                equip_state.lst_track_no.append(target_to_lock)
            
            equip_state.residual_track_num = max(0, equip_state.track_num_max - len(equip_state.lst_track_no))
            agent_observation.dict_equip_state[equip_state.str_equip_id] = equip_state

        # 5. 计算可视性矩阵
        for equip_id, equip_info in agent_observation.dict_equip_state.items():
            agent_observation.dict_detection_result[equip_id] = {}
            for target_id, system_track in agent_observation.dict_system_track.items():
                det_res = EquipmentToTargetDetectionResult(
                    str_equip_id=equip_id, str_target_id=target_id, detectable_flag=False
                )
                a, e, r = pm.geodetic2aer(
                    system_track.latitude, system_track.longitude, system_track.altitude,
                    equip_info.latitude, equip_info.longitude, equip_info.altitude
                )
                if (equip_info.azi_min <= a <= equip_info.azi_max and 
                    equip_info.ele_min <= e <= equip_info.ele_max and 
                    equip_info.range_min <= r / 1000.0 <= equip_info.range_max):
                    det_res.detectable_flag = True
                
                agent_observation.dict_detection_result[equip_id][target_id] = det_res

        self._current_obs = agent_observation # 暂存当前帧，留给以后的算分器使用
        return agent_observation, terminate_flag



        """
        ld的覆盖个数 U wx的覆盖个数 = 总覆盖个数
        可见目标个数 - 总覆盖个数 = 漏掉的可见目标
        min（漏掉的可见个数，ld重叠覆盖浪费个数+wx重叠覆盖浪费个数）
        
        漏掉目标 = 0 给最大奖励，并且不扣分
        漏掉目标 > 0 扣分 = min（漏掉的可见个数，ld重叠覆盖浪费个数+wx重叠覆盖浪费个数）* 15
        """ 
    def generate_reward(self) -> float:
        """
        面向 18(7雷达+11卫星) vs 21 目标 的【异构跨域协同版】奖励函数
        核心逻辑：
        1. 允许跨域重叠：雷达和卫星同时盯一个目标，可分别获得奖励。
        2. 严惩同域重叠：雷达之间、卫星之间依然严惩扎堆。
        3. 责任隔离的漏检惩罚：只为自己物理可见、且全局无人理睬、且自己有闲置容量的“过失”买单。
        """
        reward = 0.0
        
        if not self._current_obs or not self._last_actions: 
            return reward

        # ==========================================
        # 1. 态势解析：按兵种（雷达/卫星）分别建立专属视野
        # ==========================================
        radar_can_see = {s_id: set() for s_id in self._dict_radar_info.keys()}
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

        # ==========================================
        # 2. 动作判定与有效覆盖统计（同兵种去重，跨兵种允许重叠）
        # ==========================================
        radar_tracked_targets = set()
        sat_tracked_targets = set()

        for cmd in self._last_actions:
            if cmd.str_target_id:
                s_id = cmd.str_equip_id
                t_id = cmd.str_target_id
                
                is_radar = s_id in radar_can_see
                is_sat = s_id in sat_can_see
                
                # [规则 A：物理底线越界惩罚]
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

        # [规则 B：基础覆盖奖励] 
        # 因为 radar 和 sat 是分开的 set()，如果它们看同一个目标，这里会各自算 1 次奖励。
        # 完美支持你的需求：“卫星可以跟雷达重叠跟踪”。
        reward += (len(radar_tracked_targets) + len(sat_tracked_targets)) * 10.0

        # ==========================================
        # 3. 结果导向：责任隔离的过失漏检结算
        # ==========================================
        # 全局已经被防空网（无论雷达还是卫星）成功锁定的目标
        global_tracked = radar_tracked_targets | sat_tracked_targets

        # 漏掉的目标必须按兵种的物理视野来算！
        radar_missed = radar_visible_targets - global_tracked
        sat_missed = sat_visible_targets - global_tracked

        # 分别计算两个兵种的浪费容量
        radar_wasted_capacity = len(radar_can_see) - len(radar_tracked_targets)
        sat_wasted_capacity = len(sat_can_see) - len(sat_tracked_targets)

        # 真正的过失 (Faulty Misses)
        # 如果卫星没闲置容量了，即使天上还有一堆只有卫星能看见的目标，卫星的过失也为 0
        radar_faulty = min(radar_wasted_capacity, len(radar_missed))
        sat_faulty = min(sat_wasted_capacity, len(sat_missed))

        # [规则 C：过失漏检惩罚] 
        reward -= (radar_faulty + sat_faulty) * 15.0

        # [强烈建议保留的安全策略] 缩放奖励，防止 QMIX 在 600+ 步长下梯度爆炸
        return float(reward) / 10.0



    def generate_reward_bk4(self) -> float:
        """
        面向 7 雷达 vs 21 目标 (容量=1) 的【过失漏检惩罚版】协同奖励函数
        核心逻辑：
        1. 必然的漏检不惩罚：总容量只有7，必然有14个目标看不了，这是物理极限，不扣分。
        2. 过失的漏检严惩罚：如果有雷达重叠/待机（导致容量浪费），且全局视野里明明还有目标没被跟踪，这就是“过失漏检”，严厉扣分！
        """
        reward = 0.0
        
        if not self._current_obs or not self._last_actions: 
            return reward

        # ==========================================
        # 1. 态势解析：建立雷达专属视野 与 全局总视野
        # ==========================================
        #radar_can_see = {s_id: set() for s_id in self._dict_radar_info.keys()}
        radar_can_see = {s_id: set() for s_id in self._current_obs.dict_equip_state.keys()}
        global_visible_targets = set()  # 当前战区内，对防空网“可见”的所有目标总和
        
        for s_id, targets_dict in self._current_obs.dict_detection_result.items():
            for t_id, res in targets_dict.items():
                if res.detectable_flag:
                    radar_can_see[s_id].add(t_id)
                    global_visible_targets.add(t_id)

        # ==========================================
        # 2. 动作判定与有效覆盖统计
        # ==========================================
        successfully_tracked_targets = set()

        for cmd in self._last_actions:
            if cmd.str_target_id:
                s_id = cmd.str_equip_id
                t_id = cmd.str_target_id
                
                # [规则 A：物理底线越界惩罚] 瞎指挥，必扣分
                if t_id not in radar_can_see.get(s_id, set()):
                    reward -= 5.0  
                else:
                    successfully_tracked_targets.add(t_id)
        
        # [规则 B：基础覆盖奖励] 每合法盯住一个不同的目标，给 10 分（维持智能体积极性）
        reward += len(successfully_tracked_targets) * 10.0

        # ==========================================
        # 3. 核心创新：计算“过失漏检”并实施惩罚
        # ==========================================
        # 漏掉的可见目标 = 全局可见的总目标 - 已经被成功盯上的目标
        missed_visible_targets = global_visible_targets - successfully_tracked_targets
        
        # 浪费的雷达容量 = 雷达总数 - 当前已有效覆盖的目标数
        # total_radar_capacity = len(self._dict_radar_info) 
        # wasted_capacity = total_radar_capacity - len(successfully_tracked_targets)

        total_agent_capacity = len(self._dict_radar_info) + len(self._dict_satellite_info) 
        wasted_capacity = total_agent_capacity - len(successfully_tracked_targets)

        # 真正的过失 (Faulty Misses)：
        # 如果容量打满了 (wasted_capacity=0)，即使漏了 14 个，过失也为 0。
        # 如果没目标了 (len(missed)=0)，即使有 3 部雷达待机，过失也为 0。
        faulty_misses = min(wasted_capacity, len(missed_visible_targets))

        # [规则 C：过失漏检惩罚] 每因为重叠或待机而“白白放跑”一个可见目标，重罚！
        reward -= faulty_misses * 15.0

        return float(reward)
    

    def generate_reward_bk3(self) -> float:
        """
        面向 7 雷达 vs 21 目标 (单设备容量=1) 的极简结果导向协同奖励函数
        核心逻辑：
        1. 物理底线：雷达绝不能分配给不在自己探测范围内的目标（严惩）。
        2. 全局协同：利用 set 集合去重，计算全局真实覆盖的“唯一目标数”（奖励）。
           - 自动惩罚重叠：多部雷达看同一个目标，去重后只算1个目标的奖励。
           - 自动惩罚怠工：雷达不干活，去重后的总覆盖数直接减少，错失奖励。
        """
        reward = 0.0
        
        # 防御性判断：如果没有观测数据或没有任何动作，直接返回 0
        if not self._current_obs or not self._last_actions: 
            return reward

        # ==========================================
        # 1. 态势解析：建立每部雷达的【真实物理可用目标】白名单字典
        # 结构: {radar_id: set(可见且满足所有物理条件约束的 target_id)}
        # ==========================================
        radar_can_see = {s_id: set() for s_id in self._dict_radar_info.keys()}
        
        for s_id, targets_dict in self._current_obs.dict_detection_result.items():
            for t_id, res in targets_dict.items():
                if res.detectable_flag:  # 这个 flag 是经过严格的 R, A, E 等计算出的物理可见性
                    radar_can_see[s_id].add(t_id)

        # ==========================================
        # 2. 动作判定与有效覆盖统计
        # ==========================================
        successfully_tracked_targets = set()

        for cmd in self._last_actions:
            # 只有当智能体确实下发了某个具体目标时才进行判定（排除了动作 0 即待机指令）
            if cmd.str_target_id:
                s_id = cmd.str_equip_id
                t_id = cmd.str_target_id
                
                # [规则 A：物理底线越界惩罚]
                # 试图追踪一个根本看不见的目标，属于严重瞎指挥，必须予以打击
                if t_id not in radar_can_see.get(s_id, set()):
                    reward -= 5.0  
                else:
                    # 动作合法，将该目标加入全局覆盖集合
                    successfully_tracked_targets.add(t_id)
        
        # ==========================================
        # 3. 结果导向：全局战果结算
        # ==========================================
        # [规则 B：全局覆盖奖励] 
        # 集合(set)天然具备去重功能。
        # 如果 7 部雷达各看 1 个目标，len 为 7，总奖励为 +70.0（完美协同）。
        # 如果 7 部雷达全挤在 1 个目标上，len 为 1，总奖励仅为 +10.0（隐性协同惩罚）。
        reward += len(successfully_tracked_targets) * 10.0

        return float(reward)

    def generate_reward_bk2(self) -> float:
        """
        面向资源受限 (容量远小于目标数) 的简化协同奖励函数
        核心：防怠工、防重叠跟踪、最大化有效目标覆盖
        """
        reward = 0.0
        if not self._current_obs: 
            return reward

        # ==========================================
        # 1. 态势解析：建立雷达视角的可用目标字典
        # 结构: {radar_id: set(可见且合法的 target_id)}
        # ==========================================
        radar_can_see = {s_id: set() for s_id in self._dict_radar_info.keys()}
        
        for s_id, targets_dict in self._current_obs.dict_detection_result.items():
            for t_id, res in targets_dict.items():
                if res.detectable_flag:
                    radar_can_see[s_id].add(t_id)

        # ==========================================
        # 2. 动作统计
        # ==========================================
        target_locked_by = {t_id: [] for t_id in self._dict_target_info.keys()}
        radar_tracking = {s_id: [] for s_id in self._dict_radar_info.keys()}

        for cmd in self._last_actions:
            if cmd.str_target_id:
                s_id, t_id = cmd.str_equip_id, cmd.str_target_id
                
                # 非法动作：雷达试图跟踪一个视野外/不可见的目标
                if t_id not in radar_can_see.get(s_id, set()):
                    reward -= 5.0  
                else:
                    radar_tracking[s_id].append(t_id)
                    target_locked_by[t_id].append(s_id)

        # ==========================================
        # 3. 智能体行为结算 (雷达视角)
        # ==========================================
        for s_id, tracking_list in radar_tracking.items():
            load = len(tracking_list)
            capacity = self._dict_radar_info[s_id].track_num_max  # 你的场景下是 1
            visible_targets = radar_can_see.get(s_id, set())

            # [规则 A]: 超载惩罚
            if load > capacity:
                reward -= 10.0 * (load - capacity)

            # [规则 B]: 怠工惩罚 (核心业务要求)
            # 如果什么都没跟踪，但其实视野里是有目标的，严惩！
            if load == 0 and len(visible_targets) > 0:
                reward -= 15.0 
                
            # (补充逻辑)：如果 load == 0 且视野里本来就没目标，这是合理的待机，不奖不惩

        # ==========================================
        # 4. 全局效能结算 (目标视角)
        # ==========================================
        for t_id, locked_by_radars in target_locked_by.items():
            lock_count = len(locked_by_radars)

            # [规则 C]: 唯一跟踪奖励
            if lock_count == 1:
                reward += 10.0  # 完美！占用1个容量，贡献1个视野

            # [规则 D]: 冗余浪费惩罚
            elif lock_count > 1:
                # 依然给 10 分的覆盖基础分，但因为浪费了极其宝贵的额外容量，扣除浪费分
                # 例如：2部雷达看同1个目标，相当于浪费了1个本可以看其他目标的雷达
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