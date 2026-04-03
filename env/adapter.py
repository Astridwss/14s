# env/adapter.py
import os
import json
import numpy as np
import pymap3d as pm
from typing import List
from sim import AgentActionCommand, AgentObservation

class ScenarioAdapter:
    """
    场景与态势解析的统一枢纽：单一来源

    1. 静态推导：解析场景预案，用纯数学公式推导神经网络张量维度，向Config 注入。
    2. 动态提取：在仿真 step 中，将仿真引擎吐出的复杂对象 (AgentObservation) 转化为规整的 Numpy 张量。
    
    """
    RADAR_OBS_DIM = 8      
    TARGET_OBS_DIM = 5     
    RADAR_STATE_DIM = 5    # 全局状态: [分配目标索引, 负载率, x, y, z]
    TARGET_STATE_DIM = 8   # 全局状态: [ecf_x, ecf_y, ecf_z, ecf_vx, ecf_vy, ecf_vz, rcs/lock, type]

    @classmethod
    def parse_and_inject(cls, conf):
        """【API 层专属调用】：静态推导网络维度"""
        
        # 1. 提取路径
        local_scene_path = getattr(conf, 'local_scene_path', None)
        plan_id = getattr(conf, 'plan_id', 867)

        # 2. 调用解耦的静态方法获取数量 (使用 cls. 或者 ScenarioAdapter. 调用)
        n_radars, n_satellites, n_targets, radar_keys, target_keys, satellites_keys = cls.extract_agents(local_scene_path, plan_id)

        # 3. 动态注入到 conf 中
        conf.radar_keys = radar_keys
        conf.satellites_keys = satellites_keys
        conf.target_keys = target_keys
        
        conf.n_radars = n_radars
        conf.n_satellites = n_satellites
        conf.n_targets = n_targets

        #TODO:conf.n_agents等待传入真实的卫星数量，可能与网络维度产生报错。
        conf.n_agents = n_radars 
        conf.n_actions = n_targets + 1  # 动作0=待机，1~N=对应目标
        
        # 4. 根据公式推导强化学习的Tensor维度
        conf.radar_obs_dim = cls.RADAR_OBS_DIM + (n_targets * cls.TARGET_OBS_DIM) 
        conf.obs_shape = conf.radar_obs_dim
        conf.state_shape = (n_targets * cls.TARGET_STATE_DIM) + (n_radars * cls.RADAR_STATE_DIM) + 1
        
        #fix：智能体个数与真实不符，待接入真实WX数据
        print(f"[ScenarioAdapter] 维度注入成功: n_agents={conf.n_agents + n_satellites}, n_actions={conf.n_actions}, "
              f"obs={conf.obs_shape}, state={conf.state_shape}")


    @staticmethod
    def extract_agents(local_scene_path: str, plan_id: int = 867):
        """
        通过 PlanFileProcess 按 plan_id 解析当前场景中实体数量和真实 ID。
        任何实体数量为 0 或文件异常，抛出错误截断任务
        """        
        if not local_scene_path or not os.path.exists(local_scene_path):
            raise FileNotFoundError(f"[ScenarioAdapter]：场景文件不存在: {local_scene_path}")
            
        try:
            from sim.plan_file_process import PlanFileProcess
            processor = PlanFileProcess()
            battle_scene = processor.read_battle_scene_from_json(plan_id=plan_id, file_path=local_scene_path)
            
            # 拿到真实的列表
            radar_keys = list(battle_scene.dict_radar_id_info.keys())
            satellites_keys = list(battle_scene.dict_satellite_id_info.keys())
            target_keys = list(battle_scene.dict_target_id_info.keys())
            
            # 拿到真实的数量
            n_radars = len(radar_keys)
            n_satellites = len(satellites_keys)
            n_targets = len(target_keys)
            
            if n_radars == 0 or n_targets == 0:
                raise ValueError(
                    f"[ScenarioAdapter] 场景数据异常"
                    f"解析到 雷达={n_radars}, 卫星={n_satellites}, 目标={n_targets}。"
                    f"实体数量绝不能为 0，请检查场景 JSON 预案"
                )
            
            print(f"[ScenarioAdapter] 成功通过 PlanFileProcess 提取 (PlanID:{plan_id}): 雷达={n_radars}, 卫星={n_satellites}, 目标={n_targets}")
            
            return n_radars, n_satellites, n_targets, radar_keys, target_keys, satellites_keys
            
        except Exception as e:
            print(f"[ScenarioAdapter] 解析失败: {e}，装备数量解析错误")
            raise e


    def __init__(self, conf):
        self.conf = conf

    def extract_action_masks(self, agent_obs: AgentObservation) -> np.ndarray:
        action_mask = np.zeros((self.conf.n_radars, self.conf.n_actions), dtype=np.float32)
        action_mask[:, 0] = 1.0  
        
        # 使用注入好的实体ID列表
        radar_keys = getattr(self.conf, 'radar_keys', [])
        target_keys = getattr(self.conf, 'target_keys', [])

        for sid, target_dict in agent_obs.dict_detection_result.items():
            if sid in radar_keys:
                ri = radar_keys.index(sid)
                for tid, res in target_dict.items():
                    if res.detectable_flag and tid in target_keys:
                        ti = target_keys.index(tid)
                        action_mask[ri, ti + 1] = 1.0  
        return action_mask

    def extract_observations(self, agent_obs: AgentObservation) -> np.ndarray:
        obs = np.zeros((self.conf.n_radars, self.conf.radar_obs_dim), dtype=np.float32)
        track_dict = agent_obs.dict_system_track
        sensor_by_id = agent_obs.dict_equip_state
        
        radar_keys = getattr(self.conf, 'radar_keys', [])
        target_keys = getattr(self.conf, 'target_keys', [])
        
        target_lock_counts = {}  
        for s in sensor_by_id.values():
            for tid in s.lst_track_no:
                target_lock_counts[tid] = target_lock_counts.get(tid, 0) + 1

        for i in range(self.conf.n_radars):
            #  使用真实 ID 去底层环境要数据
            real_sid = radar_keys[i] if i < len(radar_keys) else f"RADAR_{i:03d}"
            s = sensor_by_id.get(real_sid)
            if s is not None:
                max_cap = max(1, s.track_num_max)
                obs[i, 0] = s.residual_track_num / max_cap
                if s.lst_track_no:
                    locked_tid = s.lst_track_no[0]
                    if locked_tid in target_keys:
                        obs[i, 1] = (target_keys.index(locked_tid) + 1) / self.conf.n_actions
                
                obs[i, 2] = s.range_min / 4000.0                  
                obs[i, 3] = s.range_max / 4000.0                  
                obs[i, 4] = s.azi_min / 360.0                     
                obs[i, 5] = s.azi_max / 360.0                     
                obs[i, 6] = s.ele_min / 90.0                      
                obs[i, 7] = s.ele_max / 90.0                      

        for ri in range(self.conf.n_radars):
            real_sid = radar_keys[ri] if ri < len(radar_keys) else f"RADAR_{ri:03d}"
            s = sensor_by_id.get(real_sid)
            if not s: continue
            
            for ti in range(self.conf.n_targets):
                real_tid = target_keys[ti] if ti < len(target_keys) else f"TARGET_{ti:03d}"
                res = agent_obs.dict_detection_result.get(real_sid, {}).get(real_tid)
                
                if not res or not res.detectable_flag: continue  
                
                base_idx = self.RADAR_OBS_DIM + ti * self.TARGET_OBS_DIM
                obs[ri, base_idx + 0] = 1.0  
                
                t = track_dict.get(real_tid)
                if t:
                    a, e, r = pm.geodetic2aer(t.latitude, t.longitude, t.altitude, s.latitude, s.longitude, s.altitude)
                    obs[ri, base_idx + 1] = (r / 1000.0) / 4000.0  
                    obs[ri, base_idx + 2] = a / 360.0              
                    obs[ri, base_idx + 3] = e / 90.0               
                
                obs[ri, base_idx + 4] = target_lock_counts.get(real_tid, 0) / self.conf.n_radars
        return obs

    def extract_global_state(self, agent_obs: AgentObservation) -> np.ndarray:
        """提取上帝视角全局状态"""
        state_elements = []
        
        target_lock_counts = {}
        for s in agent_obs.dict_equip_state.values():
            for tid in s.lst_track_no:
                target_lock_counts[tid] = target_lock_counts.get(tid, 0) + 1

        radar_keys = getattr(self.conf, 'radar_keys', [])
        target_keys = getattr(self.conf, 'target_keys', [])

        # 1. 目标真实状态矩阵 (n_targets, 8)
        target_states = np.zeros((self.conf.n_targets, self.TARGET_STATE_DIM), dtype=np.float32)
        track_dict = agent_obs.dict_system_track  
        
        for i in range(self.conf.n_targets):
            #  使用真实目标 ID 查字典
            real_tid = target_keys[i] if i < len(target_keys) else f"TARGET_{i:03d}"
            if real_tid in track_dict:
                t = track_dict[real_tid]
                lock_rate = target_lock_counts.get(real_tid, 0) / self.conf.n_radars
                target_states[i] = [
                    t.ecf_x / 10000.0, t.ecf_y / 10000.0, t.ecf_z / 10000.0,
                    t.ecf_vx / 10.0, t.ecf_vy / 10.0, t.ecf_vz / 10.0,
                    lock_rate, 1.0
                ]
        state_elements.append(target_states.flatten())
        
        # 2. 雷达全盘分配与空间状态矩阵 (n_radars, 5)
        radar_status = np.zeros((self.conf.n_radars, self.RADAR_STATE_DIM), dtype=np.float32)
        sensor_dict = agent_obs.dict_equip_state  
        
        for i in range(self.conf.n_radars):
            #  使用真实雷达 ID 查字典
            real_sid = radar_keys[i] if i < len(radar_keys) else f"RADAR_{i:03d}"
            s = sensor_dict.get(real_sid)
            if s is not None:
                assigned_idx = -1.0
                if s.lst_track_no:
                    locked_tid = s.lst_track_no[0]
                    if locked_tid in target_keys:
                        assigned_idx = float(target_keys.index(locked_tid)) / self.conf.n_targets
                
                max_cap = max(1, s.track_num_max)
                r_x_m, r_y_m, r_z_m = pm.geodetic2ecef(s.latitude, s.longitude, s.altitude)
                r_x_km, r_y_km, r_z_km = r_x_m / 1000.0, r_y_m / 1000.0, r_z_m / 1000.0
                
                radar_status[i] = [
                    r_x_km / 10000.0, r_y_km / 10000.0, r_z_km / 10000.0,         
                    len(s.lst_track_no) / max_cap, assigned_idx                  
                ]
            else:
                radar_status[i] = [0, 0, 0, 0, -1.0] 
                
        state_elements.append(radar_status.flatten())
        
        # 3. 追加时间
        time_norm = np.array([agent_obs.current_time / 600.0], dtype=np.float32)
        state_elements.append(time_norm)
        
        # 返回63维一维数组
        return np.concatenate(state_elements)

    def discrete_actions_to_agent_action(self, actions: List[int], current_time: float) -> List[AgentActionCommand]:
        """ RL 输出动作时，包装成含有 ID 的指令"""
        commands = []
        t_int = int(current_time) 
        radar_keys = getattr(self.conf, 'radar_keys', [])
        target_keys = getattr(self.conf, 'target_keys', [])
        
        for i in range(min(len(actions), self.conf.n_radars)):
            a = int(actions[i])
            # 下达真实 ID
            sensor_id = radar_keys[i] if i < len(radar_keys) else f"RADAR_{i:03d}"
            
            if a <= 0 or a > self.conf.n_targets:
                commands.append(AgentActionCommand(time=t_int, str_equip_id=sensor_id, str_target_id=""))
            else:
                t_idx = a - 1
                target_id = target_keys[t_idx] if t_idx < len(target_keys) else f"TARGET_{t_idx:03d}"
                commands.append(AgentActionCommand(time=t_int, str_equip_id=sensor_id, str_target_id=target_id))
                
        # 补齐未分配的雷达待机指令
        for i in range(len(commands), self.conf.n_radars):
            sensor_id = radar_keys[i] if i < len(radar_keys) else f"RADAR_{i:03d}"
            commands.append(AgentActionCommand(time=t_int, str_equip_id=sensor_id, str_target_id=""))
            
        return commands



    