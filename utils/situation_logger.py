import json
import math
from typing import List, Dict, Any

from sim.datastruct import AgentObservation, AgentActionCommand
from env.adapter import ScenarioAdapter

class SituationJSONLogger:
    """
    基于 JSON 结构的多智能体战术态势结构化日志器
    """

    @staticmethod
    def _extract_radar_and_actions(obs: AgentObservation, 
                                   actions: List[AgentActionCommand], 
                                   adapter: ScenarioAdapter) -> List[Dict[str, Any]]:
        """【原子功能 1】：抽取雷达状态、掩码可见目标与实际指令"""
        radar_keys = getattr(adapter.conf, 'radar_keys', [])
        target_keys = getattr(adapter.conf, 'target_keys', [])
        
        # 抽取掩码
        action_masks = adapter.extract_action_masks(obs)
        # 将当前动作指令列表转为字典 {radar_id: target_id} 方便 O(1) 查询
        action_dict = {cmd.str_equip_id: cmd.str_target_id for cmd in actions if cmd.str_equip_id}

        radar_list = []
        for i, r_id in enumerate(radar_keys):
            equip_state = obs.dict_equip_state.get(r_id)
            if not equip_state:
                continue
            
            # 1. 负载状态
            load_str = f"{len(equip_state.lst_track_no)}/{equip_state.track_num_max}"
            
            # 2. 解析掩码：只提取掩码为 1.0 的目标 (索引 0 通常是待机动作)
            visible_targets = []
            if i < len(action_masks):
                for t_idx, mask_val in enumerate(action_masks[i][1:]):  # 跳过待机动作掩码
                    if mask_val == 1.0 and t_idx < len(target_keys):
                        visible_targets.append(target_keys[t_idx])
            
            # 3. 实际下达的动作
            target_id = action_dict.get(r_id, "")
            action_str = target_id if target_id else "STANDBY (待机)"

            radar_list.append({
                "radar_id": r_id,
                "load": load_str,
                "visible_targets": visible_targets,
                "action_cmd": action_str
            })

        return radar_list

    @staticmethod
    def _extract_target_status(obs: AgentObservation) -> List[Dict[str, Any]]:
        """【原子功能 2】：抽取活跃目标的核心运动学特征"""
        target_list = []
        for t_id, t_info in obs.dict_system_track.items():
            # 计算速度 (勾股定理，转换为 m/s)
            speed = math.sqrt(t_info.ecf_vx**2 + t_info.ecf_vy**2 + t_info.ecf_vz**2) * 1000
            alt_km = t_info.altitude / 1000.0

            target_list.append({
                "target_id": t_id,
                "alt_km": round(alt_km, 2),
                "speed_mps": round(speed, 1),
                "type": t_info.type
            })
            
        return target_list

    @classmethod
    def generate_step_json(cls, step: int, obs: AgentObservation, actions: List[AgentActionCommand], adapter: ScenarioAdapter, 
                           reward: float) -> str:
        """
        【主控接口】：拼装上述原子信息，并转换为结构化 JSON 字符串
        """
        if not obs:
            return "{}"

        # 构建根结构
        log_dict = {
            "step": step,
            "time_s": obs.current_time,
            "reward": round(reward, 2),
            "radars": cls._extract_radar_and_actions(obs, actions, adapter),
            "targets": cls._extract_target_status(obs)
        }

        # 转换为格式化的 JSON 字符串
        return json.dumps(log_dict, ensure_ascii=False)

    # @classmethod
    # def print_log(cls, *args, **kwargs):
    #     """控制台打印入口"""
    #     json_str = cls.generate_step_json(*args, **kwargs)
    #     print(f"\n[态势帧日志 START] {'='*40}")
    #     print(json_str)
    #     print(f"[态势帧日志 END] {'='*42}\n")

    @classmethod
    def print_log(cls, step: int, obs: AgentObservation, actions: List[AgentActionCommand], adapter: ScenarioAdapter, reward: float):

        if not obs:
            return
            
        # 1. 调用之前的提取函数，拿到结构化的 List[Dict]
        radars_data = cls._extract_radar_and_actions(obs, actions, adapter)
        targets_data = cls._extract_target_status(obs)
        
        # 2. 头部信息
        print(f"\n[{'='*15} 态势日志 Step: {step} | Time: {obs.current_time}s | Reward: {reward:.2f} {'='*15}]")
        
        # 3. 打印雷达：遍历 List，把每个字典转成单行 JSON 字符串
        print(f"雷达状态 ({len(radars_data)} 部):")
        for r in radars_data:
            print(f"  {json.dumps(r, ensure_ascii=False)}")
            
        # 4. 打印目标：遍历 List，同样单行输出
        print(f"活跃目标 ({len(targets_data)} 个):")
        for t in targets_data:
            print(f"  {json.dumps(t, ensure_ascii=False)}")
            
        print("=" * 70 + "\n")