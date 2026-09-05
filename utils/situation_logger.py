"""
态势日志器 —— 纯工具，不依赖 Config 或 Adapter。

所有数据（雷达 keys、目标 keys、动作掩码）由调用方显式传入。
"""
import json
import math
from typing import List, Dict, Any
import numpy as np


class SituationJSONLogger:
    """基于 JSON 结构的多智能体战术态势结构化日志器。"""

    @staticmethod
    def _extract_radar_and_actions(obs, actions, radar_keys, target_keys, action_masks):
        """抽取雷达状态、掩码可见目标与实际指令。"""
        action_dict = {cmd.str_equip_id: cmd.str_target_id
                       for cmd in actions if cmd.str_equip_id}

        radar_list = []
        for i, r_id in enumerate(radar_keys):
            equip_state = obs.dict_equip_state.get(r_id)
            if not equip_state:
                continue

            load_str = f"{len(equip_state.lst_track_no)}/{equip_state.track_num_max}"

            visible_targets = []
            if i < len(action_masks):
                for t_idx, mask_val in enumerate(action_masks[i][1:]):
                    if mask_val == 1.0 and t_idx < len(target_keys):
                        visible_targets.append(target_keys[t_idx])

            target_id = action_dict.get(r_id, "")
            action_str = target_id if target_id else "STANDBY (待机)"

            radar_list.append({
                "radar_id": r_id,
                "load": load_str,
                "visible_targets": visible_targets,
                "action_cmd": action_str,
            })

        return radar_list

    @staticmethod
    def _extract_target_status(obs) -> List[Dict[str, Any]]:
        """抽取活跃目标的核心运动学特征。"""
        target_list = []
        for t_id, t_info in obs.dict_system_track.items():
            speed = math.sqrt(t_info.ecf_vx ** 2 + t_info.ecf_vy ** 2 + t_info.ecf_vz ** 2) * 1000
            alt_km = t_info.altitude / 1000.0
            target_list.append({
                "target_id": t_id,
                "alt_km": round(alt_km, 2),
                "speed_mps": round(speed, 1),
                "type": t_info.type,
            })
        return target_list

    @classmethod
    def generate_step_json(cls, step: int, obs, actions,
                           radar_keys, target_keys, action_masks, reward: float) -> str:
        """拼装结构化 JSON 字符串。"""
        if not obs:
            return "{}"
        log_dict = {
            "step": step,
            "time_s": obs.current_time,
            "reward": round(reward, 2),
            "radars": cls._extract_radar_and_actions(obs, actions, radar_keys, target_keys, action_masks),
            "targets": cls._extract_target_status(obs),
        }
        return json.dumps(log_dict, ensure_ascii=False)

    @classmethod
    def print_log(cls, step: int, obs, actions,
                  radar_keys, target_keys, action_masks, reward: float):
        """控制台打印入口。"""
        if not obs:
            return

        radars_data = cls._extract_radar_and_actions(obs, actions, radar_keys, target_keys, action_masks)
        targets_data = cls._extract_target_status(obs)

        print(f"\n[{'=' * 15} 态势日志 Step: {step} | "
              f"Time: {obs.current_time}s | Reward: {reward:.2f} {'=' * 15}]")

        print(f"雷达状态 ({len(radars_data)} 部):")
        for r in radars_data:
            print(f"  {json.dumps(r, ensure_ascii=False)}")

        print(f"活跃目标 ({len(targets_data)} 个):")
        for t in targets_data:
            print(f"  {json.dumps(t, ensure_ascii=False)}")

        print("=" * 70 + "\n")


class SituationLogHook:
    """态势日志 step 回调 —— 将 SituationJSONLogger 适配为 RolloutWorker 的 step_hook 接口。

    用法:
        log_hook = SituationLogHook(radar_keys, target_keys, log_interval=50)
        worker.generate_train_episode(epsilon=0.1, step_hooks=[log_hook])
    """

    def __init__(self, radar_keys: List[str], target_keys: List[str],
                 log_interval: int = 50):
        self.radar_keys = radar_keys
        self.target_keys = target_keys
        self.log_interval = log_interval

    def __call__(self, step_count: int, terminated: bool, truncated: bool,
                 next_info: dict):
        """每步回调 —— 按 log_interval 间隔打印态势。"""
        if step_count % self.log_interval != 0:
            return

        raw_obs = next_info.get('raw_obs')
        valid_cmds = next_info.get('agent_actions_list', [])
        if raw_obs is None:
            return

        SituationJSONLogger.print_log(
            step=step_count,
            obs=raw_obs,
            actions=valid_cmds,
            radar_keys=self.radar_keys,
            target_keys=self.target_keys,
            action_masks=next_info.get('avail_actions', []),
            reward=next_info.get('_last_reward', 0.0),
        )
