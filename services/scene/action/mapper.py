"""
动作映射 —— RL 离散动作 ↔ 仿真引擎指令。

自包含积木：接收实体 ID 列表和维度，不依赖 Config。
"""
import numpy as np
from typing import List, Optional
from sim import AgentActionCommand, AgentObservation


class ActionMapper:
    """RL 动作空间与引擎指令的双向映射。"""

    def __init__(self, agent_keys: List[str], target_keys: List[str],
                 n_agents: int, n_actions: int, satellite_keys: Optional[List[str]] = None):
        self.agent_keys = agent_keys
        self.target_keys = target_keys
        self.n_agents = n_agents
        self.n_actions = n_actions
        # 显式卫星 key 集合：诊断「卫星多指向」用，不靠 id 位数猜
        # （mock 场景里雷达 100xxx / 卫星 200xxx 都是 6 位，按位数会把雷达误判成卫星）。
        self._satellite_keys = set(satellite_keys or [])

        # 启动时构建 O(1) 查找表
        self._agent_idx = {sid: i for i, sid in enumerate(agent_keys)}
        self._target_idx = {tid: i for i, tid in enumerate(target_keys)}

    # ============================================================
    # 观测 → 动作掩码
    # ============================================================

    def build_action_mask(self, agent_obs: AgentObservation) -> np.ndarray:
        """根据观测中的可探测关系，生成动作掩码矩阵。"""
        mask = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        mask[:, 0] = 1.0  # 所有智能体默认可执行待机

        for sid, target_dict in agent_obs.dict_detection_result.items():
            agent_idx = self._agent_idx.get(sid)
            if agent_idx is None:
                continue
            for tid, res in target_dict.items():
                target_idx = self._target_idx.get(tid)
                if target_idx is None:
                    continue
                if res.detectable_flag:
                    mask[agent_idx, target_idx + 1] = 1.0

        return mask

    # ============================================================
    # RL 动作 → 引擎指令
    # ============================================================

    def to_engine_commands(self, actions_onehot, current_time: float) -> List[AgentActionCommand]:
        """将 RL 输出的多标签 one-hot 转为引擎可执行的 AgentActionCommand。

        actions_onehot: (n_agents, n_actions) 多标签 one-hot。
          每个 agent 的置 1 位（bit 1..21 = 目标）各发一条锁定指令：
          - LD（多选，容量 20）：可发 0..20 条；
          - WX（单选，容量 1）：  发 0 或 1 条；
          无置 1 位（含待机 bit 0）不发送，仿真对缺失 agent 默认空 lst_track_no。
        """
        t_int = int(current_time)
        commands = []

        actions_onehot = np.asarray(actions_onehot)
        n = min(actions_onehot.shape[0], self.n_agents)
        for i in range(n):
            agent_id = self.agent_keys[i] if i < len(self.agent_keys) else f"AGENT_{i:03d}"
            row = actions_onehot[i]
            for b in np.nonzero(row)[0]:
                b = int(b)
                if 1 <= b <= len(self.target_keys):
                    target_id = self.target_keys[b - 1]
                    commands.append(AgentActionCommand(
                        time=t_int, str_equip_id=agent_id, str_target_id=target_id,
                    ))

        # ---- 诊断打印（排查「卫星多指向」）：按显式卫星 key 集合判断，不按 id 位数 ----
        # from collections import Counter
        # _cnt = Counter(c.str_equip_id for c in commands)
        # _sat_multi = {k: v for k, v in _cnt.items()
        #               if v > 1 and k in self._satellite_keys}
        # if _sat_multi:
        #     print(f"[ActionMapper DEBUG] ⚠️ 卫星单步发多条指令: {_sat_multi}")
        # elif not getattr(self, "_diag_logged", False):
        #     self._diag_logged = True
        #     print(f"[ActionMapper DEBUG] 诊断已生效，本步各装备指令数: {dict(_cnt)}")

        return commands
