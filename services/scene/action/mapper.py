"""
动作映射 —— RL 离散动作 ↔ 仿真引擎指令。

自包含积木：接收实体 ID 列表和维度，不依赖 Config。
"""
import numpy as np
from typing import List, Optional
from sim import AgentActionCommand, AgentObservation
from services.scene.scene_constants import MAX_AGENTS, MAX_ACTIONS, build_agent_slot_map


class ActionMapper:
    """RL 动作空间与引擎指令的双向映射。

    可变实体数泛化：动作掩码与动作回写统一走「按类型固定槽位」——
      雷达 → agent 槽位 [0, MAX_RADARS)
      卫星 → agent 槽位 [MAX_RADARS, MAX_AGENTS)
      目标 → action 槽位 [1, MAX_ACTIONS)
    dummy 实体占满剩余槽位，掩码时靠「不在 dict_detection_result」自动归零（仅待机），
    动作回写时靠反向映射查无实体直接跳过。
    """

    def __init__(self, agent_keys: List[str], target_keys: List[str],
                 n_agents: int, n_actions: int, satellite_keys: Optional[List[str]] = None):
        self.agent_keys = list(agent_keys)
        self.target_keys = list(target_keys)
        # n_agents / n_actions 保留用于兼容旧调用方与诊断；掩码/回写维度由 MAX 常量决定。
        self.n_agents = n_agents
        self.n_actions = n_actions
        # 显式卫星 key 集合：区分雷达/卫星槽位用，不靠 id 位数猜
        # （mock 场景里雷达 100xxx / 卫星 200xxx 都是 6 位，按位数会把雷达误判成卫星）。
        self._satellite_keys = set(satellite_keys or [])

        # ---- 固定槽位映射：实体 ID → MAX 槽位 ----
        # 卫星落入 [MAX_RADARS, MAX_AGENTS)，其余（雷达）落入 [0, MAX_RADARS)。
        # 未传 satellite_keys 时全部按雷达槽位映射，满载下与固定槽位一致（dense 顺序 == 槽位顺序）。
        self._agent_idx = build_agent_slot_map(agent_keys, satellite_keys)
        self._target_idx = {tid: i for i, tid in enumerate(target_keys)}

        # ---- 反向映射：MAX 槽位 → 实体 ID（动作回写用） ----
        self._slot_to_agent = {slot: sid for sid, slot in self._agent_idx.items()}
        self._slot_to_target = {i: tid for i, tid in enumerate(target_keys)}

    # ============================================================
    # 观测 → 动作掩码
    # ============================================================

    def build_action_mask(self, agent_obs: AgentObservation) -> np.ndarray:
        """根据观测中的可探测关系，生成动作掩码矩阵。

        形状恒为 (MAX_AGENTS, MAX_ACTIONS)。dummy agent 行与 dummy 目标列
        天然全 0（仅待机位 1），无需额外逻辑。
        """
        mask = np.zeros((MAX_AGENTS, MAX_ACTIONS), dtype=np.float32)
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

        actions_onehot: (MAX_AGENTS, MAX_ACTIONS) 多标签 one-hot，行/列均按 MAX 槽位。
          每个 agent 的置 1 位（bit 1..21 = 目标）各发一条锁定指令：
          - LD（多选，容量 20）：可发 0..20 条；
          - WX（单选，容量 1）：  发 0 或 1 条；
          待机位 0 不发送；dummy 槽位（反向映射查无实体）跳过，仿真对缺失 agent 默认空 lst_track_no。
        """
        t_int = int(current_time)
        commands = []

        actions_onehot = np.asarray(actions_onehot)
        n_rows = min(actions_onehot.shape[0], MAX_AGENTS)
        for slot in range(n_rows):
            agent_id = self._slot_to_agent.get(slot)
            if agent_id is None:
                continue  # dummy agent 槽位，跳过
            row = actions_onehot[slot]
            for b in np.nonzero(row)[0]:
                b = int(b)
                if b == 0:
                    continue  # 待机位，不产生命令
                target_id = self._slot_to_target.get(b - 1)
                if target_id is None:
                    continue  # dummy 目标槽位，跳过
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
