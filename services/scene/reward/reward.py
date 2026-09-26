"""
奖励计算 —— 多标签 / 容量 20 语义（替换 sim.generate_reward）。

单步奖励 = R_ld + R_wx + R_switch，三条流物理分离、各写各的：

- R_ld      雷达流（容量 20 多标签），逐目标结算。
            把「稀缺资源」从「槽位」换成「好几何」：R_edge 给边缘锁定定价，
            替代旧 R_danger / R_relay，恢复「舒适→边缘→换手」的动态并消刷分。
- R_wx      卫星流（容量 1 单选），逐卫星结算（补盲 / 预警 / 机会成本）。
- R_switch  槽位级切换惩罚，逐智能体比较多标签 one-hot 的相邻两步变化。

对外只暴露 RewardCalculator：构造时绑定 agent_keys / target_keys，
之后每步调用 compute_reward(raw_obs, actions_onehot, prev_onehot) 即可。
系数为类属性（标定点），可在实例或子类上覆盖用于标定。

不改 sim 引擎；边缘判定（distance > 0.85 * range_max）在 env 侧用 pymap3d 复算。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pymap3d as pm

from services.scene.state.satellite_broadcast import compute_boundary_gap
from services.scene.scene_constants import build_agent_slot_map


# ============================================================
# 几何 / 观测工具（模块级纯函数，无状态，与奖励策略解耦）
# ============================================================

def _split_agents(agent_keys: List[str], equip) -> Tuple[List[str], List[str]]:
    """按装备类型把 agent_keys 拆成雷达 / 卫星两个 ID 列表（type: 1-雷达 2-卫星）。"""
    radar_ids: List[str] = []
    sat_ids: List[str] = []
    for aid in agent_keys:
        st = equip.get(aid)
        if st is None:
            continue
        if st.type == 1:
            radar_ids.append(aid)
        elif st.type == 2:
            sat_ids.append(aid)
    return radar_ids, sat_ids


def _is_edge(equip_state, system_track, edge_ratio: float) -> bool:
    """目标到该装备的距离是否超过 edge_ratio 倍射程（边缘判定）。"""
    _a, _e, dist_m = pm.geodetic2aer(
        system_track.latitude, system_track.longitude, system_track.altitude,
        equip_state.latitude, equip_state.longitude, equip_state.altitude,
    )
    return (dist_m / 1000.0) > (equip_state.range_max * edge_ratio)


def _compute_radar_edge_map(raw_obs, radar_ids: List[str], edge_ratio: float) -> Dict[str, Dict[str, bool]]:
    """雷达边缘判定矩阵 {target_id: {radar_id: is_edge}}，每步只算一次。

    只收 radar_ids 中「可见」（detectable_flag=True）的目标，is_edge 为该
    (雷达, 目标) 对的边缘判定。R_ld / R_wx 共享这一份，消除对同一几何关系
    （geodetic2aer）的重复计算。与原 _r_ld 的 visible / _r_wx 的 radar_edge 语义
    完全一致：trk 缺失视为非边缘，其余交给 _is_edge 判 0.85 倍射程。
    """
    detection = raw_obs.dict_detection_result
    equip = raw_obs.dict_equip_state
    track = raw_obs.dict_system_track
    edge_map: Dict[str, Dict[str, bool]] = {}
    for r in radar_ids:
        st = equip[r]
        for t, res in detection.get(r, {}).items():
            if not res.detectable_flag:
                continue
            trk = track.get(t)
            is_edge = (trk is not None) and _is_edge(st, trk, edge_ratio)
            edge_map.setdefault(t, {})[r] = is_edge
    return edge_map


def _nearest_radar_gap(raw_obs, radar_ids: List[str], target_id: str) -> float:
    """盲区目标到「最近雷达」的归一化边界距离（min 所有雷达的 gap，clamp [0,2]）。"""
    if not radar_ids:
        return 2.0
    gaps = [compute_boundary_gap(raw_obs, r, target_id) for r in radar_ids]
    return min(gaps)


class _RewardStream:
    """奖励分项累计器：逐项记录 value/count，最后返回 (total, breakdown)。

    让 R_ld / R_wx 各用一条 _RewardStream，逐条规则 add 一次，
    避免在奖励逻辑里散落 val/cnt 字典，便于定位某个分项由哪条规则产生。
    """

    def __init__(self, prefix: str):
        self._prefix = prefix
        self._vals = {}
        self._cnts = {}

    def add(self, name: str, coefficient: float, times: int = 1) -> None:
        """记一笔：value += coefficient * times，count += times。"""
        self._vals[name] = self._vals.get(name, 0.0) + coefficient * times
        self._cnts[name] = self._cnts.get(name, 0) + times

    def result(self):
        """返回 (total, {f"{prefix}_{name}": {"value": float, "count": int}})。"""
        total = sum(self._vals.values())
        breakdown = {
            f"{self._prefix}_{k}": {"value": v, "count": self._cnts[k]}
            for k, v in self._vals.items()
        }
        return total, breakdown


class RewardCalculator:
    """多标签 / 容量 20 语义的奖励计算器。

    除 agent_keys / target_keys 外无内部状态，每次调用结果只取决于入参。
    雷达 / 卫星按 dict_equip_state[].type（1-雷达 2-卫星）在每步动态拆分，
    不依赖 config 字段顺序，也不随 episode 缓存，避免跨局状态残留。
    """

    # ---- 系数（v4：对齐四指标加权 10/10/60/20；耗能非主指标 → 超重中性、无冗余罚）----
    K = 10                                  # 满分覆盖重数（=评价指标 10 重满分；兼作 switch 纠错豁免阈值）
    MULT_W = 1.0                            # R_mult      覆盖重数每重奖励（60% 主导，线性到 MULT_CAP 封顶）
    MULT_CAP = 10                           # R_mult 封顶重数；None 或 ≤0 = 不封顶（「能看尽看」去封顶开关）
    MISS_PENALTY = 20.0                     # R_miss      可见却 0 锁（覆盖率 20%）
    VALID_PENALTY = 5.0                     # R_valid     锁不可见
    EDGE_PENALTY = 2.0                      # R_edge      每个边缘锁定（交给卫星）
    SWITCH_PENALTY = 8.0                    # R_switch    逐装备掉锁（中断 10%，v3=5 → 8 加重）
    GAP_PENALTY = 15.0                      # R_gap       目标覆盖 >0→0 的硬中断（最重，护栏②门控）
    WX_BLIND_BASE = 4.0                     # R_wx_blind  补盲底分
    WX_GAP_WEIGHT = 2.0                     # R_wx_blind  gap 衰减权重
    WX_HANDOFF_REWARD = 3.0                 # R_wx_handoff 预警
    WX_VALID_PENALTY = 5.0                  # R_wx_valid  瞎指
    WX_IDLE_PENALTY = 1.0                   # R_wx_idle   机会成本
    WX_SEE_ALL_REWARD = 1.0                 # R_wx_see_all 能看尽看：舒适目标也跟踪（去耗能约束，< handoff 3 保持盲区/边缘优先）
    EDGE_RATIO = 0.85                       # 边缘判定阈值（distance > ratio * range_max）

    def __init__(self, agent_keys: List[str], target_keys: List[str],
                 wx_enabled: bool = True, satellite_keys: Optional[List[str]] = None,
                 mult_cap: Optional[int] = None, wx_see_all: bool = False):
        self.agent_keys = list(agent_keys)
        self.target_keys = list(target_keys)
        self.wx_enabled = wx_enabled  # phase1（纯 LD 标定）置 False，屏蔽 R_wx
        # R_mult 封顶重数：None 沿用类属性 MULT_CAP；显式传 0/负数 = 去封顶（能看尽看）
        self.mult_cap = self.MULT_CAP if mult_cap is None else mult_cap
        # WX 能看尽看：True = 卫星空闲时也跟踪舒适/冗余目标（去耗能约束），False = 现状只补盲/预警
        self.wx_see_all = wx_see_all
        # 可变实体数泛化：动作张量按固定槽位排布，读 actions 用「实体 ID → 槽位」而非 dense 下标
        # （满载两者一致；低实体数时卫星 dense 下标 != 槽位，见 build_agent_slot_map）。
        self._agent_slot = build_agent_slot_map(self.agent_keys, satellite_keys)

    # ============================================================
    # 入口
    # ============================================================

    def compute_reward(self, raw_obs, actions_onehot, prev_onehot, done: bool = False) -> float:
        """单步奖励 = R_ld + R_wx + R_switch（终态 done 跳过 R_switch）。

        Args:
            raw_obs:        AgentObservation（含 dict_detection_result / dict_equip_state
                            / dict_system_track），为 step_forward 后的当前帧观测。
            actions_onehot: (n_agents, n_actions) 多标签 one-hot，当前步动作。
            prev_onehot:    (n_agents, n_actions) 或 None，上一步动作（None = 首步无切换）。
            done:           本步是否为终态（terminated/truncated）。终态覆盖归零是
                            局结束而非智能体选择，跳过中断/掉锁结算（护栏①）。
        """
        total, _ = self.compute_reward_detailed(raw_obs, actions_onehot, prev_onehot, done=done)
        return total

    def compute_reward_detailed(self, raw_obs, actions_onehot, prev_onehot, done: bool = False):
        """单步奖励 + 分项明细，返回 (total, breakdown)。

        breakdown: {指标名: {"value": float, "count": int}}，指标名：
            ld_miss / ld_mult / ld_valid / ld_edge
            wx_blind / wx_handoff / wx_valid / wx_idle（wx_enabled 时）
            switch_penalty / switch_gap（非终态时）
        value 为该项净贡献（count×系数），count 为触发次数。
        """
        radar_ids, sat_ids = _split_agents(self.agent_keys, raw_obs.dict_equip_state)

        # 边缘判定矩阵每步只算一次，R_ld / R_wx 共享（消除重复 geodetic2aer）
        edge_map = _compute_radar_edge_map(raw_obs, radar_ids, self.EDGE_RATIO)

        total = 0.0
        breakdown = {}
        streams = [self._r_ld(raw_obs, actions_onehot, radar_ids, edge_map)]
        if self.wx_enabled:
            streams.append(self._r_wx(raw_obs, actions_onehot, radar_ids, sat_ids, edge_map))
        # 护栏①：终态不结算中断/掉锁。否则最后一步所有目标覆盖归零，会吃一记
        # 巨额 -GAP×目标数 的过失惩罚（归零是局结束造成的，非智能体选择）。
        if not done:
            streams.append(self._r_switch(raw_obs, actions_onehot, prev_onehot, edge_map))
        for _val, _bd in streams:
            total += _val
            breakdown.update(_bd)
        return float(total), breakdown

    # ============================================================
    # R_ld 雷达流（容量 20 多标签）
    # ============================================================

    def _r_ld(self, raw_obs, actions_onehot, radar_ids: List[str], edge_map: Dict[str, Dict[str, bool]]):
        radar_set = set(radar_ids)

        # 1) 雷达可见矩阵 + 边缘判定（由 compute_reward_detailed 一次性算好传入）
        visible = edge_map

        # 2) 锁定集合 {target_id: [radar_id, ...]} + R_valid
        stream = _RewardStream("ld")
        locks = {}
        actions = np.asarray(actions_onehot)
        n = min(actions.shape[0], len(self.agent_keys))
        for i in range(n):
            aid = self.agent_keys[i]
            if aid not in radar_set:
                continue
            row = actions[self._agent_slot[aid]]
            for b in np.nonzero(row)[0]:
                b = int(b)
                if b < 1 or b > len(self.target_keys):
                    continue
                t = self.target_keys[b - 1]
                if t in visible and aid in visible[t]:
                    locks.setdefault(t, []).append(aid)
                else:
                    stream.add("valid", -self.VALID_PENALTY)     # R_valid 锁不可见

        # 3) 逐目标：R_miss / R_mult / R_edge
        for t, able in visible.items():
            locked = locks.get(t, [])
            n_lock = len(locked)
            if n_lock == 0:
                stream.add("miss", -self.MISS_PENALTY)           # R_miss
                continue
            # R_mult：覆盖重数线性到 MULT_CAP 封顶（60% 主导）。超封顶中性——耗能非主指标，
            # 评价指标 min(重数,10) 封顶，>10 不奖不罚，故不再设 R_redundant。
            # MULT_CAP ≤ 0 时去封顶：重数无上限线性奖励（「能看尽看」，不再压制冗余堆锁）。
            n_eff = n_lock if (self.mult_cap is None or self.mult_cap <= 0) else min(n_lock, self.mult_cap)
            stream.add("mult", self.MULT_W * n_eff)
            edge = sum(1 for r in locked if able.get(r, False))
            stream.add("edge", -self.EDGE_PENALTY, edge)         # R_edge

        return stream.result()

    # ============================================================
    # R_wx 卫星流（容量 1 单选）
    # ============================================================

    def _r_wx(self, raw_obs, actions_onehot, radar_ids: List[str], sat_ids: List[str],
              edge_map: Dict[str, Dict[str, bool]]):
        detection = raw_obs.dict_detection_result
        sat_set = set(sat_ids)

        # 雷达可见 / 雷达边缘（从一次性边缘矩阵派生，不再逐对重算 geodetic2aer）
        radar_visible = set(edge_map.keys())
        radar_edge = {t for t, m in edge_map.items() if any(m.values())}

        stream = _RewardStream("wx")
        actions = np.asarray(actions_onehot)
        n = min(actions.shape[0], len(self.agent_keys))
        for i in range(n):
            aid = self.agent_keys[i]
            if aid not in sat_set:
                continue

            # 单选：取第一个置位的目标 bit（WX 容量 1，理论上至多一个）
            row = actions[self._agent_slot[aid]]
            t = None
            for b in np.nonzero(row)[0]:
                b = int(b)
                if 1 <= b <= len(self.target_keys):
                    t = self.target_keys[b - 1]
                    break

            seen = detection.get(aid, {})
            useful = {x for x in seen
                      if seen[x].detectable_flag
                      and (x not in radar_visible or x in radar_edge)}

            if t is not None and (t not in seen or not seen[t].detectable_flag):
                stream.add("valid", -self.WX_VALID_PENALTY)     # R_wx_valid 瞎指
                continue
            if t not in useful:                                 # 空槽(待机) 或 锁了舒适/无用目标
                if self.wx_see_all and t is not None:
                    # 能看尽看：舒适目标（雷达已覆盖 / 可能已被别的卫星跟踪）也去跟踪，
                    # 充分利用卫星、不设耗能约束。给小额奖励替代 idle 惩罚，
                    # 仍低于 blind(+4~6) / handoff(+3)，盲区/边缘优先级不变。
                    stream.add("see_all", self.WX_SEE_ALL_REWARD)
                elif useful:
                    stream.add("idle", -self.WX_IDLE_PENALTY)   # R_wx_idle
                continue
            if t not in radar_visible:                          # 盲区目标
                gap = _nearest_radar_gap(raw_obs, radar_ids, t)
                stream.add("blind",
                           self.WX_BLIND_BASE + self.WX_GAP_WEIGHT * (1.0 - gap))
            else:                                               # 边缘目标
                stream.add("handoff", self.WX_HANDOFF_REWARD)   # R_wx_handoff

        return stream.result()

    # ============================================================
    # R_switch 槽位级切换惩罚
    # ============================================================

    def _r_switch(self, raw_obs, actions_onehot, prev_onehot, edge_map):
        stream = _RewardStream("switch")
        if prev_onehot is None:
            return stream.result()
        detection = raw_obs.dict_detection_result
        prev = np.asarray(prev_onehot)
        cur = np.asarray(actions_onehot)
        if prev.shape != cur.shape:
            return stream.result()

        # 上一步 / 当前步每个目标的锁定数（按目标统计，超冗余甩掉属纠错不罚）
        prev_count = prev.sum(axis=0)
        cur_count = cur.sum(axis=0)

        # R_gap：目标覆盖从 >0 归零（硬中断，最重）。护栏②：仅当目标仍可见
        # （= 仍在 edge_map 中，即仍被 ≥1 部雷达 detectable）才罚——轨迹窗结束 /
        # 飞出射程导致的归零是被迫的，不计。终态由护栏①（done 不进入本流）兜底。
        for b in range(cur_count.shape[0]):
            if b < 1 or b > len(self.target_keys):
                continue
            target_id = self.target_keys[b - 1]
            if prev_count[b] > 0 and cur_count[b] == 0 and target_id in edge_map:
                stream.add("gap", -self.GAP_PENALTY)

        n = min(prev.shape[0], len(self.agent_keys))
        for i in range(n):
            equip_id = self.agent_keys[i]
            slot = self._agent_slot[equip_id]
            dropped = (prev[slot] > 0) & (cur[slot] == 0)
            for b in np.nonzero(dropped)[0]:
                b = int(b)
                if b < 1 or b > len(self.target_keys):
                    continue
                target_id = self.target_keys[b - 1]
                res = detection.get(equip_id, {}).get(target_id)
                if res is None or not res.detectable_flag:
                    continue                                    # 看不见切不罚
                if prev_count[b] > self.K:
                    continue                                    # 超冗余纠错不罚
                stream.add("penalty", -self.SWITCH_PENALTY)

        return stream.result()
