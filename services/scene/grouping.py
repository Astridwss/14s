"""雷达地理分组 —— K-Means 静态聚类 + 固定槽位泛化, 训练启动时调用一次。

可变实体数泛化：分组结果恒为 ``(G_max, K_max)`` 固定形状，与雷达数解耦。
  - K_max = group_size（硬上界，每组最多雷达数，由配置 ``group_size`` 决定，=20）
  - G_max = ceil(MAX_RADARS / K_max)（固定组数，=10）
  - 真实雷达聚成 ``n_groups_real = min(G_max, ceil(n_agents/K_max))`` 个组，
    每组贪心再均衡到 ≤ K_max，尾部补 -1 到 K_max，再补全 -1 空组到 G_max。
"""
from typing import Dict, List, Optional

import numpy as np

from services.scene.scene_constants import MAX_RADARS


class RadarGrouper:
    """基于雷达地理坐标 (lat, lon, alt) 的 K-Means 静态分组。

    用法:
        # 方式 1: 工厂方法（推荐，Runner 中使用）
        groups = RadarGrouper.try_build(group_size=20, radar_keys=keys,
                                        radar_info_dict=env.dict_radar_info)
        # groups 为 None 时退化为标准 QMIX

        # 方式 2: 手动实例化
        grouper = RadarGrouper(group_size=20)
        groups = grouper.cluster(radar_keys, battle_scene)
        # groups: List[List[int]], 恒 (G_max, K_max)，-1 表示空槽位
    """

    def __init__(self, group_size: int = 20, random_state: int = 42):
        if group_size < 1:
            raise ValueError(f"group_size 必须 >= 1, 收到 {group_size}")
        self.group_size = group_size   # K_max：每组雷达数硬上界
        self.random_state = random_state

    # ============================================================
    # 工厂方法 —— Runner 装配用
    # ============================================================

    @staticmethod
    def try_build(
        group_size: int,
        radar_keys: List[str],
        radar_info_dict: Optional[Dict] = None,
    ) -> Optional[List[List[int]]]:
        """从原始参数构建分组，不满足条件时返回 None。

        Args:
            group_size: 每组雷达数（硬上界 K_max；≤0 表示不启用 H-QMIX）
            radar_keys: 雷达 ID 列表
            radar_info_dict: {radar_id: SensorInfo} 字典

        Returns:
            分组列表（恒 (G_max, K_max) 固定形状），或 None（退化为标准 QMIX）
        """
        if group_size <= 0 or not radar_keys or not radar_info_dict:
            return None
        return RadarGrouper(group_size=group_size).cluster(
            radar_keys, radar_info_dict,
        )

    # ============================================================
    # 聚类
    # ============================================================

    def cluster(self, radar_keys: List[str], battle_scene) -> List[List[int]]:
        """执行聚类并返回固定形状 (G_max, K_max) 的分组索引。

        Args:
            radar_keys: 雷达实体 ID 列表 (有序, 索引 0..N-1)
            battle_scene: BattleScene 对象 (含 dict_radar_id_info) 或直接传入
                          Dict[str, SensorInfo]

        Returns:
            groups: 恒 G_max 个组, 每组恒 K_max 个槽位, 空槽位用 -1 填充
        """
        n_agents = len(radar_keys)

        # 兼容两种入参: BattleScene 或 Dict[str, SensorInfo]
        if hasattr(battle_scene, 'dict_radar_id_info'):
            radar_info_dict = battle_scene.dict_radar_id_info
        elif isinstance(battle_scene, dict):
            radar_info_dict = battle_scene
        else:
            raise TypeError(
                f"battle_scene 需为 BattleScene 或 Dict[str, SensorInfo], "
                f"收到 {type(battle_scene)}"
            )

        # ---- 固定槽位形状 ----
        K_max = self.group_size
        G_max = (MAX_RADARS + K_max - 1) // K_max      # ceil(200/20) = 10
        n_groups_real = min(G_max, (n_agents + K_max - 1) // K_max)

        # ---- 特征提取 ----
        coords = np.zeros((n_agents, 3), dtype=np.float64)
        for i, rid in enumerate(radar_keys):
            info = radar_info_dict[rid]
            coords[i] = [info.latitude, info.longitude, info.altitude]

        # 标准化
        mean = coords.mean(axis=0)
        std = coords.std(axis=0) + 1e-8
        coords = (coords - mean) / std

        # ---- K-Means 聚类（只负责地理亲密度，不保证组宽） ----
        from sklearn.cluster import KMeans
        kmeans = KMeans(
            n_clusters=n_groups_real, random_state=self.random_state, n_init=10,
        )
        labels = kmeans.fit_predict(coords)
        centroids = kmeans.cluster_centers_   # 标准化空间，冻结作距离参考

        groups = [[] for _ in range(n_groups_real)]
        for agent_idx, label in enumerate(labels):
            groups[label].append(agent_idx)

        # ---- 贪心再均衡：每组 ≤ K_max，溢出点流向「离它簇心最近且有空位」的组 ----
        # 簇心冻结（不随移动更新）→ 确定性；总容量 n_groups_real×K_max ≥ n_agents，
        # 每步严格减少总溢出量，必终止。
        def _sq_dist(a: int, c) -> float:
            return float(np.sum((coords[a] - c) ** 2))

        while True:
            over = next(
                (g for g in range(n_groups_real) if len(groups[g]) > K_max), None,
            )
            if over is None:
                break
            # 组 over 里离簇心最远的成员（边缘点）
            farthest = max(groups[over], key=lambda i: _sq_dist(i, centroids[over]))
            # 目标组：离该点簇心最近、且未满(<K_max)的组
            best = None
            best_d = float('inf')
            for h in range(n_groups_real):
                if h == over or len(groups[h]) >= K_max:
                    continue
                d = _sq_dist(farthest, centroids[h])
                if d < best_d:
                    best_d = d
                    best = h
            if best is None:   # 防御：理论不会发生
                break
            groups[over].remove(farthest)
            groups[best].append(farthest)

        # ---- 补齐：每组固定宽 K_max，尾部补空组到 G_max ----
        for g in groups:
            while len(g) < K_max:
                g.append(-1)
        while len(groups) < G_max:
            groups.append([-1] * K_max)

        n_pad = sum(1 for g in groups for i in g if i < 0)
        print(f"[RadarGrouper] {n_agents} 部雷达 → {n_groups_real} 个真实组, "
              f"固定槽位 {G_max}×{K_max} (含 {n_pad} padding)")

        return groups
