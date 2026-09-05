"""雷达地理分组 —— K-Means 静态聚类, 训练启动时调用一次。"""
from typing import Dict, List, Optional

import numpy as np


class RadarGrouper:
    """基于雷达地理坐标 (lat, lon, alt) 的 K-Means 静态分组。

    用法:
        # 方式 1: 工厂方法（推荐，Runner 中使用）
        groups = RadarGrouper.try_build(group_size=10, radar_keys=keys,
                                        radar_info_dict=env.dict_radar_info)
        # groups 为 None 时退化为标准 QMIX

        # 方式 2: 手动实例化
        grouper = RadarGrouper(group_size=10)
        groups = grouper.cluster(radar_keys, battle_scene)
        # groups: List[List[int]], G 个组, 每组统一长度 (不足位用 -1 padding)
    """

    def __init__(self, group_size: int = 10, random_state: int = 42):
        if group_size < 1:
            raise ValueError(f"group_size 必须 >= 1, 收到 {group_size}")
        self.group_size = group_size
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
            group_size: 每组雷达数 (≤0 表示不启用 H-QMIX)
            radar_keys: 雷达 ID 列表
            radar_info_dict: {radar_id: SensorInfo} 字典

        Returns:
            分组列表，或 None（退化为标准 QMIX）
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
        """执行聚类并返回分组索引。

        Args:
            radar_keys: 雷达实体 ID 列表 (有序, 索引 0..N-1)
            battle_scene: BattleScene 对象 (含 dict_radar_id_info) 或直接传入
                          Dict[str, SensorInfo]

        Returns:
            groups: G 个组, 每组为等长 agent 索引列表, 空位用 -1 填充
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

        # 特征提取
        coords = np.zeros((n_agents, 3), dtype=np.float64)
        for i, rid in enumerate(radar_keys):
            info = radar_info_dict[rid]
            coords[i] = [info.latitude, info.longitude, info.altitude]

        # 标准化
        mean = coords.mean(axis=0)
        std = coords.std(axis=0) + 1e-8
        coords = (coords - mean) / std

        # K-Means
        n_groups = max(1, n_agents // self.group_size)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_groups, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(coords)

        # 组织分组
        groups = [[] for _ in range(n_groups)]
        for agent_idx, label in enumerate(labels):
            groups[label].append(agent_idx)

        # 补齐
        max_size = max(len(g) for g in groups)
        for g in groups:
            while len(g) < max_size:
                g.append(-1)

        n_pad = sum(1 for g in groups for i in g if i < 0)
        print(f"[RadarGrouper] {n_agents} 部雷达 → {n_groups} 组, "
              f"每组 {max_size} agents (含 {n_pad} padding)")

        return groups
