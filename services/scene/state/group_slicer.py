"""分组状态切片 —— 与 ObservationBuilder 配对，从完整全局状态中按组提取局部状态。

常量来源: services.scene.scene_constants (与 ObservationBuilder 共享，避免魔数重复)。
"""

from typing import List

import torch

from services.scene.scene_constants import RADAR_STATE_FEATURES, TARGET_STATE_FEATURES

# 全局状态布局 (由 ObservationBuilder.build_global_state 定义):
#   [targets (n_targets × TARGET_STATE_FEATURES) | agents (n_agents × RADAR_STATE_FEATURES) | time (1)]


def group_local_state_dim(K: int, n_targets: int) -> int:
    """组局部状态维度: targets + K agents + time"""
    return n_targets * TARGET_STATE_FEATURES + K * RADAR_STATE_FEATURES + 1


def global_pooled_state_dim(G: int, n_targets: int) -> int:
    """池化全局状态维度: targets + G groups + time"""
    return n_targets * TARGET_STATE_FEATURES + G * RADAR_STATE_FEATURES + 1


def build_group_local_states(
    full_state: torch.Tensor,
    group_assignments: List[List[int]],
    n_targets: int,
) -> torch.Tensor:
    """从完整状态中为每组切片局部状态 s_k。

    Args:
        full_state: (..., state_dim)  完整全局状态
        group_assignments: G 个组，每组包含 agent 索引列表，-1 为 padding
        n_targets: 目标数量（决定 target 段宽度）

    Returns:
        group_states: (..., G, D_k)  D_k = n_targets*TARGET_STATE_FEATURES + K*RADAR_STATE_FEATURES + 1
    """
    *prefix, state_dim = full_state.shape
    G = len(group_assignments)
    K = max(len(g) for g in group_assignments)
    D_k = group_local_state_dim(K, n_targets)
    device = full_state.device

    flat_in = full_state.view(-1, state_dim)  # (F, state_dim)
    F = flat_in.shape[0]

    target_len = n_targets * TARGET_STATE_FEATURES
    agent_offset = target_len

    # target 部分（所有组共享）
    target_part = flat_in[:, :target_len]
    # time 部分
    time_part = flat_in[:, -1:]

    group_parts = []
    for indices in group_assignments:
        agent_parts = []
        for idx in indices:
            if idx >= 0:
                start = agent_offset + idx * RADAR_STATE_FEATURES
                agent_parts.append(flat_in[:, start:start + RADAR_STATE_FEATURES])
            else:
                agent_parts.append(torch.zeros(F, RADAR_STATE_FEATURES, device=device))
        agent_slice = torch.cat(agent_parts, dim=1)              # (F, K*RADAR_STATE_FEATURES)
        group_part = torch.cat([target_part, agent_slice, time_part], dim=1)  # (F, D_k)
        group_parts.append(group_part.unsqueeze(1))               # (F, 1, D_k)

    out = torch.cat(group_parts, dim=1)  # (F, G, D_k)
    return out.view(*prefix, G, D_k)


def build_global_pooled_state(
    full_state: torch.Tensor,
    group_assignments: List[List[int]],
    n_targets: int,
) -> torch.Tensor:
    """从完整状态构建池化全局状态 S: 每组 agent 状态取均值。

    Args:
        full_state: (..., state_dim)
        group_assignments: G 个组
        n_targets: 目标数量

    Returns:
        pooled: (..., D_u)  D_u = n_targets*TARGET_STATE_FEATURES + G*RADAR_STATE_FEATURES + 1
    """
    *prefix, state_dim = full_state.shape
    G = len(group_assignments)
    D_u = global_pooled_state_dim(G, n_targets)
    device = full_state.device

    flat_in = full_state.view(-1, state_dim)  # (F, state_dim)
    F = flat_in.shape[0]

    target_len = n_targets * TARGET_STATE_FEATURES
    agent_offset = target_len

    target_part = flat_in[:, :target_len]
    time_part = flat_in[:, -1:]

    pooled_parts = []
    for indices in group_assignments:
        group_agent_states = []
        for idx in indices:
            if idx >= 0:
                start = agent_offset + idx * RADAR_STATE_FEATURES
                group_agent_states.append(flat_in[:, start:start + RADAR_STATE_FEATURES])
        if group_agent_states:
            stacked = torch.stack(group_agent_states, dim=1)  # (F, K_active, 5)
            pooled = stacked.mean(dim=1)                       # (F, 5)
        else:
            pooled = torch.zeros(F, RADAR_STATE_FEATURES, device=device)
        pooled_parts.append(pooled)

    pooled_all = torch.cat(pooled_parts, dim=1)              # (F, G*5)
    out = torch.cat([target_part, pooled_all, time_part], dim=1)  # (F, D_u)
    return out.view(*prefix, D_u)
