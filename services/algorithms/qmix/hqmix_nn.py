"""H-QMIX 两层混频网络。

LowerMixer: 组内 K 个 agent Q 值 → Q_group (参数跨 G 组共享)
UpperMixer: G 个 group Q 值 → Q_tot (单一实例, 同标准 QMIX 结构)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LowerMixer(nn.Module):
    """组内混频 —— G 个组共享参数, 打包 batch 并行。

    agent_qs:     (B*T, G, K)   每组 K 个 agent 的 Q(s, a)
    group_states: (B*T, G, D_k)  每组局部状态
    → q_group:    (B*T, G, 1)   每组混合后的组 Q 值
    """

    def __init__(self, K: int, state_dim: int,
                 mixing_embed_dim: int = 32, hyper_embed_dim: int = 64):
        super().__init__()
        self.K = K
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim

        self.state_norm = nn.LayerNorm(state_dim)

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_embed_dim), nn.ReLU(),
            nn.Linear(hyper_embed_dim, K * mixing_embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_embed_dim), nn.ReLU(),
            nn.Linear(hyper_embed_dim, mixing_embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hyper_embed_dim), nn.ReLU(),
            nn.Linear(hyper_embed_dim, 1),
        )

    def forward(self, agent_qs, group_states):
        """
        agent_qs:     (B, G, K)  或 (B*T, G, K)
        group_states: (B, G, D)  或 (B*T, G, D)
        """
        *prefix, G, K = agent_qs.shape
        flat_b = _prod(prefix)  # B 或 B*T

        aq = agent_qs.view(flat_b, G, K)
        gs = group_states.view(flat_b, G, self.state_dim)
        gs = self.state_norm(gs)

        # W1: (F, G, K * E) → (F, G, K, E)
        w1 = torch.abs(self.hyper_w1(gs)).view(flat_b, G, K, self.mixing_embed_dim)
        b1 = self.hyper_b1(gs).unsqueeze(2)                    # (F, G, 1, E)

        hidden = F.elu((aq.unsqueeze(-2) @ w1).squeeze(-2) + b1.squeeze(2))
        # (F, G, E)

        w2 = torch.abs(self.hyper_w2(gs)).unsqueeze(-1)       # (F, G, E, 1)
        b2 = self.hyper_b2(gs)                                 # (F, G, 1)

        q_group = (hidden.unsqueeze(-2) @ w2).squeeze(-2) + b2  # (F, G, 1)

        return q_group.view(*prefix, G, 1)


class UpperMixer(nn.Module):
    """组间混频 —— 单一实例, 与标准 QMIX 混频器结构同构。

    group_qs:     (B*T, 1, G)    G 个组的 Q 值
    global_state: (B*T, D_u)     池化全局状态
    → q_tot:      (B*T, 1)
    """

    def __init__(self, G: int, state_dim: int,
                 mixing_embed_dim: int = 64, hyper_embed_dim: int = 128):
        super().__init__()
        self.G = G
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim

        self.state_norm = nn.LayerNorm(state_dim)

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_embed_dim), nn.ReLU(),
            nn.Linear(hyper_embed_dim, G * mixing_embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_embed_dim), nn.ReLU(),
            nn.Linear(hyper_embed_dim, mixing_embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hyper_embed_dim), nn.ReLU(),
            nn.Linear(hyper_embed_dim, 1),
        )

    def forward(self, group_qs, global_state):
        """
        group_qs:     (B, G)  或 (B*T, G)
        global_state: (B, D)  或 (B*T, D)
        """
        *prefix, G = group_qs.shape
        flat_b = _prod(prefix)

        gq = group_qs.view(flat_b, 1, G)
        gs = global_state.view(flat_b, self.state_dim)
        gs = self.state_norm(gs)

        w1 = torch.abs(self.hyper_w1(gs)).view(flat_b, G, self.mixing_embed_dim)
        b1 = self.hyper_b1(gs).unsqueeze(1)                  # (F, 1, E)

        hidden = F.elu(torch.bmm(gq, w1) + b1)               # (F, 1, E)

        w2 = torch.abs(self.hyper_w2(gs)).unsqueeze(-1)      # (F, E, 1)
        b2 = self.hyper_b2(gs)                                # (F, 1)

        q_tot = torch.bmm(hidden, w2).squeeze(-1) + b2       # (F, 1)
        return q_tot.view(*prefix, 1)


def _prod(xs):
    """安全 int 求积, 空列表返回 1。"""
    p = 1
    for x in xs:
        p *= x
    return p
