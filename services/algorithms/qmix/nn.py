import torch.nn as nn
import torch
import torch.nn.functional as F

class DRQN(nn.Module):
    """等价于 EPyMARL 中的 RNNAgent (src/modules/agents/rnn_agent.py)"""
    def __init__(self, input_shape, conf):
        super(DRQN, self).__init__()
        self.conf = conf

        self.fc1 = nn.Linear(input_shape, conf.drqn_hidden_dim)
        self.layer_norm = nn.LayerNorm(conf.drqn_hidden_dim)

        self.rnn = nn.GRUCell(conf.drqn_hidden_dim, conf.drqn_hidden_dim)
        self.fc2 = nn.Linear(conf.drqn_hidden_dim, conf.n_actions)

    def forward(self, obs, hidden_state):
        obs = obs.reshape(-1, obs.shape[-1])

        x = self.fc1(obs)
        x = self.layer_norm(x)
        x = F.relu(x)

        h_in = hidden_state.reshape(-1, self.conf.drqn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class DualHeadDRQN(nn.Module):
    """统一骨架 DRQN —— 共享主干 + 双输出头。

    - trunk: fc1 → LayerNorm → ReLU → GRUCell（LD 与 WX 共享，只训一遍）
    - LD 头 fc_ld: Linear(H, n_targets)，输出逐目标独立 Q（raw，无激活）。
      语义为「锁定该目标的边际价值」，Q>0 才锁，top-20 多选（容量 20）。
    - WX 头 fc_wx: Linear(H, n_actions)，输出单选 logits（softmax 在推理/损失处应用）。

    forward 返回 (q_ld, q_wx, h)，其中 q_ld (B*N, n_targets)、q_wx (B*N, n_actions)。
    两个头对全体 agent 都计算，调用方按 agent 索引（0..n_ld-1 为 LD，其余为 WX）切片使用。
    """
    def __init__(self, input_shape, conf, n_ld_actions, n_wx_actions):
        super(DualHeadDRQN, self).__init__()
        self.conf = conf
        H = conf.drqn_hidden_dim

        # 共享主干
        self.fc1 = nn.Linear(input_shape, H)
        self.layer_norm = nn.LayerNorm(H)
        self.rnn = nn.GRUCell(H, H)

        # 双输出头
        self.fc_ld = nn.Linear(H, n_ld_actions)   # LD 多标签（逐目标独立 Q）
        self.fc_wx = nn.Linear(H, n_wx_actions)   # WX 单选 logits

    def forward(self, obs, hidden_state):
        obs = obs.reshape(-1, obs.shape[-1])

        x = F.relu(self.layer_norm(self.fc1(obs)))
        h_in = hidden_state.reshape(-1, self.conf.drqn_hidden_dim)
        h = self.rnn(x, h_in)

        q_ld = self.fc_ld(h)
        q_wx = self.fc_wx(h)
        return q_ld, q_wx, h


class QMIXNET(nn.Module):
    """等价于 EPyMARL 中的 QMixer (src/modules/mixers/qmix.py)

    n_agents 参数可覆盖 conf.n_agents，用于 WX 分支（25 卫星）独立混频，
    而 conf.n_agents 仍是统一骨架 225。
    """
    def __init__(self, conf, n_agents=None):
        super(QMIXNET, self).__init__()
        self.conf = conf
        self.n_agents = conf.n_agents if n_agents is None else n_agents
        self.state_dim = self.conf.state_shape[0] if isinstance(self.conf.state_shape, tuple) else self.conf.state_shape
        self.embed_dim = getattr(self.conf, 'qmix_hidden_dim', 32)
        self.hypernet_embed = getattr(self.conf, 'hyper_hidden_dim', 64)

        self.state_norm = nn.LayerNorm(self.state_dim)

        # 1. 生成 W1 的超级网络 (输出维度: n_agents * embed_dim)
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.state_dim, self.hypernet_embed),
            nn.ReLU(),
            nn.Linear(self.hypernet_embed, self.embed_dim * self.n_agents)
        )

        # 2. 生成 W2 的超级网络 (输出维度: embed_dim * 1)
        self.hyper_w_2 = nn.Sequential(
            nn.Linear(self.state_dim, self.hypernet_embed),
            nn.ReLU(),
            nn.Linear(self.hypernet_embed, self.embed_dim)
        )

        # 3. 生成 b1 的超级网络
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # 4. 生成 b2 的超级网络 (V(s))
        self.hyper_b_2 = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        """
        agent_qs: (batch_size, max_seq_length, n_agents)
        states: (batch_size, max_seq_length, state_dim)
        """
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        states = self.state_norm(states)

        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # W1 和 W2 必须取绝对值，保证单调性 (∂Q_tot / ∂Q_a >= 0)
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)

        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        w2 = torch.abs(self.hyper_w_2(states))
        b2 = self.hyper_b_2(states)
        w2 = w2.view(-1, self.embed_dim, 1)
        b2 = b2.view(-1, 1, 1)

        # Q_tot 输出
        q_tot = torch.bmm(hidden, w2) + b2
        q_tot = q_tot.view(bs, -1, 1)
        return q_tot