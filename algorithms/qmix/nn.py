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


class QMIXNET(nn.Module):
    """等价于 EPyMARL 中的 QMixer (src/modules/mixers/qmix.py)"""
    def __init__(self, conf):
        super(QMIXNET, self).__init__()
        self.conf = conf
        self.state_dim = self.conf.state_shape[0] if isinstance(self.conf.state_shape, tuple) else self.conf.state_shape
        self.embed_dim = getattr(self.conf, 'qmix_hidden_dim', 32)
        self.hypernet_embed = getattr(self.conf, 'hyper_hidden_dim', 64)

        self.state_norm = nn.LayerNorm(self.state_dim)

        # 1. 生成 W1 的超级网络 (输出维度: n_agents * embed_dim)
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.state_dim, self.hypernet_embed),
            nn.ReLU(),
            nn.Linear(self.hypernet_embed, self.embed_dim * self.conf.n_agents)
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
        
        agent_qs = agent_qs.view(-1, 1, self.conf.n_agents)

        # W1 和 W2 必须取绝对值，保证单调性 (∂Q_tot / ∂Q_a >= 0)
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.conf.n_agents, self.embed_dim)
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