# algorithms/drqn/agent.py
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from use_cases.config.config_types import AlgorithmConfig

# 导入 qmix（EPyMARL） 网络架构，保证与强化学习维度对齐
from services.algorithms.qmix.nn import DRQN, QMIXNET
from utils.weight_naming import prefix_from_conf, weight_file_name

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class ILAgents:
    """模仿学习智能体。

    可通过两种方式构造:
      - ILAgents(conf)  —— 兼容旧版扁平 RuntimeConfig
      - ILAgents(conf, algo_config=ac) —— 新版聚焦 AlgorithmConfig
    """

    def __init__(self, conf, algo_config: Optional[AlgorithmConfig] = None):
        self.conf = conf
        self._ac = algo_config or AlgorithmConfig.from_config(conf)
        self.device = self._ac.device
        # 权重文件名前缀（实体数 + 分组数），与 qmix/policy.py 保存侧保持一致
        self.weight_prefix = prefix_from_conf(conf)
        # 保存目录：默认取 conf.model_dir，Runner 会把自己的 self.model_dir 注入进来覆盖
        self.model_dir = getattr(conf, 'model_dir', None)

        # 对齐 QMIX 的 input_shape 计算逻辑
        obs_dim = self._ac.obs_shape[0] if isinstance(self._ac.obs_shape, tuple) else self._ac.obs_shape
        input_shape = obs_dim
        if self._ac.last_action:
            input_shape += self._ac.n_actions
        if self._ac.reuse_network:
            input_shape += self._ac.n_agents

        # 实例化 QMIX 的标准网络
        self.drqn_net = DRQN(input_shape, conf).to(self.device)
        self.mixer_net = QMIXNET(conf).to(self.device)

        self.optimizer = optim.Adam(
            self.drqn_net.parameters(),
            lr=self._ac.learning_rate,
            weight_decay=1e-5,
        )

        # ---- 类别权重：缓解「待机」压倒性多数的样本不平衡 ----
        # 动作空间 = [0]待机 + [1..n_targets]锁定第 i-1 个目标。
        # 专家数据里绝大多数时刻都是待机，直接算 CE 会让网络退化成「永远待机」，
        # 故放大全部正样本（1: 之后所有目标动作），而非只放大某一个目标。
        positive_weight = getattr(conf, 'il_positive_class_weight', 20.0)
        weights = torch.ones(self._ac.n_actions)
        weights[1:] = positive_weight
        self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))
        print(f"[ILAgents] 类别权重: 待机=1.0, 目标动作×{self._ac.n_actions - 1}"
              f"={positive_weight}")


    def _forward_pass(self, batch):
        """"""
        obs = batch['obs'].to(self.device)                        # (B, T, N, obs_dim)
        expert_actions = batch['actions'].to(self.device, dtype=torch.long)
        last_actions = batch['last_actions'].to(self.device)
        avail_actions = batch['avail_actions'].to(self.device)   # 引入动作掩码

        B, T, N = obs.shape[0], obs.shape[1], self._ac.n_agents
        n_actions = self._ac.n_actions

        # 拼装全序列输入一次完成（消除 T 次 cat/reshape 小 kernel）。
        # 与时间步无关的 agent_id one-hot 也在此一次性 expand，避免循环内重建。
        inputs_list = [obs]
        if self._ac.last_action:
            inputs_list.append(last_actions)
        if self._ac.reuse_network:
            eye = torch.eye(N, device=self.device).view(1, 1, N, N).expand(B, T, -1, -1)
            inputs_list.append(eye)
        inputs = torch.cat(inputs_list, dim=-1)                   # (B, T, N, D)

        hidden_state = torch.zeros(B * N, self._ac.drqn_hidden_dim, device=self.device)

        # 循环内只保留 RNN 前向 + 逐步掩码（GRU 的时序依赖无法并行）。
        # 掩码用 masked_fill 替代布尔索引就地写（后者在昇腾上无融合算子、常触发同步），
        # 且逐时间步对 avail_actions[:, t] 施加，与原实现形状语义完全一致。
        q_list = []
        for t in range(T):
            q_t, hidden_state = self.drqn_net(inputs[:, t].reshape(B * N, -1), hidden_state)
            q_t = q_t.view(B, N, n_actions)
            q_t = q_t.masked_fill(avail_actions[:, t] == 0.0, -1e9)
            q_list.append(q_t)
        q_values = torch.stack(q_list, dim=1)                     # (B, T, N, n_actions)

        # 损失 + 准确率：整段序列一次算完（替代 T 次 CE/argmax/sum 小 kernel）。
        # 与旧实现 total_loss/seq_len 数学等价（每个 criterion 本就是对该步的均值）。
        flat_q = q_values.reshape(-1, n_actions)
        flat_a = expert_actions.reshape(-1)
        loss = self.criterion(flat_q, flat_a)

        preds = flat_q.argmax(dim=1)
        active = (flat_a > 0)
        corr_global = (preds == flat_a).sum().item()
        tot_global = flat_a.numel()
        corr_active = ((preds == flat_a) & active).sum().item()
        tot_active = active.sum().item()

        metrics = {
            'acc_global': corr_global / tot_global if tot_global > 0 else 0.0,
            'acc_active': corr_active / tot_active if tot_active > 0 else 0.0,
        }
        return loss, metrics

    def learn(self, batch):
        """训练模式"""
        self.drqn_net.train()
        loss, metrics = self._forward_pass(batch)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.drqn_net.parameters(), max_norm=10)
        self.optimizer.step()
        
        metrics['loss'] = loss.item()
        return metrics

    @torch.no_grad()
    def evaluate(self, batch):
        """验证模式"""
        self.drqn_net.eval()
        loss, metrics = self._forward_pass(batch)
        metrics['loss'] = loss.item()
        return metrics

    def save_model(self, epoch=None):
        model_dir = self.model_dir
        if not model_dir:
            return

        os.makedirs(model_dir, exist_ok=True)
        # 文件名与 qmix/policy.py 保持一致（实体数 + 分组数前缀）
        drqn_path = os.path.join(model_dir, weight_file_name(self.weight_prefix, 'drqn'))
        mixer_path = os.path.join(model_dir, weight_file_name(self.weight_prefix, 'qmix'))

        torch.save(self.drqn_net.state_dict(), drqn_path)
        # 顺手把随机初始化的 Mixer 权重也存下来，防止 RL 加载时报错
        torch.save(self.mixer_net.state_dict(), mixer_path)

        print(f"[ILAgents] 已覆盖保存模仿学习预训练权重到 {model_dir} (Epoch: {epoch})")

    def load_model(self, load_dir):
        """为离线推理加载模型权重"""
        drqn_path = os.path.join(load_dir, weight_file_name(self.weight_prefix, 'drqn'))
        mixer_path = os.path.join(load_dir, weight_file_name(self.weight_prefix, 'qmix'))

        if os.path.exists(drqn_path):
            self.drqn_net.load_state_dict(torch.load(drqn_path, map_location=self.device))
            print(f"[ILAgents] 成功加载 DRQN 权重: {drqn_path}")
        else:
            raise FileNotFoundError(f"[ILAgents] 找不到 DRQN 权重: {drqn_path}")

        # 顺便加载 Mixer，虽然推理时不用，但为了防止网络结构报错
        if os.path.exists(mixer_path):
            self.mixer_net.load_state_dict(torch.load(mixer_path, map_location=self.device))