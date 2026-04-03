import numpy as np
import torch
from .policy import QMIX

class Agents:
    """等价于 EPyMARL 中的 BasicMAC (src/controllers/basic_mac.py)"""
    def __init__(self, conf):
        self.conf = conf
        self.device = conf.device
        self.n_actions = conf.n_actions
        self.n_agents = conf.n_agents
        
        self.policy = QMIX(conf)

    def perform_inference(self, obs, last_actions, avail_actions, epsilon=0.0):
        """Rollout 时调用的前向推演，带 Epsilon-Greedy 和 Mask"""
        # 构建 PyMARL 标准输入特征: [obs, last_action, agent_id_onehot]
        inputs = []
        for i in range(self.n_agents):
            agent_feat = [obs[i].copy()]
            if self.conf.last_action:
                agent_feat.append(last_actions[i].copy())
            if getattr(self.conf, 'reuse_network', True):
                agent_id = np.zeros(self.n_agents)
                agent_id[i] = 1.0
                agent_feat.append(agent_id)
            inputs.append(np.concatenate(agent_feat))
            
        inputs_tensor = torch.tensor(np.array(inputs), dtype=torch.float32).to(self.device) #补丁 1：去掉末尾的 .unsqueeze(0)，保持 (n_agents, input_dim) 的 2D 形状
        
        # 提取当前所有 Agent 的 Hidden State
        hidden_state = self.policy.eval_hidden
        
        # DRQN 前向传播
        q_values, self.policy.eval_hidden = self.policy.eval_drqn_net(inputs_tensor, hidden_state)
        
        # EPyMARL 铁律 1：掩码
        avail_tensor = torch.tensor(avail_actions, dtype=torch.float32).to(self.device)
        q_values[avail_tensor == 0.0] = -9999999.0
        
        # EPyMARL 铁律 2：安全动作选择
        actions = []
        actions_onehot = []
        for i in range(self.n_agents):
            if np.random.uniform() < epsilon:
                # 只在可用动作里随机
                avail_idx = np.nonzero(avail_actions[i])[0]
                action = np.random.choice(avail_idx) if len(avail_idx) > 0 else 0
            else:
                action = torch.argmax(q_values[i]).item()
                
            actions.append(action)
            
            onehot = np.zeros(self.n_actions)
            onehot[action] = 1.0
            actions_onehot.append(onehot)
            
        return actions, q_values.detach().cpu().numpy(), actions_onehot

    def train_(self, batch, train_step, epsilon=None):
        return self.policy.learn(batch, train_step, epsilon)