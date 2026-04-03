import os
import torch
import torch.nn as nn
from .nn import DRQN, QMIXNET

class QMIX:
    """等价于 EPyMARL 中的 QLearner (src/learners/q_learner.py)"""
    def __init__(self, conf):
        self.conf = conf
        self.device = self.conf.device
        self.n_actions = self.conf.n_actions
        self.n_agents = self.conf.n_agents
        
        # 计算 DRQN 输入维度
        obs_dim = self.conf.obs_shape[0] if isinstance(self.conf.obs_shape, tuple) else self.conf.obs_shape
        input_shape = obs_dim
        if self.conf.last_action:
            input_shape += self.n_actions
        if getattr(self.conf, 'reuse_network', True):
            input_shape += self.n_agents

        # 初始化网络
        self.eval_drqn_net = DRQN(input_shape, self.conf).to(self.device)
        self.target_drqn_net = DRQN(input_shape, self.conf).to(self.device)
        self.eval_qmix_net = QMIXNET(self.conf).to(self.device)
        self.target_qmix_net = QMIXNET(self.conf).to(self.device)

        self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_drqn_net.parameters())
        self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.conf.learning_rate)

        self.eval_hidden = None
        self.model_dir = getattr(self.conf, 'model_dir', None)

    def init_hidden(self, batch_size):
        # 初始化 MAC 的隐状态
        self.eval_hidden = torch.zeros((batch_size, self.n_agents, self.conf.drqn_hidden_dim)).to(self.device)

    def learn(self, batch, train_step, epsilon=None):
        """完美复刻 EPyMARL 的 q_learner.py TD-error 计算逻辑"""
        # 1. 提取所有轨迹数据并送入 GPU
        states = torch.tensor(batch['state'], dtype=torch.float32).to(self.device)
        next_states = torch.tensor(batch['next_state'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.long).to(self.device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32).to(self.device)
        terminated = torch.tensor(batch['terminated'], dtype=torch.float32).to(self.device)
        mask = 1.0 - torch.tensor(batch['padded'], dtype=torch.float32).to(self.device)
        avail_actions = torch.tensor(batch['avail_actions'], dtype=torch.float32).to(self.device)
        next_avail_actions = torch.tensor(batch['next_avail_actions'], dtype=torch.float32).to(self.device)
        
        batch_size = states.size(0)
        max_seq_length = states.size(1)

        # 2. 获取 Eval 和 Target 的 MAC 序列输出
        mac_out = []
        target_mac_out = []
        
        self.init_hidden(batch_size)
        target_hidden = torch.zeros((batch_size, self.n_agents, self.conf.drqn_hidden_dim)).to(self.device)

        for t in range(max_seq_length):
            # 拼装第 t 步输入
            inputs, next_inputs = self._build_inputs(batch, t, batch_size)
            
            out, self.eval_hidden = self.eval_drqn_net(inputs, self.eval_hidden)
            target_out, target_hidden = self.target_drqn_net(next_inputs, target_hidden)
            
            mac_out.append(out.view(batch_size, self.n_agents, self.n_actions))
            target_mac_out.append(target_out.view(batch_size, self.n_agents, self.n_actions))

        mac_out = torch.stack(mac_out, dim=1)  # (bs, seq, n_agents, n_actions)
        target_mac_out = torch.stack(target_mac_out, dim=1)

        # 3. 动作选择与掩码处理
        # 取出实际执行动作的 Q 值: Q(s, a)
        chosen_action_qvals = torch.gather(mac_out, dim=3, index=actions).squeeze(3)  # (bs, seq, n_agents)

        # Target 网络的 max Q 值处理（带 Mask掩码
        target_mac_out[next_avail_actions == 0.0] = -9999999.0
        
        # 标准 Q-Learning: max_a' Q_target(s', a')
        target_max_qvals = target_mac_out.max(dim=3)[0] # (bs, seq, n_agents)

        # 4. 送入 Mixer 计算总 Q 值
        chosen_action_qvals = self.eval_qmix_net(chosen_action_qvals, states) # (bs, seq, 1)
        target_max_qvals = self.target_qmix_net(target_max_qvals, next_states) # (bs, seq, 1)

        # 5. 计算 TD-Target
        targets = rewards + self.conf.gamma * (1 - terminated) * target_max_qvals

        # 6. 计算 TD-Error 并应用 Padding 掩码
        td_error = (chosen_action_qvals - targets.detach())
        mask_td_error = td_error * mask

        # 7. 计算 MSE 损失
        loss = (mask_td_error ** 2).sum() / mask.sum()

        # 8. 梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_parameters, self.conf.grad_norm_clip)
        self.optimizer.step()

        # 9. 更新 Target 网络
        if train_step > 0 and train_step % self.conf.update_target_params == 0:
            self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        return loss.item()

    def _build_inputs(self, batch, t, batch_size):
        """提取并拼装第 t 步及其下一步的 RNN 输入"""
        obs = torch.tensor(batch['obs'][:, t], dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(batch['next_obs'][:, t], dtype=torch.float32).to(self.device)
        
        inputs = [obs]
        next_inputs = [next_obs]
        
        if self.conf.last_action:
            if t == 0:
                inputs.append(torch.zeros((batch_size, self.n_agents, self.n_actions)).to(self.device))
            else:
                inputs.append(torch.tensor(batch['actions_onehot'][:, t - 1], dtype=torch.float32).to(self.device))
            next_inputs.append(torch.tensor(batch['actions_onehot'][:, t], dtype=torch.float32).to(self.device))

        if getattr(self.conf, 'reuse_network', True):
            agent_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
            inputs.append(agent_id)
            next_inputs.append(agent_id)

        # 扁平化以送入网络
        inputs = torch.cat([x.reshape(batch_size * self.n_agents, -1) for x in inputs], dim=1)
        next_inputs = torch.cat([x.reshape(batch_size * self.n_agents, -1) for x in next_inputs], dim=1)
        
        return inputs, next_inputs

    def save_model(self, episode_idx=None):
        if not self.model_dir: return
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.eval_drqn_net.state_dict(), os.path.join(self.model_dir, 'final_drqn_net_params.pkl'))
        torch.save(self.eval_qmix_net.state_dict(), os.path.join(self.model_dir, 'final_qmix_net_params.pkl'))

    def load_state(self, load_dir):
        drqn_path = os.path.join(load_dir, 'final_drqn_net_params.pkl')
        qmix_path = os.path.join(load_dir, 'final_qmix_net_params.pkl')
        if os.path.exists(drqn_path) and os.path.exists(qmix_path):
            self.eval_drqn_net.load_state_dict(torch.load(drqn_path, map_location=self.device, weights_only=True))
            self.eval_qmix_net.load_state_dict(torch.load(qmix_path, map_location=self.device, weights_only=True))
            self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        else:
            raise FileNotFoundError(f"加载失败：未在 {load_dir} 中找到权重")