# algorithms/drqn/agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F

# 导入 qmix（EPyMARL） 网络架构，保证与强化学习维度对齐
from algorithms.qmix.nn import DRQN, QMIXNET

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # 你可以把刚才的 weights 传给 alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class ILAgents:
    def __init__(self, conf):
        self.conf = conf
        self.device = conf.device

        # 对齐 QMIX 的 input_shape 计算逻辑
        obs_dim = self.conf.obs_shape[0] if isinstance(self.conf.obs_shape, tuple) else self.conf.obs_shape
        input_shape = obs_dim
        if self.conf.last_action:
            input_shape += self.conf.n_actions
        if getattr(self.conf, 'reuse_network', True):
            input_shape += self.conf.n_agents

        #  实例化 QMIX 的标准网络，存下来的权重与RL网络结构一致
        self.drqn_net = DRQN(input_shape, conf).to(self.device)
        self.mixer_net = QMIXNET(conf).to(self.device)

        self.optimizer = optim.Adam(
            self.drqn_net.parameters(), 
            lr=getattr(self.conf, 'learning_rate', 0.001),
            weight_decay=1e-5
        )
        
        # 惩罚网络输出全 0 的行为，缓解样本不平衡
        # 初始化所有类别的权重为 1.0
        weights = torch.ones(self.conf.n_actions)
        weights[1] = 20 #放大正样本
        self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))
        #self.criterion = FocalLoss(alpha=weights.to(self.device), gamma=2.0) #解决正负样本不平衡


    def _forward_pass(self, batch):
        """"""
        obs = batch['obs'].to(self.device)
        expert_actions = batch['actions'].to(self.device, dtype=torch.long)
        last_actions = batch['last_actions'].to(self.device)
        avail_actions = batch['avail_actions'].to(self.device) #  引入动作掩码

        batch_size = obs.shape[0]
        seq_len = obs.shape[1]
        n_agents = self.conf.n_agents
        
        hidden_state = torch.zeros(batch_size * n_agents, self.conf.drqn_hidden_dim).to(self.device)
        total_loss = 0.0
        
        stats = {'corr_global': 0, 'tot_global': 0, 'corr_active': 0, 'tot_active': 0}
        
        for t in range(seq_len):
            obs_t = obs[:, t]
            last_act_t = last_actions[:, t]
            avail_t = avail_actions[:, t]
            
            # 对齐 QMIX 的拼装逻辑
            inputs_list = [obs_t]
            if self.conf.last_action:
                inputs_list.append(last_act_t)
            if getattr(self.conf, 'reuse_network', True):
                agent_id_eye = torch.eye(n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
                inputs_list.append(agent_id_eye)
                
            inputs_t = torch.cat(inputs_list, dim=-1).reshape(batch_size * n_agents, -1)
            
            # 前向传播
            q_values, hidden_state = self.drqn_net(inputs_t, hidden_state)
            q_values = q_values.view(batch_size, n_agents, -1)
            
            #  掩码：在算交叉熵 Loss 之前，把不可用的动作置为 -9999999
            q_values[avail_t == 0.0] = -9999999.0
            
            q_values_flat = q_values.view(-1, self.conf.n_actions)
            actions_t = expert_actions[:, t, :].reshape(-1)
            
            total_loss += self.criterion(q_values_flat, actions_t)
            
            # --- 向量化计算准确率指标 ---
            preds = q_values_flat.argmax(dim=1)
            active_mask = (actions_t > 0)
            
            stats['corr_global'] += (preds == actions_t).sum().item()
            stats['tot_global'] += actions_t.numel()
            stats['corr_active'] += ((preds == actions_t) & active_mask).sum().item()
            stats['tot_active'] += active_mask.sum().item()

        #平均损失
        loss = total_loss / seq_len
        
        #全局准确率与有效追踪准确率（真正下达“追踪某个目标”指令，去掉大量0了）
        metrics = {
            'acc_global': stats['corr_global'] / stats['tot_global'] if stats['tot_global'] > 0 else 0.0,
            'acc_active': stats['corr_active'] / stats['tot_active'] if stats['tot_active'] > 0 else 0.0
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
        if not getattr(self.conf, 'model_dir', None):
            return
            
        os.makedirs(self.conf.model_dir, exist_ok=True)
        # 文件名与 qmix 保持一致
        drqn_path = os.path.join(self.conf.model_dir, 'final_drqn_net_params.pkl')
        mixer_path = os.path.join(self.conf.model_dir, 'final_qmix_net_params.pkl')
        
        torch.save(self.drqn_net.state_dict(), drqn_path)
        # 顺手把随机初始化的 Mixer 权重也存下来，防止 RL 加载时报错
        torch.save(self.mixer_net.state_dict(), mixer_path)
        
        print(f"[ILAgents] 已覆盖保存模仿学习预训练权重到 {self.conf.model_dir} (Epoch: {epoch})")
    
    def load_model(self, load_dir):
        """为离线推理加载模型权重"""
        drqn_path = os.path.join(load_dir, 'final_drqn_net_params.pkl')
        mixer_path = os.path.join(load_dir, 'final_qmix_net_params.pkl')
        
        if os.path.exists(drqn_path):
            self.drqn_net.load_state_dict(torch.load(drqn_path, map_location=self.device))
            print(f"[ILAgents] 成功加载 DRQN 权重: {drqn_path}")
        else:
            raise FileNotFoundError(f"[ILAgents] 找不到 DRQN 权重: {drqn_path}")
            
        # 顺便加载 Mixer，虽然推理时不用，但为了防止网络结构报错
        if os.path.exists(mixer_path):
            self.mixer_net.load_state_dict(torch.load(mixer_path, map_location=self.device))