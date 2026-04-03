# runner/rollout.py
import numpy as np
import torch
from env.adapter import ScenarioAdapter

class RolloutWorker:
    """
    职责：执行单局环境步进 -> 收集并返回整局轨迹数据。
    """
    def __init__(self, conf, env, agents):
        self.conf = conf
        self.env = env
        self.adapter = ScenarioAdapter(self.conf)
        self.agents = agents

    def generate_train_episode(self, epsilon, episode_num=0, render_callback=None):
        """跑完完整的一局，专用于训练，返回(奖励, 步数, 训练批次数据)"""
        obs, info = self.env.reset()
        state = info['state']
        avail_actions = info['avail_actions']
        
        # RNN 隐状态与上一步动作初始化 (干净的 Numpy)
        last_actions = np.zeros((self.conf.n_agents, self.conf.n_actions), dtype=np.float32)
        
        # 清空记忆避免幻觉
        self.agents.policy.init_hidden(1)
        self.agents.policy.eval_hidden = self.agents.policy.eval_hidden.to(self.agents.device)
        
        terminated, truncated, ep_reward, step_count = False, False, 0.0, 0
        episode_data = self._init_episode_data()
        
        while not (terminated or truncated):
            # 网络前向推理
            actions, q_values, actions_onehot = self.agents.perform_inference(
                obs=obs, 
                last_actions=last_actions, 
                avail_actions=avail_actions,
                epsilon=epsilon
            )
            
            # 转为纯 Numpy
            if isinstance(actions, torch.Tensor):
                actions_np = actions.cpu().detach().numpy()
            else:
                actions_np = np.array(actions)
                
            if isinstance(actions_onehot, torch.Tensor):
                actions_onehot_np = actions_onehot.cpu().detach().numpy()
            else:
                actions_onehot_np = np.array(actions_onehot)

            # 将动作展平为纯 Python List，喂给底层引擎
            env_actions = actions_np.squeeze().tolist()
            if not isinstance(env_actions, list):
                env_actions = [env_actions]
                
            next_obs, reward, terminated, truncated, next_info = self.env.step(env_actions)
            next_state = next_info['state']
            next_avail_actions = next_info['avail_actions']

            # TV渲染回调
            if render_callback is not None:
                render_callback(next_info)

            # 经验池记录
            self._record_train_step(
                episode_data, obs, next_obs, state, next_state,
                actions_np, actions_onehot_np, reward, terminated, avail_actions, next_avail_actions
            )

            # 状态轮转
            obs = next_obs
            state = next_state
            avail_actions = next_avail_actions
            last_actions = actions_onehot_np  
            ep_reward += reward
            step_count += 1

        episode_batch = {k: np.array(v) for k, v in episode_data.items()}
        return ep_reward, step_count, episode_batch

    def generate_eval_episode(self, episode_num=0): 
        """跑完完整的一局，专用于评估推演，无条件返回(奖励, 步数, 详尽战术指标)"""
        obs, info = self.env.reset()
        state = info['state']
        avail_actions = info['avail_actions']

        last_actions = np.zeros((self.conf.n_agents, self.conf.n_actions), dtype=np.float32)
        self.agents.policy.init_hidden(1)
        self.agents.policy.eval_hidden = self.agents.policy.eval_hidden.to(self.agents.device)
        
        terminated, truncated, ep_reward, step_count = False, False, 0.0, 0
        eval_records = {} 
        
        while not truncated:
            actions, q_values, actions_onehot = self.agents.perform_inference(
                obs=obs, 
                last_actions=last_actions, 
                avail_actions=avail_actions,
                epsilon=0.0  # 推演模式贪心
            )

            if isinstance(actions, torch.Tensor):
                actions_np = actions.cpu().detach().numpy()
            else:
                actions_np = np.array(actions)
                
            if isinstance(actions_onehot, torch.Tensor):
                actions_onehot_np = actions_onehot.cpu().detach().numpy()
            else:
                actions_onehot_np = np.array(actions_onehot)

            env_actions = actions_np.squeeze().tolist()
            if not isinstance(env_actions, list):
                env_actions = [env_actions]
                
            next_obs, reward, terminated, truncated, next_info = self.env.step(env_actions)
            next_avail_actions = next_info['avail_actions']
            
            # ======================推理模式：推送的数据要符合平台格式。只保留实际发生锁定的原始指令对象============================
            action_time = int(next_info.get('action_time', 0))
            valid_cmds = [cmd for cmd in next_info.get('agent_actions_list', []) if cmd.str_target_id != ""]
            eval_records[action_time] = valid_cmds

            obs = next_obs
            avail_actions = next_avail_actions
            last_actions = actions_onehot_np
            ep_reward += reward
            step_count += 1

        return ep_reward, step_count, eval_records

    def _init_episode_data(self):
        return {
            'obs': [], 'next_obs': [],
            'state': [], 'next_state': [],
            'actions': [], 'actions_onehot': [],
            'rewards': [], 'terminated': [],
            'avail_actions': [], 'next_avail_actions': []
        }

    def _record_train_step(self, episode_data, obs, next_obs, state, next_state, 
                           actions_np, actions_onehot_np, reward, terminated, avail_actions, next_avail_actions):
        episode_data['obs'].append(obs)
        episode_data['next_obs'].append(next_obs)
        episode_data['state'].append(state)
        episode_data['next_state'].append(next_state)
        
        episode_data['actions'].append(np.array(actions_np, dtype=np.int64).reshape(-1, 1))
        episode_data['actions_onehot'].append(actions_onehot_np)
        
        episode_data['rewards'].append([reward])
        episode_data['terminated'].append([1.0 if terminated else 0.0])
        episode_data['avail_actions'].append(avail_actions)
        episode_data['next_avail_actions'].append(next_avail_actions)