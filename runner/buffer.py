# -*- coding: utf-8 -*-
import numpy as np
import threading
import torch

class EpisodeReplayBuffer:
    """
    QMIX 专属的轨迹经验池 (Off-policy Episode Replay Buffer)
    """
    def __init__(self, conf):
        self.conf = conf
        self.capacity = getattr(conf, 'buffer_size', 5000) 
        self.episode_limit = getattr(conf, 'max_episode_steps', 600)
        
        self.n_agents = conf.n_agents
        self.obs_shape = conf.obs_shape[0] if isinstance(conf.obs_shape, tuple) else conf.obs_shape
        self.state_shape = conf.state_shape[0] if isinstance(conf.state_shape, tuple) else conf.state_shape
        self.n_actions = conf.n_actions
        
        self.current_idx = 0
        self.current_size = 0
        self.lock = threading.Lock()
        
        self.buffers = self._create_empty_buffer(self.capacity)
        
    def _create_empty_buffer(self, size):
        return {
            'obs': np.zeros([size, self.episode_limit, self.n_agents, self.obs_shape], dtype=np.float32),
            'next_obs': np.zeros([size, self.episode_limit, self.n_agents, self.obs_shape], dtype=np.float32), #self.obs_shape是每个组拿到的局部观测维度（10 部雷达 × 25 维）
            'state': np.zeros([size, self.episode_limit, self.state_shape], dtype=np.float32),
            'next_state': np.zeros([size, self.episode_limit, self.state_shape], dtype=np.float32),
            'actions': np.zeros([size, self.episode_limit, self.n_agents, 1], dtype=np.int64),
            'actions_onehot': np.zeros([size, self.episode_limit, self.n_agents, self.n_actions], dtype=np.float32),
            'rewards': np.zeros([size, self.episode_limit, 1], dtype=np.float32),
            'avail_actions': np.zeros([size, self.episode_limit, self.n_agents, self.n_actions], dtype=np.float32),
            'next_avail_actions': np.zeros([size, self.episode_limit, self.n_agents, self.n_actions], dtype=np.float32),
            'terminated': np.zeros([size, self.episode_limit, 1], dtype=np.float32),
            'padded': np.ones([size, self.episode_limit, 1], dtype=np.float32) 
        }
        
    def store_episode(self, episode_batch):
        with self.lock:
            step_num = len(episode_batch['obs'])
            if step_num > self.episode_limit:
                step_num = self.episode_limit
                
            idx = self.current_idx
            
            for key in self.buffers.keys():
                self.buffers[key][idx] = 0.0
                
            for key in self.buffers.keys():
                if key in episode_batch and key != 'padded':
                    self.buffers[key][idx, :step_num] = episode_batch[key][:step_num]
            
            self.buffers['padded'][idx, :step_num] = 0.0
            if step_num < self.episode_limit:
                self.buffers['padded'][idx, step_num:] = 1.0
                
            self.current_idx = (self.current_idx + 1) % self.capacity
            self.current_size = min(self.current_size + 1, self.capacity)
            
    def sample(self, batch_size):
        with self.lock:
            indices = np.random.choice(self.current_size, batch_size, replace=False)
            batch = {key: self.buffers[key][indices] for key in self.buffers.keys()}
        return batch
        
    def can_sample(self, batch_size):
        return self.current_size >= batch_size