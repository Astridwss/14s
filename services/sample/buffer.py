# -*- coding: utf-8 -*-
import numpy as np
import threading
import torch
from typing import Optional

from use_cases.config.config_types import AlgorithmConfig, TrainingConfig


class EpisodeReplayBuffer:
    """QMIX 专属的轨迹经验池 (Off-policy Episode Replay Buffer)

    可通过两种方式构造:
      - EpisodeReplayBuffer(conf)  —— 兼容旧版扁平 RuntimeConfig
      - EpisodeReplayBuffer(conf, algo_config=ac, train_config=tc) —— 新版聚焦配置
    """

    def __init__(self, conf,
                 algo_config: Optional[AlgorithmConfig] = None,
                 train_config: Optional[TrainingConfig] = None):
        self.conf = conf
        ac = algo_config or AlgorithmConfig.from_config(conf)
        tc = train_config or TrainingConfig.from_config(conf)

        self.capacity = tc.buffer_size
        self.episode_limit = getattr(conf, 'max_episode_steps', 600)

        # RL 时间展开长度：完整 episode 过长会导致 GPU OOM，
        # 采样时随机取 train_seq_len 连续片段（<=0 或 >=episode_limit 时用完整 episode）。
        self.seq_length = getattr(tc, 'train_seq_len', 0)
        if self.seq_length <= 0 or self.seq_length >= self.episode_limit:
            self.seq_length = self.episode_limit

        self.n_agents = ac.n_agents
        self.obs_shape = ac.obs_shape[0] if isinstance(ac.obs_shape, tuple) else ac.obs_shape
        self.state_shape = ac.state_shape[0] if isinstance(ac.state_shape, tuple) else ac.state_shape
        self.n_actions = ac.n_actions

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

            # 子序列采样：从完整 episode 随机取 seq_length 连续片段，
            # 降低 DRQN 时间展开长度，避免 GPU 显存 OOM。
            if self.seq_length < self.episode_limit:
                max_start = self.episode_limit - self.seq_length
                starts = np.random.randint(0, max_start + 1, size=batch_size)
                for key in self.buffers.keys():
                    batch[key] = np.stack(
                        [batch[key][i, starts[i]:starts[i] + self.seq_length]
                         for i in range(batch_size)],
                        axis=0,
                    )
        return batch
        
    def can_sample(self, batch_size):
        return self.current_size >= batch_size