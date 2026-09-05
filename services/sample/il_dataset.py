"""
模仿学习数据集 —— PyTorch Dataset，从 CSV 加载时序张量。

不调用同层 services/ 积木。CSV 生成由调用方在上层完成。
"""
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class ExpertDataset(Dataset):
    """模仿学习真实数据集。

    期望 conf 已注入完整维度（n_agents、obs_shape 等），CSV 已由调用方预先生成。
    """

    def __init__(self, conf):
        print(f"==============================开始执行[ExpertDataset]============================")
        self.conf = conf
        self._resolve_csv_path()
        self._load_and_build_tensors()
        self._build_last_actions()
        self._build_sliding_windows()
        self._build_sample_indices()
        print(f"[ExpertDataset] 基础滑动窗口样本数: {self.num_samples} | 最终使用的总样本数: {len(self.sample_indices)}")

    # ----------------------------------------------------------
    # 构建步骤
    # ----------------------------------------------------------

    def _resolve_csv_path(self):
        self.dataset_path = getattr(self.conf, 'local_scene_path')
        base_name = os.path.splitext(self.dataset_path)[0]
        self.csv_path = os.path.abspath(f"{base_name}.csv")
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            raise FileNotFoundError(
                f"[ExpertDataset] CSV 不存在或为空: {self.csv_path}。"
                f"请先通过 generate_expert_csv() 生成专家数据。"
            )

    def _load_and_build_tensors(self):
        print(f"[ExpertDataset]正在加载 CSV 数据并构建时序张量")
        df = pd.read_csv(self.csv_path)
        if df.empty:
            raise ValueError("CSV 文件是空的，请检查底层解析逻辑。")

        time_steps = sorted(df['time_step'].unique())
        n_times = len(time_steps)
        n_agents = self.conf.n_agents
        obs_dim = self.conf.obs_shape
        state_dim = self.conf.state_shape
        n_actions = self.conf.n_actions

        all_obs = np.zeros((n_times, n_agents, obs_dim), dtype=np.float32)
        all_actions = np.zeros((n_times, n_agents), dtype=np.int64)
        all_states = np.zeros((n_times, state_dim), dtype=np.float32)
        all_avail_actions = np.ones((n_times, n_agents, n_actions), dtype=np.float32)

        grouped = df.groupby('time_step')
        for i, t in enumerate(time_steps):
            t_df = grouped.get_group(t)
            first_row = t_df.iloc[0]
            all_states[i] = np.array(first_row['state'].split('|'), dtype=np.float32)
            for row in t_df.itertuples():
                a_id = int(row.agent_id)
                if a_id < n_agents:
                    all_actions[i, a_id] = int(row.expert_action)
                    all_obs[i, a_id] = np.array(row.obs.split('|'), dtype=np.float32)
                    all_avail_actions[i, a_id] = np.array(row.avail_actions.split('|'), dtype=np.float32)

        self.all_obs = torch.tensor(all_obs)
        self.all_actions = torch.tensor(all_actions)
        self.all_states = torch.tensor(all_states)
        self.all_avail_actions = torch.tensor(all_avail_actions)
        self._n_times = n_times

    def _build_last_actions(self):
        n_agents = self.conf.n_agents
        n_actions = self.conf.n_actions
        self.all_last_actions = torch.zeros((self._n_times, n_agents, n_actions), dtype=torch.float32)
        for t in range(1, self._n_times):
            acts = self.all_actions[t - 1].unsqueeze(-1)
            self.all_last_actions[t].scatter_(1, acts, 1.0)

    def _build_sliding_windows(self):
        self.seq_len = getattr(self.conf, 'seq_len', 10)
        self.num_samples = max(0, self._n_times - self.seq_len + 1)
        if self.num_samples == 0:
            raise ValueError(
                f"数据总时长 ({self._n_times}) 小于网络要求的时间序列长度 ({self.seq_len})，无法训练"
            )

    def _build_sample_indices(self):
        if getattr(self.conf, 'use_data_augmentation', True):
            self.sample_indices = self._apply_positive_oversample(list(range(self.num_samples)))
        else:
            self.sample_indices = list(range(self.num_samples))


    def _apply_positive_oversample(self, indices):
        """正样本过采样策略: 时序内发现正样本就增强"""
        pos_indices = []
        
        for i in range(self.num_samples):
            if (self.all_actions[i : i + self.seq_len] > 0).any():
                pos_indices.append(i)

        factor = getattr(self.conf, 'aug_factor', 5)
        
        augmented_indices = indices + pos_indices * (factor - 1)
        
        # 打乱索引，让同一个正样本片段散布在 Epoch 的不同 Batch 中
        np.random.shuffle(augmented_indices) 
        
        print(f"[Augmentation] 扫描完成，发现正样本段数: {len(pos_indices)} , 放大倍数: {factor}")
        return augmented_indices


    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        """
        按滑动窗口切片，返回与强化学习Forward一致的字段
        """
        real_idx = self.sample_indices[idx]

        obs_seq = self.all_obs[real_idx : real_idx + self.seq_len]
        act_seq = self.all_actions[real_idx : real_idx + self.seq_len]
        last_act_seq = self.all_last_actions[real_idx : real_idx + self.seq_len]
        
        # 取出对应的state和avail_actions
        state_seq = self.all_states[real_idx : real_idx + self.seq_len]
        avail_act_seq = self.all_avail_actions[real_idx : real_idx + self.seq_len]
        
        return {
            'obs': obs_seq,                # (seq_len, n_agents, obs_dim)
            'state': state_seq,            # (seq_len, state_dim)  Mix网络所需的占位符
            'actions': act_seq,            # (seq_len, n_agents)
            'last_actions': last_act_seq,  # (seq_len, n_agents, n_actions)
            'avail_actions': avail_act_seq # (seq_len, n_agents, n_actions)  防止网络前向报错
        }