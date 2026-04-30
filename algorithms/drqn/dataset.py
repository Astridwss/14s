import os
import torch 
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from config import Config 
from data_pipeline.build_expert_data import generate_expert_csv

from sim.plan_file_process import PlanFileProcess
from data_pipeline.txl_test import write_plan_file_info_to_csv

class ExpertDataset(Dataset):
    "模仿学习真实数据集"
    def __init__(self, conf: Config):
        print(f"==============================开始执行[ExpertDataset]============================")
        self.conf = conf

        self.dataset_path = getattr(self.conf, 'local_scene_path')
        self.plan_id = getattr(self.conf, 'plan_id', 867) 
        
        base_name = os.path.splitext(self.dataset_path)[0]
        self.csv_path = os.path.abspath(f"{base_name}.csv")        

        # =================1. 检查并生成 CSV=======================
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            print(f"[ExpertDataset] 开始将预案 JSON 转换为模仿学习 CSV 样本...")
            
            generate_expert_csv(
                conf=self.conf, 
                dest_csv_path=self.csv_path
            )
            print(f"[ExpertDataset] CSV 样本生成完毕: {self.csv_path}")


            # ================= 打印离线数据 =================
            plan_id = 867
            scene_json_path = r"C:\webace_2026\14s\code\webace-3\test\scene.json"
            self.txl_csv_path = r"C:\webace_2026\14s\code\webace-3\txl_test.csv"

            print(f"[测试] 正在读取真实场景预案: {scene_json_path}...")
            processor = PlanFileProcess()
            real_plan_file_info = processor.read_plan_file_info_from_json(plan_id=plan_id, file_path=scene_json_path)
            
            print(f"[测试] 数据加载成功，准备写入 CSV: {self.txl_csv_path}...")
            write_plan_file_info_to_csv(
                plan_file_info=real_plan_file_info,
                dest_path=self.txl_csv_path
            )

            print("[测试] txl 转 csv 成功")
        # ================= 打印离线数据 =================


        # ===================加载 CSV 并构建张量=======================
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

        # 初始化全局矩阵
        all_obs = np.zeros((n_times, n_agents, obs_dim), dtype=np.float32)
        all_actions = np.zeros((n_times, n_agents), dtype=np.int64)
        # 全局态势矩阵 (Time, State_Dim)
        all_states = np.zeros((n_times, state_dim), dtype=np.float32)
        # 可用动作掩码 (Time, Agents, Actions) -> 默认全 1 (全部可用)
        all_avail_actions = np.ones((n_times, n_agents, n_actions), dtype=np.float32)

        # 性能优化：按时间步GroupBy，避免全表iterrows
        grouped = df.groupby('time_step')
        
        for i, t in enumerate(time_steps):
            t_df = grouped.get_group(t)
            
            # 提取 state：同一时刻所有agent的state是相同的，取第一行即可
            first_row = t_df.iloc[0]
            all_states[i] = np.array(first_row['state'].split('|'), dtype=np.float32)
            
            # 提取 obs 和 action
            for row in t_df.itertuples():
                a_id = int(row.agent_id)
                if a_id < n_agents:
                    all_actions[i, a_id] = int(row.expert_action)
                    all_obs[i, a_id] = np.array(row.obs.split('|'), dtype=np.float32)
                    all_avail_actions[i, a_id] = np.array(row.avail_actions.split('|'), dtype=np.float32)
                    

        # 转为 Tensor
        self.all_obs = torch.tensor(all_obs)
        self.all_actions = torch.tensor(all_actions)
        self.all_states = torch.tensor(all_states)
        self.all_avail_actions = torch.tensor(all_avail_actions)

        # 3. 构造强化学习必备的 last_actions (动作记忆，One-hot格式)
        self.all_last_actions = torch.zeros((n_times, n_agents, n_actions), dtype=torch.float32)
        for t in range(1, n_times): 
            acts = self.all_actions[t-1].unsqueeze(-1) 
            self.all_last_actions[t].scatter_(1, acts, 1.0)

        # 4. 计算滑动窗口
        self.seq_len = getattr(self.conf, 'seq_len', 10) 
        self.num_samples = max(0, n_times - self.seq_len + 1)
        
        if self.num_samples == 0:
            raise ValueError(f"数据总时长 ({n_times}) 小于网络要求的时间序列长度 ({self.seq_len})，无法训练")
        

        # 5. 数据增强：正样本过采样
        if getattr(self.conf, 'use_data_augmentation', True):
            self.sample_indices = self._apply_positive_oversample(list(range(self.num_samples)))
        else:
            self.sample_indices = list(range(self.num_samples))

        # 这里要加 len()，否则会打印几千个数字导致刷屏
        print(f"[ExpertDataset] 基础滑动窗口样本数: {self.num_samples} | 最终使用的总样本数: {len(self.sample_indices)}")


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