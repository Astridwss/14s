# -*- coding: utf-8 -*-
"""
训练编排：TrainRunner 负责 调度采集 → 学习 → 存档 → 日志。
"""
import os
import time
import sys
import numpy as np
import torch
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from config import Config
from env import GroupedEnvWrapper         # 使用分组环境
from algorithms.qmix.agent import Agents          # QMIX 的 Agents
from .buffer import EpisodeReplayBuffer
from .rollout import RolloutWorker

from utils.tv_display import LiveObserver
from sim.plan_file_process import PlanFileProcess


class TrainRunner:
    def __init__(self, conf: Config):
        self.conf = conf
        # 1. 注入随机种子
        np.random.seed(self.conf.seed)
        torch.manual_seed(self.conf.seed)
        random.seed(self.conf.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.conf.seed)
        
        # 2. 初始化环境
        self.env = GroupedEnvWrapper(self.conf)

        # 3. 初始化 QMIX 算法、Buffer 和 RolloutWorker
        self.agents = Agents(self.conf)
        self.buffer = EpisodeReplayBuffer(self.conf)
        self.rollout_worker = RolloutWorker(self.conf, self.env, self.agents)
        
        # ==================加载模仿学习的权重=======================
        load_dir = getattr(self.conf, 'load_dir', '')
        if load_dir and os.path.exists(load_dir):
            print(f"[TrainRunner] 正在加载预训练权重: {load_dir}")
            try:
                self.agents.policy.load_state(load_dir)
                print("[TrainRunner] 预训练权重加载成功")
            except Exception as e:
                print(f"[TrainRunner] 加载预训练权重失败，请检查路径或网络结构: {e}")
        else:
            print("[TrainRunner] 未提供 load_dir 或路径不存在，将从头开始随机初始化网络。")
        # ==========================================================

        self.episode_rewards = []

        self.env_steps = 0        # 时钟 1：环境总交互步数 (用于 epsilon 衰减)
        self.update_steps = 0    # 时钟 2：网络梯度更新次数 (用于 target network 更新)
        
        self.result_dir = self.conf.result_dir
        self.model_dir = self.conf.model_dir
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        print(f"[TrainRunner] Reward目录: {self.result_dir}")
        print(f"[TrainRunner] 模型目录: {self.model_dir}")
        print(f"[TrainRunner] 维度注入成功: n_agents={self.conf.n_agents}, obs_shape={self.conf.obs_shape}, state_shape={self.conf.state_shape}")

    def save_episode_reward_plot(self, rewards_list, save_path, label="Training Rewards", xlabel="Episode"):
        """绘制并保存奖励曲线"""
        if not rewards_list:
            return
            
        x = list(range(1, len(rewards_list) + 1))
        plt.figure(figsize=(10, 6))
        # 传入的是整个列表 rewards_list
        plt.plot(x, rewards_list, label="episode reward", color='blue', linewidth=1.5)
        plt.title(label)
        plt.xlabel(xlabel)
        plt.ylabel("Episode Total Reward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def run_learning(self, epsilon: float):
        """当 Buffer 足够大时，抽样并训练网络"""
        loss = None
        if self.buffer.can_sample(self.conf.batch_size):
            batch = self.buffer.sample(self.conf.batch_size)
            # 送入 QMIX 训练 (不再清空 buffer，因为是 off-policy 经验回放)
            loss = self.agents.train_(batch, self.update_steps, epsilon)
            self.update_steps += 1
        return loss

    def save_checkpoint(self, episode_idx: int, is_final: bool = False):
        """按频率或最后一轮保存模型"""
        freq = self.conf.save_frequency
        if is_final or (freq > 0 and episode_idx % freq == 0):
            self.agents.policy.save_model(episode_idx)
            return self.model_dir
        return None

    def log_and_report(self, episode_idx: int, ep_reward: float, step_count: int, epsilon: float):
        print(f"[Episode {episode_idx}/{self.conf.max_episodes}] ep_reward={ep_reward:.2f}, steps={step_count}, eps={epsilon:.3f}")

    def _execute_callbacks(self, episode_idx, ep_reward, loss, model_dir, metrics_callback, weight_callback):
        """执行向平台通信的回调函数"""

        # (A) 推送 Reward 和 Loss 等指标
        if metrics_callback is not None:
            metrics_callback({
                "episode": episode_idx,
                "reward": ep_reward, #回合奖励
                "loss": loss if loss is not None else 0.0,
                "planId": getattr(self.conf, "plan_id", "unknown_plan"),
            })
            
        # (B) 推送模型权重
        if model_dir is not None and weight_callback is not None:
            weight_callback({
                "episode": episode_idx,
                "model_dir": model_dir,
                "algo": getattr(self.conf, "algorithm", "qmix"),
                "task_id": getattr(self.conf, "task_id", "unknown_task"),
                "planId": getattr(self.conf, "plan_id", "unknown_plan"),
            })

    def _get_fixed_trajectories(self):
        """利用现有的 PlanFileProcess 从 json 预案中提取固定航迹"""
        plan_processor = PlanFileProcess()
        plan_id = getattr(self.conf, "plan_id", 0)
        scene_path = getattr(self.conf, "local_scene_path", "")
        
        fixed_trajectories = {}
        if not scene_path or not os.path.exists(scene_path):
            print(f"[TrainRunner] 警告: 未找到场景文件 {scene_path}，无法绘制轨迹底图")
            return fixed_trajectories
            
        # 1. 调底层工具解析整个作战场景
        battle_scene = plan_processor.read_battle_scene_from_json(plan_id, scene_path)
        
        # 2. 剥离并组装需要的 (lon, lat) 坐标点集
        for target_id, target_info in battle_scene.dict_target_id_info.items():
            pts = []
            # 必须按照时间先后顺序对字典的 key(时间点) 进行排序，保证画出的是连续的线
            for time_key in sorted(target_info.dict_target_traj_pt_info.keys()):
                traj_pt = target_info.dict_target_traj_pt_info[time_key]
                pts.append((traj_pt.longitude, traj_pt.latitude))
            if pts:
                fixed_trajectories[target_id] = pts
                
        return fixed_trajectories


    def run(self, metrics_callback=None, weight_callback=None):
        """
        主循环：计算 Epsilon → 委托 worker 采集数据 → 尝试学习 → 日志与保存。
        """
        task_id = getattr(self.conf, "task_id", "UNKNOW")
        pause_flag_file = getattr(self.conf, "pause_flag_file", "")
        terminate_flag_file = getattr(self.conf, "terminate_flag_file", "")
        # ======================TV==================================
        # 实例化tv_display观察者 (每隔 20 局看一次)
        live_viewer = LiveObserver(watch_freq=10, fps=30)
        # 新增：1. 一次性提前提取预案中的固定航迹
        fixed_path_data = self._get_fixed_trajectories()
        # ======================TV==================================


        for episode_idx in range(1, self.conf.max_episodes+1):
            # =========================终止标志位=============================
            if terminate_flag_file and os.path.exists(terminate_flag_file):
                print(f"[TrainRunner] 检测到终止信号任务 {task_id} 即将强行退出...")
                os.remove(terminate_flag_file)
                break 

            # ========================暂停标志位=====================
            if pause_flag_file and os.path.exists(pause_flag_file):
                print(f"[TrainRunner] 检测到暂停信号训练已挂起 (Episode: {episode_idx})...")
                while os.path.exists(pause_flag_file):
                    time.sleep(2)  
                print(f"[TrainRunner] 暂停信号解除训练继续...\n")
            # ======================================================

            # 计算当前局的 Epsilon (探索率线性衰减)
            epsilon = max(
                self.conf.epsilon_finish, 
                self.conf.epsilon_start - (self.conf.epsilon_start - self.conf.epsilon_finish) * (self.env_steps / self.conf.epsilon_anneal_time)
            )

            # ======================TV==================================
            #  1. 询问观察者是否需要拉起本局tv_display
            live_viewer.check_and_start(episode_idx, epsilon, fixed_trajectories=fixed_path_data)

            #  2. 传给 Rollout，让它无脑调 callback 即可
            ep_reward, step_count, ep_data = self.rollout_worker.generate_train_episode(
                epsilon=epsilon, 
                episode_num=episode_idx,
                render_callback=live_viewer.render_callback 
            )

            #  3. 通知观察者本局结束，安全关窗
            live_viewer.close()
            # ======================TV==================================

            self.env_steps += step_count
            self.episode_rewards.append(ep_reward)

            # 2. 将数据存入 Buffer
            self.buffer.store_episode(ep_data)

            # 3. 训练并获取 loss
            loss = self.run_learning(epsilon)
            print(f"loss is {loss}")

            # 4. 保存权重与日志
            is_last = (episode_idx == self.conf.max_episodes)
            model_dir = self.save_checkpoint(episode_idx, is_final=is_last)
            self.log_and_report(episode_idx, ep_reward, step_count, epsilon)

            # 5. 保存奖励曲线
            self.save_episode_reward_plot(self.episode_rewards, os.path.join(self.result_dir, "episode_reward.png"))

            # 6. 执行回调 (推送给平台)
            self._execute_callbacks(episode_idx, ep_reward, loss, model_dir, metrics_callback, weight_callback)

        print("训练管理器：本轮训练结束。")


if __name__ == "__main__":
    conf = Config(mode="train")
    runner = TrainRunner(conf)
    runner.run()