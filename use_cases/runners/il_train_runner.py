"""
ILTrainRunner —— 模仿学习训练调度器（带验证集评估）。

继承 BaseRunner，统一通过 self.push 推送指标和权重。
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from use_cases.runners.base_runner import BaseRunner
from use_cases.config.config_types import EnvConfig, TrainingConfig
from services.algorithms.drqn.agent import ILAgents
from services.sample.il_dataset import ExpertDataset
from services.sample.expert_data import generate_expert_csv
from services.scene.state.builder import ObservationBuilder
from services.scene.action.mapper import ActionMapper


class ILTrainRunner(BaseRunner):
    """模仿学习训练调度器（train/val split + epoch 循环）。"""

    def __init__(self, conf, push=None):
        super().__init__(conf, push)
        self._set_seeds()

        # ---- 提取聚焦配置 ----
        self._ec = getattr(conf, 'env', None) or EnvConfig.from_config(conf)
        tc = getattr(conf, 'train', None) or TrainingConfig.from_config(conf)

        # 1. 挂载算法层
        self.il_learner = ILAgents(self.conf)

        # 2. 组装积木 → 生成 CSV（如不存在）→ 加载数据集
        print("正在准备并校验数据集...")
        self._ensure_expert_csv()

        self.full_dataset = ExpertDataset(self.conf)
        train_ratio = tc.train_val_split
        print(f"离线数据划分比例：{train_ratio}")

        self.train_size = int(train_ratio * len(self.full_dataset))
        self.val_size = len(self.full_dataset) - self.train_size

        train_set, val_set = random_split(
            self.full_dataset, [self.train_size, self.val_size]
        )

        # 3. 构造双重 DataLoader
        batch_size = tc.batch_size
        # IL 数据加载并行度：独立于 RL 的 env 采样 rl_num_workers（Windows 开发机报错可设 0）
        dl_workers = int(getattr(self.conf, 'il_num_workers', 0) or 0)
        # pin_memory 走 CUDA 的 cudaHostAlloc 路径；昇腾 NPU 机无 CUDA，恒 True 会报错且无加速，
        # 故仅在 CUDA 可用时开启，NPU/CPU 下置 False。
        pin_memory = torch.cuda.is_available()
        self.train_loader = DataLoader(
            dataset=train_set, batch_size=batch_size,
            shuffle=True, drop_last=True,
            num_workers=dl_workers, pin_memory=pin_memory,
        )
        self.val_loader = DataLoader(
            dataset=val_set, batch_size=batch_size,
            shuffle=False, drop_last=False,
            num_workers=dl_workers, pin_memory=pin_memory,
        )

        self.model_dir = getattr(conf, 'model_dir', './models')
        self.il_learner.model_dir = self.model_dir

    # ============================================================
    # 公开 API
    # ============================================================

    def run(self):
        """主循环：train epoch → val epoch → checkpoint → push。"""
        task_id = getattr(self.conf, "task_id", "UNKNOW")
        epochs = getattr(self.conf, "epochs", 50)
        print(f"[ILTrainRunner] 开始模仿学习: "
              f"训练集 {self.train_size} 样本 | 验证集 {self.val_size} 样本")

        for epoch_idx in range(1, epochs + 1):
            if not self._should_continue(epoch_idx):
                break

            train_res = self._run_epoch(self.train_loader, is_train=True)
            val_res = self._run_epoch(self.val_loader, is_train=False)

            print(f"[Epoch {epoch_idx:03d}/{epochs}]")
            print(f"  Train | Loss: {train_res['loss']:.4f} | "
                  f"全局准度: {train_res['acc_global']:.1%} | "
                  f"追踪准度: {train_res['acc_active']:.1%}")
            print(f"  Val   | Loss: {val_res['loss']:.4f} | "
                  f"全局准度: {val_res['acc_global']:.1%} | "
                  f"追踪准度: {val_res['acc_active']:.1%}")

            is_last = (epoch_idx == epochs)
            model_dir = self._save_checkpoint(epoch_idx, is_final=is_last)
            self._push_progress(epoch_idx, train_res['loss'],
                                model_dir, is_final=is_last)

        print("[ILTrainRunner] 模仿学习结束")

    # ============================================================
    # 内部
    # ============================================================

    def _set_seeds(self):
        tc = getattr(self.conf, 'train', None) or TrainingConfig.from_config(self.conf)
        np.random.seed(tc.seed)
        torch.manual_seed(tc.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(tc.seed)

    def _ensure_expert_csv(self):
        """每次都重新生成专家数据 CSV（预案/目标键可能变化，且复用旧文件可能残破过期）。

        generate_expert_csv 内部先写 .tmp 再原子替换，中断不会在目标路径留下残缺文件。
        """
        ec = self._ec
        scene_path = ec.local_scene_path
        csv_path = os.path.splitext(scene_path)[0] + '.csv'

        print(f"[ILTrainRunner] 生成专家数据 CSV: {csv_path}")

        agent_keys = ec.agent_keys
        obs_builder = ObservationBuilder(
            agent_keys=agent_keys, target_keys=ec.target_keys,
            n_agents=ec.n_agents, n_targets=ec.n_targets,
            n_actions=ec.n_actions, radar_obs_dim=ec.radar_obs_dim,
            satellite_ids=ec.satellites_keys,
        )
        action_mapper = ActionMapper(
            agent_keys=agent_keys, target_keys=ec.target_keys,
            n_agents=ec.n_agents, n_actions=ec.n_actions,
            satellite_keys=ec.satellites_keys,
        )

        generate_expert_csv(
            conf=self.conf, dest_csv_path=csv_path,
            obs_builder=obs_builder, action_mapper=action_mapper,
            plan_id=ec.plan_id, scene_file_path=scene_path,
            radar_keys=ec.radar_keys, target_keys=ec.target_keys,
            sat_keys=ec.satellites_keys,
        )
        print(f"[ILTrainRunner] 专家数据 CSV 生成完毕: {csv_path}")

    def _run_epoch(self, dataloader, is_train=True):
        """通用 Epoch 运行器，自动处理 Train/Eval 模式和指标平均。"""
        total_metrics = {'loss': 0.0, 'acc_global': 0.0, 'acc_active': 0.0}
        num_batches = len(dataloader)
        if num_batches == 0:
            return total_metrics

        for batch in dataloader:
            if is_train:
                metrics = self.il_learner.learn(batch)
            else:
                metrics = self.il_learner.evaluate(batch)

            for k in total_metrics.keys():
                total_metrics[k] += metrics[k]

        return {k: v / num_batches for k, v in total_metrics.items()}

    def _save_checkpoint(self, epoch_idx: int, is_final: bool = False):
        freq = getattr(self.conf, "save_frequency", 10)
        if is_final or (freq > 0 and epoch_idx % freq == 0):
            self.il_learner.save_model(epoch_idx)
            return self.model_dir
        return None

    def _push_progress(self, epoch_idx, train_loss, model_dir, is_final=False):
        """推送训练指标与模型权重到平台。"""
        if self.push is None:
            return

        plan_id = getattr(self.conf, "plan_id", "unknown_plan")
        self.push.push_metrics({
            "episode": epoch_idx,
            "loss": train_loss,
            "planId": plan_id,
        })

        if model_dir is not None:
            payload = {
                "episode": epoch_idx,
                "model_dir": model_dir,
                "algo": getattr(self.conf, "algorithm", "drqn"),
                "planId": plan_id,
            }
            if is_final:
                payload["absCsvPath"] = self.full_dataset.csv_path
            self.push.push_weights(payload)
