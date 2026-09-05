"""
ILInferRunner —— 模仿学习离线数据推理评估器。

独立 Runner（无训练循环，不继承 BaseRunner）。
"""

import os

import torch
from torch.utils.data import DataLoader

from services.algorithms.drqn.agent import ILAgents
from services.sample.il_dataset import ExpertDataset
from services.sample.expert_data import generate_expert_csv
from services.scene.state.builder import ObservationBuilder
from services.scene.action.mapper import ActionMapper


class ILInferRunner:
    """模仿学习离线数据推理评估器。"""

    def __init__(self, conf):
        self.conf = conf

        # 1. 挂载算法层
        self.il_agents = ILAgents(self.conf)

        # 2. 加载平台指定的模型权重
        load_dir = getattr(self.conf, "load_dir", "")
        if not load_dir or not os.path.exists(load_dir):
            raise FileNotFoundError(
                f"[ILInferRunner] 找不到权重路径 {load_dir}"
            )
        self.il_agents.load_model(load_dir)

        # 3. 组装积木 → 生成 CSV（如不存在）→ 加载数据集
        print(f"[ILInferRunner] 正在加载离线测试数据集...")
        self._ensure_expert_csv()
        self.test_dataset = ExpertDataset(self.conf)

        dl_workers = int(getattr(self.conf, 'il_num_workers', 0) or 0)
        # pin_memory 走 CUDA 路径；昇腾 NPU 机无 CUDA，恒 True 会报错且无加速，仅在 CUDA 可用时开启。
        pin_memory = torch.cuda.is_available()
        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=getattr(self.conf, 'batch_size', 32),
            shuffle=False, drop_last=False,
            num_workers=dl_workers, pin_memory=pin_memory,
        )

    def _ensure_expert_csv(self):
        """确保专家数据 CSV 已生成（ConfigAssembler 已注入维度，直接读 conf）。"""
        scene_path = getattr(self.conf, 'local_scene_path', '')
        plan_id = getattr(self.conf, 'plan_id', 867)
        csv_path = os.path.splitext(scene_path)[0] + '.csv'

        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            return

        agent_keys = self.conf.radar_keys
        generate_expert_csv(
            conf=self.conf, dest_csv_path=csv_path,
            obs_builder=ObservationBuilder(
                agent_keys=agent_keys, target_keys=self.conf.target_keys,
                n_agents=self.conf.n_agents, n_targets=self.conf.n_targets,
                n_actions=self.conf.n_actions,
                radar_obs_dim=self.conf.radar_obs_dim,
            ),
            action_mapper=ActionMapper(
                agent_keys=agent_keys, target_keys=self.conf.target_keys,
                n_agents=self.conf.n_agents, n_actions=self.conf.n_actions,
            ),
            plan_id=plan_id, scene_file_path=scene_path,
            radar_keys=self.conf.radar_keys, target_keys=self.conf.target_keys,
            sat_keys=self.conf.satellites_keys,
        )

    def run_offline_inference(self):
        """离线数据集推演主循环。"""
        num_batches = len(self.test_loader)
        print(f"[ILInferRunner] 开始执行离线数据推理评估, "
              f"共计 {self.test_dataset.num_samples} 个样本, "
              f"{num_batches} 个批次...")

        if num_batches == 0:
            print("[ILInferRunner] 测试数据集为空")
            return None

        total_metrics = {'loss': 0.0, 'acc_global': 0.0, 'acc_active': 0.0}

        for batch_idx, batch in enumerate(self.test_loader):
            metrics = self.il_agents.evaluate(batch)
            for k in total_metrics.keys():
                total_metrics[k] += metrics[k]

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f"  [进度 {batch_idx+1}/{num_batches}] "
                      f"当前批次追踪准度: {metrics['acc_active']:.1%}")

        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

        print(f"总平均 Loss: {avg_metrics['loss']:.4f}")
        print(f"全局动作准确率: {avg_metrics['acc_global']:.1%}")
        print(f"实际追踪准确率: {avg_metrics['acc_active']:.1%}")

        return avg_metrics
