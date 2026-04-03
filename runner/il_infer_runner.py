import torch
import os
from torch.utils.data import DataLoader

from config import Config
from algorithms.drqn.agent import ILAgents
from algorithms.drqn.dataset import ExpertDataset

class ILInferRunner:
    """
    模仿学习离线数据推理评估器
    """
    def __init__(self, conf: Config):
        self.conf = conf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 挂载算法层
        self.il_agents = ILAgents(self.conf)
        
        # 2. 加载平台指定的模型权重
        load_dir = getattr(self.conf, "load_dir", "")
        if not load_dir or not os.path.exists(load_dir):
            raise FileNotFoundError(f"[ILInferRunner] 找不到权重路径 {load_dir}")
        self.il_agents.load_model(load_dir)
        
        # 3. 准备离线测试数据集 (复用 Dataset)
        print(f"[ILInferRunner] 正在加载离线测试数据集...")
        self.test_dataset = ExpertDataset(self.conf)
        
        # 推理用的DataLoader：shuffle=False，保证时序一致
        self.test_loader = DataLoader(
            dataset=self.test_dataset, 
            batch_size=getattr(self.conf, 'batch_size', 32), 
            shuffle=False,        
            drop_last=False,      
            num_workers=0,        
            pin_memory=True
        )

    def run_offline_inference(self):
        """
        离线数据集推演主循环
        """
        num_batches = len(self.test_loader)
        print(f"[ILInferRunner] 开始执行离线数据推理评估, 共计 {self.test_dataset.num_samples} 个样本, {num_batches} 个批次...")
        
        if num_batches == 0:
            print("[ILInferRunner] 测试数据集为空")
            return None
            
        total_metrics = {'loss': 0.0, 'acc_global': 0.0, 'acc_active': 0.0}
        
        # 遍历数据加载器
        for batch_idx, batch in enumerate(self.test_loader):
            # 调用 evaluate，内部自带 @torch.no_grad() 和 drqn_net.eval()
            metrics = self.il_agents.evaluate(batch)
            
            for k in total_metrics.keys():
                total_metrics[k] += metrics[k]
                
            # 每 10 个批次打印一次进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f"  [进度 {batch_idx+1}/{num_batches}] 当前批次追踪准度: {metrics['acc_active']:.1%}")

        # 计算整个测试集的平均指标
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        print(f"总平均 Loss: {avg_metrics['loss']:.4f}")
        print(f"全局动作准确率: {avg_metrics['acc_global']:.1%}")
        print(f"实际追踪准确率: {avg_metrics['acc_active']:.1%}")

        return avg_metrics