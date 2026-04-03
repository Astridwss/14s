import torch
import os 
import time
import numpy as np
from torch.utils.data import DataLoader, random_split

from config import Config
from algorithms.drqn.agent import ILAgents         
from algorithms.drqn.dataset import ExpertDataset   

class ILTrainRunner:
    """
    模仿学习训练调度器（带验证集评估机制）
    """
    def __init__(self, conf: Config):
        self.conf = conf
        self._set_seeds()
        
        # 1. 挂载算法层
        self.il_learner = ILAgents(self.conf)
        
        # 2. 准备数据集并进行 Train/Val 划分
        print("正在准备并校验数据集...")
        self.full_dataset = ExpertDataset(self.conf)
        
        train_ratio = getattr(self.conf, 'train_val_split', 0.8)
        print(f"离线数据划分比例：{train_ratio}")


        self.train_size = int(train_ratio * len(self.full_dataset))
        self.val_size = len(self.full_dataset) - self.train_size
        
        train_set, val_set = random_split(self.full_dataset, [self.train_size, self.val_size])
        
        # 3. 构造双重 DataLoader
        self.train_loader = DataLoader(
            dataset=train_set, 
            batch_size=getattr(self.conf, 'batch_size', 32), 
            shuffle=True, 
            drop_last=True,
            num_workers=0,        # 纯内存数据集必须设为 0，否则多进程拷贝会极大拖慢速度甚至报错
            pin_memory=True       
        )
        
        self.val_loader = DataLoader(
            dataset=val_set, 
            batch_size=getattr(self.conf, 'batch_size', 32), 
            shuffle=False,        
            drop_last=False,      
            num_workers=0,       
            pin_memory=True
        )
        
        self.model_dir = self.conf.model_dir

    def _set_seeds(self):
        np.random.seed(self.conf.seed)
        torch.manual_seed(self.conf.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.conf.seed)

    def _run_epoch(self, dataloader, is_train=True):
        """
        通用 Epoch 运行器，自动处理 Train/Eval 模式和指标平均
        """
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
                
        # 计算整个 Epoch 的平均值
        return {k: v / num_batches for k, v in total_metrics.items()}

    def _save_checkpoint(self, epoch_idx: int, is_final: bool = False):
        freq = getattr(self.conf, "save_frequency", 10)
        if is_final or (freq > 0 and epoch_idx % freq == 0):
            self.il_learner.save_model(epoch_idx)
            return self.model_dir
        return None

    def _execute_callbacks(self, epoch_idx, train_loss, model_dir, metrics_callback, weight_callback, is_final=False):
        """执行向平台通信的回调函数 (以验证集 Loss 为准)"""
        if metrics_callback is not None:
            metrics_callback({
                "episode": epoch_idx,  
                "loss": train_loss,
                "planId": getattr(self.conf, "plan_id", "unknown_plan"),
            })
            
        if model_dir is not None and weight_callback is not None:
            callback_payload = ({
                "episode": epoch_idx,
                "model_dir": model_dir,
                "algo": getattr(self.conf, "algorithm", "drqn"),
                "task_id": getattr(self.conf, "task_id", "unknown_task"),
                "planId": getattr(self.conf, "plan_id", "unknown_plan"),
            })
            if is_final:
                # 获取 dataset 中早已算好的绝对路径
                callback_payload["absCsvPath"] = self.full_dataset.csv_path
            
            weight_callback(callback_payload)

    def run(self, metrics_callback=None, weight_callback=None):
        """主循环"""
        task_id = getattr(self.conf, "task_id", "UNKNOW")
        epochs = getattr(self.conf, "epochs", 50)
        pause_flag_file = getattr(self.conf, "pause_flag_file", "")
        terminate_flag_file = getattr(self.conf, "terminate_flag_file", "")
        print(f"[ILTrainRunner] 开始模仿学习: 训练集 {self.train_size} 样本 | 验证集 {self.val_size} 样本")
        
        for epoch_idx in range(1, epochs + 1):
            # =========================终止标志位===================
            if terminate_flag_file and os.path.exists(terminate_flag_file):
                print(f"[TrainRunner] 检测到终止信号任务 {task_id} 即将强行退出...")
                os.remove(terminate_flag_file)
                break 

            # ========================暂停标志位=====================
            if pause_flag_file and os.path.exists(pause_flag_file):
                print(f"[TrainRunner] 检测到暂停信号训练已挂起 (Episode: {epoch_idx})...")
                while os.path.exists(pause_flag_file):
                    time.sleep(2)  
                print(f"[TrainRunner] 暂停信号解除训练继续...\n")
            # ======================================================
            
            # 1. 跑训练集
            train_res = self._run_epoch(self.train_loader, is_train=True)
            # 2. 跑验证集
            val_res = self._run_epoch(self.val_loader, is_train=False)
            
            # 3. 控制台输出
            print(f"[Epoch {epoch_idx:03d}/{epochs}]")
            print(f"  Train | Loss: {train_res['loss']:.4f} | 全局准度: {train_res['acc_global']:.1%} | 追踪准度: {train_res['acc_active']:.1%}")
            print(f"  Val   | Loss: {val_res['loss']:.4f} | 全局准度: {val_res['acc_global']:.1%} | 追踪准度: {val_res['acc_active']:.1%}")
            
            # 4. 存档判定
            is_last = (epoch_idx == epochs)
            model_dir = self._save_checkpoint(epoch_idx, is_final=is_last)
            
            # 5. 执行回调 (推送给平台验证集的表现)
            self._execute_callbacks(epoch_idx, train_res['loss'], model_dir, metrics_callback, weight_callback, is_final=is_last)

        print("[ILTrainRunner] 模仿学习结束")

if __name__ == "__main__":
    il_train = ILTrainRunner()
    conf = Config()
