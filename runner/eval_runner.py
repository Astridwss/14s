# -*- coding: utf-8 -*-
import os
import time
import json
import sys
import dataclasses
from pathlib import Path

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from config import Config
from env import GroupedEnvWrapper
from algorithms.qmix.agent import Agents
from runner.rollout import RolloutWorker

class EvalRunner:
    def __init__(self, conf):
        self.conf = conf
        
        # 1. 初始化环境
        self.env = GroupedEnvWrapper(self.conf)
        
        # 3. 实例化算法
        self.agents = Agents(self.conf)
        
        # 4. 加载平台指定的模型权重
        load_dir = getattr(self.conf, "load_dir", "")

        if not load_dir or not os.path.exists(load_dir):
            raise FileNotFoundError(f"[EvalRunner] 找不到权重路径 {load_dir}")

        self.agents.policy.load_state(load_dir)
        print(f"[EvalRunner] 成功加载推演权重: {load_dir}")

        # 5. 初始化数据采集目录
        self.eval_records_dir = self.conf.eval_records_dir
        os.makedirs(self.eval_records_dir, exist_ok=True)
        
        # 6. 实例化RolloutWorker
        self.rollout_worker = RolloutWorker(self.conf, self.env, self.agents)

    def _save_dataclass_records_to_json(self, eval_records):
        """将评估采集数据安全地序列化到本地 JSON 文件"""
        if not eval_records:
            return None
            
        task_id = getattr(self.conf, 'task_id', 'local_test')
        file_name = f"{task_id}_own_test.json"
        file_path = os.path.join(self.eval_records_dir, file_name)
        
        # ================= [新增序列化处理器] =================
        def custom_encoder(obj):
            """将 dataclass 或带有 __dict__ 的自定义对象转为标准字典"""
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
        # ======================================================

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                # 传入 default=custom_encoder
                json.dump(eval_records, f, ensure_ascii=False, indent=2, default=custom_encoder)
            print(f"[EvalRunner] 原始推理评估采集数据存入文件: {file_path}")
            return os.path.abspath(file_path)
        except IOError as e:
            print(f"[EvalRunner] 原始推理评估采集数据保存失败: {e}")
            return None

    def _save_records_to_json(self, eval_records):
        """将评估采集数据安全地序列化到本地 JSON 文件"""
        if not eval_records:
            return None
            
        task_id = getattr(self.conf, 'task_id', 'local_test')
        file_name = f"{task_id}_own_test.json"
        file_path = os.path.join(self.eval_records_dir, file_name)
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(eval_records, f, ensure_ascii=False, indent=2)
            print(f"[EvalRunner] 原始推理评估采集数据存入文件: {file_path}")
            return os.path.abspath(file_path)
        except IOError as e:
            print(f"[EvalRunner] 原始推理评估采集数据保存失败: {e}")
            return None
    
    def _save_all_records_to_json_bk(self, all_eval_data):
        """将所有局的评估数据合并序列化到同一个本地 JSON 文件，并返回绝对路径"""
        if not all_eval_data:
            return None
            
        import dataclasses #  引入 dataclasses
        
        #  定义一个能看懂数据类的 JSON 翻译官
        class DataClassEncoder(json.JSONEncoder):
            def default(self, obj):
                if dataclasses.is_dataclass(obj):
                    return dataclasses.asdict(obj)
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                return super().default(obj)

        task_id = getattr(self.conf, 'task_id', 'local_test')
        file_name = f"{task_id}_eval_records.json" 
        file_path = os.path.join(self.eval_records_dir, file_name)
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                #  加上 cls=DataClassEncoder 让它用
                json.dump(all_eval_data, f, ensure_ascii=False, indent=2, cls=DataClassEncoder)
            print(f"[EvalRunner] 所有评估数据已合并存入文件: {file_path}")
            return os.path.abspath(file_path)
        except IOError as e:
            print(f"[EvalRunner] 评估数据保存失败: {e}")
            return None


    def _save_all_records_to_json(self, all_eval_data):
        if not all_eval_data:
            print(f"[EvalRunner] 警告：没有评估数据可供保存")

        singele_episode_records = list(all_eval_data.values())[-1]

        def stringify_keys(d):
            if isinstance(d, dict):
                return {str(k): stringify_keys(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [stringify_keys(i) for i in d]
            return d
        safe_eval_data = stringify_keys(singele_episode_records)     

        task_id = getattr(self.conf, 'task_id', 'local_test')
        result_file_name = f"{task_id}_eval_records.json"
        metric_file_name = f"{task_id}_eval_metric.json"

        result_file_path = os.path.join(self.eval_records_dir, result_file_name)
        metric_file_name = os.path.join(self.eval_records_dir, metric_file_name)

        result_abs_path = os.path.abspath(result_file_path)
        metric_abs_path = os.path.abspath(metric_file_name)

        try:
            from sim import PlanFileProcess
            processor = PlanFileProcess()

            plan_file_info = processor.read_plan_file_info_from_json(
                plan_id=getattr(self.conf, 'plan_id', 867),
                file_path=self.conf.local_scene_path
            )
            battle_scene = plan_file_info.battle_scene

            print(f"[EvalRunner] 正在通过PlanFileProcess 转换并保存推演结果")

            processor.write_model_inference_result_to_json(
                dict_model_inference_result=safe_eval_data,
                dest_path=result_abs_path
            )

            processor.write_model_inference_metric_to_json(
                dict_model_inference_result=safe_eval_data,
                battle_scene = battle_scene,
                dest_path=metric_abs_path
            )

            return result_abs_path, metric_abs_path

        except Exception as e:
            print(f"[EvalRunner] 数据保存失败，错误原因：{e}")
            return None

    # def _execute_callbacks(self, episode_idx, ep_reward, step_count, eval_records_file, result_callback):
    #     """执行向平台通信的回调函数"""
    #     if result_callback is not None:
    #             result_callback({
    #                 "episode": episode_idx,
    #                 "reward": ep_reward,
    #                 "steps": step_count,
    #                 "eval_records_file": eval_records_file
    #             })

    def run(self):
        """推演循环"""
        all_eval_data = {}  #  用这个字典来装所有局的数据
        task_id = getattr(self.conf, "task_id", "UNKNOW")
        pause_flag_file = getattr(self.conf, "pause_flag_file", "")
        terminate_flag_file = getattr(self.conf, "terminate_flag_file", "")
        print(f"\n[EvalRunner] 开始推演评估，共计 {self.conf.max_episodes} 局...")

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
                print(f"[TrainRunner] 暂停信号解除训练继续\n")
            # ======================================================
            
            # 1. 交给 RolloutWorker 跑一局
            ep_reward, step_count, eval_records = self.rollout_worker.generate_eval_episode()
            print(f"[推演 {episode_idx}/{self.conf.max_episodes}] reward={ep_reward:.2f}, steps={step_count}")

            all_eval_data[f"episode_{episode_idx}"] = eval_records
        
        print(f"[EvalRunner] 智能体输出信息处理对象已将输出信息转换为装备规划动作\n")
        print("[EvalRunner] 推演评估结束\n")
        
        result_abs_path, metric_abs_path = self._save_all_records_to_json(all_eval_data)
        own_eval_file_path = self._save_dataclass_records_to_json(all_eval_data)
        
        return result_abs_path, metric_abs_path
    
if __name__ == "__main__":
    conf = Config(mode="eval")
    runner = EvalRunner(conf)
    runner.run()