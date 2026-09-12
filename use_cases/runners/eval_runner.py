"""
EvalRunner —— 推演评估运行器。

继承 BaseRunner，加载训练好的模型权重，执行推演并保存结果。
"""

import dataclasses
import glob
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 确保 sim 模块可导入
_proj_root = Path(__file__).resolve().parent.parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from use_cases.runners.base_runner import BaseRunner
from use_cases.config.config_types import EnvConfig, AlgorithmConfig, InfraConfig
from services.scene.env_wrapper import GroupedEnvWrapper
from services.algorithms.qmix.agent import Agents
from services.sample.rollout import RolloutWorker
from services.scene.grouping import RadarGrouper
from services.evaluation import (
    BaselineEvaluator, parse_metric_file, scene_target_ids,
)
from sim import PlanFileProcess
from utils.weight_naming import build_weight_prefix
from services.zmq.situation_publisher import TrainingSituationPushService
from services.zmq.situation_collector import SituationCollector
from services.zmq.situation_pool import SituationPool, SituationTrajectory
from services.zmq.situation_sender import SituationSender
from utils.situation_logger import SituationLogHook


# ============================================================
# 模块级序列化工具（纯函数，供 EvalRunner 内部使用）
# ============================================================

def _default_json_encoder(obj):
    """dataclass / 普通对象的 JSON 序列化回退编码器。"""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


def _stringify_keys(d):
    """递归将字典所有键转为 str。"""
    if isinstance(d, dict):
        return {str(k): _stringify_keys(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_stringify_keys(i) for i in d]
    return d


def _fmt_time(ts):
    """Unix 时间戳 → 可读本地时间字符串，如 '2026-09-02 14:30:15'。"""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_duration(seconds):
    """秒数 → 可读耗时字符串，如 '83.45 秒' / '1 分 23.45 秒' / '1.23 小时'。"""
    seconds = float(seconds)
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} 小时"
    if seconds >= 60:
        return f"{int(seconds // 60)} 分 {seconds % 60:.2f} 秒"
    return f"{seconds:.2f} 秒"


class EvalRunner(BaseRunner):
    """推演评估运行器。"""

    def __init__(self, conf, push=None):
        super().__init__(conf, push)

        # ---- 提取聚焦配置 ----
        ec = getattr(conf, 'env', None) or EnvConfig.from_config(conf)
        ac = getattr(conf, 'algo', None) or AlgorithmConfig.from_config(conf)

        # 1. 初始化环境
        self.env = GroupedEnvWrapper(self.conf, env_config=ec)

        # 2. 定位权重目录（提前，供分组一致性校验使用）
        load_dir = getattr(self.conf, "load_dir", "")
        if not load_dir or not os.path.exists(load_dir):
            raise FileNotFoundError(
                f"[EvalRunner] 找不到权重路径 {load_dir}"
            )

        # 3. H-QMIX 分组（需与训练时一致）
        #    优先取训练时落盘的 group_size，避免前端漏传导致分组不一致、权重加载失败。
        group_size = getattr(ac, 'group_size', 0)
        meta = self._read_train_meta(load_dir)
        if meta is not None:
            group_size = int(meta.get('group_size', group_size))
            print(f"[EvalRunner] 从训练元数据恢复 group_size={group_size} "
                  f"(grouping={meta.get('use_grouping', None)})")

        group_assignments = RadarGrouper.try_build(
            group_size=group_size,
            radar_keys=ec.radar_keys,
            radar_info_dict=getattr(self.env, 'dict_radar_info', None),
        )

        # 4. 分组模式与权重目录一致性校验（给出可操作的报错）
        self._check_grouping_weights(load_dir, group_assignments)

        # 5. 实例化算法并加载权重
        self.agents = Agents(self.conf, algo_config=ac,
                             group_assignments=group_assignments)
        # 权重文件名前缀需与训练时落盘一致：group_size 用元数据恢复后的值
        self.agents.policy.weight_prefix = build_weight_prefix(
            ec.n_radars, ec.n_satellites, ec.n_targets, group_size,
        )
        self.agents.policy.load_state(load_dir)
        print(f"[EvalRunner] 成功加载推演权重: {load_dir}")

        # 6. 初始化数据采集目录
        self.eval_records_dir = getattr(conf, 'eval_records_dir', './eval_records')
        os.makedirs(self.eval_records_dir, exist_ok=True)

        # 预案文件缓存（数 MB，推理产物与基线产物共用，全流程只读一次）
        self._plan_file_info = None

        # 7. 实例化 RolloutWorker
        self.rollout_worker = RolloutWorker(
            self.env, self.agents,
            radar_keys=ec.agent_keys,
            target_keys=ec.target_keys,
        )
        # ---- ZMQ 态势推送 ----
        self._zmq = TrainingSituationPushService.from_conf(conf)

        # ---- 态势轨迹池 + 独立发送线程（支持前端倍速，与 RL 并行模式同构） ----
        self._situation_pool = None
        self._situation_sender = None
        
        # push_interval>0 才非 None
        if self._zmq is not None:
            ic = getattr(conf, 'infra', None) or InfraConfig.from_config(conf)
            #建池
            self._situation_pool = SituationPool(
                warn_episodes=ic.situation_warn_episodes,
                max_episodes=ic.situation_pool_max,
            )
            #独立发送线程
            self._situation_sender = SituationSender(
                publisher=self._zmq.publisher,
                task_id=getattr(conf, 'task_id', 'UNKNOWN'),
                pool=self._situation_pool,
                base_interval=ic.situation_base_interval,
                speed_refresh=ic.situation_speed_refresh,
            )
            self._situation_sender.start()
            print("[EvalRunner] 态势轨迹池 + 独立发送线程已启动（前端倍速参数生效中）")

        # ---- 态势日志 hook ----
        self._situation_log_hook = SituationLogHook(
            radar_keys=ec.agent_keys,
            target_keys=ec.target_keys,
            log_interval=50,
        )

    # ============================================================
    # 分组一致性
    # ============================================================

    @staticmethod
    def _read_train_meta(load_dir):
        """读取训练时落盘的分组元数据，缺失或损坏时返回 None。"""
        meta_path = os.path.join(load_dir, "train_meta.json")
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError) as e:
            print(f"[EvalRunner] 训练元数据读取失败，回退到前端/默认 group_size: {e}")
            return None

    @staticmethod
    def _check_grouping_weights(load_dir, group_assignments):
        """校验权重目录与分组模式是否匹配，失败时给出可操作的报错。

        权重文件名带「实体数 + 分组数」前缀，故用通配符匹配角色后缀判断。
        """
        def _has(pattern):
            return bool(glob.glob(os.path.join(load_dir, pattern)))

        if group_assignments is not None:
            need = ["*_lower_mixer.pkl", "*_upper_mixer.pkl"]
            missing = [n for n in need if not _has(n)]
            if missing:
                raise FileNotFoundError(
                    f"[EvalRunner] 当前为 H-QMIX 分组模式，但权重目录缺少 {missing}。"
                    f"若训练时使用标准 QMIX，请将 group_size 置 0；"
                    f"否则请确认权重目录与训练时一致: {load_dir}"
                )
        else:
            if not _has("*_qmix.pkl"):
                raise FileNotFoundError(
                    f"[EvalRunner] 当前为标准 QMIX（group_size=0），但权重目录缺少 "
                    f"*_qmix.pkl。若训练时启用了 H-QMIX 分组，"
                    f"请下发与训练一致的 group_size。目录: {load_dir}"
                )

    # ============================================================
    # 公开 API
    # ============================================================

    def run(self):
        """推演入口：态势推送走轨迹池（支持前端倍速），否则逐帧串行。"""
        if self._situation_pool is not None:
            return self._run_pooled()
        return self._run_serial()

    def _run_pooled(self):
        """轨迹池推演：采集整局态势 → 入池 → 发送线程按前端倍速逐帧推送。"""
        all_eval_data = {}
        start_time = time.time()
        print(f"\n[EvalRunner] 开始推演评估，共计 {self.conf.max_episodes} 局...")

        try:
            for episode_idx in range(1, self.conf.max_episodes + 1):
                if not self._should_continue(episode_idx):
                    break

                # 组装 hooks：态势日志 + 轨迹采集（仅命中推送间隔或末局时采集入池）
                hooks = [self._situation_log_hook]
                collector = None
                if self._should_push_episode(episode_idx):
                    collector = SituationCollector(episode_idx)
                    hooks.append(collector)

                ep_reward, step_count, eval_records = self.rollout_worker.generate_eval_episode(
                    step_hooks=hooks,
                )
                print(f"[推演 {episode_idx}/{self.conf.max_episodes}] ", f"reward={ep_reward:.2f}, steps={step_count}")

                # 将采集到的态势数据放入轨迹池中
                if collector is not None and collector.frames:
                    self._situation_pool.put(
                        SituationTrajectory(episode_idx=episode_idx, frames=collector.frames)
                    )

                all_eval_data[f"episode_{episode_idx}"] = eval_records
        finally:
            # 发完池中剩余轨迹再退出（前端消费过慢由 finish 超时兜底）
            if self._situation_sender is not None:
                self._situation_sender.finish()
                self._situation_sender = None

        return self._finalize(all_eval_data, start_time)

    def _run_serial(self):
        """串行推演：逐 step 推送态势（原行为；推送关闭时亦走此路，仅产出评估记录）。"""
        all_eval_data = {}
        start_time = time.time()
        print(f"\n[EvalRunner] 开始推演评估，共计 {self.conf.max_episodes} 局...")

        for episode_idx in range(1, self.conf.max_episodes + 1):
            if not self._should_continue(episode_idx):
                break

            # 组装 hooks：态势日志 + ZMQ 逐帧推送（按推送间隔调度）
            hooks = [self._situation_log_hook]
            if self._zmq:
                hooks.extend(self._zmq.get_step_hooks(episode_idx))

            ep_reward, step_count, eval_records = self.rollout_worker.generate_eval_episode(
                step_hooks=hooks,
            )
            print(f"[推演 {episode_idx}/{self.conf.max_episodes}] ", f"reward={ep_reward:.2f}, steps={step_count}")

            all_eval_data[f"episode_{episode_idx}"] = eval_records

        return self._finalize(all_eval_data, start_time)
        
    
    def _should_push_episode(self, episode_idx: int) -> bool:
        """是否应推送该局态势（命中推送间隔或末局）。"""
        push_interval = getattr(self.conf, 'push_interval', 0)
        if push_interval <= 0:
            return False
        return episode_idx % push_interval == 0 or episode_idx == self.conf.max_episodes

    def _finalize(self, all_eval_data, start_time):
        """汇总评估数据，生成平台格式产物并返回响应字段。"""
        end_time = time.time()
        cost_seconds = end_time - start_time

        print(f"[EvalRunner] 智能体输出信息处理对象已将输出信息转换为装备规划动作\n")
        print(f"[EvalRunner] 推演评估结束, 总耗时: {cost_seconds:.2f} 秒\n")

        result_abs_path, metric_abs_path = self._save_all_records_to_json(all_eval_data)
        self._save_dataclass_records_to_json(all_eval_data)

        eval_fields = self._build_eval_fields(metric_abs_path, start_time, end_time, cost_seconds)

        return {
            "timeSeriesFile": result_abs_path,
            "evalFile": metric_abs_path,
            **eval_fields,
        }


    # ============================================================
    # 结果字段组装
    # ============================================================
    def _build_eval_fields(self, metric_abs_path, start_time, end_time, cost_seconds):
        """计算推演覆盖率与耗时字段，返回 dict 供 run() 组装 HTTP 响应体。

        coverage —— 全量目标算术平均覆盖率(%)，与基线侧同分母、同口径。
        """
        fields = {
            "startTime": _fmt_time(start_time),  #预案生成开始时间（可读）
            "endTime": _fmt_time(end_time),      #预案生成结束时间（可读）
            "costTime": _fmt_duration(cost_seconds),  #预案总耗时（可读）
        }

        summary = parse_metric_file(metric_abs_path, self._all_target_ids())
        fields["coverage"] = summary.coverage
        return fields

    # ============================================================
    # 预案文件
    # ============================================================

    def _load_plan_file_info(self):
        """读取并缓存预案文件（battle_scene + plan_result），全流程只读一次。"""
        if self._plan_file_info is None:
            processor = PlanFileProcess()
            self._plan_file_info = processor.read_plan_file_info_from_json(
                plan_id=self.conf.plan_id,
                file_path=self.conf.local_scene_path,
            )
        return self._plan_file_info

    def _all_target_ids(self):
        """场景全量目标 ID —— 推理与基线覆盖率平均的统一分母。"""
        try:
            return scene_target_ids(self._load_plan_file_info().battle_scene)
        except Exception as e:
            print(f"[EvalRunner] 目标集合读取失败，覆盖率分母退化为指标文件内目标: {e}")
            return None

    # ============================================================
    # 序列化
    # ============================================================

    def _save_dataclass_records_to_json(self, eval_records):
        """将 dataclass 评估数据安全序列化为 JSON。"""
        if not eval_records:
            return None

        task_id = getattr(self.conf, 'task_id', 'local_test')
        file_name = f"{task_id}_own_test.json"
        file_path = os.path.join(self.eval_records_dir, file_name)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(eval_records, f, ensure_ascii=False, indent=2,
                          default=_default_json_encoder)
            print(f"[EvalRunner] 原始推理评估采集数据存入文件: {file_path}")
            return os.path.abspath(file_path)
        except IOError as e:
            print(f"[EvalRunner] 原始推理评估采集数据保存失败: {e}")
            return None

    def _save_all_records_to_json(self, all_eval_data):
        """将所有局的评估数据合并序列化，生成平台格式的结果和指标 JSON。"""
        if not all_eval_data:
            print("[EvalRunner] 警告：没有评估数据可供保存，主动中断写入")
            return None, None

        single_episode_records = list(all_eval_data.values())[-1]
        safe_eval_data = _stringify_keys(single_episode_records)

        task_id = getattr(self.conf, 'task_id', 'local_test')
        result_file_name = f"{task_id}_eval_records.json"
        metric_file_name = f"{task_id}_eval_metric.json"
        result_file_path = os.path.join(self.eval_records_dir, result_file_name)
        metric_file_path = os.path.join(self.eval_records_dir, metric_file_name)
        result_abs_path = os.path.abspath(result_file_path)
        metric_abs_path = os.path.abspath(metric_file_path)

        try:
            processor = PlanFileProcess()
            battle_scene = self._load_plan_file_info().battle_scene

            print(f"[EvalRunner] 正在通过 PlanFileProcess 转换并保存推演结果")

            processor.write_model_inference_result_to_json(
                dict_model_inference_result=safe_eval_data,
                dest_path=result_abs_path
            )
            processor.write_model_inference_metric_to_json(
                dict_model_inference_result=safe_eval_data,
                battle_scene=battle_scene,
                dest_path=metric_abs_path
            )

            return result_abs_path, metric_abs_path

        except (OSError, ValueError, KeyError, AttributeError, ImportError) as e:
            print(f"[EvalRunner] 数据转换与保存失败 [磁盘IO或数据解析异常]: {e}")
            raise


class BaselineEvalRunner(BaseRunner):
    """基准推演评估运行器 —— 单独接口，只生成专家预案基线产物。

    基线来自 scene.json 的 ``splitQuduanResult``（专家规划），与模型推理无关，
    因此**不继承 EvalRunner**：EvalRunner.__init__ 会强制校验 load_dir、初始化
    GroupedEnvWrapper + QMIX agents 并加载权重，纯基线场景没有权重会直接
    FileNotFoundError，且无谓占用环境/网络资源。
    """

    def __init__(self, conf, push=None):
        super().__init__(conf, push)
        self.eval_records_dir = getattr(conf, 'eval_records_dir', './eval_records')
        os.makedirs(self.eval_records_dir, exist_ok=True)
        # 预案文件缓存（数 MB，全流程只读一次）
        self._plan_file_info = None

    def run(self):
        """基准推演评估主循环（同步返回基线字段）。"""
        return self._build_baseline_fields()

    def _load_plan_file_info(self):
        """读取并缓存预案文件（battle_scene + plan_result），全流程只读一次。"""
        if self._plan_file_info is None:
            processor = PlanFileProcess()
            self._plan_file_info = processor.read_plan_file_info_from_json(
                plan_id=self.conf.plan_id,
                file_path=self.conf.local_scene_path,
            )
        return self._plan_file_info

    def _build_baseline_fields(self):
        """生成专家预案基线并返回基线字段 dict，供 run() 组装 HTTP 响应体。

        返回:
            {"baselineTimeSeriesFile": records_path,
             "baselineEvalFile": metric_path,
             "baselineCoverage": coverage.coverage}

        失败/不可用时字段为 None——基线是旁路能力，任何异常都不应中断主流程，
        故整段兜底捕获。键集只声明一次：默认全 None，成功后原地覆盖对应值。
        """
        fields = {
            "timeSeriesFile": None,
            "evalFile": None,
            "coverage": None,
        }
        try:
            plan_file_info = self._load_plan_file_info()
            artifacts = BaselineEvaluator().generate_from_plan(
                plan_result=plan_file_info.plan_result,
                battle_scene=plan_file_info.battle_scene,
                out_dir=self.eval_records_dir,
                task_id=getattr(self.conf, 'task_id', 'local_test'),
            )
        except Exception as e:  # 基线是旁路能力，任何异常都不应中断主流程
            print(f"[BaselineEvalRunner] 基线数据生成失败（不中断主流程）: {e}")
            return fields

        if not artifacts.available:
            print(f"[BaselineEvalRunner] 基线数据不可用，跳过: {artifacts.reason}")
            return fields

        fields["timeSeriesFile"] = artifacts.records_path
        fields["evalFile"] = artifacts.metric_path
        fields["coverage"] = artifacts.coverage.coverage
        return fields