"""基线评估 —— 把预案里的专家规划结果转成与推理侧同构的评估产物。

基线 A（专家预案）来源：``scene.json → planInfoList[].splitQuduanResult``，
由 ``PlanFileProcess.read_plan_result_from_json`` 解析为 ``PlanResult``。

与推理侧的结构一致性靠「在 PlanResult 这一层汇合」保证——两条路走同一对写函数::

    推理: eval_records(Dict[time, List[AgentActionCommand]])
                │ write_model_inference_result_to_plan_result(time_cut=20)
                ▼
           PlanResult ──┬──► write_plan_result_to_json  → *_records.json
                        │
           PlanResult ──┴──► write_plan_metric_to_json  → *_metric.json
                ▲
                │ read_plan_result_from_json
    基线: scene.json splitQuduanResult

⚠️ 不要试图把基线塞进 ``EvalRunner._save_all_records_to_json``：那个入口的入参是
模型逐步动作流 ``Dict[time, List[AgentActionCommand]]``，而基线读出来已经是
``PlanResult``，硬塞会在 sim/plan_file_process.py:208 因缺少 ``.time`` 属性崩溃。
"""

import os
from dataclasses import dataclass
from typing import Optional

from sim import PlanFileProcess

from services.evaluation.coverage import CoverageSummary, parse_metric_file
from services.evaluation.metric_writer import write_plan_metric_to_json


@dataclass
class BaselineArtifacts:
    """基线评估产物。

    ``available=False`` 表示预案里没有专家规划结果（常见于 mock 场景），
    此时不应推送——推一个 0% 覆盖率会让后端误判「模型提升无穷大」。
    """

    available: bool = False
    reason: str = ""
    records_path: str = ""
    metric_path: str = ""
    coverage: Optional[CoverageSummary] = None


class BaselineEvaluator:
    """专家预案基线评估器。

    用法::

        evaluator = BaselineEvaluator()
        artifacts = evaluator.generate(
            scene_path="./scene.json", plan_id=867,
            out_dir="./eval_records", task_id="TASK001",
        )
        if artifacts.available:
            print(artifacts.coverage.describe())
    """

    RECORDS_SUFFIX = "_baseline_records.json"
    METRIC_SUFFIX = "_baseline_metric.json"

    def __init__(self, processor: Optional[PlanFileProcess] = None):
        self._processor = processor or PlanFileProcess()

    # ============================================================
    # 公开 API
    # ============================================================

    def generate(
        self, scene_path: str, plan_id: int, out_dir: str, task_id: str,
    ) -> BaselineArtifacts:
        """从预案文件读取基线并生成评估产物。

        调用方若已持有 ``PlanFileInfo``，请改用 :meth:`generate_from_plan`
        避免重复读盘（预案文件通常数 MB）。
        """
        if not scene_path or not os.path.exists(scene_path):
            return BaselineArtifacts(
                available=False,
                reason=f"预案文件不存在: {scene_path}",
            )

        plan_file_info = self._processor.read_plan_file_info_from_json(
            plan_id=plan_id, file_path=scene_path,
        )
        return self.generate_from_plan(
            plan_result=plan_file_info.plan_result,
            battle_scene=plan_file_info.battle_scene,
            out_dir=out_dir,
            task_id=task_id,
        )

    def generate_from_plan(
        self, plan_result, battle_scene, out_dir: str, task_id: str,
    ) -> BaselineArtifacts:
        """由已读出的 PlanResult + BattleScene 生成基线评估产物。"""
        if plan_result is None or not plan_result.dict_equip_id_target_id_detection_time:
            return BaselineArtifacts(
                available=False,
                reason="预案缺少 splitQuduanResult（专家规划结果），无基线可比",
            )

        os.makedirs(out_dir, exist_ok=True)
        records_path = os.path.abspath(
            os.path.join(out_dir, f"{task_id}{self.RECORDS_SUFFIX}")
        )
        metric_path = os.path.abspath(
            os.path.join(out_dir, f"{task_id}{self.METRIC_SUFFIX}")
        )

        # 与推理侧共用的两个写函数，保证 JSON 结构逐键同构。
        # 指标文件改用 sim 外重写版：覆盖率分母按目标自身轨迹时长（秒），
        # 与推理侧同口径，修正 sim 原实现「分母=轨迹点数」导致的覆盖率破百。
        self._processor.write_plan_result_to_json(
            plan_result=plan_result, dest_path=records_path,
        )
        write_plan_metric_to_json(
            plan_result=plan_result, battle_scene=battle_scene,
            dest_path=metric_path,
        )

        summary = parse_metric_file(
            metric_path, all_target_ids=scene_target_ids(battle_scene),
        )
        print(f"[Baseline] 基线评估产物已生成: {summary.describe()}")

        return BaselineArtifacts(
            available=True,
            records_path=records_path,
            metric_path=metric_path,
            coverage=summary,
        )


def scene_target_ids(battle_scene):
    """取场景全量目标 ID —— 覆盖率平均的统一分母。

    推理侧与基线侧必须用同一个分母，否则平均覆盖率不可比。
    """
    if battle_scene is None:
        return None
    return list(getattr(battle_scene, "dict_target_id_info", {}) or {})
