"""services.evaluation — 方案级离线评估（基线生成、覆盖率口径）。

与 ``services.scene`` 的区别：scene 是喂给 RL 环境的场景积木（env_wrapper /
action / state / reward），面向 step 循环；本包面向**整局方案产物**的评估，
被推演 Runner、离线对比脚本共用。

覆盖率口径统一在 :mod:`services.evaluation.coverage`，基线 A（专家预案）
的生成在 :mod:`services.evaluation.baseline`。
"""

from services.evaluation.coverage import CoverageSummary, parse_metric_file
from services.evaluation.metric_writer import write_plan_metric_to_json
from services.evaluation.baseline import (
    BaselineArtifacts, BaselineEvaluator, scene_target_ids,
)

__all__ = [
    "CoverageSummary", "parse_metric_file", "write_plan_metric_to_json",
    "BaselineArtifacts", "BaselineEvaluator", "scene_target_ids",
]
