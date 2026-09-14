"""评估指标文件写出 —— sim 外重写版（修正覆盖率分母）。

``sim/plan_file_process.py`` 的 ``write_plan_metric_to_json`` 把覆盖率分母写成
``len(dict_target_traj_pt_info)``（轨迹**点数**），而分子 ``over_one_cover_time``
是 ``Σ(value_max - value_min)``（覆盖**秒数**）。两者单位不一致：轨迹点采样间隔
不是 1 秒时（稀疏/密集），点数 ≠ 秒数，导致覆盖率破百或偏低，且对短航迹目标
不友好。

本模块在 sim 外重写同一个函数（sim 是冻结内核，只 import 复用其甘特图类，不改动）：
唯一改动是分母换成**每个目标自身的轨迹时长（首末轨迹点时间差，秒）**，
与分子同口径。0/1/2/≥3 重那几列 ``vecCovernumCoverage`` 用同一个分母，一并修正。

其余逻辑（甘特图重叠分析、中断次数、JSON 结构、字段名）与 sim 原实现逐字一致，
保证前端读到的 ``evalFile`` 结构不变。
"""

import json
from typing import Dict, List, Tuple

from sim.datastruct import BattleScene, EquipmentDetectionTime, PlanResult
from sim.ganttchart import GanttChart, OverlappingNumberRequirement
from sim.two_dim_coordinate import TwoDimensionMinMax


def _target_trajectory_duration(target_id: str, battle_scene: BattleScene) -> int:
    """目标自身轨迹时长（秒）= 末轨迹点时间 - 首轨迹点时间。

    覆盖率分母必须与分子（覆盖秒数）同口径，都用「时间跨度（秒）」。
    空/单点轨迹回退 1s，避免除零；目标不在场景中（异常）回退 1800s 兜底。
    """
    target_info = battle_scene.dict_target_id_info.get(target_id)
    if target_info is None:
        return 1800  # 目标不在场景中（异常），沿用 sim 原兜底时长

    traj_times = sorted(target_info.dict_target_traj_pt_info.keys())
    if len(traj_times) >= 2:
        return max(traj_times[-1] - traj_times[0], 1)
    return 1  # 空/单点轨迹：按 1s 计，避免除零


def write_plan_metric_to_json(
    plan_result: PlanResult,
    battle_scene: BattleScene,
    dest_path: str,
) -> None:
    """将规划结果写成评估指标 JSON（覆盖率分母按目标自身轨迹时长）。

    与 ``sim/plan_file_process.py:411`` 的 ``write_plan_metric_to_json`` 结构一致，
    仅把分母从「轨迹点数」改为「轨迹时长（秒）」。
    """
    # equip→target→list 反转为 target→equip→list，按目标逐条评估
    dict_target_id_equip_id_detection_time: Dict[str, Dict[str, List[EquipmentDetectionTime]]] = {}

    for str_equip_id, dict_target_id_detection_time in plan_result.dict_equip_id_target_id_detection_time.items():
        for str_target_id, lst_detection_time in dict_target_id_detection_time.items():
            if str_target_id not in dict_target_id_equip_id_detection_time:
                dict_target_id_equip_id_detection_time[str_target_id] = {}

            dict_target_id_equip_id_detection_time[str_target_id][str_equip_id] = lst_detection_time

    data = {}
    data['schemeEvaluteResult'] = {}
    data['schemeEvaluteResult']['schemeEvaResult'] = []

    dict_eva_result = {}
    dict_eva_result['vecTargetEvaResult'] = []

    for str_target_id, dict_equip_id_detection_time in dict_target_id_equip_id_detection_time.items():
        # for循环，每个目标一个
        dict_target_eva_result = {}
        dict_target_eva_result['strTargetID'] = str_target_id

        gantt_chart = GanttChart()
        gantt_chart.str_body_name = str_target_id

        for str_equip_id, lst_detection_time in dict_equip_id_detection_time.items():
            for detection_time in lst_detection_time:
                if str_equip_id not in gantt_chart.dict_activity:
                    gantt_chart.dict_activity[str_equip_id] = []

                gantt_chart.dict_activity[str_equip_id].append(detection_time.time_range)

        lst_result_over_one_cover: List[Tuple[TwoDimensionMinMax, List[str]]] = gantt_chart.overlapping_number_analysis_in_total(0, OverlappingNumberRequirement.OVERLAPPING_NUMBER_REQUIREMENT_LARGER)
        lst_result_one_cover: List[Tuple[TwoDimensionMinMax, List[str]]] = gantt_chart.overlapping_number_analysis_in_total(1, OverlappingNumberRequirement.OVERLAPPING_NUMBER_REQUIREMENT_EQUAL)
        lst_result_two_cover: List[Tuple[TwoDimensionMinMax, List[str]]] = gantt_chart.overlapping_number_analysis_in_total(2, OverlappingNumberRequirement.OVERLAPPING_NUMBER_REQUIREMENT_EQUAL)
        lst_result_over_three_cover: List[Tuple[TwoDimensionMinMax, List[str]]] = gantt_chart.overlapping_number_analysis_in_total(2, OverlappingNumberRequirement.OVERLAPPING_NUMBER_REQUIREMENT_LARGER)

        # 分母：目标自身轨迹时长（秒），与分子 over_one_cover_time（覆盖秒数）同口径
        target_traj_total_time = _target_trajectory_duration(str_target_id, battle_scene)

        if target_traj_total_time > 0:
            dict_target_eva_result['vecContinuityResult'] = []
            dict_target_eva_result['vecReliabilityResult'] = []

            over_one_cover_time = 0
            interrupt_num = 0
            for i, (time_range, lst_equip_id) in enumerate(lst_result_over_one_cover):
                over_one_cover_time = over_one_cover_time + time_range.range()

                if i < len(lst_result_over_one_cover) - 1:
                    if abs(lst_result_over_one_cover[i + 1][0].value_min - time_range.value_max) > 1e-6:
                        interrupt_num += 1

            dict_continuity_result = {'dDetectCoverAge': over_one_cover_time / target_traj_total_time * 100.0,
                                      'regionType': 'ENUM_COMPREHENSIVE', 'uiInterruputNum': interrupt_num}
            dict_target_eva_result['vecContinuityResult'].append(dict_continuity_result)

            one_cover_time = 0
            for time_range, lst_equip_id in lst_result_one_cover:
                one_cover_time = one_cover_time + time_range.range()

            two_cover_time = 0
            for time_range, lst_equip_id in lst_result_two_cover:
                two_cover_time = two_cover_time + time_range.range()

            over_three_cover_time = 0
            for time_range, lst_equip_id in lst_result_over_three_cover:
                over_three_cover_time = over_three_cover_time + time_range.range()

            dict_reliability_result = {}
            dict_reliability_result['regionType'] = 'ENUM_COMPREHENSIVE'
            dict_reliability_result['vecCovernumCoverage'] = [
                {'dVal': (target_traj_total_time - over_one_cover_time) / target_traj_total_time * 100.0,
                 'uiCoverNum': 0},
                {'dVal': one_cover_time / target_traj_total_time * 100.0, 'uiCoverNum': 1},
                {'dVal': two_cover_time / target_traj_total_time * 100.0, 'uiCoverNum': 2},
                {'dVal': over_three_cover_time / target_traj_total_time * 100.0, 'uiCoverNum': 3}
            ]

            dict_target_eva_result['vecReliabilityResult'].append(dict_reliability_result)

            dict_eva_result['vecTargetEvaResult'].append(dict_target_eva_result)

    data['schemeEvaluteResult']['schemeEvaResult'].append(dict_eva_result)

    data['schemeEvaluteResult'] = json.dumps(data['schemeEvaluteResult'])
    with open(dest_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    return
