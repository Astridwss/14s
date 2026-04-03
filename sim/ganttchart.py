# 甘特图类

from typing import List, Dict, Tuple
from enum import Enum
from .two_dim_coordinate import TwoDimensionMinMax


class OverlappingNumberRequirement(Enum):
    OVERLAPPING_NUMBER_REQUIREMENT_LARGER = 1           # 大于输入的重数
    OVERLAPPING_NUMBER_REQUIREMENT_EQUAL = 2            # 等于输入的重数
    OVERLAPPING_NUMBER_REQUIREMENT_SMALLER = 3          # 小于输入的重数


class GanttChart:
    def __init__(self) -> None:
        self.str_body_name: str = ''                                          # 主体名称
        self.dict_activity: Dict[str, List[TwoDimensionMinMax]] = {}          # 活动图，<活动名称，活动数据序列>

    # 对甘特图整体进行交叠情况分析
    def overlapping_analysis_in_total(self) -> List[Tuple[TwoDimensionMinMax, List[str]]]:
        lst_result: List[Tuple[TwoDimensionMinMax, List[str]]] = []

        lst_total_data: List[TwoDimensionMinMax] = []
        for lst_activity_data in self.dict_activity.values():
            lst_total_data.extend(lst_activity_data)

        lst_piece_data: List[TwoDimensionMinMax] = []
        lst_data_pt: List[float] = []
        for data in lst_total_data:
            lst_data_pt.append(data.value_min)
            lst_data_pt.append(data.value_max)

        lst_data_pt.sort()

        former_pt = 0.0
        for i, data_pt in enumerate(lst_data_pt):
            if i == 0:
                if abs(data_pt - former_pt) > 1e-6:
                    lst_piece_data.append(TwoDimensionMinMax(former_pt, data_pt))
                    former_pt = data_pt

            if i == len(lst_data_pt) - 1:
                break

            if abs(lst_data_pt[i + 1] - former_pt) > 1e-6:
                lst_piece_data.append(TwoDimensionMinMax(former_pt, lst_data_pt[i + 1]))
                former_pt = lst_data_pt[i + 1]

        for piece_data in lst_piece_data:
            lst_activity_name: List[str] = []

            for str_activity_name, lst_activity_data in self.dict_activity.items():
                for activity_data in lst_activity_data:
                    if activity_data.whether_contains_interval(piece_data):
                        lst_activity_name.append(str_activity_name)
                        break

            lst_result.append((piece_data, lst_activity_name))

        return lst_result

    # 获取交叠重数满足输入的交叠重数要求的活动数据集合
    def overlapping_number_analysis_in_total(self, num: int, requirement: OverlappingNumberRequirement) -> List[Tuple[TwoDimensionMinMax, List[str]]]:
        lst_result: List[Tuple[TwoDimensionMinMax, List[str]]] = []

        lst_overlapping_analysis: List[Tuple[TwoDimensionMinMax, List[str]]] = self.overlapping_analysis_in_total()

        if requirement == OverlappingNumberRequirement.OVERLAPPING_NUMBER_REQUIREMENT_LARGER:
            for activity_data, lst_activity_name in lst_overlapping_analysis:
                if len(lst_activity_name) > num:
                    lst_result.append((activity_data, lst_activity_name))

        elif requirement == OverlappingNumberRequirement.OVERLAPPING_NUMBER_REQUIREMENT_EQUAL:
            for activity_data, lst_activity_name in lst_overlapping_analysis:
                if len(lst_activity_name) == num:
                    lst_result.append((activity_data, lst_activity_name))

        elif requirement == OverlappingNumberRequirement.OVERLAPPING_NUMBER_REQUIREMENT_SMALLER:
            for activity_data, lst_activity_name in lst_overlapping_analysis:
                if len(lst_activity_name) < num:
                    lst_result.append((activity_data, lst_activity_name))
        else:
            print('输入的交叠要求有误')

        return lst_result
