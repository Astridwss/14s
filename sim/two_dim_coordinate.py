# 二维数据类定义

from typing import List


# 最小最大二维数据结构
class TwoDimensionMinMax:
    def __init__(self, value_min: float = 0.0, value_max: float = 0.0) -> None:
        self.value_min: float = value_min
        self.value_max: float = value_max

    def clear(self) -> None:
        self.value_min = 0.0
        self.value_max = 0.0

    def __eq__(self, other) -> bool:
        if not isinstance(other, TwoDimensionMinMax):
            return NotImplemented

        return abs(self.value_min - other.value_min) < 1e-6 and abs(self.value_max - other.value_max) < 1e-6

    def __lt__(self, other) -> bool:
        if not isinstance(other, TwoDimensionMinMax):
            return NotImplemented

        return self.value_min < other.value_min

    def is_valid(self) -> bool:
        return not self.value_min > self.value_max

    def whether_contains_value(self, value: float) -> bool:
        return (not value < self.value_min) and (not value > self.value_max)

    def whether_contains_interval(self, other) -> bool:
        return self.whether_contains_value(other.value_min) and self.whether_contains_value(other.value_max)

    def mid_value(self) -> float:
        return (self.value_min + self.value_max) / 2

    def range(self) -> float:
        return self.value_max - self.value_min

    def whether_intersected(self, other) -> bool:
        return not (self.value_min > other.value_max or other.value_min > self.value_max)

    def intersected(self, other):
        result = TwoDimensionMinMax()

        if self.whether_intersected(other):
            result.value_min = other.value_min if other.value_min > self.value_min else self.value_min
            result.value_max = other.value_max if other.value_max < self.value_max else self.value_max

        return result

    def united(self, other):
        lst_result: List[TwoDimensionMinMax] = []

        if self.whether_intersected(other):
            united_interval = TwoDimensionMinMax()
            united_interval.value_min = other.value_min if other.value_min < self.value_min else self.value_min
            united_interval.value_max = other.value_max if other.value_max > self.value_max else self.value_max

            lst_result.append(united_interval)

        else:
            lst_result.append(other)

            current_interval = TwoDimensionMinMax(self.value_min, self.value_max)
            lst_result.append(current_interval)

        return lst_result

    def max_range(self, other):
        result = TwoDimensionMinMax()

        result.value_min = other.value_min if other.value_min < self.value_min else self.value_min
        result.value_max = other.value_max if other.value_max > self.value_max else self.value_max

        return result
