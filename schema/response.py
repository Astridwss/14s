"""
API 响应数据校验模型

定义所有出参的 Pydantic 结构，统一接口返回格式。
"""
from typing import Optional, Any
from pydantic import BaseModel, Field


# ============================================================
# 通用基类
# ============================================================

class BaseResponse(BaseModel):
    """最简响应（仅 code + message），用于暂停 / 恢复 / 终止等控制类接口"""

    code: int = Field(default=200, description="业务状态码")
    message: str = Field(default="ok", description="提示信息")


# ============================================================
# 训练任务下发响应
# ============================================================

class TaskStartData(BaseModel):
    """训练任务成功下发后返回的摘要数据"""

    task_id: str
    algorithm: str
    status: str = "starting"

                                                                  
class TaskStartResponse(BaseResponse):
    """训练任务下发接口的完整响应"""

    data: Optional[TaskStartData] = None


# ============================================================
# 推演评估响应
# ============================================================

class EvalResultData(BaseModel):
    """推演完成后的结果数据"""

    taskId: str
    algo: str
    status: str
    timeSeriesFile: Optional[str] = None
    evalFile: Optional[str] = None
    coverage: Optional[float] = None
    startTime: Optional[str] = None
    endTime: Optional[str] = None
    costTime: Optional[str] = None
    baselineTimeSeriesFile: Optional[str] = None
    baselineEvalFile: Optional[str] = None
    baselineCoverage: Optional[float] = None


class EvalResponse(BaseResponse):
    """推演评估接口的完整响应"""

    data: Optional[EvalResultData] = None


# ============================================================
# 任务控制响应（复用 BaseResponse 的别名）
# ============================================================

class TaskActionResponse(BaseResponse):
    """暂停 / 恢复 / 终止 等控制操作响应

    与 BaseResponse 字段完全一致，作为语义别名使用。
    """
