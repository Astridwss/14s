"""
统一配置装配器 —— 单一入口，完整链路。

优先级链（由低到高）:  算法专属 YAML < 公共 YAML < 场景推导值 < 前端 API 参数

用法:
    assembler = ConfigAssembler(task_id, request_data, mode="train")
    conf = assembler.build()  # → RuntimeConfig（已锁定，不可变）
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from use_cases.config.config_types import EnvConfig, AlgorithmConfig, TrainingConfig, InfraConfig
from services.scene.scene_parser import extract_entities, compute_dimensions
from utils.task_control import TaskController
from utils.torch_device import resolve_device


# ============================================================
# 产物：纯数据容器
# ============================================================
class RuntimeConfig:
    """运行时配置 —— 纯数据容器。只承载属性，不包含任何 I/O 或解析逻辑。

    通过 ``ConfigAssembler.build()`` 创建，创建后自动锁定。
    """

    _locked: bool = False

    def __setattr__(self, key: str, value: Any) -> None:
        if self._locked:
            raise RuntimeError(
                f"RuntimeConfig 已锁定，禁止运行时修改: {key}。"
                f"请将动态值存入 Runner 实例变量。"
            )
        super().__setattr__(key, value)

    def _lock(self) -> None:
        """锁定配置，禁止后续写入。"""
        self._locked = True

    def __repr__(self) -> str:
        algo = getattr(self, "algorithm", "?")
        mode = getattr(self, "mode", "?")
        return f"RuntimeConfig(algorithm={algo}, mode={mode})"


# ============================================================
# 装配器
# ============================================================
class ConfigAssembler:
    """统一配置装配器。

    一条链路完成: YAML基线 → 场景下载+解析 → 三源合并 → 路径推导 → 锁定。

    四层数据源（优先级由低到高）:
      - 算法专属 YAML: algo.yaml 中当前算法段的参数（最低，兜底中的兜底）
      - 公共 YAML:     algo.yaml common/infra/train/eval 段
      - 场景推导值:     从场景文件解析的智能体/动作/观测维度
      - API 参数:       前端显式传入的值（最高）

    合并策略: 后层覆盖前层，API 传了什么就用什么，不传则逐级下沉取默认值。
    """

    # YAML 文件相对于本模块的路径
    _CONFIG_DIR = Path(__file__).resolve().parent

    def __init__(self, task_id: str, request_data, mode: str):
        """
        Args:
            task_id:      任务唯一 ID
            request_data: Pydantic 请求模型 或 dict（含 hyperparameters 嵌套）
            mode:         "train" | "eval"
        """
        self._task_id = task_id
        self._request = request_data
        self._mode = mode

        # ---- 四层数据源 ----
        self._algo_common: dict = {}        # algo.yaml common 段
        self._algo_mode: dict = {}          # algo.yaml train / eval 段
        self._algo_infra: dict = {}         # algo.yaml infra 段
        self._algo_cfg: dict = {}           # algo.yaml 全文（用于按需取算法段）
        self._api_params: dict = {}         # 前端传参（展平后）
        self._scene_dims: dict = {}         # 场景推导维度

        # ---- 中间状态 ----
        self._scene_path: str = ""
        self._merged: dict = {}

    # ============================================================
    # 公开入口
    # ============================================================

    def build(self) -> RuntimeConfig:
        """执行完整装配链路，返回已锁定的 RuntimeConfig。(YAML 加载→场景下载→多层合并→路径解析→冻结）"""
        self._load_yaml_baseline()
        self._flatten_api_params()
        self._download_scene()
        self._print_request_config_params()
        self._parse_scene()
        self._merge()
        self._resolve_paths()
        self._setup_flags()
        return self._freeze()

    # ============================================================
    
    # 步骤 1: YAML 基线 —— 读取本地兜底配置
    # ============================================================

    def _load_yaml_baseline(self) -> None:
        """加载 algo.yaml，拆分为 common / mode / infra / 全文。"""
        algo_raw = self._read_yaml(self._CONFIG_DIR / "algo.yaml")
        self._algo_cfg = algo_raw                          # 保留全文，后续按算法名取段
        self._algo_common = dict(algo_raw.get("common", {}))
        self._algo_mode = dict(algo_raw.get(self._mode, {}))
        self._algo_infra = dict(algo_raw.get("infra", {}))

    @staticmethod
    def _read_yaml(path: Path) -> dict:
        """读取单个 YAML 文件，统一处理 I/O 与语法异常。"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except (IOError, OSError) as e:
            raise RuntimeError(f"配置文件读取失败 [I/O]: {path}") from e
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件语法错误: {path}") from e

    # ============================================================
    # 步骤 2: API 参数扁平化
    # ============================================================

    def _flatten_api_params(self) -> None:
        """将 Pydantic 请求模型展平为单层字典。

        处理:
          - 兼容 Pydantic v1 (.dict) / v2 (.model_dump) 及裸 dict
          - 提取嵌套的 ``hyperparameters`` 字段并展平
          - 剔除值为 None 的键（表示前端未传，由下层兜底）
        """
        data = self._request

        # 归一化为 dict
        if hasattr(data, "model_dump"):
            data = data.model_dump(exclude_unset=True)
        elif hasattr(data, "dict"):
            data = data.dict(exclude_unset=True)
        elif not isinstance(data, dict):
            data = {}

        # 展平嵌套的 hyperparameters
        hypers = data.pop("hyperparameters", {}) or {}
        flat = {**data, **hypers}

        # 剔除 None：未传字段不参与覆盖
        self._api_params = {k: v for k, v in flat.items() if v is not None}

    # ============================================================
    # 步骤 3: 场景下载
    # ============================================================

    def _download_scene(self) -> None:
        """读取 scene_url 参数携带的场景文件本地路径，未传则回退到本地默认路径。

        说明：scene_url 现在承载的是平台侧已下载好的本地文件路径，
        直接使用该路径，不再二次下载（原 SceneDownloader.download_file 逻辑保留未删）。
        """
        scene_url = self._api_params.get("scene_url")

        if scene_url:
            self._scene_path = scene_url
            print(f"[ConfigAssembler] 使用平台下发的场景文件路径: {scene_url}")
        else:
            project_root = os.environ.get("PROJECT_ROOT", ".")
            self._scene_path = os.path.join(
                project_root, "scenarios", self._task_id, "scene.json"
            )

    # ============================================================
    # 步骤 4: 场景解析 —— 实体提取 + 维度推导
    # ============================================================

    def _parse_scene(self) -> None:
        """从场景文件提取实体元数据，推导智能体/动作/观测维度。"""
        plan_id = self._api_params.get("plan_id")
        if plan_id is None:
            raise ValueError(
                "[ConfigAssembler] 缺少 plan_id：训练/推演/基线请求必须显式传入 plan_id。"
            )

        n_radars, n_satellites, n_targets, radar_keys, target_keys, sat_keys = extract_entities(self._scene_path, plan_id)

        dims = compute_dimensions(n_radars, n_satellites, n_targets)

        self._scene_dims = {
            "plan_id": int(plan_id),
            "local_scene_path": self._scene_path,
            "n_radars": n_radars,
            "n_satellites": n_satellites,
            "n_targets": n_targets,
            "radar_keys": radar_keys,
            "satellites_keys": sat_keys,
            "target_keys": target_keys,
            "n_agents": dims["n_agents"],
            "n_ld": dims["n_ld"],
            "n_wx": dims["n_wx"],
            "n_actions": dims["n_actions"],
            "ld_n_actions": dims["ld_n_actions"],
            "obs_shape": dims["obs_shape"],
            "state_shape": dims["state_shape"],
            "radar_obs_dim": dims["radar_obs_dim"],
        }

    # ============================================================
    # 步骤 5: 三层合并
    # ============================================================

    def _merge(self) -> None:
        """按优先级合并四层数据源。

        合并链（由低到高，后者覆盖前者）:
          1. 算法专属 YAML  —— 兜底中的兜底（如 qmix 段的 learning_rate）
          2. 公共 YAML       —— algo.yaml common / mode / infra 段
          3. 场景推导值       —— 从场景文件提取的实体维度
          4. API 前端参数    —— 用户显式传入的值（最高优先级）
        """
        # 底层: 公共 YAML（algo common + algo mode + algo infra）
        merged = {}
        merged.update(self._algo_common)
        merged.update(self._algo_mode)
        merged.update(self._algo_infra)
        # 场景中层覆盖
        merged.update(self._scene_dims)
        # API 顶层覆盖
        merged.update(self._api_params)

        # 根据合并后的最终算法名，加载对应算法段的 YAML 默认值
        # （仅填充 API 未传的参数，已在 merged 中的不会被覆盖）
        # 前端 algorithm 可能是 "QMIX"/"PPO" 等任意字符串；统一小写后，若不在
        # algo.yaml 的真实段名中，则回退到 qmix 段（RL/推演 runner 均为 QMIX 专用），
        # 避免段名不匹配导致 drqn_hidden_dim 等网络超参数缺失。
        algo_name = str(merged.get("algorithm", "qmix")).lower().strip()
        if algo_name not in self._algo_cfg:
            algo_name = "qmix"
        algo_section = self._algo_cfg.get(algo_name, {})
        for k, v in algo_section.items():
            if k not in merged:
                merged[k] = v

        # 剔除残留的 None 值
        self._merged = {k: v for k, v in merged.items() if v is not None}

    # ============================================================
    # 步骤 6: 路径解析 + 衍生值
    # ============================================================

    def _resolve_paths(self) -> None:
        """将相对路径转为绝对路径，创建必要目录，转换衍生类型（如 device）。"""
        project_root = os.environ.get("PROJECT_ROOT", ".")

        if self._mode == "train":
            # model_dir
            model_dir = self._merged.get("model_dir", "./models")
            if not os.path.isabs(model_dir):
                model_dir = os.path.join(project_root, model_dir)
            model_dir = os.path.normpath(os.path.join(model_dir, self._task_id))
            os.makedirs(model_dir, exist_ok=True)
            self._merged["model_dir"] = model_dir

            # result_dir
            result_dir = self._merged.get("result_dir", "./results")
            if not os.path.isabs(result_dir):
                result_dir = os.path.join(project_root, result_dir)
            result_dir = os.path.normpath(os.path.join(result_dir, self._task_id))
            os.makedirs(result_dir, exist_ok=True)
            self._merged["result_dir"] = result_dir
        else:
            # eval_records_dir
            eval_dir = self._merged.get(
                "eval_records_dir", "./eval_records"
            )
            if not os.path.isabs(eval_dir):
                eval_dir = os.path.join(project_root, eval_dir)
            eval_dir = os.path.normpath(os.path.join(eval_dir, self._task_id))
            os.makedirs(eval_dir, exist_ok=True)
            self._merged["eval_records_dir"] = eval_dir

        # device 字符串 → torch.device（npu 需先注册 torch_npu 后端；torch 不可用则保持字符串）
        device_str = self._merged.get("device", "cpu")
        self._merged["device"] = resolve_device(device_str)

    # ============================================================
    # 步骤 7: 任务控制标志位
    # ============================================================

    def _setup_flags(self) -> None:
        """设置暂停/终止标志文件的绝对路径（通过 TaskController 统一入口）。"""
        self._merged["pause_flag_file"] = TaskController.pause_path(self._task_id)
        self._merged["terminate_flag_file"] = TaskController.terminate_path(self._task_id)

    # ============================================================
    # 步骤 8: 冻结为不可变配置对象
    # ============================================================

    def _freeze(self) -> RuntimeConfig:
        """将所有合并后的键值写入 RuntimeConfig 并锁定。

        直接传 _merged dict 构造 typed config（消除 from_config(RuntimeConfig) 循环）。
        """
        merged = self._merged
        merged["mode"] = self._mode
        merged["task_id"] = self._task_id

        config = RuntimeConfig()

        # 模式标识（唯一来源 config.mode，供 InfraConfig/Runner 区分 train/eval/baseline；
        # 不再设置 config.train/config.eval 布尔值 —— 二者会与下方聚焦子配置冲突）
        config.mode = self._mode
        config.task_id = self._task_id

        # 批量写入所有合并后的参数（向后兼容：Runner 中 conf.xxx 直接访问）
        for key, value in merged.items():
            setattr(config, key, value)

        # ---- 聚焦子配置（直接从 merged dict 构造，不走 RuntimeConfig） ----
        config.env = EnvConfig.from_config(merged)
        config.algo = AlgorithmConfig.from_config(merged)
        config.train = TrainingConfig.from_config(merged)
        config.infra = InfraConfig.from_config(merged)

        config._lock()
        return config

    # ============================================================
    # 步骤 9: 打印接口下发参数
    # ============================================================

    def _print_request_config_params(self) -> None:
        """打印前端实际下发的参数（已展平 + 剔除未传/None 后的有效值）。

        数据源 _api_params 由 _flatten_api_params 归一化：嵌套的 hyperparameters 被展平到
        顶层，且用 exclude_unset 只保留前端显式传入的字段 —— 这就是「接口下发了什么」。
        （与 utils.config_printer.print_config_params 互补：那边打印最终生效的 merged 配置。）
        """
        params = self._api_params
        print("\n" + "=" * 62)
        print(f"  接口下发参数 | task_id={self._task_id} | mode={self._mode} | 共 {len(params)} 项")
        print("=" * 62)
        if not params:
            print("  (前端未下发任何显式参数，全部走 YAML / 场景推导默认值)")
        for k, v in sorted(params.items()):
            print(f"  {k:<28} = {v!r}")
        print()

