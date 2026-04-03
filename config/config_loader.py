# -*- coding: utf-8 -*-
"""
配置层 - 集中管理环境参数与训练参数。
不依赖任何其他层（仅 yaml、path、可选 torch）；各层通过 Config 读取参数。
优先级: 命令行覆盖 > YAML > 默认值
"""
import os
import yaml
from pathlib import Path
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError, Exception):
    TORCH_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

class Config:
    """
    统一配置类（唯一数据源，初始化后只读）：
    - 从 env.yaml 加载环境/仿真参数（如 max_episode_steps、n_radars、n_targets 等）
    - 从 algo.yaml 加载算法与训练参数
    - 支持命令行 overrides
    - 初始化并 _apply_overrides 后，下游（Env、Runner、Buffer）仅读取，不得反向修改或追加属性。
    - 与运行环境相关的维度（n_agents、obs_shape、state_shape）由 Env 以只读属性暴露，不写回 Config。
    """
    def __init__(self, mode: str = "train", config_dir: Optional[str] = None, **overrides):
        self.mode = mode
        self.train = (mode == "train")
        self.eval = (mode == "eval")
        if config_dir is None:
            config_dir = Path(__file__).parent
        else:
            config_dir = Path(config_dir)
        self._project_root = config_dir.parent
        self._load_env_config(config_dir / "env.yaml")
        self._load_algo_config(config_dir / "algo.yaml")
        self._compute_derived_params()
        self._apply_overrides(overrides)

    def _load_env_config(self, yaml_path: Path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            
        env = cfg.get('environment', {})
        for key, value in env.items():
            setattr(self, key, value)

    def _load_algo_config(self, yaml_path: Path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            
        # 1. 存下原始字典，留给 API 切换算法时“捞取默认值”用
        self._raw_algo_cfg = cfg 
        
        # 2. 读取当前算法名称（前后端统一命名为 algorithm）
        self.algorithm = cfg.get('algorithm', 'qmix').lower()
        
        # 3. 加载 common 公共参数
        for key, value in cfg.get('common', {}).items():
            setattr(self, key, value)
            
        # 4. 根据当前的 algorithm，动态加载对应的专属参数
        for key, value in cfg.get(self.algorithm, {}).items():
            setattr(self, key, value)
            
        # 5. 加载 train 模块
        train_section = cfg.get('train', {})
        train_defaults = {'max_episodes': 5000, 'save_frequency': 100}
        for key, value in {**train_defaults, **train_section}.items():
            setattr(self, key, value)
            
        # 6. 加载 eval 模块
        for key, value in cfg.get('eval', {}).items():
            if key == 'load_model' and self.mode == 'eval':
                setattr(self, key, value)
            else:
                setattr(self, key, value)

    def _compute_derived_params(self):
        model_dir = Path(self.model_dir)
        if not model_dir.is_absolute():
            model_dir = self._project_root / model_dir
        self.model_dir = str(model_dir)

        result_dir = Path(getattr(self, 'result_dir', './results'))
        if not result_dir.is_absolute():
            result_dir = self._project_root / result_dir
        self.result_dir = str(result_dir)
        
        eval_records_dir = Path(getattr(self, 'eval_records_dir', './eval_records'))
        if not eval_records_dir.is_absolute():
            eval_records_dir = self._project_root / eval_records_dir
        self.eval_records_dir = str(eval_records_dir)
        
        if TORCH_AVAILABLE:
            device_str = getattr(self, 'device', 'cpu') 
            self.device = torch.device(device_str)
        else:
            self.device = None

    def _apply_overrides(self, overrides):
        for key, value in overrides.items():
            if not hasattr(self, key):
                import warnings
                warnings.warn(f"Config 中无此参数，已忽略: {key}", UserWarning, stacklevel=2)
                continue
            orig = getattr(self, key)
            if type(orig) is bool and isinstance(value, str):
                value = value.lower() in ('1', 'true', 'yes', 'on')
            elif orig is not None and not isinstance(value, type(orig)):
                try:
                    value = type(orig)(value)
                except (ValueError, TypeError):
                    pass
            setattr(self, key, value)

    def update_from_api(self, api_request):
        """
        接收并解析平台方发来的 API 请求数据。前后端参数名一致，原样透传
        """
        # 1. 提取有效数据（过滤掉 FastAPI 自动补全的默认 None 值）
        if hasattr(api_request, "model_dump"):
            data = api_request.model_dump(exclude_unset=True)
        elif hasattr(api_request, "dict"):
            data = api_request.dict(exclude_unset=True)
        else:
            data = api_request if isinstance(api_request, dict) else {}

        # 2. 算法热切换与 YAML 默认值打底
        if "algorithm" in data:
            new_algo = data["algorithm"].lower()
            self.algorithm = new_algo
            
            # 从原始 yaml 字典中捞取 QMIX 默认值
            new_algo_defaults = getattr(self, "_raw_algo_cfg", {}).get(new_algo, {})
            for k, v in new_algo_defaults.items():
                setattr(self, k, v)

        # 3. 参数扁平化
        hypers = data.pop("hyperparameters", {})
        merged_params = {**data, **hypers}

        # 4. 动态覆盖（删除了所有映射逻辑，按原名注入）
        for api_key, value in merged_params.items():
            setattr(self, api_key, value)

        # 5. 重算派生参数
        self._compute_derived_params()

    def __repr__(self):
        mode_str = "TRAIN" if self.train else "EVAL"
        return f"Config(mode={mode_str})"
