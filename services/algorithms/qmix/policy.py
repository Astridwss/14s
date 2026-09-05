import glob
import os
from typing import List

import torch
import torch.nn as nn

from .nn import DRQN, QMIXNET, DualHeadDRQN
from .hqmix_nn import LowerMixer, UpperMixer
from services.scene.state.group_slicer import (
    build_group_local_states,
    build_global_pooled_state,
    group_local_state_dim,
    global_pooled_state_dim,
)
from utils.weight_naming import prefix_from_conf, weight_file_name
from services.scene.scene_constants import LD_CAPACITY


# ============================================================
# 共享基类 —— DRQN 构造、输入构建、hidden 管理
# ============================================================

class _BasePolicy:
    """QMIX / HQMIX 共享基类。

    子类只需关注混频器构造 + learn() + 持久化。
    """

    def __init__(self, conf, dual_head=False):
        self.conf = conf
        self.device = conf.device
        self.n_actions = conf.n_actions
        self.n_agents = conf.n_agents
        self.n_ld = getattr(conf, 'n_ld', 0) or 0
        self.n_wx = getattr(conf, 'n_wx', 0) or 0
        self.ld_n_actions = getattr(conf, 'ld_n_actions', 0) or getattr(conf, 'n_targets', 0)
        self.eval_hidden = None
        self.model_dir = getattr(conf, 'model_dir', None)
        # 权重文件名前缀（实体数 + 分组数）；推演侧可用恢复后的 group_size 覆盖
        self.weight_prefix = prefix_from_conf(conf)

        # DRQN 输入维度
        obs_dim = conf.obs_shape[0] if isinstance(conf.obs_shape, tuple) else conf.obs_shape
        input_shape = obs_dim
        if conf.last_action:
            input_shape += self.n_actions
        if getattr(conf, 'reuse_network', True):
            input_shape += self.n_agents

        if dual_head:
            self.eval_drqn_net = DualHeadDRQN(input_shape, conf, self.ld_n_actions, self.n_actions).to(self.device)
            self.target_drqn_net = DualHeadDRQN(input_shape, conf, self.ld_n_actions, self.n_actions).to(self.device)
        else:
            self.eval_drqn_net = DRQN(input_shape, conf).to(self.device)
            self.target_drqn_net = DRQN(input_shape, conf).to(self.device)
        self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())

        # 子类在 __init__ 中调用 self._make_optimizer(head_params)
        self.eval_parameters = []
        self.optimizer = None
        # 持有主干参数组的那个优化器（QMIX 为 optimizer，HQMIX 为 opt_ld）
        self._trunk_optimizer = None

    # ============================================================
    # 优化器 —— 主干与输出头分组，支持热启动后单独降速
    # ============================================================

    def trunk_parameters(self):
        """共享主干参数（fc1 / layer_norm / rnn），与 TRUNK_PREFIXES 一一对应。

        DRQN 与 DualHeadDRQN 都有这三个子模块，故两条路径通用。
        """
        net = self.eval_drqn_net
        return (
            list(net.fc1.parameters())
            + list(net.layer_norm.parameters())
            + list(net.rnn.parameters())
        )

    def _make_optimizer(self, head_params):
        """构造分组优化器：主干一组、输出头+混频器一组。

        **两组初始 lr 相同**，因此与不分组时的行为完全等价——纯 RL 从零训练
        不受任何影响。只有在 IL 热启动成功后，``load_drqn_state`` 才会调用
        :meth:`set_trunk_lr_scale` 把主干那一组单独调慢，避免随机初始化的输出头
        把大幅度、随机方向的梯度反冲进预训练好的主干。

        Returns:
            (optimizer, flat_params) —— flat_params 供 clip_grad_norm_ 使用，
            梯度裁剪需要的是扁平参数列表而非参数组。
        """
        trunk = self.trunk_parameters()
        lr = self.conf.learning_rate
        optimizer = torch.optim.RMSprop([
            {'params': trunk, 'lr': lr},        # 下标必须为 TRUNK_GROUP
            {'params': head_params, 'lr': lr},
        ])
        self._trunk_optimizer = optimizer
        return optimizer, trunk + head_params

    def set_trunk_lr_scale(self, scale: float) -> None:
        """把主干参数组的学习率缩放为 ``base_lr * scale``。

        仅供 IL 热启动后调用。scale >= 1.0 视为不启用。
        """
        if scale is None or scale >= 1.0:
            return
        if self._trunk_optimizer is None:
            print("[WarmStart] 优化器尚未构造，跳过主干学习率调整")
            return

        group = self._trunk_optimizer.param_groups[self.TRUNK_GROUP]
        base_lr = self.conf.learning_rate
        old_lr = group['lr']
        group['lr'] = base_lr * scale
        print(f"[WarmStart] 主干学习率 {old_lr:.2e} → {group['lr']:.2e} "
              f"({scale}×)，输出头与混频器保持 {base_lr:.2e}")

    # ============================================================
    # Hidden 状态
    # ============================================================

    def init_hidden(self, batch_size):
        self.eval_hidden = torch.zeros(
            (batch_size, self.n_agents, self.conf.drqn_hidden_dim),
        ).to(self.device)

    # ============================================================
    # DRQN 输入构建
    # ============================================================

    def _build_inputs(self, batch, t, batch_size):
        """返回 (inputs, next_inputs) —— QMIX.learn() 标准循环使用。"""
        obs = torch.tensor(batch['obs'][:, t], dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(batch['next_obs'][:, t], dtype=torch.float32).to(self.device)

        inputs = [obs]
        next_inputs = [next_obs]

        if self.conf.last_action:
            if t == 0:
                inputs.append(torch.zeros((batch_size, self.n_agents, self.n_actions)).to(self.device))
            else:
                inputs.append(torch.tensor(batch['actions_onehot'][:, t - 1], dtype=torch.float32).to(self.device))
            next_inputs.append(torch.tensor(batch['actions_onehot'][:, t], dtype=torch.float32).to(self.device))

        if getattr(self.conf, 'reuse_network', True):
            agent_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
            inputs.append(agent_id)
            next_inputs.append(agent_id)

        inputs = torch.cat([x.reshape(batch_size * self.n_agents, -1) for x in inputs], dim=1)
        next_inputs = torch.cat([x.reshape(batch_size * self.n_agents, -1) for x in next_inputs], dim=1)
        return inputs, next_inputs

    def _build_inputs_for_seq(self, batch, t, batch_size, obs_key='obs', target=False):
        """单步输入构建 —— HQMIX._drqn_seq_dual() 使用。

        eval (target=False): last_action = a_{t-1}（t=0 时补零），
          与 obs_t 对齐，即「上一步动作 → 本步观测」。
        target (target=True): last_action = a_t，
          与 next_obs_t(=obs_{t+1}) 对齐，即「本步动作 → 下一步观测」。
        与 QMIX._build_inputs 的 next_inputs 约定一致。
        """
        obs = torch.tensor(batch[obs_key][:, t], dtype=torch.float32).to(self.device)
        parts = [obs]

        if self.conf.last_action:
            if target:
                parts.append(torch.tensor(batch['actions_onehot'][:, t], dtype=torch.float32).to(self.device))
            elif t == 0:
                parts.append(torch.zeros(batch_size, self.n_agents, self.n_actions).to(self.device))
            else:
                parts.append(torch.tensor(batch['actions_onehot'][:, t - 1], dtype=torch.float32).to(self.device))

        if getattr(self.conf, 'reuse_network', True):
            agent_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
            parts.append(agent_id)

        return torch.cat([x.reshape(batch_size * self.n_agents, -1) for x in parts], dim=1)

    # ============================================================
    # 权重加载（IL → RL 热启动）
    # ============================================================

    # 共享主干的参数前缀：fc1(观测编码) → layer_norm → rnn(GRUCell 时序记忆)
    # 单头 DRQN 与 DualHeadDRQN 在这三层上键名与形状完全一致，是唯一可迁移的部分。
    TRUNK_PREFIXES = ('fc1.', 'layer_norm.', 'rnn.')

    # 主干在 _make_optimizer 构造的 param_groups 中的下标
    TRUNK_GROUP = 0

    # 热启动后主干学习率的缩放系数（1.0 = 不降速）
    DEFAULT_WARM_START_TRUNK_LR_SCALE = 0.1

    def load_drqn_state(self, load_dir):
        """IL 预训练 → RL 热启动：只迁移共享主干（fc1 / layer_norm / rnn）。

        输出头不迁移，因为结构与语义都不同：
          - IL 单头 DRQN 的 fc2 是 CE logits，尺度任意、被 softmax 耦合，
            只有相对大小有意义；
          - H-QMIX 的 fc_ld 是 Q 值，零点有绝对意义（clamp(min=0) 后 top-K
            求和进混频器），fc_wx 则是另一套单选 logits。
        直接搬会让初始 Q 量级远超 reward 尺度，巨额 TD error 反冲把主干冲垮。
        故双头保持随机初始化从头学，只保留「观测 → 隐状态」的编码能力。

        混频器同样不迁移（IL 为 QMIXNET，H-QMIX 为 LowerMixer/UpperMixer）。
        """
        drqn_path = self._locate_drqn_weight(load_dir)
        checkpoint = torch.load(
            drqn_path, map_location=self.device, weights_only=True,
        )

        trunk = {
            k: v for k, v in checkpoint.items()
            if k.startswith(self.TRUNK_PREFIXES)
        }
        if not trunk:
            raise ValueError(
                f"[WarmStart] 权重文件不含任何主干参数 "
                f"{self.TRUNK_PREFIXES}，疑似文件损坏或结构不符: {drqn_path}"
            )

        self._assert_trunk_compatible(trunk, drqn_path)

        missing, unexpected = self.eval_drqn_net.load_state_dict(
            trunk, strict=False,
        )
        if unexpected:
            raise ValueError(
                f"[WarmStart] 主干过滤后仍有多余参数 {list(unexpected)}，"
                f"请检查 TRUNK_PREFIXES 是否与网络结构同步: {drqn_path}"
            )

        self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())

        skipped = [k for k in checkpoint if not k.startswith(self.TRUNK_PREFIXES)]
        n_params = sum(v.numel() for v in trunk.values())
        print(f"[WarmStart] 已迁移 IL 主干 {len(trunk)} 个张量 / {n_params} 个参数: {drqn_path}")
        print(f"[WarmStart] 保持随机初始化的输出头: {list(missing)}")
        if skipped:
            print(f"[WarmStart] 已丢弃权重文件中的输出头: {skipped}")

        # 主干降速：输出头此刻是随机的，其梯度会以「大幅度 × 随机方向」反冲进
        # 刚迁移好的主干。降速让输出头先学会使用现成表征，避免预训练被抹平。
        self.set_trunk_lr_scale(getattr(
            self.conf, 'warm_start_trunk_lr_scale',
            self.DEFAULT_WARM_START_TRUNK_LR_SCALE,
        ))

    def _locate_drqn_weight(self, load_dir):
        """定位 DRQN 权重文件，优先取与当前场景前缀一致的那份。

        权重文件名带「实体数 + 分组数」前缀，同一目录可能存在多个场景的权重，
        盲取第一个会静默加载错场景的权重，故优先精确匹配。
        """
        matches = glob.glob(os.path.join(load_dir, '*_drqn.pkl'))
        if not matches:
            raise FileNotFoundError(
                f"[WarmStart] 未找到 DRQN 权重 (*_drqn.pkl): {load_dir}"
            )

        expected = weight_file_name(self.weight_prefix, 'drqn')
        for path in matches:
            if os.path.basename(path) == expected:
                return path

        if len(matches) > 1:
            print(f"[WarmStart] 警告：目录内有 {len(matches)} 份 DRQN 权重且均不匹配"
                  f"当前场景前缀 {expected}，回退取 {os.path.basename(matches[0])}")
        return matches[0]

    def _assert_trunk_compatible(self, trunk, drqn_path):
        """逐 key 校验主干形状，不一致时给出可操作的报错。

        最常见的失配原因是 IL 与 RL 跑在不同场景上（雷达/卫星/目标数不同
        → input_shape 不同），此时直接 load 会抛出难以定位的 size mismatch。
        """
        current = self.eval_drqn_net.state_dict()

        for key, tensor in trunk.items():
            if key not in current:
                raise ValueError(
                    f"[WarmStart] 权重文件含当前网络没有的主干参数 '{key}'，"
                    f"IL 与 RL 的网络结构不一致: {drqn_path}"
                )
            if tuple(current[key].shape) != tuple(tensor.shape):
                raise ValueError(
                    f"[WarmStart] IL 预训练权重与当前场景不匹配。\n"
                    f"  参数 : {key}\n"
                    f"  当前 : {tuple(current[key].shape)}\n"
                    f"  权重 : {tuple(tensor.shape)}\n"
                    f"  权重文件: {drqn_path}\n"
                    f"  {self._shape_hint(key, current[key], tensor)}\n"
                    f"  IL 与 RL 必须在同一场景（相同雷达/卫星/目标数）上训练。"
                )

    def _shape_hint(self, key, current_tensor, ckpt_tensor):
        """把形状差异翻译成场景维度的人话提示。"""
        if key != 'fc1.weight':
            return "提示: 请确认两侧使用了相同的 drqn_hidden_dim。"
        return (
            f"提示: input_shape 当前={current_tensor.shape[1]}，"
            f"权重={ckpt_tensor.shape[1]}。"
            f"input_shape = obs_dim + n_actions(last_action) + n_agents(reuse_network)，"
            f"三者任一不同都会导致此错。"
        )

    # ============================================================
    # 子类覆盖
    # ============================================================

    def learn(self, batch, train_step, epsilon=None):
        raise NotImplementedError

    def save_model(self, episode_idx=None):
        raise NotImplementedError

    def load_state(self, load_dir):
        raise NotImplementedError


# ============================================================
# QMIX —— 标准单层混频
# ============================================================

class QMIX(_BasePolicy):
    """等价于 EPyMARL 中的 QLearner (src/learners/q_learner.py)"""

    def __init__(self, conf):
        super().__init__(conf)

        self.eval_qmix_net = QMIXNET(self.conf).to(self.device)
        self.target_qmix_net = QMIXNET(self.conf).to(self.device)
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        # 主干(fc1/layer_norm/rnn) 与 输出头(fc2)+混频器 分成两个参数组，
        # 初始 lr 相同 => 与不分组时等价；热启动后主干可单独降速。
        head_params = (
            list(self.eval_drqn_net.fc2.parameters())
            + list(self.eval_qmix_net.parameters())
        )
        self.optimizer, self.eval_parameters = self._make_optimizer(head_params)

    # ============================================================
    # 训练
    # ============================================================

    def learn(self, batch, train_step, epsilon=None):
        """EPyMARL 的 q_learner.py TD-error 计算逻辑"""
        states = torch.tensor(batch['state'], dtype=torch.float32).to(self.device)
        next_states = torch.tensor(batch['next_state'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.long).to(self.device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32).to(self.device)
        terminated = torch.tensor(batch['terminated'], dtype=torch.float32).to(self.device)
        mask = 1.0 - torch.tensor(batch['padded'], dtype=torch.float32).to(self.device)
        next_avail_actions = torch.tensor(batch['next_avail_actions'], dtype=torch.float32).to(self.device)

        batch_size = states.size(0)
        max_seq_length = states.size(1)

        # DRQN 序列前向
        mac_out = []
        target_mac_out = []

        self.init_hidden(batch_size)
        target_hidden = torch.zeros(
            (batch_size, self.n_agents, self.conf.drqn_hidden_dim),
        ).to(self.device)

        for t in range(max_seq_length):
            inputs, next_inputs = self._build_inputs(batch, t, batch_size)
            out, self.eval_hidden = self.eval_drqn_net(inputs, self.eval_hidden)
            with torch.no_grad():
                target_out, target_hidden = self.target_drqn_net(next_inputs, target_hidden)
            mac_out.append(out.view(batch_size, self.n_agents, self.n_actions))
            target_mac_out.append(target_out.view(batch_size, self.n_agents, self.n_actions))

        mac_out = torch.stack(mac_out, dim=1)
        target_mac_out = torch.stack(target_mac_out, dim=1)

        # 选中动作 Q 值
        chosen_action_qvals = torch.gather(mac_out, dim=3, index=actions).squeeze(3)

        # Target max Q
        target_mac_out[next_avail_actions == 0.0] = -9999999.0
        target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mixer
        chosen_action_qvals = self.eval_qmix_net(chosen_action_qvals, states)
        target_max_qvals = self.target_qmix_net(target_max_qvals, next_states)

        # TD loss
        targets = rewards + self.conf.gamma * (1 - terminated) * target_max_qvals
        td_error = (chosen_action_qvals - targets.detach()) * mask
        loss = (td_error ** 2).sum() / mask.sum()

        # 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_parameters, self.conf.grad_norm_clip)
        self.optimizer.step()

        # Target 同步
        if train_step > 0 and train_step % self.conf.update_target_params == 0:
            self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        return loss.item()

    # ============================================================
    # 持久化
    # ============================================================

    def save_model(self, episode_idx=None):
        if not self.model_dir:
            return
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.eval_drqn_net.state_dict(),
                   os.path.join(self.model_dir, weight_file_name(self.weight_prefix, 'drqn')))
        torch.save(self.eval_qmix_net.state_dict(),
                   os.path.join(self.model_dir, weight_file_name(self.weight_prefix, 'qmix')))

    def load_state(self, load_dir):
        drqn_path = os.path.join(load_dir, weight_file_name(self.weight_prefix, 'drqn'))
        qmix_path = os.path.join(load_dir, weight_file_name(self.weight_prefix, 'qmix'))
        if not os.path.exists(drqn_path) or not os.path.exists(qmix_path):
            raise FileNotFoundError(f"加载失败：未在 {load_dir} 中找到权重")
        self.eval_drqn_net.load_state_dict(
            torch.load(drqn_path, map_location=self.device, weights_only=True))
        self.eval_qmix_net.load_state_dict(
            torch.load(qmix_path, map_location=self.device, weights_only=True))
        self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())


# ============================================================
# HQMIX —— 统一骨架两阶段（LD 层次化 + WX 独立 QMIX）
# ============================================================

class HQMIX(_BasePolicy):
    """统一骨架两阶段 H-QMIX：LD 层次化混频 + WX 独立 QMIX 混频，共享 DRQN 主干。

    值分解:  Q_tot = Q_tot_ld(LowerMixer→UpperMixer) + (phase>=2 ? Q_tot_wx(QMIXNET) : 0)

    phase1（只训 LD）:
      - 仅 step opt_ld（trunk + LD 头 + lower/upper），WX 头 + wx_mixer 冻结；
      - Q_tot_wx 不进入损失，WX 随机输出不污染 LD 训练；
      - 推理时 WX 恒待机（不参与动作）。
    phase2（联合训练）:
      - opt_ld + opt_wx 都 step，trunk 共享、被两侧梯度共同更新。
    """

    def __init__(self, conf, group_assignments: List[List[int]]):
        super().__init__(conf, dual_head=True)

        # ---- 分组元数据（仅 LD 分组；WX 走独立 QMIX） ----
        self.group_assignments = group_assignments
        self.G = len(group_assignments)
        self.K = max(len(g) for g in group_assignments)
        self.n_targets = getattr(conf, 'n_targets', 21)
        self.ld_capacity = LD_CAPACITY
        self.phase = 1

        # padding mask: (G, K), 1=有效 0=padding
        self._agent_mask = torch.ones(self.G, self.K)
        for g, indices in enumerate(group_assignments):
            for k, idx in enumerate(indices):
                if idx < 0:
                    self._agent_mask[g, k] = 0.0

        # ---- 状态维度 ----
        self.s_k_dim = group_local_state_dim(self.K, self.n_targets)
        self.S_dim = global_pooled_state_dim(self.G, self.n_targets)

        # ---- LD 分支：LowerMixer + UpperMixer ----
        self.eval_lower_mixer = LowerMixer(self.K, self.s_k_dim).to(self.device)
        self.target_lower_mixer = LowerMixer(self.K, self.s_k_dim).to(self.device)
        self.eval_upper_mixer = UpperMixer(self.G, self.S_dim).to(self.device)
        self.target_upper_mixer = UpperMixer(self.G, self.S_dim).to(self.device)
        self.target_lower_mixer.load_state_dict(self.eval_lower_mixer.state_dict())
        self.target_upper_mixer.load_state_dict(self.eval_upper_mixer.state_dict())

        # ---- WX 分支：独立 QMIXNET（25 卫星，覆盖 conf.n_agents=225） ----
        self.eval_wx_mixer = QMIXNET(conf, n_agents=self.n_wx).to(self.device)
        self.target_wx_mixer = QMIXNET(conf, n_agents=self.n_wx).to(self.device)
        self.target_wx_mixer.load_state_dict(self.eval_wx_mixer.state_dict())

        # ---- 双优化器参数分组 ----
        # opt_ld: trunk(fc1/ln/rnn) + LD 头(fc_ld) + lower/upper
        # opt_wx: WX 头(fc_wx) + wx_mixer
        # opt_ld 内部再分两个 param_group：初始 lr 相同 => 与不分组时等价；
        # 热启动后由 set_trunk_lr_scale 把主干那组单独降速。
        ld_head_params = (
            list(self.eval_drqn_net.fc_ld.parameters())
            + list(self.eval_lower_mixer.parameters())
            + list(self.eval_upper_mixer.parameters())
        )
        self.opt_ld, self._opt_ld_params = self._make_optimizer(ld_head_params)

        # opt_wx 不含主干（主干由 opt_ld 持有），无需分组
        self._opt_wx_params = (
            list(self.eval_drqn_net.fc_wx.parameters())
            + list(self.eval_wx_mixer.parameters())
        )
        self.opt_wx = torch.optim.RMSprop(self._opt_wx_params, lr=self.conf.learning_rate)

        # 兼容基类接口
        self.eval_parameters = self._opt_ld_params + self._opt_wx_params
        self.optimizer = self.opt_ld

    def set_phase(self, phase: int):
        """phase1 = 只训 LD；phase2 = 联合训练。由 Runner 在切阈值时调用。"""
        self.phase = phase

    # ============================================================
    # 训练
    # ============================================================

    def learn(self, batch, train_step, epsilon=None):
        device = self.device
        B, T = batch['obs'].shape[0], batch['obs'].shape[1]

        states      = torch.tensor(batch['state'], dtype=torch.float32).to(device)
        next_states = torch.tensor(batch['next_state'], dtype=torch.float32).to(device)
        actions_onehot = torch.tensor(batch['actions_onehot'], dtype=torch.float32).to(device)
        rewards     = torch.tensor(batch['rewards'], dtype=torch.float32).to(device)
        terminated  = torch.tensor(batch['terminated'], dtype=torch.float32).to(device)
        mask        = 1.0 - torch.tensor(batch['padded'], dtype=torch.float32).to(device)
        next_avail_actions = torch.tensor(batch['next_avail_actions'], dtype=torch.float32).to(device)

        # DRQN 双头序列前向
        q_ld, q_wx = self._drqn_seq_dual(batch, B, T, target=False)
        with torch.no_grad():
            tq_ld, tq_wx = self._drqn_seq_dual(batch, B, T, target=True)

        mac_out = self._combine_heads(q_ld, q_wx)              # (B,T,N,n_actions)
        target_mac_out = self._combine_heads(tq_ld, tq_wx)

        # 选中动作 Q（LD 多标签求和 / WX 单热求和，统一用 multi-hot 加权）
        chosen_q = (mac_out * actions_onehot).sum(dim=3)       # (B,T,N)
        target_max_q = self._target_max_q(target_mac_out, next_avail_actions)  # (B,T,N)

        # LD 分支混频
        q_tot = self._hierarchical_mix(chosen_q, states, self.eval_lower_mixer, self.eval_upper_mixer)
        with torch.no_grad():
            target_q_tot = self._hierarchical_mix(target_max_q, next_states, self.target_lower_mixer, self.target_upper_mixer)

        # WX 分支（phase1 排除，避免随机 WX 输出污染 LD 训练）
        if self.phase >= 2:
            q_tot = q_tot + self.eval_wx_mixer(chosen_q[:, :, self.n_ld:], states)
            with torch.no_grad():
                target_q_tot = target_q_tot + self.target_wx_mixer(target_max_q[:, :, self.n_ld:], next_states)

        # TD loss
        targets = rewards + self.conf.gamma * (1 - terminated) * target_q_tot
        td_error = (q_tot - targets.detach()) * mask
        loss = (td_error ** 2).sum() / mask.sum()

        # 梯度更新（分参数组）
        self.opt_ld.zero_grad()
        self.opt_wx.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._opt_ld_params, self.conf.grad_norm_clip)
        if self.phase >= 2:
            nn.utils.clip_grad_norm_(self._opt_wx_params, self.conf.grad_norm_clip)
            self.opt_wx.step()
        self.opt_ld.step()

        # Target 同步
        if train_step > 0 and train_step % self.conf.update_target_params == 0:
            self._sync_target()

        return loss.item()

    # ============================================================
    # 双头前向 / 值结算
    # ============================================================

    def _drqn_seq_dual(self, batch, B, T, target=False):
        """展开时间维，返回 (q_ld (B,T,N,ld_n_actions), q_wx (B,T,N,n_actions))。"""
        drqn = self.target_drqn_net if target else self.eval_drqn_net
        hidden = torch.zeros(B * self.n_agents, self.conf.drqn_hidden_dim).to(self.device)
        obs_key = 'next_obs' if target else 'obs'

        q_ld_list, q_wx_list = [], []
        for t in range(T):
            inputs = self._build_inputs_for_seq(batch, t, B, obs_key, target=target)
            q_ld, q_wx, hidden = drqn(inputs, hidden)
            q_ld_list.append(q_ld.view(B, self.n_agents, self.ld_n_actions))
            q_wx_list.append(q_wx.view(B, self.n_agents, self.n_actions))
        return torch.stack(q_ld_list, 1), torch.stack(q_wx_list, 1)

    def _combine_heads(self, q_ld, q_wx):
        """把 LD 头(21) 与 WX 头(22) 拼成统一 (B,T,N,22)。LD 行用 q_ld（待机位=0）。"""
        out = q_wx.clone()                                   # (B,T,N,22)
        out[:, :, :self.n_ld, 1:] = q_ld[:, :, :self.n_ld, :]  # 目标 0..20 → 动作位 1..21
        out[:, :, :self.n_ld, 0] = 0.0                       # LD 待机位恒 0
        return out

    def _target_max_q(self, target_mac_out, next_avail):
        B, T, N, _ = target_mac_out.shape
        # LD：可探测掩码 → top-20 正值求和（Q≤0 不锁，等价于不选）
        ld_q = target_mac_out[:, :, :self.n_ld, 1:]
        ld_avail = next_avail[:, :, :self.n_ld, 1:]
        ld_q = ld_q.masked_fill(ld_avail == 0, float('-inf'))
        ld_pos = torch.clamp(ld_q, min=0.0)
        k = min(self.ld_capacity, self.ld_n_actions)
        ld_topk, _ = torch.topk(ld_pos, k, dim=3)
        ld_max = ld_topk.sum(dim=3)                          # (B,T,n_ld)
        # WX：掩码后单热 max
        wx_q = target_mac_out[:, :, self.n_ld:, :]
        wx_avail = next_avail[:, :, self.n_ld:, :]
        wx_q = wx_q.masked_fill(wx_avail == 0, float('-inf'))
        wx_max = wx_q.max(dim=3)[0]                          # (B,T,n_wx)
        return torch.cat([ld_max, wx_max], dim=2)            # (B,T,N)

    # ============================================================
    # 层次化混频（LD 分支，复用原逻辑）
    # ============================================================

    def _hierarchical_mix(self, agent_qs, full_state, lower_mixer, upper_mixer):
        """agent_qs (B, T, N) → Q_tot (B, T, 1)"""
        B, T, _N = agent_qs.shape
        device = agent_qs.device

        # 重组为分组张量 + padding mask
        q_grouped = torch.zeros(B, T, self.G, self.K, device=device)
        for g, indices in enumerate(self.group_assignments):
            for k, idx in enumerate(indices):
                if idx >= 0:
                    q_grouped[:, :, g, k] = agent_qs[:, :, idx]
        q_grouped = q_grouped * self._agent_mask.to(device).view(1, 1, self.G, self.K)

        # 组局部状态
        group_states = build_group_local_states(
            full_state, self.group_assignments, self.n_targets,
        )

        # Lower: 组内混频
        q_groups = lower_mixer(q_grouped, group_states).squeeze(-1)

        # 全局池化状态
        S = build_global_pooled_state(
            full_state, self.group_assignments, self.n_targets,
        )

        # Upper: 组间混频
        return upper_mixer(q_groups, S)

    def _sync_target(self):
        self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
        self.target_lower_mixer.load_state_dict(self.eval_lower_mixer.state_dict())
        self.target_upper_mixer.load_state_dict(self.eval_upper_mixer.state_dict())
        self.target_wx_mixer.load_state_dict(self.eval_wx_mixer.state_dict())

    # ============================================================
    # 持久化
    # ============================================================

    def save_model(self, episode_idx=None):
        if not self.model_dir:
            return
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.eval_drqn_net.state_dict(),
                   os.path.join(self.model_dir, weight_file_name(self.weight_prefix, 'drqn')))
        torch.save(self.eval_lower_mixer.state_dict(),
                   os.path.join(self.model_dir, weight_file_name(self.weight_prefix, 'lower_mixer')))
        torch.save(self.eval_upper_mixer.state_dict(),
                   os.path.join(self.model_dir, weight_file_name(self.weight_prefix, 'upper_mixer')))
        torch.save(self.eval_wx_mixer.state_dict(),
                   os.path.join(self.model_dir, weight_file_name(self.weight_prefix, 'wx_mixer')))

    def load_state(self, load_dir):
        def _load(name):
            path = os.path.join(load_dir, weight_file_name(self.weight_prefix, name))
            if not os.path.exists(path):
                raise FileNotFoundError(f"加载失败：未在 {load_dir} 中找到 {name}")
            return torch.load(path, map_location=self.device, weights_only=True)

        self.eval_drqn_net.load_state_dict(_load('drqn'))
        self.eval_lower_mixer.load_state_dict(_load('lower_mixer'))
        self.eval_upper_mixer.load_state_dict(_load('upper_mixer'))
        self.eval_wx_mixer.load_state_dict(_load('wx_mixer'))
        self._sync_target()
