from typing import List, Optional

import numpy as np
import torch

from use_cases.config.config_types import AlgorithmConfig, _get

from .policy import QMIX, HQMIX
from services.scene.scene_constants import LD_CAPACITY


class Agents:
    """等价于 EPyMARL 中的 BasicMAC (src/controllers/basic_mac.py)

    可通过两种方式构造:
      - Agents(conf)  —— 兼容旧版扁平 RuntimeConfig
      - Agents(conf, algo_config=algo_cfg) —— 新版聚焦 AlgorithmConfig

    当传入 group_assignments 时自动切换为 HQMIX（统一骨架双头），
    否则为标准 QMIX（单头单选）。
    """

    def __init__(self, conf, algo_config: Optional[AlgorithmConfig] = None,
                 group_assignments: Optional[List[List[int]]] = None):
        self.conf = conf
        ac = algo_config or AlgorithmConfig.from_config(conf)

        self.device = ac.device
        self.n_actions = ac.n_actions
        self.n_agents = ac.n_agents
        self.n_targets = _get(conf, "n_targets", self.n_actions - 1)

        # ---- 统一骨架维度（LD 多标签 / WX 单选） ----
        self.n_ld = getattr(ac, 'n_ld', 0) or 0
        self.n_wx = getattr(ac, 'n_wx', 0) or 0
        self.ld_n_actions = getattr(ac, 'ld_n_actions', 0) or self.n_targets
        self.ld_capacity = LD_CAPACITY
        self.wx_epsilon = _get(conf, 'wx_epsilon', 0.3)
        # 锁粘滞（LD 多标签解码）：上一帧已锁、当前仍可用且 Q 非强负的目标继续锁，
        # 掐断「仍可见但 Q 被反超 → 放掉 → 下帧又捡回」的掉锁-起锁中断抖动。
        self.lock_sticky = _get(conf, 'lock_sticky', True)
        self.lock_sticky_margin = float(_get(conf, 'lock_sticky_margin', 0.0))
        self.phase = 1  # phase1 = 只训 LD（WX 待机），phase2 = 联合

        # ---- 分组元数据（None = 标准 QMIX, 非空 = HQMIX 双头） ----
        self.group_assignments = group_assignments

        if group_assignments is not None and len(group_assignments) > 0:
            self.policy = HQMIX(conf, group_assignments)
        else:
            self.policy = QMIX(conf)

    def set_phase(self, phase: int):
        """phase1 = 只训 LD（WX 待机）；phase2 = 联合训练。由 Runner 在切阈值时调用。"""
        self.phase = phase
        if isinstance(self.policy, HQMIX):
            self.policy.set_phase(phase)

    def init_episode(self):
        """初始化一个 episode 的 RNN hidden state（供 RolloutWorker 调用）。"""
        self.policy.init_hidden(1)
        self.policy.eval_hidden = self.policy.eval_hidden.to(self.device)

    def perform_inference(self, obs, last_actions, avail_actions, epsilon=0.0):
        """Rollout 时调用的前向推演。

        HQMIX 双头: LD 多标签 top-k + WX 单选（phase1 待机）。
        标准 QMIX:  并行单选（与原版一致）。
        """
        if isinstance(self.policy, HQMIX):
            return self._perform_dual_head_inference(
                obs, last_actions, avail_actions, epsilon,
            )
        return self._perform_parallel_inference(
            obs, last_actions, avail_actions, epsilon,
        )

    # ============================================================
    # 并行推理（标准 QMIX —— 单头单选）
    # ============================================================

    def _perform_parallel_inference(self, obs, last_actions, avail_actions, epsilon):
        """所有 agent 同时决策（原版逻辑）。"""
        inputs = self._build_inputs(obs, last_actions)
        inputs_tensor = torch.tensor(np.array(inputs), dtype=torch.float32).to(self.device)

        hidden_state = self.policy.eval_hidden
        q_values, self.policy.eval_hidden = self.policy.eval_drqn_net(
            inputs_tensor, hidden_state,
        )

        avail_tensor = torch.tensor(avail_actions, dtype=torch.float32).to(self.device)
        q_values[avail_tensor == 0.0] = -9999999.0

        actions, actions_onehot = self._epsilon_greedy_batch(
            q_values, avail_actions, epsilon,
        )
        return actions, q_values.detach().cpu().numpy(), actions_onehot

    # ============================================================
    # 双头推理（HQMIX —— LD 多标签 + WX 单选）
    # ============================================================

    def _perform_dual_head_inference(self, obs, last_actions, avail_actions, epsilon):
        """统一骨架双头推理。

        - LD 头输出逐目标独立 Q（21 维），多标签 top-k（Q>0 才锁，最多 ld_capacity 个）。
        - WX 头输出单选 logits（22 维），phase1 恒待机，phase2 用 wx_epsilon 探索。

        返回:
          actions:       (n_agents,)  argmax 代表动作（buffer 兼容）
          combined_q:    (n_agents, n_actions) 已掩码合并 Q（日志用）
          actions_onehot:(n_agents, n_actions) 多标签 one-hot（LD 多位置 1）
        """
        inputs = self._build_inputs(obs, last_actions)
        inputs_tensor = torch.tensor(np.array(inputs), dtype=torch.float32).to(self.device)

        hidden_state = self.policy.eval_hidden
        q_ld, q_wx, self.policy.eval_hidden = self.policy.eval_drqn_net(
            inputs_tensor, hidden_state,
        )
        q_ld_np = q_ld.detach().cpu().numpy()   # (N, ld_n_actions)
        q_wx_np = q_wx.detach().cpu().numpy()   # (N, n_actions)

        avail = np.asarray(avail_actions, dtype=np.float32)  # (N, n_actions)

        # 合并为统一 (N, n_actions)：LD 行用 q_ld（待机位恒 -inf），WX 行用 q_wx
        combined = q_wx_np.copy()
        combined[:self.n_ld, 1:] = q_ld_np[:self.n_ld, :]
        combined[:self.n_ld, 0] = -1e9
        combined[avail == 0] = -1e9

        actions_onehot = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        actions = [0] * self.n_agents

        # ---- LD：多标签 top-k 正 Q ----
        for i in range(self.n_ld):
            a = avail[i, 1:]          # (ld_n_actions,) 目标可用掩码
            q = q_ld_np[i].copy()
            q[a == 0] = -1e9

            if np.random.uniform() < epsilon:
                # 探索：随机选 0..capacity 个可用目标
                cand = np.nonzero(a)[0]
                n_pick = np.random.randint(0, min(self.ld_capacity, len(cand)) + 1)
                chosen = np.random.choice(cand, size=n_pick, replace=False)
            else:
                # 锁粘滞：上一帧已锁的目标，只要当前仍可用且 Q 非强负（≥ -margin），
                # 就优先保留；剩余容量再用「未锁过的正 Q」按 top-K 填补。
                prev = None
                if self.lock_sticky and last_actions is not None:
                    prev = np.asarray(last_actions[i])[1:]      # 上一帧目标锁位（0..ld_n_actions-1）
                keep = []
                if prev is not None:
                    keep = [int(b) for b in np.nonzero(prev > 0)[0]
                            if b < self.ld_n_actions and a[b] > 0
                            and q[b] >= -self.lock_sticky_margin]
                used = set(keep)
                free = self.ld_capacity - len(keep)
                order = [int(b) for b in np.argsort(-q)
                         if b not in used and a[b] > 0 and q[b] > 0]
                chosen = keep + order[:max(0, free)]

            for b in chosen:
                b = int(b)
                actions_onehot[i, b + 1] = 1.0
            actions[i] = int(np.argmax(combined[i]))

        # ---- WX：单选（phase1 待机） ----
        for i in range(self.n_ld, self.n_agents):
            a = avail[i, :]           # (n_actions,) 全动作可用掩码
            q = q_wx_np[i].copy()
            q[a == 0] = -1e9

            if self.phase < 2:
                action = 0            # phase1：卫星不参与动作，恒待机
            elif epsilon == 0.0:
                action = int(np.argmax(q))   # 纯推理：WX 也贪心，不发随机探索
            elif np.random.uniform() < self.wx_epsilon:
                cand = np.nonzero(a)[0]
                action = int(np.random.choice(cand)) if len(cand) > 0 else 0
            else:
                action = int(np.argmax(q))

            actions_onehot[i, action] = 1.0
            actions[i] = action

        return np.array(actions), combined, actions_onehot

    # ============================================================
    # 输入构建（复用）
    # ============================================================

    def _build_inputs(self, obs, last_actions):
        """构建 PyMARL 标准输入特征: [obs, last_action, agent_id_onehot]"""
        inputs = []
        for i in range(self.n_agents):
            agent_feat = [obs[i].copy()]
            if self.conf.last_action:
                agent_feat.append(last_actions[i].copy())
            if getattr(self.conf, 'reuse_network', True):
                agent_id = np.zeros(self.n_agents)
                agent_id[i] = 1.0
                agent_feat.append(agent_id)
            inputs.append(np.concatenate(agent_feat))
        return inputs

    def _epsilon_greedy_batch(self, q_values, avail_actions, epsilon):
        """批量 epsilon-greedy 动作选择（标准 QMIX 单选用）。"""
        actions = []
        actions_onehot = []
        for i in range(self.n_agents):
            if np.random.uniform() < epsilon:
                avail_idx = np.nonzero(avail_actions[i])[0]
                action = np.random.choice(avail_idx) if len(avail_idx) > 0 else 0
            else:
                action = torch.argmax(q_values[i]).item()
            actions.append(action)
            onehot = np.zeros(self.n_actions)
            onehot[action] = 1.0
            actions_onehot.append(onehot)
        return actions, actions_onehot

    def train_(self, batch, train_step, epsilon=None):
        return self.policy.learn(batch, train_step, epsilon)

    # ============================================================
    # 并行 rollout 权重同步
    # ============================================================

    def get_inference_weights(self) -> dict:
        """导出推理主干权重（CPU 张量），供并行 worker 进程同步。

        只同步 eval_drqn_net（rollout 推理唯一用到的网络），
        混频器 / target 网络不参与采样，无需搬运。
        """
        return {k: v.detach().cpu()
                for k, v in self.policy.eval_drqn_net.state_dict().items()}

    def set_inference_weights(self, weights: dict) -> None:
        """加载推理主干权重（跨进程同步用）。"""
        self.policy.eval_drqn_net.load_state_dict(
            {k: v.to(self.device) for k, v in weights.items()}
        )
