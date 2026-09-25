"""奖励分项诊断 —— 独立工具类（只消费 reward.py 原生产出的分项，不重算奖励）。

奖励分项的唯一来源是 services/scene/reward/reward.py 的
``RewardCalculator.compute_reward_detailed``（原生分项，单点真相）。
本文件不做任何奖励逻辑的镜像重算，只负责：
  - RewardBreakdownTracker：逐步累计 next_info['reward_breakdown'] 并汇总打印；
  - RewardBreakdownHook：挂在 rollout step_hooks 上，按局间隔打印分项表；
  - diagnose_episode：独立跑一局打印分项（不改训练主链路）。

因此 reward.py 一旦调整系数/规则，分项自动随之变化，不存在「两套代码漂移」的问题。

用法（独立跑一局，ε 可控）:
    from utils.reward_diagnostics import diagnose_episode
    diagnose_episode(env, agents, epsilon=0.0)
"""

# ---- 实体数与单步奖励的量化关系（系数见 RewardCalculator 类属性，v4）----
# 设  N_r=雷达数(LD)  N_s=卫星数(WX)  N_t=目标数  T=每局步数；
#     每步可见目标数 V_vis，覆盖(≥1锁)目标数 C，漏警 M=V_vis-C；
#     E=边缘锁定次数  V=瞎指锁定次数  S=切换惩罚次数  G=覆盖归零次数；
#     Mult=Σ_t min(n_lock(t), K)（覆盖重数，K=10）。
#
#     单步 R_ld      = MULT_W·Mult - 20M - 2E - 5V   （覆盖重数线性到10封顶，超重中性）
#     单步 R_wx      = Σ卫星项，每部卫星 ∈ [-5, +6]
#     单步 R_switch  = -8S - 15G   （中断 10%；终态 done 不结算）
#
#     R_episode = Σ_{t=1..T} (R_ld + R_wx + R_switch)
#
#     覆盖重数 R_mult 是主导项（对齐评价指标 60% 权重）：每个目标 1→10 重线性加分，
#     >10 不奖不罚（耗能非主指标，不再设冗余罚）。中断 R_switch/R_gap 加重，呼应
#     「极重视极小中断次数」。

import numpy as np


# 分项打印顺序与中文名（与 reward.py 原生 breakdown key 一一对应）
_COLS = [
    ("ld_miss", "雷达漏警"),
    ("ld_mult", "覆盖重数"),
    ("ld_valid", "雷达瞎指"),
    ("ld_edge", "雷达边缘"),
    ("wx_blind", "卫星补盲"),
    ("wx_handoff", "卫星预警"),
    ("wx_valid", "卫星瞎指"),
    ("wx_idle", "卫星空耗"),
    ("switch_penalty", "槽位切换"),
    ("switch_gap", "覆盖归零"),
]


def _to_np(x):
    """torch.Tensor → numpy，兼容普通 ndarray。"""
    return x.cpu().detach().numpy() if hasattr(x, "cpu") else np.asarray(x)


# ============================================================
# 累计 + 打印
# ============================================================

class RewardBreakdownTracker:
    """累计每步 reward 与 breakdown，输出分项汇总。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self._total = 0.0
        self._steps = 0
        self._acc = {}   # key -> {"value": float, "count": int}

    def add(self, reward: float, breakdown: dict = None):
        self._total += float(reward)
        self._steps += 1
        if not breakdown:
            return
        for key, item in breakdown.items():
            slot = self._acc.setdefault(key, {"value": 0.0, "count": 0})
            slot["value"] += float(item.get("value", 0.0))
            slot["count"] += int(item.get("count", 0))

    @property
    def total(self) -> float:
        return self._total

    @property
    def steps(self) -> int:
        return self._steps

    def summary(self) -> str:
        avg = self._total / self._steps if self._steps else 0.0
        return f"total={self._total:.2f} steps={self._steps} avg/step={avg:.4f}"

    def report(self, episode_idx: int = None) -> str:
        """格式化分项表：打印并返回字符串。"""
        head = (f"---- 奖励分项诊断 (episode {episode_idx}) ----"
                if episode_idx is not None else "---- 奖励分项诊断 ----")
        lines = [head, f"累计: {self.summary()}",
                 f"{'分项':<14}{'value':>12}{'count':>8}"]
        for key, name in _COLS:
            slot = self._acc.get(key)
            if slot is None:
                continue
            lines.append(f"{name:<14}{slot['value']:>12.2f}{slot['count']:>8}")
        lines.append("-" * 34)
        report = "\n".join(lines)
        print(report)
        return report


class RewardBreakdownHook:
    """训练 step 回调 —— 累计原生奖励分项并按间隔在每局结束打印。

    供 runner 挂到 step_hooks（定义集中在 utils/reward_diagnostics.py，这里只调用）:

        hook = RewardBreakdownHook(report_interval=50)
        worker.generate_train_episode(epsilon=..., step_hooks=[hook])

    分项直接读 next_info['reward_breakdown']（env_wrapper 已原生产出），
    无额外奖励计算开销。
    """

    def __init__(self, report_interval: int = 50):
        self.report_interval = report_interval
        self._tracker = RewardBreakdownTracker()
        self._ep = 0

    def __call__(self, step_idx, terminated, truncated, next_info):
        if step_idx == 0:
            self._tracker.reset()
            self._ep += 1

        self._tracker.add(
            next_info.get('_last_reward', 0.0),
            next_info.get('reward_breakdown'),
        )

        if terminated or truncated:
            if self.report_interval > 0 and self._ep % self.report_interval == 0:
                self._tracker.report(self._ep)


def diagnose_episode(env, agents, epsilon: float = 0.0, max_steps: int = None):
    """独立跑一局并打印奖励分项表（不改训练主链路）。

    Args:
        env:       GroupedEnvWrapper（step 返回 info['reward_breakdown']）。
        agents:    Agents（需有 perform_inference / n_agents / n_actions / init_episode）。
        epsilon:   0.0 = 纯贪心（推演口径）；>0 可复现训练探索。
        max_steps: 每局步数上限，默认取 env._max_episode_steps。

    Returns:
        RewardBreakdownTracker（内部已打印 report）。
    """
    max_steps = max_steps or getattr(env, '_max_episode_steps', 600)
    n_agents = agents.n_agents
    n_actions = agents.n_actions

    obs, info = env.reset()
    avail_actions = info['avail_actions']
    last_actions = np.zeros((n_agents, n_actions), dtype=np.float32)
    agents.init_episode()

    tracker = RewardBreakdownTracker()
    terminated = truncated = False
    step = 0
    while not (terminated or truncated) and step < max_steps:
        _, _, actions_onehot = agents.perform_inference(
            obs=obs, last_actions=last_actions,
            avail_actions=avail_actions, epsilon=epsilon,
        )
        onehot = _to_np(actions_onehot)
        next_obs, reward, terminated, truncated, next_info = env.step(onehot)

        tracker.add(reward, next_info.get('reward_breakdown'))

        obs = next_obs
        avail_actions = next_info['avail_actions']
        last_actions = onehot
        step += 1

    tracker.report()
    return tracker
