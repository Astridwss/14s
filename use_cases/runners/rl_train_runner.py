"""
RLTrainRunner —— 强化学习训练主循环（QMIX / H-QMIX）。

继承 BaseRunner，从 conf 自建积木，统一通过 self.push 推送。
"""

import json
import os
import time

from use_cases.runners.base_runner import BaseRunner
from use_cases.config.config_types import EnvConfig, AlgorithmConfig, TrainingConfig, InfraConfig
from services.scene.env_wrapper import GroupedEnvWrapper
from services.algorithms.qmix.agent import Agents
from services.sample.buffer import EpisodeReplayBuffer
from services.sample.rollout import RolloutWorker
from utils.situation_logger import SituationLogHook
from utils.epsilon_schedule import EpsilonSchedule
from services.zmq.situation_publisher import TrainingSituationPushService
from services.zmq.situation_pool import SituationTrajectory
from utils.seed import set_seeds
from utils.cpu_cores import clamp_workers
from utils.reward_plot import save_reward_plot, save_loss_plot
from utils.reward_diagnostics import RewardBreakdownHook
from services.scene.grouping import RadarGrouper

try:
    from utils.tv_display import LiveObserver
except ImportError:
    # 机器未安装 pygame(-ce) 时，退化为空实现，自动关闭 2D 态势实时渲染
    class LiveObserver:
        """无 pygame 环境的空实现：所有渲染接口均为 no-op。"""
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def begin_episode(self, *args, **kwargs):
            return False
        def render_callback(self, info):
            pass
        def end_episode(self):
            pass
        def shutdown(self):
            pass
    print("[RLTrainRunner] 未检测到 pygame(-ce)，已自动关闭 2D 态势实时渲染")


class RLTrainRunner(BaseRunner):
    """RL 训练主循环（自动检测 H-QMIX 分组）。"""

    def __init__(self, conf, push=None):
        super().__init__(conf, push)

        # ---- 提取聚焦配置 ----
        ec = getattr(conf, 'env', None) or EnvConfig.from_config(conf)
        ac = getattr(conf, 'algo', None) or AlgorithmConfig.from_config(conf)
        tc = getattr(conf, 'train', None) or TrainingConfig.from_config(conf)
        ic = getattr(conf, 'infra', None) or InfraConfig.from_config(conf)

        # ---- 自建积木（注入聚焦配置） ----
        self.env = GroupedEnvWrapper(conf, env_config=ec)

        # H-QMIX 分组（group_size > 0 时自动启用）
        group_assignments = RadarGrouper.try_build(
            group_size=getattr(ac, 'group_size', 0),
            radar_keys=ec.radar_keys,
            radar_info_dict=getattr(self.env, 'dict_radar_info', None),
        )

        # ---- 分组信息打印 ----
        if group_assignments is not None:
            n_groups = len(group_assignments)
            k_max = max(len(g) for g in group_assignments)
            group_sizes = [sum(1 for idx in g if idx >= 0) for g in group_assignments]
            satellites = len(ec.satellites_keys)
            print(f"\n{'='*60}")
            print(f"[H-QMIX] 统一骨架双头 + 层次化分组混频已启用")
            print(f"[软件版本号]: V0.0.0.5")
            print(f"  雷达(LD):   {ec.n_radars} (多标签 top-{getattr(ac, 'ld_n_actions', 0)} 选 20)")
            print(f"  卫星(WX):   {satellites} (单选, phase2 解冻)")
            print(f"  目标数量:   {ec.n_targets}")
            print(f"  每组雷达数: ~{ac.group_size}")
            print(f"  分组结果:   {ec.n_radars} 部雷达 → {n_groups} 组, 每组最多 {k_max} agents")
            print(f"  各组实际大小: {group_sizes}")
            print(f"  总 agent 数: {ec.n_agents} (LD {ec.n_ld} + WX {ec.n_wx})")
            print(f"  观测维度:   {ec.obs_shape} (per-agent)")
            print(f"  全局状态:   {ec.state_shape}")
            print(f"  两阶段训练: phase1_episodes={tc.phase1_episodes}, wx_epsilon={tc.wx_epsilon}")
            print(f"{'='*60}\n")
        else:
            print(f"\n[QMIX] 标准单层混频 (未启用分组)")
            print(f"  agents={ec.n_agents}, obs_shape={ec.obs_shape}, "
                  f"state_shape={ec.state_shape}\n")

        self.agents = Agents(conf, algo_config=ac, group_assignments=group_assignments)

        # IL → RL 热启动：仅加载 DRQN 预训练权重（混频器从零训练）
        self._maybe_warm_start(ic)

        self.buffer = EpisodeReplayBuffer(conf, algo_config=ac, train_config=tc)
        self.rollout_worker = RolloutWorker(
            self.env, self.agents,
            radar_keys=ec.agent_keys,
            target_keys=ec.target_keys,
        )
        self.epsilon = EpsilonSchedule.from_conf(tc)

        # ---- 训练控制参数 ----
        self.max_episodes = tc.max_episodes
        self.model_dir = ic.model_dir
        self.result_dir = ic.result_dir
        self.save_frequency = tc.save_frequency
        self.batch_size = tc.batch_size
        self.utd_ratio = max(1, int(getattr(tc, 'utd_ratio', 1)))
        self.update_target_every = ac.update_target_params
        self.seed = tc.seed

        # ---- 两阶段冻结训练控制 ----
        self.phase1_episodes = tc.phase1_episodes  # -1=纯phase1标定 / 0=直接联合 / N=第N局切phase2
        self._current_phase = None

        # ---- ZMQ 态势推送 ----
        self._zmq = TrainingSituationPushService.from_conf(conf)

        # ---- 态势日志 hook ----
        self._situation_log_hook = SituationLogHook(
            agent_keys=ec.agent_keys,
            satellite_keys=ec.satellites_keys,
            target_keys=ec.target_keys,
            log_interval=50,
        )

        # ---- 奖励明细打印 hook（按需开启：REWARD_BREAKDOWN_INTERVAL>0 才挂载） ----
        try:
            bd_interval = int(os.environ.get("REWARD_BREAKDOWN_INTERVAL", "0"))
        except ValueError:
            bd_interval = 0
        self._breakdown_hook = (RewardBreakdownHook(report_interval=bd_interval) if bd_interval > 0 else None)

        # ---- 2D 态势可视化 (训练全程常驻窗口) ----无 GPU/显示器可通过环境变量 LIVE_WATCH_FREQ=0 关闭，
        try:
            watch_freq = int(os.environ.get("LIVE_WATCH_FREQ", "1"))
        except ValueError:
            watch_freq = 1
        self._live_observer = LiveObserver(
            watch_freq=watch_freq,                # 每局都渲染 (调试用); 生产可改大或 0 = 关闭
            fps=30,
            group_assignments=group_assignments,
            radar_keys=ec.radar_keys,
            max_episodes=tc.max_episodes,
        )

        # ---- 内部状态 ----
        self._update_steps = 0
        self.env_steps = 0
        self.episode_rewards = []
        self.episode_losses = []

        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        set_seeds(self.seed)

        # 落盘训练分组元数据，供推演端在未下发 group_size 时自动恢复一致性
        self._write_train_meta(ec, ac, group_assignments)

        # ---- 并行环境采样（rl_num_workers > 1 或 NUM_ENV_WORKERS 环境变量） ----
        # 瓶颈在 sim/client.py 的 step_forward 纯 Python 双循环（CPU 侧），
        # 多进程 worker 各持一份冻结 sim 环境副本独立跑局，让 CPU 核吃满。
        self.n_workers = int(getattr(conf, 'rl_num_workers', 0) or 0)
        _env_workers = os.environ.get("NUM_ENV_WORKERS", "")
        if _env_workers:
            try:
                self.n_workers = int(_env_workers)
            except ValueError:
                pass
        # 并行数是纯 Python CPU-bound，有效上限 = 部署机器物理核数；启动时读一次并 clamp，
        # 避免核数少的机器超订（超订会因上下文切换反而变慢）。0/1 表示不启用并行。
        if self.n_workers > 1:
            self.n_workers = clamp_workers(self.n_workers)
        self._parallel = None
        self._situation_pool = None
        self._situation_sender = None
        if self.n_workers > 1:
            from services.sample.parallel_rollout import ParallelRollout
            self._parallel = ParallelRollout(conf, group_assignments, self.n_workers)
            print(f"[RLTrainRunner] 并行环境采样已启用: {self.n_workers} workers")
            # 并行模式：态势轨迹走「采集→入池→独立发送线程」路径，ZMQ 推送移出 step 循环
            # （训练 replay buffer 与发送池是两个独立池子；发送池无上限 FIFO、不丢最旧，
            #   水位由前端倍速参数调节）。
            if self._zmq is not None:
                from services.zmq.situation_pool import SituationPool
                from services.zmq.situation_sender import SituationSender
                self._situation_pool = SituationPool(
                    warn_episodes=ic.situation_warn_episodes,
                    max_episodes=ic.situation_pool_max,
                )
                self._situation_sender = SituationSender(
                    publisher=self._zmq.publisher,
                    task_id=getattr(conf, 'task_id', 'UNKNOWN'),
                    pool=self._situation_pool,
                    base_interval=ic.situation_base_interval,
                    speed_refresh=ic.situation_speed_refresh,
                )
                self._situation_sender.start()
                print("[RLTrainRunner] 态势轨迹池 + 独立发送线程已启动（前端倍速参数生效中）")

    # ============================================================
    # 公开 API
    # ============================================================

    def run(self):
        """训练入口：并行可用时走 _run_parallel，否则走 _run_serial。"""
        if self._parallel is not None:
            self._run_parallel()
        else:
            self._run_serial()

    def _run_parallel(self):
        """并行主循环：每轮广播权重，N 个 worker 各跑一局，回收后学习。"""
        task_id = getattr(self.conf, 'task_id', 'UNKNOWN')
        round_size = self.n_workers
        episode_idx = 0

        try:
            while episode_idx < self.max_episodes:
                if not self._should_continue(episode_idx + 1):
                    break

                eps = self.epsilon.get(self.env_steps)
                phase = self._phase_for_episode(episode_idx + 1)
                if phase != self._current_phase:
                    self.agents.set_phase(phase)
                    self.env.set_phase(phase)
                    self._current_phase = phase
                    print(f"[RLTrainRunner] 进入 phase{phase} "
                          f"(episode {episode_idx + 1}, phase1_episodes={self.phase1_episodes})")

                weights = self.agents.get_inference_weights()
                n_this_round = min(round_size, self.max_episodes - episode_idx)
                episode_indices = list(range(episode_idx + 1, episode_idx + n_this_round + 1))

                # ---- 计时埋点：rollout 墙钟（N 局并发，含权重广播与轨迹回收的 pickle 开销） ----
                t_rollout = time.perf_counter()
                results = self._parallel.generate_episodes(weights, phase, eps, episode_indices)
                t_rollout = time.perf_counter() - t_rollout
                learn_total = 0.0
                learn_steps = 0

                # 态势轨迹先整批入池：并行本轮已一次产出 N 局，若留在下方 learn_batch
                # 循环里逐局入池，投喂节奏会被学习/保存/推送拖成「1 局/次」，发送线程随之
                # 饿死（池空）。先整批投喂，发送线程才有连续积压可发，speed 才能调节流量。
                if self._situation_pool is not None:
                    for (_, _, _, frames), eidx in zip(results, episode_indices):
                        if frames and self._should_push_episode(eidx):
                            self._situation_pool.put(SituationTrajectory(episode_idx=eidx, frames=frames))

                for ep_reward, step_count, ep_data, frames in results:
                    episode_idx += 1
                    self.env_steps += step_count
                    print(f"强化学习环境构建的训练样本池样本量: {self.env_steps}")
                    self.episode_rewards.append(ep_reward)
                    self.buffer.store_episode(ep_data) #训练buffer
                   
                    t_learn = time.perf_counter()
                    loss = self._learn_batch(eps, self.utd_ratio)
                    t_learn = time.perf_counter() - t_learn
                    learn_total += t_learn
                    if loss is not None:
                        learn_steps += self.utd_ratio
                    self.episode_losses.append(loss)

                    print(f"[Episode {episode_idx}/{self.max_episodes}] "
                          f"ep_reward={ep_reward:.2f}, steps={step_count}, eps={eps:.3f}")

                    is_last = (episode_idx == self.max_episodes)
                    self._maybe_save(episode_idx, is_last)
                    save_reward_plot(self.episode_rewards,
                                     os.path.join(self.result_dir, "episode_reward.png"))
                    save_loss_plot(self.episode_losses,
                                   os.path.join(self.result_dir, "episode_loss.png"))

                    if self.push is not None:
                        self.push.push_metrics({
                            "episode": episode_idx, "reward": ep_reward,
                            "loss": loss if loss is not None else 0.0,
                            "planId": getattr(self.conf, 'plan_id', 'unknown'),
                        })

                    if self.push is not None and (
                        is_last or (self.save_frequency > 0 and episode_idx % self.save_frequency == 0)
                    ):
                        self.push.push_weights({
                            "episode": episode_idx, "model_dir": self.model_dir,
                            "algo": "qmix", "task_id": task_id,
                            "planId": getattr(self.conf, 'plan_id', 'unknown'),
                        })


                # ---- 计时汇总：本轮 rollout vs learning 墙钟，折算 learning 成为瓶颈的 worker 上限 ----
                if learn_steps > 0:
                    t_grad = learn_total / learn_steps
                    w_max = t_rollout / max(self.utd_ratio * t_grad, 1e-9)
                    tail = (f"单步梯度={t_grad * 1000:.2f}ms | "
                            f"learning 成为瓶颈的 worker 上限≈{int(w_max)}")
                else:
                    tail = "learning 未启动（buffer 未攒满 batch_size）"
                print(f"[计时] 本轮(episode 至 {episode_idx}): "
                      f"rollout={t_rollout:.2f}s (N={n_this_round} 并发) | "
                      f"learning={learn_total:.2f}s ({learn_steps} 步) | "
                      f"learn/rollout={learn_total / max(t_rollout, 1e-6):.3f} | {tail}")
        finally:
            # 训练结束（跑满 / 暂停 / 终止 / 异常）都必须回收 spawn 出的 worker 进程：
            # 否则 N 个 worker 各持一份 env + 网络副本常驻，多次训练任务累加泄漏进程与显存。
            self._parallel.close()
            # 停发送线程：放 DRAIN 哨兵，发送线程发完池中剩余轨迹后自行退出（不丢帧）
            if self._situation_sender is not None:
                self._situation_sender.drain()
                self._situation_sender = None

        print("训练管理器：本轮训练结束（并行模式）。")

    def _run_serial(self):
        """串行主循环：epsilon → rollout → learn → save → repeat。"""
        task_id = getattr(self.conf, 'task_id', 'UNKNOWN')

        # 可视化：训练开始时打开持久窗口
        self._live_observer.start()

        for episode_idx in range(1, self.max_episodes + 1):
            if not self._should_continue(episode_idx):
                break

            eps = self.epsilon.get(self.env_steps)

            # 两阶段冻结训练：按局切 phase（phase1 只训 LD，phase2 解冻 WX）
            phase = self._phase_for_episode(episode_idx)
            if phase != self._current_phase:
                self.agents.set_phase(phase)
                self.env.set_phase(phase)
                self._current_phase = phase
                print(f"[RLTrainRunner] 进入 phase{phase} "
                      f"(episode {episode_idx}, phase1_episodes={self.phase1_episodes})")

            # 组装 hooks：态势日志 + 奖励分项诊断(可选) + ZMQ 推送（按推送间隔调度）
            hooks = [self._situation_log_hook]
            if self._breakdown_hook is not None:
                hooks.append(self._breakdown_hook)
            if self._zmq:
                hooks.extend(self._zmq.get_step_hooks(episode_idx))

            # 2D可视化：渲染局逐帧绘制，非渲染局内部只泵事件保持窗口响应
            self._live_observer.begin_episode(episode_idx, eps)
            render_cb = self._live_observer.render_callback

            ep_reward, step_count, ep_data = self.rollout_worker.generate_train_episode(
                epsilon=eps, episode_num=episode_idx,
                step_hooks=hooks,
                render_callback=render_cb,
            )

            self._live_observer.end_episode()

            self.env_steps += step_count
            print(f"强化学习环境构建的训练样本池样本量: {self.env_steps}")
            
            self.episode_rewards.append(ep_reward)
            self.buffer.store_episode(ep_data)

            loss = self._learn_batch(eps, self.utd_ratio)
            self.episode_losses.append(loss)

            print(f"[Episode {episode_idx}/{self.max_episodes}] ", f"ep_reward={ep_reward:.2f}, steps={step_count}, eps={eps:.3f}")

            is_last = (episode_idx == self.max_episodes)
            self._maybe_save(episode_idx, is_last)
            save_reward_plot(self.episode_rewards,
                             os.path.join(self.result_dir, "episode_reward.png"))
            save_loss_plot(self.episode_losses,
                           os.path.join(self.result_dir, "episode_loss.png"))

            if self.push is not None:
                self.push.push_metrics({
                    "episode": episode_idx, "reward": ep_reward,
                    "loss": loss if loss is not None else 0.0,
                    "planId": getattr(self.conf, 'plan_id', 'unknown'),
                })
            if self.push is not None and (
                is_last or (self.save_frequency > 0 and episode_idx % self.save_frequency == 0)
            ):
                self.push.push_weights({
                    "episode": episode_idx, "model_dir": self.model_dir,
                    "algo": "qmix", "task_id": task_id,
                    "planId": getattr(self.conf, 'plan_id', 'unknown'),
                })

        # 可视化：关闭持久窗口
        self._live_observer.shutdown()

        
        print("训练管理器：本轮训练结束。")

    # ============================================================
    # 内部
    # ============================================================

    def _should_push_episode(self, episode_idx: int) -> bool:
        """该局是否应推送态势（命中推送间隔或末局），与串行 get_step_hooks 同口径。"""
        push_interval = getattr(self.conf, 'push_interval', 0)
        if push_interval <= 0:
            return False
        return episode_idx % push_interval == 0 or episode_idx == self.max_episodes

    def _phase_for_episode(self, episode_idx: int) -> int:
        """根据 phase1_episodes 计算当前局所属训练阶段。

        - -1: 纯 phase1 标定（只训 LD，永不切 WX）；
        -  0: 直接联合训练（跳过 phase1）；
        -  N: 第 N 局起自动切 phase2（解冻 WX）。
        """
        p1 = self.phase1_episodes
        if p1 == 0:
            return 2
        if p1 == -1:
            return 1
        return 1 if episode_idx < p1 else 2

    def _write_train_meta(self, ec, ac, group_assignments) -> None:
        """将训练分组元数据落盘，供推演端自动恢复一致的 group_size。

        推演前端若漏传 group_size，EvalRunner 会读取此文件兜底，
        避免「推演与训练分组不一致 → 权重加载失败」。
        """
        meta = {
            "algorithm": "qmix",
            "group_size": int(getattr(ac, 'group_size', 0) or 0),
            "use_grouping": group_assignments is not None,
            "n_agents": int(ec.n_agents),
            "n_actions": int(ec.n_actions),
            "n_targets": int(ec.n_targets),
        }
        path = os.path.join(self.model_dir, "train_meta.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"[RLTrainRunner] 训练元数据已写入: {path} "
                  f"(group_size={meta['group_size']}, grouping={meta['use_grouping']})")
        except OSError as e:
            print(f"[RLTrainRunner] 训练元数据写入失败（不影响训练）: {e}")

    def _maybe_warm_start(self, ic) -> None:
        """若 load_dir 指定且存在，则用 IL 预训练权重热启动 DRQN。

        细节（文件名、state_dict 加载）下沉到算法层 load_drqn_state。
        """
        load_dir = getattr(ic, 'load_dir', '')
        if not load_dir or not os.path.exists(load_dir):
            return
        self.agents.policy.load_drqn_state(load_dir)
        print(f"[RLTrainRunner] 已用预训练权重热启动 DRQN: {load_dir}")

    def _learn_batch(self, epsilon: float, n_steps: int):
        """连续做 n_steps 步梯度更新（replay 复用，UTD ratio）。

        每步从 buffer 重新采样一个 batch（batch_size 局）；buffer 未攒满
        batch_size 时一步都不做、返回 None，否则返回最后一步的 loss。
        """
        if not self.buffer.can_sample(self.batch_size):
            return None
        last_loss = None
        
        #3. learn N 次（ratio = N）
        for _ in range(n_steps):
            batch = self.buffer.sample(self.batch_size) 
            last_loss = self.agents.train_(batch, self._update_steps, epsilon)  # forward + backward + step
            self._update_steps += 1
        return last_loss

    def _maybe_save(self, episode_idx, is_last):
        if is_last or (self.save_frequency > 0
                       and episode_idx % self.save_frequency == 0):
            self.agents.policy.save_model(episode_idx)
