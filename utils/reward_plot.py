"""训练曲线绘图工具 —— 纯函数，无状态。"""
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_reward_plot(rewards: list, save_path: str):
    """将奖励曲线保存为 PNG。"""
    if not rewards:
        return
    x = list(range(1, len(rewards) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(x, rewards, label="episode reward", color='blue', linewidth=1.5)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Episode Total Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_loss_plot(losses: list, save_path: str):
    """将 TD loss 曲线保存为 PNG。

    loss 可能含 None（buffer 未攒满 batch_size 前的空值），统一转成 NaN，
    让 matplotlib 在该处断线，不影响其余段绘制。
    """
    if not losses:
        return
    ys = [float('nan') if l is None else float(l) for l in losses]
    if all(math.isnan(y) for y in ys):
        return
    x = list(range(1, len(ys) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(x, ys, label="TD loss", color='red', linewidth=1.5)
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("TD Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
