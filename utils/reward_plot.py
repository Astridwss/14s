"""奖励曲线绘图工具 —— 纯函数，无状态。"""
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
