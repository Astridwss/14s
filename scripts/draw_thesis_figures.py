# -*- coding: utf-8 -*-
"""
学术结构图绘制（参考 Transformer 架构图风格）。

风格要点（对照 transformer.png 的柔和低饱和配色）：
- 白底 + 粉彩填充的圆角模块块 + 细黑边框与文字，留白充足；
- 模块按「功能族」分色（输入/主干/雷达头/卫星头/混频器/终端）；
- 主信息流用细直线箭头，循环/残差跳接用右侧圆弧箭头（类 Transformer 的 skip）；
- 字体宋体 + STIX 数学。

用法：
    python scripts/draw_thesis_figures.py            # 全部
    python scripts/draw_thesis_figures.py fig3_1     # 指定
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, 'docs', '设计方案', 'figures')

_FONT_DIR = 'C:/Windows/Fonts'
for _f in ['simsun.ttc', 'simhei.ttf', 'times.ttf', 'cambria.ttc']:
    _p = os.path.join(_FONT_DIR, _f)
    if os.path.exists(_p):
        try:
            font_manager.fontManager.addfont(_p)
        except Exception:
            pass

plt.rcParams['font.family'] = ['SimSun', 'SimHei', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------- 配色（取自 transformer.png，低饱和粉彩）
C_TEXT = '#1A1A1A'
C_EDGE = '#3A3A3C'
C_LINE = '#4A4A4C'
C_SUB = '#5A5A5A'

C_INPUT = '#E4DDD4'   # 米色 —— 输入/嵌入
C_TRUNK = '#C6BDDD'   # 淡紫 —— 共享主干
C_LD = '#EBD6C8'      # 淡橙 —— 雷达分支 / LD 头
C_WX = '#D3E0CB'      # 淡绿 —— 卫星分支 / WX 头
C_MIXER = '#C9DCEA'   # 淡蓝 —— 混频器
C_OUT = '#D8CBE4'     # 紫 —— 终端/输出
C_ACC = '#F0E0C8'     # 强调（迁移主线 / 关键结果）

FS = 10.5
FS_SUB = 8.0


def new_ax(w=7.4, h=6.6):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100.0 * h / w)
    ax.axis('off')
    return fig, ax


def block(ax, x, y, w, h, title, color, sub=None, fs=FS, subfs=FS_SUB,
          bold=False, ec=None, tc=C_TEXT, lw=1.2, zorder=3):
    """以 (x,y) 为中心的圆角填充块；title 为主标签，sub 为下方小字注释。"""
    ec = ec or C_EDGE
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle='round,pad=0.25,rounding_size=1.3',
                       fc=color, ec=ec, lw=lw, zorder=zorder, mutation_aspect=1.0)
    ax.add_patch(p)
    if sub:
        ax.text(x, y + 0.9, title, ha='center', va='center', fontsize=fs, color=tc,
                zorder=zorder + 1, fontweight='bold' if bold else 'normal')
        ax.text(x, y - 1.3, sub, ha='center', va='center', fontsize=subfs, color=C_SUB,
                zorder=zorder + 1, linespacing=1.3)
    else:
        ax.text(x, y, title, ha='center', va='center', fontsize=fs, color=tc,
                zorder=zorder + 1, fontweight='bold' if bold else 'normal', linespacing=1.35)
    return p


def label(ax, x, y, text, fs=FS_SUB, color=C_SUB, ha='center', va='center', zorder=4):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, zorder=zorder, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, color=C_LINE, lw=1.1, zorder=2, ms=11):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=ms,
                        lw=lw, color=color, zorder=zorder, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def curved(ax, x1, y1, x2, y2, rad=0.35, color=C_LINE, lw=1.0, zorder=2, ms=11):
    """右侧圆弧跳接（类 Transformer 残差/skip 连接）。"""
    a = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle='arc3,rad=%g' % rad,
                        arrowstyle='-|>', mutation_scale=ms, lw=lw, color=color,
                        zorder=zorder, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def elbow(ax, points, color=C_LINE, lw=1.1, zorder=2, ms=11):
    verts = list(points)
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
    ax.add_patch(PathPatch(Path(verts, codes), fc='none', ec=color, lw=lw, zorder=zorder))
    x1, y1 = verts[-2]
    x2, y2 = verts[-1]
    arrow(ax, x1, y1, x2, y2, color=color, lw=lw, zorder=zorder, ms=ms)


def save(fig, name, dpi=300):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.06, facecolor='white')
    plt.close(fig)
    print('saved:', path)
    return path


# ---------------------------------------------------------------- 图 3-2 双头 DRQN（旗舰，最贴近 Transformer）
def fig3_2():
    fig, ax = new_ax(w=6.6, h=7.4)
    X, LD, WX = 50, 29, 72

    block(ax, X, 103, 46, 7.5, r'输入 $x_i \in \mathbb{R}^{482}$', C_INPUT,
          sub=r'观测 177 + 上一动作 22 + 身份 283')
    arrow(ax, X, 99, X, 96.5)
    block(ax, X, 92.5, 40, 6, r'$fc_1$：$Linear(482,128)$', C_TRUNK)
    arrow(ax, X, 89.5, X, 87)
    block(ax, X, 84, 40, 6, r'$LayerNorm(128)$', C_TRUNK)
    arrow(ax, X, 81, X, 78.5)
    block(ax, X, 75.5, 40, 6, r'$ReLU$', C_TRUNK)
    arrow(ax, X, 72.5, X, 70)
    block(ax, X, 67, 40, 6, r'$GRUCell(128,128)$', C_TRUNK)

    # GRU 循环回边（右侧圆弧）
    curved(ax, X + 20, 67, X + 20, 84, rad=-0.4, color=C_LINE, lw=1.0)
    label(ax, X + 24.5, 75.5, r'隐状态 $h_i$', ha='left')

    label(ax, X, 60, r'$h_i \in \mathbb{R}^{128}$（共享表征）')
    arrow(ax, X, 63.5, X, 61.5)

    block(ax, LD, 49, 38, 12.5, r'$fc_{ld}$：$Linear(128,21)$', C_LD,
          sub=r'逐目标独立 $Q$（无激活）' + '\n' + r'多标签 top-20 解码')
    block(ax, WX, 49, 38, 12.5, r'$fc_{wx}$：$Linear(128,22)$', C_WX,
          sub=r'单选 logits（无激活）' + '\n' + r'softmax → argmax 解码')
    elbow(ax, [(X - 3, 58.5), (LD, 58.5), (LD, 55.25)])
    elbow(ax, [(X + 3, 58.5), (WX, 58.5), (WX, 55.25)])

    block(ax, LD, 36, 38, 6, r'雷达动作（容量 20）', C_LD)
    block(ax, WX, 36, 38, 6, r'卫星动作（容量 1）', C_WX)
    arrow(ax, LD, 42.75, LD, 39)
    arrow(ax, WX, 42.75, WX, 39)

    save(fig, 'fig3-2_dualhead_drqn.png')


# ---------------------------------------------------------------- 图 3-1 H-QMIX 总体结构
def fig3_1():
    fig, ax = new_ax(w=7.8, h=7.8)
    LD, WX, CT = 28, 72, 50

    block(ax, CT, 98, 34, 6, r'输入 $x_i \in \mathbb{R}^{482}$', C_INPUT)
    arrow(ax, CT, 95, CT, 92.75)
    block(ax, CT, 88, 66, 9, r'共享主干 DualHeadDRQN（283 个智能体共享参数）', C_TRUNK,
          sub=r'$fc_1 \rightarrow LayerNorm \rightarrow ReLU \rightarrow GRU$（隐 128）', bold=True)
    arrow(ax, 40, 85, LD, 79)   # 主干左出 → LD 头
    arrow(ax, 61, 85, WX, 79)   # 主干右出 → WX 头

    block(ax, LD, 72, 31, 11, r'LD 头 $fc_{ld}$', C_LD,
          sub=r'$Linear(128,21)$' + '\n' + r'多标签 top-20')
    block(ax, WX, 72, 31, 11, r'WX 头 $fc_{wx}$', C_WX,
          sub=r'$Linear(128,22)$' + '\n' + r'单选 softmax')
    label(ax, LD - 2, 64.5, r'个体价值 $Q_1\cdots Q_{200}$', ha='right')
    label(ax, WX + 2, 64.5, r'个体价值 $Q_{201}\cdots Q_{283}$', ha='left')
    arrow(ax, LD, 66.5, LD, 59)
    arrow(ax, WX, 66.5, WX, 57)

    block(ax, LD, 54, 44, 8, r'K-Means 地理分组（静态、非可学习）', C_MIXER,
          sub=r'$200 \rightarrow 10$ 组 $\times\,20$')
    arrow(ax, LD, 50, LD, 46.5)
    block(ax, LD, 41.5, 42, 9, r'LowerMixer 组内混频（$\times G$ 共享）', C_MIXER,
          sub=r'组状态 $s_g$ → 组价值 $q_g$')
    arrow(ax, LD, 37, LD, 33.5)
    block(ax, LD, 29, 42, 9, r'UpperMixer 组间混频（$\times 1$）', C_MIXER,
          sub=r'全局状态 $S$ → $Q_{tot}^{ld}$')

    block(ax, WX, 48, 36, 15, r'标准 QMIX 混频器', C_MIXER,
          sub=r'QMIXNET（$n=83$）' + '\n' + r'→ $Q_{tot}^{wx}$')

    block(ax, CT, 11, 68, 8, r'$Q_{tot} = Q_{tot}^{ld} + \mathbf{1}[\,phase \geq 2\,]\cdot Q_{tot}^{wx}$',
          C_OUT, bold=True)
    elbow(ax, [(LD, 24.5), (LD, 18), (CT, 18), (CT, 15)])
    elbow(ax, [(WX, 40.5), (WX, 18), (CT, 18)])

    save(fig, 'fig3-1_hqmix.png')


# ---------------------------------------------------------------- 图 4-1 模仿学习阶段单头 DRQN
def fig4_1():
    fig, ax = new_ax(w=6.6, h=7.0)
    X = 50

    block(ax, X, 101, 46, 8, r'输入 $x_i \in \mathbb{R}^{482}$', C_INPUT,
          sub=r'观测 177 + 上一动作 22 + agent_id 283')
    arrow(ax, X, 96.5, X, 94)
    block(ax, X, 90.5, 40, 6, r'$fc_1(482 \rightarrow 128)$', C_TRUNK)
    arrow(ax, X, 87.5, X, 85)
    block(ax, X, 82, 40, 6, r'$LayerNorm$', C_TRUNK)
    arrow(ax, X, 79, X, 76.5)
    block(ax, X, 73.5, 40, 6, r'$ReLU$', C_TRUNK)
    arrow(ax, X, 70.5, X, 68)
    block(ax, X, 65, 40, 6, r'$GRU(128 \rightarrow 128)$', C_TRUNK)

    # 共享主干虚线框
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((X - 24, 63.5), 48, 31.5, fc='none', ec='#999999',
                           lw=1.0, ls=(0, (4, 3)), zorder=1))
    label(ax, X - 22, 93.5, '共享主干', ha='left', color='#666666')

    label(ax, X, 58.5, r'$h_i$')
    arrow(ax, X, 61.5, X, 59.5)
    block(ax, X, 53, 40, 8, r'$fc_2(128 \rightarrow 22)$', C_WX,
          sub=r'22 维 logits（待机 + 21 批目标）')
    arrow(ax, X, 49, X, 46.5)
    block(ax, X, 43, 40, 6, r'$softmax$', C_MIXER)
    arrow(ax, X, 40, X, 37.5)
    block(ax, X, 33.5, 40, 7, r'交叉熵损失（对齐专家动作）', C_OUT, bold=True)

    save(fig, 'fig4-1_singlehead_drqn.png')


# ---------------------------------------------------------------- 图 2-1 系统总体技术路线
def fig2_1():
    fig, ax = new_ax(w=8.2, h=7.8)
    CT = 50

    block(ax, CT, 99, 46, 6.5, r'前端任务管理平台', C_INPUT,
          sub=r'下发 task_id / plan_id / scene_url / 超参数')
    arrow(ax, CT, 95.5, CT, 93)
    block(ax, CT, 89.5, 46, 6.5, r'配置装配层 ConfigAssembler', C_INPUT,
          sub=r'YAML 基线 → 场景解析 → 四层合并 → 锁定')

    IL, RL = 30, 70
    arrow(ax, 40, 86, IL, 82)   # 配置 → IL
    arrow(ax, 60, 86, RL, 82)   # 配置 → RL
    block(ax, IL, 76, 34, 11, r'模仿学习（IL）阶段', C_LD,
          sub=r'专家预案 → 行为克隆' + '\n' + r'单头 DRQN 预训练')
    block(ax, RL, 76, 34, 11, r'强化学习（RL）阶段', C_WX,
          sub=r'H-QMIX 层次化值分解' + '\n' + r'+ 两阶段冻结训练')
    # 迁移主线（强调色，粗箭头）
    a = FancyArrowPatch((IL + 17, 76), (RL - 17, 76), arrowstyle='-|>',
                        mutation_scale=13, lw=1.8, color='#8A6D3B', zorder=2)
    ax.add_patch(a)
    label(ax, CT, 79.5, r'主干权重迁移（$fc_1$ / LayerNorm / rnn）', color='#8A6D3B', fs=8.5)

    # RL 下三支撑件
    c1, c2, c3 = 24, 50, 76
    arrow(ax, RL, 70.5, RL, 66)
    block(ax, c1, 58, 30, 11, r'统一骨架 DRQN 双输出头', C_TRUNK,
          sub=r'$fc_1$ / LayerNorm / rnn')
    block(ax, c2, 58, 30, 11, r'混频器（训练期）', C_MIXER,
          sub=r'LD：Lower + Upper' + '\n' + r'WX：独立 QMIXNET')
    block(ax, c3, 58, 30, 11, r'多进程并行 Rollout', C_MIXER,
          sub=r'+ UTD 数据复用' + '\n' + r'+ 态势倍速推送')
    elbow(ax, [(RL, 63.5), (RL, 64.5), (c1, 64.5), (c1, 63.5)])
    elbow(ax, [(RL, 63.5), (RL, 64.5), (c2, 64.5), (c2, 63.5)])
    elbow(ax, [(RL, 63.5), (RL, 64.5), (c3, 64.5), (c3, 63.5)])

    block(ax, CT, 40, 52, 7, r'推理：仅 DRQN 前向 + 组内序列决策', C_TRUNK)
    elbow(ax, [(c1, 52.5), (c1, 47), (CT, 47), (CT, 43.5)])
    elbow(ax, [(c2, 52.5), (c2, 47), (CT, 47)])
    elbow(ax, [(c3, 52.5), (c3, 47), (CT, 47)])

    arrow(ax, CT, 36.5, CT, 34)
    block(ax, CT, 30, 52, 7, r'输出 LD 多选指令 / WX 单选指令', C_OUT,
          sub=r'混频器已移除，无需全局信息', bold=True)

    save(fig, 'fig2-1_system_route.png')


# ---------------------------------------------------------------- 图 4-2 完整训练流程
def fig4_2():
    fig, ax = new_ax(w=7.6, h=8.0)
    CT = 50

    block(ax, CT, 96, 64, 12, r'阶段 A：专家数据生成（IL 前置）', C_INPUT,
          sub=r'读预案 splitQuduanResult → 分段展开 → 逐时间步重建观测 →' + '\n'
               r'提取专家动作 → 向量化 → 并行分块 → 拼接 CSV')
    arrow(ax, CT, 89.5, CT, 86.5)
    block(ax, CT, 78, 64, 12, r'阶段 B：模仿学习预训练（IL）', C_TRUNK,
          sub=r'单头 DRQN + 加权交叉熵 + 正样本过采样' + '\n'
               r'训练 50 epoch，评价全局/追踪准确率，保存主干权重')
    # 迁移箭头
    a = FancyArrowPatch((CT, 71.5), (CT, 66.5), arrowstyle='-|>', mutation_scale=13,
                        lw=1.8, color='#8A6D3B', zorder=2)
    ax.add_patch(a)
    label(ax, CT + 26, 69, r'只迁移 $fc_1$ / layer_norm / rnn', color='#8A6D3B', ha='left')
    label(ax, CT + 26, 66.5, r'（主干热启动）', color='#8A6D3B', ha='left')

    block(ax, CT, 56, 64, 18, r'阶段 C：强化学习训练（H-QMIX）', C_WX,
          sub=r'phase1（前 500 局）：只训 LD，WX 冻结' + '\n'
               r'phase2（之后）：解冻 WX，联合训练' + '\n'
               r'每局：并行 rollout → 三流奖励 → 存经验池 → UTD 次梯度更新' + '\n'
               r'每 200 步硬同步 target 网络；$\varepsilon$ 从 1.0 退火到 0.1')

    save(fig, 'fig4-2_training_flow.png')


# ---------------------------------------------------------------- 第 6 章 关键技术全景图（6 面板）
def fig6():
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2))
    for axr in axes:
        for ax in axr:
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.axis('off')

    def pt(ax, x, y, s, **kw):
        kw.setdefault('color', C_SUB)
        ax.text(x, y, s, ha='center', va='center', fontsize=8.0, zorder=4,
                linespacing=1.3, **kw)

    # 面板 1 总体技术架构
    ax = axes[0][0]
    ax.text(50, 96, '总体技术架构', ha='center', va='center', fontsize=12, fontweight='bold', color=C_TEXT)
    block(ax, 50, 84, 74, 12, '应用层：训练 · 推理 · 评估', C_INPUT, sub='强化学习训练 / 推理预案生成 / 四指标加权评估')
    block(ax, 50, 64, 74, 15, '算法层：层次化多智能体强化学习', C_TRUNK, sub='双头 DRQN（策略网络）/ H-QMIX（层次化值分解）/ 三流奖励塑形')
    block(ax, 50, 42, 74, 15, '环境层：雷达-卫星-目标协同跟踪仿真引擎', C_MIXER, sub='观测构建 / 动作映射 / 仿真内核（冻结，不可改）')
    pt(ax, 50, 20, '四类掩码贯穿推理/续训：动作可用 · 智能体数量 · 分组 · 时间步')
    arrow(ax, 50, 77.5, 50, 72)
    arrow(ax, 50, 55.5, 50, 50)

    # 面板 2 H-QMIX 层次化值分解
    ax = axes[0][1]
    ax.text(50, 96, 'H-QMIX 层次化值分解结构', ha='center', va='center', fontsize=12, fontweight='bold', color=C_TEXT)
    ys = [84, 74, 64, 54, 44]
    labs = [
        (r'组内序列化 DRQN $\times K$（覆盖掩码）', C_TRUNK, None),
        (r'个体价值 $Q_1\cdots Q_K$（$\times K$）', C_TRUNK, None),
        (r'Lower Mixer 组内混频 $\times G$ 共享', C_MIXER, r'输入：组局部状态 $s_g$'),
        (r'组价值 $q_1\cdots q_G$（$\times G$）', C_MIXER, None),
        (r'Upper Mixer 组间混频 $\times 1$', C_MIXER, r'输入：池化全局状态 $S$'),
    ]
    for i, (t, c, s) in enumerate(labs):
        block(ax, 50, ys[i], 58, 7.5, t, c, sub=s)
    block(ax, 50, 32, 58, 7.5, r'$Q_{tot}^{ld}$', C_OUT, bold=True)
    block(ax, 78, 84, 30, 9, r'DRQN 卫星输出头（单选）', C_WX)
    block(ax, 78, 72, 30, 9, r'卫星个体价值 $\times 83$', C_WX)
    block(ax, 78, 60, 30, 9, r'卫星单调混频 $\times 1$', C_WX)
    block(ax, 78, 48, 30, 9, r'$Q_{tot}^{wx}$', C_OUT, bold=True)
    for i in range(4):
        arrow(ax, 50, ys[i] - 4, 50, ys[i + 1] + 4)
    arrow(ax, 50, ys[4] - 4, 50, 36)
    arrow(ax, 78, 79, 78, 77)
    arrow(ax, 78, 67, 78, 65)
    arrow(ax, 78, 55, 78, 53)
    pt(ax, 50, 14, r'$\partial q_g/\partial Q_i \geq 0$，$\partial Q_{tot}/\partial q_g \geq 0$'
                  + '\n' + r'链式传导 $\Rightarrow$ $\partial Q_{tot}/\partial Q_i \geq 0$，全局 IGM 成立')

    # 面板 3 双头 DRQN
    ax = axes[0][2]
    ax.text(50, 96, '双头 DRQN 结构（共享主干 + 双输出头）', ha='center', va='center', fontsize=12, fontweight='bold', color=C_TEXT)
    block(ax, 50, 84, 60, 10, r'输入 482 维', C_INPUT, sub='观测 177 + 上一动作 + 智能体身份')
    block(ax, 50, 68, 60, 10, r'共享主干（$fc_1$ → LayerNorm → ReLU → GRU）', C_TRUNK)
    block(ax, 28, 48, 38, 12, r'雷达动作', C_LD, sub='容量 20 多标签')
    block(ax, 72, 48, 38, 12, r'卫星动作', C_WX, sub='容量 1 单选')
    arrow(ax, 50, 78.5, 50, 73.5)
    elbow(ax, [(42, 63), (28, 63), (28, 54)])
    elbow(ax, [(58, 63), (72, 63), (72, 54)])
    pt(ax, 50, 26, '感知层：共享   /   决策层：分离\n两类价值不可通约，须独立输出')

    # 面板 4 训练流程
    ax = axes[1][0]
    ax.text(50, 96, '训练流程（专家知识 → 可用模型）', ha='center', va='center', fontsize=12, fontweight='bold', color=C_TEXT)
    block(ax, 16, 84, 26, 11, r'历史专家预案', C_INPUT, sub='装备 → 目标 → 时间段')
    block(ax, 50, 84, 26, 11, r'解析逐时间步', C_INPUT, sub='（观测，动作）监督样本')
    block(ax, 84, 84, 26, 11, r'行为克隆预训练（IL）', C_TRUNK, sub='单输出头 · 类别加权交叉熵')
    block(ax, 50, 62, 26, 11, r'主干权重迁移', C_ACC, sub='迁主干 · 头与混频随机初始化')
    block(ax, 84, 62, 26, 11, r'两阶段冻结 RL 训练', C_WX, sub='phase1 只训雷达\nphase2 联合训练')
    block(ax, 50, 40, 26, 11, r'推理与评估', C_OUT, sub='四指标加权', bold=True)
    arrow(ax, 30, 84, 36, 84)
    arrow(ax, 64, 84, 70, 84)
    elbow(ax, [(84, 78), (84, 73), (50, 73), (50, 68)])
    elbow(ax, [(50, 56), (50, 51), (84, 51), (84, 68)])  # 迁移 → RL
    arrow(ax, 84, 56, 84, 68)
    elbow(ax, [(50, 56), (50, 46)])

    # 面板 5 三流奖励塑形
    ax = axes[1][1]
    ax.text(50, 96, '三流奖励塑形', ha='center', va='center', fontsize=12, fontweight='bold', color=C_TEXT)
    block(ax, 50, 86, 70, 8, r'单步奖励  $R = R_{ld} + R_{wx} + R_{switch}$', C_ACC, bold=True)
    block(ax, 18, 60, 32, 34, r'$R_{ld}$ 雷达流', C_LD, sub='多标签 · 逐目标\n漏检惩罚：可见却 0 锁\n覆盖重数奖励：线性到封顶\n瞎指惩罚：锁不可见目标\n边缘锁定惩罚：让渡卫星')
    block(ax, 50, 60, 32, 34, r'$R_{wx}$ 卫星流', C_WX, sub='单选 · 逐卫星\n补盲奖励：盲区目标\n预警接力奖励：边缘目标\n瞎指惩罚\n机会成本惩罚：空置不用')
    block(ax, 82, 60, 32, 34, r'$R_{switch}$ 切换流', C_MIXER, sub='槽位级\n掉锁惩罚：频繁换锁抖动\n硬中断惩罚：覆盖由正变 0\n反刷分：边缘几何定价\n反刷分：终态护栏')
    arrow(ax, 36, 82, 18, 77)
    arrow(ax, 50, 82, 50, 77)
    arrow(ax, 64, 82, 82, 77)
    pt(ax, 50, 20, '逐项对齐四指标评价：覆盖率 / 中断次数 / 覆盖重数 / 跟踪覆盖率')

    # 面板 6 多进程并行化训练架构
    ax = axes[1][2]
    ax.text(50, 96, '多进程并行化训练架构', ha='center', va='center', fontsize=12, fontweight='bold', color=C_TEXT)
    for i in range(4):
        block(ax, 16, 86 - i * 13, 26, 9, r'采样进程 %d' % (i + 1), C_INPUT, sub='独立仿真副本')
    block(ax, 50, 70, 28, 12, r'轨迹池', C_MIXER, sub='经验回放')
    block(ax, 82, 70, 28, 14, r'NPU / GPU 多卡', C_TRUNK, sub='worker 按启动序\n轮询认领物理卡')
    block(ax, 50, 44, 42, 12, r'态势推送（独立线程）', C_WX, sub='ZMQ 轨迹池 · 倍速回放 · 与训练解耦')
    elbow(ax, [(29, 70), (35, 70), (35, 70), (36, 70)])
    arrow(ax, 30, 70, 35, 70)
    arrow(ax, 64, 70, 67, 70)
    elbow(ax, [(50, 64), (50, 50)])
    pt(ax, 50, 20, 'UTD 数据复用：每采 1 局 → 多次梯度更新（喂饱加速器）', color=C_TEXT)

    save(fig, 'fig6_key_tech.png')


_FIGS = {
    'fig2_1': fig2_1,
    'fig3_1': fig3_1,
    'fig3_2': fig3_2,
    'fig4_1': fig4_1,
    'fig4_2': fig4_2,
    'fig6': fig6,
}


def main():
    only = [a for a in sys.argv[1:] if not a.startswith('-')]
    if only:
        for name in only:
            _FIGS[name]()
    else:
        for fn in _FIGS.values():
            fn()


if __name__ == '__main__':
    main()
