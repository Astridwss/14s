# -*- coding: utf-8 -*-
"""
按「图清单与画图提示词.md」生成学术结构图 .drawio（diagrams.net 可编辑）。

风格对齐 transformer.png 的低饱和粉彩 + 圆角填充块 + 细黑边框（与 draw_thesis_figures.py 的 PNG 同源配色）。
输出到 docs/设计方案/figures/。
"""
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'docs', '设计方案', 'figures')

# 配色（与 PNG 版一致）
C_TEXT = '#1A1A1A'
C_EDGE = '#3A3A3C'
C_LINE = '#4A4A4C'
C_SUB = '#5A5A5A'
C_INPUT = '#E4DDD4'
C_TRUNK = '#C6BDDD'
C_LD = '#EBD6C8'
C_WX = '#D3E0CB'
C_MIXER = '#C9DCEA'
C_OUT = '#D8CBE4'
C_ACC = '#F0E0C8'
ACC_BORDER = '#8A6D3B'
FONT = 'SimSun'


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def sub(s):
    return '<sub>' + s + '</sub>'


def sup(s):
    return '<sup>' + s + '</sup>'


class D:
    def __init__(self, name, w, h):
        self.name = name
        self.w = w
        self.h = h
        self.cells = []
        self.cnt = 2

    def _id(self):
        self.cnt += 1
        return 'n%d' % self.cnt

    def box(self, x, y, w, h, label, c=C_INPUT, bold=False, fs=12, ec=None):
        i = self._id()
        ec = ec or C_EDGE
        st = ('rounded=1;arcSize=10;whiteSpace=wrap;html=1;align=center;verticalAlign=middle;'
              'fillColor=%s;strokeColor=%s;strokeWidth=1;fontColor=%s;fontFamily=%s;fontSize=%d;'
              % (c, ec, C_TEXT, FONT, fs))
        if bold:
            st += 'fontStyle=1;'
        self.cells.append(
            '<mxCell id="%s" value="%s" style="%s" vertex="1" parent="1">'
            '<mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/></mxCell>'
            % (i, esc(label), st, x, y, w, h))
        return i

    def text(self, x, y, w, h, label, fs=11, color=C_SUB, bold=False):
        i = self._id()
        st = ('text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;'
              'fontColor=%s;fontFamily=%s;fontSize=%d;' % (color, FONT, fs))
        if bold:
            st += 'fontStyle=1;'
        self.cells.append(
            '<mxCell id="%s" value="%s" style="%s" vertex="1" parent="1">'
            '<mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/></mxCell>'
            % (i, esc(label), st, x, y, w, h))
        return i

    def group(self, x, y, w, h, label=''):
        i = self._id()
        st = ('rounded=1;arcSize=8;whiteSpace=wrap;html=1;dashed=1;fillColor=none;'
              'strokeColor=#999999;strokeWidth=1;fontColor=#666666;fontFamily=%s;fontSize=11;'
              'verticalAlign=top;align=left;spacingLeft=8;spacingTop=4;' % FONT)
        self.cells.append(
            '<mxCell id="%s" value="%s" style="%s" vertex="1" parent="1">'
            '<mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/></mxCell>'
            % (i, esc(label), st, x, y, w, h))
        return i

    def edge(self, s, t, label='', dashed=False, accent=False, ex=(0.5, 1), en=(0.5, 0),
             exd=None, end=None):
        i = self._id()
        st = 'edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;'
        if dashed:
            st += 'dashed=1;'
        st += 'endArrow=%s;strokeColor=%s;strokeWidth=1;' % (
            'none' if dashed else 'block', ACC_BORDER if accent else C_LINE)
        st += 'exitX=%.3f;exitY=%.3f;entryX=%.3f;entryY=%.3f;' % (ex[0], ex[1], en[0], en[1])
        if exd:
            st += 'exitDx=%d;exitDy=0;' % exd
        if end:
            st += 'entryDx=%d;entryDy=0;' % end
        if label:
            st += 'fontColor=%s;fontSize=10;fontFamily=%s;' % (ACC_BORDER if accent else '#333333', FONT)
        self.cells.append(
            '<mxCell id="%s" value="%s" style="%s" edge="1" parent="1" source="%s" target="%s">'
            '<mxGeometry relative="1" as="geometry"/></mxCell>'
            % (i, esc(label), st, s, t))
        return i


def build(d):
    cells = '\n'.join(d.cells)
    return ('<mxfile host="app.diagrams.net" agent="python" version="24.0.0" type="device">\n'
            '  <diagram id="%s" name="%s">\n'
            '    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" '
            'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="%d" pageHeight="%d" '
            'math="0" shadow="0" background="#FFFFFF">\n'
            '      <root>\n'
            '        <mxCell id="0"/>\n'
            '        <mxCell id="1" parent="0"/>\n'
            '%s\n'
            '      </root>\n'
            '    </mxGraphModel>\n'
            '  </diagram>\n'
            '</mxfile>') % (d.name, d.name, d.w, d.h, cells)


def save(d, fn):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, fn)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(build(d))
    print('saved:', p)


# ---------------------------------------------------------------- 图 2-1 系统总体技术路线
def fig2_1():
    d = D('fig2-1 系统总体技术路线', 920, 1160)
    b = d.box
    e = d.edge

    l1 = b(250, 30, 420, 58,
           '前端任务管理平台<br><font style="font-size:10px">下发 task_id / plan_id / scene_url / 超参数</font>',
           C_INPUT)
    l2 = b(250, 140, 420, 62,
           '配置装配层 ConfigAssembler<br><font style="font-size:10px">YAML 基线 → 场景解析 → 四层合并 → 锁定</font>',
           C_INPUT)
    il = b(70, 290, 330, 100,
           '模仿学习（IL）阶段<br><font style="font-size:10px">专家预案 → 行为克隆<br>单头 DRQN 预训练</font>',
           C_LD)
    rl = b(520, 290, 330, 100,
           '强化学习（RL）阶段<br><font style="font-size:10px">H-QMIX 层次化值分解<br>+ 两阶段冻结训练</font>',
           C_WX)
    c1 = b(70, 470, 250, 104,
           '统一骨架 DRQN 双输出头<br><font style="font-size:10px">fc' + sub('1') + ' / LayerNorm / rnn</font>',
           C_TRUNK)
    c2 = b(335, 470, 250, 104,
           '混频器（训练期）<br><font style="font-size:10px">LD：Lower + Upper<br>WX：独立 QMIXNET</font>',
           C_MIXER)
    c3 = b(600, 470, 250, 104,
           '多进程并行 Rollout<br><font style="font-size:10px">+ UTD 数据复用<br>+ 态势倍速推送</font>',
           C_MIXER)
    infer = b(260, 660, 400, 64, '推理：仅 DRQN 前向 + 组内序列决策', C_TRUNK)
    out = b(260, 780, 400, 60,
            '输出 LD 多选指令 / WX 单选指令<br><font style="font-size:10px">混频器已移除，无需全局信息</font>',
            C_OUT, bold=True)

    e(l1, l2)
    e(l2, il, ex=(0.0, 1), en=(0.5, 0))
    e(l2, rl, ex=(1.0, 1), en=(0.5, 0))
    e(il, rl, label='主干权重迁移（fc' + sub('1') + ' / LayerNorm / rnn）',
      accent=True, ex=(1, 0.5), en=(0, 0.5))
    e(rl, c1, ex=(0.15, 1), en=(0.5, 0))
    e(rl, c2, ex=(0.5, 1), en=(0.5, 0))
    e(rl, c3, ex=(0.85, 1), en=(0.5, 0))
    e(c1, infer, ex=(0.5, 1), en=(0.35, 0))
    e(c2, infer, ex=(0.5, 1), en=(0.5, 0))
    e(c3, infer, ex=(0.5, 1), en=(0.65, 0))
    e(infer, out)
    save(d, 'fig2-1_系统总体技术路线.drawio')


# ---------------------------------------------------------------- 图 3-1 H-QMIX 总体结构
def fig3_1():
    d = D('fig3-1 H-QMIX 总体结构', 860, 1040)
    b = d.box
    e = d.edge
    t = d.text

    inp = b(310, 30, 240, 46, '输入 x' + sub('i') + ' ∈ ℝ' + sup('482'), C_INPUT)
    trunk = b(190, 124, 480, 70,
              '共享主干 DualHeadDRQN（283 个智能体共享参数）<br>'
              '<font style="font-size:10px">fc' + sub('1') + ' → LayerNorm → ReLU → GRU（隐 128）</font>',
              C_TRUNK, bold=True)
    ld = b(120, 262, 230, 92,
           'LD 头 fc' + sub('ld') + '<br><font style="font-size:10px">Linear(128, 21)<br>多标签 top-20</font>',
           C_LD)
    wx = b(510, 262, 230, 92,
           'WX 头 fc' + sub('wx') + '<br><font style="font-size:10px">Linear(128, 22)<br>单选 softmax</font>',
           C_WX)
    t(120, 370, 230, 26, '个体价值 Q' + sub('1') + ' ⋯ Q' + sub('200') + '（雷达）')
    t(510, 370, 230, 26, '个体价值 Q' + sub('201') + ' ⋯ Q' + sub('283') + '（卫星）')
    kmeans = b(90, 424, 290, 70,
               'K-Means 地理分组（静态、非可学习）<br><font style="font-size:10px">200 → 10 组 × 20</font>',
               C_MIXER)
    lower = b(90, 548, 290, 76,
              'LowerMixer 组内混频（×G 共享）<br>'
              '<font style="font-size:10px">组状态 s' + sub('g') + ' → 组价值 q' + sub('g') + '</font>',
              C_MIXER)
    upper = b(90, 678, 290, 76,
              'UpperMixer 组间混频（×1）<br>'
              '<font style="font-size:10px">全局状态 S → Q' + sub('tot') + sup('ld') + '</font>',
              C_MIXER)
    qmix = b(500, 430, 250, 130,
             '标准 QMIX 混频器<br><font style="font-size:10px">QMIXNET（n=83）<br>→ Q' + sub('tot') + sup('wx') + '</font>',
             C_MIXER)
    qtot = b(190, 836, 480, 60,
             'Q' + sub('tot') + ' = Q' + sub('tot') + sup('ld') + ' + 1[ phase ≥ 2 ] · Q' + sub('tot') + sup('wx'),
             C_OUT, bold=True)

    e(inp, trunk)
    e(trunk, ld, ex=(0.25, 1), en=(0.5, 0))
    e(trunk, wx, ex=(0.75, 1), en=(0.5, 0))
    e(ld, kmeans, ex=(0.5, 1), en=(0.5, 0))
    e(kmeans, lower)
    e(lower, upper)
    e(wx, qmix)
    e(upper, qtot, ex=(0.5, 1), en=(0.3, 0))
    e(qmix, qtot, ex=(0.5, 1), en=(0.7, 0))
    save(d, 'fig3-1_HQMIX总体结构.drawio')


# ---------------------------------------------------------------- 图 3-2 双头 DRQN 网络结构
def fig3_2():
    d = D('fig3-2 双头 DRQN 网络结构', 760, 940)
    b = d.box
    e = d.edge
    t = d.text

    inp = b(230, 30, 300, 50, '输入 x' + sub('i') + ' ∈ ℝ' + sup('482'), C_INPUT)
    t(230, 84, 300, 22, '（观测 177 + 上一动作 22 + 身份 283）')
    fc1 = b(250, 112, 260, 46, 'fc' + sub('1') + '：Linear(482, 128)', C_TRUNK)
    ln = b(250, 182, 260, 46, 'LayerNorm(128)', C_TRUNK)
    relu = b(250, 252, 260, 46, 'ReLU', C_TRUNK)
    gru = b(250, 322, 260, 46, 'GRUCell(128, 128)', C_TRUNK)
    t(250, 388, 260, 26, 'h' + sub('i') + ' ∈ ℝ' + sup('128') + '（共享表征）')
    ld = b(120, 442, 250, 104,
           'fc' + sub('ld') + '：Linear(128, 21)<br><font style="font-size:10px">逐目标独立 Q（无激活）<br>多标签 top-20 解码</font>',
           C_LD)
    wx = b(390, 442, 250, 104,
           'fc' + sub('wx') + '：Linear(128, 22)<br><font style="font-size:10px">单选 logits（无激活）<br>softmax → argmax 解码</font>',
           C_WX)
    act_ld = b(120, 588, 250, 46, '雷达动作（容量 20）', C_LD)
    act_wx = b(390, 588, 250, 46, '卫星动作（容量 1）', C_WX)

    e(inp, fc1)
    e(fc1, ln)
    e(ln, relu)
    e(relu, gru)
    e(gru, gru, label='隐状态 h' + sub('i'), dashed=True, ex=(1, 0.5), en=(1, 0.5), exd=26, end=26)
    e(gru, ld, ex=(0.2, 1), en=(0.5, 0))
    e(gru, wx, ex=(0.8, 1), en=(0.5, 0))
    e(ld, act_ld)
    e(wx, act_wx)
    save(d, 'fig3-2_双头DRQN.drawio')


# ---------------------------------------------------------------- 图 4-1 模仿学习阶段单头 DRQN
def fig4_1():
    d = D('fig4-1 模仿学习阶段单头 DRQN', 760, 850)
    b = d.box
    e = d.edge
    t = d.text

    inp = b(230, 30, 300, 52,
            '输入 x' + sub('i') + ' ∈ ℝ' + sup('482') + '<br>'
            '<font style="font-size:10px">观测 177 + 上一动作 22 + agent_id 283</font>', C_INPUT)
    d.group(210, 112, 340, 304, '共享主干')
    fc1 = b(250, 142, 260, 46, 'fc' + sub('1') + '(482→128)', C_TRUNK)
    ln = b(250, 212, 260, 46, 'LayerNorm', C_TRUNK)
    relu = b(250, 282, 260, 46, 'ReLU', C_TRUNK)
    gru = b(250, 352, 260, 46, 'GRU(128→128)', C_TRUNK)
    t(250, 434, 260, 26, 'h' + sub('i'))
    fc2 = b(250, 468, 260, 60,
            'fc' + sub('2') + '(128→22)<br><font style="font-size:10px">22 维 logits（待机 + 21 批目标）</font>',
            C_WX)
    sm = b(250, 560, 260, 46, 'softmax', C_MIXER)
    loss = b(250, 636, 260, 56, '交叉熵损失（对齐专家动作）', C_OUT, bold=True)

    e(inp, fc1)
    e(fc1, ln)
    e(ln, relu)
    e(relu, gru)
    e(gru, fc2)
    e(fc2, sm)
    e(sm, loss)
    save(d, 'fig4-1_模仿学习单头DRQN.drawio')


# ---------------------------------------------------------------- 图 4-2 完整训练流程
def fig4_2():
    d = D('fig4-2 完整训练流程', 820, 1000)
    b = d.box
    e = d.edge

    a = b(110, 30, 600, 130,
          '阶段 A：专家数据生成（IL 前置）<br><font style="font-size:10px">'
          '读预案 splitQuduanResult → 分段展开 → 逐时间步重建观测 →<br>'
          '提取专家动作 → 向量化 → 并行分块 → 拼接 CSV</font>', C_INPUT)
    bb = b(110, 216, 600, 120,
           '阶段 B：模仿学习预训练（IL）<br><font style="font-size:10px">'
           '单头 DRQN + 加权交叉熵 + 正样本过采样<br>'
           '训练 50 epoch，评价全局/追踪准确率，保存主干权重</font>', C_TRUNK)
    cc = b(110, 460, 600, 200,
           '阶段 C：强化学习训练（H-QMIX）<br><font style="font-size:10px">'
           'phase1（前 500 局）：只训 LD，WX 冻结<br>'
           'phase2（之后）：解冻 WX，联合训练<br>'
           '每局：并行 rollout → 三流奖励 → 存经验池 → UTD 次梯度更新<br>'
           '每 200 步硬同步 target 网络；ε 从 1.0 退火到 0.1</font>', C_WX)

    e(a, bb)
    e(bb, cc, label='只迁移 fc' + sub('1') + ' / layer_norm / rnn（主干热启动）', accent=True)
    save(d, 'fig4-2_完整训练流程.drawio')


# ---------------------------------------------------------------- 第 6 章 关键技术全景图（6 面板）
def fig6():
    d = D('第6章 关键技术全景图', 1880, 1440)
    b = d.box
    e = d.edge
    t = d.text
    g = d.group

    W, H = 890, 430
    cols = [40, 950]
    rows = [40, 500, 960]

    def panel(px, py, title):
        t(px + 10, py + 4, W - 20, 26, '<b>' + title + '</b>', fs=13, color='#222222', bold=True)
        g(px, py + 34, W, H - 34, '')

    # ---- 面板 1 总体技术架构
    px, py = cols[0], rows[0]
    panel(px, py, '总体技术架构')
    a1 = b(px + 60, py + 62, W - 120, 60,
           '应用层：训练 · 推理 · 评估<br><font style="font-size:10px">强化学习训练 / 推理预案生成 / 四指标加权评估</font>',
           C_INPUT)
    a2 = b(px + 60, py + 152, W - 120, 72,
           '算法层：层次化多智能体强化学习<br><font style="font-size:10px">双头 DRQN（策略网络） / H-QMIX（层次化值分解） / 三流奖励塑形</font>',
           C_TRUNK)
    a3 = b(px + 60, py + 254, W - 120, 72,
           '环境层：雷达-卫星-目标协同跟踪仿真引擎<br><font style="font-size:10px">观测构建 / 动作映射 / 仿真内核（冻结，不可改）</font>',
           C_MIXER)
    t(px + 60, py + 356, W - 120, 30, '四类掩码贯穿推理/续训：动作可用 · 智能体数量 · 分组 · 时间步')
    e(a1, a2)
    e(a2, a3)

    # ---- 面板 2 H-QMIX 层次化值分解
    px, py = cols[1], rows[0]
    panel(px, py, 'H-QMIX 层次化值分解结构')
    items = [
        ('组内序列化 DRQN ×K（覆盖掩码）', C_TRUNK),
        ('个体价值 Q' + sub('1') + ' … Q' + sub('K') + '（×K）', C_TRUNK),
        ('Lower Mixer 组内混频 ×G 共享<br><font style="font-size:10px">输入：组局部状态 s' + sub('g') + '</font>', C_MIXER),
        ('组价值 q' + sub('1') + ' … q' + sub('G') + '（×G）', C_MIXER),
        ('Upper Mixer 组间混频 ×1<br><font style="font-size:10px">输入：池化全局状态 S</font>', C_MIXER),
    ]
    b2 = []
    yy = py + 62
    for lab, c in items:
        b2.append(b(px + 60, yy, 320, 48, lab, c))
        yy += 58
    b2tot = b(px + 60, yy, 320, 48, 'Q' + sub('tot') + sup('ld'), C_OUT, bold=True)
    sat = b(px + 470, py + 62, 340, 48, 'DRQN 卫星输出头（单选）', C_WX)
    sat_q = b(px + 470, py + 128, 340, 48, '卫星个体价值 Q' + sub('i') + sup('wx') + '（×83）', C_WX)
    sat_mix = b(px + 470, py + 194, 340, 48, '卫星单调混频 ×1', C_WX)
    sat_tot = b(px + 470, py + 260, 340, 48, 'Q' + sub('tot') + sup('wx'), C_OUT, bold=True)
    t(px + 60, py + 396, W - 120, 24,
      '∂q' + sub('g') + '/∂Q' + sub('i') + ' ≥ 0   &   ∂Q' + sub('tot') + '/∂q' + sub('g') + ' ≥ 0'
      '   →   ∂Q' + sub('tot') + '/∂Q' + sub('i') + ' ≥ 0，全局 IGM 成立')
    for i in range(len(b2) - 1):
        e(b2[i], b2[i + 1])
    e(b2[-1], b2tot)
    e(sat, sat_q)
    e(sat_q, sat_mix)
    e(sat_mix, sat_tot)

    # ---- 面板 3 双头 DRQN
    px, py = cols[0], rows[1]
    panel(px, py, '双头 DRQN 结构（共享主干 + 双输出头）')
    p3in = b(px + 60, py + 62, 320, 52, '输入 482 维<br><font style="font-size:10px">观测 177 + 上一动作 + 智能体身份</font>', C_INPUT)
    p3tr = b(px + 60, py + 152, 320, 52, '共享主干（fc₁ → LayerNorm → ReLU → GRU）', C_TRUNK)
    p3ld = b(px + 40, py + 262, 300, 60, '雷达动作<br><font style="font-size:10px">容量 20 多标签</font>', C_LD)
    p3wx = b(px + 430, py + 262, 300, 60, '卫星动作<br><font style="font-size:10px">容量 1 单选</font>', C_WX)
    t(px + 60, py + 360, W - 120, 28, '感知层：共享   /   决策层：分离   ·   两类价值不可通约，须独立输出')
    e(p3in, p3tr)
    e(p3tr, p3ld, ex=(0.25, 1), en=(0.5, 0))
    e(p3tr, p3wx, ex=(0.75, 1), en=(0.5, 0))

    # ---- 面板 4 训练流程
    px, py = cols[1], rows[1]
    panel(px, py, '训练流程（专家知识 → 可用模型）')
    s1 = b(px + 30, py + 62, 220, 64, '历史专家预案<br><font style="font-size:10px">装备 → 目标 → 时间段分配表</font>', C_INPUT)
    s2 = b(px + 280, py + 62, 220, 64, '解析为逐时间步<br><font style="font-size:10px">（观测，动作）监督样本</font>', C_INPUT)
    s3 = b(px + 530, py + 62, 220, 64, '行为克隆预训练（IL）<br><font style="font-size:10px">单输出头 · 类别加权交叉熵</font>', C_TRUNK)
    s4 = b(px + 280, py + 200, 220, 64, '主干权重迁移<br><font style="font-size:10px">迁主干 · 头与混频随机初始化</font>', C_ACC, ec=ACC_BORDER)
    s5 = b(px + 530, py + 200, 220, 64, '两阶段冻结强化学习训练', C_WX)
    s5a = b(px + 530, py + 290, 220, 44, '阶段一（前 500 局）只训雷达 · 卫星冻结', C_WX)
    s5b = b(px + 530, py + 346, 220, 44, '阶段二 解冻卫星 · 联合训练 · 卫星高探索', C_WX)
    s6 = b(px + 280, py + 340, 220, 56, '推理与评估<br><font style="font-size:10px">四指标加权</font>', C_OUT, bold=True)
    t(px + 30, py + 300, 200, 40, '冷启动：专家 → 初值<br><font style="font-size:10px">主干降学习率</font>')
    e(s1, s2, ex=(1, 0.5), en=(0, 0.5))
    e(s2, s3, ex=(1, 0.5), en=(0, 0.5))
    e(s3, s4, ex=(0, 1), en=(0.5, 0))
    e(s4, s5, ex=(1, 0.5), en=(0, 0.5))
    e(s5, s5a)
    e(s5a, s5b)
    e(s4, s6, ex=(0.5, 1), en=(0.5, 0))

    # ---- 面板 5 三流奖励塑形
    px, py = cols[0], rows[2]
    panel(px, py, '三流奖励塑形')
    r0 = b(px + 60, py + 54, W - 120, 46,
           '单步奖励  R = R' + sub('ld') + ' + R' + sub('wx') + ' + R' + sub('switch'), C_ACC, bold=True, ec=ACC_BORDER)
    r1 = b(px + 40, py + 122, 260, 152,
           'R' + sub('ld') + ' 雷达流（多标签 · 逐目标）<br><font style="font-size:10px">'
           '漏检惩罚：可见却 0 锁<br>覆盖重数奖励：线性到封顶<br>瞎指惩罚：锁不可见目标<br>边缘锁定惩罚：让渡卫星</font>',
           C_LD)
    r2 = b(px + 310, py + 122, 260, 152,
           'R' + sub('wx') + ' 卫星流（单选 · 逐卫星）<br><font style="font-size:10px">'
           '补盲奖励：盲区目标<br>预警接力奖励：边缘目标<br>瞎指惩罚<br>机会成本惩罚：空置不用</font>',
           C_WX)
    r3 = b(px + 580, py + 122, 260, 152,
           'R' + sub('switch') + ' 切换流（槽位级）<br><font style="font-size:10px">'
           '掉锁惩罚：频繁换锁抖动<br>硬中断惩罚：覆盖由正变 0<br>反刷分：边缘几何定价<br>反刷分：终态护栏</font>',
           C_MIXER)
    t(px + 60, py + 292, W - 120, 40, '逐项对齐四指标评价：覆盖率 / 中断次数 / 覆盖重数 / 跟踪覆盖率')
    e(r0, r1, ex=(0.15, 1), en=(0.5, 0))
    e(r0, r2, ex=(0.5, 1), en=(0.5, 0))
    e(r0, r3, ex=(0.85, 1), en=(0.5, 0))

    # ---- 面板 6 多进程并行化训练架构
    px, py = cols[1], rows[2]
    panel(px, py, '多进程并行化训练架构')
    procs = [b(px + 40, py + 72 + i * 60, 200, 48,
               '采样进程 ' + str(i + 1) + '<br><font style="font-size:10px">独立仿真副本</font>', C_INPUT)
             for i in range(4)]
    pool = b(px + 320, py + 150, 180, 56, '轨迹池<br><font style="font-size:10px">经验回放</font>', C_MIXER)
    npu = b(px + 560, py + 140, 240, 76,
            'NPU / GPU 多卡<br><font style="font-size:10px">worker 按启动序轮询认领物理卡</font>', C_TRUNK)
    push = b(px + 320, py + 280, 300, 70,
             '态势推送（独立线程）<br><font style="font-size:10px">ZMQ 轨迹池 · 倍速回放 · 与训练解耦</font>', C_WX)
    t(px + 40, py + 362, W - 120, 30, 'UTD 数据复用：每采 1 局 → 多次梯度更新（喂饱加速器）', bold=True, color=C_TEXT)
    for i in range(len(procs) - 1):
        e(procs[i], procs[i + 1])
    e(procs[-1], pool, ex=(1, 0.5), en=(0, 0.5))
    e(pool, npu, ex=(1, 0.5), en=(0, 0.5))
    e(pool, push, ex=(0.5, 1), en=(0.5, 0))
    save(d, '第6章_关键技术全景图.drawio')


if __name__ == '__main__':
    fig2_1()
    fig3_1()
    fig3_2()
    fig4_1()
    fig4_2()
    fig6()
