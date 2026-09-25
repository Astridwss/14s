# -*- coding: utf-8 -*-
"""
将 docs/设计方案/第1-6章_*.md 按学术论文格式合并为单个 Word 文档。

新增特性（相对早期版本）：
- 参考文献合并：各章章末的「参考文献（第 N 章）」小节被剥离，合并为文末唯一的
  「参考文献」列表，按全局编号 [1]..[N] 自动编号；
- 交叉引用：正文中的上标式引用 [N] 转为 Word 交叉引用域（REF 字段 + 书签），
  与文末自动编号列表联动，引用顺序从 1 开始。

约定：
- 正文宋体小四(12pt)、西文 Times New Roman、1.5 倍行距、首行缩进 2 字符；
- 章标题黑体三号(16pt)居中、节黑体四号(14pt)、小节黑体小四(12pt)；
- 行间公式（$$...$$）以居中 LaTeX 源码呈现（供粘贴进 WPS 公式编辑器），行内公式
  （$...$）转换为可读 Unicode 文本；
- 表格以 Word 表格呈现，表题/图题居中加粗；代码块以等宽小字占位。
"""
import re
import glob
import os

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from lxml import etree as _ET
from latex2mathml.converter import convert as _latex2mml

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(BASE, "docs", "设计方案")
OUT = os.path.join(DOCS, "论文正文_第1至6章.docx")
FIG_DIR = os.path.join(DOCS, "figures")

# 图题编号 → 配图 PNG（由 scripts/draw_thesis_figures.py 生成，与同名 .drawio 同源）
FIGURE_MAP = {
    '图 2-1': 'fig2-1_system_route.png',
    '图 3-1': 'fig3-1_hqmix.png',
    '图 3-2': 'fig3-2_dualhead_drqn.png',
    '图 4-1': 'fig4-1_singlehead_drqn.png',
    '图 4-2': 'fig4-2_training_flow.png',
    '图 6-1': 'fig6_key_tech.png',
}
# 通栏图（多面板）放宽到接近版心宽；其余单栏图默认 12 cm
FIG_WIDTHS = {'图 6-1': 15.0}
FIG_DEFAULT_WIDTH = 12.0

# ---------------------------------------------------------------- 字体
BODY_EA = "宋体"
BODY_ASCII = "Times New Roman"
HEAD_EA = "黑体"
HEAD_ASCII = "Times New Roman"
MONO = "Consolas"
GRAY = RGBColor(0x59, 0x59, 0x59)
NOTE_GRAY = RGBColor(0x80, 0x80, 0x80)

# 全局合法引用编号集合（main 中填充）
VALID_REFS = set()

# ---------------------------------------------------------------- 基础工具


def set_run(run, size=12, ea=BODY_EA, ascii_font=BODY_ASCII, bold=False,
            italic=False, color=None):
    run.font.name = ascii_font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.append(rFonts)
    rFonts.set(qn('w:ascii'), ascii_font)
    rFonts.set(qn('w:hAnsi'), ascii_font)
    rFonts.set(qn('w:eastAsia'), ea)


def set_first_line_indent(p, chars=200):
    pPr = p._p.get_or_add_pPr()
    ind = pPr.find(qn('w:ind'))
    if ind is None:
        ind = OxmlElement('w:ind')
        pPr.append(ind)
    ind.set(qn('w:firstLineChars'), str(chars))
    ind.set(qn('w:firstLine'), str(int(chars * 2.4)))


def set_para_spacing(p, line=1.5, before=0, after=0):
    pf = p.paragraph_format
    pf.line_spacing = line
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)


# ---------------------------------------------------------------- LaTeX -> Unicode
_GREEK = {
    r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
    r'\epsilon': 'ε', r'\varepsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η',
    r'\theta': 'θ', r'\vartheta': 'ϑ', r'\iota': 'ι', r'\kappa': 'κ',
    r'\lambda': 'λ', r'\mu': 'μ', r'\nu': 'ν', r'\xi': 'ξ', r'\pi': 'π',
    r'\rho': 'ρ', r'\sigma': 'σ', r'\tau': 'τ', r'\upsilon': 'υ',
    r'\phi': 'φ', r'\varphi': 'φ', r'\chi': 'χ', r'\psi': 'ψ', r'\omega': 'ω',
    r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
    r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Phi': 'Φ', r'\Psi': 'Ψ',
    r'\Omega': 'Ω',
}

_CMDS = {
    r'\arg\max': 'arg max', r'\arg\min': 'arg min',
    r'\ge': '≥', r'\geq': '≥', r'\le': '≤', r'\leq': '≤', r'\ne': '≠', r'\neq': '≠',
    r'\in': '∈', r'\notin': '∉', r'\subset': '⊂', r'\subseteq': '⊆',
    r'\cup': '∪', r'\cap': '∩', r'\times': '×', r'\cdot': '·', r'\circ': '∘',
    r'\ast': '*', r'\star': '⋆',
    r'\to': '→', r'\rightarrow': '→', r'\leftarrow': '←',
    r'\longrightarrow': '→', r'\longleftarrow': '←', r'\mapsto': '↦',
    r'\Rightarrow': '⇒', r'\Leftarrow': '⇐', r'\Leftrightarrow': '⇔', r'\iff': '⇔',
    r'\partial': '∂', r'\nabla': '∇', r'\infty': '∞', r'\sum': '∑',
    r'\prod': '∏', r'\int': '∫',
    r'\forall': '∀', r'\exists': '∃', r'\neg': '¬',
    r'\wedge': '∧', r'\vee': '∨', r'\land': '∧', r'\lor': '∨',
    r'\approx': '≈', r'\equiv': '≡', r'\sim': '∼', r'\propto': '∝',
    r'\pm': '±', r'\mp': '∓',
    r'\langle': '⟨', r'\rangle': '⟩', r'\lVert': '‖', r'\rVert': '‖',
    r'\lfloor': '⌊', r'\rfloor': '⌋', r'\lceil': '⌈', r'\rceil': '⌉',
    r'\emptyset': '∅', r'\varnothing': '∅',
    r'\max': 'max', r'\min': 'min', r'\sup': 'sup', r'\inf': 'inf',
    r'\lim': 'lim', r'\log': 'log', r'\exp': 'exp', r'\ln': 'ln',
    r'\sin': 'sin', r'\cos': 'cos', r'\tan': 'tan', r'\arg': 'arg',
    r'\det': 'det', r'\dim': 'dim', r'\Pr': 'Pr',
    r'\top': '⊤', r'\bot': '⊥', r'\prime': '′', r'\ell': 'ℓ',
    r'\quad': '  ', r'\qquad': '   ', r'\;': ' ', r'\,': '', r'\:': ' ',
    r'\!': '', r'\ ': ' ',
    r'\{': '{', r'\}': '}', r'\%': '%', r'\$': '$', r'\&': '&',
    r'\_': '_', r'\#': '#', r'\backslash': '\\',
    r'\ldots': '…', r'\cdots': '⋯', r'\dots': '…', r'\vdots': '⋮', r'\ddots': '⋱',
}


def latex_to_unicode(s):
    if not s:
        return s
    s = re.sub(r'\\left\.?|\\right\.?', '', s)
    s = re.sub(r'\\big[lr]?|\\Big[lr]?|\\bigg[lr]?|\\Bigg[lr]?', '', s)
    for _ in range(6):
        s2 = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}',
                    lambda m: '(%s)/(%s)' % (latex_to_unicode(m.group(1)),
                                             latex_to_unicode(m.group(2))), s)
        if s2 == s:
            break
        s = s2
    s = re.sub(r'\\underbrace\{([^{}]*)\}_\{([^{}]*)\}',
               lambda m: '%s_(%s)' % (latex_to_unicode(m.group(1)),
                                       latex_to_unicode(m.group(2))), s)
    s = re.sub(r'\\underbrace\{([^{}]*)\}', lambda m: latex_to_unicode(m.group(1)), s)
    for a, b in [('R', 'ℝ'), ('N', 'ℕ'), ('Z', 'ℤ'), ('Q', 'ℚ'), ('C', 'ℂ')]:
        s = s.replace('\\mathbb{%s}' % a, b)
    s = re.sub(r'\\(?:text|mathrm|mathbf|mathcal|mathbb|boldsymbol|bm|mathit|'
               r'smathsf|mathtt|operatorname|mbox|textbf|textrm|textit)\{([^{}]*)\}',
               r'\1', s)
    s = re.sub(r'\\(?:bar|hat|tilde|dot|ddot|vec|widehat|widetilde|overline|'
               r'underline|breve|check|acute|grave|mathring)\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\(?:bar|hat|tilde|dot|ddot|vec)\s*([A-Za-z])', r'\1', s)
    for k in sorted(_CMDS, key=len, reverse=True):
        s = s.replace(k, _CMDS[k])
    for k in sorted(_GREEK, key=len, reverse=True):
        s = s.replace(k, _GREEK[k])
    s = s.replace('{', '').replace('}', '')
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ---------------------------------------------------------------- LaTeX -> OMML
_OM_TRANSFORM = None


def _get_omml_transform():
    global _OM_TRANSFORM
    if _OM_TRANSFORM is None:
        _OM_TRANSFORM = False
        for candidate in (
            r'C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL',
            r'C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.XSL',
        ):
            if os.path.exists(candidate):
                try:
                    _OM_TRANSFORM = _ET.XSLT(_ET.parse(candidate))
                    break
                except Exception:
                    continue
    return _OM_TRANSFORM


def latex_to_omml(latex):
    """LaTeX → OMML(<m:oMath> lxml 元素)；失败返回 None。"""
    latex = (latex or '').strip()
    if not latex:
        return None
    transform = _get_omml_transform()
    if not transform:
        return None
    try:
        mml = _latex2mml(latex)
        mml_tree = _ET.fromstring(mml.encode('utf-8'))
        root = transform(mml_tree).getroot()
        return root
    except Exception:
        return None


# ---------------------------------------------------------------- 交叉引用域
CITE_RE = re.compile(r'\[(\d{1,2})\]')


def add_citation_field(p, n, size=12, bold=False, color=None):
    """在段落 p 末尾插入一个指向参考文献 [n] 的 REF 交叉引用域。"""
    bm = '_Ref%d' % n
    r = p.add_run(); set_run(r, size=size, bold=bold, color=color)
    el = OxmlElement('w:fldChar'); el.set(qn('w:fldCharType'), 'begin'); r._element.append(el)
    r = p.add_run(); set_run(r, size=size, bold=bold, color=color)
    el = OxmlElement('w:instrText'); el.set(qn('xml:space'), 'preserve')
    el.text = ' REF %s \\r \\h ' % bm
    r._element.append(el)
    r = p.add_run(); set_run(r, size=size, bold=bold, color=color)
    el = OxmlElement('w:fldChar'); el.set(qn('w:fldCharType'), 'separate'); r._element.append(el)
    r = p.add_run('[%d]' % n); set_run(r, size=size, bold=bold, color=color)
    r = p.add_run(); set_run(r, size=size, bold=bold, color=color)
    el = OxmlElement('w:fldChar'); el.set(qn('w:fldCharType'), 'end'); r._element.append(el)


def add_text_runs(p, text, bold=False, size=12, color=None):
    """把普通文本按引用 [N] 与正文切分：引用转交叉引用域，其余为普通 run。"""
    pos = 0
    for m in CITE_RE.finditer(text):
        if m.start() > pos:
            run = p.add_run(text[pos:m.start()])
            set_run(run, size=size, bold=bold, color=color)
        n = int(m.group(1))
        if n in VALID_REFS:
            add_citation_field(p, n, size=size, bold=bold, color=color)
        else:
            run = p.add_run(m.group(0))
            set_run(run, size=size, bold=bold, color=color)
        pos = m.end()
    if pos < len(text):
        run = p.add_run(text[pos:])
        set_run(run, size=size, bold=bold, color=color)


# ---------------------------------------------------------------- 行内排版
MATH_RE = re.compile(r'(\$[^$]+\$)')


def add_math_runs(p, text, bold=False, size=12, color=None):
    for part in MATH_RE.split(text):
        if not part:
            continue
        if part.startswith('$') and part.endswith('$') and len(part) >= 2:
            omml = latex_to_omml(part[1:-1])
            if omml is not None:
                p._p.append(omml)
            else:
                run = p.add_run(latex_to_unicode(part[1:-1]))
                set_run(run, size=size, bold=bold, italic=False, color=color)
        else:
            add_text_runs(p, part, bold=bold, size=size, color=color)


BOLD_RE = re.compile(r'(\*\*[^*]+\*\*)')


def add_inline_runs(p, text, bold=False, size=12, color=None):
    for part in BOLD_RE.split(text):
        if not part:
            continue
        if part.startswith('**') and part.endswith('**') and len(part) >= 4:
            add_math_runs(p, part[2:-2], bold=True, size=size, color=color)
        else:
            add_math_runs(p, part, bold=bold, size=size, color=color)


# ---------------------------------------------------------------- 段落元素
def add_heading(doc, text, level):
    p = doc.add_paragraph(style='Heading %d' % level)
    sizes = {1: 16, 2: 14, 3: 12}
    run = p.add_run(latex_to_unicode(re.sub(r'[#*`]', '', text)).strip())
    set_run(run, size=sizes[level], ea=HEAD_EA, ascii_font=HEAD_ASCII, bold=True)
    pf = p.paragraph_format
    pf.line_spacing = 1.3
    pf.space_before = Pt(12 if level == 1 else 8)
    pf.space_after = Pt(8 if level == 1 else 4)
    if level == 1:
        pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return p


def add_body(doc, text):
    p = doc.add_paragraph()
    add_inline_runs(p, text)
    set_para_spacing(p, 1.5)
    set_first_line_indent(p, 200)
    return p


def add_note(doc, text):
    p = doc.add_paragraph()
    add_inline_runs(p, text, size=10.5, color=NOTE_GRAY)
    set_para_spacing(p, 1.3, before=2, after=6)
    p.paragraph_format.left_indent = Cm(0.5)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    add_inline_runs(p, text, size=10.5)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para_spacing(p, 1.3, before=6, after=3)
    return p


def add_figure(doc, img_path, caption_text, width_cm=FIG_DEFAULT_WIDTH):
    """插入居中配图，图下方跟居中图题。"""
    doc.add_picture(img_path, width=Cm(width_cm))
    p = doc.paragraphs[-1]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(0)
    add_caption(doc, caption_text)


def _emit_figure(doc, line):
    """处理「**图 X-Y 标题**…」行：插入对应配图并加图题。

    返回是否需要跳过紧随其后的 ASCII 结构描述代码块（旧占位符）。
    """
    m = re.match(r'(\*\*图\s*\d+-\d+[^*]*\*\*)', line)
    caption = m.group(1) if m else line
    num_m = re.match(r'\*\*图\s*(\d+-\d+)', caption)
    key = '图 %s' % num_m.group(1) if num_m else None
    img = FIGURE_MAP.get(key)
    inserted = False
    if img:
        path = os.path.join(FIG_DIR, img)
        if os.path.exists(path):
            add_figure(doc, path, caption, FIG_WIDTHS.get(key, FIG_DEFAULT_WIDTH))
            inserted = True
    if not inserted:
        add_caption(doc, caption)
    return inserted and ('WPS 绘图工具插入' in line)


def add_list_item(doc, text, numbered=None):
    p = doc.add_paragraph()
    marker = ('%s. ' % numbered) if numbered else '• '
    run = p.add_run(marker)
    set_run(run, size=12)
    add_inline_runs(p, text)
    set_para_spacing(p, 1.5)
    p.paragraph_format.left_indent = Cm(0.74)
    p.paragraph_format.first_line_indent = Cm(-0.74)
    return p


def add_equation(doc, latex, number=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.5
    pf.space_before = Pt(4)
    pf.space_after = Pt(6)
    omml = latex_to_omml(latex)
    if number:
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.tab_stops.add_tab_stop(Cm(7.75), WD_TAB_ALIGNMENT.CENTER)
        pf.tab_stops.add_tab_stop(Cm(15.5), WD_TAB_ALIGNMENT.RIGHT)
        r = p.add_run('\t')
        set_run(r, size=10.5)
        if omml is not None:
            p._p.append(omml)
        else:
            run = p.add_run(latex)
            set_run(run, size=10.5, ea=BODY_EA, ascii_font=MONO, color=GRAY)
        r2 = p.add_run('\t(%s)' % number)
        set_run(r2, size=10.5)
    else:
        pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if omml is not None:
            p._p.append(omml)
        else:
            run = p.add_run(latex)
            set_run(run, size=10.5, ea=BODY_EA, ascii_font=MONO, color=GRAY)
    return p


def add_code_block(doc, lines):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.0
    pf.space_before = Pt(4)
    pf.space_after = Pt(8)
    pf.left_indent = Cm(0.4)
    for i, line in enumerate(lines):
        run = p.add_run(line if line else ' ')
        set_run(run, size=7.5, ea=BODY_EA, ascii_font=MONO, color=GRAY)
        if i < len(lines) - 1:
            run.add_break()
    return p


def add_table(doc, header, rows):
    ncol = len(header)
    table = doc.add_table(rows=1 + len(rows), cols=ncol)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, cell in enumerate(header):
        c = table.cell(0, j)
        c.text = ''
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_inline_runs(p, latex_to_unicode(cell), bold=True, size=10.5)
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = table.cell(i + 1, j)
            c.text = ''
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline_runs(p, latex_to_unicode(cell), size=10.5)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(6)
    sp.paragraph_format.line_spacing = 1.0
    sp.add_run('').font.size = Pt(2)
    return table


# ---------------------------------------------------------------- 参考文献（自动编号 + 书签）
_REF_ABS_ID = '100'
_REF_NUM_ID = '100'


def ensure_ref_numbering(doc):
    part = doc.part.numbering_part
    numbering = part.element
    for an in numbering.findall(qn('w:abstractNum')):
        if an.get(qn('w:abstractNumId')) == _REF_ABS_ID:
            return int(_REF_NUM_ID)
    abstractNum = OxmlElement('w:abstractNum')
    abstractNum.set(qn('w:abstractNumId'), _REF_ABS_ID)
    mt = OxmlElement('w:multiLevelType'); mt.set(qn('w:val'), 'singleLevel')
    abstractNum.append(mt)
    lvl = OxmlElement('w:lvl'); lvl.set(qn('w:ilvl'), '0')
    start = OxmlElement('w:start'); start.set(qn('w:val'), '1'); lvl.append(start)
    numFmt = OxmlElement('w:numFmt'); numFmt.set(qn('w:val'), 'decimal'); lvl.append(numFmt)
    lvlText = OxmlElement('w:lvlText'); lvlText.set(qn('w:val'), '[%1]'); lvl.append(lvlText)
    lvlJc = OxmlElement('w:lvlJc'); lvlJc.set(qn('w:val'), 'left'); lvl.append(lvlJc)
    lpPr = OxmlElement('w:pPr')
    lind = OxmlElement('w:ind'); lind.set(qn('w:left'), '567'); lind.set(qn('w:hanging'), '567')
    lpPr.append(lind)
    lvl.append(lpPr)
    lrPr = OxmlElement('w:rPr')
    lfont = OxmlElement('w:rFonts')
    lfont.set(qn('w:ascii'), BODY_ASCII); lfont.set(qn('w:hAnsi'), BODY_ASCII)
    lfont.set(qn('w:eastAsia'), BODY_EA)
    lrPr.append(lfont)
    lsz = OxmlElement('w:sz'); lsz.set(qn('w:val'), '21'); lrPr.append(lsz)
    lvl.append(lrPr)
    abstractNum.append(lvl)
    # abstractNum 必须位于所有 w:num 之前
    first_num = numbering.find(qn('w:num'))
    if first_num is not None:
        first_num.addprevious(abstractNum)
    else:
        numbering.append(abstractNum)
    num = OxmlElement('w:num'); num.set(qn('w:numId'), _REF_NUM_ID)
    anId = OxmlElement('w:abstractNumId'); anId.set(qn('w:val'), _REF_ABS_ID); num.append(anId)
    numbering.append(num)
    return int(_REF_NUM_ID)


def add_reference_entry(doc, n, text, numId):
    p = doc.add_paragraph()
    p_el = p._p
    pPr = OxmlElement('w:pPr')
    numPr = OxmlElement('w:numPr')
    ilvl = OxmlElement('w:ilvl'); ilvl.set(qn('w:val'), '0'); numPr.append(ilvl)
    num = OxmlElement('w:numId'); num.set(qn('w:val'), str(numId)); numPr.append(num)
    pPr.append(numPr)
    ind = OxmlElement('w:ind'); ind.set(qn('w:left'), '567'); ind.set(qn('w:hanging'), '567')
    pPr.append(ind)
    spacing = OxmlElement('w:spacing')
    spacing.set(qn('w:line'), '312'); spacing.set(qn('w:lineRule'), 'auto')
    pPr.append(spacing)
    existing = p_el.find(qn('w:pPr'))
    if existing is not None:
        p_el.remove(existing)
    p_el.insert(0, pPr)
    bmS = OxmlElement('w:bookmarkStart')
    bmS.set(qn('w:id'), str(n)); bmS.set(qn('w:name'), '_Ref%d' % n)
    p_el.append(bmS)
    r = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')
    rFonts = OxmlElement('w:rFonts')
    rFonts.set(qn('w:ascii'), BODY_ASCII); rFonts.set(qn('w:hAnsi'), BODY_ASCII)
    rFonts.set(qn('w:eastAsia'), BODY_EA)
    rPr.append(rFonts)
    sz = OxmlElement('w:sz'); sz.set(qn('w:val'), '21'); rPr.append(sz)
    szCs = OxmlElement('w:szCs'); szCs.set(qn('w:val'), '21'); rPr.append(szCs)
    r.append(rPr)
    t = OxmlElement('w:t'); t.set(qn('xml:space'), 'preserve'); t.text = text
    r.append(t)
    p_el.append(r)
    bmE = OxmlElement('w:bookmarkEnd'); bmE.set(qn('w:id'), str(n))
    p_el.append(bmE)
    return p


# ---------------------------------------------------------------- Markdown 行解析
def split_table_row(line):
    line = line.strip()
    if line.startswith('|'):
        line = line[1:]
    if line.endswith('|'):
        line = line[:-1]
    return [c.strip() for c in line.split('|')]


def is_table_sep(line):
    return bool(re.match(r'^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$', line))


def split_references(lines):
    """把章文件拆成 (正文行, 参考文献映射)。参考文献映射 {编号: 条目文本}。"""
    ref_map = {}
    ref_start = None
    for i, line in enumerate(lines):
        if re.match(r'^##\s*参考文献', line.strip()):
            ref_start = i
            break
    if ref_start is None:
        return lines, ref_map
    body_lines = lines[:ref_start]
    for line in lines[ref_start + 1:]:
        m = re.match(r'^\[(\d+)\]\s+(.*)$', line.strip())
        if m:
            ref_map[int(m.group(1))] = m.group(2).strip()
    return body_lines, ref_map


def process_chapter(doc, lines, first=False):
    i = 0
    n = len(lines)
    skip_code_block = False
    while i < n:
        stripped = lines[i].strip()

        if stripped == '':
            i += 1
            continue

        if stripped.startswith('```'):
            i += 1
            code = []
            while i < n and not lines[i].strip().startswith('```'):
                code.append(lines[i])
                i += 1
            i += 1
            if skip_code_block:
                # 插图后紧随的 ASCII 结构描述块（旧占位符）：图已实体化，跳过
                skip_code_block = False
                continue
            add_code_block(doc, code)
            continue

        # 到达此处即遇到普通非空非代码行：清空「跳过下一代码块」标志
        skip_code_block = False

        if stripped.startswith('|'):
            if i + 1 < n and is_table_sep(lines[i + 1]):
                header = split_table_row(stripped)
                rows = []
                i += 2
                while i < n and lines[i].strip().startswith('|'):
                    rows.append(split_table_row(lines[i]))
                    i += 1
                add_table(doc, header, rows)
                continue
            else:
                add_body(doc, stripped)
                i += 1
                continue

        m = re.match(r'^(#{1,6})\s+(.*)$', stripped)
        if m:
            level = len(m.group(1))
            if level == 1 and not first:
                doc.add_page_break()
            add_heading(doc, m.group(2), level)
            i += 1
            continue

        if re.match(r'^-{3,}\s*$', stripped) or re.match(r'^\*{3,}\s*$', stripped):
            i += 1
            continue

        if stripped.startswith('>'):
            bq_lines = []
            while i < n and lines[i].strip().startswith('>'):
                bq_lines.append(lines[i].strip()[1:].strip())
                i += 1
            _emit_blockquote(doc, bq_lines)
            continue

        m = re.match(r'^[-*]\s+(.*)$', stripped)
        if m:
            add_list_item(doc, m.group(1))
            i += 1
            continue
        m = re.match(r'^(\d+)\.\s+(.*)$', stripped)
        if m:
            add_list_item(doc, m.group(2), numbered=m.group(1))
            i += 1
            continue

        if stripped.startswith('$$') and stripped.endswith('$$') and len(stripped) >= 4:
            add_equation(doc, stripped[2:-2].strip())
            i += 1
            continue

        if stripped.startswith('**图'):
            i += 1
            skip_code_block = _emit_figure(doc, stripped)
            continue

        if stripped.startswith('**表'):
            add_caption(doc, stripped)
            i += 1
            continue

        add_body(doc, stripped)
        i += 1


def _emit_blockquote(doc, bq_lines):
    has_math = any(l.startswith('$$') for l in bq_lines)
    if not has_math:
        text = ' '.join(bq_lines).strip()
        if text:
            add_note(doc, text)
        return

    current_label = None
    note_buf = []
    for l in bq_lines:
        if l == '':
            continue
        if l.startswith('$$') and l.endswith('$$') and len(l) >= 4:
            if note_buf:
                add_note(doc, ' '.join(note_buf))
                note_buf = []
            add_equation(doc, l[2:-2].strip(), number=current_label)
            current_label = None
        else:
            m = re.search(r'式\(([^)]+)\)', l)
            if m:
                current_label = m.group(1)
                rest = re.sub(r'\*\*式\([^)]+\)\*\*', '', l)
                rest = re.sub(r'[（(]WPS[^）)]*[）)]', '', rest).strip(' ：:')
                if rest:
                    note_buf.append(rest)
            else:
                note_buf.append(l)
    if note_buf:
        add_note(doc, ' '.join(note_buf))


# ---------------------------------------------------------------- 主流程
def main():
    global VALID_REFS
    files = sorted(glob.glob(os.path.join(DOCS, '第*章_*.md')))
    files = [f for f in files if re.search(r'第[1-6]章_', os.path.basename(f))]
    files.sort(key=lambda f: int(re.search(r'第(\d)章_', os.path.basename(f)).group(1)))

    global_refs = {}
    bodies = []
    for f in files:
        with open(f, encoding='utf-8') as fh:
            lines = fh.read().split('\n')
        body_lines, ref_map = split_references(lines)
        for num, txt in ref_map.items():
            if num not in global_refs:
                global_refs[num] = txt
        bodies.append((os.path.basename(f), body_lines))

    VALID_REFS = set(global_refs.keys())

    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    sec.left_margin = Cm(3.0)
    sec.right_margin = Cm(2.5)

    normal = doc.styles['Normal']
    normal.font.name = BODY_ASCII
    normal.font.size = Pt(12)
    normal.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_EA)

    for idx, (bname, body_lines) in enumerate(bodies):
        process_chapter(doc, body_lines, first=(idx == 0))

    doc.add_page_break()
    add_heading(doc, '参考文献', 1)
    numId = ensure_ref_numbering(doc)
    for n in sorted(global_refs.keys()):
        add_reference_entry(doc, n, global_refs[n], numId)

    doc.save(OUT)
    print('OK ->', OUT)
    print('章节数:', len(files))
    print('参考文献条数:', len(global_refs), '编号范围:', min(global_refs), '-', max(global_refs))


if __name__ == '__main__':
    main()
