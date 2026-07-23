#!/usr/bin/env python3
"""Convert the F Ring User's Guide .docx to a LaTeX project.

Reads the unzipped docx in DOCX_DIR and writes .tex files into PROJ.
Tracked changes are accepted: w:ins content kept, w:del content dropped.
"""
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter

DOCX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docx')
PROJ = os.path.expanduser('~/DS/f-ring-users-guide-latex')

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'

warns = []
def warn(msg):
    warns.append(msg)

# ----------------------------------------------------------------- loading
rels = {}
for rel in ET.parse(f'{DOCX}/word/_rels/document.xml.rels').getroot():
    rels[rel.get('Id')] = rel.get('Target')
fn_rels = {}
try:
    for rel in ET.parse(f'{DOCX}/word/_rels/footnotes.xml.rels').getroot():
        fn_rels[rel.get('Id')] = rel.get('Target')
except FileNotFoundError:
    pass

doc = ET.parse(f'{DOCX}/word/document.xml')
body = doc.getroot().find(W + 'body')
blocks = list(body)

foot_root = ET.parse(f'{DOCX}/word/footnotes.xml').getroot()
footnotes = {fn.get(W + 'id'): fn for fn in foot_root.findall(W + 'footnote')}

# numbering: numId -> {ilvl: (fmt, lvltext)}
nroot = ET.parse(f'{DOCX}/word/numbering.xml').getroot()
absmap = {}
for an in nroot.findall(W + 'abstractNum'):
    lvls = {}
    for lvl in an.findall(W + 'lvl'):
        fmt = lvl.find(W + 'numFmt')
        lvls[lvl.get(W + 'ilvl')] = fmt.get(W + 'val') if fmt is not None else 'bullet'
    absmap[an.get(W + 'abstractNumId')] = lvls
numfmt = {}
for n in nroot.findall(W + 'num'):
    aid = n.find(W + 'abstractNumId').get(W + 'val')
    numfmt[n.get(W + 'numId')] = absmap.get(aid, {})

# character styles -> font
sroot = ET.parse(f'{DOCX}/word/styles.xml').getroot()
charstyle_font = {}
parstyle_font = {}
for s in sroot.findall(W + 'style'):
    sid = s.get(W + 'styleId')
    rf = s.find(f'{W}rPr/{W}rFonts')
    font = rf.get(W + 'ascii') if rf is not None else None
    if s.get(W + 'type') == 'character' and font:
        charstyle_font[sid] = font
    if s.get(W + 'type') == 'paragraph' and font:
        parstyle_font[sid] = font

SYM = {'F0B0': '°', 'F02D': '−', 'F0B4': '×', 'F044': 'Δ', 'F076': 'ϖ',
       'F057': 'Ω', 'F0B7': '•', 'F06C': 'λ'}

CODE_FONTS = {'Cascadia Code': 'code', 'Courier New': 'literal',
              'Lucida Sans': 'labelxml'}
# CIDFont+F1/F10 are artifacts of text pasted from a PDF; they are body text.

# ----------------------------------------------------------------- helpers
def pstyle(p):
    ps = p.find(f'{W}pPr/{W}pStyle')
    return ps.get(W + 'val') if ps is not None else 'Normal'

def par_jc(p):
    jc = p.find(f'{W}pPr/{W}jc')
    return jc.get(W + 'val') if jc is not None else None

def parmark_deleted(p):
    rpr = p.find(f'{W}pPr/{W}rPr')
    return rpr is not None and rpr.find(W + 'del') is not None

def accepted_text(el):
    out = []
    def walk(e):
        for c in e:
            if c.tag in (W + 'del', W + 'moveFrom'):
                continue
            if c.tag == W + 't':
                out.append(c.text or '')
            elif c.tag == W + 'sym':
                out.append(SYM.get(c.get(W + 'char'), '?'))
            else:
                walk(c)
    walk(el)
    return ''.join(out)

def has_kept_drawing(el):
    def walk(e):
        for c in e:
            if c.tag in (W + 'del', W + 'moveFrom'):
                continue
            if c.tag == W + 'drawing':
                return True
            if walk(c):
                return True
        return False
    return walk(el)

def is_block_empty(el):
    return not accepted_text(el).strip() and not has_kept_drawing(el)

# LaTeX escaping for text mode
ESC = {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', '$': r'\$',
       '&': r'\&', '#': r'\#', '^': r'\textasciicircum{}',
       '_': r'\_',
       '%': r'\%', '~': r'\textasciitilde{}'}
def esc(s):
    return ''.join(ESC.get(ch, ch) for ch in s)

def slugify(text):
    s = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
    return s[:48].rstrip('-') or 'x'

MERGEABLE = ('code', 'literal', 'labelxml', 'textit', 'textbf')
def polish(s):
    """Merge adjacent identical macro fragments and strip trailing breaks."""
    for mac in MERGEABLE:
        prev = None
        while prev != s:
            prev = s
            s = re.sub(r'\\%s\{([^{}]*)\}\\%s\{' % (mac, mac),
                       lambda m: '\\%s{%s' % (mac, m.group(1)), s)
    s = s.strip()
    s = re.sub(r'(?:\\newline\s*|\\quad\s*)+$', '', s).rstrip()
    # empty macro leftovers
    s = re.sub(r'\\(?:code|literal|labelxml|textit|textbf)\{\}', '', s)
    return s.strip()

# ----------------------------------------------------------------- pre-pass
# Bookmarks that are targets of REF fields
whole_xml = ET.tostring(doc.getroot(), encoding='unicode')
foot_xml = ET.tostring(foot_root, encoding='unicode')
reftargets = set(re.findall(r'REF (_Ref\d+)', whole_xml + foot_xml))

# Identify kept blocks and their roles; map bookmarks -> labels
FIGSLUG = {1: 'fmovie-mosaic', 2: 'paired-mosaics', 3: 'm3-mosaics',
           4: 'm4-reproj-imgs', 5: 'non-corot-mosaic', 6: 'bkg-sub-mosaic'}

# figure tables detected by containing a SEQ Figure field (kept)
def is_figure_table(tbl):
    s = ET.tostring(tbl, encoding='unicode')
    return 'SEQ Figure' in s and not is_block_empty(tbl)

bookmark_label = {}   # _RefNNN -> latex label
heading_labels = {}   # block idx -> list of labels to emit
used_slugs = set()
fig_counter = 0
fig_of_block = {}

for i, el in enumerate(blocks):
    tag = el.tag.split('}')[1]
    if tag == 'tbl' and is_figure_table(el):
        fig_counter += 1
        fig_of_block[i] = fig_counter

def heading_label_for(idx, el):
    txt = accepted_text(el).strip()
    slug = slugify(txt)
    if slug in used_slugs:
        n = 2
        while f'{slug}-{n}' in used_slugs:
            n += 1
        slug = f'{slug}-{n}'
    used_slugs.add(slug)
    return f'sec:{slug}'

# walk blocks; for each bookmarkStart that is a reftarget, find its block
for i, el in enumerate(blocks):
    tag = el.tag.split('}')[1]
    names = [bm.get(W + 'name') for bm in el.iter(W + 'bookmarkStart')]
    names = [n for n in names if n in reftargets]
    if not names:
        continue
    if i in fig_of_block:
        for n in names:
            bookmark_label[n] = f'fig:{FIGSLUG[fig_of_block[i]]}'
    elif tag == 'p' and pstyle(el).startswith('Heading') and accepted_text(el).strip():
        lab = heading_labels.get(i)
        if lab is None:
            lab = heading_label_for(i, el)
            heading_labels[i] = lab
        for n in names:
            bookmark_label[n] = lab
    else:
        for n in names:
            bookmark_label[n] = None   # deleted/unrenderable target
            warn(f'REF target {n} at block {i} has no kept target (deleted content)')

# ----------------------------------------------------------------- OMML math
def omml_to_latex(el):
    t = el.tag
    if t in (M + 'oMath', M + 'oMathPara', M + 'e', M + 'num', M + 'den',
             M + 'sub', M + 'sup', M + 'fName'):
        return ''.join(omml_to_latex(c) for c in el)
    if t == M + 'r':
        txt = ''.join(x.text or '' for x in el.iter(M + 't'))
        # inside w:del?
        if el.find(f'{W}del') is not None:
            return ''
        rep = {'ϖ': r'\varpi ', 'Ω': r'\Omega ', 'θ': r'\theta ',
               'λ': r'\lambda ', '−': '-', '–': '-', ' ': r'\ '}
        out = []
        for ch in txt:
            out.append(rep.get(ch, ch))
        return ''.join(out)
    if t == M + 'acc':
        chr_el = el.find(f'{M}accPr/{M}chr')
        ch = chr_el.get(M + 'val') if chr_el is not None else '^'
        base = ''.join(omml_to_latex(c) for c in el.findall(M + 'e'))
        if '̇' in (ch or ''):
            return r'\dot{%s}' % base
        return r'\hat{%s}' % base
    if t == M + 'sSub':
        base = omml_to_latex(el.find(M + 'e'))
        sub = omml_to_latex(el.find(M + 'sub'))
        return '%s_{\\mathrm{%s}}' % (base, sub) if len(sub) > 1 else '%s_{%s}' % (base, sub)
    if t == M + 'sSup':
        base = omml_to_latex(el.find(M + 'e'))
        sup = omml_to_latex(el.find(M + 'sup'))
        return '%s^{%s}' % (base, sup)
    if t == M + 'f':
        return r'\frac{%s}{%s}' % (omml_to_latex(el.find(M + 'num')),
                                   omml_to_latex(el.find(M + 'den')))
    if t == M + 'func':
        name = omml_to_latex(el.find(M + 'fName')).strip()
        arg = ''.join(omml_to_latex(c) for c in el.findall(M + 'e'))
        macro = {'cos': r'\cos', 'sin': r'\sin', 'tan': r'\tan'}.get(name, r'\mathrm{%s}' % name)
        return '%s %s' % (macro, arg)
    if t == M + 'd':   # delimiters
        inner = ''.join(omml_to_latex(c) for c in el.findall(M + 'e'))
        return r'\left(%s\right)' % inner
    if t in (W + 'ins',):
        return ''.join(omml_to_latex(c) for c in el)
    if t in (W + 'del', M + 'ctrlPr', M + 'accPr', M + 'sSubPr', M + 'sSupPr',
             M + 'fPr', M + 'funcPr', M + 'dPr', M + 'rPr'):
        return ''
    return ''.join(omml_to_latex(c) for c in el)

# ----------------------------------------------------------------- run rendering
zotero_cites = []

class Renderer:
    """Renders the accepted (tracked-changes applied) inline content of a
    paragraph, handling field codes via a stack."""
    def __init__(self):
        self.out = []
        self.fields = []       # stack of dicts: {instr:[], phase, suppress}
        self.suppress = 0

    def emit(self, s):
        if self.suppress == 0:
            self.out.append(s)
        elif self.fields and self.fields[-1]['phase'] == 'cached':
            self.fields[-1]['cached'].append(s)

    def field_begin(self):
        self.fields.append({'instr': [], 'phase': 'instr', 'cached': []})

    def field_separate(self):
        if not self.fields:
            return
        f = self.fields[-1]
        f['phase'] = 'cached'
        instr = ''.join(f['instr']).strip()
        # decide whether cached result should be suppressed & replaced
        if re.match(r'(REF|PAGEREF|SEQ|HYPERLINK)\b', instr) or 'ZOTERO_ITEM' in instr or instr.startswith('TOC'):
            f['sup'] = True
            self.suppress += 1
        else:
            f['sup'] = False

    def field_end(self):
        if not self.fields:
            return
        f = self.fields.pop()
        if f['phase'] == 'instr':
            # field with no separate (rare): treat instr as nothing
            instr = ''.join(f['instr']).strip()
            rep = self.replacement(instr, '')
            if rep:
                self.emit(rep)
            return
        if f.get('sup'):
            self.suppress -= 1
            instr = ''.join(f['instr']).strip()
            cached = ''.join(f['cached'])
            rep = self.replacement(instr, cached)
            if rep:
                self.emit(rep)

    def replacement(self, instr, cached):
        m = re.match(r'REF\s+(_Ref\d+)', instr)
        if m:
            name = m.group(1)
            lab = bookmark_label.get(name)
            if lab is None:
                if name in bookmark_label:
                    warn(f'dropping REF to deleted target {name} (cached {cached!r})')
                else:
                    warn(f'REF to unknown bookmark {name} (cached {cached!r})')
                return cached  # best effort: keep whatever Word displayed
            if lab.startswith('fig:'):
                return r'Figure~\ref{%s}' % lab
            return r'\ref{%s}' % lab
        if instr.startswith('SEQ Figure'):
            return '\x00SEQFIG\x00'   # marker used by caption handling
        if instr.startswith('HYPERLINK'):
            m = re.match(r'HYPERLINK\s+"([^"]+)"', instr)
            url = m.group(1) if m else ''
            plain = cached
            if url and plain and plain.strip('\x00') :
                # if display text is the url itself, use \url
                return r'\href{%s}{%s}' % (url.replace('%', r'\%').replace('#', r'\#'), plain)
            if url:
                return r'\url{%s}' % url.replace('%', r'\%').replace('#', r'\#')
            return plain
        if 'ZOTERO_ITEM' in instr:
            jm = re.search(r'CSL_CITATION\s+(\{.*)', instr, re.S)
            plain = None
            if jm:
                try:
                    data = json.loads(jm.group(1))
                    plain = data['properties'].get('plainCitation')
                    zotero_cites.append(data)
                except Exception as e:
                    warn(f'zotero JSON parse failed: {e}')
            if plain is None:
                return cached
            return r'\zcite{%s}' % esc(plain)
        if instr.startswith('PAGEREF') or instr.startswith('TOC'):
            return ''
        warn(f'unknown field {instr[:60]!r} kept cached text')
        return cached

    # ---- run walking
    def run_font(self, r, para_font):
        rpr = r.find(W + 'rPr')
        if rpr is not None:
            rf = rpr.find(W + 'rFonts')
            if rf is not None and rf.get(W + 'ascii'):
                return rf.get(W + 'ascii')
            rs = rpr.find(W + 'rStyle')
            if rs is not None and rs.get(W + 'val') in charstyle_font:
                return charstyle_font[rs.get(W + 'val')]
        return para_font

    def render_run(self, r, para_font):
        rpr = r.find(W + 'rPr')
        bold = italic = under = sub = sup = False
        if rpr is not None:
            def on(tag):
                e = rpr.find(W + tag)
                return e is not None and e.get(W + 'val') not in ('0', 'false', 'none')
            bold = on('b'); italic = on('i'); under = on('u')
            va = rpr.find(W + 'vertAlign')
            if va is not None:
                sub = va.get(W + 'val') == 'subscript'
                sup = va.get(W + 'val') == 'superscript'
        font = self.run_font(r, para_font)

        def wrap(txt):
            if not txt:
                return ''
            code_kind = CODE_FONTS.get(font)
            if code_kind == 'code':
                txt = r'\code{%s}' % txt
            elif code_kind == 'literal':
                txt = r'\literal{%s}' % txt
            elif code_kind == 'labelxml':
                txt = r'\labelxml{%s}' % txt
            elif font == 'Arial':
                txt = r'\textsf{%s}' % txt
            if sub:
                txt = r'\textsubscript{%s}' % txt
            if sup:
                txt = r'\textsuperscript{%s}' % txt
            if under:
                txt = r'\underline{%s}' % txt
            if italic:
                txt = r'\textit{%s}' % txt
            if bold:
                txt = r'\textbf{%s}' % txt
            return txt

        pieces = []       # rendered output pieces
        buf = []          # pending text to be font-wrapped
        def flush():
            if buf:
                pieces.append(wrap(''.join(buf)))
                del buf[:]
        for c in r:
            if c.tag == W + 't':
                buf.append(esc(c.text or ''))
            elif c.tag == W + 'delText':
                pass
            elif c.tag == W + 'sym':
                ch = SYM.get(c.get(W + 'char'))
                if ch is None:
                    warn(f'unknown symbol {c.get(W+"char")}')
                    ch = '?'
                buf.append(ch)
            elif c.tag == W + 'tab':
                flush()
                pieces.append(r'\quad ')
            elif c.tag == W + 'br':
                flush()
                if c.get(W + 'type') != 'page':
                    pieces.append('\\newline ')
            elif c.tag == W + 'noBreakHyphen':
                buf.append('-')
            elif c.tag == W + 'footnoteReference':
                flush()
                pieces.append(render_footnote(c.get(W + 'id')))
            elif c.tag == W + 'drawing':
                flush()
                img = render_drawing(c)
                if img:
                    pieces.append(img)
            elif c.tag == W + 'fldChar':
                flush()
                t = c.get(W + 'fldCharType')
                if t == 'begin':
                    self.field_begin()
                elif t == 'separate':
                    self.field_separate()
                elif t == 'end':
                    self.field_end()
            elif c.tag == W + 'instrText':
                if self.fields and self.fields[-1]['phase'] == 'instr':
                    self.fields[-1]['instr'].append(c.text or '')
        flush()
        return ''.join(pieces)

    def walk_para_content(self, e, para_font):
        for c in e:
            if c.tag in (W + 'del', W + 'moveFrom'):
                # still need field chars inside dels? Word keeps field pairs
                # outside dels in this doc; skip entirely.
                continue
            elif c.tag == W + 'r':
                self.emit(self.render_run(c, para_font))
            elif c.tag in (W + 'ins', W + 'moveTo', W + 'smartTag', W + 'sdt', W + 'sdtContent'):
                self.walk_para_content(c, para_font)
            elif c.tag == W + 'hyperlink':
                rid = c.get(R + 'id')
                url = rels.get(rid) if rid else None
                innerr = Renderer()
                innerr.walk_para_content(c, para_font)
                inner = ''.join(innerr.out)
                if url:
                    u = url.replace('%', r'\%').replace('#', r'\#')
                    # if the visible text is just the url, prefer \url
                    if inner.replace(r'\_', '_').strip() == url:
                        self.emit(r'\url{%s}' % u)
                    else:
                        self.emit(r'\href{%s}{%s}' % (u, inner))
                else:
                    self.emit(inner)
            elif c.tag == M + 'oMath':
                self.emit('$%s$' % omml_to_latex(c))
            elif c.tag == M + 'oMathPara':
                self.emit(r'\begin{equation*}%s\end{equation*}' %
                          ''.join(omml_to_latex(x) for x in c.findall(M + 'oMath')))
            elif c.tag == W + 'pPr':
                continue
            else:
                pass

def render_para_inline(p):
    """Render paragraph inline content to latex string."""
    font = parstyle_font.get(pstyle(p))
    r = Renderer()
    r.walk_para_content(p, font)
    while r.fields:
        r.field_end()
    return ''.join(r.out)

def render_footnote(fid):
    fn = footnotes.get(fid)
    if fn is None:
        warn(f'missing footnote {fid}')
        return ''
    parts = []
    for p in fn.findall(W + 'p'):
        t = render_para_inline(p)
        t = t.replace('\x00SEQFIG\x00', '')
        if t.strip():
            parts.append(t.strip())
    return r'\footnote{%s}' % ' '.join(parts)

IMG_OUT_NAME = {}
def render_drawing(d):
    blip = d.find(f'.//{A}blip')
    if blip is None:
        return ''
    target = rels.get(blip.get(R + 'embed'))
    if not target:
        return ''
    base = os.path.basename(target)
    name = IMG_OUT_NAME.get(base, base)
    ext = d.find(f'.//{A}ext')
    width_in = int(ext.get('cx')) / 914400 if ext is not None and ext.get('cx') else None
    if width_in and width_in < 6.4:
        return r'\includegraphics[width=%.2fin]{figures/%s}' % (width_in, name)
    return r'\noindent\includegraphics[width=\linewidth]{figures/%s}' % name

# ----------------------------------------------------------------- paragraph kinds
def effective_para_font(p):
    return parstyle_font.get(pstyle(p))

def num_props(p):
    npr = p.find(f'{W}pPr/{W}numPr')
    if npr is None:
        return None
    ilvl = npr.find(W + 'ilvl')
    numid = npr.find(W + 'numId')
    if numid is None:
        return None
    lvl = ilvl.get(W + 'val') if ilvl is not None else '0'
    nid = numid.get(W + 'val')
    if nid == '0':
        return None
    fmt = numfmt.get(nid, {}).get(lvl, 'bullet')
    return (int(lvl), 'enumerate' if fmt == 'decimal' else 'itemize')

def codeish_kind(p):
    """Return 'code'/'literal'/'labelxml' if the whole paragraph is in a
    code-like font, else None."""
    st = pstyle(p)
    if num_props(p):
        return None
    kinds = set()
    has_text = False
    r = Renderer()          # only for font resolution
    font_default = parstyle_font.get(st)
    def walk(e):
        nonlocal has_text
        for c in e:
            if c.tag in (W + 'del', W + 'moveFrom'):
                continue
            if c.tag == W + 'r':
                txt = ''.join(t.text or '' for t in c.findall(W + 't'))
                syms = c.findall(W + 'sym')
                if (txt and txt.strip()) or syms:
                    has_text = True
                    f = r.run_font(c, font_default)
                    kinds.add(CODE_FONTS.get(f))
            elif c.tag in (W + 'ins', W + 'moveTo'):
                walk(c)
    walk(p)
    if not has_text:
        # empty paragraph: code-style empties belong to code blocks
        if st in ('Code', 'Literal'):
            return 'empty-code'
        return None
    if len(kinds) == 1:
        k = kinds.pop()
        if k:
            return k
    return None

def code_size_class(p):
    """Word font sizes (half-points) -> LaTeX size command for code blocks."""
    szs = []
    def walk(e):
        for c in e:
            if c.tag in (W + 'del', W + 'moveFrom'):
                continue
            if c.tag == W + 'r':
                rpr = c.find(W + 'rPr')
                if rpr is not None:
                    sz = rpr.find(W + 'sz')
                    if sz is not None:
                        szs.append(int(sz.get(W + 'val')))
            elif c.tag == W + 'ins':
                walk(c)
    walk(p)
    if not szs:
        # fall back to paragraph-mark size
        rpr = p.find(f'{W}pPr/{W}rPr')
        if rpr is not None:
            sz = rpr.find(W + 'sz')
            if sz is not None:
                szs.append(int(sz.get(W + 'val')))
    m = max(szs) if szs else 22
    if m <= 16:
        return r'\scriptsize'
    if m <= 18:
        return r'\footnotesize'
    return None

def raw_code_text(p):
    """Plain text of a code paragraph (tabs -> 4 spaces)."""
    out = []
    def walk(e):
        for c in e:
            if c.tag in (W + 'del', W + 'moveFrom'):
                continue
            if c.tag == W + 't':
                out.append(c.text or '')
            elif c.tag == W + 'tab':
                out.append('    ')
            elif c.tag == W + 'sym':
                out.append(SYM.get(c.get(W + 'char'), '?'))
            else:
                walk(c)
    walk(p)
    return ''.join(out)

# ----------------------------------------------------------------- tables
def cell_props(tc):
    tcpr = tc.find(W + 'tcPr')
    span = 1
    vmerge = None
    if tcpr is not None:
        gs = tcpr.find(W + 'gridSpan')
        if gs is not None:
            span = int(gs.get(W + 'val'))
        vm = tcpr.find(W + 'vMerge')
        if vm is not None:
            vmerge = vm.get(W + 'val') or 'continue'
    return span, vmerge

BREAK_AFTER = re.compile(r'([./:])')
def _breakable(m):
    inner = BREAK_AFTER.sub(lambda x: x.group(1) + '\x01', m.group(2))
    return '\\%s{%s}' % (m.group(1), inner)

def cell_breakable(s):
    """Insert break opportunities in long code-like tokens (Word wraps them)."""
    s = re.sub(r'\\(code|literal|labelxml)\{([^{}]*)\}', _breakable, s)
    s = s.replace(r'\_', r'\_\allowbreak{}')
    s = s.replace('\x01', r'\allowbreak{}')
    return s

def render_cell(tc):
    paras = []
    for p in tc.findall(W + 'p'):
        t = render_para_inline(p).strip()
        if t:
            paras.append(t)
    return r' \newline '.join(cell_breakable(polish(x)) for x in paras)

def render_table(tbl, idx):
    grid = [int(g.get(W + 'w')) for g in tbl.findall(f'{W}tblGrid/{W}gridCol')]
    total = sum(grid) or 1
    ncols = len(grid)
    fracs = [w / total * 0.92 for w in grid]
    colspec = '|' + '|'.join('>{\\raggedright\\arraybackslash}p{%.3f\\linewidth}' % f
                             for f in fracs) + '|'
    rows = [tr for tr in tbl.findall(W + 'tr') if not is_row_deleted(tr)]
    lines = []
    lines.append(r'\begingroup\small')
    lines.append(r'\begin{longtable}{%s}' % colspec)
    lines.append(r'\hline')
    first = True
    header_row = None
    for ri, tr in enumerate(rows):
        trpr = tr.find(W + 'trPr')
        is_header = trpr is not None and trpr.find(W + 'tblHeader') is not None
        cells = tr.findall(W + 'tc')
        rend = []
        merge_cont_cols = []   # column positions (1-based) continuing a vmerge
        colpos = 1
        for tc in cells:
            span, vmerge = cell_props(tc)
            content = render_cell(tc)
            if vmerge == 'continue':
                merge_cont_cols.extend(range(colpos, colpos + span))
                content = ''
            if span > 1:
                content = r'\multicolumn{%d}{|>{\raggedright\arraybackslash}p{%.3f\linewidth}|}{%s}' % (
                    span, sum(fracs[colpos - 1:colpos - 1 + span]), content)
            rend.append(content)
            colpos += span
        lines.append(' & '.join(rend) + r' \\')
        # border under this row: suppress segments where next row continues a merge
        nxt = rows[ri + 1] if ri + 1 < len(rows) else None
        if nxt is not None:
            ncont = []
            cp = 1
            for tc in nxt.findall(W + 'tc'):
                span, vmerge = cell_props(tc)
                if vmerge == 'continue':
                    ncont.extend(range(cp, cp + span))
                cp += span
            if ncont:
                segs = []
                cstart = None
                for c in range(1, ncols + 1):
                    if c not in ncont:
                        if cstart is None:
                            cstart = c
                    else:
                        if cstart is not None:
                            segs.append((cstart, c - 1))
                            cstart = None
                if cstart is not None:
                    segs.append((cstart, ncols))
                lines.append(''.join(r'\cline{%d-%d}' % s for s in segs))
            else:
                lines.append(r'\hline')
        else:
            lines.append(r'\hline')
        if is_header and first:
            lines.append(r'\endhead')
        first = False
    lines.append(r'\end{longtable}')
    lines.append(r'\endgroup')
    return '\n'.join(lines)

def is_row_deleted(tr):
    trpr = tr.find(W + 'trPr')
    if trpr is not None and trpr.find(W + 'del') is not None:
        return True
    # row whose entire content is deleted
    return False

# ----------------------------------------------------------------- figures
def render_figure_table(tbl, fignum):
    """Word wraps figures in 1-col tables: image rows + caption row."""
    label = f'fig:{FIGSLUG[fignum]}'
    img_rows = []       # list of lists of image latex
    caption = None
    for tr in tbl.findall(W + 'tr'):
        row_imgs = []
        row_text = []
        for tc in tr.findall(W + 'tc'):
            cell_imgs = []
            for d in tc.iter(W + 'drawing'):
                # skip deleted drawings
                anc = d
                deleted = False
                # cheap check: search parents via tostring context not avail;
                # rely on accepted structure: drawings inside w:del skipped later
                cell_imgs.append(d)
            # determine deleted drawings properly:
            kept = []
            def walk(e, indel):
                for c in e:
                    d2 = indel or c.tag in (W + 'del', W + 'moveFrom')
                    if c.tag == W + 'drawing' and not d2:
                        kept.append(c)
                    else:
                        walk(c, d2)
            kept = []
            walk(tc, False)
            cell_imgs = kept
            if cell_imgs:
                row_imgs.append(cell_imgs)
            t = accepted_text(tc).strip()
            if t:
                row_text.append(tc)
        if row_imgs and not row_text:
            img_rows.append(row_imgs)
        elif row_text:
            caption = row_text[0]   # cell containing caption paragraph
        # reset for next row
        row_imgs = []
    # rebuild: iterate again properly (row_imgs got reset incorrectly above)
    img_matrix = []
    caption_cell = None
    for tr in tbl.findall(W + 'tr'):
        cells_imgs = []
        text_cell = None
        for tc in tr.findall(W + 'tc'):
            kept = []
            def walk(e, indel):
                for c in e:
                    d2 = indel or c.tag in (W + 'del', W + 'moveFrom')
                    if c.tag == W + 'drawing' and not d2:
                        kept.append(c)
                    else:
                        walk(c, d2)
            walk(tc, False)
            if kept:
                cells_imgs.append(kept)
            if accepted_text(tc).strip():
                text_cell = tc
        if cells_imgs:
            img_matrix.append(cells_imgs)
        if text_cell is not None:
            caption_cell = text_cell
    lines = [r'\begin{figure}[tbp]', r'\centering']
    for cells in img_matrix:
        if len(cells) == 1:
            for d in cells[0]:
                lines.append(render_drawing(d) + r'\\[2pt]')
        else:
            # side-by-side images
            frac = 0.98 / len(cells)
            parts = []
            for cell in cells:
                for d in cell:
                    blip = d.find(f'.//{A}blip')
                    target = rels.get(blip.get(R + 'embed'))
                    base = os.path.basename(target)
                    name = IMG_OUT_NAME.get(base, base)
                    parts.append(r'\includegraphics[width=%.3f\linewidth]{figures/%s}' % (frac, name))
            lines.append('\n\\hfill\n'.join(parts) + r'\\[2pt]')
    # last image row: strip trailing \\[2pt]
    if lines[-1].endswith(r'\\[2pt]'):
        lines[-1] = lines[-1][:-len(r'\\[2pt]')]
    cap = ''
    if caption_cell is not None:
        cap_parts = []
        for p in caption_cell.findall(W + 'p'):
            t = render_para_inline(p).strip()
            if t:
                cap_parts.append(t)
        cap = ' '.join(cap_parts)
        # strip "Figure <SEQ>: " prefix
        cap = re.sub(r'^\s*Figure\s*\x00SEQFIG\x00\s*:?\s*', '', cap)
        cap = cap.replace('\x00SEQFIG\x00', str(fignum))
        cap = cell_breakable(polish(cap))
    else:
        warn(f'figure {fignum}: no caption found')
    lines.append(r'\caption{%s}' % cap)
    lines.append(r'\label{%s}' % label)
    lines.append(r'\end{figure}')
    return '\n'.join(lines)

# ----------------------------------------------------------------- image renaming
IMG_OUT_NAME.update({
    'image1.png': 'fig1-fmovie-mosaic.png',
    'image2.png': 'fig2a-paired-mosaic-1.png',
    'image3.png': 'fig2b-paired-mosaic-2.png',
    'image4.png': 'fig3a-m3-mosaic-1.png',
    'image5.png': 'fig3b-m3-mosaic-2.png',
    'image6.png': 'fig3c-m3-mosaic-3.png',
    'image7.png': 'fig4a-reproj-img-1.png',
    'image8.png': 'fig4b-reproj-img-2.png',
    'image9.png': 'fig4c-reproj-img-3.png',
    'image10.png': 'fig5-non-corot-mosaic.png',
    'image11.png': 'fig6a-mosaic-original.png',
    'image12.png': 'fig6b-mosaic-bkg-sub.png',
    'image13.png': 'screenshot-display-reproj-img.png',
    'image14.png': 'screenshot-plot-ews.png',
    'image15.png': 'screenshot-find-prometheus.png',
})

# ----------------------------------------------------------------- main walk
class Doc:
    def __init__(self):
        self.chunks = []          # list of (section_index, latex string)
        self.cur = []
        self.sections = []        # (num, title, lines)
        self.front = []
        self.list_stack = []      # active list envs: list of 'itemize'/'enumerate'
        self.code_buf = []        # (kind, [lines])
        self.code_kind = None
        self.code_size = None

    def out(self, s):
        self.close_code()
        self.target().append(s)

    def target(self):
        return self.sections[-1][2] if self.sections else self.front

    # ---- code block buffering
    def code_line(self, kind, text, size=None):
        self.close_lists()
        if self.code_kind is not None and kind != 'empty-code' and \
                (kind != self.code_kind or size != self.code_size):
            self.close_code()
        if self.code_kind is None:
            if kind == 'empty-code':
                return
            self.code_kind = kind
            self.code_size = size
        self.code_buf.append(text)

    def close_code(self):
        if self.code_kind is None:
            return
        env = {'code': 'CodeBlock', 'literal': 'LiteralBlock',
               'labelxml': 'LabelBlock'}[self.code_kind]
        buf = list(self.code_buf)
        while buf and not buf[-1].strip():
            buf.pop()
        while buf and not buf[0].strip():
            buf.pop(0)
        t = self.target()
        opt = '[fontsize=%s]' % self.code_size if self.code_size else ''
        t.append(r'\begin{%s}%s' % (env, opt))
        t.extend(buf)
        t.append(r'\end{%s}' % env)
        t.append('')
        self.code_buf = []
        self.code_kind = None
        self.code_size = None

    # ---- lists
    def open_list(self, env):
        self.target().append(r'\begin{%s}[nosep]' % env if False else r'\begin{%s}' % env)
        self.list_stack.append(env)

    def close_one_list(self):
        env = self.list_stack.pop()
        self.target().append(r'\end{%s}' % env)

    def close_lists(self):
        while self.list_stack:
            self.close_one_list()

    def item(self, lvl, env, latex):
        self.close_code()
        # adjust stack depth
        while len(self.list_stack) > lvl + 1:
            self.close_one_list()
        while len(self.list_stack) < lvl + 1:
            self.open_list(env)
        if self.list_stack[-1] != env and len(self.list_stack) == lvl + 1:
            self.close_one_list()
            self.open_list(env)
        self.target().append(r'\item %s' % cell_breakable(polish(latex)))

    def para(self, latex, jc=None):
        self.close_code()
        self.close_lists()
        if '\\includegraphics' not in latex and '\\begin{' not in latex:
            latex = cell_breakable(polish(latex))
        if not latex:
            return
        t = self.target()
        if jc == 'center':
            t.append(r'\begin{center}')
            t.append(latex)
            t.append(r'\end{center}')
        else:
            t.append(latex)
        t.append('')

    def heading(self, level, latex, labels):
        self.close_code()
        self.close_lists()
        cmd = {1: 'section', 2: 'subsection', 3: 'subsubsection', 4: 'paragraph'}[level]
        t = self.target()
        t.append('')
        t.append(r'\%s{%s}' % (cmd, latex))
        for lab in labels:
            t.append(r'\label{%s}' % lab)
        t.append('')


# blocks with an explicit (kept) page break: honor it when the next content
# block is a heading, so major sections start on a fresh page as in Word
clearpage_before = set()
for i, el in enumerate(blocks):
    if el.tag != W + 'p':
        continue
    has_page_br = any(br.get(W + 'type') == 'page' for br in el.iter(W + 'br'))
    if not has_page_br:
        continue
    j = i + 1
    while j < len(blocks):
        e2 = blocks[j]
        if e2.tag == W + 'p' and pstyle(e2).startswith('Heading') and accepted_text(e2).strip():
            clearpage_before.add(j)
            break
        if e2.tag == W + 'sdt' or (e2.tag in (W + 'p', W + 'tbl') and not is_block_empty(e2)):
            break
        j += 1

d = Doc()

SECTION_FILES = []
sec_no = 0

for i, el in enumerate(blocks):
    tag = el.tag.split('}')[1]
    if tag == 'sectPr':
        continue
    if tag == 'sdt':
        # the TOC block
        d.out('%% Table of contents (Word TOC replaced by LaTeX)')
        continue
    if tag == 'tbl':
        if is_block_empty(el):
            continue
        d.close_code(); d.close_lists()
        if i in fig_of_block:
            d.para(render_figure_table(el, fig_of_block[i]))
        else:
            d.para(render_table(el, i))
        continue
    if tag != 'p':
        continue
    p = el
    st = pstyle(p)
    if st.startswith('TOC'):
        if st == 'TOCHeading' and accepted_text(p).strip() and not d.sections:
            title = polish(render_para_inline(p).strip())
            d.front.append(r'\begin{center}')
            d.front.append(r'{\usgtitlefont %s\par}' % title)
            d.front.append(r'\end{center}')
            d.front.append('')
        continue
    # fully deleted paragraph (paragraph mark deleted and no kept content)
    if parmark_deleted(p) and is_block_empty(p):
        continue
    if st.startswith('Heading'):
        level = int(st[7:])
        txt = render_para_inline(p).strip()
        if not txt:
            continue
        if level == 1:
            # start a new section file
            sec_no += 1
            title_txt = accepted_text(p).strip()
            d.close_code(); d.close_lists()
            d.sections.append((sec_no, title_txt, []))
        elif i in clearpage_before:
            d.out(r'\clearpage')
        txt = polish(txt)
        labels = []
        if i in heading_labels:
            labels.append(heading_labels[i])
        else:
            lab = heading_label_for(i, p)
            heading_labels[i] = lab
            labels.append(lab)
        d.heading(level, txt, labels)
        continue
    if st == 'Bibliography':
        txt = render_para_inline(p).strip()
        if txt:
            def _url(m):
                u = m.group(0)
                for a, b in ((r'\_\allowbreak{}', '_'), (r'\_', '_'),
                             (r'\%', '%'), (r'\#', '#'), (r'\&', '&')):
                    u = u.replace(a, b)
                return r'\url{%s}' % u.rstrip('.,')
            txt = re.sub(r'https?://[^\s]+', _url, txt)
            d.para(r'\noindent\hangindent=2em\hangafter=1 ' + txt)
        continue
    # code-ish?
    ck = codeish_kind(p)
    if ck:
        if ck == 'empty-code':
            d.code_line('empty-code', '')
        else:
            d.code_line(ck, raw_code_text(p).rstrip(), code_size_class(p))
        continue
    # list item?
    np_ = num_props(p)
    if np_ and st == 'ListParagraph':
        lvl, env = np_
        txt = render_para_inline(p).strip()
        txt = txt.replace('\x00SEQFIG\x00', '')
        if txt:
            d.item(lvl, env, txt)
        continue
    if st == 'ListParagraph' and not np_:
        # continuation paragraph inside a list – attach to current item
        txt = render_para_inline(p).strip()
        if txt:
            if d.list_stack:
                d.target().append('')
                d.target().append(txt)
            else:
                d.para(txt)
        continue
    # normal paragraph
    txt = render_para_inline(p).strip()
    txt = txt.replace('\x00SEQFIG\x00', '')
    if not txt:
        continue
    d.para(txt, par_jc(p))

d.close_code()
d.close_lists()

# ----------------------------------------------------------------- output
os.makedirs(f'{PROJ}/sections', exist_ok=True)

slug_titles = []
main_inputs = []
for num, title, lines in d.sections:
    slug = slugify(title)
    fname = f'{num:02d}-{slug}.tex'
    with open(f'{PROJ}/sections/{fname}', 'w') as f:
        f.write('%% Section %d: %s\n' % (num, title))
        f.write('\n'.join(lines).rstrip() + '\n')
    main_inputs.append(f'sections/{fname[:-4]}')
    slug_titles.append((num, title, fname))

with open(f'{PROJ}/frontmatter-body.tex', 'w') as f:
    f.write('\n'.join(d.front).rstrip() + '\n')

with open(f'{PROJ}/citations/zotero-citations.json', 'w') as f:
    json.dump(zotero_cites, f, indent=1)

print('sections written:')
for num, title, fname in slug_titles:
    print(f'  {fname}  ({title})')
print()
print('front matter lines:', len(d.front))
print('zotero citations:', len(zotero_cites))
print()
print('WARNINGS (%d):' % len(warns))
seen = set()
for w_ in warns:
    if w_ not in seen:
        print('  -', w_)
        seen.add(w_)
