"""Helpers for Fig 3 v6: edit text in matplotlib SVGs without touching the plotted marks.
glyph_text_to_live(): replace matplotlib glyph-path text groups (<g id="text_N"> with a <!-- text --> comment and <use> glyphs)
by live <text> elements at the same anchor and size (Arial), optionally rewording or dropping them.
set_live_text(): reword or drop live <text> elements (svg.fonttype none output)."""
import re, html


def _groups(s, prefix="text_"):
    """yield (start, end, id) of every <g id="text_N"> ... </g> block (balanced)."""
    for m in re.finditer(r'<g id="(%s\d+)"' % prefix, s):
        i = m.start(); depth = 0
        for t in re.finditer(r'<g\b|</g>', s[i:]):
            depth += 1 if t.group(0) == '<g' else -1
            if depth == 0:
                yield i, i + t.end(), m.group(1); break


def glyph_text_to_live(s, reword=None, drop=(), font="Arial", size_mult=1.0):
    reword = reword or {}; out = []; last = 0
    for a, b, gid in list(_groups(s)):
        blk = s[a:b]; cm = re.search(r'<!--(.*?)-->', blk, re.S)
        if not cm: continue
        txt = html.unescape(cm.group(1).strip()); out.append(s[last:a]); last = b
        if any(d in txt for d in drop): out.append(f'<g id="{gid}"/>'); continue
        lines = txt.split('\n')
        heads = re.findall(r'<g style="fill: ?([^;"]+)[^"]*" transform="translate\(([-\d.]+) ([-\d.]+)\)( rotate\([^)]*\))? scale\(([-\d.]+) -[-\d.]+\)"', blk)
        pre = blk[:blk.index(cm.group(0))]   # bbox patches etc. stay
        new = []
        for k, (fill, x, y, rot, sc) in enumerate(heads[:len(lines)]):
            t = reword.get(lines[k], lines[k]); fs = float(sc) * 100 * size_mult
            tf = f' transform="{rot.strip()}"' if rot else ''
            if rot:   # rotate about the anchor
                tf = f' transform="translate({x} {y}){rot}"'; x, y = 0, 0
            new.append(f'<text x="{x}" y="{y}"{tf} style="font-family: {font}, Helvetica, sans-serif; font-size: {fs:.2f}px; fill: {fill}">{html.escape(t)}</text>')
        out.append(pre.replace(f'<g id="{gid}">', f'<g id="{gid}">', 1) + "\n".join(new) + "\n</g>")
    out.append(s[last:]); return "".join(out)


def set_live_text(s, reword=None, drop=()):
    reword = reword or {}
    def f(m):
        inner = html.unescape(re.sub(r'<[^>]+>', '', m.group(2)))
        if any(d == inner or (d.endswith('*') and inner.startswith(d[:-1])) for d in drop): return ''
        if inner in reword: return m.group(1) + html.escape(reword[inner]) + '</text>'
        return m.group(0)
    return re.sub(r'(<text\b[^>]*>)(.*?)</text>', f, s, flags=re.S)
