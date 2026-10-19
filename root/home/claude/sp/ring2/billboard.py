# Billboard-style tiles for the arena ring. Every tile: f(w, h, t, spec) -> PIL RGB image, periodic in LOOP where animated.
# Colour rule: the asset's brand colour is the base; variants (solid / deep / light / dark+accent / gradient / hue-shift)
# create many combinations; the strip builder keeps neighbouring tiles in different hue families.
import sys, math, colorsys, numpy as np, random
sys.path.insert(0, '/home/claude/dca-media/kit/scene'); sys.path.insert(0, '/home/claude/sp/ring2')
import os
from PIL import Image, ImageDraw
import content as C, universe as U

for s in U.SYMS:                                    # register prices so content.Ticker works for every symbol
    C.BASE[s] = U.PRICE[s]
    if s not in C.DAY: C.DAY[s] = float(C.rng('day' + s).choice([-1, 1]) * C.rng('dm' + s).uniform(0.25, 2.6))
LOOP = C.LOOP; F = C.F; fit = C.fit
UP, DN, WH, DK = (40, 205, 115), (240, 72, 72), (250, 250, 248), (18, 18, 22)
END_T = float(os.environ.get('VIDEO_END', 3600.0))   # countdown reaches 0:00 here

def lum(c): return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
def mix(a, b, k): return tuple(int(x * (1 - k) + y * k) for x, y in zip(a, b))
def hsv(c): return colorsys.rgb_to_hsv(*[v / 255 for v in c])
def from_hsv(h, s, v): return tuple(int(255 * x) for x in colorsys.hsv_to_rgb(h % 1, max(0, min(1, s)), max(0, min(1, v))))
def neutral(c): return hsv(c)[1] < 0.22 or hsv(c)[2] < 0.18
def hue_family(c):
    if neutral(c): return 'n'
    return int(((hsv(c)[0] * 360 + 15) % 360) // 30)          # 12 families of 30 degrees

VARIANTS = ['solid', 'deep', 'light', 'dark', 'grad', 'shift']
def colours(brand, var):
    """returns bg (or (top,bottom) for gradient), fg text colour, accent"""
    h, s, v = hsv(brand)
    if neutral(brand):
        brand = (40, 40, 46) if v < 0.5 else (200, 204, 210)
        if var in ('solid', 'grad', 'shift'): var = 'dark' if v < 0.5 else 'light'
    if var == 'solid': bg = brand
    elif var == 'deep': bg = from_hsv(h, min(1, s * 1.05), v * (0.78 if 0.05 <= h <= 0.2 else 0.55)) if not (0.05 <= h <= 0.2) else from_hsv(h + 0.03, min(1, s * 1.1), min(1, v * 0.92))
    elif var == 'light': bg = from_hsv(h, s * 0.85, min(1, v * 1.12 + 0.08))
    elif var == 'dark': bg = (14, 14, 18)
    elif var == 'grad': bg = (brand, from_hsv(h, s, v * 0.45))
    else: bg = from_hsv(h + 0.055, s, min(1, v * 1.02))
    base = bg[1] if isinstance(bg, tuple) and isinstance(bg[0], tuple) else bg
    fg = DK if lum(base) > 160 else WH
    acc = brand if var in ('dark',) else (from_hsv(h, s * 0.9, max(0.35, v * 0.42)) if var == 'light' else (WH if fg == WH else DK))
    return bg, fg, acc

def chart_colours(brand):
    h_, s_, v_ = hsv(brand)
    band = brand if not neutral(brand) else (200, 204, 210)
    band = from_hsv(h_, max(0.7, s_), max(0.85, v_)) if not neutral(brand) else band
    return (12, 13, 18), WH, band
def chart_header(im, d, w, h, band, title, pct=None):
    hh = int(h * 0.2); d.rectangle((0, 0, w, hh), fill=band); tc = DK if lum(band) > 160 else WH
    f, sz = fit(d, title, w * 0.62, hh * 0.72, 'k'); bb = d.textbbox((0, 0), title, font=f)
    d.text((w * 0.04, hh / 2 - (bb[1] + bb[3]) / 2), title, font=f, fill=tc)
    if pct is not None: badge(d, w * 0.04 + d.textlength(title, font=f) + hh * 0.3, hh * 0.22, pct, hh * 0.36, band)
    return hh
def canvas(w, h, bg):
    if isinstance(bg, tuple) and isinstance(bg[0], tuple):
        a = np.linspace(0, 1, h)[:, None, None]; top, bot = np.array(bg[0], float), np.array(bg[1], float)
        arr = (top * (1 - a) + bot * a).repeat(w, 1).astype(np.uint8); return Image.fromarray(arr)
    return Image.new('RGB', (w, h), bg)

def quote(sym, chan, rate, t):
    return C.Ticker.get(sym, chan, rate).at(t)          # price, pct, age, dir
def fmt(sym, p): return C.fmt(sym, p) if p >= 5 else f"{p:,.3f}"
def badge(d, x, y, pct, size, on):
    s = f"{'▲' if pct >= 0 else '▼'} {abs(pct):.2f}%"; f = F(size); tw = d.textlength(s, font=f); pad = size * 0.35
    col = UP if pct >= 0 else DN
    d.rounded_rectangle((x - 2, y - 2, x + tw + 2 * pad + 2, y + size * 1.45 + 2), radius=size * 0.3, fill=WH if lum(on) < 140 else DK)
    d.rounded_rectangle((x, y, x + tw + 2 * pad, y + size * 1.45), radius=size * 0.28, fill=col)
    d.text((x + pad, y + size * 0.1), s, font=f, fill=WH)
    return tw + 2 * pad
def walk(key, n, trend):
    r = C.rng(key); v = np.cumsum(r.normal(0, 1, n)); v = (v - v.min()) / (np.ptp(v) + 1e-9)
    return v * 0.7 + np.linspace(0, 1, n) * (0.3 if trend >= 0 else -0.3)

# ---------------------------------------------------------------- tiles
def t_card(w, h, t, sp):
    sym = sp['sym']; bg, fg, acc = colours(U.BRAND[sym], sp['var']); im = canvas(w, h, bg); d = ImageDraw.Draw(im)
    p, pct, age, dr = quote(sym, 'c' + sym, 5, t); fl = C.flash_amt(age, 1.2)
    x0 = w * 0.07; fT, sT = fit(d, sym, w * 0.5, h * 0.30, 'k'); d.text((x0, h * 0.08), sym, font=fT, fill=fg)
    ps = fmt(sym, p); fP, sP = fit(d, ps, w * 0.5, h * 0.17); d.text((x0, h * 0.08 + sT * 1.3), ps, font=fP, fill=mix(fg, UP if dr > 0 else DN, fl * 0.8))
    badge(d, x0, h * 0.08 + sT * 1.3 + sP * 1.45, pct, sP * 0.42, bg if not isinstance(bg[0], tuple) else bg[1])
    v = walk('cw' + sym, 60, pct); X0, X1, Y0, Y1 = w * 0.58, w * 0.95, h * 0.2, h * 0.8
    pts = [(X0 + (X1 - X0) * i / 59, Y1 - y * (Y1 - Y0)) for i, y in enumerate(v)]
    d.line(pts, fill=acc if sp['var'] != 'dark' else (UP if pct >= 0 else DN), width=max(3, int(h * 0.018)), joint='curve')
    return im

def t_trio(w, h, t, sp):
    syms = sp['syms']; im = Image.new('RGB', (w, h)); d = ImageDraw.Draw(im); cw = w / 3
    vars_ = sp.get('vars', ('solid', 'deep', 'solid'))
    for i, s in enumerate(syms):
        bg, fg, acc = colours(U.BRAND[s], vars_[i]); bg = bg if not isinstance(bg[0], tuple) else bg[0]
        x = int(i * cw); d.rectangle((x, 0, int((i + 1) * cw), h), fill=bg)
        p, pct, age, dr = quote(s, 'tr' + s, 4, t); fl = C.flash_amt(age, 1.2)
        fT, sT = fit(d, s, cw * 0.8, h * 0.22, 'k'); d.text((x + cw * 0.09, h * 0.1), s, font=fT, fill=fg)
        ps = fmt(s, p); fP, sP = fit(d, ps, cw * 0.8, h * 0.14); d.text((x + cw * 0.09, h * 0.1 + sT * 1.35), ps, font=fP, fill=mix(fg, UP if dr > 0 else DN, fl * 0.8))
        badge(d, x + cw * 0.09, h * 0.1 + sT * 1.35 + sP * 1.5, pct, sP * 0.45, bg)
        v = walk('tw' + s, 40, pct); pts = [(x + cw * 0.08 + cw * 0.84 * j / 39, h * 0.92 - y * h * 0.2) for j, y in enumerate(v)]
        d.line(pts, fill=acc, width=max(3, int(h * 0.014)))
    return im

def t_candle(w, h, t, sp):
    sym = sp['sym']; bg, fg, band = chart_colours(U.BRAND[sym])
    im = canvas(w, h, bg); d = ImageDraw.Draw(im); iv = 2.0; N = int(LOOP / iv); r = C.rng('cd' + sym)
    inc = r.normal(0, 1, N); inc -= inc.mean(); path = np.cumsum(inc) * U.PRICE[sym] * 0.0016
    closes = U.PRICE[sym] * (1 + C.DAY[sym] / 100) + path - path[-1]
    nshow = 30; cw = w * 0.84 / nshow; k = t / iv; i_now = int(k); fr = k - i_now; seg = []
    for j in range(i_now - nshow + 1, i_now + 1):
        o = closes[(j - 1) % N]; c = closes[j % N]
        if j == i_now: c = o + (c - o) * min(1, fr * 1.4)
        rr = C.rng(f'w{sym}{j % N}'); hi = max(o, c) + abs(rr.normal()) * U.PRICE[sym] * 0.0012; lo = min(o, c) - abs(rr.normal()) * U.PRICE[sym] * 0.0012
        seg.append((o, hi, lo, c))
    L = min(s[2] for s in seg); Hh = max(s[1] for s in seg); Y = lambda v: h * 0.92 - (v - L) / (Hh - L + 1e-9) * h * 0.6
    for gy in range(2, 5): d.line((0, h * gy / 5, w, h * gy / 5), fill=(34, 36, 46))
    for j, (o, hi, lo, c) in enumerate(seg):
        x = w * 0.03 + j * cw + cw / 2; col = UP if c >= o else DN
        d.line((x, Y(hi), x, Y(lo)), fill=col, width=max(2, int(cw * 0.12))); d.rectangle((x - cw * 0.33, min(Y(o), Y(c)), x + cw * 0.33, max(Y(o), Y(c)) + 1), fill=col)
    last = seg[-1][3]; pct = (last / U.PRICE[sym] - 1) * 100; chart_header(im, d, w, h, band, sym, pct)
    return im

def t_trend(w, h, t, sp):
    sym = sp['sym']; bg, fg, acc = colours(U.BRAND[sym], sp['var']); im = canvas(w, h, bg).convert('RGBA'); d = ImageDraw.Draw(im)
    p, pct, age, dr = quote(sym, 'tn' + sym, 4, t)
    v = walk('tn' + sym, 90, pct); ph = (t % 30) / 30; n = max(2, int(len(v) * min(1, 0.15 + ph * 1.25)))
    X0, X1, Y0, Y1 = w * 0.04, w * 0.96, h * 0.32, h * 0.92
    pts = [(X0 + (X1 - X0) * i / (len(v) - 1), Y1 - y * (Y1 - Y0)) for i, y in enumerate(v)][:n]
    ov = Image.new('RGBA', im.size); od = ImageDraw.Draw(ov)
    od.polygon(pts + [(pts[-1][0], Y1), (pts[0][0], Y1)], fill=(*(acc if sp['var'] == 'dark' else fg), 55))
    im = Image.alpha_composite(im, ov); d = ImageDraw.Draw(im)
    d.line(pts, fill=acc if sp['var'] == 'dark' else fg, width=max(4, int(h * 0.02)), joint='curve')
    d.ellipse((pts[-1][0] - h * 0.02, pts[-1][1] - h * 0.02, pts[-1][0] + h * 0.02, pts[-1][1] + h * 0.02), fill=fg)
    fT, sT = fit(d, sym, w * 0.3, h * 0.17, 'k'); d.text((w * 0.04, h * 0.06), sym, font=fT, fill=fg)
    ps = fmt(sym, p); fP, sP = fit(d, ps, w * 0.3, h * 0.12); d.text((w * 0.04 + d.textlength(sym, font=fT) + sT * 0.4, h * 0.1), ps, font=fP, fill=fg)
    badge(d, w * 0.72, h * 0.08, pct, h * 0.05, bg if not isinstance(bg[0], tuple) else bg[0])
    return im.convert('RGB')

def t_rank(w, h, t, sp):
    syms = sp['syms']; im = Image.new('RGB', (w, h)); d = ImageDraw.Draw(im); n = len(syms); rh = h / n
    fN, _ = fit(d, 'WWWWW', w * 0.40, rh * 0.52, 'k')                     # one size for every row
    fB, _ = fit(d, '▼0.00%', w * 0.27, rh * 0.40)
    bw = d.textlength('▼00.00%', font=fB); bpad = rh * 0.16; bx1 = w * 0.955; bx0 = bx1 - bw - 2 * bpad
    def vc(txt, f, y0, y1):
        bb = d.textbbox((0, 0), txt, font=f); return (y0 + y1) / 2 - (bb[1] + bb[3]) / 2
    for i, s_ in enumerate(syms):
        bg, fg, acc = colours(U.BRAND[s_], 'solid'); bg = bg if not isinstance(bg[0], tuple) else bg[0]; y0 = i * rh; y1 = y0 + rh
        d.rectangle((0, y0, w, y1), fill=bg)
        p, pct, age, dr = quote(s_, 'rk' + s_, 3, t); fl = C.flash_amt(age, 1.4)
        if fl > 0: d.rectangle((0, y0, w, y1), fill=mix(bg, UP if dr > 0 else DN, fl * 0.45))
        num = f'{i + 1}'; d.text((w * 0.045, vc(num, fN, y0, y1)), num, font=fN, fill=mix(fg, bg, 0.35))
        d.text((w * 0.045 + d.textlength('0 ', font=fN), vc(s_, fN, y0, y1)), s_, font=fN, fill=fg)
        cs = f"{'▲' if pct >= 0 else '▼'}{abs(pct):.2f}%"; tw = d.textlength(cs, font=fB)
        d.rounded_rectangle((bx0, y0 + rh * 0.2, bx1, y1 - rh * 0.2), radius=rh * 0.12, fill=UP if pct >= 0 else DN)
        d.text(((bx0 + bx1) / 2 - tw / 2, vc(cs, fB, y0, y1)), cs, font=fB, fill=WH)
    return im

def t_bars(w, h, t, sp):
    syms = sp['syms']; bgc, fgc, _ = colours(U.BRAND[syms[0]], 'deep'); bgc = bgc if not isinstance(bgc[0], tuple) else bgc[1]
    im = Image.new('RGB', (w, h), bgc); d = ImageDraw.Draw(im); n = len(syms)
    fT, sT = fit(d, 'VOLUME', w * 0.4, h * 0.09, 'k'); d.text((w * 0.04, h * 0.04), 'VOLUME', font=fT, fill=mix(fgc, bgc, 0.2))
    rh = h * 0.82 / n; ph = (t % 40) / 40; grow = min(1, ph * 3)
    for i, s in enumerate(syms):
        y = h * 0.16 + i * rh; base = 0.35 + 0.6 * C.rng('bv' + s).random(); wob = 0.06 * math.sin(2 * math.pi * (t / 6 + i * 0.37))
        L = w * 0.62 * min(1, base * grow + wob * grow); col = U.BRAND[s] if not neutral(U.BRAND[s]) else (200, 204, 210)
        col = from_hsv(hsv(col)[0], max(0.75, hsv(col)[1]), max(0.85, hsv(col)[2])) if not neutral(col) else (235, 235, 238)
        if hue_family(col) == hue_family(bgc): col = (245, 245, 245)
        f, sz = fit(d, s, w * 0.24, rh * 0.5, 'k'); d.text((w * 0.04, y + rh * 0.22), s, font=f, fill=WH)
        d.rounded_rectangle((w * 0.3, y + rh * 0.2, w * 0.3 + max(4, L), y + rh * 0.8), radius=rh * 0.15, fill=col)
    return im

def t_heat(w, h, t, sp):
    syms = sp['syms']; cols = sp.get('cols', 4); rows = math.ceil(len(syms) / cols); cw, ch = w / cols, h / rows
    im = Image.new('RGB', (w, h), (8, 8, 10)); d = ImageDraw.Draw(im)
    for i, s in enumerate(syms):
        p, pct, age, dr = quote(s, 'ht' + s, 2, t); fl = C.flash_amt(age, 1.5)
        a = min(1, 0.4 + abs(pct) / 3); col = mix((8, 8, 10), UP if pct >= 0 else DN, a * 0.9); col = mix(col, WH, fl * 0.15)
        x = (i % cols) * cw; y = (i // cols) * ch; d.rectangle((x + 3, y + 3, x + cw - 3, y + ch - 3), fill=col)
        f, sz = fit(d, s, cw * 0.8, min(cw, ch) * 0.26, 'k'); tw = d.textlength(s, font=f); d.text((x + (cw - tw) / 2, y + ch * 0.22), s, font=f, fill=WH)
        st = f"{pct:+.2f}%"; f2, _ = fit(d, st, cw * 0.8, min(cw, ch) * 0.18); tw = d.textlength(st, font=f2); d.text((x + (cw - tw) / 2, y + ch * 0.58), st, font=f2, fill=WH)
    return im

def squarify(items, x, y, w, h):
    out = []; items = sorted(items, key=lambda a: -a[1]); tot = sum(v for _, v in items)
    def lay(items, x, y, w, h):
        if not items: return
        if len(items) == 1: out.append((items[0][0], x, y, w, h)); return
        s = sum(v for _, v in items); acc = 0; k = 0
        for k, (n, v) in enumerate(items):
            acc += v
            if acc >= s / 2: break
        a, b = items[:k + 1], items[k + 1:]; fa = sum(v for _, v in a) / s
        if w >= h: lay(a, x, y, w * fa, h); lay(b, x + w * fa, y, w * (1 - fa), h)
        else: lay(a, x, y, w, h * fa); lay(b, x, y + h * fa, w, h * (1 - fa))
    lay(items, x, y, w, h); return out
def t_tree(w, h, t, sp):
    syms = sp['syms']; im = Image.new('RGB', (w, h), (8, 8, 10)); d = ImageDraw.Draw(im)
    items = [(s, U.PRICE[s] ** 0.35 * (0.6 + C.rng('tw' + s).random())) for s in syms]
    for s, x, y, ww, hh in squarify(items, 0, 0, w, h):
        p, pct, age, dr = quote(s, 'tm' + s, 2, t); fl = C.flash_amt(age, 1.6)
        col = mix((10, 10, 12), UP if pct >= 0 else DN, min(1, 0.45 + abs(pct) / 3)); col = mix(col, WH, fl * 0.12)
        d.rectangle((x + 2, y + 2, x + ww - 2, y + hh - 2), fill=col)
        if ww > h * 0.18 and hh > h * 0.12:
            f, sz = fit(d, s, ww * 0.8, min(hh * 0.3, ww * 0.3), 'k'); tw = d.textlength(s, font=f)
            d.text((x + (ww - tw) / 2, y + hh * 0.5 - sz * 0.9), s, font=f, fill=WH)
            st = f"{pct:+.2f}%"; f2, s2 = fit(d, st, ww * 0.7, sz * 0.6); tw = d.textlength(st, font=f2); d.text((x + (ww - tw) / 2, y + hh * 0.5 + sz * 0.25), st, font=f2, fill=WH)
    return im

def t_score(w, h, t, sp): return C.s_score(w, h, t, {})
def t_glyph(w, h, t, sp):
    sym = sp['sym']; bg, fg, acc = colours(U.BRAND[sym], sp['var']); im = canvas(w, h, bg); d = ImageDraw.Draw(im)
    p, pct, age, dr = quote(sym, 'gl' + sym, 2, t); g = '▲' if pct >= 0 else '▼'
    f, sz = fit(d, g, w * 0.8, h * 0.52); tw = d.textlength(g, font=f)
    d.text(((w - tw) / 2, h * 0.04), g, font=f, fill=UP if pct >= 0 else DN)
    s = f'{sym}  {pct:+.2f}%'; f2, s2 = fit(d, s, w * 0.85, h * 0.16, 'k'); tw = d.textlength(s, font=f2); d.text(((w - tw) / 2, h * 0.72), s, font=f2, fill=fg)
    return im

DOT = None
def t_led(w, h, t, sp):
    """amber dot-matrix quote wall; rows page every 10 s with a flip"""
    syms = sp['syms']; rows = 3; per = 10.0; pg = int(t // per); ph = (t % per) / per
    small = Image.new('L', (w // 8, h // 8), 0); d = ImageDraw.Draw(small); rh = small.height / rows
    for r in range(rows):
        s = syms[(pg * rows + r) % len(syms)]; p, pct, *_ = quote(s, 'ld' + s, 1, t)
        txt = f"{s:<5} {fmt(s, p):>9} {'+' if pct >= 0 else '-'}{abs(pct):.2f}%"
        f, sz = fit(d, txt, small.width * 0.94, rh * 0.8, 'm'); d.text((small.width * 0.03, r * rh + rh * 0.1), txt, font=f, fill=255)
    m = np.asarray(small, np.float32) / 255
    if ph < 0.06: m *= ph / 0.06
    big = cv2_resize(m, (w, h)); yy, xx = np.mgrid[0:h, 0:w]
    rr = ((xx % 8) - 3.5) ** 2 + ((yy % 8) - 3.5) ** 2
    dots = np.clip(1.35 - rr / 11.0, 0, 1)                       # round lamps, almost touching
    lit = big * dots
    import cv2 as _cv
    glow = _cv.GaussianBlur(lit, (0, 0), 3.0)
    amber = np.array([255, 200, 80], np.float32); off = np.array([70, 46, 20], np.float32)
    arr = np.full((h, w, 3), 8, np.float32) + (dots * (1 - big))[..., None] * off * 0.55 + lit[..., None] * amber * 1.25 + glow[..., None] * amber * 0.9
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
def cv2_resize(m, size):
    import cv2; return cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)

def t_tape(w, h, t, sp): return C.s_tape(w, h, t, {'syms': sp['syms'], 'lanes': 2, 'laps': [2, 1], 'bg': sp.get('bg', (8, 8, 10))})

def t_depth(w, h, t, sp):
    sym = sp['sym']; bg, fg, band = chart_colours(U.BRAND[sym]); im = canvas(w, h, bg).convert('RGBA'); d = ImageDraw.Draw(im)
    n = 40; r = C.rng('dp' + sym); wob = lambda i: 1 + 0.08 * math.sin(2 * math.pi * (t / 8 + i * 0.13))
    bid = np.cumsum([r.uniform(0.5, 1.5) * wob(i) for i in range(n)]); ask = np.cumsum([r.uniform(0.5, 1.5) * wob(i + 50) for i in range(n)])
    top = max(bid[-1], ask[-1]); X0, X1 = w * 0.05, w * 0.95; mid = (X0 + X1) / 2; Y0, Y1 = h * 0.32, h * 0.84
    grid = (40, 42, 52)
    for k in range(1, 4): y = Y1 - (Y1 - Y0) * k / 4; d.line((X0, y, X1, y), fill=grid, width=1)
    p0 = U.PRICE[sym]; step = p0 * 0.002
    fS, sS = fit(d, fmt(sym, p0 * 1.004), (X1 - X0) * 0.22, h * 0.06)
    for k in (-2, -1, 0, 1, 2):
        x = mid + k * (X1 - X0) / 4.2; d.line((x, Y1, x, Y1 + h * 0.025), fill=mix(grid, fg, 0.5), width=2)
        lab = fmt(sym, p0 + k * step); tw = d.textlength(lab, font=fS); d.text((x - tw / 2, Y1 + h * 0.035), lab, font=fS, fill=(190, 192, 200))
    bp = [(mid - (i + 1) / n * (mid - X0), Y1 - bid[i] / top * (Y1 - Y0)) for i in range(n)]
    ap = [(mid + (i + 1) / n * (X1 - mid), Y1 - ask[i] / top * (Y1 - Y0)) for i in range(n)]
    ov = Image.new('RGBA', im.size); od = ImageDraw.Draw(ov)
    od.polygon([(mid, Y1)] + bp + [(bp[-1][0], Y1)], fill=(*UP, 150)); od.polygon([(mid, Y1)] + ap + [(ap[-1][0], Y1)], fill=(*DN, 150))
    im = Image.alpha_composite(im, ov); d = ImageDraw.Draw(im); d.line([(mid, Y1)] + bp, fill=UP, width=4); d.line([(mid, Y1)] + ap, fill=DN, width=4)
    d.line((mid, Y0 - h * 0.02, mid, Y1), fill=mix(grid, fg, 0.6), width=2)
    for lab, x, col in (('BID', X0 + w * 0.02, UP), ('ASK', X1 - w * 0.02, DN)):
        f, sz = fit(d, lab, w * 0.15, h * 0.07, 'k'); tw = d.textlength(lab, font=f); d.text((x if col == UP else x - tw, Y0 - h * 0.08), lab, font=f, fill=col)
    p, pct, *_ = quote(sym, 'dp' + sym, 3, t); d = ImageDraw.Draw(im); chart_header(im, d, w, h, band, f'{sym} {fmt(sym, p)}')
    return im.convert('RGB')

def t_book(w, h, t, sp):
    sym = sp['sym']; bg, fg, band = chart_colours(U.BRAND[sym]); im = Image.new('RGB', (w, h), bg); d = ImageDraw.Draw(im)
    hh = chart_header(im, d, w, h, band, sym)
    p0 = U.PRICE[sym]; tick = max(0.01, round(p0 * 0.0004, 2)); rows = 4; rh = (h - hh - h * 0.04) / (2 * rows); y0 = hh + h * 0.02
    beat = (t * 2.0) % 1; hit = int(t * 2.0)
    f, sz = fit(d, '00000.00', w * 0.46, rh * 0.68, 'm')
    for i in range(2 * rows):
        side = 1 if i < rows else -1; k = rows - i if side > 0 else i - rows + 1
        price = p0 + side * k * tick; y = y0 + i * rh
        size = 0.25 + 0.7 * C.rng(f'bs{sym}{i}').random() + 0.12 * math.sin(2 * math.pi * (t / 5 + i * 0.3))
        col = (255, 90, 90) if side > 0 else (60, 225, 130); L = w * 0.5 * size
        fl = (1 - beat) ** 2 if (hit % (2 * rows)) == i else 0
        d.rectangle((w - L, y + 2, w, y + rh - 2), fill=mix(mix(bg, col, 0.38), WH, fl * 0.45))
        bb = d.textbbox((0, 0), '0', font=f); ty = y + rh / 2 - (bb[1] + bb[3]) / 2
        d.text((w * 0.05, ty), fmt(sym, price), font=f, fill=col)
        q = f'{int(size * 900):>4}'; tw = d.textlength(q, font=f); d.text((w * 0.95 - tw, ty), q, font=f, fill=WH)
    return im

LAND = None
MAPIDX = [('S&P', 0.20, 0.36, 'VOO', 1), ('DAX', 0.50, 0.28, 'EWG', -1), ('NIKKEI', 0.86, 0.36, 'EWJ', -1),
          ('NIFTY', 0.69, 0.50, 'INDA', -1), ('IBOV', 0.33, 0.68, 'EWZ', 1)]
def t_map(w, h, t, sp):
    global LAND
    import cv2
    if LAND is None:
        LAND = cv2.imread('/home/claude/dca-media/kit/cockpit/data/land_mask.png', cv2.IMREAD_GRAYSCALE)
    im = Image.new('RGB', (w, h), (4, 10, 24)); d = ImageDraw.Draw(im); sp_ = 7
    lm = cv2.resize(LAND, (w // sp_, h // sp_), interpolation=cv2.INTER_AREA)
    for yy in range(lm.shape[0]):
        for xx in range(lm.shape[1]):
            cx, cy = xx * sp_ + sp_ / 2, yy * sp_ + sp_ / 2
            if lm[yy, xx] > 90: d.ellipse((cx - 2.7, cy - 2.7, cx + 2.7, cy + 2.7), fill=(110, 190, 255))
            else: d.ellipse((cx - 1.2, cy - 1.2, cx + 1.2, cy + 1.2), fill=(14, 30, 56))
    arr = np.asarray(im).astype(np.float32); glow = cv2.GaussianBlur(arr, (0, 0), 3.5)
    im = Image.fromarray(np.clip(arr + glow * 0.45, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    for i, (nm, fx, fy, s, side) in enumerate(MAPIDX):
        p, pct, age, dr = quote(s, 'mp' + s, 2, t); x, y = fx * w, fy * h; pulse = 0.5 + 0.5 * math.sin(2 * math.pi * (t / 4 + i * 0.27))
        col = UP if pct >= 0 else DN; r = h * (0.012 + 0.01 * pulse)
        d.ellipse((x - r * 2.2, y - r * 2.2, x + r * 2.2, y + r * 2.2), fill=mix((4, 10, 24), col, 0.45)); d.ellipse((x - r, y - r, x + r, y + r), fill=col)
        lab = f'{nm} {pct:+.1f}%'; f, sz = fit(d, lab, w * 0.24, h * 0.085, 'k'); tw = d.textlength(lab, font=f)
        lx = x + r * 2.8 if side > 0 else x - r * 2.8 - tw; d.rectangle((lx - 6, y - sz * 0.75, lx + tw + 6, y + sz * 0.6), fill=(4, 10, 24)); d.text((lx, y - sz * 0.7), lab, font=f, fill=WH)
    return im

SEG7 = {'0': 'abcdef', '1': 'bc', '2': 'abged', '3': 'abgcd', '4': 'fgbc', '5': 'afgcd', '6': 'afgedc', '7': 'abc', '8': 'abcdefg', '9': 'abcdfg'}
def _seg_mask(text, W, H, th_px=None):
    """seven-segment layout -> (lit mask, all-segments mask), float arrays H x W"""
    import cv2 as _cv
    lit = np.zeros((H, W), np.float32); allm = np.zeros((H, W), np.float32)
    n_d = sum(c.isdigit() for c in text); n_s = len(text) - n_d
    dw = W / (n_d + 0.45 * n_s); x = 0.0; th = th_px or dw * 0.17; g = th * 0.9
    for c in text:
        if not c.isdigit():
            cx = x + dw * 0.225
            for yy in ((0.33, 0.67) if c == ':' else (0.93,)):
                _cv.circle(lit, (int(cx), int(H * yy)), int(th * 0.62), 1, -1); _cv.circle(allm, (int(cx), int(H * yy)), int(th * 0.62), 1, -1)
            x += dw * 0.45; continue
        x0, x1 = x + dw * 0.2, x + dw * 0.8; y0, ym, y1 = H * 0.05, H * 0.5, H * 0.95
        segs = {'a': ((x0 + g, y0), (x1 - g, y0)), 'd': ((x0 + g, y1), (x1 - g, y1)), 'g': ((x0 + g, ym), (x1 - g, ym)),
                'f': ((x0, y0 + g), (x0, ym - g)), 'e': ((x0, ym + g), (x0, y1 - g)), 'b': ((x1, y0 + g), (x1, ym - g)), 'c': ((x1, ym + g), (x1, y1 - g))}
        for k, (p0, p1) in segs.items():
            for m in ([allm, lit] if k in SEG7[c] else [allm]):
                _cv.line(m, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 1, int(th))
        x += dw
    return lit, allm
CLOCK_TEXT = None                      # override: str ('' = all segments unlit) or callable -> str
def t_clock(w, h, t, sp):
    """basketball game clock: dot-matrix seven-segment digits, amber, unlit segments faintly visible"""
    import cv2 as _cv
    rem = max(0.0, END_T - t)
    s = f'{int(rem // 60):02d}:{int(rem % 60):02d}' if rem >= 60 else f'{int(rem):02d}.{int(rem * 10) % 10}'
    if CLOCK_TEXT is not None: s = CLOCK_TEXT if isinstance(CLOCK_TEXT, str) else CLOCK_TEXT()
    blank = s == ''
    if blank: s = '88:88'
    pad_x, pad_y = int(w * 0.08), int(h * 0.16); W_, H_ = w - 2 * pad_x, h - 2 * pad_y
    P = max(6, int(h / 22)); lit, allm = _seg_mask(s, W_, H_, th_px=2.0 * P)
    if blank: lit = lit * 0
    yy, xx = np.mgrid[0:H_, 0:W_]
    dots = np.clip(1.4 - (((xx % P) - (P - 1) / 2) ** 2 + ((yy % P) - (P - 1) / 2) ** 2) / (P * P * 0.13), 0, 1)
    on = (_cv.GaussianBlur(lit, (0, 0), 0.6) > 0.5) * dots; off = (_cv.GaussianBlur(allm, (0, 0), 0.6) > 0.5) * dots * (1 - (lit > 0.5))
    amber = np.array([255, 178, 60], np.float32); core = np.array([255, 236, 190], np.float32)
    glow = _cv.GaussianBlur(on, (0, 0), P * 0.9)
    arr = np.zeros((h, w, 3), np.float32) + np.array([16, 8, 8], np.float32)
    sub = arr[pad_y:pad_y + H_, pad_x:pad_x + W_]
    sub += off[..., None] * np.array([40, 36, 28], np.float32) + glow[..., None] * amber * 0.7
    sub[:] = sub * (1 - on[..., None]) + (on[..., None] * (amber * 0.55 + core * 0.45))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

R = dict(card=t_card, trio=t_trio, candle=t_candle, trend=t_trend, rank=t_rank, bars=t_bars, heat=t_heat, tree=t_tree, score=t_score,
         glyph=t_glyph, led=t_led, tape=t_tape, depth=t_depth, book=t_book, map=t_map, clock=t_clock)
WIDTH = dict(card=1.8, trio=3.0, candle=2.3, trend=2.6, rank=1.15, bars=1.7, heat=2.2, tree=2.2, score=1.25, glyph=1.0, led=2.6,
             tape=3.2, depth=1.55, book=1.25, map=2.8, clock=2.2)
