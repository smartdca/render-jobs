# Arena ring renderer for the Gemini photo (works at 2x: 2390x1792), standalone.
import os, sys, math, json, random
S2 = 2
os.environ.update(RING_W=str(1195 * S2), RING_H=str(896 * S2), RING_MODEL='/home/claude/sp/ring2/modelg2.json',
                  LIP_Y=str(408 * S2), RING_SRC='/home/claude/sp/ring2/gem.jpg')
sys.path.insert(0, '/home/claude/sp/ring'); sys.path.insert(0, '/home/claude/dca-media/kit/scene')
import numpy as np, cv2
from PIL import Image, ImageDraw
import geo, strip as S
import content as C

H, W = geo.H, geo.W
HS, HL, HF = 600, 80, 40
REV = 120.0
_x2 = '/home/claude/sp/ring2/gem_x2.png'
BASE = (cv2.imread(_x2) if os.path.exists(_x2) else
        cv2.resize(cv2.imread('/home/claude/sp/ring2/gem.jpg'), (W, H), interpolation=cv2.INTER_CUBIC)).astype(np.float32)
_g8 = cv2.cvtColor(np.clip(BASE, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
_wm = np.zeros(_g8.shape, np.uint8)
for x0, y0, x1, y1 in [(240, 366, 470, 408), (1080, 452, 1155, 510)]:
    x0, y0, x1, y1 = x0 * S2, y0 * S2, x1 * S2, y1 * S2; _wm[y0:y1, x0:x1] = (_g8[y0:y1, x0:x1] > 105) * 255
BASE = cv2.inpaint(np.clip(BASE, 0, 255).astype(np.uint8), cv2.dilate(_wm, np.ones((7, 7), np.uint8)), 7, cv2.INPAINT_TELEA).astype(np.float32)
CORNER_R = int(os.environ.get('CORNER_R', 60))
G = geo.build(1, HS, HL, HS)
# ---- outer band ends traced from the original photo (1x coords). Two layers, each fitted separately:
#  screen face: tapers to a sharp tip; its bottom edge rises from the tip along the arc.
#  thick rim (ticker band): between the screen bottom and the white edge line; its outer edge recedes from the
#  screen tip back inward as it goes down (the band wrapping round).
_ENDS = [
  dict(side=0, x0=0, x1=200, xb=150,
       sb=[(5,442),(10,433),(20,425),(30,418),(50,408),(70,400),(100,392),(150,383)],      # screen bottom
       wb=[(13,479),(15,477),(22,468),(40,453),(67,442),(120,432)],                        # rim bottom (white line)
       limb=[(35,330),(27.5,350),(20,370),(16,380),(12,420),(7.5,432),(5,442)],             # screen outer edge (top part = model edge)
       rimedge=[(5,442),(10,467),(13,479)]),                                               # rim outer edge
  dict(side=1, x0=1040, x1=1195, xb=1090,
       sb=[(1090,458),(1110,467),(1130,478),(1147,490),(1160,500),(1167,507)],
       wb=[(1060,477),(1093,487),(1127,500),(1147,513),(1160,527),(1163,532)],
       limb=[(1063,303),(1090,327),(1107,353),(1120,380),(1133,412),(1147,453),(1157,480),(1167,507)],   # traced from the photo's screen edge (2026-10-25: was a notch at y 420)
       rimedge=[(1167,507),(1168,513),(1167,527),(1163,532)]),
]
def _ip(pts, x):
    p = np.array(pts, float); o = np.argsort(p[:, 0]); return np.interp(x, p[o, 0], p[o, 1])
def _ipy(pts, y):
    p = np.array(pts, float); o = np.argsort(p[:, 1]); return np.interp(y, p[o, 1], p[o, 0])
_mN0, _mL0 = G['mN'].copy(), G['mL'].copy()
_keys = ('sN', 'vN', 'sL', 'vL')
_src = {k: G[k].copy() for k in _keys}
for E in _ENDS:
    side = E['side']; xs1 = np.arange(E['x0'], E['x1'], 0.5)
    tipx = E['sb'][0][0] if side == 0 else E['sb'][-1][0]
    for x1 in xs1:
        x2 = int(round(x1 * S2))
        if not (0 <= x2 < W): continue
        # nearest column where the model has both layers (used for columns past the model's end)
        xr = x2
        while 0 <= xr < W and not (_mN0[:, xr].any() and _mL0[:, xr].any()): xr += 1 if side == 0 else -1
        if not (0 <= xr < W): continue
        rN = np.nonzero(_mN0[:, xr])[0]; rL = np.nonzero(_mL0[:, xr])[0]
        top, bN, bL = rN.min(), rN.max(), rL.max()
        # blend weight: 1 = traced line, 0 = model (keeps the long middle untouched)
        k = np.clip((E['xb'] - x1) / (E['xb'] - tipx), 0, 1) if side == 0 else np.clip((x1 - E['xb']) / (tipx - E['xb']), 0, 1)
        k = k * k * (3 - 2 * k)
        sb = bN + (_ip(E['sb'], x1) * S2 - bN) * k
        wb = bL + (_ip(E['wb'], x1) * S2 - bL) * k
        if wb <= sb + 2 or sb <= top + 2: continue
        ys = np.arange(top, int(round(wb)) + 1)
        src = np.where(ys <= sb, top + (ys - top) * (bN - top) / (sb - top), bN + 1 + (ys - sb) * (bL - bN - 1) / (wb - sb))
        src = np.clip(src, 0, H - 1); f0 = np.floor(src).astype(int); f1 = np.minimum(f0 + 1, H - 1); w = src - f0
        for kk in _keys: G[kk][:, x2] = 0; G[kk][ys, x2] = _src[kk][f0, xr] * (1 - w) + _src[kk][f1, xr] * w
        colN = np.zeros(H, bool); colL = np.zeros(H, bool)
        colN[top:int(np.floor(sb)) + 1] = True; colL[int(np.floor(sb)) + 1:int(round(wb)) + 1] = True
        # outer edges: screen right of / left of the limb; rim right of / left of its receding edge
        yy = np.arange(H) / S2
        tipy = E['limb'][-1][1]
        edge = np.where(yy <= tipy, _ipy(E['limb'], yy), _ipy(E['rimedge'], yy))     # one outer outline for both layers
        lim = (yy >= E['limb'][0][1]) & (yy <= E['rimedge'][-1][1] + 0.5)
        outside = (lim & ((x1 < edge) if side == 0 else (x1 > edge))) | (yy > E['rimedge'][-1][1] + 0.5)
        colN &= ~outside; colL &= ~outside
        colN &= ~(yy > tipy + 0.5)
        G['mN'][:, x2] = colN; G['mL'][:, x2] = colL
        if x2 + 1 < W and x1 % 1 == 0.5 and False: pass


# side edges: per row, extend each layer out to the traced outer edge by extrapolating its content coordinates
for E in _ENDS:
    side = E['side']; tipy = E['limb'][-1][1]; step = -1 if side == 0 else 1
    for y2 in range(int(E['limb'][0][1] * S2), int(E['rimedge'][-1][1] * S2) + 1):
        y1 = y2 / S2
        ex = (_ipy(E['limb'], y1) if y1 <= tipy else _ipy(E['rimedge'], y1)) * S2
        row = G['mN'][y2] | G['mL'][y2]
        zc = np.nonzero(row[int(E['x0'] * S2):int(E['x1'] * S2)])[0]
        if len(zc) < 6: continue
        zc = zc + int(E['x0'] * S2)
        xa = zc.min() if side == 0 else zc.max()                       # current outermost band pixel
        lay = 'mN' if G['mN'][y2, xa] else 'mL'
        sk, vk = ('sN', 'vN') if lay == 'mN' else ('sL', 'vL')
        xb = xa - 6 * step                                               # a pixel further in, for the slope
        if not (0 <= xb < W) or not G[lay][y2, xb]: continue
        slope = (G[sk][y2, xa] - G[sk][y2, xb]) / (xa - xb)
        xt = int(np.floor(ex)) if side == 0 else int(np.ceil(ex))
        rng_ = range(xa - 1, xt - 1, -1) if side == 0 else range(xa + 1, xt + 1)
        for x2 in rng_:
            if not (0 <= x2 < W): break
            G[lay][y2, x2] = True; G[sk][y2, x2] = G[sk][y2, xa] + slope * (x2 - xa); G[vk][y2, x2] = G[vk][y2, xa]
# final outline clip over the whole end zone (also covers columns the loop skipped)
_yy1 = (np.arange(H) / S2)[:, None]; _xx1 = (np.arange(W) / S2)[None, :]
for E in _ENDS:
    side = E['side']; tipy = E['limb'][-1][1]
    edge = np.where(_yy1 <= tipy, _ipy(E['limb'], _yy1), _ipy(E['rimedge'], _yy1))
    zone = (_xx1 >= E['x0']) & (_xx1 < E['x1'])
    lim = (_yy1 >= E['limb'][0][1])
    out = zone & lim & (((_xx1 < edge) if side == 0 else (_xx1 > edge)) | (_yy1 > E['rimedge'][-1][1] + 0.5))
    G['mN'] &= ~(out | (zone & lim & (_yy1 > tipy + 0.5))); G['mL'] &= ~out
CN, CL = G['CN'], G['CL']
CROP = (200, 1544)                                   # 16:9 window (y0, y1) in 2x pixels

def vis_interval(s, m, C_):
    v = np.sort(np.mod(s[m], C_)); gaps = np.diff(np.concatenate([v, [v[0] + C_]]))
    i = int(np.argmax(gaps)); a = v[(i + 1) % len(v)]; b = v[i]
    return (a, b + C_) if b < a else (a, b)
aN, bN = vis_interval(G['sN'], G['mN'], CN)

def sample(strip, s, v, hh):
    L = strip.shape[1]; wrap = np.concatenate([strip, strip[:, :4]], 1)
    return cv2.remap(wrap, np.mod(s, L).astype(np.float32), (np.clip(v, 0, 1) * (hh - 1)).astype(np.float32),
                     cv2.INTER_LINEAR).astype(np.float32)
def fit_len(img, L): return cv2.resize(img, (int(round(L)), img.shape[0]), interpolation=cv2.INTER_AREA)

# ---------------- market strips
SEG_N = S.SEG + S.SEG3
SEG_F = [x for x in S.SEG2 if x[1].get('type') != 'tape' and not (x[1].get('type') == 'candle' and x[1].get('sym') == 'META')]
_mc = {}
def market(t):
    k = round(t * 30) / 30
    if k not in _mc:
        _mc.clear()
        _mc[k] = (fit_len(np.array(S.make(HS, k, seg=SEG_N))[:, :, ::-1], CN).astype(np.float32),
                  fit_len(np.array(S.make(HS, k, seg=SEG_F))[:, :, ::-1], CF).astype(np.float32))
    return _mc[k]

import universe as _U, random as _rnd
_pool = _U.SYMS[:]; _rnd.Random(7).shuffle(_pool)
SYMS = _pool[:70]
SYMS_F = _pool[70:140]
try:
    import billboard as _BB                                    # registers prices for all symbols
except Exception:
    pass
def ticker_tex(h, bg, gapw=6, chan='lip', fs=0.52, boost=False, syms=None):
    f = C.F(h * fs); fk = C.F(h * fs, 'k'); d0 = ImageDraw.Draw(Image.new('RGB', (8, 8)))
    items = []
    for s_ in (syms or SYMS):
        p, pct, *_ = C.Ticker.get(s_, chan, 0).at(0.0); items.append((s_, C.fmt(s_, p), f"{'▲' if pct >= 0 else '▼'}{abs(pct):.2f}%", pct))
    ws = [d0.textlength(a + '  ', font=fk) + d0.textlength(b + '  ', font=f) + d0.textlength(c + ' ' * gapw, font=f) for a, b, c, _ in items]
    im = Image.new('RGB', (int(sum(ws)) + 1, h), bg); d = ImageDraw.Draw(im); x = 0
    for (a, b, c, pct), w in zip(items, ws):
        y = h * (0.12 if boost else 0.2); d.text((x, y), a, font=fk, fill=(255, 255, 255)); xx = x + d.textlength(a + '  ', font=fk)
        d.text((xx, y), b, font=f, fill=(235, 235, 240) if boost else (200, 200, 206)); xx += d.textlength(b + '  ', font=f)
        d.text((xx, y), c, font=f, fill=((70, 240, 140) if pct >= 0 else (255, 95, 95)) if boost else (C.UP if pct >= 0 else C.DN)); x += w
    return np.array(im)[:, :, ::-1].astype(np.float32)
TT = ticker_tex(HL, (8, 8, 12)); n = max(1, round(CL / TT.shape[1])); TICK = fit_len(np.concatenate([TT] * n, 1), CL)

# ---------------- audience fascia strip
FA = json.load(open('/home/claude/sp/ring2/fascia_g.json'))
fx = np.arange(FA['x0'] * S2, FA['x1'] * S2)
ft = np.polyval(FA['top'], fx / S2) * S2; fb = np.polyval(FA['bot'], fx / S2) * S2; fth = fb - ft
fs_ = np.concatenate([[0], np.cumsum(1.0 / fth[:-1])]) * HF
FMASK = np.zeros((H, W), bool); FU = np.zeros((H, W), np.float32); FV = np.zeros((H, W), np.float32)
for i, x in enumerate(fx):
    for y in range(int(np.floor(ft[i])), int(np.ceil(fb[i])) + 1):
        vv = (y + 0.5 - ft[i]) / fth[i]
        if 0 <= vv <= 1: FMASK[y, x] = True; FU[y, x] = fs_[i]; FV[y, x] = vv
FSOFT = cv2.GaussianBlur(FMASK.astype(np.float32), (3, 3), 0)[..., None]
FTEX = ticker_tex(HF, (4, 5, 10), gapw=5, chan='fas', fs=0.70, boost=True, syms=SYMS_F)

# ---------------- slogan texture (single phrase, evenly repeated)
PALETTES = {'orange': ((20, 15, 11), (232, 150, 60)), 'green': ((6, 18, 12), (40, 205, 115)), 'red': ((22, 8, 8), (240, 72, 72)),
            'mono': ((10, 10, 12), (245, 245, 240)), 'gold': ((18, 14, 6), (226, 182, 72))}
def ink(text, h):
    f = C.F(h * 0.56, 'k'); d = ImageDraw.Draw(Image.new('L', (8, 8))); bb = d.textbbox((0, 0), text, font=f)
    im = Image.new('L', (int(math.ceil(bb[2] - bb[0])) + 2, h), 0)
    ImageDraw.Draw(im).text((1 - bb[0], (h - (bb[3] - bb[1])) / 2 - bb[1]), text, font=f, fill=255)
    return np.asarray(im, np.float32) / 255
def loop_mask(text, L, h, g=0.8):
    L = int(round(L)); it = ink(text, h); tw = it.shape[1]; n = max(1, int(round(L / (tw + g * h))))
    while n > 1 and (L - n * tw) / n < 0.3 * h: n -= 1                   # long phrases: never overlap the repeats
    gap = (L - n * tw) / n
    m = np.zeros((h, L), np.float32); t_ = int(round(h * 0.035)); m[:t_] = 1; m[-t_:] = 1; x = gap / 2
    for _ in range(n):
        xi = int(round(x)); w = max(0, min(tw, L - xi)); m[:, xi:xi + w] = np.maximum(m[:, xi:xi + w], it[:, :w]); x += tw + gap
    return m
def colour(mask, pal, inv):
    bg, fg = [np.array(c[::-1], np.float32) for c in PALETTES[pal]]
    if inv: bg, fg = fg, bg
    return bg[None, None] * (1 - mask[..., None]) + fg[None, None] * mask[..., None]

# ---------------- spotlights hung under the rim, flashes in the crowd
lipb = np.array([np.nonzero(G['mL'][:, x])[0].max() if G['mL'][:, x].any() else -1 for x in range(W)])
ORIG = [(x, float(lipb[x] + 24)) for x in np.linspace(260, 2100, 7).astype(int)]
SW = 0.5; _YY, _XX = np.mgrid[0:int(H * SW), 0:int(W * SW)].astype(np.float32)
def beams(t, amt, t0):
    h, w = int(H * SW), int(W * SW); acc = np.zeros((h, w), np.float32)
    for i, (ox, oy) in enumerate(ORIG):
        ang = math.radians(28 * math.sin(2 * math.pi * 0.23 * (t - t0) + i * 0.9) + (i - 3) * 6); L = 1100; sp = math.radians(6.5)
        o = np.array([ox, oy]) * SW
        p1 = o + np.array([math.sin(ang - sp), math.cos(ang - sp)]) * L * SW; p2 = o + np.array([math.sin(ang + sp), math.cos(ang + sp)]) * L * SW
        cone = np.zeros((h, w), np.float32); cv2.fillConvexPoly(cone, np.array([o, p1, p2]).astype(np.int32), 1.0)
        cone *= np.clip(1 - np.sqrt((_XX - o[0]) ** 2 + (_YY - o[1]) ** 2) / (L * SW), 0, 1) ** 1.3
        c = (p1 + p2) / 2; cv2.ellipse(cone, (int(c[0]), int(c[1])), (int(85 * SW), int(26 * SW)), 0, 0, 360, 0.55, -1)
        acc += cone
    return cv2.resize(cv2.GaussianBlur(acc, (0, 0), 4) * 0.8 + cv2.GaussianBlur(acc, (0, 0), 22) * 0.7, (W, H)) * amt
CY0, CY1 = 1290, 1536                                  # crowd rows (2x) below the strip, inside the 16:9 crop (was 1780: half the flashes fell outside the picture)
FLASH_RATE = (0.9, 7.5)                                # phone flashes per second, all on screen: normal / cheering (2026-10-25)
BURST = 0.55                                           # while cheering, about half the flashes go off in bursts (several phones at once)
def make_flashes(T, busy, seed=11):
    r = np.random.default_rng(seed); ev = []; t = 0.0
    def one(t):
        for _ in range(60):
            x = int(r.uniform(30, W - 30)); y = int(r.uniform(CY0, CY1))
            if not (G['mN'][y, x] or G['mF'][y, x] or G['mL'][y, x] or FMASK[max(0, y - 8):y + 9, x].any()): break
        near = (y - CY0) / (CY1 - CY0)
        ev.append((t, x, y, (2.6 + 5.5 * near) * 1.2, 0.55 + 0.45 * near, r.uniform(0.08, 0.16)))
    while t < T:
        b = busy(t); rate = FLASH_RATE[1] if b else FLASH_RATE[0]
        if b and r.random() < BURST / 5.0 * 5 / (1 + 4 * BURST):          # a burst stands for ~5 flashes
            t += r.exponential(5 / rate)
            for _ in range(int(r.integers(3, 9))): one(t + r.uniform(0, 0.05))
        else:
            t += r.exponential(1 / rate); one(t)
    ev.sort(key=lambda e: e[0])
    return ev
_fy, _fx2 = np.mgrid[0:81, 0:81].astype(np.float32) - 40
def add_flashes(out, t, ev):
    for (t0, x, y, sz, br, du) in ev:
        if not (t0 <= t < t0 + du + 0.12): continue
        k = br * (1.0 if t < t0 + du else max(0, 1 - (t - t0 - du) / 0.12)) ** 1.5
        rr = np.sqrt(_fx2 ** 2 + _fy ** 2)
        g = (np.exp(-(rr / sz) ** 2) * 1.4 + np.exp(-(rr / (sz * 3.5))) * 0.45 +
             (np.exp(-(np.abs(_fy) / (0.6 + sz * 0.15)) ** 2 - (np.abs(_fx2) / (sz * 6)) ** 2) +
              np.exp(-(np.abs(_fx2) / (0.6 + sz * 0.15)) ** 2 - (np.abs(_fy) / (sz * 6)) ** 2)) * 0.35) * k * 255
        y0, x0 = y - 40, x - 40; ys = slice(max(0, y0), min(H, y0 + 81)); xs = slice(max(0, x0), min(W, x0 + 81))
        out[ys, xs] += g[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0, None] * np.array([1.0, 0.98, 0.94])
    return out

# leftover original-screen pixels just outside the fitted areas -> dark bezel
# inner screen continues up behind the hanging lights: extend the inner band, replace only screen-coloured pixels
_y2t0 = geo.Y2T; geo.Y2T = _y2t0 + 0.06; os.environ['YIF'] = str(geo.Y2B + 0.07)
G2 = geo.build(1, HS, HL, HS); geo.Y2T = _y2t0; os.environ.pop('YIF')
CF = G2['CF']; aF, bF = vis_interval(G2['sF'], G2['mF'], CF)
_FLAG = (644, 676, 485, 550)                                    # Canada flag hangs in front (1x box)
_fb = BASE[_FLAG[2] * S2:_FLAG[3] * S2, _FLAG[0] * S2:_FLAG[1] * S2]
_fh = cv2.cvtColor(np.clip(_fb, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
_flag = (((_fh[..., 0] < 12) | (_fh[..., 0] > 165)) & (_fh[..., 1] > 70) & (_fh[..., 2] > 60)) | ((_fh[..., 1] < 50) & (_fh[..., 2] > 150))
_flag = cv2.morphologyEx(_flag.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
_fm = np.zeros((H, W), bool); _fm[_FLAG[2] * S2:_FLAG[3] * S2, _FLAG[0] * S2:_FLAG[1] * S2] = _flag
G['mF'] &= ~_fm
# old-screen leftovers poking above the inner band's arc (middle): repaint with the surrounding ceiling
_ft = np.array([np.nonzero(G['mF'][:, x])[0].min() if G['mF'][:, x].any() else -1 for x in range(W)])
_hs = cv2.cvtColor(np.clip(BASE, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
_old = ((_hs[..., 1] > 55) & (_hs[..., 2] > 50)) | (_hs[..., 2] > 115)
_lamp = cv2.dilate(((_hs[..., 2] > 165) & (_hs[..., 1] < 70)).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
_old &= ~_lamp                                                     # keep the hanging lamps (inpainting them smeared them into haze)
_strip = np.zeros((H, W), bool)
for _x2 in range(380 * S2, 1010 * S2):
    if _ft[_x2] > 0: _strip[max(0, _ft[_x2] - 50 * S2):_ft[_x2] + 2, _x2] = True
_cand = (_strip & _old & ~_fm).astype(np.uint8)
_n, _lab = cv2.connectedComponents(cv2.dilate(_cand, np.ones((3, 3), np.uint8)))
_touch = np.zeros(_n, bool)                                   # keep only leftovers attached to the band's top edge
for _x2 in range(380 * S2, 1010 * S2):
    if _ft[_x2] > 2: _touch[_lab[_ft[_x2] - 3:_ft[_x2] + 1, _x2]] = True
_touch[0] = False
_er = (cv2.dilate((_touch[_lab] & _cand.astype(bool)).astype(np.uint8), np.ones((5, 5), np.uint8)) & _strip.astype(np.uint8) & (~_fm).astype(np.uint8))
BASE = cv2.inpaint(np.clip(BASE, 0, 255).astype(np.uint8), _er * 255, 11, cv2.INPAINT_TELEA).astype(np.float32)

_rows = np.arange(H)[:, None]
MFX = G2['mF'] & ~G['mF'] & ~G['mN'] & ~G['mL'] & (_rows > np.maximum(lipb, 0)[None, :] + 2)
_hsv = cv2.cvtColor(np.clip(BASE, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
_scr = (((_hsv[..., 1] > 70) & (_hsv[..., 2] > 55)) | ((BASE[..., 0] > BASE[..., 2] + 25) & (BASE[..., 0] > 55))).astype(np.float32)
_scr = cv2.morphologyEx(_scr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
_ftop = np.array([np.nonzero(G['mF'][:, x])[0].min() if G['mF'][:, x].any() else H for x in range(W)])
MFX &= (_rows > _ftop[None, :] - 70)                                     # only just above the inner screen (gaps between lights)
_cols = np.arange(W)[None, :]
_ends = (_cols < 330) | (_cols > 2250)
MFX &= ((_cols < 420 * S2) | (_cols > 990 * S2))                                    # edge-on ends: no lights in front -> replace fully
XA = (np.clip(cv2.GaussianBlur(np.maximum(_scr, _ends.astype(np.float32)) * MFX, (0, 0), 1.2) * 1.4, 0, 1) * MFX)[..., None]
_ring = (G['mN'] | G['mF'] | G['mL'] | MFX).astype(np.uint8)
_near = cv2.dilate(_ring, np.ones((41, 41), np.uint8)).astype(bool) & ~_ring.astype(bool)
_b, _g, _r = BASE[..., 0], BASE[..., 1], BASE[..., 2]
_blue = (_b > _r + 35) & (_b > 70)
_fix = (_near & _blue).astype(np.float32)
_fix = np.clip(cv2.GaussianBlur(_fix, (0, 0), 1.5) * 1.6, 0, 1)[..., None]
BASE = BASE * (1 - _fix) + np.array([14, 13, 12], np.float32) * _fix
# edge-on inner-screen ends: hand-marked outlines of the old screen ends; filled by stretching the nearest inner-band column
_PL = [(150,452),(115,462),(85,477),(67,490),(60,500),(62,520),(67,540),(85,546),(102,550),(122,555),(150,565)]
_PR = [(1040,490),(1078,502),(1103,519),(1123,532),(1138,550),(1138,566),(1126,578),(1102,588),(1065,601),(1040,608)]
_mFall = G['mF'] | (XA[..., 0] > 0.5)
ENDX = []
for poly, side in ((_PL, 0), (_PR, 1)):
    pm = np.zeros((H, W), np.uint8); cv2.fillPoly(pm, [np.array(poly, np.int32) * S2], 1)
    pm = pm.astype(bool) & ~G['mN'] & ~G['mL']
    for y in np.nonzero(pm.any(1))[0]:
        band = np.nonzero(_mFall[y])[0]; cols = np.nonzero(pm[y] & ~_mFall[y])[0]
        if len(band) < 5 or len(cols) == 0: continue
        xe = band.min() + 2 if side == 0 else band.max() - 2
        cols = cols[cols < xe] if side == 0 else cols[cols > xe]
        if len(cols): ENDX.append((y, cols.min(), cols.max(), xe))
AA_N = cv2.GaussianBlur(G['mN'].astype(np.float32), (0, 0), 0.9)[..., None]
AA_L = cv2.GaussianBlur(G['mL'].astype(np.float32), (0, 0), 0.9)[..., None]
AA_F = cv2.GaussianBlur(G['mF'].astype(np.float32), (0, 0), 1.1)[..., None]
LED = np.ones((H, W, 1), np.float32); LED[::3] *= 0.93; LED[:, ::3] *= 0.95


# ================= realism: LED panel structure, halation, light spill, reflection, lens & film ==================
def _dots(u, v, pitch):
    fu = (u / pitch) % 1.0 - 0.5; fv = (v / pitch) % 1.0 - 0.5
    return np.exp(-(fu * fu + fv * fv) / 0.09)                       # 1 at LED centre, ~0.2 between LEDs
def _seams(u, step, w=0.012):
    f = (u / step) % 1.0; return 1 - 0.28 * np.exp(-(np.minimum(f, 1 - f) / w) ** 2)
_dN = _dots(G['sN'], G['vN'] * HS, 9.0) * _seams(G['sN'], 300) * _seams(G['vN'] * HS, 300)
_dF = _dots(G2['sF'], G2['vF'] * HS, 9.0) * _seams(G2['sF'], 300)
_dL = _dots(G['sL'], G['vL'] * HL, 6.0)
LEDTEX = np.ones((H, W), np.float32)
for _m, _d in ((G['mN'], _dN), (G2['mF'] | G['mF'], _dF), (G['mL'], _dL)):
    LEDTEX = np.where(_m, 0.80 + 0.20 * _d, LEDTEX)
LEDTEX = cv2.GaussianBlur(LEDTEX, (0, 0), 0.7)[..., None]
_yy0 = np.arange(H, dtype=np.float32)[:, None]
_lb = np.where(lipb > 0, lipb, 0).astype(np.float32)[None, :]
_dist = _yy0 - _lb
REFL_W = (np.exp(-np.clip(_dist, 0, None) / 46.0) * (_dist > 2) * (lipb[None, :] > 0)).astype(np.float32)
_scr_any = (G['mN'] | G['mL'] | G['mF'] | G2['mF']).astype(np.float32)
REFL_W *= (1 - cv2.dilate(_scr_any, np.ones((5, 5), np.uint8)))
REFL_W = (REFL_W * 0.20)[..., None]
_mirror_y = np.clip(2 * _lb - _yy0, 0, H - 1).astype(np.float32).repeat(W, 1) if False else None
MAPX = np.tile(np.arange(W, dtype=np.float32)[None, :], (H, 1)); MAPY = np.clip(2 * _lb - _yy0, 0, H - 1).astype(np.float32) * np.ones((1, W), np.float32)
_vy, _vx = np.mgrid[0:H, 0:W].astype(np.float32)
VIGN = (1 - 0.32 * (((_vx - W / 2) / (W / 2)) ** 2 + ((_vy - H * 0.45) / (H / 2)) ** 2) ** 1.3).clip(0.45, 1)[..., None]
RECEIVE = (1 - _scr_any)[..., None]
def realism(out, A, t):
    a = A[..., None]
    emis = out * a                                                    # light emitted by the screens
    # LED panel: lifted black level, LED grid, cabinet seams, slight highlight roll-off
    scr = out * LEDTEX * 0.97 + np.array([9, 8, 8], np.float32)
    scr = 255 * (1 - np.exp(-scr / 255 * 1.25)) / (1 - np.exp(-1.25))
    out = out * (1 - a) + scr * a
    # light spill / halation (low-res big blurs)
    small = cv2.resize(emis, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    near_g = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (W, H), interpolation=cv2.INTER_LINEAR)
    wide_g = cv2.resize(cv2.GaussianBlur(small, (0, 0), 14), (W, H), interpolation=cv2.INTER_LINEAR)
    out = out + RECEIVE * (out * wide_g / 255 * 1.6 + wide_g * 0.14 + near_g * 0.14)   # surroundings lit by the ring + haze
    out = out + near_g * 0.10 * a + cv2.GaussianBlur(np.clip(out - 170, 0, 255) * a, (0, 0), 7) * 0.30   # halation over the panels
    # reflection of the ring on the dark underside just below the rim
    ref = cv2.remap(cv2.GaussianBlur(emis, (0, 0), 4), MAPX, MAPY, cv2.INTER_LINEAR)
    out = out + ref * REFL_W
    return out
ROI_ORIGIN = None
GRAIN = True
def finish(out, t, seed=None):
    # lens: slight chromatic aberration, softness, vignette; film: grain; grade: cool shadows / warm highlights
    b, g, r = out[..., 0], out[..., 1], out[..., 2]
    sh = 1.5
    ox, oy, Wf, Hf = ROI_ORIGIN if ROI_ORIGIN else (0, 0, W, H)          # region renders: same lens geometry as the full frame
    r = cv2.warpAffine(r, np.float32([[1.0008, 0, 0.0008 * ox - Wf * 0.0004], [0, 1.0008, 0.0008 * oy - Hf * 0.0004]]), (W, H), borderMode=cv2.BORDER_REFLECT)
    b = cv2.warpAffine(b, np.float32([[0.9992, 0, -0.0008 * ox + Wf * 0.0004], [0, 0.9992, -0.0008 * oy + Hf * 0.0004]]), (W, H), borderMode=cv2.BORDER_REFLECT)
    out = np.stack([b, g, r], -1)
    out = cv2.GaussianBlur(out, (0, 0), 0.6) * 0.5 + out * 0.5
    lum = out.mean(-1, keepdims=True) / 255
    out = out + (1 - lum) ** 2 * np.array([4, 1, -2], np.float32) + lum ** 2 * np.array([-3, 1, 5], np.float32)
    out = out * VIGN
    if not GRAIN: return out                                          # ar1: grain is added at output size in the final pass
    rng = np.random.default_rng((int(round(t * 30)) if seed is None else seed) + 7)
    gr = cv2.resize(rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    out = out + gr[..., None] * (3.2 + 2.0 * (1 - lum))
    return out

def compose(near, far, rim, phi, t, ring_k=1.0, glow=0.0, beam_amt=0.0, beam_t0=0.0, flashes=(), rim_k=1.0, rim_off=0.0,
            fglow=None, beam_tint=None, fas_off=None, fas=None, seed=None):
    """near/far: strip images (or None=off); rim: strip (L=CL) ; phi: revolutions"""
    out = BASE.copy(); A = np.zeros((H, W), np.float32)
    for strip, s, v, m, C_, gain in [(near, G['sN'] + phi * CN, G['vN'], G['mN'], CN, 0.95),
                                     (far, G2['sF'] - phi * CF, G2['vF'], G['mF'], CF, 0.80)]:
        img = np.full((H, W, 3), 9, np.float32) if strip is None else sample(strip.astype(np.uint8), s, v, HS)
        img = img * ring_k + 9 * (1 - ring_k)
        if m is G['mF']:
            out = out * (1 - AA_F) + img * gain * AA_F; A = np.maximum(A, AA_F[..., 0])
        else:
            al = AA_N if m is G['mN'] else None
            al = AA_N
            out = out * (1 - al) + img * gain * al; A = np.maximum(A, al[..., 0])
        if m is G['mF']:
            ix = img
            ix = ix * ring_k + 9 * (1 - ring_k)
            out = out * (1 - XA) + ix * gain * XA; A = np.maximum(A, XA[..., 0])
    img = np.full((H, W, 3), 8, np.float32) if rim is None else sample(rim.astype(np.uint8), G['sL'] + rim_off, G['vL'], HL)
    img = img * rim_k + 8 * (1 - rim_k)
    out = out * (1 - AA_L) + img * 0.9 * AA_L; A = np.maximum(A, AA_L[..., 0])
    if ENDX:
        cols_ = {}
        for (y, x0, x1, xs_) in ENDX:
            side = 0 if x1 < xs_ else 1
            ref = out[y, max(0, xs_ - 40):xs_ + 1] if side else out[y, xs_:xs_ + 41]
            cols_.setdefault(side, []).append((y, x0, x1, ref.mean(0)))
        for side, lst in cols_.items():
            lst.sort(); ys_ = np.array([l[0] for l in lst]); cs_ = np.array([l[3] for l in lst])
            k_ = np.exp(-0.5 * ((ys_[:, None] - ys_[None, :]) / 14.0) ** 2); cs_ = (k_ @ cs_) / k_.sum(1, keepdims=True)
            for (y, x0, x1, _), c in zip(lst, cs_):
                n = x1 - x0 + 1; kk = np.linspace(0.5, 0.9, n) if side == 0 else np.linspace(0.9, 0.5, n)
                out[y, x0:x1 + 1] = c[None] * kk[:, None]; A[y, x0:x1 + 1] = 1
    a = cv2.GaussianBlur(A, (3, 3), 0)[..., None]
    out = realism(out, a[..., 0], t)
    if glow:
        yy = np.arange(H)[:, None, None]; wgt = np.clip((yy - 1260) / 400, 0, 1)
        out = out * (1 + glow * wgt * (fglow if fglow is not None else 1.0) * (1 - a))
    FT_ = FTEX if fas is None else fas; L_ = FT_.shape[1]; wrap = np.concatenate([FT_, FT_[:, :4]], 1)
    fimg = cv2.remap(np.clip(wrap, 0, 255).astype(np.uint8), np.mod(FU + (55 * S2 * t if fas_off is None else fas_off), L_).astype(np.float32), (FV * (HF - 1)).astype(np.float32), cv2.INTER_LINEAR).astype(np.float32)
    out = out * (1 - FSOFT) + np.clip(fimg * 1.25, 0, 255) * FSOFT
    if beam_amt > 0:
        tint = np.array([200, 225, 255], np.float32) if beam_tint is None else beam_tint
        out = out + beams(t, beam_amt, beam_t0)[..., None] * tint * 0.62
    out = add_flashes(out, t, flashes)
    out = finish(out, t, seed)
    return np.clip(out, 0, 255).astype(np.uint8)[CROP[0]:CROP[1]]
