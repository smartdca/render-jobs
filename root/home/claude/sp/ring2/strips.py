# Build the long content strips (one per band per music segment) from billboard tiles.
import sys, random, numpy as np
sys.path.insert(0, '/home/claude/sp/ring2')
import billboard as BB, universe as U, content as C
from PIL import Image
import cv2
C.WM = Image.open('/home/claude/dca-media/kit/scene/dcascore-wordmark-white.png').convert('RGBA'); C.WM = C.WM.crop(C.WM.getbbox())

FAM = {s: BB.hue_family(U.BRAND[s]) for s in U.SYMS}
BY_FAM = {}
for s, f in FAM.items(): BY_FAM.setdefault(f, []).append(s)
def WARM(s):
    c = U.BRAND[s]; return (not BB.neutral(c)) and 18 <= BB.hsv(c)[0] * 360 <= 65
SINGLE = ['card', 'candle', 'trend', 'glyph', 'depth', 'book']
GROUP_SAME = ['trio', 'rank']            # same hue family, different shades
MARKET = ['heat', 'tree', 'bars', 'led', 'map']

def build(seg, band, length_h, clock=True):
    """returns list of (width_px_ratio, spec); total width == length_h (in strip heights)"""
    r = random.Random(1000 * seg + (0 if band == 'near' else 1))
    # weighted shuffle: warm (yellow / orange) and bright brands come up more often
    def wgt(s):
        h, sat, v = BB.hsv(U.BRAND[s])
        if BB.neutral(U.BRAND[s]): return 0.6
        hd = h * 360
        return 3.2 if 18 <= hd <= 65 else (1.6 if v > 0.75 else 1.0)
    pool = sorted(U.SYMS, key=lambda s: -r.random() ** (1.0 / wgt(s))); used = set()
    def take(cond=lambda s: True, n=1):
        out = []
        for s in pool:
            if s not in used and cond(s): out.append(s); used.add(s)
            if len(out) == n: break
        if len(out) < n:                                          # pool exhausted: allow repeats
            extra = [s for s in pool if cond(s) and s not in out]; r.shuffle(extra); out += extra[:n - len(out)]
        return out
    fixed = (['clock', 'score'] if clock else ['score']) if band == 'near' else []
    kinds = SINGLE * 2 + GROUP_SAME + MARKET + (['tape'] if band == 'near' else []); deck = []
    seq = []; total = 0.0; prev_kind = None; prev_fam = None; vi = r.randrange(6); since_warm = 2
    must = fixed[:]; r.shuffle(must)
    while total < length_h:
        if must and (len(seq) % 4 == 2): kind = must.pop()
        else:
            for _try in range(40):
                if not deck: deck = kinds[:]; r.shuffle(deck)
                kind = deck.pop()
                if kind != prev_kind and not (kind in MARKET + ['tape'] and prev_fam == 'm'): break
                deck.insert(0, kind)
        sp = {'type': kind}
        if kind in SINGLE:
            want_warm = since_warm >= 1
            s = take(lambda s: FAM[s] != prev_fam and (WARM(s) or not want_warm))[0]; sp['sym'] = s
            since_warm = 0 if WARM(s) else since_warm + 1
            wts = [6, 0.8, 2, 2, 0.4, 0.3] if WARM(s) else [6, 1.2, 2.5, 1.5, 1.2, 0.6]
            sp['var'] = r.choices(['solid', 'light', 'shift', 'grad', 'deep', 'dark'], weights=wts)[0]; fam = FAM[s]
        elif kind in GROUP_SAME:                               # several assets, each in its own colour family (no two alike in a row)
            n = 3 if kind == 'trio' else 5; got = []; fams = {prev_fam}
            sp['vars'] = tuple(r.choice(['solid', 'solid', 'solid', 'shift']) for _ in range(3))
            for _ in range(n):
                last = FAM[got[-1]] if got else prev_fam
                c = take(lambda s, last=last, fams=frozenset(fams): FAM[s] != 'n' and FAM[s] != last and FAM[s] not in fams)
                got += c; fams.add(FAM[c[0]])
            sp['syms'] = got; fam = 'm'
        elif kind in ('heat', 'tree'):
            sp['syms'] = take(n=8 if kind == 'heat' else 10); sp['cols'] = 4; fam = 'm'
        elif kind == 'bars': sp['syms'] = take(n=5); fam = 'm'
        elif kind == 'led': sp['syms'] = take(n=9); fam = 'm'
        elif kind == 'tape': sp['syms'] = take(n=14); fam = 'm'
        else: fam = 'm'                                        # map, clock, score
        seq.append([BB.WIDTH[kind], sp]); total += BB.WIDTH[kind]; prev_kind = kind; prev_fam = fam
    while must:                                                 # make sure fixed tiles are present
        sp = {'type': must.pop()}; seq.insert(len(seq) // 2, [BB.WIDTH[sp['type']], sp]); total += BB.WIDTH[sp['type']]
    k = length_h / total
    for it in seq: it[0] *= k
    return seq

def render(seq, h, t, gap=0.03):
    W = int(round(sum(w for w, _ in seq) * h)); im = Image.new('RGB', (W, h), (6, 8, 14)); x = 0.0; g = int(gap * h)
    for w, sp in seq:
        x0 = int(round(x)); x1 = int(round(x + w * h)); tile = BB.R[sp['type']](x1 - x0 - g, h, t, sp); im.paste(tile, (x0 + g // 2, 0)); x += w * h
    return im

# ---------------- neighbour colour check: render each tile small, compare the colours that actually touch
def _lab(img):
    a = cv2.cvtColor(np.asarray(img, np.uint8)[None] if np.asarray(img).ndim == 1 else np.asarray(img, np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return a.mean(0)
def _hue(im):
    """dominant hue (deg) of the coloured pixels, or None if the area is mostly neutral/dark"""
    hsv = cv2.cvtColor(np.ascontiguousarray(np.asarray(im, np.uint8)), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    m = (hsv[:, 1] > 70) & (hsv[:, 2] > 45)
    if m.mean() < 0.25: return None
    hist = np.bincount((hsv[m, 0] / 5).astype(int) % 36, minlength=36).astype(float)
    hist = hist + np.roll(hist, 1) + np.roll(hist, -1)
    return float(np.argmax(hist) * 10)
def _bgc(im):
    """dominant (most frequent) colour of an area = its background"""
    q = (np.asarray(im).reshape(-1, 3) // 24).astype(np.int32); key = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
    v, c = np.unique(key, return_counts=True); k = v[np.argmax(c)]; m = key == k
    return np.asarray(im).reshape(-1, 3)[m].mean(0)
def _sig(sp, w, h=72, t=37.0):
    im = np.asarray(BB.R[sp['type']](max(8, int(w * h)), h, t, sp).convert('RGB'))
    q = max(2, im.shape[1] // 3)
    b = [_bgc(im), _bgc(im[:, :q]), _bgc(im[:, -q:])]
    labs = [cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32) for x in b]
    hues = [_hue(np.uint8([[x]] * 4)) for x in b]
    return dict(lab=labs[0], L=labs[1], R=labs[2], hue=hues[0], hL=hues[1], hR=hues[2])
def _close(a, b): return float(np.linalg.norm(a - b))
def _hd(a, b): d = abs(a - b) % 360; return min(d, 360 - d)
def _bad(sa, sb, thr=36.0):
    """neighbours too alike: both dark, same hue family (any lightness), or near-identical colour"""
    if sa['R'][0] < 55 and sb['L'][0] < 55: return True
    if _close(sa['R'], sb['L']) < 42: return True
    for ha, hb in ((sa['hue'], sb['hue']), (sa['hR'], sb['hL'])):
        if ha is not None and hb is not None and _hd(ha, hb) < 28: return True
    return _close(sa['lab'], sb['lab']) < thr or _close(sa['R'], sb['L']) < thr * 0.9

def fix_colours(seq, seed=0, passes=20):
    r = random.Random(seed); n = len(seq)
    sig = [_sig(sp, w) for w, sp in seq]
    def bad_at(i): return _bad(sig[i], sig[(i + 1) % n])
    for _ in range(passes):
        bads = [i for i in range(n) if bad_at(i)]
        if not bads: break
        for i in bads:
            j = (i + 1) % n
            if not bad_at(i): continue
            w, sp = seq[j]
            done = False
            if sp['type'] in SINGLE:                                  # try other colour variants of the same tile
                for v in ['solid', 'light', 'shift', 'grad', 'deep', 'dark']:
                    if v == sp.get('var'): continue
                    sp2 = dict(sp, var=v); s2 = _sig(sp2, w)
                    if not _bad(sig[i], s2) and not _bad(s2, sig[(j + 1) % n]):
                        seq[j][1] = sp2; sig[j] = s2; done = True; break
            if not done and sp['type'] in SINGLE + ['bars', 'book', 'depth', 'candle', 'trend', 'glyph', 'card']:
                cand = [s for s in U.SYMS if WARM(s)]; r.shuffle(cand); cand += r.sample(U.SYMS, 60)
                for s_new in cand[:120]:                       # try another asset with a different colour (warm first)
                    sp2 = dict(sp); sp2['sym'] = s_new
                    if sp['type'] == 'bars': sp2['syms'] = [s_new] + list(sp['syms'][1:])
                    s2 = _sig(sp2, w)
                    if not _bad(sig[i], s2) and not _bad(s2, sig[(j + 1) % n]):
                        seq[j][1] = sp2; sig[j] = s2; done = True; break
            if done: continue
            swapped = False
            for k in r.sample(range(n), n):                           # swap with another tile that fits both sides
                if k in (i, j): continue
                si, sk = sig[j], sig[k]
                ok_k_here = not _bad(sig[i], sk) and not _bad(sk, sig[(j + 1) % n])
                ok_j_there = not _bad(sig[(k - 1) % n], si) and not _bad(si, sig[(k + 1) % n])
                if ok_k_here and ok_j_there:
                    seq[j], seq[k] = seq[k], seq[j]; sig[j], sig[k] = sig[k], sig[j]; swapped = True; break
            if not swapped and seq[j][1]['type'] not in ('clock', 'score'):     # last resort: a bright single card that fits
                cand = [s for s in U.SYMS if WARM(s)]; r.shuffle(cand); cand += r.sample(U.SYMS, 80)
                for s_new in cand[:140]:
                    for v in ('solid', 'light', 'shift', 'grad'):
                        sp2 = {'type': 'card', 'sym': s_new, 'var': v}; s2 = _sig(sp2, w)
                        if not _bad(sig[i], s2) and not _bad(s2, sig[(j + 1) % n]):
                            seq[j][1] = sp2; sig[j] = s2; swapped = True; break
                    if swapped: break
    return seq, [i for i in range(n) if bad_at(i)]
