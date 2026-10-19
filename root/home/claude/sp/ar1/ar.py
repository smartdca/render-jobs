# ar1 full-video renderer core: 120 s loops per section, warm-up clips, clock ROI patches.
# Ring time tau: every ring-borne animation (rotation, tiles, rim ticker, fascia ticker, loop flashes) is a function of tau,
# periodic in 120 s. tau runs 1:1 with video time except during warm-ups, where the ring decelerates, holds and accelerates.
import os, sys, json, math, random
sys.path.insert(0, '/home/claude/sp/ring2')
import numpy as np, cv2
import arena as A, strips as ST, billboard as BB
A.GRAIN = False                                   # film grain goes on in the final pass (keeps the intermediate videos small)

D = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(D, 'plan_ar1.json')))
FPS = 30; LOOP = 120.0; NL = int(LOOP * FPS)
HS, HL, HF = A.HS, A.HL, A.HF
CN, CF, CL, REV = A.CN, A.CF, A.CL, A.REV
aN, bN = A.aN, A.bN; VIS = bN - aN
OW, OH = 1920, 1080

# anti-alias fringe pixels just outside each band sampled extrapolated strip coordinates (thin lines of unrelated content
# along the band edges); give them the coordinates of the nearest pixel inside the band instead
def _nearest_fill(valid, *maps):
    d, lab = cv2.distanceTransformWithLabels((~valid).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(valid); idx = np.zeros(lab.max() + 1, np.int64)
    zl = lab[valid]; idx[zl] = np.arange(len(zl))
    sel = idx[lab]; out = []
    for m in maps:
        v = m[ys, xs][sel]; out.append(np.where(valid, m, v).astype(m.dtype))
    return out
A.G['sN'], A.G['vN'] = _nearest_fill(A.G['mN'], A.G['sN'], A.G['vN'])
A.G['sL'], A.G['vL'] = _nearest_fill(A.G['mL'], A.G['sL'], A.G['vL'])
A.G2['sF'], A.G2['vF'] = _nearest_fill(A.G2['mF'] | A.G['mF'] | (A.XA[..., 0] > 0), A.G2['sF'], A.G2['vF'])   # far band is drawn with G's mask
LN = int(round(CN))

# ---------------- tickers on the rim and the audience fascia, speeds quantised so they repeat every 120 s
TICK = A.TICK; LR = TICK.shape[1]; VR = LR / LOOP                       # rim: one full lap per loop (~150 px/s)
def _fas_tex():
    best = None
    for n in range(20, 71):
        t = A.ticker_tex(HF, (4, 5, 10), gapw=5, chan='fas', fs=0.70, boost=True, syms=A.SYMS_F[:n])
        if best is None or abs(t.shape[1] - 55 * A.S2 * LOOP) < abs(best.shape[1] - 55 * A.S2 * LOOP): best = t
    return best
FTX = _fas_tex(); LF = FTX.shape[1]; VF = LF / LOOP

# ---------------- content strips per section
_SEQ = {}
def seqs(s):
    if s not in _SEQ:
        n = ST.build(s, 'near', CN / HS, clock=s > 0); n, _ = ST.fix_colours(n, s)
        f = ST.build(s, 'far', CF / HS); f, _ = ST.fix_colours(f, s + 50)
        if s == len(P['sections']) - 1: n = _clock_to_front(n)
        _SEQ[s] = (n, f)
    return _SEQ[s]
def _clock_to_front(seq):
    """last section: turn the tile order (whole tiles, neighbours unchanged) so the game clock faces the camera at 00:00"""
    W = json.load(open(os.path.join(D, 'warmups.json')))
    phi = (tau_of(P['clock_zero'], W) % LOOP) / REV; want = (phi * CN + (aN + bN) / 2) % LN
    best = None
    for r in range(len(seq)):
        sq = seq[r:] + seq[:r]; x = 0.0
        for w, sp in sq:
            if sp['type'] == 'clock': c = x + w * HS / 2
            x += w * HS
        d = abs((c - want + LN / 2) % LN - LN / 2)
        if best is None or d < best[0]: best = (d, sq)
    return best[1]
def tile_spans(seq, h=HS, gap=0.03):
    """[(x0, x1, spec)] pixel spans of each tile slot (incl. gap) in the rendered strip"""
    out = []; x = 0.0
    for w, sp in seq: out.append((int(round(x)), int(round(x + w * h)), sp)); x += w * h
    return out
def render_strip(seq, t, L, lo=None, hi=None, gap=0.03, h=HS):
    """strip image (h x L, BGR float32); with lo/hi only tiles touching [lo, hi] (mod L) are drawn"""
    im = np.full((h, L, 3), (14, 8, 6), np.float32); g = int(gap * h)
    for x0, x1, sp in tile_spans(seq, h, gap):
        if lo is not None:
            ok = False
            for sh in (-L, 0, L):
                if x1 + sh >= lo and x0 + sh <= hi: ok = True
            if not ok: continue
        tw = x1 - x0 - g
        if tw <= 0 or x0 + g // 2 >= L: continue
        tile = np.asarray(BB.R[sp['type']](tw, h, t, sp).convert('RGB'))[:, :, ::-1]
        a = x0 + g // 2; b = min(L, a + tw); im[:, a:b] = tile[:, :b - a]
    return im

def near_window(phi, margin=HS * 1.2):
    c = phi * CN; return c + aN - margin, c + bN + margin

# ---------------- flashes in the crowd, periodic for the loops
_LEV = A.make_flashes(LOOP, lambda t: False, seed=29)
LOOP_FLASH = _LEV + [(e[0] - LOOP,) + tuple(e[1:]) for e in _LEV if e[0] > LOOP - 1] + [(e[0] + LOOP,) + tuple(e[1:]) for e in _LEV if e[0] < 1]

def down(im):
    return cv2.resize(im, (OW, OH), interpolation=cv2.INTER_AREA)

# ---------------- loop frame
def loop_frame(s, i):
    """section s, loop frame i (0..NL-1) -> 1920x1080 BGR uint8; clock tile drawn with all segments unlit"""
    tau = i / FPS; phi = tau / REV; seqN, seqF = seqs(s)
    BB.CLOCK_TEXT = ''
    lo, hi = near_window(phi); lo, hi = lo % LN, lo % LN + (hi - lo)
    near = render_strip(seqN, tau, LN, lo, hi)
    far = A.fit_len(render_strip(seqF, tau, int(round(sum(w for w, _ in seqF) * HS))), CF).astype(np.float32)
    im = A.compose(near, far, TICK, phi=phi, t=tau, flashes=LOOP_FLASH, rim_off=VR * tau, fas=FTX, fas_off=VF * tau, seed=s * 10000 + i)
    return down(im)

# ================= warm-ups ("hype"): schedule
DEC = ACC = 2.0
FLASH_LEN = {'invert': 4.0, 'strobe': 3.2, 'pulse': 4.8}
NEG = {'DIP', 'LOW', 'FEAR', 'BEAR', 'KILLS', 'CRASH'}
def _hold(pat): return 0.7 + FLASH_LEN[pat] + 2.6                     # stop -> resume
def schedule():
    """two passes: the tip-off warm-up must flash exactly on the tip-off horn; the difference is absorbed by the holds
    of the pre-game warm-ups"""
    W1 = _schedule({}, None)
    tw = next(w for w in W1 if w['sec'] == 1 and w['idx'] == 0)
    d = P['tipoff'] - tw['t_fl']; prev = [w for w in W1 if w['sec'] == 0]
    extra = {(w['sec'], w['idx']): round(d / len(prev) * FPS) / FPS for w in prev}
    W2 = _schedule(extra, tw['run'])
    tw = next(w for w in W2 if w['sec'] == 1 and w['idx'] == 0)
    assert abs(tw['t_fl'] - P['tipoff']) < 0.05, (tw['t_fl'], P['tipoff'])
    return W2
def _schedule(extra, tip_run):
    rnd = random.Random(2026); out = []; lag = 0.0; last_pal = None; pats = []
    for s, sec in enumerate(P['sections']):
        phrases = sec['phrases'][:]; rnd.shuffle(phrases)
        if s == 1: phrases.remove('RIDE THE BULL'); phrases.insert(0, 'RIDE THE BULL')   # tip-off
        a, b = sec['start'], sec['music_end']
        targets = [P['tipoff'], a + 0.52 * (b - a), a + 0.80 * (b - a)] if s == 1 else [a + f * (b - a) for f in (0.25, 0.52, 0.79)]
        spans = tile_spans(seqs(s)[0]); n = len(spans)
        for k, (ph, tgt) in enumerate(zip(phrases, targets)):
            if not pats: pats = ['invert', 'strobe', 'pulse']; rnd.shuffle(pats)
            pat = pats.pop()
            neg = bool(set(ph.replace("'", ' ').split()) & NEG)
            pal = 'red' if neg else rnd.choice([p for p in A.PALETTES if p not in ('red', last_pal)])
            last_pal = pal
            hold = _hold(pat) + extra.get((s, k), 0.0); t_stop_want = tgt - 0.7                    # flash starts at the target (tip-off: on the horn)
            tau_want = t_stop_want - DEC / 2 - lag
            best = None
            for i in range(n):                                            # candidate runs of whole tiles, no clock inside
                j = i; span = 0
                while span < VIS + 0.9 * HS and j < i + n:
                    x0, x1, sp = spans[j % n]; span += x1 - x0; j += 1
                if span > VIS + 2.6 * HS: continue
                if any(spans[q % n][2]['type'] == 'clock' for q in range(i, j)): continue
                if tip_run and s == 1 and k == 0 and [i, j] != tip_run: continue
                p = spans[i][0]; c = (p + span / 2 - (aN + bN) / 2) / CN * REV % LOOP
                tau_s = c + LOOP * round((tau_want - c) / LOOP)
                if best is None or abs(tau_s - tau_want) < abs(best[0] - tau_want): best = (tau_s, i, j, p, span)
            tau_s, i, j, p, span = best
            # the stop must land on a whole frame of ring time
            tau_s = round(tau_s * FPS) / FPS
            t_stop = tau_s + DEC / 2 + lag
            tau_in = tau_s - (span / 2 + VIS / 2) / CN * REV; tau_out = tau_s + (span / 2 + VIS / 2) / CN * REV
            w = dict(sec=s, idx=k, phrase=ph, pat=pat, pal=pal, run=[i, j], p=p, ws=span, tau_stop=tau_s, lag_before=round(lag, 4),
                     t_dec=t_stop - DEC, t_stop=t_stop, t_fl=t_stop + 0.7, t_fe=t_stop + 0.7 + FLASH_LEN[pat], t_res=t_stop + hold, hold=hold,
                     t_a=math.floor((tau_in - 1.0 + lag) * FPS) / FPS)
            lag += DEC / 2 + hold + ACC / 2
            w['t_b'] = math.ceil((tau_out + 1.0 + lag) * FPS) / FPS; w['lag_after'] = round(lag, 4)
            out.append(w)
    return out

def tau_of(t, W):
    """ring time at video time t"""
    lag = 0.0
    for w in W:
        if t < w['t_dec']: break
        if t < w['t_stop']: x = (t - w['t_dec']) / DEC; return w['t_dec'] - lag + DEC * (x - x * x / 2)
        ps = w['t_dec'] - lag + DEC / 2
        if t < w['t_res']: return ps
        if t < w['t_res'] + ACC: x = (t - w['t_res']) / ACC; return ps + ACC * x * x / 2
        lag = w['lag_after']
    return t - lag

# ================= game clock
def game_running(t):
    """seconds of game time elapsed at video time t (clock stops during cards)"""
    tip = P['tipoff']
    if t <= tip: return 0.0
    run = t - tip
    for c, sec in zip(P['cards'][1:], P['sections'][2:]):
        if t > c: run -= min(t, sec['start']) - c
    return min(2880.0, run)
def game_text(t):
    rem = 2880.0 - game_running(t)
    if rem >= 60: return f'{int(rem // 60):02d}:{int(rem % 60):02d}'
    return f'{int(rem):02d}.{int(rem * 10) % 10}'

# ================= warm-up clip frames
def look(w, t):
    """inv (bool), k (brightness 0..1), mode"""
    if t < w['t_stop']: return False, 1.0, 'pre'
    if t < w['t_fl']: return False, 1.0, 'stop'
    if t < w['t_fe']:
        u = t - w['t_fl']; pat = w['pat']
        if pat == 'invert':
            seq = [(0, .22), (1, .42), (0, .22), (2, .42), (0, .22), (1, .42), (0, .22), (2, .42), (0, .22), (1, .42)]; acc = 0
            for st, d in seq:
                if u < acc + d: break
                acc += d
            return st == 1, (0.0 if st == 0 else 1.0), 'flash'
        if pat == 'strobe': return int(u / 0.48) % 2 == 1, (1.0 if int(u / 0.12) % 2 == 0 else 0.05), 'flash'
        k = 0.5 - 0.5 * math.cos(2 * math.pi * u / 1.6); return int(u / 1.6) % 2 == 1, 0.15 + 0.85 * k, 'flash'
    return True, 1.0, 'hold' if t < w['t_res'] else 'post'

_WC = {}
def _wc(w):
    key = (w['sec'], w['idx'])
    if key not in _WC:
        ws = int(round(w['ws']))
        mN = A.loop_mask(w['phrase'], ws, HS, g=0.8); mF = A.loop_mask(w['phrase'], CF, HS, g=0.8)
        mL = A.loop_mask(w['phrase'], CL, HL, g=2.2); mA = A.loop_mask(w['phrase'], LF, HF, g=2.6)
        # crowd flashes: the loop's (by ring time) while the ring runs, a busy random set around the hold
        ev = []
        for e in LOOP_FLASH:
            for n in range(-2, 40):
                ta = e[0] + n * LOOP + w['lag_before']; tb = e[0] + n * LOOP + w['lag_after']
                if w['t_a'] - 1 <= ta < w['t_dec']: ev.append((ta,) + tuple(e[1:]))
                if w['t_res'] + ACC <= tb < w['t_b'] + 1: ev.append((tb,) + tuple(e[1:]))
        T0 = w['t_dec']; busy = A.make_flashes(w['t_res'] + ACC - T0, lambda u: w['t_stop'] - 0.5 - T0 <= u <= w['t_res'] + 2 - T0, seed=100 + 7 * w['sec'] + w['idx'])
        ev += [(e[0] + T0,) + tuple(e[1:]) for e in busy]
        _WC[key] = dict(mN=mN, mF=mF, mL=mL, mA=mA, ev=ev, fg=np.array(A.PALETTES[w['pal']][1][::-1], np.float32))
    return _WC[key]

def clip_frame(W, w, f):
    """warm-up w, global frame index f -> 1920x1080 BGR uint8"""
    t = f / FPS; c = _wc(w); s = w['sec']; seqN, seqF = seqs(s)
    tau = tau_of(t, W) % LOOP; phi = tau / REV                     # wrap: strips are whole pixels, ring lengths are not
    inv, k, mode = look(w, t)
    BB.CLOCK_TEXT = game_text(t)
    lo, hi = near_window(phi); lo, hi = lo % LN, lo % LN + (hi - lo)
    near = render_strip(seqN, tau, LN, lo, hi)
    sec = A.colour(c['mN'], w['pal'], inv); p = int(round(w['p'])); ws = sec.shape[1]
    idx = (np.arange(ws) + p) % LN; near[:, idx] = sec
    far_dark = (w['t_stop'] <= t < w['t_fl']) or (w['t_res'] <= t < w['t_res'] + 0.3)
    if far_dark: far = None
    elif w['t_fl'] <= t < w['t_res']: far = np.roll(A.colour(c['mF'], w['pal'], inv), int(round(-phi * CF)) % int(round(CF)), axis=1)
    else: far = A.fit_len(render_strip(seqF, tau, int(round(sum(x for x, _ in seqF) * HS))), CF).astype(np.float32)
    if mode in ('pre', 'post'): rim, rk, ro = TICK, 1.0, VR * tau; fas, fo = FTX, VF * tau
    elif mode == 'stop': rim, rk, ro = None, 1.0, 0.0; fas, fo = np.full_like(FTX, 6.0), 0.0
    else:
        rim, rk, ro = A.colour(c['mL'], w['pal'], inv), k, 0.0
        fas, fo = A.colour(c['mA'], w['pal'], inv) * k + 6 * (1 - k), 0.0
    glow = amt = 0.0
    if t >= w['t_stop']:
        amt = min(1, (t - w['t_stop']) / 0.5) * (max(0, 1 - (t - w['t_res']) / 2.0) if t > w['t_res'] else 1)
        glow = (0.10 if inv else 0.0) * k * (max(0, 1 - (t - w['t_res']) / 6) if t > w['t_res'] else 1)
    FG = c['fg']
    im = A.compose(near, far, rim, phi=phi, t=t, ring_k=k if mode == 'flash' else 1.0, glow=glow,
                   fglow=(FG / FG.max())[None, None], beam_amt=amt, beam_t0=w['t_stop'], flashes=c['ev'],
                   rim_k=rk, rim_off=ro, beam_tint=np.array([200, 225, 255], np.float32) * 0.8 + FG * 0.2,
                   fas=fas, fas_off=fo, seed=f)
    return down(im)

# ================= clock patch: re-render only the clock tile's screen region with the real game-clock digits
_COL = {}
def _near_cols(phi):
    return np.mod(A.G['sN'] + phi * CN, LN)
_ROI_NAMES = ['BASE', 'AA_N', 'AA_F', 'AA_L', 'XA', 'FU', 'FV', 'FSOFT', 'LEDTEX', 'RECEIVE', 'REFL_W', 'MAPX', 'MAPY', 'VIGN']
GX, GY = 239, 56                                     # 2x-render blocks that map to whole output pixels (192 x 45)
def clock_patch(s, tau, text, seed):
    """-> (patch uint8, alpha float32, (x, y) in output pixels) or None when the clock is not on screen"""
    seqN, _ = seqs(s); tau = tau % LOOP; phi = tau / REV
    sp_ = [(x0, x1) for x0, x1, sp in tile_spans(seqN) if sp['type'] == 'clock']
    if not sp_: return None
    x0, x1 = sp_[0]; c0 = A.CROP[0]
    col = _near_cols(phi); band = A.AA_N[..., 0] > 0.002
    rel = np.mod(col - x0, LN); inside = band & (rel < (x1 - x0))
    inside[:c0] = False; inside[A.CROP[1]:] = False
    if not inside.any(): return None
    ys, xs = np.nonzero(inside)
    ya = max(c0, ys.min() - 72); yb = min(A.CROP[1], ys.max() + 73); xa = max(0, xs.min() - 72); xb = min(A.W, xs.max() + 73)
    ya = c0 + (ya - c0) // GY * GY; yb = min(A.CROP[1], c0 + -(-(yb - c0) // GY) * GY)
    xa = xa // GX * GX; xb = min(A.W, -(-xb // GX) * GX)
    ysl, xsl = slice(ya, yb), slice(xa, xb)
    saved = {k: getattr(A, k) for k in _ROI_NAMES + ['G', 'G2', 'ENDX', 'W', 'H', 'CROP', 'ROI_ORIGIN']}
    try:
        cs = lambda a: np.ascontiguousarray(a)
        for k in _ROI_NAMES: setattr(A, k, cs(saved[k][ysl, xsl]))
        A.MAPY = cs(np.clip(saved['MAPY'][ysl, xsl] - ya, 0, yb - ya - 1).astype(np.float32)); A.MAPX = cs((saved['MAPX'][ysl, xsl] - xa).astype(np.float32))
        A.G = {k: (cs(v[ysl, xsl]) if isinstance(v, np.ndarray) and v.shape[:2] == (saved['H'], saved['W']) else v) for k, v in saved['G'].items()}
        A.G2 = {k: (cs(v[ysl, xsl]) if isinstance(v, np.ndarray) and v.shape[:2] == (saved['H'], saved['W']) else v) for k, v in saved['G2'].items()}
        A.ENDX = []; A.W, A.H = xb - xa, yb - ya; A.CROP = (0, yb - ya); A.ROI_ORIGIN = (xa, ya, saved['W'], saved['H'])
        BB.CLOCK_TEXT = text
        lo = x0 - 2 * HS; hi = x1 + 2 * HS
        near = render_strip(seqN, tau, LN, lo, hi)
        im = A.compose(near, None, TICK, phi=phi, t=tau, rim_off=VR * tau, fas=FTX, fas_off=VF * tau, seed=seed)
    finally:
        for k, v in saved.items(): setattr(A, k, v)
    m = cv2.GaussianBlur(inside[ysl, xsl].astype(np.float32), (0, 0), 1.2)
    ow, oh = (xb - xa) * 192 // GX, (yb - ya) * 45 // GY
    pt = cv2.resize(im, (ow, oh), interpolation=cv2.INTER_AREA); al = cv2.resize(m, (ow, oh), interpolation=cv2.INTER_AREA)
    return pt, al, (xa * 192 // GX, (ya - c0) * 45 // GY)
def apply_patch(frame, patch):
    if patch is None: return frame
    pt, al, (x, y) = patch; h, w = al.shape
    sub = frame[y:y + h, x:x + w].astype(np.float32); a = al[..., None]
    frame[y:y + h, x:x + w] = np.clip(sub * (1 - a) + pt.astype(np.float32) * a + 0.5, 0, 255).astype(np.uint8)
    return frame
