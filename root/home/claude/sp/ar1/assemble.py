# ar1 final picture: loops (+ clock patch) or warm-up clips, section switch under the card blur, cards, fades.
# usage: python3 assemble.py <first_frame> <last_frame_exclusive> <out.mp4>    (needs loops/L{s}.mp4 and clips/w{n}.mp4)
#        python3 assemble.py still <seconds> <out.jpg>
import os, sys, json, math, subprocess
import numpy as np, cv2
import ar
D = '/home/claude/sp/ar1/' if not os.environ.get('AR_DIR') else os.environ['AR_DIR'].rstrip('/') + '/'
P = ar.P; W = json.load(open(D + 'warmups.json')); FPS = 30; OW, OH = 1920, 1080
CT = P['cards']; TE = P['end_card']; T = P['total']; SECS = P['sections']
SL = ['score', 'backtest', 'trending', 'insights', 'learn']
XA, XB = 5.0, 15.0
def ss(x, a, b):
    x = min(max((x - a) / (b - a), 0.0), 1.0); return x * x * (3 - 2 * x)
def sections_at(t):
    """[(section, weight)]: the ring switches content under the card blur"""
    for k, c in enumerate(CT):
        if c + XA <= t < c + XB: w = ss(t, c + XA, c + XB); return [(k, 1 - w), (k + 1, w)]
    return [(sum(1 for c in CT if t >= c + XB), 1.0)]
def load(name):
    im = cv2.imread(D + f'cards/card_{name}.png', cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    return im[..., :3], im[..., 3:], [int(v) for v in json.load(open(D + f'cards/box_{name}.json'))]
CARD = {n: load(n) for n in SL + ['brand']}
def card_state(t):
    for i, Tk in enumerate(CT):
        if Tk - 3 <= t <= Tk + 24:
            bl = ss(t, Tk - 2.5, Tk + 0.5) * (1 - ss(t, Tk + 20.5, Tk + 23.5))
            ca = ss(t, Tk, Tk + 1.2) * (1 - ss(t, Tk + 20.3, Tk + 21.6))
            rem = 1 - min(max((t - (Tk + 0.8)) / 20.0, 0), 1)
            return bl, ca, SL[i], rem
    if t >= TE - 3: return ss(t, TE - 2.5, TE + 0.5), ss(t, TE, TE + 1.2), 'brand', None
    return 0, 0, None, None
def clip_at(f):
    for n, w in enumerate(W):
        if int(round(w['t_a'] * FPS)) <= f < int(round(w['t_b'] * FPS)): return n, w
    return None, None

class Reader:
    """sequential reader of an mp4 that restarts (with an exact seek) whenever a non-consecutive frame is asked for"""
    def __init__(self, path, n=None): self.path = path; self.n = n; self.p = None; self.cur = None
    def get(self, i):
        if self.n: i %= self.n
        if self.p is None or i != self.cur:
            if self.p: self.p.kill()
            self.p = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{max(0, (i - 0.3) / FPS):.4f}', '-i', self.path, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                                      stdout=subprocess.PIPE, bufsize=OW * OH * 3 * 4)
            self.cur = i
        b = self.p.stdout.read(OW * OH * 3)
        if len(b) < OW * OH * 3:                       # loop end reached: wrap
            self.p.kill(); self.p = None; return self.get(0 if self.n else i)
        self.cur = (i + 1) % self.n if self.n else i + 1
        return np.frombuffer(b, np.uint8).reshape(OH, OW, 3).copy()
LOOPS = {}; CLIPS = {}
GRAIN = True
DIRECT = bool(os.environ.get('DIRECT'))           # samples: render loop/clip frames on the fly instead of reading videos
def loop_reader(s):
    if s not in LOOPS: LOOPS[s] = Reader(D + f'loops/L{s}.mp4', ar.NL)
    return LOOPS[s]

def frame(f, use_patch=True):
    t = f / FPS; bl, ca, name, rem = card_state(t)
    n, w = clip_at(f)
    if w is not None:
        if DIRECT: fr = ar.clip_frame(W, w, f)
        else:
            if n not in CLIPS: CLIPS[n] = Reader(D + f'clips/w{n}.mp4')
            fr = CLIPS[n].get(f - int(round(w['t_a'] * FPS)))
    else:
        tau = ar.tau_of(t, W); i = int(round(tau * FPS)) % ar.NL; acc = None
        for s, wt in sections_at(t):
            im = ar.loop_frame(s, i) if DIRECT else loop_reader(s).get(i)
            if s > 0 and use_patch and bl < 0.97: im = ar.apply_patch(im, ar.clock_patch(s, tau, ar.game_text(t), f))
            acc = im.astype(np.float32) * wt if acc is None else acc + im.astype(np.float32) * wt
        fr = acc
    fr = fr.astype(np.float32) / 255.0
    if bl > 0:
        b = cv2.resize(cv2.GaussianBlur(cv2.resize(fr, (240, 135), interpolation=cv2.INTER_AREA), (0, 0), 2.2), (OW, OH)) * 0.45
        fr = fr * (1 - bl) + b * bl
    if ca > 0:
        rgb, a, (bx, by, bw, bh) = CARD[name]; a = a * ca; fr = fr * (1 - a) + rgb * a
        if rem is not None:
            yb = by + bh - 30; x0 = bx + 76; x1 = bx + bw - 76
            trk = np.float32([70, 64, 58]) / 255; org = np.float32([58, 129, 200]) / 255
            fr[yb - 1:yb + 2, x0:x1] = fr[yb - 1:yb + 2, x0:x1] * (1 - ca) + trk * ca
            xe = x0 + int((x1 - x0) * rem)
            if xe > x0: fr[yb - 1:yb + 2, x0:xe] = fr[yb - 1:yb + 2, x0:xe] * (1 - ca) + org * ca
    k = min(ss(t, 0, 2.0), 1 - ss(t, T - 1.6, T))
    fr = fr * 255.0
    if GRAIN:                                         # temporal film grain at output size (same character as the 2x render's)
        lum = fr.mean(-1, keepdims=True) / 255
        g = cv2.resize(np.random.default_rng(f + 7).normal(0, 1, (OH // 2, OW // 2)).astype(np.float32), (OW, OH), interpolation=cv2.INTER_LINEAR)
        fr = fr + g[..., None] * (3.2 + 2.0 * (1 - lum)) * 0.85
    return np.clip(fr * k + 0.5, 0, 255).astype(np.uint8)

def main(f0, f1, out):
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-preset', 'medium', '-crf', '21', '-g', '60', '-keyint_min', '60', '-sc_threshold', '0',
                            '-pix_fmt', 'yuv420p', '-profile:v', 'high', out], stdin=subprocess.PIPE)
    for f in range(f0, f1): enc.stdin.write(frame(f).tobytes())
    enc.stdin.close(); enc.wait(); print('done', out, flush=True)

if __name__ == '__main__':
    if sys.argv[1] == 'still': cv2.imwrite(sys.argv[3], frame(int(round(float(sys.argv[2]) * FPS))))
    else: main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
