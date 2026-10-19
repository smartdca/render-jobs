# 3D ring geometry -> per-pixel maps for near band, rim (ticker) and inner far band, with real occlusion.
import numpy as np, json, cv2
import os as _os
H, W = int(_os.environ.get('RING_H', 1440)), int(_os.environ.get('RING_W', 1920))
import os
p = np.array(json.load(open(os.environ.get('RING_MODEL', '/home/claude/sp/ring/model7.json')))['p'])
f, pitch, roll, cx, cz, K, YT, YB, Y2T, Y2B = p[:10]
B, PSI = (p[10], p[11]) if len(p) > 10 else (1.0, 0.0)
_c, _s = np.cos(PSI), np.sin(PSI)
def to_local(X, Z):            # world xz (relative to ring centre) -> unit-circle space
    lx = _c * X + _s * Z; lz = -_s * X + _c * Z; return lx, lz / B
def from_local(lx, lz):
    lz = lz * B; return cx + _c * lx - _s * lz, cz + _s * lx + _c * lz
# arc length along the ellipse (radius 1) as a function of circle-space angle
_T = np.linspace(-np.pi, np.pi, 20001)
_ds = np.hypot(np.sin(_T), B * np.cos(_T)); _S = np.concatenate([[0], np.cumsum((_ds[1:] + _ds[:-1]) / 2 * np.diff(_T))])
_S -= np.interp(0, _T, _S)
PERIM = _S[-1] - _S[0]
def arclen(th): return np.interp(th, _T, _S)
YLB = None  # rim (lip) bottom height, calibrated below

def rays(ss=1):
    u, v = np.meshgrid((np.arange(W * ss) + 0.5) / ss, (np.arange(H * ss) + 0.5) / ss)
    x2 = (u - W / 2) / f; y2 = -(v - H / 2) / f; z1 = np.ones_like(x2)
    cr, sr = np.cos(roll), np.sin(roll); X = x2 * cr + y2 * sr; y1 = -x2 * sr + y2 * cr
    cp, sp = np.cos(pitch), np.sin(pitch); Y = y1 * cp + z1 * sp; Z = -y1 * sp + z1 * cp
    return X, Y, Z

def roots(X, Z, r):
    ox, oz = to_local(-cx, -cz); dx, dz = to_local(X, Z)
    a = dx * dx + dz * dz; b = 2 * (ox * dx + oz * dz); c = ox * ox + oz * oz - r * r; D = b * b - 4 * a * c
    ok = D > 0; sq = np.sqrt(np.maximum(D, 0))
    return ok, (-b - sq) / (2 * a), (-b + sq) / (2 * a)
def angle(X, Z, t):
    lx, lz = to_local(X * t - cx, Z * t - cz); return np.arctan2(lz, lx)

def proj(X, Y, Z):
    cp, sp = np.cos(pitch), np.sin(pitch); y1 = Y * cp - Z * sp; z1 = Y * sp + Z * cp
    cr, sr = np.cos(roll), np.sin(roll); x2 = X * cr - y1 * sr; y2 = X * sr + y1 * cr
    return W / 2 + f * x2 / z1, H / 2 - f * y2 / z1

def calibrate_lip(img_y_center=None):
    if img_y_center is None: img_y_center = float(os.environ.get('LIP_Y', 662))
    """find rim-bottom height so the rim's lower edge at the ring's front-centre lands where the photo shows it"""
    lx, lz = to_local(-cx, -cz); th = np.arctan2(lz, lx)   # ring point facing the camera
    X, Z = from_local(np.cos(th), np.sin(th))
    lo, hi = YB - 1.0, YB
    for _ in range(60):
        m = (lo + hi) / 2; _, y = proj(X, m, Z)
        if y > img_y_center: lo = m
        else: hi = m
    return (lo + hi) / 2

def build(ss=1, hs_near=600, hs_lip=80, hs_far=600):
    global YLB
    YLB = float(p[12]) if len(p) > 12 else calibrate_lip()
    X, Y, Z = rays(ss)
    ok, t1, _ = roots(X, Z, 1.0)
    Y1 = Y * t1; th1 = angle(X, Z, t1)
    near = ok & (t1 > 0) & (Y1 <= YT) & (Y1 >= YB)
    lip = ok & (t1 > 0) & (Y1 < YB) & (Y1 >= YLB)
    blocked = ok & (t1 > 0) & (Y1 <= YT) & (Y1 >= YLB)          # anything on the outer face hides what is behind it
    # inner far face: ray must pass up through the hole (inside inner radius at the underside level)
    okk, _, t2 = roots(X, Z, K)
    Y2 = Y * t2; th2 = angle(X, Z, t2)
    YIF = float(os.environ.get('YIF', Y2B)); tb = YIF / np.where(np.abs(Y) < 1e-9, 1e-9, Y)
    _lx, _lz = to_local(X * tb - cx, Z * tb - cz); inhole = np.hypot(_lx, _lz) < K
    far = okk & (t2 > 0) & (Y2 <= Y2T) & (Y2 >= Y2B) & inhole & ~blocked
    # strip coordinates (pixels). arc length scaled so content keeps its aspect ratio on the surface
    kn = hs_near / (YT - YB); kl = hs_lip / (YB - YLB); kf = hs_far / (Y2T - Y2B)
    a1 = arclen(th1); a2 = arclen(th2) * K
    sN = a1 * kn; vN = (YT - Y1) / (YT - YB)
    sL = a1 * kl; vL = (YB - Y1) / (YB - YLB)
    sF = a2 * kf; vF = (Y2T - Y2) / (Y2T - Y2B)
    # make strips read left->right on screen
    xs = X  # monotone with screen x for fixed row
    def orient(s, m):
        rows, cols = np.nonzero(m)
        if len(rows) < 10: return 1
        i = np.argsort(cols); c = cols[i]; sv = s[rows[i], c]
        mid = len(c) // 2; return 1 if np.median(np.diff(np.unwrap(sv[mid - 200:mid + 200] * 0 + sv[mid - 200:mid + 200]))) >= 0 else -1
    sgnN = 1 if np.nanmean(np.diff(sN[int(H * ss * 0.25), :][near[int(H * ss * 0.25), :]])) > 0 else -1
    rowf = int(np.argmax(far.sum(1))); cols = np.nonzero(far[rowf])[0]; c0 = cols[len(cols) // 2]
    sgnF = 1 if sF[rowf, c0 + 10] > sF[rowf, c0 - 10] else -1
    rown = int(np.argmax(near.sum(1))); cols = np.nonzero(near[rown])[0]; c1 = cols[len(cols) // 2]
    sgnN = 1 if sN[rown, c1 + 10] > sN[rown, c1 - 10] else -1
    return dict(sN=sN * sgnN, vN=vN, mN=near, sL=sL * sgnN, vL=vL, mL=lip, sF=sF * sgnF, vF=vF, mF=far,
                CN=PERIM * kn, CL=PERIM * kl, CF=PERIM * K * kf, sgnN=sgnN, sgnF=sgnF, YLB=YLB)

if __name__ == '__main__':
    g = build()
    print({k: (float(v) if np.isscalar(v) or np.ndim(v) == 0 else v.shape) for k, v in g.items()})
    src = cv2.imread(os.environ.get('RING_SRC', '/mnt/user-data/uploads/Intuit-Dome-Los-Angeles-Clippers.jpg'))
    ov = src.copy()
    for m, c in [(g['mN'], (0, 200, 255)), (g['mL'], (0, 255, 0)), (g['mF'], (255, 0, 255))]:
        ov[m] = (ov[m] * 0.45 + np.array(c) * 0.55).astype(np.uint8)
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cnt, -1, (255, 255, 255), 1)
    cv2.imwrite('/home/claude/sp/ring/geo_ov.jpg', ov[150:1230])
