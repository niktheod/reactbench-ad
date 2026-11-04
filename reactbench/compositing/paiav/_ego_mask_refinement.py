"""Frozen numerical method v2_outer_contour_04; no dataset or experiment paths.

Ported from the accepted 50-clip development and 30-clip validation experiments.
The constants below intentionally retain the tested search ranges and thresholds.
"""
import math
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks
from scipy.optimize import minimize, least_squares

X = np.arange(600, 1321)
OFF = np.arange(-32, 33)
def subdivide(v, f):
    for _ in range(2):
        tr = v[f]
        n = len(v)
        m = len(f)
        v = np.concatenate([v, (tr[:, 0] + tr[:, 1]) / 2, (tr[:, 1] + tr[:, 2]) / 2, (tr[:, 2] + tr[:, 0]) / 2])
        ab = np.arange(n, n + m)
        bc = ab + m
        ca = bc + m
        f = np.concatenate([np.c_[f[:, 0], ab, ca], np.c_[ab, f[:, 1], bc], np.c_[ca, bc, f[:, 2]], np.c_[ab, bc, ca]])
    edges = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
    return (v, edges)

def bivar(a, b, c):
    n = int((math.sqrt(8 * len(c) + 1) - 3) / 2)
    out = np.zeros_like(a)
    k = 0
    for j in range(n + 1):
        for i in range(n - j + 1):
            out += c[k] * a ** i * b ** j
            k += 1
    return out

def project(v, row, alignment):
    a = alignment
    s = a['scale']
    v = np.c_[s * (a['rear_y'] - v[:, 1]), s * (v[:, 0] - a['center_x']), s * (v[:, 2] - a['roof_z']) + a['height'] + a['dz']]
    pose = np.array(row['rig_from_camera'])
    p = (v - pose[:3, 3]) @ pose[:3, :3]
    z = p[:, 2].copy()
    c = row['raw_calibration']
    e = c.get('external_distortion_parameters')
    if e:
        if e['reference_poly'] != 'FORWARD':
            raise ValueError('Unsupported external distortion reference_poly; frozen method not changed')
        norm = np.linalg.norm(p, axis=1)
        aa = np.arcsin(p[:, 0] / norm)
        bb = np.arcsin(p[:, 1] / norm)
        xx = np.sin(bivar(aa, bb, e['horizontal_poly']))
        yy = np.sin(bivar(aa, bb, e['vertical_poly']))
        p = np.c_[xx, yy, np.sqrt(np.maximum(0, 1 - xx * xx - yy * yy))]
    rho = np.linalg.norm(p[:, :2], axis=1)
    theta = np.arctan2(rho, p[:, 2])
    dist = np.polynomial.polynomial.polyval(theta, c['angle_to_pixeldist_poly'])
    if c.get('reference_poly') == 'PIXELDIST_TO_ANGLE':
        coef = c['pixeldist_to_angle_poly']
        der = np.polynomial.polynomial.polyder(coef)
        for _ in range(5):
            dist -= (np.polynomial.polynomial.polyval(dist, coef) - theta) / np.polynomial.polynomial.polyval(dist, der)
    lin = c.get('linear_cde', [1.0, 0.0, 0.0])
    A = np.array([[lin[0], lin[1]], [lin[2], 1.0]])
    return (p[:, :2] * (dist / np.maximum(rho, 1e-09))[:, None] @ A.T + c['principal_point'], z)

def prior_boundary(v, edges, row):
    (uv, z) = project(v, row, row["alignment"])
    q = uv[edges]
    validz = (z[edges] > 0.02).all(1)
    prior = []
    for x in X:
        (a, b) = (q[:, 0, 0], q[:, 1, 0])
        ok = validz & (np.minimum(a, b) <= x) & (np.maximum(a, b) >= x) & (np.abs(b - a) > 1e-09)
        if not ok.any():
            raise ValueError(f'Projected hood lacks coverage at x={x}')
        tr = (x - a[ok]) / (b[ok] - a[ok])
        prior.append(np.min(q[ok, 0, 1] + tr * (q[ok, 1, 1] - q[ok, 0, 1])))
    return np.array(prior)

def solve(prior, score, valid):
    score = gaussian_filter1d(score, 1.2, axis=1)
    unary = -score + (OFF[:, None] / 32.0) ** 2
    unary = np.where(valid, unary, 100000.0)
    cost = unary[:, 0].copy()
    n = len(OFF)
    w = len(X)
    index = np.arange(n)
    back = np.zeros((w, n), np.int16)
    for x in range(1, w):
        opts = np.full((5, n), np.inf)
        for (k, delta) in enumerate(range(-2, 3)):
            src = index + delta
            ok = (src >= 0) & (src < n)
            opts[k, ok] = cost[src[ok]] + 0.12 * delta * delta
        win = opts.argmin(0)
        cost = unary[:, x] + opts[win, index]
        back[x] = index + win - 2
    path = np.zeros(w, np.int16)
    path[-1] = cost.argmin()
    for x in range(w - 1, 0, -1):
        path[x - 1] = back[x, path[x]]
    return prior + gaussian_filter1d(OFF[path].astype(float), 1.2)

def project_rig(v, row):
    a = row["alignment"]
    s = a['scale']
    native = np.c_[v[:, 1] / s + a['center_x'], a['rear_y'] - v[:, 0] / s, (v[:, 2] - a['height'] - a['dz']) / s + a['roof_z']]
    return project(native, row, row["alignment"])[0]

def panel(params, row):
    (x, y, z, wt, wb) = params
    zb = 0.75
    profile = np.array([[-wt * 0.35, z], [wt * 0.35, z], [wt * 0.48, z - 0.006], [wt * 0.5, z - 0.014], [wb * 0.5, zb], [-wb * 0.5, zb], [-wt * 0.5, z - 0.014], [-wt * 0.48, z - 0.006]])
    v = np.c_[np.full(len(profile), x), y + profile[:, 0], profile[:, 1]]
    return project_rig(v, row)

def features(poly, side_y):
    top = poly[:2].mean(0)
    q = np.r_[poly, poly[:1]]
    xs = []
    for (a, b) in zip(q[:-1], q[1:]):
        if min(a[1], b[1]) <= side_y <= max(a[1], b[1]) and abs(b[1] - a[1]) > 1e-08:
            xs.append(a[0] + (side_y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]))
    if len(xs) < 2:
        return np.r_[top, top[0] - 5, top[0] + 5]
    return np.r_[top, min(xs), max(xs)]

def score_pose(labs, poly, dx, dy, scale):
    p = poly.copy()
    ctr = p[:2].mean(0)
    p = (p - ctr) * scale + ctr + [dx, dy]
    top = p[:2].mean(0)
    ys = np.linspace(top[1] + 10, top[1] + 42, 12)
    contrasts = []
    for im in labs:
        ls = []
        rs = []
        for yy in ys:
            (_, _, l, r) = features(p, yy)
            l = int(round(l))
            r = int(round(r))
            v = int(round(yy))
            if l < 12 or r > 1907 or v < 6 or (v >= 1074):
                continue
            ls.append(float(np.median(im[v - 2:v + 3, l - 9:l - 3, 0]) - np.median(im[v - 2:v + 3, l + 3:l + 9, 0])))
            rs.append(float(np.median(im[v - 2:v + 3, r + 3:r + 9, 0]) - np.median(im[v - 2:v + 3, r - 9:r - 3, 0])))
        cx = int(round(top[0]))
        ty = int(round(top[1]))
        tc = float(np.median(im[ty - 8:ty - 3, cx - 12:cx + 13, 0]) - np.median(im[ty + 4:ty + 10, cx - 12:cx + 13, 0]))
        contrasts.append([float(np.median(ls)) if ls else -100, float(np.median(rs)) if rs else -100, tc])
    c = np.median(contrasts, axis=0)
    score = float(np.mean(np.minimum(c, 35)) + 0.5 * min(c)) - 0.015 * (dx * dx + dy * dy) - 20 * (scale - 1) ** 2
    return (score, c, p)

def sample(a, x, y):
    return cv2.remap(a.astype(np.float32), np.asarray(x, np.float32).reshape(1, -1), np.asarray(y, np.float32).reshape(1, -1), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1)

def hood(rgb, z, w, stock, constrained):
    x = z['x']
    prior = z['geometry']
    old = z['refined']
    xs = w['x']
    ls = np.stack([cv2.cvtColor(im[:, 500:1421], cv2.COLOR_RGB2LAB)[:, :, 0] for im in rgb]).astype(np.float32)
    gs = np.stack([cv2.Sobel(cv2.GaussianBlur(im, (0, 0), 1.2), cv2.CV_32F, 0, 1, ksize=3) / 8 for im in ls])
    grad = gaussian_filter1d(np.median(gs, axis=0), 4, axis=1)
    support = np.mean(gs > 0.45, axis=0)
    ax = []
    ay = []
    aw = []
    for col in x[::4]:
        p = prior[col - 600]
        lo = max(4, int(p - 18) - 850)
        hi = min(225, int(p + 45) - 850)
        if hi <= lo:
            continue
        g = grad[lo:hi, col - 500]
        (peaks, props) = find_peaks(g, prominence=0.35)
        if not len(peaks):
            continue
        mx = float(g[peaks].max())
        peaks = peaks[g[peaks] >= max(0.55, 0.28 * mx)]
        peaks = [k for k in peaks if support[lo + k, col - 500] >= 0.42]
        if not peaks:
            continue
        k = peaks[0]
        ax.append(col)
        ay.append(lo + k + 850 - 0.6)
        aw.append(min(float(g[k]), 6.0))
    ax = np.array(ax)
    ay = np.array(ay)
    aw = np.array(aw)
    reliable = len(ax) >= 45 and np.ptp(ax) > 500
    if reliable:
        knots = np.linspace(600, 1320, 10)
        target = ay - np.interp(ax, x, prior)
        weights = np.sqrt(np.minimum(aw, 3))

        def residual(v):
            return np.r_[weights * (np.interp(ax, knots, v) - target), 0.22 * np.diff(v, 2), 0.025 * v]
        fit = least_squares(residual, np.full(10, np.median(target)), bounds=(-25, 45), loss='soft_l1', f_scale=2.5)
        aligned = prior + np.interp(x, knots, fit.x)
        offsets = np.arange(-5, 6)
        scores = np.stack([sample(grad, x - 500, aligned + o - 850) for o in offsets])
        scores -= 0.075 * offsets[:, None] ** 2
        local = offsets[scores.argmax(0)].astype(float)
        curve = aligned + gaussian_filter1d(local, 4)
        residuals = ay - np.interp(ax, x, curve)
        unc = np.interp(x, ax, np.minimum(abs(residuals), 30)) > 5
    else:
        aligned = old.copy()
        curve = old.copy()
        unc = np.ones(len(x), bool)
    dark = gaussian_filter1d(median_filter(w['dark_foreground'], size=41, mode='nearest'), 2)
    delta = np.interp(xs, x, curve - prior)
    line = np.minimum(w['geometry'] + delta, dark)
    mask = stock.copy()
    mask[:, xs] = np.arange(1080)[:, None] >= line
    uncertainty = np.zeros_like(mask)
    uncx = np.interp(xs, x, unc.astype(float)) > 0.5
    uncertainty[:, xs] = uncx[None, :] & (abs(np.arange(1080)[:, None] - line) < 7)
    return (mask, uncertainty, dict(x=x, prior=prior, old=old, aligned=aligned, curve=curve, wide_x=xs, wide_curve=line, anchor_x=ax, anchor_y=ay, anchor_weight=aw), dict(resolved_rim=bool(reliable), anchor_count=len(ax), median_shift_from_previous_px=float(np.median(curve - constrained)), uncertain_column_fraction=float(unc.mean())))

def contour(params, base):
    (cx, top, wt, wb, lean, radius) = params
    h = max(base - top, 40.0)
    r = min(radius, wt * 0.6)
    left_t = np.linspace(1, 0, 25)
    left = np.c_[cx - (wt + (wb - wt) * left_t) + lean * h * left_t, top + r + (h - r) * left_t]
    a = np.linspace(np.pi, 1.5 * np.pi, 9)
    lc = np.c_[cx - wt + r + r * np.cos(a), top + r + r * np.sin(a)]
    tx = np.linspace(cx - wt + r, cx + wt - r, 22)
    tt = np.c_[tx, np.full(len(tx), top)]
    a = np.linspace(1.5 * np.pi, 2 * np.pi, 9)
    rc = np.c_[cx + wt - r + r * np.cos(a), top + r + r * np.sin(a)]
    rt = np.linspace(0, 1, 25)
    right = np.c_[cx + wt + (wb - wt) * rt + lean * h * rt, top + r + (h - r) * rt]
    points = np.concatenate([left, lc, tt, rc, right])
    v = np.gradient(points, axis=0)
    norm = np.c_[-v[:, 1], v[:, 0]]
    norm /= np.maximum(np.linalg.norm(norm, axis=1, keepdims=True), 1e-06)
    return (points, norm, [slice(2, 23), slice(28, 62), slice(67, 88)])

def close_attachment(points, hoodline):
    xx = np.arange(int(points[0, 0]) - 3, int(points[-1, 0]) + 4)
    bottom = float(np.interp(xx, hoodline['wide_x'], hoodline['wide_curve']).max()) + 8
    ends = []
    for (p, q) in [(points[0], points[1]), (points[-1], points[-2])]:
        dx = (p[0] - q[0]) / max(p[1] - q[1], 0.1)
        ends.append([p[0] + dx * (bottom - p[1]), bottom])
    return np.concatenate([np.asarray(ends[:1]), points, np.asarray(ends[1:])])

def attachment(rgb, previous, hoodline):
    poly = np.array(previous['refined_polygon'])
    gpoly = np.array(previous['geometry_polygon'])
    cx0 = float(poly[:2, 0].mean())
    ty0 = float(poly[:2, 1].mean())
    base = float(np.interp(cx0, hoodline['wide_x'], hoodline['wide_curve'])) - 2
    x0 = int(cx0) - 100
    roi = rgb[:, :, x0:x0 + 201]
    labs = np.stack([cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32) for im in roi])
    smooth = np.stack([cv2.GaussianBlur(im, (0, 0), 1.0) for im in labs])
    gx = np.stack([cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3) / 8 for im in smooth])
    gy = np.stack([cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3) / 8 for im in smooth])
    ex = np.quantile(np.sqrt(gx[:, :, :, 0] ** 2 + 0.4 * (gx[:, :, :, 1:] ** 2).sum(3)), 0.35, axis=0).astype(np.float32)
    ey = np.quantile(np.sqrt(gy[:, :, :, 0] ** 2 + 0.4 * (gy[:, :, :, 1:] ** 2).sum(3)), 0.35, axis=0).astype(np.float32)
    p0 = np.array([cx0, ty0, 30.0, 38.0, 0.0, 5.0])
    geomcx = float(gpoly[:2, 0].mean())
    geomty = float(gpoly[:2, 1].mean())
    bounds = [(geomcx - 22, geomcx + 22), (geomty - 20, geomty + 20), (23, 38), (27, 48), (-0.1, 0.1), (2, 10)]

    def evidence(p):
        (pts, norm, parts) = contour(p, base)
        x = pts[:, 0] - x0
        y = pts[:, 1] - 850
        e = abs(norm[:, 0]) * sample(ex, x, y) + abs(norm[:, 1]) * sample(ey, x, y)
        bg = sum((abs(norm[:, 0]) * sample(ex, x + d * norm[:, 0], y + d * norm[:, 1]) + abs(norm[:, 1]) * sample(ey, x + d * norm[:, 0], y + d * norm[:, 1]) for d in [-7, 7])) / 2
        v = e - 0.45 * bg
        edges = np.array([np.mean(np.minimum(v[s], 20)) for s in parts])
        return (edges, pts, e, v)

    def objective(p):
        (e, _, _, _) = evidence(p)
        return -(float(e.mean()) + 0.6 * float(e.min())) + 0.003 * ((p[0] - geomcx) ** 2 + (p[1] - geomty) ** 2) + 0.006 * ((p[2] - 30) ** 2 + (p[3] - 37) ** 2) + 0.5 * p[4] ** 2
    coarse = []
    for dc in range(-18, 19, 3):
        for dt in range(-18, 19, 3):
            for width in [27.0, 30.0, 33.0]:
                q = np.array([geomcx + dc, geomty + dt, width, width + 8, 0.0, 5.0])
                coarse.append((objective(q), q))
    coarse.sort(key=lambda q: q[0])
    starts = [p0]
    for (cost, q) in coarse:
        if all((np.linalg.norm(q[:2] - p[:2]) > 3 for p in starts)):
            starts.append(q)
        if len(starts) >= 5:
            break
    fits = [minimize(objective, p, method='Powell', bounds=bounds, options=dict(maxiter=65, xtol=0.12, ftol=0.002)) for p in starts]
    fit = min(fits, key=lambda f: f.fun)
    p = fit.x
    (edge, points, raw, prom) = evidence(p)
    fit_edge = edge.copy()
    edge[1] = float(np.median(prom[36:54]))
    present = bool(edge.min() > 1.0 and edge.mean() > 2.0)
    status = 'present' if present else 'uncertain' if edge.mean() > 1.0 else 'absent'
    mask_points = close_attachment(points, hoodline)
    m = np.zeros((1080, 1920), np.uint8)
    cv2.fillPoly(m, [np.rint(mask_points).astype(np.int32)], 255)
    band = np.zeros_like(m)
    cv2.polylines(band, [np.rint(points).astype(np.int32)], False, 255, 5)
    out = dict(presence_prediction=status, previous_prediction=previous['presence_prediction'], parameters=p, edge_scores_left_top_right=edge, fit_edge_scores=fit_edge, fit_cost=float(fit.fun), polygon=mask_points, fit_polygon=points, context_frames=len(rgb), automatic=True)
    return (m > 0, band > 0, out)


def solve_constrained(prior, score, valid, strength=8):
    off = OFF
    score = gaussian_filter1d(score, 1.2, axis=1)
    unary = -score + strength * (off[:, None] / 32) ** 2
    unary = np.where(valid, unary, 100000.0)
    cost = unary[:, 0].copy()
    n = len(off)
    w = len(prior)
    ix = np.arange(n)
    back = np.zeros((w, n), np.int16)
    for x in range(1, w):
        opts = np.full((5, n), np.inf)
        for (k, d) in enumerate(range(-2, 3)):
            src = ix + d
            ok = (src >= 0) & (src < n)
            opts[k, ok] = cost[src[ok]] + 0.24 * d * d
        win = opts.argmin(0)
        cost = unary[:, x] + opts[win, ix]
        back[x] = ix + win - 2
    path = np.zeros(w, np.int16)
    path[-1] = cost.argmin()
    for x in range(w - 1, 0, -1):
        path[x - 1] = back[x, path[x]]
    return prior + gaussian_filter1d(off[path].astype(float), 1.2)

def geometry_line(row, xs, vertices, faces):
    v, e = subdivide(vertices, faces)
    (uv, z) = project(v, row, row["alignment"])
    q = uv[e]
    validz = (z[e] > 0.02).all(1)
    (a, b) = (q[:, 0, 0], q[:, 1, 0])
    out = []
    for x in xs:
        ok = validz & (np.minimum(a, b) <= x) & (np.maximum(a, b) >= x) & (np.abs(b - a) > 1e-09)
        out.append(float(np.min(q[ok, 0, 1] + (x - a[ok]) / (b[ok] - a[ok]) * (q[ok, 1, 1] - q[ok, 0, 1]))) if ok.any() else 1200.0)
    return np.array(out)


def initial_boundaries(rgb, stock, row, vertices, faces):
    """Original image-led fallback and wider dashboard foreground estimate."""
    v, edges = subdivide(vertices, faces)
    prior = prior_boundary(v, edges, row)
    mx = np.tile(X - 550, (len(OFF), 1)).astype(np.float32)
    my = (prior[None, :] + OFF[:, None] - 850).astype(np.float32)
    stack, bottom = [], []
    for image in rgb:
        lab = cv2.cvtColor(image[:, 550:1371], cv2.COLOR_RGB2LAB).astype(np.float32)
        blur = cv2.GaussianBlur(lab, (0, 0), 1.3)
        dy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3) / 8.
        grad = np.sqrt(dy[:, :, 0]**2 + .5*dy[:, :, 1]**2 + .5*dy[:, :, 2]**2)
        score = cv2.remap(grad, mx, my, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        stack.append(np.minimum(score / max(float(np.percentile(score, 95)), 1.5), 2.))
        bottom.append(cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32))
    stack = np.stack(stack)
    valid = (my >= 4) & (my < 226)
    if not valid.any(0).all():
        raise ValueError('Frozen search band falls outside the image')
    ref = solve(prior, np.median(stack, 0), valid)
    initial = dict(x=X, geometry=prior, refined=ref,
                   early=solve(prior, np.median(stack[:11], 0), valid),
                   late=solve(prior, np.median(stack[-11:], 0), valid))
    constrained = solve_constrained(prior, np.median(stack, 0), valid)
    xs = np.arange(250, 1651)
    geo = geometry_line(row, xs, vertices, faces)
    smooth = cv2.GaussianBlur(np.median(np.stack(bottom), axis=0), (0, 0), 2)
    dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) / 8
    yy = np.arange(850, 1080)
    dark = []
    for j, col in enumerate(xs):
        lb = np.flatnonzero(stock[850:, col])
        lower = int(lb[0] + 850) if len(lb) else 1079
        candidate = (yy >= max(930, lower)) & (yy >= min(geo[j]-45, 1040)) & (yy < 1075)
        score = -dy[:, col] - .003*(yy - np.clip(geo[j], 1000, 1060))**2
        score[~candidate] = -1e6
        k = int(score.argmax())
        dark.append(yy[k] if smooth[min(k+8, 229), col] < 30 and -dy[k, col] > 1.5 else 1080.)
    dark = gaussian_filter1d(np.array(dark, float), 3)
    return initial, dict(x=xs, geometry=geo, dark_foreground=dark), constrained


def initialize_attachment(history_rgb, row, parameters):
    """Frozen coarse fit; its polygon seeds v2, whose decision is authoritative."""
    poly = panel(parameters, row)
    labs = [cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32) for im in history_rgb]
    best = (-1e10, None, None, None)
    for dx in range(-14, 15, 2):
        for dy in range(-14, 15, 2):
            for scale in [.94, 1., 1.06]:
                score, contrast, points = score_pose(labs, poly, dx, dy, scale)
                if score > best[0]:
                    best = score, contrast, points, [dx, dy, scale]
    score, contrast, points, adjustment = best
    present = bool(min(contrast) > 4 and np.mean(contrast) > 7)
    status = 'present' if present else ('uncertain' if min(contrast) > 0 and np.mean(contrast) > 3 else 'absent')
    return dict(presence_prediction=status, contrast_left_right_top=contrast,
                score=float(score), adjustment=adjustment,
                geometry_polygon=poly, refined_polygon=points)


def compute(rgb, history_rgb, stock, row, vertices, faces, config):
    """Compute masks from 33 RGB bottom crops and four full original histories."""
    if rgb.shape != (33, 230, 1920, 3) or np.shape(history_rgb) != (4, 1080, 1920, 3):
        raise ValueError('Frozen method requires 33 bottom crops and four 1920x1080 histories')
    row = {**row, 'alignment': config['alignment']}
    initial, wide, constrained = initial_boundaries(rgb, stock, row, vertices, faces)
    initialization = initialize_attachment(history_rgb, row, config['attachment_parameters'])
    h, u, curves, stats = hood(rgb, initial, wide, stock, constrained)
    a, band, attachment_stats = attachment(rgb, initialization, curves)
    applied = a if attachment_stats['presence_prediction'] == 'present' else np.zeros_like(a)
    masks = dict(hood=h, attachment_proposal=a, attachment_applied=applied,
                 refined=h | applied, uncertain=u | band)
    diagnostics = dict(hood=stats, attachment=attachment_stats,
                       attachment_initialization=initialization)
    return masks, curves, diagnostics
