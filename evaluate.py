#!/usr/bin/env python
"""evaluate.py — Paper-only evaluator (MICCAI 2026, Paper ID 4807).

Computes ONLY the metrics reported in the paper's main table:
  CD, Curvature WD, Tortuosity WD, Radius WD,
  Tapering WD (per-vessel paper definition + per-edge |dr/ds| thesis extension),
  Murray's law compliance (±20%), Bounds compliance (overall).

Slimmed from the original evaluator (which also computed JSD/MMD/SWD/1-NN/PR/F1/
Coverage/Density/Smoothness/Topological/Manifold-Coverage/Absolute/Intra-vessel/
Variance-ratio analyses — all removed). Numerics for the retained metrics are
byte-equivalent to the original; non-paper analyses are simply not computed.
"""
import os, sys, glob, argparse, json, warnings, time, gc
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from tqdm import tqdm

try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None

warnings.filterwarnings('ignore')

EPS = 1e-10
N_SAMPLE_POINTS = 2048
CD_BATCH_SIZE = 50
GLOBAL_SEED = 42


def set_global_seed(seed: int = GLOBAL_SEED):
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


# ==================== Paper-only dataclasses ====================

@dataclass
class AnatomicalMetrics:
    curvature_wasserstein: float = 0.0
    segment_tortuosity_wasserstein: float = 0.0
    murrays_law_compliance_strict: float = 0.0  # ±20% rate
    murrays_law_n_samples: int = 0
    tapering_ratio_wasserstein: float = 0.0     # per-vessel (paper)
    tapering_slope_wasserstein: float = 0.0     # per-edge |dr/ds| (thesis extension)
    tapering_slope_n_edges: int = 0


@dataclass
class RadiusMetrics:
    radius_wasserstein: float = 0.0
    radius_mean: float = 0.0
    gt_radius_mean: float = 0.0


@dataclass
class BoundsMetrics:
    curvature_in_bounds: float = 0.0
    torsion_in_bounds: float = 0.0
    radius_in_bounds: float = 0.0
    length_in_bounds: float = 0.0
    tortuosity_in_bounds: float = 0.0
    overall_in_bounds: float = 0.0  # paper "Bounds" = mean of 5
    gt_curvature_range: Tuple[float, float] = (0.0, 0.0)
    gt_radius_range: Tuple[float, float] = (0.0, 0.0)
    gt_length_range: Tuple[float, float] = (0.0, 0.0)


@dataclass
class FidelityMetrics:
    chamfer_distance_mean: float = 0.0  # paper CD (×10^3 when reported in table)


@dataclass
class EvaluationResults:
    gt_anatomical: AnatomicalMetrics = field(default_factory=AnatomicalMetrics)
    baseline_anatomical: Optional[AnatomicalMetrics] = None
    ours_anatomical: AnatomicalMetrics = field(default_factory=AnatomicalMetrics)
    baseline_radius: Optional[RadiusMetrics] = None
    ours_radius: RadiusMetrics = field(default_factory=RadiusMetrics)
    baseline_bounds: Optional[BoundsMetrics] = None
    ours_bounds: BoundsMetrics = field(default_factory=BoundsMetrics)
    baseline_fidelity: Optional[FidelityMetrics] = None
    ours_fidelity: FidelityMetrics = field(default_factory=FidelityMetrics)


# ==================== Loading & normalization (unchanged) ====================

def load_ply(path: str) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    points, radius, edges = [], [], []
    with open(path, 'r') as f:
        lines = f.readlines()
    header_end, n_vertices, n_edges = 0, 0, 0
    has_radius, radius_col_idx = False, 3
    property_count, in_vertex_section = 0, False
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('element vertex'):
            n_vertices = int(s.split()[-1]); in_vertex_section = True; property_count = 0
        elif s.startswith('element edge'):
            n_edges = int(s.split()[-1]); in_vertex_section = False
        elif in_vertex_section and s.startswith('property'):
            parts = s.split()
            if len(parts) >= 3 and parts[2].lower() in ['r', 'radius', 'scalar', 'quality', 'rad']:
                has_radius = True; radius_col_idx = property_count
            property_count += 1
        elif s == 'end_header':
            header_end = i + 1; break
    for i in range(header_end, header_end + n_vertices):
        parts = lines[i].strip().split()
        if len(parts) >= 3:
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            radius.append(float(parts[radius_col_idx]) if (has_radius and len(parts) > radius_col_idx) else 0.1)
    for i in range(header_end + n_vertices, header_end + n_vertices + n_edges):
        parts = lines[i].strip().split()
        if len(parts) >= 2:
            edges.append((int(parts[0]), int(parts[1])))
    return np.array(points, dtype=np.float64), np.array(radius, dtype=np.float64), edges


def load_held_out_filenames(split_csv: str) -> set:
    import pandas as pd
    df = pd.read_csv(split_csv)
    out = set()
    for f in df['filenames'].tolist():
        out.add(f.replace('.pt', '.ply').replace('.npz', '.ply'))
    return out


def load_all_vessels(directory: str, max_files=None, filter_filenames=None) -> List[Dict]:
    files = sorted(glob.glob(os.path.join(directory, "*.ply")))
    if filter_filenames is not None:
        files = [p for p in files if os.path.basename(p) in filter_filenames]
    if max_files:
        files = files[:max_files]
    out = []
    for path in tqdm(files, desc="    Loading", leave=False, ncols=80):
        try:
            pts, rad, edges = load_ply(path)
            if len(pts) > 0:
                out.append({'points': pts, 'radius': rad, 'edges': edges, 'filename': os.path.basename(path)})
        except Exception:
            pass
    return out


def normalize_vessels_with_radius(vessels: List[Dict]) -> List[Dict]:
    out = []
    for v in vessels:
        pts = v['points'].copy(); rad = v['radius'].copy(); scale = 1.0
        if len(pts) > 0:
            pts = pts - np.mean(pts, axis=0)
            scale = np.max(np.abs(pts)) + EPS
            pts = pts / scale; rad = rad / scale
        out.append({'points': pts, 'radius': rad, 'edges': v.get('edges', []).copy() if v.get('edges') else [],
                    'filename': v.get('filename', ''), 'scale': scale})
    return out


# ==================== Topology & segments (unchanged) ====================

def build_adjacency(edges, n_points):
    adj = defaultdict(list)
    for i, j in edges:
        if 0 <= i < n_points and 0 <= j < n_points:
            adj[i].append(j); adj[j].append(i)
    return dict(adj)


def find_bifurcations(adj):
    return [n for n, ne in adj.items() if len(ne) >= 3]


def find_endpoints(adj):
    return [n for n, ne in adj.items() if len(ne) == 1]


def order_points_by_edges(points, edges):
    if len(edges) == 0 or len(points) == 0: return points
    adj = build_adjacency(edges, len(points))
    endpoints = find_endpoints(adj)
    start = endpoints[0] if endpoints else (list(adj.keys())[0] if adj else 0)
    visited, ordered, current = set(), [], start
    while current is not None and current not in visited:
        visited.add(current); ordered.append(current); nxt = None
        for nb in adj.get(current, []):
            if nb not in visited: nxt = nb; break
        current = nxt
    return points[ordered] if len(ordered) >= 2 else points


def _extract_segments_core(points, edges, with_indices: bool):
    if len(edges) == 0 or len(points) < 2:
        if len(points) >= 2:
            return [(points, list(range(len(points))))] if with_indices else [points]
        return []
    n = len(points); adj = build_adjacency(edges, n)
    if not adj:
        if len(points) >= 2:
            return [(points, list(range(len(points))))] if with_indices else [points]
        return []
    special = {nd for nd, ne in adj.items() if len(ne) != 2}
    if not special:
        ordered_idx = []
        endpoints = find_endpoints(adj)
        start = endpoints[0] if endpoints else list(adj.keys())[0]
        visited, current = set(), start
        while current is not None and current not in visited:
            visited.add(current); ordered_idx.append(current); nxt = None
            for nb in adj.get(current, []):
                if nb not in visited: nxt = nb; break
            current = nxt
        if len(ordered_idx) >= 2:
            return [(points[ordered_idx], ordered_idx)] if with_indices else [points[ordered_idx]]
        return [(points, list(range(len(points))))] if with_indices else [points]
    segs, visited_edges = [], set()
    for s in special:
        for nb in adj.get(s, []):
            ek = (min(s, nb), max(s, nb))
            if ek in visited_edges: continue
            seg_idx = [s]; prev, current = s, nb
            while current not in special:
                ek = (min(prev, current), max(prev, current))
                visited_edges.add(ek); seg_idx.append(current); nxt = None
                for nn in adj.get(current, []):
                    if nn != prev: nxt = nn; break
                if nxt is None: break
                prev, current = current, nxt
            ek = (min(prev, current), max(prev, current))
            visited_edges.add(ek); seg_idx.append(current)
            if len(seg_idx) >= 2:
                segs.append((points[seg_idx], seg_idx) if with_indices else points[seg_idx])
    if not segs:
        ordered = order_points_by_edges(points, edges)
        if len(ordered) >= 2:
            return [(ordered, list(range(len(points))))] if with_indices else [ordered]
        return []
    return segs


def extract_segments(points, edges):
    return _extract_segments_core(points, edges, False)


def extract_segments_with_indices(points, edges):
    return _extract_segments_core(points, edges, True)


def extract_per_segment_attributes(vessel):
    pts, rad = vessel['points'], vessel['radius']
    edges = vessel.get('edges', [])
    result = []
    for seg_pts, seg_idx in extract_segments_with_indices(pts, edges):
        if len(seg_pts) < 2: continue
        seg_len = compute_segment_length(seg_pts)
        if seg_len < EPS: continue
        attr = {'length': seg_len, 'tortuosity': compute_tortuosity(seg_pts)}
        attr['curvature'] = float(np.mean(compute_curvature_robust(seg_pts))) if len(seg_pts) >= 3 else 0.0
        attr['torsion'] = float(np.mean(compute_torsion_robust(seg_pts))) if len(seg_pts) >= 4 else 0.0
        valid = [i for i in seg_idx if i < len(rad)]
        attr['radius'] = float(np.mean(rad[valid])) if valid else 0.1
        result.append(attr)
    return result


def compute_curvature_robust(points):
    if len(points) < 3: return np.array([0.0])
    out = []
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i-1]; v2 = points[i+1] - points[i]
        l1, l2 = norm(v1), norm(v2)
        if l1 < EPS or l2 < EPS: out.append(0.0); continue
        ca = np.clip(np.dot(v1, v2) / (l1*l2), -1.0, 1.0)
        ds = (l1 + l2) / 2
        out.append(np.arccos(ca) / ds if ds > EPS else 0.0)
    return np.array(out) if out else np.array([0.0])


def compute_torsion_robust(points):
    if len(points) < 4: return np.array([0.0])
    out = []
    for i in range(1, len(points) - 2):
        t1 = points[i] - points[i-1]; t2 = points[i+1] - points[i]; t3 = points[i+2] - points[i+1]
        b1, b2 = np.cross(t1, t2), np.cross(t2, t3)
        nb1, nb2, nt2 = norm(b1), norm(b2), norm(t2)
        if nb1 < EPS or nb2 < EPS or nt2 < EPS: out.append(0.0); continue
        ca = np.clip(np.dot(b1/nb1, b2/nb2), -1.0, 1.0)
        out.append(float(np.clip(np.arccos(ca) / nt2, 0, 100)))
    return np.array(out) if out else np.array([0.0])


def compute_tortuosity(points):
    if len(points) < 2: return 1.0
    chord = norm(points[-1] - points[0])
    if chord < EPS: return 1.0
    return float(np.sum(norm(np.diff(points, axis=0), axis=1)) / chord)


def compute_segment_length(points):
    if len(points) < 2: return 0.0
    return float(np.sum(norm(np.diff(points, axis=0), axis=1)))


def compute_edge_tapering_slopes(points, edges, radius):
    """Per-edge radius decrease rate |dr/ds| (least-squares slope of radius on cumulative arc length)."""
    slopes = []
    for seg_pts, seg_idx in extract_segments_with_indices(points, edges):
        if len(seg_pts) < 2: continue
        idx = np.asarray(seg_idx)
        if idx.size != len(seg_pts) or idx.max() >= len(radius): continue
        r = np.asarray(radius, dtype=np.float64)[idx]
        d = norm(np.diff(seg_pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)])
        if s[-1] < EPS or np.var(s) < EPS: continue
        slopes.append(abs(float(np.cov(s, r, bias=True)[0, 1] / np.var(s))))
    return slopes


# ==================== Anatomical metrics (paper-only) ====================

def _safe_w(a, b):
    if len(a) > 0 and len(b) > 0:
        return float(wasserstein_distance(a, b))
    return 0.0


def _accumulate_anatomical(vessels):
    """Pooled arrays needed by paper anatomical metrics."""
    curv, tort, tap_old, tap_new, murr = [], [], [], [], []
    for v in vessels:
        pts, rad = v['points'], v['radius']
        segs = extract_segments(pts, v.get('edges', []))
        for seg in segs:
            if len(seg) < 2: continue
            if compute_segment_length(seg) > EPS:
                tort.append(compute_tortuosity(seg))
            if len(seg) >= 3:
                curv.append(float(np.mean(compute_curvature_robust(seg))))
        # Tapering — per-vessel (paper)
        if len(segs) > 0 and len(rad) >= 2:
            adj_t = build_adjacency(v['edges'], len(pts))
            eps = find_endpoints(adj_t)
            er = [(e, rad[e]) for e in eps if e < len(rad)]
            if len(er) >= 2:
                er.sort(key=lambda x: x[1], reverse=True)
                r_prox, r_dist = er[0][1], er[-1][1]
                if r_prox > EPS:
                    tap_old.append(r_dist / r_prox)
        # Tapering — per-edge (thesis extension)
        tap_new.extend(compute_edge_tapering_slopes(pts, v.get('edges', []), rad))
        # Murray
        adj = build_adjacency(v.get('edges', []), len(pts))
        for bif in find_bifurcations(adj):
            ne = adj.get(bif, [])
            if len(ne) < 3 or bif >= len(pts): continue
            nr = [rad[n] for n in ne if n < len(rad) and rad[n] > EPS]
            if len(nr) >= 3:
                sr = sorted(nr, reverse=True)
                if sr[0] > EPS:
                    murr.append(sum(r**3 for r in sr[1:]) / (sr[0]**3))
    return curv, tort, tap_old, tap_new, murr


def compute_anatomical_metrics(vessels, gt_vessels=None):
    curv, tort, tap_old, tap_new, murr = _accumulate_anatomical(vessels)
    gt_curv = gt_tort = gt_tap_old = gt_tap_new = []
    if gt_vessels:
        gt_curv, gt_tort, gt_tap_old, gt_tap_new, _ = _accumulate_anatomical(gt_vessels)
    murray_strict = float(np.mean([abs(r - 1.0) < 0.2 for r in murr])) if murr else 0.0
    return AnatomicalMetrics(
        curvature_wasserstein=_safe_w(curv, gt_curv),
        segment_tortuosity_wasserstein=_safe_w(tort, gt_tort),
        murrays_law_compliance_strict=murray_strict,
        murrays_law_n_samples=len(murr),
        tapering_ratio_wasserstein=_safe_w(tap_old, gt_tap_old),
        tapering_slope_wasserstein=_safe_w(tap_new, gt_tap_new),
        tapering_slope_n_edges=len(tap_new),
    )


# ==================== Radius metric ====================

def compute_radius_metrics(gen_vessels, gt_vessels):
    gen_r = np.array([sa['radius'] for v in gen_vessels for sa in extract_per_segment_attributes(v)])
    gt_r = np.array([sa['radius'] for v in gt_vessels for sa in extract_per_segment_attributes(v)])
    if len(gen_r) == 0 or len(gt_r) == 0:
        return RadiusMetrics()
    return RadiusMetrics(
        radius_wasserstein=float(wasserstein_distance(gen_r, gt_r)),
        radius_mean=float(np.mean(gen_r)),
        gt_radius_mean=float(np.mean(gt_r)),
    )


# ==================== Bounds compliance ====================

def _extract_all_attributes(vessels):
    a = {k: [] for k in ['curvature', 'torsion', 'radius', 'length', 'tortuosity']}
    for v in vessels:
        for sa in extract_per_segment_attributes(v):
            for k in a: a[k].append(sa[k])
    return {k: np.array(vv) for k, vv in a.items()}


def compute_bounds_metrics(gen_vessels, gt_vessels):
    g = _extract_all_attributes(gen_vessels); t = _extract_all_attributes(gt_vessels)

    def ib(gv, tv, lp=2.5, hp=97.5):
        if len(gv) == 0 or len(tv) == 0: return 0.0, (0.0, 0.0)
        lo, hi = np.percentile(tv, lp), np.percentile(tv, hp)
        return float(np.mean((gv >= lo) & (gv <= hi))), (float(lo), float(hi))

    cib, cr = ib(g['curvature'], t['curvature'])
    sib, _ = ib(g['torsion'],   t['torsion'])
    rib, rr = ib(g['radius'],    t['radius'])
    lib, lr = ib(g['length'],    t['length'])
    tib, _ = ib(g['tortuosity'], t['tortuosity'])
    return BoundsMetrics(
        curvature_in_bounds=cib, torsion_in_bounds=sib, radius_in_bounds=rib,
        length_in_bounds=lib, tortuosity_in_bounds=tib,
        overall_in_bounds=float(np.mean([cib, sib, rib, lib, tib])),
        gt_curvature_range=cr, gt_radius_range=rr, gt_length_range=lr,
    )


# ==================== Chamfer Distance ====================

def chamfer_distance_gpu_batch(pcs1, pcs2, batch_size=CD_BATCH_SIZE):
    N1, N2 = pcs1.shape[0], pcs2.shape[0]
    out = torch.zeros(N1, N2, device=pcs1.device, dtype=torch.float32)
    for i in tqdm(range(N1), desc="      GPU CD", leave=False, ncols=80):
        pc1 = pcs1[i]
        for js in range(0, N2, batch_size):
            je = min(js + batch_size, N2)
            batch = pcs2[js:je]; B = batch.shape[0]
            pc1e = pc1.unsqueeze(0).expand(B, -1, -1)
            dm = torch.cdist(pc1e, batch, p=2) ** 2
            out[i, js:je] = dm.min(dim=2)[0].mean(dim=1) + dm.min(dim=1)[0].mean(dim=1)
    return out


def pairwise_chamfer_distance(pcs1, pcs2):
    N1, N2 = len(pcs1), len(pcs2)
    print(f"      Computing {N1}x{N2}={N1*N2:,} pairs...")
    if TORCH_AVAILABLE and DEVICE is not None and DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        a = torch.tensor(pcs1, dtype=torch.float32, device=DEVICE)
        b = torch.tensor(pcs2, dtype=torch.float32, device=DEVICE)
        r = chamfer_distance_gpu_batch(a, b).cpu().numpy()
        del a, b; torch.cuda.empty_cache()
        return r
    out = np.zeros((N1, N2), dtype=np.float32)
    for i in tqdm(range(N1), desc="      CPU CD", leave=False):
        for j in range(N2):
            d = cdist(pcs1[i], pcs2[j], 'sqeuclidean')
            out[i, j] = np.mean(np.min(d, axis=1)) + np.mean(np.min(d, axis=0))
    return out


def sample_points_from_vessel(vessel, n_points=N_SAMPLE_POINTS):
    pts = vessel['points']; edges = vessel.get('edges', []); N = len(pts)
    if N == 0: return np.zeros((n_points, 3), dtype=np.float32)
    if len(edges) == 0 or N < 4:
        idx = np.random.choice(N, n_points, replace=(N < n_points))
        return pts[idx].astype(np.float32)
    ve, el = [], []
    for i, j in edges:
        if i < N and j < N:
            L = norm(pts[i] - pts[j])
            if L > EPS: ve.append((i, j)); el.append(L)
    if not ve:
        idx = np.random.choice(N, n_points, replace=(N < n_points))
        return pts[idx].astype(np.float32)
    total = sum(el); samp = []
    for (i, j), L in zip(ve, el):
        ns = max(1, int(round(n_points * L / total)))
        for t in np.random.uniform(0, 1, ns):
            samp.append(pts[i] * (1 - t) + pts[j] * t)
    samp = np.array(samp, dtype=np.float32)
    if len(samp) > n_points:
        idx = np.random.choice(len(samp), n_points, replace=False); return samp[idx]
    if len(samp) < n_points:
        idx = np.random.choice(len(samp), n_points, replace=True); return samp[idx]
    return samp


def compute_chamfer_distance_metric(gen_pcs, gt_pcs, cd_gen_gt):
    if len(gen_pcs) == 0:
        return FidelityMetrics()
    return FidelityMetrics(
        chamfer_distance_mean=float((np.mean(np.min(cd_gen_gt, axis=1)) + np.mean(np.min(cd_gen_gt, axis=0))) / 2)
    )


# ==================== Run / Print / Save ====================

class _Timer:
    def __init__(self, n): self.n = n
    def __enter__(self): self.t = time.time(); return self
    def __exit__(self, *a): print(f"    [{self.n}] {time.time()-self.t:.1f}s")


def run_evaluation(gt_dir, baseline_dir, ours_dir, output_dir,
                   max_files=None, eval_mode="full", split_csv=None):
    set_global_seed(GLOBAL_SEED)
    os.makedirs(output_dir, exist_ok=True)
    R = EvaluationResults()
    print("[1/7] Loading vessels...")
    gt_filter = None
    if eval_mode == "held-out":
        if not split_csv or not os.path.exists(split_csv):
            raise ValueError(f"--split-csv required for held-out mode: {split_csv}")
        gt_filter = load_held_out_filenames(split_csv)
        print(f"    held-out: filter GT to {len(gt_filter)} files")
    with _Timer("load"):
        gt_o = load_all_vessels(gt_dir, max_files, filter_filenames=gt_filter)
        bl_o = load_all_vessels(baseline_dir, max_files) if baseline_dir else []
        ou_o = load_all_vessels(ours_dir, max_files)
    n_gt, n_bl, n_ou = len(gt_o), len(bl_o), len(ou_o)
    has_bl = n_bl > 0
    print(f"    GT={n_gt}, Baseline={n_bl}, Ours={n_ou}")

    print("[2/7] Normalizing (scale-fair)...")
    with _Timer("normalize"):
        gt = normalize_vessels_with_radius(gt_o)
        bl = normalize_vessels_with_radius(bl_o) if has_bl else []
        ou = normalize_vessels_with_radius(ou_o)

    print("[3/7] Anatomical (Curv/Tortu/Murray/Tapering)...")
    with _Timer("anatomical"):
        R.gt_anatomical = compute_anatomical_metrics(gt, gt)
        if has_bl: R.baseline_anatomical = compute_anatomical_metrics(bl, gt)
        R.ours_anatomical = compute_anatomical_metrics(ou, gt)

    print("[4/7] Radius WD...")
    with _Timer("radius"):
        if has_bl: R.baseline_radius = compute_radius_metrics(bl, gt)
        R.ours_radius = compute_radius_metrics(ou, gt)

    print("[5/7] Bounds compliance...")
    with _Timer("bounds"):
        if has_bl: R.baseline_bounds = compute_bounds_metrics(bl, gt)
        R.ours_bounds = compute_bounds_metrics(ou, gt)

    print("[6/7] Sampling point clouds...")
    with _Timer("sampling"):
        gt_pcs = np.array([sample_points_from_vessel(v) for v in gt])
        bl_pcs = np.array([sample_points_from_vessel(v) for v in bl]) if has_bl else np.array([])
        ou_pcs = np.array([sample_points_from_vessel(v) for v in ou])
    gc.collect()
    if TORCH_AVAILABLE and DEVICE is not None and DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    print("[7/7] Pairwise CD + set-level CD...")
    if has_bl:
        with _Timer("baseline CD"):
            cd = pairwise_chamfer_distance(bl_pcs, gt_pcs)
            R.baseline_fidelity = compute_chamfer_distance_metric(bl_pcs, gt_pcs, cd)
    with _Timer("ours CD"):
        cd = pairwise_chamfer_distance(ou_pcs, gt_pcs)
        R.ours_fidelity = compute_chamfer_distance_metric(ou_pcs, gt_pcs, cd)
    return R, n_gt, n_bl, n_ou


def print_results(R, n_gt, n_bl, n_ou):
    def f(v, p=4): return "—" if v is None else f"{v:.{p}f}"
    print("\n" + "=" * 80)
    print(f"  PAPER METRICS  (GT={n_gt}, Baseline={n_bl}, Ours={n_ou})")
    print("=" * 80)
    print(f"  {'Metric':<34} {'Baseline':>14} {'Ours':>14}")
    print("  " + "-" * 64)
    bf, of = R.baseline_fidelity, R.ours_fidelity
    ba, oa = R.baseline_anatomical, R.ours_anatomical
    br, or_ = R.baseline_radius, R.ours_radius
    bb, ob = R.baseline_bounds, R.ours_bounds
    rows = [
        ("CD (×10³ in paper) ↓",            (bf.chamfer_distance_mean if bf else None), of.chamfer_distance_mean),
        ("Curvature Wasserstein ↓",         (ba.curvature_wasserstein if ba else None), oa.curvature_wasserstein),
        ("Tortuosity Wasserstein ↓",        (ba.segment_tortuosity_wasserstein if ba else None), oa.segment_tortuosity_wasserstein),
        ("Radius Wasserstein ↓",            (br.radius_wasserstein if br else None), or_.radius_wasserstein),
        ("Tapering Wass (per-vessel) ↓",    (ba.tapering_ratio_wasserstein if ba else None), oa.tapering_ratio_wasserstein),
        ("Tapering Wass (per-edge) ↓",      (ba.tapering_slope_wasserstein if ba else None), oa.tapering_slope_wasserstein),
        ("Murray's law (±20%) ↑",           (ba.murrays_law_compliance_strict if ba else None), oa.murrays_law_compliance_strict),
        ("Bounds compliance ↑",             (bb.overall_in_bounds if bb else None), ob.overall_in_bounds),
    ]
    for name, b, o in rows:
        print(f"  {name:<34} {f(b):>14} {f(o):>14}")
    print("=" * 80)


def save_results(R, output_dir, n_gt, n_bl, n_ou, gt_dir, baseline_dir, ours_dir,
                 eval_mode="full", split_csv=None):
    out = {
        "meta": {"n_gt": n_gt, "n_baseline": n_bl, "n_ours": n_ou,
                 "eval_mode": eval_mode, "split_csv": split_csv,
                 "gt_dir": gt_dir, "baseline_dir": baseline_dir, "ours_dir": ours_dir,
                 "timestamp": datetime.now().isoformat(), "seed": GLOBAL_SEED},
        "results": asdict(R),
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)


def main():
    p = argparse.ArgumentParser(description="Paper-only evaluator (MICCAI 2026, Paper 4807)")
    p.add_argument("--gt-dir", type=str, required=True)
    p.add_argument("--baseline-dir", type=str, default="")
    p.add_argument("--ours-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="./evaluation_results")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--eval-mode", type=str, choices=["full", "held-out"], default="full")
    p.add_argument("--split-csv", type=str, default=None)
    a = p.parse_args()
    R, n_gt, n_bl, n_ou = run_evaluation(a.gt_dir, a.baseline_dir, a.ours_dir, a.output_dir,
                                          a.max_files, eval_mode=a.eval_mode, split_csv=a.split_csv)
    print_results(R, n_gt, n_bl, n_ou)
    save_results(R, a.output_dir, n_gt, n_bl, n_ou, a.gt_dir, a.baseline_dir, a.ours_dir,
                 eval_mode=a.eval_mode, split_csv=a.split_csv)


if __name__ == "__main__":
    main()
