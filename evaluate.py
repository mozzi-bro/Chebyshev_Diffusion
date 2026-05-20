
import os
import sys
import glob
import argparse
import json
import warnings
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import math

import gc
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance, entropy
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from numpy.linalg import norm
from tqdm import tqdm

try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        DEVICE = torch.device('cpu')
        GPU_NAME = None
        GPU_MEM_GB = 0
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    GPU_NAME = None
    GPU_MEM_GB = 0

warnings.filterwarnings('ignore')

EPS = 1e-10
N_SAMPLE_POINTS = 2048
JSD_RESOLUTION = 28
CD_BATCH_SIZE = 50
GLOBAL_SEED = 42

PR_THRESHOLDS = [99, 95, 90, 85, 75, 50, 25]


def set_global_seed(seed: int = GLOBAL_SEED):
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def print_header():
    print("\n" + "=" * 100)
    print("  Coronary Vessel Generation Evaluation")
    print("  Per-Segment Unification + Diversity Metrics")
    print("=" * 100)
    
    if TORCH_AVAILABLE and GPU_NAME:
        print(f"\n  * GPU: {GPU_NAME} | VRAM: {GPU_MEM_GB:.1f} GB | CUDA: {torch.version.cuda}")
    else:
        print(f"\n  [!] GPU not available - using CPU")
    print()


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


class Timer:
    def __init__(self, name=""):
        self.name = name
        self.start = None
        self.elapsed = 0
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.name:
            print(f"    -> {self.name}: {format_time(self.elapsed)}")


@dataclass
class SmoothnessMetrics:
    curvature_mean: float = 0.0
    curvature_std: float = 0.0
    curvature_variance_mean: float = 0.0
    jerk_rms_mean: float = 0.0
    jerk_rms_std: float = 0.0
    acceleration_smoothness: float = 0.0


@dataclass
class AnatomicalMetrics:
    branching_angle_mean: float = 0.0
    branching_angle_std: float = 0.0
    branching_angle_wasserstein: float = 0.0
    n_bifurcations_total: int = 0
    
    murrays_law_mean_ratio: float = 0.0
    murrays_law_std_ratio: float = 0.0
    murrays_law_compliance_strict: float = 0.0
    murrays_law_compliance_loose: float = 0.0
    murrays_law_n_samples: int = 0
    
    segment_tortuosity_mean: float = 0.0
    segment_tortuosity_std: float = 0.0
    segment_tortuosity_wasserstein: float = 0.0
    segment_length_mean: float = 0.0
    segment_length_wasserstein: float = 0.0
    
    curvature_mean: float = 0.0
    curvature_wasserstein: float = 0.0
    torsion_mean: float = 0.0
    torsion_wasserstein: float = 0.0
    
    tapering_ratio_mean: float = 0.0
    tapering_ratio_wasserstein: float = 0.0


@dataclass
class TopologicalMetrics:
    n_bifurcations_mean: float = 0.0
    n_bifurcations_wasserstein: float = 0.0
    n_endpoints_mean: float = 0.0
    n_endpoints_wasserstein: float = 0.0
    n_segments_mean: float = 0.0
    tree_depth_mean: float = 0.0


@dataclass
class PrecisionRecallMetrics:
    precision_p95: float = 0.0
    recall_p95: float = 0.0
    f1_p95: float = 0.0
    precision_p90: float = 0.0
    recall_p90: float = 0.0
    f1_p90: float = 0.0
    precision_p85: float = 0.0
    recall_p85: float = 0.0
    f1_p85: float = 0.0
    precision_p99: float = 0.0
    recall_p99: float = 0.0
    f1_p99: float = 0.0
    precision_p75: float = 0.0
    recall_p75: float = 0.0
    f1_p75: float = 0.0
    precision_p50: float = 0.0
    recall_p50: float = 0.0
    f1_p50: float = 0.0
    precision_p25: float = 0.0
    recall_p25: float = 0.0
    f1_p25: float = 0.0
    threshold_p95: float = 0.0
    threshold_p90: float = 0.0
    threshold_p50: float = 0.0
    threshold_p25: float = 0.0


@dataclass
class AnatomicalBoundsMetrics:
    curvature_in_bounds: float = 0.0
    torsion_in_bounds: float = 0.0
    radius_in_bounds: float = 0.0
    length_in_bounds: float = 0.0
    tortuosity_in_bounds: float = 0.0
    overall_in_bounds: float = 0.0
    
    curvature_ood: float = 0.0
    torsion_ood: float = 0.0
    radius_ood: float = 0.0
    length_ood: float = 0.0
    tortuosity_ood: float = 0.0
    overall_ood: float = 0.0
    
    gt_curvature_range: Tuple[float, float] = (0.0, 0.0)
    gt_radius_range: Tuple[float, float] = (0.0, 0.0)
    gt_length_range: Tuple[float, float] = (0.0, 0.0)


@dataclass
class LegitimateDiversityMetrics:
    raw_diversity: float = 0.0
    legitimate_diversity: float = 0.0
    legitimacy_ratio: float = 0.0
    manifold_coverage: float = 0.0
    curvature_legit_div: float = 0.0
    radius_legit_div: float = 0.0
    length_legit_div: float = 0.0


@dataclass
class FidelityMetrics:
    chamfer_distance_mean: float = 0.0
    chamfer_distance_std: float = 0.0
    chamfer_distance_median: float = 0.0
    jsd_3d: float = 0.0
    mmd_rbf: float = 0.0
    sliced_wasserstein: float = 0.0
    one_nn_accuracy: float = 0.0
    one_nn_deviation: float = 0.0
    one_nn_interpretation: str = ""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    coverage: float = 0.0
    density_score: float = 0.0


@dataclass
class AbsoluteMetrics:
    scale_valid: bool = True
    seg_length_mean_gen: float = 0.0
    seg_length_mean_gt: float = 0.0
    seg_length_ratio: float = 0.0
    seg_length_wasserstein: float = 0.0
    seg_radius_mean_gen: float = 0.0
    seg_radius_mean_gt: float = 0.0
    seg_radius_ratio: float = 0.0
    seg_radius_wasserstein: float = 0.0
    tree_length_mean_gen: float = 0.0
    tree_length_mean_gt: float = 0.0
    tree_length_ratio: float = 0.0


@dataclass
class IntraVesselDiversityMetrics:
    curvature_cv_mean: float = 0.0
    torsion_cv_mean: float = 0.0
    radius_cv_mean: float = 0.0
    length_cv_mean: float = 0.0
    tortuosity_cv_mean: float = 0.0
    overall_cv_mean: float = 0.0


@dataclass
class VarianceRatioMetrics:
    curvature_ratio: float = 0.0
    torsion_ratio: float = 0.0
    radius_ratio: float = 0.0
    length_ratio: float = 0.0
    tortuosity_ratio: float = 0.0
    overall_ratio: float = 0.0


@dataclass
class RadiusMetrics:
    radius_mean: float = 0.0
    radius_std: float = 0.0
    radius_wasserstein: float = 0.0
    radius_mean_diff: float = 0.0
    radius_in_bounds: float = 0.0
    gt_radius_mean: float = 0.0
    gt_radius_std: float = 0.0


@dataclass 
class EvaluationResults:
    gt_smoothness: SmoothnessMetrics = field(default_factory=SmoothnessMetrics)
    baseline_smoothness: SmoothnessMetrics = field(default_factory=SmoothnessMetrics)
    ours_smoothness: SmoothnessMetrics = field(default_factory=SmoothnessMetrics)
    
    gt_anatomical: AnatomicalMetrics = field(default_factory=AnatomicalMetrics)
    baseline_anatomical: AnatomicalMetrics = field(default_factory=AnatomicalMetrics)
    ours_anatomical: AnatomicalMetrics = field(default_factory=AnatomicalMetrics)
    
    gt_topological: TopologicalMetrics = field(default_factory=TopologicalMetrics)
    baseline_topological: TopologicalMetrics = field(default_factory=TopologicalMetrics)
    ours_topological: TopologicalMetrics = field(default_factory=TopologicalMetrics)
    
    baseline_precision_recall: PrecisionRecallMetrics = field(default_factory=PrecisionRecallMetrics)
    ours_precision_recall: PrecisionRecallMetrics = field(default_factory=PrecisionRecallMetrics)
    
    baseline_bounds: AnatomicalBoundsMetrics = field(default_factory=AnatomicalBoundsMetrics)
    ours_bounds: AnatomicalBoundsMetrics = field(default_factory=AnatomicalBoundsMetrics)
    
    baseline_legit_diversity: LegitimateDiversityMetrics = field(default_factory=LegitimateDiversityMetrics)
    ours_legit_diversity: LegitimateDiversityMetrics = field(default_factory=LegitimateDiversityMetrics)
    
    baseline_fidelity: FidelityMetrics = field(default_factory=FidelityMetrics)
    ours_fidelity: FidelityMetrics = field(default_factory=FidelityMetrics)
    
    baseline_radius: RadiusMetrics = field(default_factory=RadiusMetrics)
    ours_radius: RadiusMetrics = field(default_factory=RadiusMetrics)

    baseline_absolute: AbsoluteMetrics = field(default_factory=AbsoluteMetrics)
    ours_absolute: AbsoluteMetrics = field(default_factory=AbsoluteMetrics)

    baseline_intra_diversity: IntraVesselDiversityMetrics = field(default_factory=IntraVesselDiversityMetrics)
    ours_intra_diversity: IntraVesselDiversityMetrics = field(default_factory=IntraVesselDiversityMetrics)
    gt_intra_diversity: IntraVesselDiversityMetrics = field(default_factory=IntraVesselDiversityMetrics)
    baseline_variance_ratio: VarianceRatioMetrics = field(default_factory=VarianceRatioMetrics)
    ours_variance_ratio: VarianceRatioMetrics = field(default_factory=VarianceRatioMetrics)


def load_ply(path: str) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    points, radius, edges = [], [], []
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    header_end, n_vertices, n_edges = 0, 0, 0
    has_radius = False
    radius_col_idx = 3

    property_count = 0
    in_vertex_section = False
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        if line_stripped.startswith('element vertex'):
            n_vertices = int(line_stripped.split()[-1])
            in_vertex_section = True
            property_count = 0
        elif line_stripped.startswith('element edge'):
            n_edges = int(line_stripped.split()[-1])
            in_vertex_section = False
        elif in_vertex_section and line_stripped.startswith('property'):
            parts = line_stripped.split()
            if len(parts) >= 3:
                prop_name = parts[2].lower()
                if prop_name in ['r', 'radius', 'scalar', 'quality', 'rad']:
                    has_radius = True
                    radius_col_idx = property_count
            property_count += 1
        elif line_stripped == 'end_header':
            header_end = i + 1
            break
    
    for i in range(header_end, header_end + n_vertices):
        parts = lines[i].strip().split()
        if len(parts) >= 3:
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if has_radius and len(parts) > radius_col_idx:
                radius.append(float(parts[radius_col_idx]))
            else:
                radius.append(0.1)
    
    for i in range(header_end + n_vertices, header_end + n_vertices + n_edges):
        parts = lines[i].strip().split()
        if len(parts) >= 2:
            edges.append((int(parts[0]), int(parts[1])))
    
    return np.array(points, dtype=np.float64), np.array(radius, dtype=np.float64), edges


def load_held_out_filenames(split_csv: str) -> set:
    import pandas as pd
    df = pd.read_csv(split_csv)
    ply_names = set()
    for f in df['filenames'].tolist():
        ply_name = f.replace('.pt', '.ply').replace('.npz', '.ply')
        ply_names.add(ply_name)
    return ply_names


def load_all_vessels(directory: str, max_files: Optional[int] = None,
                     filter_filenames: Optional[set] = None) -> List[Dict]:
    ply_files = sorted(glob.glob(os.path.join(directory, "*.ply")))
    if filter_filenames is not None:
        ply_files = [p for p in ply_files if os.path.basename(p) in filter_filenames]
    if max_files:
        ply_files = ply_files[:max_files]
    
    vessels = []
    load_errors = []
    for path in tqdm(ply_files, desc="    Loading", leave=False, ncols=80):
        try:
            points, radius, edges = load_ply(path)
            if len(points) > 0:
                vessels.append({
                    'points': points,
                    'radius': radius,
                    'edges': edges,
                    'filename': os.path.basename(path)
                })
        except Exception as e:
            load_errors.append((os.path.basename(path), str(e)))
    
    if load_errors:
        print(f"    [!] Failed to load {len(load_errors)} files:")
        for fname, err in load_errors[:5]:
            print(f"      - {fname}: {err}")
        if len(load_errors) > 5:
            print(f"      ... and {len(load_errors) - 5} more")
    
    return vessels


def normalize_vessels_with_radius(vessels: List[Dict]) -> List[Dict]:
    normalized = []
    for v in vessels:
        pts = v['points'].copy()
        rad = v['radius'].copy()
        scale = 1.0
        
        if len(pts) > 0:
            centroid = np.mean(pts, axis=0)
            pts = pts - centroid
            max_extent = np.max(np.abs(pts)) + EPS
            scale = max_extent
            pts = pts / max_extent
            rad = rad / max_extent
        
        normalized.append({
            'points': pts,
            'radius': rad,
            'edges': v['edges'].copy() if 'edges' in v else [],
            'filename': v.get('filename', ''),
            'scale': scale
        })
    
    return normalized


def build_adjacency(edges: List[Tuple[int, int]], n_points: int) -> Dict[int, List[int]]:
    adj = defaultdict(list)
    for i, j in edges:
        if 0 <= i < n_points and 0 <= j < n_points:
            adj[i].append(j)
            adj[j].append(i)
    return dict(adj)


def find_bifurcations(adj: Dict[int, List[int]]) -> List[int]:
    return [n for n, neigh in adj.items() if len(neigh) >= 3]


def find_endpoints(adj: Dict[int, List[int]]) -> List[int]:
    return [n for n, neigh in adj.items() if len(neigh) == 1]


def order_points_by_edges(points: np.ndarray, edges: List[Tuple[int, int]]) -> np.ndarray:
    if len(edges) == 0 or len(points) == 0:
        return points
    
    adj = build_adjacency(edges, len(points))
    endpoints = find_endpoints(adj)
    
    if not endpoints:
        start = list(adj.keys())[0] if adj else 0
    else:
        start = endpoints[0]
    
    visited = set()
    ordered = []
    current = start
    
    while current is not None and current not in visited:
        visited.add(current)
        ordered.append(current)
        next_node = None
        for neighbor in adj.get(current, []):
            if neighbor not in visited:
                next_node = neighbor
                break
        current = next_node
    
    if len(ordered) < 2:
        return points
    
    return points[ordered]


def extract_segments(points: np.ndarray, edges: List[Tuple[int, int]]) -> List[np.ndarray]:
    if len(edges) == 0 or len(points) < 2:
        return [points] if len(points) >= 2 else []
    
    n_points = len(points)
    adj = build_adjacency(edges, n_points)
    
    if not adj:
        return [points] if len(points) >= 2 else []
    
    special_nodes = set()
    for n, neigh in adj.items():
        if len(neigh) != 2:
            special_nodes.add(n)
    
    if not special_nodes:
        ordered = order_points_by_edges(points, edges)
        return [ordered] if len(ordered) >= 2 else []
    
    segments = []
    visited_edges = set()
    
    for start in special_nodes:
        for neighbor in adj.get(start, []):
            edge_key = (min(start, neighbor), max(start, neighbor))
            if edge_key in visited_edges:
                continue
            
            segment_indices = [start]
            prev = start
            current = neighbor
            
            while current not in special_nodes:
                edge_key = (min(prev, current), max(prev, current))
                visited_edges.add(edge_key)
                segment_indices.append(current)
                
                next_node = None
                for n in adj.get(current, []):
                    if n != prev:
                        next_node = n
                        break
                
                if next_node is None:
                    break
                prev = current
                current = next_node
            
            edge_key = (min(prev, current), max(prev, current))
            visited_edges.add(edge_key)
            segment_indices.append(current)
            
            if len(segment_indices) >= 2:
                segments.append(points[segment_indices])
    
    if not segments:
        ordered = order_points_by_edges(points, edges)
        return [ordered] if len(ordered) >= 2 else []
    
    return segments


def count_segments(points: np.ndarray, edges: List[Tuple[int, int]]) -> int:
    segs = extract_segments(points, edges)
    return len(segs)


def extract_segments_with_indices(points: np.ndarray, edges: List[Tuple[int, int]]) -> List[Tuple[np.ndarray, List[int]]]:
    if len(edges) == 0 or len(points) < 2:
        if len(points) >= 2:
            return [(points, list(range(len(points))))]
        return []

    n_points = len(points)
    adj = build_adjacency(edges, n_points)

    if not adj:
        if len(points) >= 2:
            return [(points, list(range(len(points))))]
        return []

    special_nodes = set()
    for n, neigh in adj.items():
        if len(neigh) != 2:
            special_nodes.add(n)

    if not special_nodes:
        endpoints = find_endpoints(adj)
        start = endpoints[0] if endpoints else list(adj.keys())[0]
        ordered_indices = []
        visited = set()
        current = start
        while current is not None and current not in visited:
            visited.add(current)
            ordered_indices.append(current)
            next_node = None
            for n in adj.get(current, []):
                if n not in visited:
                    next_node = n
                    break
            current = next_node
        if len(ordered_indices) >= 2:
            return [(points[ordered_indices], ordered_indices)]
        return [(points, list(range(len(points))))]

    segments = []
    visited_edges = set()

    for start in special_nodes:
        for neighbor in adj.get(start, []):
            edge_key = (min(start, neighbor), max(start, neighbor))
            if edge_key in visited_edges:
                continue

            segment_indices = [start]
            prev = start
            current = neighbor

            while current not in special_nodes:
                edge_key = (min(prev, current), max(prev, current))
                visited_edges.add(edge_key)
                segment_indices.append(current)

                next_node = None
                for n in adj.get(current, []):
                    if n != prev:
                        next_node = n
                        break

                if next_node is None:
                    break
                prev = current
                current = next_node

            edge_key = (min(prev, current), max(prev, current))
            visited_edges.add(edge_key)
            segment_indices.append(current)

            if len(segment_indices) >= 2:
                segments.append((points[segment_indices], segment_indices))

    if not segments:
        ordered = order_points_by_edges(points, edges)
        return [(ordered, list(range(len(points))))] if len(ordered) >= 2 else []

    return segments


def extract_per_segment_attributes(vessel: Dict) -> List[Dict[str, float]]:
    pts = vessel['points']
    rad = vessel['radius']
    edges = vessel.get('edges', [])

    segments_with_idx = extract_segments_with_indices(pts, edges)
    result = []

    for seg_pts, seg_idx in segments_with_idx:
        if len(seg_pts) < 2:
            continue
        seg_len = compute_segment_length(seg_pts)
        if seg_len < EPS:
            continue

        attr = {'length': seg_len, 'tortuosity': compute_tortuosity(seg_pts)}

        if len(seg_pts) >= 3:
            attr['curvature'] = float(np.mean(compute_curvature_robust(seg_pts)))
        else:
            attr['curvature'] = 0.0

        if len(seg_pts) >= 4:
            attr['torsion'] = float(np.mean(compute_torsion_robust(seg_pts)))
        else:
            attr['torsion'] = 0.0

        valid_idx = [i for i in seg_idx if i < len(rad)]
        seg_radii = rad[valid_idx] if valid_idx else np.array([0.1])
        attr['radius'] = float(np.mean(seg_radii))

        result.append(attr)

    return result


def compute_curvature_robust(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.array([0.0])
    
    curvatures = []
    for i in range(1, len(points) - 1):
        p0, p1, p2 = points[i-1], points[i], points[i+1]
        v1 = p1 - p0
        v2 = p2 - p1
        len_v1 = norm(v1)
        len_v2 = norm(v2)
        
        if len_v1 < EPS or len_v2 < EPS:
            curvatures.append(0.0)
            continue
        
        cos_angle = np.clip(np.dot(v1, v2) / (len_v1 * len_v2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        ds = (len_v1 + len_v2) / 2
        curvature = angle / ds if ds > EPS else 0.0
        curvatures.append(curvature)
    
    return np.array(curvatures) if curvatures else np.array([0.0])


def compute_torsion_robust(points: np.ndarray) -> np.ndarray:
    if len(points) < 4:
        return np.array([0.0])
    
    torsions = []
    for i in range(1, len(points) - 2):
        p0, p1, p2, p3 = points[i-1], points[i], points[i+1], points[i+2]
        t1 = p1 - p0
        t2 = p2 - p1
        t3 = p3 - p2
        
        b1 = np.cross(t1, t2)
        b2 = np.cross(t2, t3)
        
        norm_b1 = norm(b1)
        norm_b2 = norm(b2)
        norm_t2 = norm(t2)
        
        if norm_b1 < EPS or norm_b2 < EPS or norm_t2 < EPS:
            torsions.append(0.0)
            continue
        
        b1_hat = b1 / norm_b1
        b2_hat = b2 / norm_b2
        
        cos_angle = np.clip(np.dot(b1_hat, b2_hat), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        torsion = angle / norm_t2
        torsion = np.clip(torsion, 0, 100)
        torsions.append(torsion)
    
    return np.array(torsions) if torsions else np.array([0.0])


def compute_jerk(points: np.ndarray) -> float:
    if len(points) < 4:
        return 0.0
    edge_lengths = norm(np.diff(points, axis=0), axis=1)
    arc_length = float(np.sum(edge_lengths))
    if arc_length < EPS:
        return 0.0
    ds = arc_length / (len(points) - 1)
    if ds < EPS:
        return 0.0
    velocity = np.diff(points, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)
    if len(jerk) == 0:
        return 0.0
    jerk_magnitudes = norm(jerk, axis=1) / (ds ** 3)
    return float(np.sqrt(np.mean(jerk_magnitudes ** 2)))


def compute_tortuosity(points: np.ndarray) -> float:
    if len(points) < 2:
        return 1.0
    chord_length = norm(points[-1] - points[0])
    if chord_length < EPS:
        return 1.0
    arc_length = np.sum(norm(np.diff(points, axis=0), axis=1))
    return float(arc_length / chord_length)


def compute_segment_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(norm(np.diff(points, axis=0), axis=1)))


def chamfer_distance_gpu_batch(pcs1: torch.Tensor, pcs2: torch.Tensor,
                                batch_size: int = CD_BATCH_SIZE) -> torch.Tensor:
    N1, P1, _ = pcs1.shape
    N2, P2, _ = pcs2.shape
    all_cd = torch.zeros(N1, N2, device=pcs1.device, dtype=torch.float32)
    
    for i in tqdm(range(N1), desc="      GPU CD", leave=False, ncols=80):
        pc1 = pcs1[i]
        for j_start in range(0, N2, batch_size):
            j_end = min(j_start + batch_size, N2)
            batch = pcs2[j_start:j_end]
            B = batch.shape[0]
            pc1_exp = pc1.unsqueeze(0).expand(B, -1, -1)
            dist_matrix = torch.cdist(pc1_exp, batch, p=2) ** 2
            min_1to2 = dist_matrix.min(dim=2)[0].mean(dim=1)
            min_2to1 = dist_matrix.min(dim=1)[0].mean(dim=1)
            all_cd[i, j_start:j_end] = min_1to2 + min_2to1
    
    return all_cd


def compute_min_chamfer_gpu(source: torch.Tensor, target: torch.Tensor,
                            batch_size: int = CD_BATCH_SIZE,
                            desc: str = "MinCD") -> torch.Tensor:
    N_src, N_tgt = len(source), len(target)
    min_dists = torch.full((N_src,), float('inf'), device=source.device, dtype=torch.float32)
    
    for i in tqdm(range(N_src), desc=f"      {desc}", leave=False, ncols=80):
        pc_src = source[i]
        batch_min = float('inf')
        for j_start in range(0, N_tgt, batch_size):
            j_end = min(j_start + batch_size, N_tgt)
            batch = target[j_start:j_end]
            B = batch.shape[0]
            pc_src_exp = pc_src.unsqueeze(0).expand(B, -1, -1)
            dist_matrix = torch.cdist(pc_src_exp, batch, p=2) ** 2
            min_1to2 = dist_matrix.min(dim=2)[0].mean(dim=1)
            min_2to1 = dist_matrix.min(dim=1)[0].mean(dim=1)
            cd_batch = min_1to2 + min_2to1
            batch_min = min(batch_min, cd_batch.min().item())
        min_dists[i] = batch_min
    
    return min_dists


def pairwise_chamfer_distance(pcs1: np.ndarray, pcs2: np.ndarray) -> np.ndarray:
    N1, N2 = len(pcs1), len(pcs2)
    print(f"      Computing {N1}x{N2}={N1*N2:,} pairs...")

    if TORCH_AVAILABLE and DEVICE is not None and DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        pcs1_t = torch.tensor(pcs1, dtype=torch.float32, device=DEVICE)
        pcs2_t = torch.tensor(pcs2, dtype=torch.float32, device=DEVICE)
        result = chamfer_distance_gpu_batch(pcs1_t, pcs2_t)
        ret = result.cpu().numpy()
        del pcs1_t, pcs2_t, result
        torch.cuda.empty_cache()
        return ret
    else:
        all_cd = np.zeros((N1, N2), dtype=np.float32)
        for i in tqdm(range(N1), desc="      CPU CD", leave=False):
            for j in range(N2):
                d = cdist(pcs1[i], pcs2[j], 'sqeuclidean')
                all_cd[i, j] = np.mean(np.min(d, axis=1)) + np.mean(np.min(d, axis=0))
        return all_cd


def compute_smoothness_metrics(vessels: List[Dict]) -> SmoothnessMetrics:
    all_curvatures = []
    curvature_variances = []
    jerks = []
    
    for v in vessels:
        segments = extract_segments(v['points'], v.get('edges', []))
        vessel_curvatures = []
        for seg in segments:
            if len(seg) < 3:
                continue
            curv = compute_curvature_robust(seg)
            vessel_curvatures.extend(curv.tolist())
            if len(seg) >= 4:
                jerks.append(compute_jerk(seg))
        
        all_curvatures.extend(vessel_curvatures)
        if len(vessel_curvatures) > 1:
            curvature_variances.append(np.var(vessel_curvatures))
    
    return SmoothnessMetrics(
        curvature_mean=float(np.mean(all_curvatures)) if all_curvatures else 0.0,
        curvature_std=float(np.std(all_curvatures)) if all_curvatures else 0.0,
        curvature_variance_mean=float(np.mean(curvature_variances)) if curvature_variances else 0.0,
        jerk_rms_mean=float(np.mean(jerks)) if jerks else 0.0,
        jerk_rms_std=float(np.std(jerks)) if jerks else 0.0,
        acceleration_smoothness=float(1.0 / (np.mean(jerks) + EPS)) if jerks else 0.0
    )


def compute_anatomical_metrics(vessels: List[Dict], gt_vessels: Optional[List[Dict]] = None) -> AnatomicalMetrics:
    branching_angles = []
    murrays_ratios = []
    tortuosities = []
    lengths = []
    all_curvatures = []
    all_torsions = []
    tapering_ratios = []
    n_bifurcations_total = 0
    
    for v in vessels:
        pts_orig = v['points']
        rad = v['radius']

        segments = extract_segments(pts_orig, v.get('edges', []))

        for seg in segments:
            if len(seg) < 2:
                continue
            seg_len = compute_segment_length(seg)
            if seg_len > EPS:
                lengths.append(seg_len)
                tortuosities.append(compute_tortuosity(seg))
            if len(seg) >= 3:
                all_curvatures.append(float(np.mean(compute_curvature_robust(seg))))
            if len(seg) >= 4:
                all_torsions.append(float(np.mean(compute_torsion_robust(seg))))
        
        if len(segments) > 0 and len(rad) >= 2:
            adj_tap = build_adjacency(v['edges'], len(pts_orig))
            endpoints_tap = find_endpoints(adj_tap)
            if len(endpoints_tap) >= 2:
                ep_radii = [(ep, rad[ep]) for ep in endpoints_tap if ep < len(rad)]
                if len(ep_radii) >= 2:
                    ep_radii.sort(key=lambda x: x[1], reverse=True)
                    r_proximal = ep_radii[0][1]
                    r_distal = ep_radii[-1][1]
                    if r_proximal > EPS:
                        tapering_ratios.append(r_distal / r_proximal)
        
        adj = build_adjacency(v['edges'], len(pts_orig))
        bifurcations = find_bifurcations(adj)
        n_bifurcations_total += len(bifurcations)
        
        for bif in bifurcations:
            neighbors = adj.get(bif, [])
            if len(neighbors) < 3 or bif >= len(pts_orig):
                continue
            
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    ni, nj = neighbors[i], neighbors[j]
                    if ni < len(pts_orig) and nj < len(pts_orig):
                        v1 = pts_orig[ni] - pts_orig[bif]
                        v2 = pts_orig[nj] - pts_orig[bif]
                        len_v1, len_v2 = norm(v1), norm(v2)
                        if len_v1 > EPS and len_v2 > EPS:
                            cos_a = np.clip(np.dot(v1, v2) / (len_v1 * len_v2), -1, 1)
                            branching_angles.append(np.degrees(np.arccos(cos_a)))
            
            neighbor_radii = [rad[n] for n in neighbors if n < len(rad) and rad[n] > EPS]
            if len(neighbor_radii) >= 3:
                sorted_radii = sorted(neighbor_radii, reverse=True)
                parent_r = sorted_radii[0]
                children_r = sorted_radii[1:]
                if parent_r > EPS:
                    ratio = sum(r**3 for r in children_r) / (parent_r**3)
                    murrays_ratios.append(ratio)
    
    gt_tortuosities, gt_curvatures, gt_torsions = [], [], []
    gt_lengths, gt_angles, gt_tapering = [], [], []
    
    if gt_vessels:
        for v in gt_vessels:
            pts_orig = v['points']
            rad = v['radius']
            segments = extract_segments(pts_orig, v.get('edges', []))

            for seg in segments:
                if len(seg) < 2:
                    continue
                seg_len = compute_segment_length(seg)
                if seg_len > EPS:
                    gt_lengths.append(seg_len)
                    gt_tortuosities.append(compute_tortuosity(seg))
                if len(seg) >= 3:
                    gt_curvatures.append(float(np.mean(compute_curvature_robust(seg))))
                if len(seg) >= 4:
                    gt_torsions.append(float(np.mean(compute_torsion_robust(seg))))
            
            if len(segments) > 0 and len(rad) >= 2:
                adj_tap = build_adjacency(v['edges'], len(pts_orig))
                endpoints_tap = find_endpoints(adj_tap)
                if len(endpoints_tap) >= 2:
                    ep_radii = [(ep, rad[ep]) for ep in endpoints_tap if ep < len(rad)]
                    if len(ep_radii) >= 2:
                        ep_radii.sort(key=lambda x: x[1], reverse=True)
                        r_prox = ep_radii[0][1]
                        r_dist = ep_radii[-1][1]
                        if r_prox > EPS:
                            gt_tapering.append(r_dist / r_prox)
            
            adj = build_adjacency(v['edges'], len(pts_orig))
            for bif in find_bifurcations(adj):
                neighbors = adj.get(bif, [])
                if len(neighbors) >= 3 and bif < len(pts_orig):
                    for i in range(len(neighbors)):
                        for j in range(i + 1, len(neighbors)):
                            ni, nj = neighbors[i], neighbors[j]
                            if ni < len(pts_orig) and nj < len(pts_orig):
                                v1 = pts_orig[ni] - pts_orig[bif]
                                v2 = pts_orig[nj] - pts_orig[bif]
                                l1, l2 = norm(v1), norm(v2)
                                if l1 > EPS and l2 > EPS:
                                    cos_a = np.clip(np.dot(v1, v2)/(l1*l2), -1, 1)
                                    gt_angles.append(np.degrees(np.arccos(cos_a)))
    
    def safe_wasserstein(a, b):
        if len(a) > 0 and len(b) > 0:
            return float(wasserstein_distance(a, b))
        return 0.0
    
    murray_compliance_strict = 0.0
    murray_compliance_loose = 0.0
    if murrays_ratios:
        murray_compliance_strict = float(np.mean([abs(r - 1.0) < 0.2 for r in murrays_ratios]))
        murray_compliance_loose = float(np.mean([abs(r - 1.0) < 0.5 for r in murrays_ratios]))
    
    return AnatomicalMetrics(
        branching_angle_mean=float(np.mean(branching_angles)) if branching_angles else 0.0,
        branching_angle_std=float(np.std(branching_angles)) if branching_angles else 0.0,
        branching_angle_wasserstein=safe_wasserstein(branching_angles, gt_angles),
        n_bifurcations_total=n_bifurcations_total,
        murrays_law_mean_ratio=float(np.mean(murrays_ratios)) if murrays_ratios else 0.0,
        murrays_law_std_ratio=float(np.std(murrays_ratios)) if murrays_ratios else 0.0,
        murrays_law_compliance_strict=murray_compliance_strict,
        murrays_law_compliance_loose=murray_compliance_loose,
        murrays_law_n_samples=len(murrays_ratios),
        segment_tortuosity_mean=float(np.mean(tortuosities)) if tortuosities else 0.0,
        segment_tortuosity_std=float(np.std(tortuosities)) if tortuosities else 0.0,
        segment_tortuosity_wasserstein=safe_wasserstein(tortuosities, gt_tortuosities),
        segment_length_mean=float(np.mean(lengths)) if lengths else 0.0,
        segment_length_wasserstein=safe_wasserstein(lengths, gt_lengths),
        curvature_mean=float(np.mean(all_curvatures)) if all_curvatures else 0.0,
        curvature_wasserstein=safe_wasserstein(all_curvatures, gt_curvatures),
        torsion_mean=float(np.mean(all_torsions)) if all_torsions else 0.0,
        torsion_wasserstein=safe_wasserstein(all_torsions, gt_torsions),
        tapering_ratio_mean=float(np.mean(tapering_ratios)) if tapering_ratios else 0.0,
        tapering_ratio_wasserstein=safe_wasserstein(tapering_ratios, gt_tapering)
    )


def compute_topological_metrics(vessels: List[Dict], gt_vessels: Optional[List[Dict]] = None) -> TopologicalMetrics:
    n_bifs, n_eps, n_segs = [], [], []
    
    for v in vessels:
        adj = build_adjacency(v['edges'], len(v['points']))
        n_bifs.append(len(find_bifurcations(adj)))
        n_eps.append(len(find_endpoints(adj)))
        n_segs.append(count_segments(v['points'], v['edges']))
    
    gt_bifs, gt_eps = [], []
    if gt_vessels:
        for v in gt_vessels:
            adj = build_adjacency(v['edges'], len(v['points']))
            gt_bifs.append(len(find_bifurcations(adj)))
            gt_eps.append(len(find_endpoints(adj)))
    
    def safe_wass(a, b):
        return float(wasserstein_distance(a, b)) if a and b else 0.0
    
    return TopologicalMetrics(
        n_bifurcations_mean=float(np.mean(n_bifs)) if n_bifs else 0.0,
        n_bifurcations_wasserstein=safe_wass(n_bifs, gt_bifs),
        n_endpoints_mean=float(np.mean(n_eps)) if n_eps else 0.0,
        n_endpoints_wasserstein=safe_wass(n_eps, gt_eps),
        n_segments_mean=float(np.mean(n_segs)) if n_segs else 0.0,
        tree_depth_mean=0.0
    )


def compute_gt_internal_distances(gt_pcs: np.ndarray, max_samples: int = 50) -> np.ndarray:
    n = min(len(gt_pcs), max_samples)
    gt_subset = gt_pcs[:n]
    internal_dists = []

    if TORCH_AVAILABLE and DEVICE is not None and DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        for i in range(n):
            pc1 = torch.tensor(gt_subset[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            for j in range(i + 1, n):
                pc2 = torch.tensor(gt_subset[j], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                dist = torch.cdist(pc1, pc2, p=2)[0] ** 2
                cd = dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()
                internal_dists.append(cd.item())
            del pc1
        torch.cuda.empty_cache()
    else:
        for i in range(n):
            for j in range(i + 1, n):
                d = cdist(gt_subset[i], gt_subset[j], 'sqeuclidean')
                cd = np.mean(np.min(d, axis=1)) + np.mean(np.min(d, axis=0))
                internal_dists.append(cd)
    
    return np.array(internal_dists)


def compute_precision_recall_multi_threshold(gen_pcs: np.ndarray, gt_pcs: np.ndarray) -> PrecisionRecallMetrics:
    if len(gen_pcs) == 0 or len(gt_pcs) == 0:
        return PrecisionRecallMetrics()

    print("      Computing GT internal distances for thresholds...")
    internal_dists = compute_gt_internal_distances(gt_pcs)

    if len(internal_dists) == 0:
        return PrecisionRecallMetrics()

    thresholds = {p: np.percentile(internal_dists, p) for p in PR_THRESHOLDS}
    print(f"      Thresholds: p95={thresholds[95]:.4f}, p50={thresholds[50]:.4f}, p25={thresholds[25]:.4f}")

    if TORCH_AVAILABLE and DEVICE is not None and DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        gen_t = torch.tensor(gen_pcs, dtype=torch.float32, device=DEVICE)
        gt_t = torch.tensor(gt_pcs, dtype=torch.float32, device=DEVICE)

        print("      Computing Precision (Gen->GT)...")
        gen_min_dists = compute_min_chamfer_gpu(gen_t, gt_t, desc="Precision").cpu().numpy()
        print("      Computing Recall (GT->Gen)...")
        gt_min_dists = compute_min_chamfer_gpu(gt_t, gen_t, desc="Recall").cpu().numpy()
        del gen_t, gt_t
        torch.cuda.empty_cache()
    else:
        gen_min_dists = np.array([np.min([
            np.mean(np.min(cdist(gen_pcs[i], gt_pcs[j], 'sqeuclidean'), axis=1)) +
            np.mean(np.min(cdist(gen_pcs[i], gt_pcs[j], 'sqeuclidean'), axis=0))
            for j in range(len(gt_pcs))
        ]) for i in tqdm(range(len(gen_pcs)), desc="      Precision", leave=False)])

        gt_min_dists = np.array([np.min([
            np.mean(np.min(cdist(gt_pcs[i], gen_pcs[j], 'sqeuclidean'), axis=1)) +
            np.mean(np.min(cdist(gt_pcs[i], gen_pcs[j], 'sqeuclidean'), axis=0))
            for j in range(len(gen_pcs))
        ]) for i in tqdm(range(len(gt_pcs)), desc="      Recall", leave=False)])

    results = {}
    for p in [99, 95, 90, 85, 75, 50, 25]:
        thresh = thresholds[p]
        precision = float(np.mean(gen_min_dists <= thresh))
        recall = float(np.mean(gt_min_dists <= thresh))
        f1 = 2 * precision * recall / (precision + recall + EPS)
        results[p] = {'precision': precision, 'recall': recall, 'f1': f1}

    return PrecisionRecallMetrics(
        precision_p99=results[99]['precision'],
        recall_p99=results[99]['recall'],
        f1_p99=results[99]['f1'],
        precision_p95=results[95]['precision'],
        recall_p95=results[95]['recall'],
        f1_p95=results[95]['f1'],
        precision_p90=results[90]['precision'],
        recall_p90=results[90]['recall'],
        f1_p90=results[90]['f1'],
        precision_p85=results[85]['precision'],
        recall_p85=results[85]['recall'],
        f1_p85=results[85]['f1'],
        precision_p75=results[75]['precision'],
        recall_p75=results[75]['recall'],
        f1_p75=results[75]['f1'],
        precision_p50=results[50]['precision'],
        recall_p50=results[50]['recall'],
        f1_p50=results[50]['f1'],
        precision_p25=results[25]['precision'],
        recall_p25=results[25]['recall'],
        f1_p25=results[25]['f1'],
        threshold_p95=float(thresholds[95]),
        threshold_p90=float(thresholds[90]),
        threshold_p50=float(thresholds[50]),
        threshold_p25=float(thresholds[25])
    )


def extract_all_attributes(vessels: List[Dict]) -> Dict[str, np.ndarray]:
    attrs = {k: [] for k in ['curvature', 'torsion', 'radius', 'length', 'tortuosity']}

    for v in vessels:
        seg_attrs = extract_per_segment_attributes(v)
        for sa in seg_attrs:
            for k in attrs:
                attrs[k].append(sa[k])

    return {k: np.array(v) for k, v in attrs.items()}


def compute_bounds_metrics(gen_vessels: List[Dict], gt_vessels: List[Dict]) -> AnatomicalBoundsMetrics:
    gen_attrs = extract_all_attributes(gen_vessels)
    gt_attrs = extract_all_attributes(gt_vessels)
    
    def in_bounds_rate(gen_vals, gt_vals, low_p=2.5, high_p=97.5):
        if len(gen_vals) == 0 or len(gt_vals) == 0:
            return 0.0, (0.0, 0.0)
        low = np.percentile(gt_vals, low_p)
        high = np.percentile(gt_vals, high_p)
        rate = float(np.mean((gen_vals >= low) & (gen_vals <= high)))
        return rate, (float(low), float(high))
    
    def ood_rate(gen_vals, gt_vals, low_p=1.0, high_p=99.0):
        if len(gen_vals) == 0 or len(gt_vals) == 0:
            return 0.0
        low = np.percentile(gt_vals, low_p)
        high = np.percentile(gt_vals, high_p)
        return float(np.mean((gen_vals < low) | (gen_vals > high)))
    
    curv_ib, curv_range = in_bounds_rate(gen_attrs['curvature'], gt_attrs['curvature'])
    tors_ib, _ = in_bounds_rate(gen_attrs['torsion'], gt_attrs['torsion'])
    rad_ib, rad_range = in_bounds_rate(gen_attrs['radius'], gt_attrs['radius'])
    len_ib, len_range = in_bounds_rate(gen_attrs['length'], gt_attrs['length'])
    tort_ib, _ = in_bounds_rate(gen_attrs['tortuosity'], gt_attrs['tortuosity'])
    
    curv_ood = ood_rate(gen_attrs['curvature'], gt_attrs['curvature'])
    tors_ood = ood_rate(gen_attrs['torsion'], gt_attrs['torsion'])
    rad_ood = ood_rate(gen_attrs['radius'], gt_attrs['radius'])
    len_ood = ood_rate(gen_attrs['length'], gt_attrs['length'])
    tort_ood = ood_rate(gen_attrs['tortuosity'], gt_attrs['tortuosity'])
    
    overall_ib = float(np.mean([curv_ib, tors_ib, rad_ib, len_ib, tort_ib]))
    overall_ood = float(np.mean([curv_ood, tors_ood, rad_ood, len_ood, tort_ood]))
    
    return AnatomicalBoundsMetrics(
        curvature_in_bounds=curv_ib,
        torsion_in_bounds=tors_ib,
        radius_in_bounds=rad_ib,
        length_in_bounds=len_ib,
        tortuosity_in_bounds=tort_ib,
        overall_in_bounds=overall_ib,
        curvature_ood=curv_ood,
        torsion_ood=tors_ood,
        radius_ood=rad_ood,
        length_ood=len_ood,
        tortuosity_ood=tort_ood,
        overall_ood=overall_ood,
        gt_curvature_range=curv_range,
        gt_radius_range=rad_range,
        gt_length_range=len_range
    )


def compute_legitimate_diversity_metrics(gen_vessels: List[Dict], gt_vessels: List[Dict]) -> LegitimateDiversityMetrics:
    gen_attrs = extract_all_attributes(gen_vessels)
    gt_attrs = extract_all_attributes(gt_vessels)
    
    def raw_diversity(vals):
        if len(vals) < 2:
            return 0.0
        mean = np.mean(vals)
        return float(np.std(vals) / (abs(mean) + EPS))
    
    def legit_diversity(gen_vals, gt_vals, low_p=2.5, high_p=97.5):
        if len(gen_vals) == 0 or len(gt_vals) == 0:
            return 0.0
        low = np.percentile(gt_vals, low_p)
        high = np.percentile(gt_vals, high_p)
        in_bounds = gen_vals[(gen_vals >= low) & (gen_vals <= high)]
        if len(in_bounds) < 2:
            return 0.0
        gt_range = high - low
        return float(np.std(in_bounds) / (gt_range + EPS))
    
    raw_divs = [raw_diversity(gen_attrs[k]) for k in ['curvature', 'radius', 'length'] if len(gen_attrs[k]) > 1]
    raw_div = float(np.mean(raw_divs)) if raw_divs else 0.0
    
    curv_legit = legit_diversity(gen_attrs['curvature'], gt_attrs['curvature'])
    rad_legit = legit_diversity(gen_attrs['radius'], gt_attrs['radius'])
    len_legit = legit_diversity(gen_attrs['length'], gt_attrs['length'])
    
    legit_divs = [d for d in [curv_legit, rad_legit, len_legit] if d > 0]
    legit_div = float(np.mean(legit_divs)) if legit_divs else 0.0
    
    legitimacy_ratio = legit_div / raw_div if raw_div > EPS else 0.0
    
    def vessel_features(v):
        pts = v['points']
        if len(pts) < 3:
            return None
        segments = extract_segments(pts, v.get('edges', []))
        extent = np.max(pts, axis=0) - np.min(pts, axis=0)
        total_length = sum(compute_segment_length(s) for s in segments if len(s) >= 2)
        all_curv = []
        for seg in segments:
            if len(seg) >= 3:
                all_curv.extend(compute_curvature_robust(seg).tolist())
        mean_curv = np.mean(all_curv) if all_curv else 0.0
        return np.array([extent[0], extent[1], extent[2], total_length, 
                        mean_curv, np.mean(v['radius'])])
    
    gen_feats = np.array([f for f in [vessel_features(v) for v in gen_vessels] if f is not None])
    gt_feats = np.array([f for f in [vessel_features(v) for v in gt_vessels] if f is not None])
    
    manifold_cov = 0.0
    if len(gen_feats) >= 20 and len(gt_feats) >= 20:
        gt_mean = gt_feats.mean(axis=0)
        gt_std = gt_feats.std(axis=0) + EPS
        gt_norm = (gt_feats - gt_mean) / gt_std
        gen_norm = (gen_feats - gt_mean) / gt_std
        
        n_clusters = min(20, len(gt_feats) // 3)
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=GLOBAL_SEED, n_init=10).fit(gt_norm)
            gt_clusters = set(kmeans.labels_)
            gen_clusters = set(kmeans.predict(gen_norm))
            manifold_cov = float(len(gt_clusters & gen_clusters) / len(gt_clusters))
    
    return LegitimateDiversityMetrics(
        raw_diversity=raw_div,
        legitimate_diversity=legit_div,
        legitimacy_ratio=float(legitimacy_ratio),
        manifold_coverage=manifold_cov,
        curvature_legit_div=curv_legit,
        radius_legit_div=rad_legit,
        length_legit_div=len_legit
    )


def sample_points_from_vessel(vessel: Dict, n_points: int = N_SAMPLE_POINTS) -> np.ndarray:
    pts = vessel['points']
    edges = vessel.get('edges', [])
    N = len(pts)
    
    if N == 0:
        return np.zeros((n_points, 3), dtype=np.float32)
    
    if len(edges) == 0 or N < 4:
        idx = np.random.choice(N, n_points, replace=(N < n_points))
        return pts[idx].astype(np.float32)
    
    valid_edges = []
    edge_lengths = []
    for i, j in edges:
        if i < N and j < N:
            length = norm(pts[i] - pts[j])
            if length > EPS:
                valid_edges.append((i, j))
                edge_lengths.append(length)
    
    if not valid_edges:
        idx = np.random.choice(N, n_points, replace=(N < n_points))
        return pts[idx].astype(np.float32)
    
    total_length = sum(edge_lengths)
    
    sampled = []
    for (i, j), length in zip(valid_edges, edge_lengths):
        n_samples = max(1, int(round(n_points * length / total_length)))
        t_values = np.random.uniform(0, 1, n_samples)
        for t in t_values:
            sampled.append(pts[i] * (1 - t) + pts[j] * t)
    
    sampled = np.array(sampled, dtype=np.float32)
    
    if len(sampled) > n_points:
        idx = np.random.choice(len(sampled), n_points, replace=False)
        return sampled[idx]
    elif len(sampled) < n_points:
        idx = np.random.choice(len(sampled), n_points, replace=True)
        return sampled[idx]
    
    return sampled


def jsd_between_point_clouds(sample_pcs: np.ndarray, ref_pcs: np.ndarray, 
                              resolution: int = JSD_RESOLUTION) -> float:
    if len(sample_pcs) == 0 or len(ref_pcs) == 0:
        return 0.0
    
    spacing = 2.0 / (resolution - 1)
    coords = []
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                pt = [i * spacing - 1.0, j * spacing - 1.0, k * spacing - 1.0]
                if norm(pt) <= 1.0:
                    coords.append(pt)
    grid = np.array(coords)
    
    nn = NearestNeighbors(n_neighbors=1).fit(grid)
    
    def occupancy(pcs):
        counts = np.zeros(len(grid))
        for pc in pcs:
            _, idx = nn.kneighbors(pc)
            for i in idx.flatten():
                counts[i] += 1
        return counts / (counts.sum() + EPS)
    
    P = occupancy(sample_pcs)
    Q = occupancy(ref_pcs)
    M = (P + Q) / 2
    
    jsd = 0.5 * np.sum(P * np.log((P + EPS) / (M + EPS))) + \
          0.5 * np.sum(Q * np.log((Q + EPS) / (M + EPS)))
    
    return float(jsd)


def compute_mmd(gen_pcs: np.ndarray, gt_pcs: np.ndarray, sigma: float = 0.1) -> float:
    if len(gen_pcs) == 0 or len(gt_pcs) == 0:
        return 0.0
    
    n_subsample = min(256, gen_pcs.shape[1])
    n_vessels_subsample = min(50, len(gen_pcs), len(gt_pcs))
    
    gen_sub_idx = np.random.choice(len(gen_pcs), n_vessels_subsample, replace=False)
    gt_sub_idx = np.random.choice(len(gt_pcs), n_vessels_subsample, replace=False)
    
    gen_features = np.array([gen_pcs[i][np.random.choice(gen_pcs.shape[1], n_subsample, replace=False)].flatten() 
                             for i in gen_sub_idx])
    gt_features = np.array([gt_pcs[i][np.random.choice(gt_pcs.shape[1], n_subsample, replace=False)].flatten() 
                            for i in gt_sub_idx])
    
    pairwise_dists = cdist(gen_features[:20], gt_features[:20], 'sqeuclidean')
    sigma_adapted = np.sqrt(np.median(pairwise_dists) / 2) + EPS
    
    if TORCH_AVAILABLE and DEVICE is not None and DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        gen_t = torch.tensor(gen_features, dtype=torch.float32, device=DEVICE)
        gt_t = torch.tensor(gt_features, dtype=torch.float32, device=DEVICE)

        def rbf(X, Y):
            dist_sq = torch.cdist(X, Y, p=2) ** 2
            return torch.exp(-dist_sq / (2 * sigma_adapted ** 2))

        K_gg = rbf(gen_t, gen_t)
        K_rr = rbf(gt_t, gt_t)
        K_gr = rbf(gen_t, gt_t)

        mmd = K_gg.mean() + K_rr.mean() - 2 * K_gr.mean()
        del gen_t, gt_t, K_gg, K_rr, K_gr
        torch.cuda.empty_cache()
        return float(max(0, mmd.item()))
    else:
        def rbf(X, Y):
            return np.exp(-cdist(X, Y, 'sqeuclidean') / (2 * sigma_adapted ** 2))
        
        K_gg = rbf(gen_features, gen_features)
        K_rr = rbf(gt_features, gt_features)
        K_gr = rbf(gen_features, gt_features)
        
        return float(max(0, K_gg.mean() + K_rr.mean() - 2 * K_gr.mean()))


def compute_sliced_wasserstein(gen_pcs: np.ndarray, gt_pcs: np.ndarray, n_proj: int = 100) -> float:
    if len(gen_pcs) == 0 or len(gt_pcs) == 0:
        return 0.0
    
    gen_all = np.vstack(gen_pcs)
    gt_all = np.vstack(gt_pcs)
    
    n_samples = min(5000, len(gen_all), len(gt_all))
    np.random.seed(GLOBAL_SEED)
    gen_sub = gen_all[np.random.choice(len(gen_all), n_samples, replace=False)]
    gt_sub = gt_all[np.random.choice(len(gt_all), n_samples, replace=False)]
    
    projs = np.random.randn(n_proj, 3)
    projs /= norm(projs, axis=1, keepdims=True)
    
    if TORCH_AVAILABLE and DEVICE is not None and DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        gen_t = torch.tensor(gen_sub, dtype=torch.float32, device=DEVICE)
        gt_t = torch.tensor(gt_sub, dtype=torch.float32, device=DEVICE)
        proj_t = torch.tensor(projs, dtype=torch.float32, device=DEVICE)

        swd = 0.0
        for p in proj_t:
            gen_proj = (gen_t @ p).sort()[0]
            gt_proj = (gt_t @ p).sort()[0]
            swd += torch.mean(torch.abs(gen_proj - gt_proj)).item()

        del gen_t, gt_t, proj_t
        torch.cuda.empty_cache()
        return float(swd / n_proj)
    else:
        swd = 0.0
        for p in projs:
            gen_proj = np.sort(gen_sub @ p)
            gt_proj = np.sort(gt_sub @ p)
            swd += np.mean(np.abs(gen_proj - gt_proj))
        
        return float(swd / n_proj)


def compute_1nn_accuracy(cd_rr: np.ndarray, cd_ss: np.ndarray, cd_rs: np.ndarray) -> float:
    if cd_rr.size == 0 or cd_ss.size == 0 or cd_rs.size == 0:
        return 0.5
    
    N_ref, N_sample = cd_rr.shape[0], cd_ss.shape[0]
    
    ref_correct = 0
    for i in range(N_ref):
        m_rr = cd_rr[i].copy()
        m_rr[i] = np.inf
        if np.min(m_rr) <= np.min(cd_rs[:, i]):
            ref_correct += 1
    
    sample_correct = 0
    for i in range(N_sample):
        m_ss = cd_ss[i].copy()
        m_ss[i] = np.inf
        if np.min(m_ss) <= np.min(cd_rs[i, :]):
            sample_correct += 1
    
    return float((ref_correct / N_ref + sample_correct / N_sample) / 2)


def compute_fidelity_metrics(gen_pcs: np.ndarray, gt_pcs: np.ndarray,
                              cd_gen_gt: np.ndarray, cd_gt_gt: np.ndarray,
                              cd_gen_gen: np.ndarray,
                              pr_metrics: PrecisionRecallMetrics) -> FidelityMetrics:
    if len(gen_pcs) == 0:
        return FidelityMetrics()

    min_per_gen = np.min(cd_gen_gt, axis=1)
    min_per_gt = np.min(cd_gen_gt, axis=0)
    all_mins = np.concatenate([min_per_gen, min_per_gt])

    cd_mean = float((np.mean(min_per_gen) + np.mean(min_per_gt)) / 2)
    cd_std = float(np.std(all_mins))
    cd_median = float(np.median(all_mins))

    print("      Computing JSD...")
    jsd = jsd_between_point_clouds(gen_pcs, gt_pcs)

    print("      Computing MMD (full PC)...")
    mmd = compute_mmd(gen_pcs, gt_pcs)

    print("      Computing Sliced Wasserstein...")
    swd = compute_sliced_wasserstein(gen_pcs, gt_pcs)

    print("      Computing 1-NN Accuracy...")
    one_nn = compute_1nn_accuracy(cd_gt_gt, cd_gen_gen, cd_gen_gt)

    one_nn_deviation = abs(one_nn - 0.5)
    if one_nn > 0.6:
        one_nn_interpretation = "gen_lacks_diversity"
    elif one_nn < 0.4:
        one_nn_interpretation = "gen_too_diverse"
    else:
        one_nn_interpretation = "good"

    thresh_coverage = pr_metrics.threshold_p50 if pr_metrics.threshold_p50 > 0 else np.median(min_per_gt)
    coverage = float(np.mean(min_per_gt <= thresh_coverage))
    density_score = float(np.mean(min_per_gen <= thresh_coverage))

    return FidelityMetrics(
        chamfer_distance_mean=cd_mean,
        chamfer_distance_std=cd_std,
        chamfer_distance_median=cd_median,
        jsd_3d=jsd,
        mmd_rbf=mmd,
        sliced_wasserstein=swd,
        one_nn_accuracy=one_nn,
        one_nn_deviation=one_nn_deviation,
        one_nn_interpretation=one_nn_interpretation,
        precision=pr_metrics.precision_p95,
        recall=pr_metrics.recall_p95,
        f1_score=pr_metrics.f1_p95,
        coverage=coverage,
        density_score=density_score
    )


def compute_radius_metrics(gen_vessels: List[Dict], gt_vessels: List[Dict]) -> RadiusMetrics:
    gen_seg_radii = []
    for v in gen_vessels:
        for sa in extract_per_segment_attributes(v):
            gen_seg_radii.append(sa['radius'])

    gt_seg_radii = []
    for v in gt_vessels:
        for sa in extract_per_segment_attributes(v):
            gt_seg_radii.append(sa['radius'])

    gen_seg_radii = np.array(gen_seg_radii)
    gt_seg_radii = np.array(gt_seg_radii)

    if len(gen_seg_radii) == 0 or len(gt_seg_radii) == 0:
        return RadiusMetrics()

    gt_low = np.percentile(gt_seg_radii, 2.5)
    gt_high = np.percentile(gt_seg_radii, 97.5)
    in_bounds = float(np.mean((gen_seg_radii >= gt_low) & (gen_seg_radii <= gt_high)))

    return RadiusMetrics(
        radius_mean=float(np.mean(gen_seg_radii)),
        radius_std=float(np.std(gen_seg_radii)),
        radius_wasserstein=float(wasserstein_distance(gen_seg_radii, gt_seg_radii)),
        radius_mean_diff=float(abs(np.mean(gen_seg_radii) - np.mean(gt_seg_radii))),
        radius_in_bounds=in_bounds,
        gt_radius_mean=float(np.mean(gt_seg_radii)),
        gt_radius_std=float(np.std(gt_seg_radii))
    )


def compute_absolute_metrics(gen_vessels_orig: List[Dict], gt_vessels_orig: List[Dict]) -> AbsoluteMetrics:
    gen_seg_lengths, gen_seg_radii, gen_tree_lengths = [], [], []
    for v in gen_vessels_orig:
        seg_attrs = extract_per_segment_attributes(v)
        if seg_attrs:
            gen_tree_lengths.append(sum(sa['length'] for sa in seg_attrs))
            for sa in seg_attrs:
                gen_seg_lengths.append(sa['length'])
                gen_seg_radii.append(sa['radius'])

    gt_seg_lengths, gt_seg_radii, gt_tree_lengths = [], [], []
    for v in gt_vessels_orig:
        seg_attrs = extract_per_segment_attributes(v)
        if seg_attrs:
            gt_tree_lengths.append(sum(sa['length'] for sa in seg_attrs))
            for sa in seg_attrs:
                gt_seg_lengths.append(sa['length'])
                gt_seg_radii.append(sa['radius'])

    gt_seg_len_mean = float(np.mean(gt_seg_lengths)) if gt_seg_lengths else 1.0
    gen_seg_len_mean = float(np.mean(gen_seg_lengths)) if gen_seg_lengths else 0.0
    seg_len_ratio = gen_seg_len_mean / (gt_seg_len_mean + EPS)

    SCALE_THRESHOLD = 0.1
    scale_valid = seg_len_ratio >= SCALE_THRESHOLD

    if not scale_valid:
        print(f"      [!] SCALE MISMATCH: gen_seg_len={gen_seg_len_mean:.2f} vs gt={gt_seg_len_mean:.2f}")
        print(f"        ratio={seg_len_ratio:.4f} < {SCALE_THRESHOLD} -> Normalized scale")
        return AbsoluteMetrics(scale_valid=False)

    gen_seg_rad_mean = float(np.mean(gen_seg_radii)) if gen_seg_radii else 0.0
    gt_seg_rad_mean = float(np.mean(gt_seg_radii)) if gt_seg_radii else 0.0
    gen_tree_len_mean = float(np.mean(gen_tree_lengths)) if gen_tree_lengths else 0.0
    gt_tree_len_mean = float(np.mean(gt_tree_lengths)) if gt_tree_lengths else 0.0

    seg_len_wass = float(wasserstein_distance(gen_seg_lengths, gt_seg_lengths)) if gen_seg_lengths and gt_seg_lengths else 0.0
    seg_rad_wass = float(wasserstein_distance(gen_seg_radii, gt_seg_radii)) if gen_seg_radii and gt_seg_radii else 0.0

    return AbsoluteMetrics(
        scale_valid=True,
        seg_length_mean_gen=gen_seg_len_mean,
        seg_length_mean_gt=gt_seg_len_mean,
        seg_length_ratio=seg_len_ratio,
        seg_length_wasserstein=seg_len_wass,
        seg_radius_mean_gen=gen_seg_rad_mean,
        seg_radius_mean_gt=gt_seg_rad_mean,
        seg_radius_ratio=gen_seg_rad_mean / (gt_seg_rad_mean + EPS),
        seg_radius_wasserstein=seg_rad_wass,
        tree_length_mean_gen=gen_tree_len_mean,
        tree_length_mean_gt=gt_tree_len_mean,
        tree_length_ratio=gen_tree_len_mean / (gt_tree_len_mean + EPS),
    )


def compute_intra_vessel_diversity(vessels: List[Dict]) -> IntraVesselDiversityMetrics:
    per_vessel_cvs = {k: [] for k in ['curvature', 'torsion', 'radius', 'length', 'tortuosity']}

    for v in vessels:
        seg_attrs = extract_per_segment_attributes(v)
        if len(seg_attrs) < 2:
            continue
        for k in per_vessel_cvs:
            vals = [sa[k] for sa in seg_attrs]
            mean_val = np.mean(vals)
            if abs(mean_val) > EPS:
                per_vessel_cvs[k].append(float(np.std(vals) / abs(mean_val)))

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    cv_means = {k: safe_mean(v) for k, v in per_vessel_cvs.items()}
    overall = float(np.mean(list(cv_means.values())))

    return IntraVesselDiversityMetrics(
        curvature_cv_mean=cv_means['curvature'],
        torsion_cv_mean=cv_means['torsion'],
        radius_cv_mean=cv_means['radius'],
        length_cv_mean=cv_means['length'],
        tortuosity_cv_mean=cv_means['tortuosity'],
        overall_cv_mean=overall,
    )


def compute_variance_ratio(gen_vessels: List[Dict], gt_vessels: List[Dict]) -> VarianceRatioMetrics:
    gen_attrs = extract_all_attributes(gen_vessels)
    gt_attrs = extract_all_attributes(gt_vessels)

    ratios = {}
    for k in ['curvature', 'torsion', 'radius', 'length', 'tortuosity']:
        gt_std = float(np.std(gt_attrs[k])) if len(gt_attrs[k]) > 1 else EPS
        gen_std = float(np.std(gen_attrs[k])) if len(gen_attrs[k]) > 1 else 0.0
        ratios[k] = gen_std / (gt_std + EPS)

    overall = float(np.mean(list(ratios.values())))

    return VarianceRatioMetrics(
        curvature_ratio=ratios['curvature'],
        torsion_ratio=ratios['torsion'],
        radius_ratio=ratios['radius'],
        length_ratio=ratios['length'],
        tortuosity_ratio=ratios['tortuosity'],
        overall_ratio=overall,
    )


def run_evaluation(gt_dir: str, baseline_dir: str, ours_dir: str,
                   output_dir: str, max_files: Optional[int] = None,
                   eval_mode: str = "full", split_csv: Optional[str] = None) -> Tuple[EvaluationResults, int, int, int]:
    print_header()
    set_global_seed(GLOBAL_SEED)
    os.makedirs(output_dir, exist_ok=True)

    results = EvaluationResults()
    total_start = time.time()

    print("[1/14] Loading vessel data...")
    gt_filter = None
    if eval_mode == "held-out":
        if not split_csv or not os.path.exists(split_csv):
            raise ValueError(f"--split-csv required for held-out mode: {split_csv}")
        gt_filter = load_held_out_filenames(split_csv)
        print(f"    [held-out mode] Filtering GT to {len(gt_filter)} held-out files")

    with Timer("Data loading"):
        gt_vessels_orig = load_all_vessels(gt_dir, max_files, filter_filenames=gt_filter)
        baseline_vessels_orig = load_all_vessels(baseline_dir, max_files) if baseline_dir else []
        ours_vessels_orig = load_all_vessels(ours_dir, max_files)
    
    n_gt = len(gt_vessels_orig)
    n_baseline = len(baseline_vessels_orig)
    n_ours = len(ours_vessels_orig)
    has_baseline = n_baseline > 0
    
    print(f"    Loaded: GT={n_gt}, Baseline={n_baseline}, Ours={n_ours}")
    
    print("\n[2/14] Normalizing coordinates AND radius (Scale-Fair)...")
    with Timer("Normalization"):
        gt_norm = normalize_vessels_with_radius(gt_vessels_orig)
        baseline_norm = normalize_vessels_with_radius(baseline_vessels_orig) if has_baseline else []
        ours_norm = normalize_vessels_with_radius(ours_vessels_orig)
    
    if n_gt > 0:
        gt_scales = [v['scale'] for v in gt_norm]
        print(f"    GT scale factors: mean={np.mean(gt_scales):.2f}, range=[{np.min(gt_scales):.2f}, {np.max(gt_scales):.2f}]")
    if has_baseline:
        bl_scales = [v['scale'] for v in baseline_norm]
        print(f"    Baseline scale factors: mean={np.mean(bl_scales):.4f}, range=[{np.min(bl_scales):.4f}, {np.max(bl_scales):.4f}]")
    if n_ours > 0:
        ours_scales = [v['scale'] for v in ours_norm]
        print(f"    Ours scale factors: mean={np.mean(ours_scales):.2f}, range=[{np.min(ours_scales):.2f}, {np.max(ours_scales):.2f}]")
    
    if has_baseline and n_gt > 0:
        gt_rad_norm = np.concatenate([v['radius'] for v in gt_norm if len(v['radius']) > 0])
        bl_rad_norm = np.concatenate([v['radius'] for v in baseline_norm if len(v['radius']) > 0])
        print(f"    [Scale Check] GT norm radius: mean={np.mean(gt_rad_norm):.6f}")
        print(f"    [Scale Check] Baseline norm radius: mean={np.mean(bl_rad_norm):.6f}")
        ratio = np.mean(gt_rad_norm) / (np.mean(bl_rad_norm) + EPS)
        if 0.5 < ratio < 2.0:
            print(f"    * Radius scales are comparable (ratio={ratio:.2f})")
        else:
            print(f"    [!] Radius scale mismatch remains (ratio={ratio:.2f}), check data format")
    
    
    print("\n[3/14] Computing Smoothness Metrics...")
    with Timer("Smoothness"):
        results.gt_smoothness = compute_smoothness_metrics(gt_norm)
        if has_baseline:
            results.baseline_smoothness = compute_smoothness_metrics(baseline_norm)
        results.ours_smoothness = compute_smoothness_metrics(ours_norm)
    
    print("\n[4/14] Computing Anatomical Metrics...")
    with Timer("Anatomical"):
        results.gt_anatomical = compute_anatomical_metrics(gt_norm, gt_norm)
        if has_baseline:
            results.baseline_anatomical = compute_anatomical_metrics(baseline_norm, gt_norm)
        results.ours_anatomical = compute_anatomical_metrics(ours_norm, gt_norm)
    
    print("\n[5/14] Computing Topological Metrics...")
    with Timer("Topological"):
        results.gt_topological = compute_topological_metrics(gt_norm, gt_norm)
        if has_baseline:
            results.baseline_topological = compute_topological_metrics(baseline_norm, gt_norm)
        results.ours_topological = compute_topological_metrics(ours_norm, gt_norm)
    
    print("\n[6/14] Computing Anatomical Bounds (Scale-Fair)...")
    with Timer("Anatomical Bounds"):
        if has_baseline:
            results.baseline_bounds = compute_bounds_metrics(baseline_norm, gt_norm)
        results.ours_bounds = compute_bounds_metrics(ours_norm, gt_norm)
    
    print("\n[7/14] Computing Legitimate Diversity (Scale-Fair)...")
    with Timer("Legitimate Diversity"):
        if has_baseline:
            results.baseline_legit_diversity = compute_legitimate_diversity_metrics(baseline_norm, gt_norm)
        results.ours_legit_diversity = compute_legitimate_diversity_metrics(ours_norm, gt_norm)
    
    print("\n[8/14] Sampling point clouds...")
    with Timer("Sampling"):
        gt_pcs = np.array([sample_points_from_vessel(v) for v in gt_norm])
        baseline_pcs = np.array([sample_points_from_vessel(v) for v in baseline_norm]) if has_baseline else np.array([])
        ours_pcs = np.array([sample_points_from_vessel(v) for v in ours_norm])
    
    gc.collect()
    if TORCH_AVAILABLE and DEVICE is not None and DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    print("\n[9/14] Computing Multi-Threshold Precision-Recall...")
    if has_baseline:
        print("    [9a] Baseline...")
        with Timer("Baseline PR"):
            results.baseline_precision_recall = compute_precision_recall_multi_threshold(baseline_pcs, gt_pcs)
    
    print("    [9b] Ours...")
    with Timer("Ours PR"):
        results.ours_precision_recall = compute_precision_recall_multi_threshold(ours_pcs, gt_pcs)
    
    print("\n[10/14] Computing Pairwise Chamfer Distance...")
    if has_baseline:
        print("    [10a] Baseline vs GT...")
        with Timer("Baseline vs GT"):
            cd_baseline_gt = pairwise_chamfer_distance(baseline_pcs, gt_pcs)
        print("    [10b] Baseline vs Baseline...")
        with Timer("Baseline vs Baseline"):
            cd_baseline_baseline = pairwise_chamfer_distance(baseline_pcs, baseline_pcs)
    else:
        cd_baseline_gt, cd_baseline_baseline = np.array([]), np.array([])
    
    print("    [10c] Ours vs GT...")
    with Timer("Ours vs GT"):
        cd_ours_gt = pairwise_chamfer_distance(ours_pcs, gt_pcs)
    
    print("    [10d] GT vs GT...")
    with Timer("GT vs GT"):
        cd_gt_gt = pairwise_chamfer_distance(gt_pcs, gt_pcs)
    
    print("    [10e] Ours vs Ours...")
    with Timer("Ours vs Ours"):
        cd_ours_ours = pairwise_chamfer_distance(ours_pcs, ours_pcs)
    
    print("\n[11/14] Computing Fidelity Metrics...")
    if has_baseline:
        print("    [11a] Baseline Fidelity...")
        with Timer("Baseline Fidelity"):
            results.baseline_fidelity = compute_fidelity_metrics(
                baseline_pcs, gt_pcs, cd_baseline_gt, cd_gt_gt, cd_baseline_baseline,
                results.baseline_precision_recall)
    
    print("    [11b] Ours Fidelity...")
    with Timer("Ours Fidelity"):
        results.ours_fidelity = compute_fidelity_metrics(
            ours_pcs, gt_pcs, cd_ours_gt, cd_gt_gt, cd_ours_ours,
            results.ours_precision_recall)
    
    print("    [11c] Radius (Scale-Fair)...")
    with Timer("Radius"):
        if has_baseline:
            results.baseline_radius = compute_radius_metrics(baseline_norm, gt_norm)
        results.ours_radius = compute_radius_metrics(ours_norm, gt_norm)
    
    print("\n[12/14] Computing Absolute Metrics (mm scale, per-segment)...")
    if has_baseline:
        print("    [12a] Baseline Absolute...")
        with Timer("Baseline Absolute"):
            results.baseline_absolute = compute_absolute_metrics(baseline_vessels_orig, gt_vessels_orig)
    print("    [12b] Ours Absolute...")
    with Timer("Ours Absolute"):
        results.ours_absolute = compute_absolute_metrics(ours_vessels_orig, gt_vessels_orig)

    print("\n[13/14] Computing Intra-Vessel Segment Diversity...")
    with Timer("Intra-Vessel Diversity"):
        results.gt_intra_diversity = compute_intra_vessel_diversity(gt_norm)
        if has_baseline:
            results.baseline_intra_diversity = compute_intra_vessel_diversity(baseline_norm)
        results.ours_intra_diversity = compute_intra_vessel_diversity(ours_norm)

    print("\n[14/14] Computing Variance Ratio (mode collapse detection)...")
    with Timer("Variance Ratio"):
        if has_baseline:
            results.baseline_variance_ratio = compute_variance_ratio(baseline_norm, gt_norm)
        results.ours_variance_ratio = compute_variance_ratio(ours_norm, gt_norm)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  Total Time: {format_time(total_elapsed)}")
    print(f"{'='*70}")
    
    return results, n_gt, n_baseline, n_ours


def print_results(results: EvaluationResults, n_gt: int, n_baseline: int, n_ours: int):
    has_baseline = n_baseline > 0
    
    print("\n" + "=" * 115)
    print("  EVALUATION RESULTS (Per-Segment Unification + Diversity)")
    print("=" * 115)
    print(f"\n  Samples: GT={n_gt}, Baseline={n_baseline}, Ours={n_ours}")
    
    def fmt(v, p=4):
        if v is None:
            return "N/A"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, tuple):
            return f"({v[0]:.3f}, {v[1]:.3f})"
        if v == 0:
            return "0.0000"
        if abs(v) < 0.0001:
            return f"{v:.2e}"
        return f"{v:.{p}f}"
    
    def win(b, o, lower=True):
        if not has_baseline:
            return "-"
        if lower:
            return "Ours *" if o < b else ("Baseline" if b < o else "Tie")
        return "Ours *" if o > b else ("Baseline" if b > o else "Tie")
    
    print("\n" + "-" * 115)
    print("  [1] PRECISION-RECALL (Multi-Threshold)")
    print("-" * 115)
    b_pr, o_pr = results.baseline_precision_recall, results.ours_precision_recall
    print(f"  {'Threshold':<20} {'Baseline Prec/Rec/F1':>30} {'Ours Prec/Rec/F1':>30} {'Winner':>15}")
    print(f"  {'-'*20} {'-'*30} {'-'*30} {'-'*15}")
    for p in [99, 95, 90, 85, 75, 50, 25]:
        bp = getattr(b_pr, f'precision_p{p}')
        br = getattr(b_pr, f'recall_p{p}')
        bf = getattr(b_pr, f'f1_p{p}')
        op = getattr(o_pr, f'precision_p{p}')
        ore = getattr(o_pr, f'recall_p{p}')
        of = getattr(o_pr, f'f1_p{p}')
        winner = win(bf, of, lower=False)
        print(f"  {'p' + str(p):<20} {fmt(bp)+'/'+fmt(br)+'/'+fmt(bf):>30} {fmt(op)+'/'+fmt(ore)+'/'+fmt(of):>30} {winner:>15}")
    print(f"  {'Threshold p50':<20} {fmt(b_pr.threshold_p50):>30} {fmt(o_pr.threshold_p50):>30}")
    print(f"  {'Threshold p25':<20} {fmt(b_pr.threshold_p25):>30} {fmt(o_pr.threshold_p25):>30}")
    
    print("\n" + "-" * 115)
    print("  [2] ANATOMICAL BOUNDS (Scale-Fair)")
    print("-" * 115)
    b_ab, o_ab = results.baseline_bounds, results.ours_bounds
    print(f"  {'Metric':<30} {'Baseline':>15} {'Ours':>15} {'Winner':>15} {'GT Range':>25}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*15} {'-'*25}")
    print(f"  {'Curvature In-Bounds (^)':<30} {fmt(b_ab.curvature_in_bounds):>15} {fmt(o_ab.curvature_in_bounds):>15} {win(b_ab.curvature_in_bounds, o_ab.curvature_in_bounds, False):>15} {fmt(o_ab.gt_curvature_range):>25}")
    print(f"  {'Torsion In-Bounds (^)':<30} {fmt(b_ab.torsion_in_bounds):>15} {fmt(o_ab.torsion_in_bounds):>15} {win(b_ab.torsion_in_bounds, o_ab.torsion_in_bounds, False):>15}")
    print(f"  {'Radius In-Bounds (^)':<30} {fmt(b_ab.radius_in_bounds):>15} {fmt(o_ab.radius_in_bounds):>15} {win(b_ab.radius_in_bounds, o_ab.radius_in_bounds, False):>15} {fmt(o_ab.gt_radius_range):>25}")
    print(f"  {'Length In-Bounds (^)':<30} {fmt(b_ab.length_in_bounds):>15} {fmt(o_ab.length_in_bounds):>15} {win(b_ab.length_in_bounds, o_ab.length_in_bounds, False):>15} {fmt(o_ab.gt_length_range):>25}")
    print(f"  {'Tortuosity In-Bounds (^)':<30} {fmt(b_ab.tortuosity_in_bounds):>15} {fmt(o_ab.tortuosity_in_bounds):>15} {win(b_ab.tortuosity_in_bounds, o_ab.tortuosity_in_bounds, False):>15}")
    print(f"  {'Overall In-Bounds (^)':<30} {fmt(b_ab.overall_in_bounds):>15} {fmt(o_ab.overall_in_bounds):>15} {win(b_ab.overall_in_bounds, o_ab.overall_in_bounds, False):>15}")
    print(f"  {'Overall OOD Rate (v)':<30} {fmt(b_ab.overall_ood):>15} {fmt(o_ab.overall_ood):>15} {win(b_ab.overall_ood, o_ab.overall_ood):>15}")
    
    print("\n" + "-" * 115)
    print("  [3] LEGITIMATE DIVERSITY")
    print("-" * 115)
    b_ld, o_ld = results.baseline_legit_diversity, results.ours_legit_diversity
    print(f"  {'Raw Diversity':<30} {fmt(b_ld.raw_diversity):>15} {fmt(o_ld.raw_diversity):>15} {'(reference)':>15}")
    print(f"  {'Legitimate Diversity (^)':<30} {fmt(b_ld.legitimate_diversity):>15} {fmt(o_ld.legitimate_diversity):>15} {win(b_ld.legitimate_diversity, o_ld.legitimate_diversity, False):>15}")
    print(f"  {'Legitimacy Ratio (^)':<30} {fmt(b_ld.legitimacy_ratio):>15} {fmt(o_ld.legitimacy_ratio):>15} {win(b_ld.legitimacy_ratio, o_ld.legitimacy_ratio, False):>15}")
    print(f"  {'Manifold Coverage (^)':<30} {fmt(b_ld.manifold_coverage):>15} {fmt(o_ld.manifold_coverage):>15} {win(b_ld.manifold_coverage, o_ld.manifold_coverage, False):>15}")
    
    print("\n" + "-" * 115)
    print("  [4] GEOMETRIC FIDELITY")
    print("-" * 115)
    b_f, o_f = results.baseline_fidelity, results.ours_fidelity
    print(f"  {'Chamfer Distance (v)':<30} {fmt(b_f.chamfer_distance_mean,6):>15} {fmt(o_f.chamfer_distance_mean,6):>15} {win(b_f.chamfer_distance_mean, o_f.chamfer_distance_mean):>15}")
    print(f"  {'JSD 3D (v)':<30} {fmt(b_f.jsd_3d):>15} {fmt(o_f.jsd_3d):>15} {win(b_f.jsd_3d, o_f.jsd_3d):>15}")
    print(f"  {'MMD (RBF, full PC) (v)':<30} {fmt(b_f.mmd_rbf,6):>15} {fmt(o_f.mmd_rbf,6):>15} {win(b_f.mmd_rbf, o_f.mmd_rbf):>15}")
    print(f"  {'Sliced Wasserstein (v)':<30} {fmt(b_f.sliced_wasserstein,6):>15} {fmt(o_f.sliced_wasserstein,6):>15} {win(b_f.sliced_wasserstein, o_f.sliced_wasserstein):>15}")
    print(f"  {'1-NN Accuracy (->0.5)':<30} {fmt(b_f.one_nn_accuracy):>15} {fmt(o_f.one_nn_accuracy):>15}")
    print(f"  {'  1-NN Deviation (v)':<30} {fmt(b_f.one_nn_deviation):>15} {fmt(o_f.one_nn_deviation):>15} {win(b_f.one_nn_deviation, o_f.one_nn_deviation):>15}")
    print(f"  {'  1-NN Interpretation':<30} {b_f.one_nn_interpretation:>15} {o_f.one_nn_interpretation:>15}")
    print(f"  {'Coverage (^)':<30} {fmt(b_f.coverage):>15} {fmt(o_f.coverage):>15} {win(b_f.coverage, o_f.coverage, False):>15}")
    print(f"  {'Density Score (^)':<30} {fmt(b_f.density_score):>15} {fmt(o_f.density_score):>15} {win(b_f.density_score, o_f.density_score, False):>15}")
    
    print("\n" + "-" * 115)
    print("  [5] ANATOMICAL PLAUSIBILITY")
    print("-" * 115)
    gt_a, b_a, o_a = results.gt_anatomical, results.baseline_anatomical, results.ours_anatomical
    print(f"  {'Tortuosity Wass (v)':<30} {fmt(b_a.segment_tortuosity_wasserstein):>15} {fmt(o_a.segment_tortuosity_wasserstein):>15} {win(b_a.segment_tortuosity_wasserstein, o_a.segment_tortuosity_wasserstein):>15}")
    print(f"  {'Curvature Wass (v)':<30} {fmt(b_a.curvature_wasserstein):>15} {fmt(o_a.curvature_wasserstein):>15} {win(b_a.curvature_wasserstein, o_a.curvature_wasserstein):>15}")
    print(f"  {'Torsion Wass (v)':<30} {fmt(b_a.torsion_wasserstein):>15} {fmt(o_a.torsion_wasserstein):>15} {win(b_a.torsion_wasserstein, o_a.torsion_wasserstein):>15}")
    print(f"  {'Branching Angle Wass (v)':<30} {fmt(b_a.branching_angle_wasserstein):>15} {fmt(o_a.branching_angle_wasserstein):>15} {win(b_a.branching_angle_wasserstein, o_a.branching_angle_wasserstein):>15}")
    print(f"  {'Tapering Wass (v)':<30} {fmt(b_a.tapering_ratio_wasserstein):>15} {fmt(o_a.tapering_ratio_wasserstein):>15} {win(b_a.tapering_ratio_wasserstein, o_a.tapering_ratio_wasserstein):>15}")
    print(f"  {'Murray Compliance (strict) (^)':<30} {fmt(b_a.murrays_law_compliance_strict):>15} {fmt(o_a.murrays_law_compliance_strict):>15} {win(b_a.murrays_law_compliance_strict, o_a.murrays_law_compliance_strict, False):>15}")
    print(f"  {'Murray Compliance (loose) (^)':<30} {fmt(b_a.murrays_law_compliance_loose):>15} {fmt(o_a.murrays_law_compliance_loose):>15} {win(b_a.murrays_law_compliance_loose, o_a.murrays_law_compliance_loose, False):>15}")
    print(f"  {'  (n_samples)':<30} {fmt(b_a.murrays_law_n_samples):>15} {fmt(o_a.murrays_law_n_samples):>15}")
    
    print("\n" + "-" * 115)
    print("  [6] RADIUS (normalized scale)")
    print("-" * 115)
    b_r, o_r = results.baseline_radius, results.ours_radius
    print(f"  {'Radius Wass (v)':<30} {fmt(b_r.radius_wasserstein,6):>15} {fmt(o_r.radius_wasserstein,6):>15} {win(b_r.radius_wasserstein, o_r.radius_wasserstein):>15}")
    print(f"  {'Radius Mean Diff (v)':<30} {fmt(b_r.radius_mean_diff,6):>15} {fmt(o_r.radius_mean_diff,6):>15} {win(b_r.radius_mean_diff, o_r.radius_mean_diff):>15}")
    print(f"  {'Radius In-Bounds (^)':<30} {fmt(b_r.radius_in_bounds):>15} {fmt(o_r.radius_in_bounds):>15} {win(b_r.radius_in_bounds, o_r.radius_in_bounds, False):>15}")
    print(f"  {'  (GT mean+-std)':<30} {fmt(b_r.gt_radius_mean)+'+-'+fmt(b_r.gt_radius_std):>15}")
    
    print("\n" + "-" * 115)
    print("  [7] ABSOLUTE METRICS (mm scale, per-segment, unpaired)")
    print("-" * 115)
    b_abs, o_abs = results.baseline_absolute, results.ours_absolute
    b_valid, o_valid = b_abs.scale_valid, o_abs.scale_valid
    print(f"  {'Scale Validity':<30} {'VALID' if b_valid else 'INVALID':>15} {'VALID' if o_valid else 'INVALID':>15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*15}")
    print(f"  {'Metric':<30} {'Baseline':>15} {'Ours':>15} {'Ideal':>15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*15}")

    def fmt_valid(val, p, valid):
        return fmt(val, p) if valid else "N/A"

    print(f"  {'Seg Length Ratio (->1.0)':<30} {fmt_valid(b_abs.seg_length_ratio, 4, b_valid):>15} {fmt_valid(o_abs.seg_length_ratio, 4, o_valid):>15} {'1.0000':>15}")
    print(f"  {'Seg Length Wass (v)':<30} {fmt_valid(b_abs.seg_length_wasserstein, 4, b_valid):>15} {fmt_valid(o_abs.seg_length_wasserstein, 4, o_valid):>15} {'0.0000':>15}")
    print(f"  {'Seg Radius Ratio (->1.0)':<30} {fmt_valid(b_abs.seg_radius_ratio, 4, b_valid):>15} {fmt_valid(o_abs.seg_radius_ratio, 4, o_valid):>15} {'1.0000':>15}")
    print(f"  {'Seg Radius Wass (v)':<30} {fmt_valid(b_abs.seg_radius_wasserstein, 4, b_valid):>15} {fmt_valid(o_abs.seg_radius_wasserstein, 4, o_valid):>15} {'0.0000':>15}")
    print(f"  {'Tree Length Ratio (->1.0)':<30} {fmt_valid(b_abs.tree_length_ratio, 4, b_valid):>15} {fmt_valid(o_abs.tree_length_ratio, 4, o_valid):>15} {'1.0000':>15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")
    print(f"  {'GT Seg Length Mean (mm)':<30} {fmt(o_abs.seg_length_mean_gt, 2):>15}")
    print(f"  {'GT Seg Radius Mean (mm)':<30} {fmt(o_abs.seg_radius_mean_gt, 4):>15}")
    if not b_valid:
        print(f"\n  [!] Baseline uses normalized coordinates -> absolute metrics invalid.")

    print("\n" + "-" * 115)
    print("  [8] INTRA-VESSEL SEGMENT DIVERSITY (within-vessel CV)")
    print("-" * 115)
    gt_iv = results.gt_intra_diversity
    b_iv = results.baseline_intra_diversity
    o_iv = results.ours_intra_diversity
    print(f"  {'Attribute CV':<30} {'GT':>15} {'Baseline':>15} {'Ours':>15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*15}")
    for attr in ['curvature', 'torsion', 'radius', 'length', 'tortuosity']:
        gt_val = getattr(gt_iv, f'{attr}_cv_mean')
        b_val_iv = getattr(b_iv, f'{attr}_cv_mean')
        o_val_iv = getattr(o_iv, f'{attr}_cv_mean')
        print(f"  {attr.capitalize()+' CV':30} {fmt(gt_val):>15} {fmt(b_val_iv):>15} {fmt(o_val_iv):>15}")
    print(f"  {'Overall CV':<30} {fmt(gt_iv.overall_cv_mean):>15} {fmt(b_iv.overall_cv_mean):>15} {fmt(o_iv.overall_cv_mean):>15}")
    print(f"  (Higher = more structural diversity within each vessel)")

    print("\n" + "-" * 115)
    print("  [9] VARIANCE RATIO (gen_std / gt_std, ideal=1.0)")
    print("-" * 115)
    b_vr, o_vr = results.baseline_variance_ratio, results.ours_variance_ratio
    print(f"  {'Attribute':<30} {'Baseline':>15} {'Ours':>15} {'Interpretation':>20}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*20}")
    for attr in ['curvature', 'torsion', 'radius', 'length', 'tortuosity']:
        b_val_vr = getattr(b_vr, f'{attr}_ratio')
        o_val_vr = getattr(o_vr, f'{attr}_ratio')
        def interp_vr(v):
            if v < 0.5: return "mode collapse"
            elif v < 0.8: return "low diversity"
            elif v <= 1.2: return "good"
            else: return "over-dispersed"
        print(f"  {attr.capitalize():<30} {fmt(b_val_vr):>15} {fmt(o_val_vr):>15} {interp_vr(o_val_vr):>20}")
    print(f"  {'Overall':<30} {fmt(b_vr.overall_ratio):>15} {fmt(o_vr.overall_ratio):>15}")

    print("\n" + "=" * 115)
    print("  SUMMARY")
    print("=" * 115)

    metrics_comparison = [
        ("Precision (p95)", b_pr.precision_p95, o_pr.precision_p95, False),
        ("Precision (p50)", b_pr.precision_p50, o_pr.precision_p50, False),
        ("Recall (p95)", b_pr.recall_p95, o_pr.recall_p95, False),
        ("In-Bounds", b_ab.overall_in_bounds, o_ab.overall_in_bounds, False),
        ("OOD Rate", b_ab.overall_ood, o_ab.overall_ood, True),
        ("Legit Diversity", b_ld.legitimate_diversity, o_ld.legitimate_diversity, False),
        ("Manifold Coverage", b_ld.manifold_coverage, o_ld.manifold_coverage, False),
        ("Chamfer", b_f.chamfer_distance_mean, o_f.chamfer_distance_mean, True),
        ("1-NN Deviation", b_f.one_nn_deviation, o_f.one_nn_deviation, True),
        ("Tortuosity Wass", b_a.segment_tortuosity_wasserstein, o_a.segment_tortuosity_wasserstein, True),
        ("Curvature Wass", b_a.curvature_wasserstein, o_a.curvature_wasserstein, True),
    ]

    ours_wins = 0
    baseline_wins = 0
    ties = 0

    for name, b_val, o_val, lower_better in metrics_comparison:
        if has_baseline:
            if lower_better:
                if o_val < b_val: ours_wins += 1
                elif b_val < o_val: baseline_wins += 1
                else: ties += 1
            else:
                if o_val > b_val: ours_wins += 1
                elif b_val > o_val: baseline_wins += 1
                else: ties += 1

    print(f"\n  Key metrics: Ours {ours_wins} wins, Baseline {baseline_wins} wins, Tie {ties}")
    print(f"\n  Key design choices:")
    print(f"    1. Per-segment unification: all 5 attributes at segment granularity")
    print(f"    2. Intra-vessel diversity: GT CV={fmt(gt_iv.overall_cv_mean)}, Ours={fmt(o_iv.overall_cv_mean)}")
    print(f"    3. Variance ratio: Ours={fmt(o_vr.overall_ratio)} (ideal=1.0)")
    print(f"    4. Absolute metrics: per-segment, unpaired distribution comparison")

    print("\n" + "=" * 115)


def save_results(results: EvaluationResults, output_dir: str,
                 n_gt: int, n_baseline: int, n_ours: int,
                 gt_dir: str, baseline_dir: str, ours_dir: str,
                 eval_mode: str = "full", split_csv: Optional[str] = None):
    json_path = os.path.join(output_dir, "evaluation_results_v6.json")

    data = {
        "metadata": {
            "version": "6.0",
            "gpu": GPU_NAME,
            "n_gt": n_gt,
            "n_baseline": n_baseline,
            "n_ours": n_ours,
            "timestamp": datetime.now().isoformat(),
            "eval_mode": eval_mode,
            "split_csv": split_csv if split_csv else "N/A",
            "gt_dir": gt_dir,
            "baseline_dir": baseline_dir,
            "ours_dir": ours_dir,
            "features": [
                "Per-segment unification: all 5 attributes at segment granularity",
                "Intra-vessel segment diversity: within-vessel CV for mode collapse",
                "Variance ratio: gen_std/gt_std per attribute (ideal=1.0)",
                "Absolute metrics: per-segment, unpaired distribution comparison",
                "Radius metrics: per-segment mean radii",
            ],
        },
        "precision_recall": {
            "baseline": asdict(results.baseline_precision_recall),
            "ours": asdict(results.ours_precision_recall)
        },
        "anatomical_bounds": {
            "baseline": asdict(results.baseline_bounds),
            "ours": asdict(results.ours_bounds)
        },
        "legitimate_diversity": {
            "baseline": asdict(results.baseline_legit_diversity),
            "ours": asdict(results.ours_legit_diversity)
        },
        "fidelity": {
            "baseline": asdict(results.baseline_fidelity),
            "ours": asdict(results.ours_fidelity)
        },
        "anatomical": {
            "gt": asdict(results.gt_anatomical),
            "baseline": asdict(results.baseline_anatomical),
            "ours": asdict(results.ours_anatomical)
        },
        "topological": {
            "gt": asdict(results.gt_topological),
            "baseline": asdict(results.baseline_topological),
            "ours": asdict(results.ours_topological)
        },
        "smoothness": {
            "gt": asdict(results.gt_smoothness),
            "baseline": asdict(results.baseline_smoothness),
            "ours": asdict(results.ours_smoothness)
        },
        "radius": {
            "baseline": asdict(results.baseline_radius),
            "ours": asdict(results.ours_radius)
        },
        "absolute_metrics": {
            "baseline": asdict(results.baseline_absolute),
            "ours": asdict(results.ours_absolute)
        },
        "intra_vessel_diversity": {
            "gt": asdict(results.gt_intra_diversity),
            "baseline": asdict(results.baseline_intra_diversity),
            "ours": asdict(results.ours_intra_diversity)
        },
        "variance_ratio": {
            "baseline": asdict(results.baseline_variance_ratio),
            "ours": asdict(results.ours_variance_ratio)
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n  * Results saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Coronary Vessel Generation Evaluation (Per-Segment Unification)")
    parser.add_argument("--gt-dir", type=str, required=True, help="GT PLY directory")
    parser.add_argument("--baseline-dir", type=str, default="", help="Baseline PLY directory")
    parser.add_argument("--ours-dir", type=str, required=True, help="Ours PLY directory")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results_v6")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--eval-mode", type=str, choices=["full", "held-out"], default="full",
                        help="Evaluation mode: 'full' uses all GT, 'held-out' uses only val split")
    parser.add_argument("--split-csv", type=str, default=None,
                        help="Path to val_filename.csv (required if eval-mode is held-out)")
    args = parser.parse_args()

    results, n_gt, n_baseline, n_ours = run_evaluation(
        args.gt_dir, args.baseline_dir, args.ours_dir,
        args.output_dir, args.max_files,
        eval_mode=args.eval_mode, split_csv=args.split_csv
    )

    print_results(results, n_gt, n_baseline, n_ours)
    save_results(results, args.output_dir, n_gt, n_baseline, n_ours,
                 args.gt_dir, args.baseline_dir, args.ours_dir,
                 eval_mode=args.eval_mode, split_csv=args.split_csv)
    
    print("\n  * Evaluation Complete!")
    print("=" * 115)


if __name__ == "__main__":
    main()
