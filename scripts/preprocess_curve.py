
import os
import sys
import glob
import argparse
import json
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial import chebyshev as cheb
from tqdm import tqdm
import pandas as pd

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


NUM_SAMPLES = 2048
CHEBYSHEV_K = 48
CHEBYSHEV_K_RADIUS = 6
COEFF_STATISTICS_FILENAME = "coeff_statistics.npz"
FALLBACK_THETA0_HINT_WEIGHTS = {"weights": [1.0, 1.0]}


def get_hints_per_segment_from_tree_pt(root_dir: str, dataset: str) -> int:
    tree_pt_path = os.path.join(root_dir, 'data', f'{dataset}_tree.pt')

    if not os.path.exists(tree_pt_path):
        print(f"  [WARNING] {tree_pt_path} not found. Using default hints_per_segment=2")
        return 2

    try:
        tree_data = torch.load(tree_pt_path, weights_only=False)
        hint_stats = tree_data.get('hint_stats', {})
        hints_per_segment = hint_stats.get('hints_per_segment', 2)
        print(f"  [INFO] Loaded hints_per_segment={hints_per_segment} from {tree_pt_path}")
        return hints_per_segment
    except Exception as e:
        print(f"  [WARNING] Failed to load {tree_pt_path}: {e}. Using default hints_per_segment=2")
        return 2


@dataclass
class SkeletonNode:
    idx: int
    pos: np.ndarray
    degree: int
    original_indices: List[int]
    is_hint: bool = False


@dataclass
class SkeletonEdge:
    start_node_idx: int
    end_node_idx: int
    path_indices: List[int]
    radius_values: np.ndarray
    parent_edge_idx: int = -1
    curve_points: Optional[np.ndarray] = None


def compute_arc_lengths(curve: np.ndarray) -> np.ndarray:
    if len(curve) < 2:
        return np.zeros(len(curve), dtype=np.float64)
    diffs = np.diff(curve, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])


def resample_curve(curve: np.ndarray, num_points: int) -> Tuple[np.ndarray, float]:
    if len(curve) < 2:
        return curve.copy(), 0.0
    arc_lengths = compute_arc_lengths(curve)
    L = arc_lengths[-1]
    if L <= 1e-12:
        return np.vstack([curve[0]] * num_points), 0.0
    s_query = np.linspace(0.0, L, num_points)
    resampled = np.stack([np.interp(s_query, arc_lengths, curve[:, i]) for i in range(3)], axis=1)
    return resampled, L


def compute_deterministic_frame(P_start: np.ndarray, P_end: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    chord = P_end - P_start
    L = np.linalg.norm(chord)
    if L < 1e-12:
        return (np.array([1.0, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
                np.array([0.0, 0.0, 1.0], dtype=np.float64), 0.0)
    T = chord / L
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(T, world_up)) > 0.99:
        world_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    N = np.cross(T, world_up)
    N = N / np.linalg.norm(N)
    B = np.cross(T, N)
    return T, N, B, L


def fit_chebyshev(t_params: np.ndarray, values: np.ndarray, K: int) -> np.ndarray:
    MIN_POINTS = max(2 * K, 100)
    
    n_points = len(t_params)
    
    if n_points < MIN_POINTS and n_points >= 2:
        from scipy.interpolate import interp1d
        
        kind = 'cubic' if n_points >= 4 else 'linear'
        try:
            interp_func = interp1d(t_params, values, kind=kind, fill_value='extrapolate')
            t_resampled = np.linspace(t_params.min(), t_params.max(), MIN_POINTS)
            values_resampled = interp_func(t_resampled)
            t_params = t_resampled
            values = values_resampled
        except Exception:
            pass
    
    x = 2.0 * t_params - 1.0
    try:
        return cheb.chebfit(x, values, K - 1)
    except Exception:
        return np.zeros(K, dtype=np.float64)


def interpolate_radius_on_curve(radius_values: np.ndarray, arc_lengths: np.ndarray, target_s: float) -> float:
    return float(np.interp(target_s, arc_lengths, radius_values))


def normalize_for_ply(points: np.ndarray, radius: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    center = (min_coords + max_coords) / 2
    extent = np.max(max_coords - min_coords)
    
    scale = 2.0 / extent if extent > 1e-6 else 1.0
    
    normalized_points = (points - center) * scale
    normalized_radius = radius / extent * 2.0
    
    scale_info = {
        'center': center.tolist(),
        'extent': float(extent),
        'scale': float(scale)
    }
    
    return normalized_points, normalized_radius, scale_info


def save_ply(filepath: str, points: np.ndarray, radius: np.ndarray, edges: List[Tuple[int, int]]) -> None:
    n_vertices = len(points)
    n_edges = len(edges)
    
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float radius\n")
        f.write(f"element edge {n_edges}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")
        
        for i in range(n_vertices):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} {radius[i]:.6f}\n")
        
        for e in edges:
            f.write(f"{e[0]} {e[1]}\n")


def load_vtp_for_ply(vtp_path: str, merge_tolerance: float = 0.001
                     ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], Dict]:
    if not HAS_PYVISTA:
        raise ImportError("pyvista required: pip install pyvista")
    
    mesh = pv.read(vtp_path)
    raw_points = np.array(mesh.points, dtype=np.float64)
    
    raw_radius = None
    if mesh.point_data:
        for name in mesh.point_data.keys():
            if 'radius' in name.lower():
                raw_radius = np.array(mesh.point_data[name], dtype=np.float64)
                break
        
        if raw_radius is None and len(mesh.point_data.keys()) > 0:
            first_key = list(mesh.point_data.keys())[0]
            raw_radius = np.array(mesh.point_data[first_key], dtype=np.float64)
    
    if raw_radius is None:
        raw_radius = np.ones(len(raw_points), dtype=np.float64) * 1.5
    
    coord_to_new_idx = {}
    old_to_new = {}
    merged_points = []
    merged_radius = []
    
    decimals = int(-np.log10(merge_tolerance))
    
    for old_idx, (pt, r) in enumerate(zip(raw_points, raw_radius)):
        key = tuple(np.round(pt, decimals))
        
        if key not in coord_to_new_idx:
            new_idx = len(merged_points)
            coord_to_new_idx[key] = new_idx
            merged_points.append(pt)
            merged_radius.append(r)
        else:
            new_idx = coord_to_new_idx[key]
        
        old_to_new[old_idx] = new_idx
    
    merged_points = np.array(merged_points, dtype=np.float64)
    merged_radius = np.array(merged_radius, dtype=np.float64)
    
    edges = set()
    
    if mesh.lines is not None and len(mesh.lines) > 0:
        lines = mesh.lines
        i = 0
        while i < len(lines):
            n_pts = lines[i]
            for j in range(n_pts - 1):
                old_idx0 = lines[i + 1 + j]
                old_idx1 = lines[i + 2 + j]
                
                new_idx0 = old_to_new[old_idx0]
                new_idx1 = old_to_new[old_idx1]
                
                if new_idx0 != new_idx1:
                    edge = (min(new_idx0, new_idx1), max(new_idx0, new_idx1))
                    edges.add(edge)
            
            i += n_pts + 1
    
    edges = list(edges)
    
    degree = defaultdict(int)
    for e in edges:
        degree[e[0]] += 1
        degree[e[1]] += 1
    
    n_bifurcations = sum(1 for d in degree.values() if d >= 3)
    n_endpoints = sum(1 for d in degree.values() if d == 1)
    
    stats = {
        'n_raw_points': len(raw_points),
        'n_merged_points': len(merged_points),
        'n_edges': len(edges),
        'n_bifurcations': n_bifurcations,
        'n_endpoints': n_endpoints,
        'merge_ratio': 1.0 - len(merged_points) / len(raw_points) if len(raw_points) > 0 else 0
    }
    
    return merged_points, merged_radius, edges, stats


def process_single_vtp_to_ply(vtp_path: str, ply_output_dir: str, normalize: bool = False
                              ) -> Tuple[str, str, Dict]:
    basename = os.path.basename(vtp_path)
    
    try:
        points, radius, edges, stats = load_vtp_for_ply(vtp_path)
        
        if len(points) == 0:
            return basename, "SKIP: Empty", {}
        
        if normalize:
            points, radius, _ = normalize_for_ply(points, radius)
        
        ply_filename = basename.replace('.vtp', '.ply')
        ply_path = os.path.join(ply_output_dir, ply_filename)
        save_ply(ply_path, points, radius, edges)
        
        return basename, "OK", stats
    
    except Exception as ex:
        return basename, f"ERROR: {ex}", {}


def compute_max_deviation(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    
    start = points[0]
    end = points[-1]
    chord = end - start
    chord_length = np.linalg.norm(chord)
    
    if chord_length < 1e-6:
        return float(np.max(np.linalg.norm(points - start, axis=1)))
    
    chord_dir = chord / chord_length
    max_deviation = 0.0
    
    for p in points:
        v = p - start
        proj_length = np.dot(v, chord_dir)
        proj = proj_length * chord_dir
        perp = v - proj
        deviation = np.linalg.norm(perp)
        max_deviation = max(max_deviation, deviation)
    
    return float(max_deviation)


def needs_hint_node(curve_points: np.ndarray, deviation_threshold: float = 2.0) -> bool:
    return compute_max_deviation(curve_points) >= deviation_threshold


def chebyshev_to_curve_np(coeffs_t, coeffs_n, coeffs_b, P_start, P_end, num_points=64):
    K = len(coeffs_t)
    t_params = np.linspace(0, 1, num_points)
    T, N, B, L = compute_deterministic_frame(P_start, P_end)
    chord = P_end - P_start
    if L < 1e-8:
        return np.tile(P_start, (num_points, 1))
    curve = []
    for t in t_params:
        boundary = t * (1 - t)
        x = 2 * t - 1
        basis = np.array([np.cos(k * np.arccos(np.clip(x, -1, 1))) for k in range(K)])
        delta_t = boundary * np.dot(coeffs_t, basis)
        delta_n = boundary * np.dot(coeffs_n, basis)
        delta_b = boundary * np.dot(coeffs_b, basis)
        baseline = P_start + t * chord
        offset = delta_t * T + delta_n * N + delta_b * B
        curve.append(baseline + offset)
    return np.array(curve)


def extract_curve_recipe_28d(curve_points, P_start, P_end):
    T, N, B, L_straight = compute_deterministic_frame(P_start, P_end)
    resampled, L_curve = resample_curve(curve_points, NUM_SAMPLES)
    if L_straight < 1e-12:
        return {
            'intermediate_offsets': np.zeros(4, dtype=np.float64),
            'chebyshev_coeffs_t': np.zeros(CHEBYSHEV_K, dtype=np.float64),
            'chebyshev_coeffs_n': np.zeros(CHEBYSHEV_K, dtype=np.float64),
            'chebyshev_coeffs_b': np.zeros(CHEBYSHEV_K, dtype=np.float64),
            'L_curve': 0.0, 'L_straight': 0.0,
        }
    local_coords = resampled - P_start
    proj_t = np.dot(local_coords, T)
    proj_n = np.dot(local_coords, N)
    proj_b = np.dot(local_coords, B)
    t_params = np.linspace(0, 1, NUM_SAMPLES)
    baseline_t = t_params * L_straight
    delta_t = proj_t - baseline_t
    delta_n = proj_n
    delta_b = proj_b
    EPS = 1e-4
    boundary_factor = t_params * (1.0 - t_params)
    valid_mask = (t_params > EPS) & (t_params < 1.0 - EPS)
    t_valid = t_params[valid_mask]
    bf_valid = boundary_factor[valid_mask]
    coeffs_t = fit_chebyshev(t_valid, delta_t[valid_mask] / bf_valid, CHEBYSHEV_K)
    coeffs_n = fit_chebyshev(t_valid, delta_n[valid_mask] / bf_valid, CHEBYSHEV_K)
    coeffs_b = fit_chebyshev(t_valid, delta_b[valid_mask] / bf_valid, CHEBYSHEV_K)
    idx_33, idx_66 = int(NUM_SAMPLES * 1/3), int(NUM_SAMPLES * 2/3)
    intermediate_offsets = np.array([delta_n[idx_33], delta_b[idx_33], delta_n[idx_66], delta_b[idx_66]], dtype=np.float64)
    return {
        'intermediate_offsets': intermediate_offsets,
        'chebyshev_coeffs_t': coeffs_t, 'chebyshev_coeffs_n': coeffs_n, 'chebyshev_coeffs_b': coeffs_b,
        'L_curve': L_curve, 'L_straight': L_straight,
    }


def extract_radius_recipe(radius_values, K):
    if len(radius_values) < 2:
        return np.zeros(K, dtype=np.float64)
    t_params = np.linspace(0, 1, len(radius_values))
    return fit_chebyshev(t_params, radius_values, K)


def compute_total_curvature(curve_points):
    if len(curve_points) < 3:
        return 0.0
    v = np.gradient(curve_points, axis=0)
    a = np.gradient(v, axis=0)
    v_norm = np.maximum(np.linalg.norm(v, axis=1), 1e-10)
    cross_norm = np.linalg.norm(np.cross(v, a), axis=1)
    curvature = cross_norm / (v_norm ** 3)
    diff = np.diff(curve_points, axis=0)
    seg_lengths = np.linalg.norm(diff, axis=1)
    total = sum((curvature[i] + curvature[min(i+1, len(curvature)-1)]) / 2 * seg_lengths[i] for i in range(len(seg_lengths)))
    return float(total)


def extract_radius_ratio_chebyshev(radius_values, radius_mean, K=CHEBYSHEV_K_RADIUS):
    MIN_POINTS = max(2 * K, 100)
    
    if len(radius_values) < 2 or radius_mean < 1e-6:
        coeffs = np.zeros(K, dtype=np.float64)
        coeffs[0] = 1.0
        return coeffs
    
    n_points = len(radius_values)
    t = np.linspace(0, 1, n_points)
    
    if n_points < MIN_POINTS:
        from scipy.interpolate import interp1d
        
        kind = 'cubic' if n_points >= 4 else 'linear'
        try:
            interp_func = interp1d(t, radius_values, kind=kind, fill_value='extrapolate')
            t_resampled = np.linspace(0, 1, MIN_POINTS)
            radius_values = interp_func(t_resampled)
            t = t_resampled
            n_points = MIN_POINTS
        except Exception:
            pass
    
    x = 2 * t - 1
    
    ratio = radius_values / radius_mean
    
    try:
        return cheb.chebfit(x, ratio, K - 1).astype(np.float64)
    except:
        coeffs = np.zeros(K, dtype=np.float64)
        coeffs[0] = 1.0
        return coeffs


def extract_skeleton_from_vtp(vtp_path):
    if not HAS_PYVISTA:
        raise ImportError("pyvista required: pip install pyvista")
    
    mesh = pv.read(vtp_path)
    all_points = np.array(mesh.points, dtype=np.float64)
    if 'Radius' not in mesh.point_data:
        raise ValueError(f"No Radius data in {vtp_path}")
    all_radius = np.array(mesh.point_data['Radius'], dtype=np.float64)
    
    lines = mesh.lines
    line_data = []
    idx = 0
    while idx < len(lines):
        n_pts = lines[idx]
        if n_pts >= 2:
            pt_indices = list(lines[idx+1:idx+1+n_pts])
            line_data.append((pt_indices[0], pt_indices[-1], pt_indices))
        idx += n_pts + 1
    
    if not line_data:
        raise ValueError("No lines found")
    
    pos_to_info = defaultdict(lambda: {'indices': [], 'line_ids': []})
    for line_id, (start_idx, end_idx, _) in enumerate(line_data):
        pos_start = tuple(all_points[start_idx].round(6))
        pos_to_info[pos_start]['indices'].append(start_idx)
        pos_to_info[pos_start]['line_ids'].append(line_id)
        pos_end = tuple(all_points[end_idx].round(6))
        pos_to_info[pos_end]['indices'].append(end_idx)
        pos_to_info[pos_end]['line_ids'].append(line_id)
    
    skeleton_nodes = []
    pos_to_skeleton_idx = {}
    for pos, info in pos_to_info.items():
        degree = len(info['line_ids'])
        if degree == 1 or degree >= 3:
            skeleton_idx = len(skeleton_nodes)
            skeleton_nodes.append(SkeletonNode(idx=skeleton_idx, pos=np.array(pos, dtype=np.float64),
                                               degree=degree, original_indices=info['indices'], is_hint=False))
            pos_to_skeleton_idx[pos] = skeleton_idx
    
    skeleton_edges = []
    processed_edges = set()
    for line_id, (start_idx, end_idx, pt_indices) in enumerate(line_data):
        pos_start = tuple(all_points[start_idx].round(6))
        pos_end = tuple(all_points[end_idx].round(6))
        if pos_start not in pos_to_skeleton_idx or pos_end not in pos_to_skeleton_idx:
            continue
        sk_start, sk_end = pos_to_skeleton_idx[pos_start], pos_to_skeleton_idx[pos_end]
        edge_key = (min(sk_start, sk_end), max(sk_start, sk_end))
        if edge_key in processed_edges:
            continue
        processed_edges.add(edge_key)
        skeleton_edges.append(SkeletonEdge(start_node_idx=sk_start, end_node_idx=sk_end,
                                           path_indices=pt_indices, radius_values=all_radius[pt_indices], parent_edge_idx=-1))
    
    return skeleton_nodes, skeleton_edges, all_points, all_radius


def add_hint_nodes(skeleton_nodes, skeleton_edges, all_points, all_radius, deviation_threshold=2.0, hints_per_segment=2):
    new_nodes = list(skeleton_nodes)
    new_edges = []
    stats = {'original_edges': len(skeleton_edges), 'long_edges': 0, 'hint_nodes_added': 0,
             'segments_created': 0, 'short_edges_kept': 0, 'hints_per_segment': hints_per_segment}

    for orig_edge_idx, edge in enumerate(skeleton_edges):
        curve_points = all_points[edge.path_indices]
        radius_values = edge.radius_values

        if needs_hint_node(curve_points, deviation_threshold):
            stats['long_edges'] += 1
            P_start = skeleton_nodes[edge.start_node_idx].pos
            P_end = skeleton_nodes[edge.end_node_idx].pos
            recipe = extract_curve_recipe_28d(curve_points, P_start, P_end)

            reconstructed = chebyshev_to_curve_np(
                recipe['chebyshev_coeffs_t'], recipe['chebyshev_coeffs_n'], recipe['chebyshev_coeffs_b'],
                P_start, P_end, num_points=64
            )
            N_pts = len(reconstructed)

            arc_lengths = compute_arc_lengths(curve_points)
            L_curve = arc_lengths[-1]

            hint_positions = []
            hint_radii = []
            for i in range(1, hints_per_segment + 1):
                t = i / (hints_per_segment + 1)
                hint_idx = int(t * N_pts)
                hint_idx = max(1, min(N_pts - 2, hint_idx))
                hint_pos = reconstructed[hint_idx].copy()
                hint_radius = interpolate_radius_on_curve(radius_values, arc_lengths, L_curve * t)
                hint_positions.append(hint_pos)
                hint_radii.append(hint_radius)

            split_indices = []
            for hint_pos in hint_positions:
                dist = np.linalg.norm(curve_points - hint_pos, axis=1)
                split_idx = int(np.argmin(dist))
                split_indices.append(split_idx)

            for i in range(1, len(split_indices)):
                if split_indices[i] <= split_indices[i-1]:
                    split_indices[i] = min(len(curve_points) - 1, split_indices[i-1] + 1)

            hint_node_indices = []
            for j, (hint_pos, hint_radius) in enumerate(zip(hint_positions, hint_radii)):
                hint_idx = len(new_nodes)
                new_nodes.append(SkeletonNode(idx=hint_idx, pos=hint_pos, degree=2, original_indices=[], is_hint=True))
                hint_node_indices.append(hint_idx)
            stats['hint_nodes_added'] += len(hint_node_indices)

            all_node_indices = [edge.start_node_idx] + hint_node_indices + [edge.end_node_idx]
            all_split_points = [0] + split_indices + [len(curve_points) - 1]
            all_radii_at_splits = [radius_values[0]] + hint_radii + [radius_values[-1]]

            for seg_i in range(len(all_node_indices) - 1):
                start_node = all_node_indices[seg_i]
                end_node = all_node_indices[seg_i + 1]
                start_pt_idx = all_split_points[seg_i]
                end_pt_idx = all_split_points[seg_i + 1]

                if seg_i == 0:
                    seg_curve = np.vstack([curve_points[:end_pt_idx+1], hint_positions[0].reshape(1, 3)])
                    seg_radius = np.concatenate([radius_values[:end_pt_idx+1], [hint_radii[0]]])
                elif seg_i == len(all_node_indices) - 2:
                    seg_curve = np.vstack([hint_positions[-1].reshape(1, 3), curve_points[start_pt_idx+1:]])
                    seg_radius = np.concatenate([[hint_radii[-1]], radius_values[start_pt_idx+1:]])
                else:
                    seg_curve = np.vstack([
                        hint_positions[seg_i-1].reshape(1, 3),
                        curve_points[start_pt_idx+1:end_pt_idx+1],
                        hint_positions[seg_i].reshape(1, 3)
                    ])
                    seg_radius = np.concatenate([
                        [hint_radii[seg_i-1]],
                        radius_values[start_pt_idx+1:end_pt_idx+1],
                        [hint_radii[seg_i]]
                    ])

                new_edges.append(SkeletonEdge(start_node, end_node,
                                              edge.path_indices[start_pt_idx:end_pt_idx+1],
                                              seg_radius, orig_edge_idx, seg_curve))

            stats['segments_created'] += hints_per_segment + 1
        else:
            new_edges.append(SkeletonEdge(edge.start_node_idx, edge.end_node_idx, edge.path_indices,
                                          edge.radius_values, orig_edge_idx, curve_points))
            stats['short_edges_kept'] += 1

    degree_count = defaultdict(int)
    for e in new_edges:
        degree_count[e.start_node_idx] += 1
        degree_count[e.end_node_idx] += 1
    for node in new_nodes:
        node.degree = degree_count[node.idx]

    stats['final_nodes'] = len(new_nodes)
    stats['final_edges'] = len(new_edges)
    return new_nodes, new_edges, stats


def compute_segment_groups(skeleton_edges, skeleton_nodes, node_degrees):
    E = len(skeleton_edges)
    if E == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    
    node_to_edges = defaultdict(list)
    for ei, e in enumerate(skeleton_edges):
        node_to_edges[e.start_node_idx].append(ei)
        node_to_edges[e.end_node_idx].append(ei)
    
    visited = np.zeros(E, dtype=bool)
    segment_group_id = np.full(E, -1, dtype=np.int64)
    group_members = []
    current_group_id = 0
    
    for start_edge in range(E):
        if visited[start_edge]:
            continue
        group = []
        queue = deque([start_edge])
        visited[start_edge] = True
        while queue:
            edge_idx = queue.popleft()
            group.append(edge_idx)
            segment_group_id[edge_idx] = current_group_id
            e = skeleton_edges[edge_idx]
            for node_idx in [e.start_node_idx, e.end_node_idx]:
                if node_degrees[node_idx] == 2:
                    for nei in node_to_edges[node_idx]:
                        if not visited[nei]:
                            visited[nei] = True
                            queue.append(nei)
        group_members.append(group)
        current_group_id += 1
    
    position_in_group = np.zeros(E, dtype=np.float64)
    for group in group_members:
        gs = len(group)
        for pos, ei in enumerate(group):
            position_in_group[ei] = pos / (gs - 1) if gs > 1 else 0.5
    
    return segment_group_id, position_in_group


def compute_dist_to_leaf(skeleton_edges, node_degrees, num_nodes):
    E, N = len(skeleton_edges), num_nodes
    if E == 0 or N == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    
    node_neighbors = defaultdict(set)
    for e in skeleton_edges:
        node_neighbors[e.start_node_idx].add(e.end_node_idx)
        node_neighbors[e.end_node_idx].add(e.start_node_idx)
    
    leaf_nodes = [i for i in range(N) if node_degrees[i] == 1]
    node_dist = np.full(N, np.inf, dtype=np.float64)
    
    if not leaf_nodes:
        node_dist[:] = 0.0
    else:
        queue = deque()
        for leaf in leaf_nodes:
            node_dist[leaf] = 0.0
            queue.append(leaf)
        while queue:
            node = queue.popleft()
            for neighbor in node_neighbors[node]:
                if node_dist[neighbor] > node_dist[node] + 1:
                    node_dist[neighbor] = node_dist[node] + 1
                    queue.append(neighbor)
    
    dist_start = np.array([node_dist[e.start_node_idx] for e in skeleton_edges])
    dist_end = np.array([node_dist[e.end_node_idx] for e in skeleton_edges])
    max_dist = max(dist_start.max(), dist_end.max(), 1.0)
    return dist_start / max_dist, dist_end / max_dist


def process_single_vtp_to_npz(vtp_path, npz_output_dir, deviation_threshold=2.0, hints_per_segment=2):
    basename = os.path.basename(vtp_path)
    stats = {}

    try:
        skeleton_nodes, skeleton_edges, all_points, all_radius = extract_skeleton_from_vtp(vtp_path)
        if len(skeleton_nodes) < 2 or len(skeleton_edges) < 1:
            return basename, "SKIP: too few nodes/edges", stats

        original_nodes = skeleton_nodes
        original_edges = skeleton_edges
        E_original = len(original_edges)
        
        skeleton_nodes, skeleton_edges, hint_stats = add_hint_nodes(
            original_nodes, original_edges, all_points, all_radius, deviation_threshold, hints_per_segment
        )
        
        stats = {
            'n_nodes': len(skeleton_nodes), 'n_edges': len(skeleton_edges),
            'n_hint': sum(1 for n in skeleton_nodes if n.is_hint),
            'hint_stats': hint_stats, 'n_original_edges': E_original,
        }
        
        skeleton_pos = np.array([n.pos for n in skeleton_nodes], dtype=np.float64)
        centroid = skeleton_pos.mean(axis=0)
        skeleton_pos_centered = skeleton_pos - centroid
        scale = np.linalg.norm(skeleton_pos_centered, axis=1).max()
        if scale < 1e-6:
            scale = 1.0
        skeleton_pos_normalized = skeleton_pos_centered / scale
        
        edge_index = np.array([[e.start_node_idx for e in skeleton_edges], [e.end_node_idx for e in skeleton_edges]], dtype=np.int64)
        node_degrees = np.array([n.degree for n in skeleton_nodes], dtype=np.int64)
        
        edge_radius_mean = np.array([np.mean(e.radius_values) if e.radius_values is not None and len(e.radius_values) > 0 else 1.5 for e in skeleton_edges], dtype=np.float64)
        
        E = len(skeleton_edges)
        recipe_intermediate = np.zeros((E, 4), dtype=np.float64)
        recipe_coeffs_t = np.zeros((E, CHEBYSHEV_K), dtype=np.float64)
        recipe_coeffs_n = np.zeros((E, CHEBYSHEV_K), dtype=np.float64)
        recipe_coeffs_b = np.zeros((E, CHEBYSHEV_K), dtype=np.float64)
        recipe_coeffs_r = np.zeros((E, CHEBYSHEV_K), dtype=np.float64)
        edge_L_curve = np.zeros(E, dtype=np.float64)
        edge_L_straight = np.zeros(E, dtype=np.float64)
        edge_P_start = np.zeros((E, 3), dtype=np.float64)
        edge_P_end = np.zeros((E, 3), dtype=np.float64)
        segment_parent_edge = np.zeros(E, dtype=np.int64)
        
        for ei, e in enumerate(skeleton_edges):
            curve_points = e.curve_points if e.curve_points is not None else all_points[e.path_indices]
            P_start = skeleton_nodes[e.start_node_idx].pos
            P_end = skeleton_nodes[e.end_node_idx].pos
            recipe = extract_curve_recipe_28d(curve_points, P_start, P_end)
            recipe_intermediate[ei] = recipe['intermediate_offsets']
            recipe_coeffs_t[ei] = recipe['chebyshev_coeffs_t']
            recipe_coeffs_n[ei] = recipe['chebyshev_coeffs_n']
            recipe_coeffs_b[ei] = recipe['chebyshev_coeffs_b']
            edge_L_curve[ei] = recipe['L_curve']
            edge_L_straight[ei] = recipe['L_straight']
            edge_P_start[ei] = P_start
            edge_P_end[ei] = P_end
            segment_parent_edge[ei] = e.parent_edge_idx
            recipe_coeffs_r[ei] = extract_radius_recipe(e.radius_values, CHEBYSHEV_K)
        
        segment_group_id, position_in_group = compute_segment_groups(skeleton_edges, skeleton_nodes, node_degrees)
        dist_to_leaf_start, dist_to_leaf_end = compute_dist_to_leaf(skeleton_edges, node_degrees, len(skeleton_nodes))
        
        original_edge_index = np.array([[e.start_node_idx for e in original_edges], [e.end_node_idx for e in original_edges]], dtype=np.int64)
        recipe_coeffs_t_original = np.zeros((E_original, CHEBYSHEV_K), dtype=np.float64)
        recipe_coeffs_n_original = np.zeros((E_original, CHEBYSHEV_K), dtype=np.float64)
        recipe_coeffs_b_original = np.zeros((E_original, CHEBYSHEV_K), dtype=np.float64)
        recipe_coeffs_r_original = np.zeros((E_original, CHEBYSHEV_K_RADIUS), dtype=np.float64)
        edge_P_start_original = np.zeros((E_original, 3), dtype=np.float64)
        edge_P_end_original = np.zeros((E_original, 3), dtype=np.float64)
        edge_L_straight_original = np.zeros(E_original, dtype=np.float64)
        edge_L_curve_original = np.zeros(E_original, dtype=np.float64)
        edge_radius_mean_original = np.zeros(E_original, dtype=np.float64)
        total_curvature_original = np.zeros(E_original, dtype=np.float64)
        has_hint_original = np.zeros(E_original, dtype=np.bool_)
        hint_positions_original = np.zeros((E_original, hints_per_segment, 3), dtype=np.float64)

        for ei, e in enumerate(original_edges):
            curve_points = all_points[e.path_indices]
            radius_values = e.radius_values if e.radius_values is not None else all_radius[e.path_indices]
            P_start = original_nodes[e.start_node_idx].pos
            P_end = original_nodes[e.end_node_idx].pos
            recipe = extract_curve_recipe_28d(curve_points, P_start, P_end)
            recipe_coeffs_t_original[ei] = recipe['chebyshev_coeffs_t']
            recipe_coeffs_n_original[ei] = recipe['chebyshev_coeffs_n']
            recipe_coeffs_b_original[ei] = recipe['chebyshev_coeffs_b']
            edge_P_start_original[ei] = P_start
            edge_P_end_original[ei] = P_end
            edge_L_straight_original[ei] = recipe['L_straight']
            edge_L_curve_original[ei] = recipe['L_curve']
            radius_mean = float(np.mean(radius_values))
            edge_radius_mean_original[ei] = radius_mean
            recipe_coeffs_r_original[ei] = extract_radius_ratio_chebyshev(radius_values, radius_mean)
            total_curvature_original[ei] = compute_total_curvature(curve_points)

            if needs_hint_node(curve_points, deviation_threshold):
                has_hint_original[ei] = True
                reconstructed = chebyshev_to_curve_np(
                    recipe['chebyshev_coeffs_t'], recipe['chebyshev_coeffs_n'], recipe['chebyshev_coeffs_b'],
                    P_start, P_end, num_points=64
                )
                N_pts = len(reconstructed)
                for h_i in range(hints_per_segment):
                    t = (h_i + 1) / (hints_per_segment + 1)
                    hint_idx = max(1, min(N_pts - 2, int(t * N_pts)))
                    hint_positions_original[ei, h_i] = (reconstructed[hint_idx] - centroid) / scale
        
        npz_name = basename.replace('.vtp', '.npz')
        np.savez_compressed(
            os.path.join(npz_output_dir, npz_name),
            skeleton_pos=skeleton_pos, skeleton_pos_normalized=skeleton_pos_normalized,
            skeleton_degree=node_degrees, edge_index=edge_index,
            scale=np.array([scale]), centroid=centroid,
            edge_P_start=edge_P_start, edge_P_end=edge_P_end,
            edge_L_curve=edge_L_curve, edge_L_straight=edge_L_straight,
            edge_radius_mean=edge_radius_mean,
            recipe_intermediate=recipe_intermediate,
            recipe_coeffs_t=recipe_coeffs_t, recipe_coeffs_n=recipe_coeffs_n,
            recipe_coeffs_b=recipe_coeffs_b, recipe_coeffs_r=recipe_coeffs_r,
            chebyshev_K=np.array([CHEBYSHEV_K]),
            node_is_hint=np.array([n.is_hint for n in skeleton_nodes], dtype=np.bool_),
            segment_parent_edge=segment_parent_edge,
            deviation_threshold_mm=np.array([deviation_threshold]),
            segment_group_id=segment_group_id, position_in_group=position_in_group,
            dist_to_leaf_start=dist_to_leaf_start, dist_to_leaf_end=dist_to_leaf_end,
            edge_index_original=original_edge_index,
            recipe_coeffs_t_original=recipe_coeffs_t_original,
            recipe_coeffs_n_original=recipe_coeffs_n_original,
            recipe_coeffs_b_original=recipe_coeffs_b_original,
            recipe_coeffs_r_original=recipe_coeffs_r_original,
            edge_P_start_original=edge_P_start_original,
            edge_P_end_original=edge_P_end_original,
            edge_L_straight_original=edge_L_straight_original,
            edge_L_curve_original=edge_L_curve_original,
            edge_radius_mean_original=edge_radius_mean_original,
            total_curvature=total_curvature_original,
            has_hint=has_hint_original,
            hint_positions=hint_positions_original,
            num_edges_original=np.array([E_original]),
            num_edges_segment=np.array([E]),
            chebyshev_K_radius=np.array([CHEBYSHEV_K_RADIUS]),
            hints_per_segment=np.array([hints_per_segment]),
            version=np.array([171]),
        )
        
        return basename, "OK", stats
    
    except Exception as ex:
        import traceback
        traceback.print_exc()
        return basename, f"ERROR: {ex}", stats


def create_npz_splits(npz_dir, train_ratio=0.81, seed=42):
    npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz') and f != COEFF_STATISTICS_FILENAME])
    
    def get_original_name(fn):
        base = fn.replace('.npz', '')
        return base.split('_aug')[0] if '_aug' in base else base
    
    groups = defaultdict(list)
    for f in npz_files:
        groups[get_original_name(f)].append(f)
    
    originals = sorted(groups.keys())
    np.random.seed(seed)
    np.random.shuffle(originals)
    
    n_train = int(len(originals) * train_ratio)
    train_orig = set(originals[:n_train])
    val_orig = set(originals[n_train:])
    
    train_files = [f for o in train_orig for f in groups[o]]
    val_files = [f for o in val_orig for f in groups[o]]
    
    pd.DataFrame({'filenames': sorted([f.replace('.npz', '.pt') for f in train_files])}).to_csv(os.path.join(npz_dir, 'train_filename.csv'), index=False)
    pd.DataFrame({'filenames': sorted([f.replace('.npz', '.pt') for f in val_files])}).to_csv(os.path.join(npz_dir, 'val_filename.csv'), index=False)
    
    print(f"\n  [Data Split] Train: {len(train_files)}, Val: {len(val_files)} (from {len(originals)} originals)")
    return train_files, val_files


def compute_and_save_coeff_statistics(npz_dir, output_path):
    npz_files = sorted([f for f in glob.glob(os.path.join(npz_dir, '*.npz')) if not f.endswith(COEFF_STATISTICS_FILENAME)])
    if not npz_files:
        print("[WARNING] No NPZ files")
        return
    
    print(f"\n[Coeff Stats] Loading {len(npz_files)} files...")
    all_t, all_n, all_b, all_r = [], [], [], []
    all_t_orig, all_n_orig, all_b_orig, all_r_orig, all_curv = [], [], [], [], []
    
    for f in tqdm(npz_files, desc="Loading"):
        try:
            data = np.load(f)
            all_t.append(data['recipe_coeffs_t'])
            all_n.append(data['recipe_coeffs_n'])
            all_b.append(data['recipe_coeffs_b'])
            all_r.append(data['recipe_coeffs_r'])
            if 'recipe_coeffs_t_original' in data:
                all_t_orig.append(data['recipe_coeffs_t_original'])
                all_n_orig.append(data['recipe_coeffs_n_original'])
                all_b_orig.append(data['recipe_coeffs_b_original'])
                all_r_orig.append(data['recipe_coeffs_r_original'])
                all_curv.append(data['total_curvature'])
        except Exception as e:
            print(f"  [WARN] {f}: {e}")
    
    ct, cn, cb, cr = np.vstack(all_t), np.vstack(all_n), np.vstack(all_b), np.vstack(all_r)
    save_dict = {
        'mean_t': ct.mean(0), 'std_t': np.maximum(ct.std(0), 1e-8),
        'mean_n': cn.mean(0), 'std_n': np.maximum(cn.std(0), 1e-8),
        'mean_b': cb.mean(0), 'std_b': np.maximum(cb.std(0), 1e-8),
        'mean_r': cr.mean(0), 'std_r': np.maximum(cr.std(0), 1e-8),
        'total_segments': np.array([len(ct)]), 'chebyshev_K': np.array([CHEBYSHEV_K]),
    }
    
    if all_t_orig:
        cto, cno, cbo, cro = np.vstack(all_t_orig), np.vstack(all_n_orig), np.vstack(all_b_orig), np.vstack(all_r_orig)
        curv = np.concatenate(all_curv)
        save_dict.update({
            'mean_t_original': cto.mean(0), 'std_t_original': np.maximum(cto.std(0), 1e-8),
            'mean_n_original': cno.mean(0), 'std_n_original': np.maximum(cno.std(0), 1e-8),
            'mean_b_original': cbo.mean(0), 'std_b_original': np.maximum(cbo.std(0), 1e-8),
            'mean_r_original': cro.mean(0), 'std_r_original': np.maximum(cro.std(0), 1e-8),
            'mean_curvature': np.array([curv.mean()]), 'std_curvature': np.array([max(curv.std(), 1e-8)]),
            'total_original_edges': np.array([len(cto)]), 'chebyshev_K_radius': np.array([CHEBYSHEV_K_RADIUS]),
            'version': np.array([140]),
        })
    
    np.savez_compressed(output_path, **save_dict)
    print(f"  Saved: {output_path}")


def compute_and_save_gt_statistics(npz_dir, output_path, n_bins=5, l_bin_method="quantile"):
    from scipy.stats import vonmises
    
    npz_files = sorted([f for f in glob.glob(os.path.join(npz_dir, "*.npz")) if not f.endswith(COEFF_STATISTICS_FILENAME)])
    if not npz_files:
        raise ValueError(f"No NPZ files in {npz_dir}")
    
    print(f"\n[GT Stats] Loading {len(npz_files)} files...")
    all_L, all_theta0, all_r0, all_radius, all_hint = [], [], [], [], []
    all_coeffs_r = []
    all_theta_full = []
    detected_hints_per_segment = None

    for f in tqdm(npz_files, desc="Loading"):
        try:
            data = np.load(f)
            L = data['edge_L_straight_original'].astype(np.float32)
            cn = data['recipe_coeffs_n_original'].astype(np.float32)
            cb = data['recipe_coeffs_b_original'].astype(np.float32)
            all_L.append(L)
            all_theta0.append(np.arctan2(cb[:, 0], cn[:, 0]))
            all_r0.append(np.sqrt(cn[:, 0]**2 + cb[:, 0]**2))
            all_radius.append(data.get('edge_radius_mean_original', np.ones_like(L) * 1.5))
            all_hint.append(data.get('has_hint', np.zeros_like(L)))

            if 'recipe_coeffs_r_original' in data:
                all_coeffs_r.append(data['recipe_coeffs_r_original'])

            theta_full = np.arctan2(cb, cn)
            all_theta_full.append(theta_full)

            if detected_hints_per_segment is None:
                if 'hints_per_segment' in data:
                    detected_hints_per_segment = int(data['hints_per_segment'].item())
                elif 'hint_positions' in data:
                    detected_hints_per_segment = data['hint_positions'].shape[1]
                else:
                    detected_hints_per_segment = 2
        except:
            continue

    if detected_hints_per_segment is None:
        detected_hints_per_segment = 2
    print(f"  hints_per_segment = {detected_hints_per_segment} (auto-detected from NPZ)")
    
    L_all = np.concatenate(all_L)
    theta0_all = np.concatenate(all_theta0)
    theta_full_all = np.concatenate(all_theta_full, axis=0)
    
    L_valid = L_all[(~np.isnan(L_all)) & (L_all > 0)]
    if l_bin_method == "quantile":
        boundaries = np.percentile(L_valid, np.linspace(0, 100, n_bins + 1)).tolist()
        boundaries[0], boundaries[-1] = 0, float('inf')
    else:
        boundaries = [0, 15, 25, 40, 80, float('inf')]
    
    bin_names = []
    for i in range(len(boundaries) - 1):
        if i == 0:
            bin_names.append(f"<{boundaries[1]:.1f}mm")
        elif np.isinf(boundaries[i + 1]):
            bin_names.append(f">{boundaries[i]:.1f}mm")
        else:
            bin_names.append(f"{boundaries[i]:.1f}-{boundaries[i+1]:.1f}mm")
    
    l_bin_config = {
        "method": l_bin_method,
        "n_bins": len(bin_names),
        "boundaries": [b if not np.isinf(b) else "inf" for b in boundaries],
        "bin_names": bin_names,
    }
    
    bin_counts = {}
    for i, name in enumerate(bin_names):
        low = boundaries[i]
        high = boundaries[i + 1] if not np.isinf(boundaries[i + 1]) else float('inf')
        mask = (L_all >= low) & (L_all < high)
        bin_counts[name] = int(mask.sum())
    
    total_samples = sum(bin_counts.values())
    
    original_ratios = {name: count / total_samples for name, count in bin_counts.items()}
    
    inv_ratios = {name: 1.0 / (ratio + 1e-6) for name, ratio in original_ratios.items()}
    inv_sum = sum(inv_ratios.values())
    stratified_ratios = {name: inv_ratio / inv_sum for name, inv_ratio in inv_ratios.items()}
    
    stratified_sampling_config = {
        "enabled": True,
        "method": "inverse_frequency",
        "ratios": stratified_ratios,
        "original_ratios": original_ratios,
        "l_bin_config": l_bin_config.copy(),
    }
    
    coeffs_r_inverse_variance_weights = None
    if all_coeffs_r:
        coeffs_r_all = np.vstack(all_coeffs_r)
        variances = np.var(coeffs_r_all, axis=0)
        variances = np.maximum(variances, 1e-10)
        inv_var = 1.0 / variances
        coeffs_r_inverse_variance_weights = (inv_var / inv_var.sum()).tolist()
        print(f"  coeffs_r variances: {variances.tolist()}")
        print(f"  coeffs_r inv_var weights: {coeffs_r_inverse_variance_weights}")
    
    sin_mean, cos_mean = np.mean(np.sin(theta0_all)), np.mean(np.cos(theta0_all))
    circular_mean = np.arctan2(sin_mean, cos_mean)
    
    K = theta_full_all.shape[1]
    theta_params = {}
    
    print(f"\n[Theta Params] Fitting von Mises for k=1 to {K-1}...")
    n_fitted = 0
    for k in range(1, K):
        theta_k = theta_full_all[:, k]
        
        try:
            kappa_fit, mu_fit, _ = vonmises.fit(theta_k, fscale=1)
            theta_params[k] = {"mu": float(mu_fit), "kappa": float(kappa_fit)}
            if kappa_fit >= 0.1:
                n_fitted += 1
        except Exception as e:
            theta_params[k] = {"mu": 0.0, "kappa": 0.0}
    
    print(f"  {n_fitted}/{K-1} coefficients have kappa >= 0.1 (non-uniform)")
    
    stats_dir = os.path.dirname(output_path)
    os.makedirs(stats_dir, exist_ok=True)
    theta_params_path = os.path.join(stats_dir, "theta_params.json")
    
    theta_params_data = {
        "version": "17.0",
        "created_at": datetime.now().isoformat(),
        "source_dir": npz_dir,
        "n_edges": len(theta_full_all),
        "K": K,
        "params": theta_params,
    }
    
    with open(theta_params_path, 'w') as f:
        json.dump(theta_params_data, f, indent=2)
    
    print(f"  Saved: {theta_params_path}")
    
    gt_stats = {
        "version": "18.0",
        "created_at": datetime.now().isoformat(),
        "source_dir": npz_dir,
        "n_files": len(npz_files),
        "n_edges": len(L_all),

        "hints_per_segment": detected_hints_per_segment,

        "coeffs_r_encoding": "ratio",

        "l_bin_config": l_bin_config,

        "theta0_hint_weights": {"weights": [1.0] * detected_hints_per_segment},
        
        "stratified_sampling_config": stratified_sampling_config,
        
        "coeffs_r_inverse_variance_weights": coeffs_r_inverse_variance_weights,
        
        "statistics": {
            "L_straight": {"min": float(L_all.min()), "max": float(L_all.max()), "mean": float(L_all.mean())},
            "theta0_gt": {"circular_mean": float(circular_mean)},
            "hint_ratio": float(np.mean(np.concatenate(all_hint))),
            "bin_counts": bin_counts,
        },
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(gt_stats, f, indent=2)
    
    print(f"\n  Saved: {output_path}")
    print(f"  Included fields:")
    print(f"    - hints_per_segment: {detected_hints_per_segment}")
    print(f"    - coeffs_r_encoding: ratio")
    print(f"    - l_bin_config: {len(bin_names)} bins")
    print(f"    - theta0_hint_weights: {gt_stats['theta0_hint_weights']}")
    print(f"    - stratified_sampling_config: {len(stratified_ratios)} ratios")
    print(f"    - coeffs_r_inverse_variance_weights: {coeffs_r_inverse_variance_weights is not None}")
    print(f"    - theta_params.json: {len(theta_params)} coefficients (separate file)")


def run_vtp2npz(args):
    if not HAS_PYVISTA:
        print("[ERROR] pyvista required: pip install pyvista")
        sys.exit(1)

    vtp_dir = args.vtp_dir or os.path.join(args.root, 'data', args.dataset, 'raw_vtp')
    npz_dir = args.npz_dir or os.path.join(args.root, 'data', args.dataset, 'recipe_npz')
    os.makedirs(npz_dir, exist_ok=True)

    hints_per_segment = get_hints_per_segment_from_tree_pt(args.root, args.dataset)

    print("=" * 60)
    print("  VTP -> NPZ Preprocessor")
    print("=" * 60)
    print(f"  Dataset: {args.dataset}")
    print(f"  VTP dir: {vtp_dir}")
    print(f"  NPZ dir: {npz_dir}")
    print(f"  Hint threshold: {args.deviation_threshold} mm (max deviation)")
    print(f"  Hints per segment: {hints_per_segment} (from {args.dataset}_tree.pt)")
    print("=" * 60)

    vtp_files = sorted(glob.glob(os.path.join(vtp_dir, "*.vtp")))
    if not vtp_files:
        print(f"[ERROR] No VTP files in {vtp_dir}")
        sys.exit(1)

    print(f"\n[INFO] Found {len(vtp_files)} VTP files")

    ok, skip, err = 0, 0, 0
    for vtp_path in tqdm(vtp_files, desc="Processing"):
        _, status, _ = process_single_vtp_to_npz(vtp_path, npz_dir, args.deviation_threshold, hints_per_segment)
        if status == "OK":
            ok += 1
        elif "SKIP" in status:
            skip += 1
        else:
            err += 1
            print(f"  {status}")

    print(f"\n[Done] OK: {ok}, Skip: {skip}, Error: {err}")

    create_npz_splits(npz_dir, args.train_ratio, args.seed)
    compute_and_save_coeff_statistics(npz_dir, os.path.join(npz_dir, COEFF_STATISTICS_FILENAME))


def run_vtp2ply(args):
    if not HAS_PYVISTA:
        print("[ERROR] pyvista required: pip install pyvista")
        sys.exit(1)
    
    vtp_dir = args.vtp_dir or os.path.join(args.root, 'data', args.dataset, 'raw_vtp')
    ply_dir = args.ply_dir or os.path.join(args.root, 'data', args.dataset, 'reference_ply')
    os.makedirs(ply_dir, exist_ok=True)
    
    print("=" * 60)
    print("  VTP -> PLY Converter")
    print("=" * 60)
    print(f"  VTP dir: {vtp_dir}")
    print(f"  PLY dir: {ply_dir}")
    print(f"  Normalize: {args.normalize}")
    print("=" * 60)
    
    vtp_files = sorted(glob.glob(os.path.join(vtp_dir, "*.vtp")))
    if not vtp_files:
        print(f"[ERROR] No VTP files in {vtp_dir}")
        sys.exit(1)
    
    print(f"\n[INFO] Found {len(vtp_files)} VTP files")
    
    stats = {
        'total': len(vtp_files),
        'success': 0,
        'failed': 0,
        'total_raw_points': 0,
        'total_merged_points': 0,
        'total_edges': 0,
        'total_bifurcations': 0,
        'total_endpoints': 0
    }
    
    for vtp_path in tqdm(vtp_files, desc="Converting to PLY"):
        basename, status, file_stats = process_single_vtp_to_ply(vtp_path, ply_dir, args.normalize)
        
        if status == "OK":
            stats['success'] += 1
            stats['total_raw_points'] += file_stats.get('n_raw_points', 0)
            stats['total_merged_points'] += file_stats.get('n_merged_points', 0)
            stats['total_edges'] += file_stats.get('n_edges', 0)
            stats['total_bifurcations'] += file_stats.get('n_bifurcations', 0)
            stats['total_endpoints'] += file_stats.get('n_endpoints', 0)
        elif "SKIP" in status:
            pass
        else:
            stats['failed'] += 1
            print(f"  {basename}: {status}")
    
    print(f"\n{'='*60}")
    print("  PLY Conversion Complete!")
    print(f"{'='*60}")
    print(f"  Total:   {stats['total']}")
    print(f"  Success: {stats['success']}")
    print(f"  Failed:  {stats['failed']}")
    if stats['success'] > 0:
        n = stats['success']
        print(f"\n  [Point Merge Statistics]")
        print(f"  Avg raw points:      {stats['total_raw_points'] // n}")
        print(f"  Avg merged points:   {stats['total_merged_points'] // n}")
        print(f"  Avg edges:           {stats['total_edges'] // n}")
        print(f"  Avg bifurcations:    {stats['total_bifurcations'] / n:.2f}")
        print(f"  Avg endpoints:       {stats['total_endpoints'] / n:.2f}")
        if stats['total_raw_points'] > 0:
            merge_ratio = 1.0 - stats['total_merged_points'] / stats['total_raw_points']
            print(f"  Point merge ratio:   {merge_ratio*100:.1f}%")
    print(f"\n  Output directory: {ply_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage-2B Preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (NPZ + statistics + PLY)
  python scripts/preprocess_curve.py --mode all --dataset lca

  # VTP -> NPZ only
  python scripts/preprocess_curve.py --mode vtp2npz --dataset lca

  # VTP -> PLY only
  python scripts/preprocess_curve.py --mode ply --dataset lca

  # PLY with normalization
  python scripts/preprocess_curve.py --mode ply --dataset lca --normalize

  # Regenerate statistics only
  python scripts/preprocess_curve.py --mode statistics --dataset lca
        """
    )
    
    parser.add_argument('--mode', required=True, 
                        choices=['vtp2npz', 'statistics', 'ply', 'all'],
                        help='Processing mode: vtp2npz, statistics, ply, or all')
    
    parser.add_argument('--root', default='.', help='Project root directory')
    parser.add_argument('--dataset', default='lca', help='Dataset name (e.g., lca, rca, cow)')
    
    parser.add_argument('--vtp-dir', default=None, 
                        help='Override VTP input directory (default: {root}/data/{dataset}/raw_vtp)')
    parser.add_argument('--npz-dir', default=None,
                        help='Override NPZ output directory (default: {root}/data/{dataset}/recipe_npz)')
    parser.add_argument('--ply-dir', default=None,
                        help='Override PLY output directory (default: {root}/data/{dataset}/reference_ply)')
    parser.add_argument('--output', default=None, 
                        help='Override gt_statistics.json path')
    
    parser.add_argument('--deviation-threshold', type=float, default=2.0, 
                        help='Hint threshold in mm (default: 2.0)')
    parser.add_argument('--train-ratio', type=float, default=0.81,
                        help='Train/val split ratio (default: 0.81)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for split (default: 42)')
    parser.add_argument('--n-bins', type=int, default=5,
                        help='Number of L-bins for stratified sampling (default: 5)')
    parser.add_argument('--l-bin-method', default='quantile', 
                        choices=['quantile', 'hardcoded'],
                        help='L-bin boundary method (default: quantile)')
    
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize PLY coordinates to [-1, 1] range')
    
    args = parser.parse_args()
    
    if args.mode == 'vtp2npz':
        run_vtp2npz(args)
        
    elif args.mode == 'statistics':
        npz_dir = args.npz_dir or os.path.join(args.root, 'data', args.dataset, 'recipe_npz')
        output = args.output or os.path.join(args.root, 'data', args.dataset, 'statistics', 'gt_statistics.json')
        compute_and_save_gt_statistics(npz_dir, output, args.n_bins, args.l_bin_method)
        
    elif args.mode == 'ply':
        run_vtp2ply(args)
        
    elif args.mode == 'all':
        print("\n" + "="*70)
        print("  [1/3] VTP -> NPZ Conversion")
        print("="*70)
        run_vtp2npz(args)
        
        print("\n" + "="*70)
        print("  [2/3] Statistics Generation")
        print("="*70)
        npz_dir = args.npz_dir or os.path.join(args.root, 'data', args.dataset, 'recipe_npz')
        output = args.output or os.path.join(args.root, 'data', args.dataset, 'statistics', 'gt_statistics.json')
        compute_and_save_gt_statistics(npz_dir, output, args.n_bins, args.l_bin_method)
        
        print("\n" + "="*70)
        print("  [3/3] VTP -> PLY Conversion (Reference Set)")
        print("="*70)
        run_vtp2ply(args)
        
        ply_dir = args.ply_dir or os.path.join(args.root, 'data', args.dataset, 'reference_ply')
        stats_dir = os.path.join(args.root, 'data', args.dataset, 'statistics')
        
        print("\n" + "="*70)
        print("  All preprocessing complete!")
        print("="*70)
        print(f"  Dataset: {args.dataset}")
        print(f"  Outputs:")
        print(f"    - NPZ:        {npz_dir}/")
        print(f"    - Statistics: {stats_dir}/")
        print(f"    - PLY:        {ply_dir}/")
        print("="*70)


if __name__ == "__main__":
    main()
