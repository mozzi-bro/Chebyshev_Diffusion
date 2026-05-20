
import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from numpy.polynomial import chebyshev as cheb

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


CHEBYSHEV_K = 48
CHEBYSHEV_K_RADIUS = 6
CONDITION_DIM = 9
DEFAULT_PREDICTED_SCALE = 100.0
NUM_CURVE_SAMPLES = 256


def load_gt_statistics(stats_path: str) -> dict:
    if not stats_path or not Path(stats_path).exists():
        raise FileNotFoundError(f"gt_statistics.json not found: {stats_path}")
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    if "l_bin_config" not in stats:
        raise ValueError(f"l_bin_config not found in {stats_path}")
    if "theta0_hint_weights" not in stats:
        raise ValueError(f"theta0_hint_weights not found in {stats_path}")
    
    boundaries = stats["l_bin_config"]["boundaries"]
    stats["l_bin_config"]["boundaries"] = [
        float('inf') if (isinstance(b, str) and b.lower() == "inf") else b
        for b in boundaries
    ]
    
    return stats


def load_temperature_config(config_path: str) -> dict:
    if not config_path or not Path(config_path).exists():
        raise FileNotFoundError(f"temperature_config.json not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if "phase1_temperature" not in config:
        raise ValueError(f"phase1_temperature not found in {config_path}")
    
    return config


def get_l_bin_dynamic(L: float, l_bin_config: dict) -> str:
    boundaries = l_bin_config["boundaries"]
    bin_names = l_bin_config["bin_names"]
    
    for i in range(len(boundaries) - 1):
        low = boundaries[i]
        high = boundaries[i + 1]
        if isinstance(high, str) and high.lower() == "inf":
            high = float('inf')
        if low <= L < high:
            return bin_names[i]
    
    return bin_names[-1]


def get_temperature_dynamic(L: float, l_bin_config: dict, temperature_map: dict) -> float:
    bin_name = get_l_bin_dynamic(L, l_bin_config)
    return temperature_map.get(bin_name, 1.0)


def load_theta_params(stats_dir: str) -> dict:
    theta_params_path = Path(stats_dir) / "theta_params.json"
    
    if not theta_params_path.exists():
        raise FileNotFoundError(
            f"theta_params.json not found: {theta_params_path}\n"
            f"Generate with: python models/curve_diffusion.py (ThetaSampler will auto-fit from NPZ)"
        )
    
    with open(theta_params_path, 'r') as f:
        data = json.load(f)
    
    raw_params = data.get("params", {})
    params = {int(k): v for k, v in raw_params.items()}
    
    print(f"Theta params loaded: {len(params)} coefficients from {theta_params_path}")
    
    return params


def compute_deterministic_frame(P_start: np.ndarray, P_end: np.ndarray):
    chord = P_end - P_start
    L = np.linalg.norm(chord)
    
    if L < 1e-12:
        return (np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                0.0)
    
    T = chord / L
    
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(T, world_up)) > 0.99:
        world_up = np.array([1.0, 0.0, 0.0])
    
    N = np.cross(T, world_up)
    N = N / np.linalg.norm(N)
    B = np.cross(T, N)
    
    return T, N, B, L


def compute_hint_direction(
    P_start: np.ndarray,
    P_end: np.ndarray,
    hint_positions: np.ndarray,
) -> np.ndarray:
    T, N, B, L = compute_deterministic_frame(P_start, P_end)

    H = hint_positions.shape[0]

    if L < 1e-12:
        return np.zeros(2 * H, dtype=np.float64)

    result = np.zeros(2 * H, dtype=np.float64)
    for i in range(H):
        t = (i + 1) / (H + 1)
        baseline = P_start + t * (P_end - P_start)
        delta = hint_positions[i] - baseline
        result[2 * i] = float(np.dot(delta, N))
        result[2 * i + 1] = float(np.dot(delta, B))

    return result


def evaluate_chebyshev_at_t(coeffs: np.ndarray, t: float) -> float:
    x = 2.0 * t - 1.0
    return float(cheb.chebval(x, coeffs))


def reconstruct_curve_from_coeffs(
    coeffs_n: np.ndarray,
    coeffs_b: np.ndarray,
    P_start: np.ndarray,
    P_end: np.ndarray,
    num_samples: int = NUM_CURVE_SAMPLES
) -> np.ndarray:
    T, N, B, L = compute_deterministic_frame(P_start, P_end)
    
    t_params = np.linspace(0, 1, num_samples)
    curve_points = []
    
    for t in t_params:
        baseline = P_start + t * (P_end - P_start)
        boundary = t * (1.0 - t)
        
        delta_n = boundary * evaluate_chebyshev_at_t(coeffs_n, t)
        delta_b = boundary * evaluate_chebyshev_at_t(coeffs_b, t)
        
        point = baseline + delta_n * N + delta_b * B
        curve_points.append(point)
    
    return np.array(curve_points)


def reconstruct_radii_from_coeffs(
    coeffs_r: np.ndarray,
    base_radius: float,
    num_samples: int = NUM_CURVE_SAMPLES
) -> np.ndarray:
    t_params = np.linspace(0, 1, num_samples)
    radii = np.zeros(num_samples)
    
    for i, t in enumerate(t_params):
        ratio = evaluate_chebyshev_at_t(coeffs_r, t)
        radii[i] = base_radius * ratio
    
    radii = np.clip(radii, 0.1, 10.0)
    return radii


def load_treevae(checkpoint_path: str, device: str = 'cuda'):
    from models.tree_vae import RecursiveEncoder, RecursiveDecoder
    
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    
    encoder_weight_key = 'leaf_encoder.attribute_lin_encoder_1.weight'
    if encoder_weight_key in ckpt['encoder']:
        input_size = ckpt['encoder'][encoder_weight_key].shape[1]
    else:
        input_size = 3
    
    latent_size = ckpt['encoder']['sample_encoder.mlp1.weight'].shape[1]
    hidden_size = ckpt['encoder']['sample_encoder.mlp1.weight'].shape[0]

    class Args:
        pass
    args = Args()
    args.device = device
    args.input_size = input_size
    
    encoder = RecursiveEncoder(input_size, latent_size, hidden_size).to(device)
    decoder = RecursiveDecoder(latent_size, hidden_size, input_size, args).to(device)
    
    encoder.load_state_dict(ckpt['encoder'])
    decoder.load_state_dict(ckpt['decoder'])
    encoder.eval()
    decoder.eval()
    
    print(f"Tree VAE Loaded: {checkpoint_path}")
    print(f"     Input size: {input_size}D (auto-detected)")
    
    return encoder, decoder, args, latent_size


def decode_tree(vector, max_depth, decoder):
    from utils.node import create_node
    
    edge_radii = {}
    node_latents = {}
    
    def decode_node(vector, parent_latent, parent_idx, max_depth, decoder):
        cl = decoder.nodeClassifier(vector)
        _, label = torch.max(cl, 1)
        label = label.item()
        
        current_idx = create_node.count
        node_latents[current_idx] = vector
        
        if parent_latent is not None and parent_idx is not None:
            with torch.no_grad():
                pred_radius = decoder.predict_edge_radius(parent_latent, vector)
                edge_radii[(parent_idx, current_idx)] = pred_radius.item()

        if label == 0 and create_node.count <= max_depth:
            node = decoder.featureDecoder(vector)
            return create_node(create_node.count, node)
        elif label == 1 and create_node.count <= max_depth:
            right, node = decoder.internalDecoder(vector)
            d = create_node(create_node.count, node)
            d.right = decode_node(right, vector, current_idx, max_depth, decoder)
            return d
        elif label == 2 and create_node.count <= max_depth:
            left, right, node = decoder.bifurcationDecoder(vector)
            d = create_node(create_node.count, node)
            d.right = decode_node(right, vector, current_idx, max_depth, decoder)
            d.left = decode_node(left, vector, current_idx, max_depth, decoder)
            return d
        return None

    create_node.count = 0
    root_latent = decoder.sample_decoder(vector)
    
    with torch.no_grad():
        predicted_scale_tensor = decoder.predict_scale(root_latent)
        predicted_scale_raw = predicted_scale_tensor.item()
        predicted_scale_raw = np.clip(predicted_scale_raw, 0.005, 0.5)
        predicted_scale_mm = 1.0 / predicted_scale_raw
    
    tree = decode_node(root_latent, None, None, max_depth, decoder)
    
    return tree, edge_radii, predicted_scale_mm


def collect_skeleton_raw(node, parent_idx=None, positions=None, edges=None, node_idx=None):
    if positions is None:
        positions = []
        edges = []
        node_idx = [0]
    
    if node is None:
        return positions, edges
    
    radius = node.radius
    if hasattr(radius, 'cpu'):
        pos = radius.cpu().detach().numpy()
    else:
        pos = np.array(radius)
    pos = np.squeeze(pos).flatten()[:3]
    
    current_idx = node_idx[0]
    positions.append(pos.copy())
    node_idx[0] += 1
    
    if parent_idx is not None:
        edges.append((parent_idx, current_idx))
    
    collect_skeleton_raw(node.left, current_idx, positions, edges, node_idx)
    collect_skeleton_raw(node.right, current_idx, positions, edges, node_idx)
    
    return positions, edges


def compute_degrees(n_nodes: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    degrees = np.zeros(n_nodes, dtype=np.int32)
    for src, dst in edges:
        degrees[src] += 1
        degrees[dst] += 1
    return degrees


@dataclass
class Segment:
    start_idx: int
    end_idx: int
    hint_indices: List[int]
    edge_list: List[Tuple[int, int]]


def build_segments_from_tree(
    positions: List[np.ndarray],
    edges: List[Tuple[int, int]]
) -> Tuple[List[Segment], np.ndarray, int]:
    n_nodes = len(positions)
    degrees = compute_degrees(n_nodes, edges)
    
    adj = defaultdict(list)
    for src, dst in edges:
        adj[src].append(dst)
        adj[dst].append(src)
    
    visited_edges = set()
    segments = []
    
    endpoints = [i for i in range(n_nodes) if degrees[i] != 2]
    
    for start_node in endpoints:
        for neighbor in adj[start_node]:
            edge_key = (min(start_node, neighbor), max(start_node, neighbor))
            if edge_key in visited_edges:
                continue
            
            segment_edges = []
            hint_indices = []
            current = start_node
            next_node = neighbor
            
            while True:
                e_key = (min(current, next_node), max(current, next_node))
                if e_key in visited_edges:
                    break
                visited_edges.add(e_key)
                segment_edges.append((current, next_node))
                
                if degrees[next_node] == 2:
                    hint_indices.append(next_node)
                    neighbors_of_next = adj[next_node]
                    found_next = False
                    for n in neighbors_of_next:
                        if n != current:
                            current = next_node
                            next_node = n
                            found_next = True
                            break
                    if not found_next:
                        break
                else:
                    break
            
            if segment_edges:
                first_edge = segment_edges[0]
                last_edge = segment_edges[-1]
                start_idx = first_edge[0]
                end_idx = last_edge[1]
                
                segments.append(Segment(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    hint_indices=hint_indices,
                    edge_list=segment_edges
                ))
    
    n_hints = sum(1 for d in degrees if d == 2)
    return segments, degrees, n_hints


def load_phase1_model(ckpt_path: str, device: torch.device):
    from models.curve_diffusion import MagnitudeDiffusion
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model = MagnitudeDiffusion.from_config(config)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "diffusion_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["diffusion_state_dict"])
    
    model = model.to(device)
    model.eval()
    
    r_mean = checkpoint.get('r_mean', np.zeros(CHEBYSHEV_K))
    r_std = checkpoint.get('r_std', np.ones(CHEBYSHEV_K))
    if isinstance(r_mean, torch.Tensor): r_mean = r_mean.cpu().numpy()
    if isinstance(r_std, torch.Tensor): r_std = r_std.cpu().numpy()
    
    condition_scaler = checkpoint.get('condition_scaler', None)
    
    print(f"Phase 1 (MagnitudeDiffusion) Loaded: T={model.T}")
    
    return model, r_mean, r_std, condition_scaler


def load_phase2_model(ckpt_path: str, device: torch.device):
    from models.curve_diffusion import ThetaDiffusion
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model = ThetaDiffusion.from_config(config)
    
    if "diffusion_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["diffusion_state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model = model.to(device)
    model.eval()
    
    condition_scaler = checkpoint.get('condition_scaler', None)
    
    print(f"Phase 2 (ThetaDiffusion) Loaded: T={model.T}")
    
    return model, condition_scaler


def load_phase3_model(ckpt_path: str, device: torch.device):
    from models.curve_diffusion import RadiusDiffusion
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model = RadiusDiffusion.from_config(config)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "diffusion_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["diffusion_state_dict"])
    
    model = model.to(device)
    model.eval()
    
    coeffs_r_mean = checkpoint.get('coeffs_r_mean', np.zeros(CHEBYSHEV_K_RADIUS))
    coeffs_r_std = checkpoint.get('coeffs_r_std', np.ones(CHEBYSHEV_K_RADIUS))
    if isinstance(coeffs_r_mean, torch.Tensor): coeffs_r_mean = coeffs_r_mean.cpu().numpy()
    if isinstance(coeffs_r_std, torch.Tensor): coeffs_r_std = coeffs_r_std.cpu().numpy()
    
    condition_scaler = checkpoint.get('condition_scaler', None)
    
    print(f"Phase 3 (RadiusDiffusion) Loaded: K={CHEBYSHEV_K_RADIUS}, T={model.T}")
    
    return model, coeffs_r_mean, coeffs_r_std, condition_scaler


def sample_phase1_with_temperature(
    model,
    condition: torch.Tensor,
    L_values: np.ndarray,
    r_mean: np.ndarray,
    r_std: np.ndarray,
    device: torch.device,
    l_bin_config: dict,
    temperature_map: dict
) -> np.ndarray:
    B = condition.shape[0]
    K = CHEBYSHEV_K
    
    r_mean_tensor = torch.tensor(r_mean, dtype=torch.float32, device=device)
    r_std_tensor = torch.tensor(r_std, dtype=torch.float32, device=device)
    
    temperatures = torch.tensor(
        [get_temperature_dynamic(L, l_bin_config, temperature_map) for L in L_values],
        dtype=torch.float32, device=device
    ).view(B, 1)
    
    T_steps = model.T
    beta_start, beta_end = 1e-4, 0.02
    betas = torch.linspace(beta_start, beta_end, T_steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    r_t = torch.randn(B, K, device=device) * temperatures
    
    with torch.no_grad():
        for t in reversed(range(T_steps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = model.forward(r_t, t_batch, condition)
            
            alpha = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]
            beta = betas[t]
            
            if t > 0:
                noise = torch.randn_like(r_t) * temperatures
            else:
                noise = 0
            
            r_t = (1 / torch.sqrt(alpha)) * (
                r_t - (beta / torch.sqrt(1 - alpha_cumprod_t)) * noise_pred
            ) + torch.sqrt(beta) * noise
    
    r_normalized = r_t
    r_pred = r_normalized * r_std_tensor + r_mean_tensor
    r_pred = torch.clamp(r_pred, min=0)
    
    return r_pred.cpu().numpy()


def sample_phase2(model, condition: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        theta_0 = model.sample(condition, n=1)
    return theta_0.cpu().numpy().flatten()


def sample_phase3_with_temperature(
    model,
    condition: torch.Tensor,
    L_values: np.ndarray,
    coeffs_r_mean: np.ndarray,
    coeffs_r_std: np.ndarray,
    device: torch.device,
    l_bin_config: dict,
    temperature_map: dict
) -> np.ndarray:
    B = condition.shape[0]
    K = CHEBYSHEV_K_RADIUS
    
    coeffs_r_mean_tensor = torch.tensor(coeffs_r_mean, dtype=torch.float32, device=device)
    coeffs_r_std_tensor = torch.tensor(coeffs_r_std, dtype=torch.float32, device=device)
    
    temperatures = torch.tensor(
        [get_temperature_dynamic(L, l_bin_config, temperature_map) for L in L_values],
        dtype=torch.float32, device=device
    ).view(B, 1)
    
    T_steps = model.T
    beta_start, beta_end = 1e-4, 0.02
    betas = torch.linspace(beta_start, beta_end, T_steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    r_t = torch.randn(B, K, device=device) * temperatures
    
    with torch.no_grad():
        for t in reversed(range(T_steps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = model.forward(r_t, t_batch, condition)
            
            alpha = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]
            beta = betas[t]
            
            if t > 0:
                noise = torch.randn_like(r_t) * temperatures
            else:
                noise = 0
            
            r_t = (1 / torch.sqrt(alpha)) * (
                r_t - (beta / torch.sqrt(1 - alpha_cumprod_t)) * noise_pred
            ) + torch.sqrt(beta) * noise
    
    coeffs_r_pred = r_t * coeffs_r_std_tensor + coeffs_r_mean_tensor
    
    return coeffs_r_pred.cpu().numpy()


def compute_theta0_deterministic(
    hint_dirs: np.ndarray,
    theta0_weights: dict,
) -> float:
    weights = theta0_weights["weights"]
    H = len(weights)
    numerator = sum(weights[i] * hint_dirs[2 * i + 1] for i in range(H))
    denominator = sum(weights[i] * hint_dirs[2 * i] for i in range(H))
    return np.arctan2(numerator, denominator)


def sample_theta_rest(n_samples: int, theta_params: dict, K: int = CHEBYSHEV_K) -> np.ndarray:
    theta = np.zeros((n_samples, K - 1))
    
    for k in range(1, K):
        if k in theta_params:
            mu = theta_params[k]["mu"]
            kappa = theta_params[k]["kappa"]
            
            if kappa < 0.1:
                theta[:, k - 1] = np.random.uniform(-np.pi, np.pi, n_samples)
            else:
                theta[:, k - 1] = np.random.vonmises(mu, kappa, n_samples)
        else:
            theta[:, k - 1] = np.random.uniform(-np.pi, np.pi, n_samples)
    
    return theta


def polar_to_NB(r: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N = r * np.cos(theta)
    B = r * np.sin(theta)
    return N, B


def write_ply(
    filepath: str,
    all_points: np.ndarray,
    all_radii: np.ndarray,
    edges: List[Tuple[int, int]],
):
    n_vertices = len(all_points)
    n_edges = len(edges)
    
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float r\n")
        f.write(f"element edge {n_edges}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")
        
        for i in range(n_vertices):
            pt = all_points[i]
            r = all_radii[i]
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {r:.6f}\n")
        
        for v1, v2 in edges:
            f.write(f"{v1} {v2}\n")


def export_vessel_tree_to_ply(
    filepath: str,
    curves: List[np.ndarray],
    radii: List[np.ndarray],
    merge_threshold: float = 1e-5
):
    if len(curves) == 0:
        return
    
    all_points = []
    all_radii = []
    edges = []
    
    endpoint_coords = []
    endpoint_to_idx = {}
    
    def coord_to_key(point: np.ndarray) -> tuple:
        decimals = int(-np.log10(merge_threshold))
        return tuple(np.round(point, decimals=decimals))
    
    def get_or_create_endpoint_idx(point: np.ndarray, radius_val: float) -> int:
        key = coord_to_key(point)
        
        if key in endpoint_to_idx:
            return endpoint_to_idx[key]
        
        new_idx = len(all_points)
        endpoint_to_idx[key] = new_idx
        all_points.append(point.copy())
        all_radii.append(radius_val)
        return new_idx
    
    for curve, radius in zip(curves, radii):
        num_pts = len(curve)
        if num_pts == 0:
            continue
        
        if num_pts == 1:
            get_or_create_endpoint_idx(curve[0], radius[0])
            continue
        
        start_idx = get_or_create_endpoint_idx(curve[0], radius[0])
        
        end_idx = get_or_create_endpoint_idx(curve[-1], radius[-1])
        
        curve_indices = [start_idx]
        for i in range(1, num_pts - 1):
            new_idx = len(all_points)
            all_points.append(curve[i].copy())
            all_radii.append(radius[i])
            curve_indices.append(new_idx)
        curve_indices.append(end_idx)
        
        for i in range(len(curve_indices) - 1):
            idx1, idx2 = curve_indices[i], curve_indices[i + 1]
            if idx1 != idx2:
                edges.append((idx1, idx2))
    
    all_points = np.array(all_points) if all_points else np.zeros((0, 3))
    all_radii = np.array(all_radii) if all_radii else np.zeros(0)
    
    write_ply(filepath, all_points, all_radii, edges)


def process_single_tree(
    tree,
    edge_radii: dict,
    predicted_scale: float,
    phase1_model,
    phase2_model,
    phase3_model,
    r_mean: np.ndarray,
    r_std: np.ndarray,
    coeffs_r_mean: np.ndarray,
    coeffs_r_std: np.ndarray,
    cond_scaler1,
    cond_scaler2,
    cond_scaler3,
    device: torch.device,
    theta0_weights: dict,
    theta_params: dict,
    l_bin_config: dict,
    temperature_map: dict
) -> Optional[Tuple[List[np.ndarray], List[np.ndarray], int]]:
    positions, edges = collect_skeleton_raw(tree)
    positions = np.array(positions)
    n_nodes = len(positions)
    
    if len(edges) == 0:
        return None
    
    segments, degrees, n_hints = build_segments_from_tree(positions.tolist(), edges)
    n_segments = len(segments)
    
    if n_segments == 0:
        return None
    
    conditions = []
    L_values = []
    segment_endpoints_mm = []
    has_hint_list = []
    hint_directions = []
    
    for seg in segments:
        P_start_norm = positions[seg.start_idx]
        P_end_norm = positions[seg.end_idx]
        
        P_start_mm = P_start_norm * predicted_scale
        P_end_mm = P_end_norm * predicted_scale
        
        L_straight = np.linalg.norm(P_end_mm - P_start_mm)
        
        deg_src = float(degrees[seg.start_idx])
        deg_dst = float(degrees[seg.end_idx])
        
        segment_radii = []
        for (src, dst) in seg.edge_list:
            for key in [(src, dst), (dst, src)]:
                if key in edge_radii:
                    segment_radii.append(edge_radii[key])
                    break
        
        if segment_radii:
            radius_mean = np.mean(segment_radii)
        else:
            if cond_scaler3 is not None and hasattr(cond_scaler3, 'mean_'):
                L_mean = cond_scaler3.mean_[0]
                R_mean = cond_scaler3.mean_[1]
                L_std = cond_scaler3.scale_[0]
                R_std = cond_scaler3.scale_[1]
                z_L = (L_straight - L_mean) / (L_std + 1e-6)
                radius_mean = R_mean + z_L * R_std * 0.3
                radius_mean = np.clip(radius_mean, 0.5, 4.0)
            else:
                radius_mean = 1.5
        
        H = len(theta0_weights["weights"])
        if len(seg.hint_indices) >= 1:
            hint_pos_array = np.array([positions[idx] for idx in seg.hint_indices])
            if len(seg.hint_indices) >= H:
                selected = np.linspace(0, len(seg.hint_indices) - 1, H, dtype=int)
                hint_pos_array = hint_pos_array[selected]
            else:
                padded = np.zeros((H, 3))
                padded[:len(seg.hint_indices)] = hint_pos_array
                for j in range(len(seg.hint_indices), H):
                    padded[j] = hint_pos_array[-1]
                hint_pos_array = padded

            hint_dirs = compute_hint_direction(P_start_norm, P_end_norm, hint_pos_array)
            has_hint = 1.0
        else:
            hint_dirs = np.zeros(2 * H)
            has_hint = 0.0

        cond = [L_straight, radius_mean, deg_src, deg_dst]
        cond.extend(hint_dirs.tolist())
        cond.append(has_hint)
        conditions.append(cond)
        L_values.append(L_straight)
        segment_endpoints_mm.append((P_start_mm, P_end_mm))
        has_hint_list.append(has_hint > 0.5)
        hint_directions.append(hint_dirs)
    
    conditions = np.array(conditions, dtype=np.float32)
    L_values = np.array(L_values)
    R_values = conditions[:, 1]
    
    if cond_scaler1 is not None:
        conditions_norm_p1 = cond_scaler1.transform(conditions)
    else:
        conditions_norm_p1 = conditions
    conditions_tensor_p1 = torch.tensor(conditions_norm_p1, dtype=torch.float32, device=device)
    
    if cond_scaler2 is not None:
        conditions_norm_p2 = cond_scaler2.transform(conditions)
    else:
        conditions_norm_p2 = conditions
    conditions_tensor_p2 = torch.tensor(conditions_norm_p2, dtype=torch.float32, device=device)
    
    if cond_scaler3 is not None:
        conditions_norm_p3 = cond_scaler3.transform(conditions)
    else:
        conditions_norm_p3 = conditions
    conditions_tensor_p3 = torch.tensor(conditions_norm_p3, dtype=torch.float32, device=device)
    
    r_pred = sample_phase1_with_temperature(
        phase1_model, conditions_tensor_p1, L_values, r_mean, r_std, device,
        l_bin_config, temperature_map
    )
    
    theta_0_pred = sample_phase2(phase2_model, conditions_tensor_p2, device)
    
    for i in range(n_segments):
        if has_hint_list[i]:
            theta_0_pred[i] = compute_theta0_deterministic(hint_directions[i], theta0_weights)
    
    theta_rest = sample_theta_rest(n_segments, theta_params, CHEBYSHEV_K)
    
    theta_all = np.zeros((n_segments, CHEBYSHEV_K))
    theta_all[:, 0] = theta_0_pred
    theta_all[:, 1:] = theta_rest
    
    coeffs_n_pred, coeffs_b_pred = polar_to_NB(r_pred, theta_all)
    
    coeffs_r_pred = sample_phase3_with_temperature(
        phase3_model, conditions_tensor_p3, L_values, coeffs_r_mean, coeffs_r_std, device,
        l_bin_config, temperature_map
    )
    
    curves = []
    radii = []
    
    for i in range(n_segments):
        P_start_mm, P_end_mm = segment_endpoints_mm[i]
        base_r = R_values[i]
        
        curve = reconstruct_curve_from_coeffs(
            coeffs_n_pred[i], coeffs_b_pred[i], P_start_mm, P_end_mm
        )
        curves.append(curve)
        
        radius_profile = reconstruct_radii_from_coeffs(coeffs_r_pred[i], base_r)
        radii.append(radius_profile)
    
    return curves, radii, n_segments


def main():
    parser = argparse.ArgumentParser(
        description='Tree VAE -> Stage-2B -> PLY Export Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 vessels and export to PLY
  python export.py \\
      --treevae-ckpt ./logs/tree_lca_tree_01_29_03_03/models/best_model_gwd.pth \\
      --phase1-ckpt ./outputs/lca/stage2b/checkpoints/best_magnitude.pt \\
      --phase2-ckpt ./outputs/lca/stage2b/checkpoints/best_theta.pt \\
      --phase3-ckpt ./outputs/lca/stage2b/checkpoints/best_radius.pt \\
      --gt-stats data/lca/statistics/gt_statistics.json \\
      --temperature-config data/lca/statistics/temperature_config.json \\
      --stats-dir data/lca/statistics \\
      --output-dir ./outputs/lca/export_ply \\
      --num-samples 100 --sigma 1.0
        """
    )
    
    parser.add_argument('--treevae-ckpt', type=str, required=True,
                        help='Tree VAE checkpoint path')
    parser.add_argument('--phase1-ckpt', type=str, required=True,
                        help='Phase 1 (MagnitudeDiffusion) checkpoint path')
    parser.add_argument('--phase2-ckpt', type=str, required=True,
                        help='Phase 2 (ThetaDiffusion) checkpoint path')
    parser.add_argument('--phase3-ckpt', type=str, required=True,
                        help='Phase 3 (RadiusDiffusion) checkpoint path')
    parser.add_argument('--gt-stats', type=str, required=True,
                        help='GT statistics JSON path')
    parser.add_argument('--temperature-config', type=str, required=True,
                        help='Temperature config JSON path')
    parser.add_argument('--stats-dir', type=str, required=True,
                        help='Statistics directory (contains theta_params.json)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for PLY files')
    
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Standard deviation for latent space sampling')
    parser.add_argument('--max-depth', type=int, default=20,
                        help='Maximum tree depth for decoding')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible generation')

    args = parser.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"\n[Seed] Random seed set to {args.seed}")

    print("=" * 70)
    print("  export.py")
    print("  Tree VAE -> Stage-2B (Phase 1/2/3) -> PLY Export Pipeline")
    print("  Using learned ScaleHead + EdgeRadiusHead values")
    print("  Ratio-based Radius Decoding")
    print("=" * 70)
    print(f"\n[Configuration]")
    print(f"  Tree VAE:     {args.treevae_ckpt}")
    print(f"  Phase 1:      {args.phase1_ckpt}")
    print(f"  Phase 2:      {args.phase2_ckpt}")
    print(f"  Phase 3:      {args.phase3_ckpt}")
    print(f"  GT Stats:     {args.gt_stats}")
    print(f"  Temp Config:  {args.temperature_config}")
    print(f"  Stats Dir:    {args.stats_dir}")
    print(f"  Output Dir:   {args.output_dir}")
    print(f"  Num Samples:  {args.num_samples}")
    print(f"  Sigma:        {args.sigma}")
    print(f"  Max Depth:    {args.max_depth}")
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, using CPU")
        device = 'cpu'
    device = torch.device(device)
    print(f"\n  Device: {device}")
    
    print("\n[Loading Configurations...]")
    gt_stats = load_gt_statistics(args.gt_stats)
    temp_config = load_temperature_config(args.temperature_config)
    
    theta0_weights_raw = gt_stats["theta0_hint_weights"]
    if "weights" in theta0_weights_raw:
        theta0_weights = theta0_weights_raw
    elif "w1" in theta0_weights_raw and "w2" in theta0_weights_raw:
        theta0_weights = {"weights": [theta0_weights_raw["w1"], theta0_weights_raw["w2"]]}
    else:
        raise ValueError("Invalid theta0_hint_weights format")

    H = gt_stats.get("hints_per_segment", len(theta0_weights["weights"]))
    l_bin_config = gt_stats["l_bin_config"]
    temperature_map = temp_config["phase1_temperature"]
    
    coeffs_r_encoding = gt_stats.get("coeffs_r_encoding", "residual")
    if coeffs_r_encoding == "ratio":
        print(f"  coeffs_r_encoding: ratio")
        print(f"     Decoding formula: radius = base_radius * ratio")
    else:
        print(f"  WARNING: coeffs_r_encoding: {coeffs_r_encoding} (legacy)")
        print(f"     Recommend re-generating with: python preprocess_curve.py --mode all")
    
    radius_stats = gt_stats.get("radius_statistics", None)
    if radius_stats:
        print(f"  Radius Statistics (GT reference):")
        print(f"     mean: {radius_stats['mean']:.4f} mm, std: {radius_stats['std']:.4f} mm")
        if 'L_R_regression' in radius_stats:
            lr = radius_stats['L_R_regression']
            print(f"     L-R regression: R = {lr['slope']:.6f}*L + {lr['intercept']:.4f} (R^2={lr['r_squared']:.4f})")
    else:
        print(f"  WARNING: Radius Statistics not found in gt_stats")

    print(f"  radius_mean is predicted by Tree VAE EdgeRadiusHead")
    
    print(f"  GT Statistics loaded")
    print(f"     hints_per_segment: {H}")
    print(f"     theta[0] hint weights: {theta0_weights['weights']}")
    print(f"  Temperature Config loaded")
    print(f"     L-bins: {l_bin_config['bin_names']}")
    
    theta_params = load_theta_params(args.stats_dir)
    
    print("\n[Loading Models...]")
    
    _, decoder, _, latent_size = load_treevae(args.treevae_ckpt, device)
    
    phase1_model, r_mean, r_std, cond_scaler1 = load_phase1_model(args.phase1_ckpt, device)
    phase2_model, cond_scaler2 = load_phase2_model(args.phase2_ckpt, device)
    phase3_model, coeffs_r_mean, coeffs_r_std, cond_scaler3 = load_phase3_model(args.phase3_ckpt, device)
    
    def build_scaler(scaler_data):
        if scaler_data is None:
            return None
        scaler = StandardScaler()
        scaler.mean_ = np.array(scaler_data['mean_'])
        scaler.scale_ = np.array(scaler_data['scale_'])
        scaler.var_ = np.array(scaler_data['var_'])
        return scaler
    
    cond_scaler1 = build_scaler(cond_scaler1)
    cond_scaler2 = build_scaler(cond_scaler2)
    cond_scaler3 = build_scaler(cond_scaler3)
    
    if cond_scaler1 is None:
        cond_scaler1 = cond_scaler3
    if cond_scaler2 is None:
        cond_scaler2 = cond_scaler3
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n[Generating {args.num_samples} vessels...]")
    
    valid_count = 0
    empty_count = 0
    failed_count = 0
    
    all_segment_counts = []
    
    with torch.no_grad():
        debug_scales = []
        debug_radii = []
        
        for i in tqdm(range(args.num_samples), desc="Generating", ncols=80):
            z = torch.randn(1, latent_size, device=device) * args.sigma
            
            tree, edge_radii, predicted_scale = decode_tree(z, args.max_depth, decoder)
            
            if tree is None:
                empty_count += 1
                continue
            
            if i < 5:
                debug_scales.append(predicted_scale)
                if edge_radii:
                    debug_radii.extend(list(edge_radii.values()))
            

            result = process_single_tree(
                tree=tree,
                edge_radii=edge_radii,
                predicted_scale=predicted_scale,
                phase1_model=phase1_model,
                phase2_model=phase2_model,
                phase3_model=phase3_model,
                r_mean=r_mean,
                r_std=r_std,
                coeffs_r_mean=coeffs_r_mean,
                coeffs_r_std=coeffs_r_std,
                cond_scaler1=cond_scaler1,
                cond_scaler2=cond_scaler2,
                cond_scaler3=cond_scaler3,
                device=device,
                theta0_weights=theta0_weights,
                theta_params=theta_params,
                l_bin_config=l_bin_config,
                temperature_map=temperature_map
            )
            
            if result is None:
                failed_count += 1
                continue
            
            curves, radii, n_segments = result
            all_segment_counts.append(n_segments)
            
            ply_path = os.path.join(args.output_dir, f"vessel_{i:05d}.ply")
            export_vessel_tree_to_ply(ply_path, curves, radii)
            
            valid_count += 1
    
    print("\n" + "=" * 70)
    print("  EXPORT COMPLETED")
    print("=" * 70)
    print(f"  Total attempts:     {args.num_samples}")
    print(f"  Valid vessels:      {valid_count} ({100*valid_count/args.num_samples:.1f}%)")
    print(f"  Empty trees:        {empty_count}")
    print(f"  Failed processing:  {failed_count}")
    
    if debug_scales:
        print(f"\n  [DEBUG: Predicted Scale (first 5 samples)]")
        for j, s in enumerate(debug_scales):
            print(f"    Sample {j}: scale={s:.4f} mm (extent~{2/s:.1f}mm)")
        print(f"    Mean scale: {np.mean(debug_scales):.4f} mm")

    if debug_radii:
        print(f"\n  [DEBUG: Predicted EdgeRadius (first 5 samples)]")
        print(f"    n_edges: {len(debug_radii)}")
        print(f"    mean: {np.mean(debug_radii):.4f} mm")
        print(f"    std:  {np.std(debug_radii):.4f} mm")
        print(f"    range: [{np.min(debug_radii):.4f}, {np.max(debug_radii):.4f}] mm")
    
    if all_segment_counts:
        print(f"\n  [Segment Statistics]")
        print(f"    Mean +- Std: {np.mean(all_segment_counts):.1f} +- {np.std(all_segment_counts):.1f}")
        print(f"    Range: [{min(all_segment_counts)}, {max(all_segment_counts)}]")
    
    print(f"\n  Output directory: {args.output_dir}")
    print(f"  PLY files: {valid_count} files")
    
    print(f"\n=== PLY File Format ===")
    print(f"  - Vertex: x, y, z, r (with radius)")
    print(f"  - Edge: vertex1, vertex2")
    print(f"  - Compatible with the original PartVessel format")


if __name__ == "__main__":
    main()
