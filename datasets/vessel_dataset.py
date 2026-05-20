
import os
import json
import glob
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


CHEBYSHEV_K = 48
CHEBYSHEV_K_RADIUS = 6


def _missing_config_error(config_type: str, config_path: str) -> str:
    return f"""
================================================================================
[ERROR] Required configuration file not found: {config_path}
================================================================================

The '{config_type}' configuration is required but was not found.

To generate this file, run:

  1. For gt_statistics.json (L-bin, theta[0] weights, Stratified config):
     python scripts/preprocess_curve.py --dataset <name> --compute-statistics

  2. For temperature_config.json (Temperature maps):
     python calibrate_temperature.py \\
         --phase1-ckpt outputs/<dataset>/stage2b/checkpoints/best_magnitude.pt \\
         --phase3-ckpt outputs/<dataset>/stage2b/checkpoints/best_radius.pt \\
         --npz-dir data/<dataset>/recipe_npz \\
         --split-files data/<dataset>/recipe_npz/val_filename.csv \\
         --gt-stats data/<dataset>/statistics/gt_statistics.json \\
         --output data/<dataset>/statistics/temperature_config.json

JSON configuration files are now REQUIRED.
================================================================================
"""


def _convert_inf_strings(boundaries: List) -> List:
    return [
        float('inf') if (isinstance(b, str) and b.lower() == "inf") else b
        for b in boundaries
    ]


def _validate_l_bin_config(config: Dict, source: str) -> Dict:
    required_keys = ["boundaries", "bin_names"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Invalid l_bin_config from {source}: missing '{key}'")
    
    config["boundaries"] = _convert_inf_strings(config["boundaries"])
    
    n_bins = len(config["bin_names"])
    n_boundaries = len(config["boundaries"])
    if n_boundaries != n_bins + 1:
        raise ValueError(
            f"Invalid l_bin_config from {source}: "
            f"expected {n_bins + 1} boundaries for {n_bins} bins, got {n_boundaries}"
        )
    
    return config


def load_l_bin_config(config_path: str) -> Dict:
    if not config_path:
        raise ValueError("config_path is required")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(_missing_config_error("l_bin_config", config_path))

    try:
        with open(path, 'r') as f:
            stats = json.load(f)

        if "l_bin_config" not in stats:
            raise ValueError(f"No 'l_bin_config' found in {config_path}")
        
        l_bin_config = _validate_l_bin_config(stats["l_bin_config"], config_path)
        
        print(f"[L-bin Config] Loaded from: {config_path}")
        print(f"  Method: {l_bin_config.get('method', 'unknown')}")
        print(f"  N-bins: {l_bin_config.get('n_bins', len(l_bin_config['bin_names']))}")
        print(f"  Boundaries: {l_bin_config['boundaries']}")
        print(f"  Bin names: {l_bin_config['bin_names']}")
        
        return l_bin_config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {config_path}: {e}")


def load_stratified_config(config_path: str) -> Dict:
    if not config_path:
        raise ValueError("config_path is required")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(_missing_config_error("stratified_sampling_config", config_path))

    try:
        with open(path, 'r') as f:
            stats = json.load(f)

        stratified_config = stats.get("stratified_sampling_config", None)

        if stratified_config is None:
            raise ValueError(
                f"No 'stratified_sampling_config' found in {config_path}\n"
                "Re-run: python scripts/preprocess_curve.py --dataset <name> --compute-statistics"
            )
        
        if "l_bin_config" in stratified_config:
            stratified_config["l_bin_config"] = _validate_l_bin_config(
                stratified_config["l_bin_config"],
                f"{config_path}:stratified_sampling_config"
            )

        if "ratios" not in stratified_config:
            raise ValueError(f"No 'ratios' found in stratified_sampling_config of {config_path}")
        
        print(f"[Stratified Config] Loaded from: {config_path}")
        print(f"  Method: {stratified_config.get('method', 'unknown')}")
        print(f"  N-bins: {len(stratified_config['ratios'])}")
        print(f"  Ratios: {stratified_config['ratios']}")
        
        return stratified_config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {config_path}: {e}")


def get_l_bin_dynamic(L: float, l_bin_config: Dict) -> str:
    if l_bin_config is None:
        raise ValueError(
            "l_bin_config is required.\n"
            "Load with: l_bin_config = load_l_bin_config('path/to/gt_statistics.json')"
        )

    boundaries = l_bin_config.get("boundaries")
    bin_names = l_bin_config.get("bin_names")

    if boundaries is None or bin_names is None:
        raise ValueError("l_bin_config must contain 'boundaries' and 'bin_names'")

    for i in range(len(boundaries) - 1):
        low = boundaries[i]
        high = boundaries[i + 1]

        if low <= L < high:
            return bin_names[i]

    return bin_names[-1]


def compute_deterministic_frame(P_start: np.ndarray, P_end: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    chord = P_end - P_start
    L = np.linalg.norm(chord)
    
    if L < 1e-12:
        return (np.array([1.0, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
                0.0)
    
    T = chord / L
    
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(T, world_up)) > 0.99:
        world_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    
    N = np.cross(T, world_up)
    N = N / (np.linalg.norm(N) + 1e-12)
    B = np.cross(T, N)
    
    return T, N, B, L


def compute_hint_direction(
    P_start: np.ndarray,
    P_end: np.ndarray,
    hint_positions: np.ndarray,
) -> np.ndarray:
    T, N_axis, B_axis, L = compute_deterministic_frame(P_start, P_end)

    H = hint_positions.shape[0]
    result = np.zeros(2 * H, dtype=np.float64)

    for i in range(H):
        t = (i + 1) / (H + 1)
        baseline = P_start + t * (P_end - P_start)
        delta = hint_positions[i] - baseline
        result[2 * i] = float(np.dot(delta, N_axis))
        result[2 * i + 1] = float(np.dot(delta, B_axis))

    return result


class PolarDataset(Dataset):
    
    def __init__(self, npz_dir: str, split_file: str):
        self.npz_dir = npz_dir
        
        if os.path.exists(split_file):
            df = pd.read_csv(split_file)
            files = df['filenames'].tolist()
            self.npz_files = []
            for f in files:
                if f.endswith('.pt'):
                    self.npz_files.append(f.replace('.pt', '.npz'))
                elif f.endswith('.npz'):
                    self.npz_files.append(f)
                else:
                    self.npz_files.append(f + '.npz')
        else:
            self.npz_files = sorted(glob.glob(os.path.join(npz_dir, '*.npz')))
            self.npz_files = [os.path.basename(f) for f in self.npz_files]
        
        self.npz_files = [
            f for f in self.npz_files 
            if os.path.exists(os.path.join(npz_dir, f))
        ]
        
        self.edges = []
        self._load_all_edges()
    
    def _load_all_edges(self):
        for npz_name in self.npz_files:
            npz_path = os.path.join(self.npz_dir, npz_name)
            try:
                data = np.load(npz_path, allow_pickle=True)
                edges = self._extract_edges(data)
                self.edges.extend(edges)
            except Exception as e:
                print(f"Error loading {npz_name}: {e}")
    
    def _extract_edges(self, data: Dict) -> List[Dict]:
        edges = []

        coeffs_n = data['recipe_coeffs_n_original'].astype(np.float64)
        coeffs_b = data['recipe_coeffs_b_original'].astype(np.float64)
        L_straight = data['edge_L_straight_original'].astype(np.float64)
        radius_mean = data['edge_radius_mean_original'].astype(np.float64)
        edge_index = data['edge_index_original'].astype(np.int64)
        degrees = data['skeleton_degree'].astype(np.int64)

        if 'recipe_coeffs_r_original' in data:
            coeffs_r = data['recipe_coeffs_r_original'].astype(np.float64)
        else:
            coeffs_r = np.zeros((coeffs_n.shape[0], CHEBYSHEV_K_RADIUS), dtype=np.float64)

        has_hint = data['has_hint'].astype(bool)
        hint_positions_all = data['hint_positions'].astype(np.float64)
        H = hint_positions_all.shape[1]
        self._hints_per_segment = H
        skeleton_pos_normalized = data['skeleton_pos_normalized'].astype(np.float64)

        num_edges = edge_index.shape[1]
        actual_K = coeffs_n.shape[1]

        for ei in range(num_edges):
            src_node = edge_index[0, ei]
            dst_node = edge_index[1, ei]

            N = coeffs_n[ei]
            B = coeffs_b[ei]

            if actual_K < CHEBYSHEV_K:
                N = np.pad(N, (0, CHEBYSHEV_K - actual_K), mode='constant')
                B = np.pad(B, (0, CHEBYSHEV_K - actual_K), mode='constant')
            elif actual_K > CHEBYSHEV_K:
                N = N[:CHEBYSHEV_K]
                B = B[:CHEBYSHEV_K]

            polar_r = np.sqrt(N**2 + B**2).astype(np.float32)
            polar_theta = np.arctan2(B, N).astype(np.float32)

            if has_hint[ei]:
                P_start = skeleton_pos_normalized[src_node]
                P_end = skeleton_pos_normalized[dst_node]
                hint_dirs = compute_hint_direction(
                    P_start, P_end, hint_positions_all[ei]
                )
                has_hint_val = 1.0
            else:
                hint_dirs = np.zeros(2 * H, dtype=np.float64)
                has_hint_val = 0.0

            edges.append({
                'L_straight': np.float32(L_straight[ei]),
                'radius_mean': np.float32(radius_mean[ei]),
                'deg_src': np.float32(degrees[src_node]),
                'deg_dst': np.float32(degrees[dst_node]),
                'hint_dirs': hint_dirs.astype(np.float32),
                'has_hint': np.float32(has_hint_val),
                'polar_r': polar_r,
                'polar_theta': polar_theta,
                'coeffs_r': coeffs_r[ei].astype(np.float32),
            })

        return edges
    
    def __len__(self) -> int:
        return len(self.edges)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        edge = self.edges[idx]
        return {
            'L_straight': torch.tensor(edge['L_straight'], dtype=torch.float32),
            'radius_mean': torch.tensor(edge['radius_mean'], dtype=torch.float32),
            'deg_src': torch.tensor(edge['deg_src'], dtype=torch.float32),
            'deg_dst': torch.tensor(edge['deg_dst'], dtype=torch.float32),
            'hint_dirs': torch.tensor(edge['hint_dirs'], dtype=torch.float32),
            'has_hint': torch.tensor(edge['has_hint'], dtype=torch.float32),
            'polar_r': torch.tensor(edge['polar_r'], dtype=torch.float32),
            'polar_theta': torch.tensor(edge['polar_theta'], dtype=torch.float32),
            'coeffs_r': torch.tensor(edge['coeffs_r'], dtype=torch.float32),
        }


def polar_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    L_straight = torch.stack([b['L_straight'] for b in batch])
    radius_mean = torch.stack([b['radius_mean'] for b in batch])
    deg_src = torch.stack([b['deg_src'] for b in batch])
    deg_dst = torch.stack([b['deg_dst'] for b in batch])

    hint_dirs = torch.stack([b['hint_dirs'] for b in batch])
    has_hint = torch.stack([b['has_hint'] for b in batch])

    condition = torch.cat([
        L_straight.unsqueeze(1),
        radius_mean.unsqueeze(1),
        deg_src.unsqueeze(1),
        deg_dst.unsqueeze(1),
        hint_dirs,
        has_hint.unsqueeze(1),
    ], dim=1)

    polar_r = torch.stack([b['polar_r'] for b in batch])
    polar_theta = torch.stack([b['polar_theta'] for b in batch])
    coeffs_r = torch.stack([b['coeffs_r'] for b in batch])

    return {
        'condition': condition,
        'polar_r': polar_r,
        'polar_theta': polar_theta,
        'coeffs_r': coeffs_r,
        'L_straight': L_straight,
        'radius_mean': radius_mean,
        'deg_src': deg_src,
        'deg_dst': deg_dst,
        'hint_dirs': hint_dirs,
        'has_hint': has_hint,
    }


def compute_polar_statistics(dataloader: DataLoader) -> Dict[str, np.ndarray]:
    all_polar_r = []
    all_polar_theta = []
    all_coeffs_r = []
    all_condition = []
    all_L = []
    all_has_hint = []
    
    for batch in dataloader:
        all_polar_r.append(batch['polar_r'].numpy())
        all_polar_theta.append(batch['polar_theta'].numpy())
        all_coeffs_r.append(batch['coeffs_r'].numpy())
        all_condition.append(batch['condition'].numpy())
        all_L.append(batch['L_straight'].numpy())
        all_has_hint.append(batch['has_hint'].numpy())
    
    all_polar_r = np.concatenate(all_polar_r, axis=0)
    all_polar_theta = np.concatenate(all_polar_theta, axis=0)
    all_coeffs_r = np.concatenate(all_coeffs_r, axis=0)
    all_condition = np.concatenate(all_condition, axis=0)
    all_L = np.concatenate(all_L, axis=0)
    all_has_hint = np.concatenate(all_has_hint, axis=0)
    
    return {
        'polar_r_all': all_polar_r,
        'polar_r_mean': np.mean(all_polar_r, axis=0),
        'polar_r_std': np.std(all_polar_r, axis=0),
        'polar_r_min': np.min(all_polar_r, axis=0),
        'polar_r_max': np.max(all_polar_r, axis=0),
        'polar_theta_all': all_polar_theta,
        'polar_theta_k0_mean': np.mean(all_polar_theta[:, 0]),
        'polar_theta_k0_std': np.std(all_polar_theta[:, 0]),
        'coeffs_r_all': all_coeffs_r,
        'coeffs_r_mean': np.mean(all_coeffs_r, axis=0),
        'coeffs_r_std': np.std(all_coeffs_r, axis=0),
        'condition_all': all_condition,
        'condition_mean': np.mean(all_condition, axis=0),
        'condition_std': np.std(all_condition, axis=0),
        'L_straight_all': all_L,
        'has_hint_all': all_has_hint,
        'hint_ratio': np.mean(all_has_hint),
    }


class StratifiedBatchSampler(Sampler):
    
    def __init__(
        self, 
        dataset: Dataset,
        batch_size: int,
        stratified_config: Dict,
        drop_last: bool = True,
        seed: Optional[int] = None,
    ):
        if stratified_config is None:
            raise ValueError(
                "stratified_config is required.\n"
                "Load with: stratified_config = load_stratified_config('path/to/gt_statistics.json')"
            )
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        
        
        if "ratios" not in stratified_config:
            raise ValueError("stratified_config must contain 'ratios'")
        
        self.ratios = stratified_config["ratios"]
        
        if "l_bin_config" not in stratified_config:
            raise ValueError("stratified_config must contain 'l_bin_config'")
        
        l_bin_config = stratified_config["l_bin_config"]
        boundaries = _convert_inf_strings(l_bin_config["boundaries"])
        bin_names = l_bin_config["bin_names"]
        method = stratified_config.get("method", "auto")
        
        print(f"\n  [StratifiedBatchSampler]")
        print(f"    Method: {method}")
        print(f"    Batch size: {batch_size}")
        print(f"    N-bins: {len(bin_names)}")
        
        self.L_values = np.array([edge['L_straight'] for edge in dataset.edges])

        self.indices_by_bin = {}
        for i, name in enumerate(bin_names):
            low = boundaries[i]
            high = boundaries[i + 1]
            
            mask = (self.L_values >= low) & (self.L_values < high)
            self.indices_by_bin[name] = np.where(mask)[0]
        
        self.counts_per_batch = {}
        remaining = batch_size
        
        bin_names_list = list(self.ratios.keys())
        for i, bin_name in enumerate(bin_names_list):
            ratio = self.ratios[bin_name]
            if i == len(bin_names_list) - 1:
                count = remaining
            else:
                count = max(1, int(batch_size * ratio))
                remaining -= count
            self.counts_per_batch[bin_name] = count
        
        batches_per_bin = []
        for bin_name, indices in self.indices_by_bin.items():
            count = self.counts_per_batch.get(bin_name, 0)
            if count > 0 and len(indices) > 0:
                batches_per_bin.append(len(indices) // count)
        
        self.num_batches = min(batches_per_bin) if batches_per_bin else 0
        
        print(f"    Total batches per epoch: {self.num_batches}")
        
        for bin_name in bin_names:
            n_samples = len(self.indices_by_bin.get(bin_name, []))
            n_total = len(self.L_values)
            orig_pct = 100 * n_samples / n_total if n_total > 0 else 0
            target_pct = 100 * self.ratios.get(bin_name, 0)
            boost = target_pct / orig_pct if orig_pct > 0 else 0
            count_per_batch = self.counts_per_batch.get(bin_name, 0)
            print(f"    {bin_name}: {n_samples:,} samples, "
                  f"{count_per_batch}/batch ({orig_pct:.1f}% -> {target_pct:.1f}%, {boost:.2f}x)")
    
    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        
        shuffled_indices = {
            bin_name: rng.permutation(indices).tolist()
            for bin_name, indices in self.indices_by_bin.items()
        }

        positions = {bin_name: 0 for bin_name in self.indices_by_bin}
        
        for _ in range(self.num_batches):
            batch = []
            
            for bin_name, count in self.counts_per_batch.items():
                indices = shuffled_indices[bin_name]
                pos = positions[bin_name]
                
                selected = indices[pos:pos + count]
                
                if len(selected) < count:
                    remaining = count - len(selected)
                    shuffled_indices[bin_name] = rng.permutation(
                        self.indices_by_bin[bin_name]
                    ).tolist()
                    selected = selected + shuffled_indices[bin_name][:remaining]
                    positions[bin_name] = remaining
                else:
                    positions[bin_name] = pos + count
                
                batch.extend(selected)
            
            rng.shuffle(batch)
            yield batch
    
    def __len__(self):
        return self.num_batches


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="vessel_dataset.py test")
    parser.add_argument('--npz_dir', type=str, required=True,
                        help='NPZ directory path')
    parser.add_argument('--split_file', type=str, required=True,
                        help='Split CSV file path')
    parser.add_argument('--gt_stats', type=str, required=True,
                        help='gt_statistics.json path (required)')
    args = parser.parse_args()

    print("=" * 70)
    print("vessel_dataset.py test")
    print("=" * 70)

    dataset = PolarDataset(args.npz_dir, args.split_file)
    print(f"\nDataset loaded: {len(dataset)} edges")

    print(f"\nLoading required configurations...")

    l_bin_config = load_l_bin_config(args.gt_stats)

    try:
        stratified_config = load_stratified_config(args.gt_stats)

        sampler = StratifiedBatchSampler(
            dataset=dataset,
            batch_size=128,
            stratified_config=stratified_config,
            seed=42
        )
        dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=polar_collate_fn)
        print(f"\n  StratifiedBatchSampler: {len(sampler)} batches")
    except ValueError as e:
        print(f"\n  Stratified sampling disabled: {e}")
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=polar_collate_fn)

    batch = next(iter(dataloader))
    print(f"\nBatch test:")
    print(f"  condition: {batch['condition'].shape}")
    print(f"  polar_r: {batch['polar_r'].shape}")
    print(f"  polar_theta: {batch['polar_theta'].shape}")
    print(f"  coeffs_r: {batch['coeffs_r'].shape}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
