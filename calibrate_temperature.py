
import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CHEBYSHEV_K = 48
CHEBYSHEV_K_RADIUS = 6
CONDITION_DIM = 9


from models.curve_diffusion import MagnitudeDiffusion, RadiusDiffusion
from datasets.vessel_dataset import PolarDataset, polar_collate_fn


def _missing_gt_statistics_error(stats_path: str) -> str:
    return f"""
================================================================================
[ERROR] gt_statistics.json not found
================================================================================

Expected path: {stats_path}

This file is REQUIRED and contains:
  - l_bin_config: L-bin boundaries and names

To generate this file:
  python scripts/preprocess_curve.py --dataset <n> --compute-statistics

================================================================================
"""


def load_gt_statistics(stats_path: str) -> Dict:
    from pathlib import Path
    if not stats_path or not Path(stats_path).exists():
        raise FileNotFoundError(_missing_gt_statistics_error(stats_path or "None"))
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    print(f"[GT Statistics] Loaded: {stats_path}")
    
    if "l_bin_config" not in stats:
        raise ValueError(f"l_bin_config not found in {stats_path}")
    
    return stats


def get_l_bin_dynamic(L: float, l_bin_config: Dict) -> str:
    boundaries = l_bin_config["boundaries"]
    bin_names = l_bin_config["bin_names"]
    
    for i in range(len(boundaries) - 1):
        low = boundaries[i]
        high = boundaries[i + 1]
        if isinstance(high, str) and high.lower() == "inf":
            high = float('inf')
        if isinstance(low, str) and low.lower() == "inf":
            low = float('inf')
        if low <= L < high:
            return bin_names[i]
    
    return bin_names[-1]


def load_data_from_dataset(npz_dir: str, split_files: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_filenames = []
    for split_file in split_files:
        if not os.path.exists(split_file):
            continue
        df = pd.read_csv(split_file)
        col = 'filenames' if 'filenames' in df.columns else 'filename'
        if col in df.columns:
            all_filenames.extend(df[col].tolist())
    
    filenames = list(set(all_filenames))
    print(f"  Total unique files: {len(filenames)}")
    
    temp_split_path = os.path.join(npz_dir, '_temp_calibration_split.csv')
    temp_df = pd.DataFrame({'filenames': filenames})
    temp_df.to_csv(temp_split_path, index=False)
    
    try:
        dataset = PolarDataset(npz_dir, temp_split_path)
        print(f"  Loaded {len(dataset)} edges via PolarDataset")
        
        all_polar_r = []
        all_coeffs_r = []
        all_condition = []
        all_L = []
        
        for i in range(len(dataset)):
            item = dataset[i]
            all_polar_r.append(item['polar_r'].numpy())
            all_coeffs_r.append(item['coeffs_r'].numpy())

            cond_parts = [
                item['L_straight'].item(),
                item['radius_mean'].item(),
                item['deg_src'].item(),
                item['deg_dst'].item(),
            ]
            cond_parts.extend(item['hint_dirs'].numpy().tolist())
            cond_parts.append(item['has_hint'].item())
            all_condition.append(np.array(cond_parts, dtype=np.float32))
            all_L.append(item['L_straight'].item())
        
        polar_r = np.array(all_polar_r)
        coeffs_r = np.array(all_coeffs_r)
        condition = np.array(all_condition)
        L_straight = np.array(all_L)
        
    finally:
        if os.path.exists(temp_split_path):
            os.remove(temp_split_path)
    
    return polar_r, coeffs_r, condition, L_straight


def split_by_lbin(target_data: np.ndarray, cond_data: np.ndarray,
                  L_data: np.ndarray, l_bin_config: Dict) -> Dict:
    if l_bin_config is None:
        raise ValueError("l_bin_config is required")
    
    bin_names = l_bin_config["bin_names"]
    bin_data = {name: {'target': [], 'cond': [], 'L': []} for name in bin_names}
    
    for i, L in enumerate(L_data):
        bin_name = get_l_bin_dynamic(L, l_bin_config)
        
        if bin_name in bin_data:
            bin_data[bin_name]['target'].append(target_data[i])
            bin_data[bin_name]['cond'].append(cond_data[i])
            bin_data[bin_name]['L'].append(L)
    
    for name in bin_data:
        bin_data[name]['target'] = np.array(bin_data[name]['target']) if bin_data[name]['target'] else np.array([])
        bin_data[name]['cond'] = np.array(bin_data[name]['cond']) if bin_data[name]['cond'] else np.array([])
    
    return bin_data


def calibrate_temperature(gt_variance: float, gen_variance: float, 
                         min_temp: float = 0.70, max_temp: float = 1.50) -> float:
    if gt_variance < 1e-10:
        return 1.0
    
    VR = gen_variance / gt_variance
    
    if VR > 1.2:
        T = 1.0 / np.sqrt(VR)
        T = max(T, min_temp)
    elif VR < 0.8:
        T = np.sqrt(1.0 / VR)
        T = min(T, max_temp)
    else:
        T = 1.0
    
    return round(T, 2)


def compute_variance_phase1(model, bin_data: Dict, r_mean: np.ndarray, r_std: np.ndarray,
                            cond_scaler: StandardScaler, device: torch.device,
                            temperature: float = 1.0, n_samples: int = 30) -> Dict:
    results = {}
    
    for bin_name, data in bin_data.items():
        if len(data['target']) == 0:
            continue
        
        r_gt = data['target']
        cond = data['cond']
        
        cond_scaled = cond_scaler.transform(cond)
        cond_tensor = torch.tensor(cond_scaled, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            samples_normalized = model.sample(
                cond_tensor, n=n_samples, temperature=temperature
            )
            samples = samples_normalized.cpu().numpy() * r_std + r_mean
        
        gt_var_per_k = np.var(r_gt, axis=0)
        gt_var_avg = np.mean(gt_var_per_k)
        
        gen_flat = samples.reshape(-1, CHEBYSHEV_K)
        gen_var_per_k = np.var(gen_flat, axis=0)
        gen_var_avg = np.mean(gen_var_per_k)
        
        vr_per_k = gen_var_per_k / (gt_var_per_k + 1e-10)
        vr_avg = np.mean(vr_per_k)
        
        results[bin_name] = {
            'n_samples': len(r_gt),
            'gt_variance': float(gt_var_avg),
            'gen_variance': float(gen_var_avg),
            'variance_ratio': float(vr_avg),
            'gt_variance_per_k': gt_var_per_k.tolist(),
            'gen_variance_per_k': gen_var_per_k.tolist(),
        }
    
    return results


def compute_variance_phase3(model, bin_data: Dict, coeffs_r_mean: np.ndarray,
                            coeffs_r_std: np.ndarray, cond_scaler: StandardScaler,
                            device: torch.device, temperature: float = 1.0,
                            n_samples: int = 30) -> Dict:
    results = {}
    
    for bin_name, data in bin_data.items():
        if len(data['target']) == 0:
            continue
        
        coeffs_r_gt = data['target']
        cond = data['cond']
        
        cond_scaled = cond_scaler.transform(cond)
        cond_tensor = torch.tensor(cond_scaled, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            samples_normalized = model.sample(cond_tensor, n=n_samples, temperature=temperature)
            samples = samples_normalized.cpu().numpy() * coeffs_r_std + coeffs_r_mean
        
        gt_var_per_k = np.var(coeffs_r_gt, axis=0)
        gt_var_avg = np.mean(gt_var_per_k)
        
        gen_flat = samples.reshape(-1, CHEBYSHEV_K_RADIUS)
        gen_var_per_k = np.var(gen_flat, axis=0)
        gen_var_avg = np.mean(gen_var_per_k)
        
        vr_per_k = gen_var_per_k / (gt_var_per_k + 1e-10)
        vr_avg = np.mean(vr_per_k)
        
        results[bin_name] = {
            'n_samples': len(coeffs_r_gt),
            'gt_variance': float(gt_var_avg),
            'gen_variance': float(gen_var_avg),
            'variance_ratio': float(vr_avg),
            'gt_variance_per_k': gt_var_per_k.tolist(),
            'gen_variance_per_k': gen_var_per_k.tolist(),
        }
    
    return results


def calibrate_phase1(model, bin_data: Dict, r_mean: np.ndarray, r_std: np.ndarray,
                     cond_scaler: StandardScaler, device: torch.device,
                     n_samples: int = 50, bin_names: List[str] = None) -> Tuple[Dict, Dict]:
    
    print("\n" + "=" * 80)
    print("  Phase 1: MagnitudeDiffusion Temperature Calibration")
    print("=" * 80)
    
    if bin_names is None:
        bin_names = list(bin_data.keys())
    
    print("\n[Step 1] Computing variance with T=1.0...")
    variance_results = compute_variance_phase1(
        model, bin_data, r_mean, r_std, cond_scaler, device, 
        temperature=1.0, n_samples=n_samples
    )
    
    print("\n[Step 2] Calibrating optimal temperature...")
    temperature_map = {}
    calibration_details = {}
    
    print(f"\n  {'L-bin':<20} | {'GT Var':>12} | {'Gen Var':>12} | {'VR':>8} | {'Opt T':>8}")
    print(f"  {'-'*75}")
    
    for bin_name in bin_names:
        if bin_name not in variance_results:
            continue
        
        res = variance_results[bin_name]
        opt_temp = calibrate_temperature(res['gt_variance'], res['gen_variance'])
        temperature_map[bin_name] = opt_temp
        
        calibration_details[bin_name] = {
            'gt_variance': res['gt_variance'],
            'gen_variance_t1': res['gen_variance'],
            'variance_ratio': res['variance_ratio'],
            'optimal_temperature': opt_temp,
            'n_samples': res['n_samples'],
        }
        
        vr = res['variance_ratio']
        vr_status = "OK" if 0.8 <= vr <= 1.2 else "!!"
        
        print(f"  {bin_name:<20} | {res['gt_variance']:>12.6f} | {res['gen_variance']:>12.6f} | "
              f"{vr:>7.3f} {vr_status} | {opt_temp:>7.2f}")
    
    return temperature_map, calibration_details


def calibrate_phase3(model, bin_data: Dict, coeffs_r_mean: np.ndarray,
                     coeffs_r_std: np.ndarray, cond_scaler: StandardScaler,
                     device: torch.device, n_samples: int = 50,
                     bin_names: List[str] = None) -> Tuple[Dict, Dict]:
    
    print("\n" + "=" * 80)
    print("  Phase 3: RadiusDiffusion Temperature Calibration")
    print("=" * 80)
    
    if bin_names is None:
        bin_names = list(bin_data.keys())
    
    print("\n[Step 1] Computing variance with T=1.0...")
    variance_results = compute_variance_phase3(
        model, bin_data, coeffs_r_mean, coeffs_r_std, cond_scaler, device, 
        temperature=1.0, n_samples=n_samples
    )
    
    print("\n[Step 2] Calibrating optimal temperature...")
    temperature_map = {}
    calibration_details = {}
    
    print(f"\n  {'L-bin':<20} | {'GT Var':>12} | {'Gen Var':>12} | {'VR':>8} | {'Opt T':>8}")
    print(f"  {'-'*75}")
    
    for bin_name in bin_names:
        if bin_name not in variance_results:
            continue
        
        res = variance_results[bin_name]
        opt_temp = calibrate_temperature(res['gt_variance'], res['gen_variance'])
        temperature_map[bin_name] = opt_temp
        
        calibration_details[bin_name] = {
            'gt_variance': res['gt_variance'],
            'gen_variance_t1': res['gen_variance'],
            'variance_ratio': res['variance_ratio'],
            'optimal_temperature': opt_temp,
            'n_samples': res['n_samples'],
        }
        
        vr = res['variance_ratio']
        vr_status = "OK" if 0.8 <= vr <= 1.2 else "!!"
        
        print(f"  {bin_name:<20} | {res['gt_variance']:>12.6f} | {res['gen_variance']:>12.6f} | "
              f"{vr:>7.3f} {vr_status} | {opt_temp:>7.2f}")
    
    return temperature_map, calibration_details


def create_visualization(phase1_details: Dict, phase3_details: Dict,
                         output_dir: str, bin_names: List[str]):
    viz_dir = os.path.join(output_dir, 'calibration_viz')
    os.makedirs(viz_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    if phase1_details:
        ax1 = axes[0, 0]
        bins = [bn for bn in bin_names if bn in phase1_details]
        gt_vars = [phase1_details[bn]['gt_variance'] for bn in bins]
        gen_vars = [phase1_details[bn]['gen_variance_t1'] for bn in bins]
        
        x = np.arange(len(bins))
        width = 0.35
        ax1.bar(x - width/2, gt_vars, width, label='GT Variance', color='steelblue')
        ax1.bar(x + width/2, gen_vars, width, label='Gen Variance (T=1)', color='coral')
        ax1.set_xlabel('L-bin')
        ax1.set_ylabel('Variance')
        ax1.set_title('Phase 1: GT vs Generated Variance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(bins, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        temps = [phase1_details[bn]['optimal_temperature'] for bn in bins]
        colors = ['green' if 0.95 <= t <= 1.05 else 'orange' if 0.8 <= t <= 1.2 else 'red' for t in temps]
        ax2.bar(bins, temps, color=colors)
        ax2.axhline(y=1.0, color='black', linestyle='--', label='T=1.0')
        ax2.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5)
        ax2.axhline(y=1.2, color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel('L-bin')
        ax2.set_ylabel('Optimal Temperature')
        ax2.set_title('Phase 1: Optimal Temperature per L-bin')
        ax2.set_xticklabels(bins, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    if phase3_details:
        ax3 = axes[1, 0]
        bins = [bn for bn in bin_names if bn in phase3_details]
        gt_vars = [phase3_details[bn]['gt_variance'] for bn in bins]
        gen_vars = [phase3_details[bn]['gen_variance_t1'] for bn in bins]
        
        x = np.arange(len(bins))
        ax3.bar(x - width/2, gt_vars, width, label='GT Variance', color='steelblue')
        ax3.bar(x + width/2, gen_vars, width, label='Gen Variance (T=1)', color='coral')
        ax3.set_xlabel('L-bin')
        ax3.set_ylabel('Variance')
        ax3.set_title('Phase 3: GT vs Generated Variance')
        ax3.set_xticks(x)
        ax3.set_xticklabels(bins, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        temps = [phase3_details[bn]['optimal_temperature'] for bn in bins]
        colors = ['green' if 0.95 <= t <= 1.05 else 'orange' if 0.8 <= t <= 1.2 else 'red' for t in temps]
        ax4.bar(bins, temps, color=colors)
        ax4.axhline(y=1.0, color='black', linestyle='--', label='T=1.0')
        ax4.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5)
        ax4.axhline(y=1.2, color='gray', linestyle=':', alpha=0.5)
        ax4.set_xlabel('L-bin')
        ax4.set_ylabel('Optimal Temperature')
        ax4.set_title('Phase 3: Optimal Temperature per L-bin')
        ax4.set_xticklabels(bins, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz_path = os.path.join(viz_dir, 'calibration_summary.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Visualization saved: {viz_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage-2B Temperature Calibration")
    parser.add_argument('--phase1-ckpt', type=str, help='Phase 1 (MagnitudeDiffusion) checkpoint')
    parser.add_argument('--phase3-ckpt', type=str, help='Phase 3 (RadiusDiffusion) checkpoint')
    parser.add_argument('--npz-dir', type=str, required=True, help='NPZ directory')
    parser.add_argument('--split-files', type=str, nargs='+', required=True, help='Split CSV files')
    parser.add_argument('--gt-stats', type=str, required=True, help='gt_statistics.json path')
    parser.add_argument('--output', type=str, required=True, help='Output temperature_config.json path')
    parser.add_argument('--n-samples', type=int, default=50, help='Samples per condition')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    args = parser.parse_args()
    
    print("=" * 80)
    print("  Stage-2B Temperature Auto-Calibration")
    print("=" * 80)
    print("\n  PolarDataset integration - single source of truth")
    print("  - Data pipeline identical to training")
    print("  - Condition vector order guaranteed consistent")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    print(f"  N samples: {args.n_samples}")
    
    do_phase1 = args.phase1_ckpt is not None
    do_phase3 = args.phase3_ckpt is not None
    print(f"  Phase 1: {'Yes' if do_phase1 else 'No'}")
    print(f"  Phase 3: {'Yes' if do_phase3 else 'No'}")
    
    gt_stats = load_gt_statistics(args.gt_stats)
    l_bin_config = gt_stats["l_bin_config"]
    
    boundaries = l_bin_config["boundaries"]
    l_bin_config["boundaries"] = [
        float('inf') if (isinstance(b, str) and b.lower() == "inf") else b
        for b in boundaries
    ]
    
    bin_names = l_bin_config["bin_names"]
    
    print(f"\n[L-bin Configuration]")
    print(f"  Method: {l_bin_config.get('method', 'unknown')}")
    print(f"  Boundaries: {l_bin_config['boundaries']}")
    print(f"  Bin names: {bin_names}")
    
    print("\n" + "-" * 80)
    print("  Loading data via PolarDataset...")
    print("-" * 80)
    
    polar_r, coeffs_r, condition, L_straight = load_data_from_dataset(
        args.npz_dir, args.split_files
    )
    print(f"  Total samples: {len(polar_r)}")
    
    phase1_temperature = {}
    phase1_details = {}
    
    if do_phase1:
        print("\n" + "-" * 80)
        print("  Phase 1: Loading model and calibrating...")
        print("-" * 80)
        
        bin_data = split_by_lbin(polar_r, condition, L_straight, l_bin_config)
        
        print("\n  [L-bin Distribution]")
        for bn in bin_names:
            n = len(bin_data[bn]['target'])
            pct = n / len(polar_r) * 100 if len(polar_r) > 0 else 0
            print(f"    {bn:<20}: {n:>6} ({pct:>5.1f}%)")
        
        print(f"\n  Loading checkpoint: {args.phase1_ckpt}")
        ckpt = torch.load(args.phase1_ckpt, map_location=device, weights_only=False)
        
        config = ckpt.get('config', {})
        mag_cfg = config.get('magnitude_diffusion', {})
        hidden_dims = mag_cfg.get('hidden_dims', [256, 512, 512, 256])
        use_film = mag_cfg.get('use_film', True)
        T = mag_cfg.get('T', 500)
        
        print(f"  [Phase 1] T={T}, hidden_dims={hidden_dims}, use_film={use_film}")
        
        cond_dim = mag_cfg.get('cond_dim', CONDITION_DIM)
        model = MagnitudeDiffusion(
            output_dim=CHEBYSHEV_K, cond_dim=cond_dim,
            hidden_dims=hidden_dims, use_film=use_film, T=T
        ).to(device)
        
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        r_mean = np.array(ckpt['r_mean'])
        r_std = np.array(ckpt['r_std'])
        
        cond_scaler = StandardScaler()
        cond_scaler.mean_ = np.array(ckpt['condition_scaler']['mean_'])
        cond_scaler.scale_ = np.array(ckpt['condition_scaler']['scale_'])
        cond_scaler.var_ = np.array(ckpt['condition_scaler']['var_'])
        
        phase1_temperature, phase1_details = calibrate_phase1(
            model, bin_data, r_mean, r_std, cond_scaler, device,
            n_samples=args.n_samples, bin_names=bin_names
        )
    
    phase3_temperature = {}
    phase3_details = {}
    
    if do_phase3:
        print("\n" + "-" * 80)
        print("  Phase 3: Loading model and calibrating...")
        print("-" * 80)
        
        bin_data = split_by_lbin(coeffs_r, condition, L_straight, l_bin_config)
        
        print("\n  [L-bin Distribution]")
        for bn in bin_names:
            n = len(bin_data[bn]['target'])
            pct = n / len(coeffs_r) * 100 if len(coeffs_r) > 0 else 0
            print(f"    {bn:<20}: {n:>6} ({pct:>5.1f}%)")
        
        print(f"\n  Loading checkpoint: {args.phase3_ckpt}")
        ckpt = torch.load(args.phase3_ckpt, map_location=device, weights_only=False)
        
        config = ckpt.get('config', {})
        radius_cfg = config.get('radius_diffusion', {})
        hidden_dims = radius_cfg.get('hidden_dims', [128, 256, 256, 128])
        T = radius_cfg.get('T', 200)
        use_film = radius_cfg.get('use_film', True)
        use_weighted_loss = radius_cfg.get('use_weighted_loss', True)
        
        if 'loss_weights' not in radius_cfg:
            raise ValueError("loss_weights not found in checkpoint config")
        loss_weights = radius_cfg['loss_weights']
        
        cond_dim = radius_cfg.get('cond_dim', CONDITION_DIM)
        model = RadiusDiffusion(
            output_dim=CHEBYSHEV_K_RADIUS, cond_dim=cond_dim,
            hidden_dims=hidden_dims, T=T, use_film=use_film,
            use_weighted_loss=use_weighted_loss, loss_weights=loss_weights
        ).to(device)
        
        state_dict_key = 'diffusion_state_dict' if 'diffusion_state_dict' in ckpt else 'model_state_dict'
        model.load_state_dict(ckpt[state_dict_key])
        model.eval()
        
        coeffs_r_mean = np.array(ckpt.get('coeffs_r_mean', np.zeros(CHEBYSHEV_K_RADIUS)))
        coeffs_r_std = np.array(ckpt.get('coeffs_r_std', np.ones(CHEBYSHEV_K_RADIUS)))
        
        cond_scaler = StandardScaler()
        if 'condition_scaler' in ckpt:
            cond_scaler.mean_ = np.array(ckpt['condition_scaler']['mean_'])
            cond_scaler.scale_ = np.array(ckpt['condition_scaler']['scale_'])
            cond_scaler.var_ = np.array(ckpt['condition_scaler']['var_'])
        else:
            cond_scaler.fit(condition)
        
        phase3_temperature, phase3_details = calibrate_phase3(
            model, bin_data, coeffs_r_mean, coeffs_r_std, cond_scaler, device,
            n_samples=args.n_samples, bin_names=bin_names
        )
    
    print("\n" + "=" * 80)
    print("  Saving temperature_config.json")
    print("=" * 80)
    
    if not phase1_temperature:
        print("  [Warning] Phase 1 not calibrated - phase1_temperature will be empty")
    if not phase3_temperature:
        print("  [Warning] Phase 3 not calibrated - phase3_temperature will be empty")
    
    output_config = {
        "version": "1.0",
        "dataset": os.path.basename(os.path.dirname(args.npz_dir)),
        "created_at": datetime.now().isoformat(),
        "calibration_method": "variance_ratio",
        "n_samples_per_condition": args.n_samples,
        
        "phase1_checkpoint": args.phase1_ckpt if args.phase1_ckpt else None,
        "phase3_checkpoint": args.phase3_ckpt if args.phase3_ckpt else None,
        "gt_statistics_path": args.gt_stats,
        
        "l_bin_config": {
            "method": l_bin_config.get("method", "unknown"),
            "n_bins": len(bin_names),
            "boundaries": [str(b) if b == float('inf') else b for b in l_bin_config["boundaries"]],
            "bin_names": bin_names,
        },
        
        "phase1_temperature": phase1_temperature,
        "phase3_temperature": phase3_temperature,
        
        "calibration_details": {
            "phase1": phase1_details if phase1_details else None,
            "phase3": phase3_details if phase3_details else None,
        }
    }
    
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(output_config, f, indent=2)
    
    print(f"\n  Saved: {args.output}")
    
    print("\n  [Phase 1 Temperature Map]")
    if phase1_temperature:
        for bn in bin_names:
            if bn in phase1_temperature:
                print(f"    {bn}: {phase1_temperature[bn]}")
    else:
        print("    (Not calibrated)")
    
    print("\n  [Phase 3 Temperature Map]")
    if phase3_temperature:
        for bn in bin_names:
            if bn in phase3_temperature:
                print(f"    {bn}: {phase3_temperature[bn]}")
    else:
        print("    (Not calibrated)")
    
    if not args.no_viz and (phase1_details or phase3_details):
        create_visualization(phase1_details, phase3_details, output_dir or '.', bin_names)
    
    print("\n" + "=" * 80)
    print("  Temperature calibration complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
