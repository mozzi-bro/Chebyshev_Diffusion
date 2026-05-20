
import os
import sys
import argparse
import random
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from datasets.vessel_dataset import (
    PolarDataset, polar_collate_fn, compute_polar_statistics, CHEBYSHEV_K,
    CHEBYSHEV_K_RADIUS, StratifiedBatchSampler,
    load_stratified_config,
    load_l_bin_config, get_l_bin_dynamic,
)
from models.curve_diffusion import (
    MagnitudeDiffusion, ThetaDiffusion, RadiusDiffusion
)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if "paths" not in config:
        return config
    
    dataset = config.get("dataset", "lca")
    paths = config["paths"]
    

    for key in list(paths.keys()):
        if isinstance(paths[key], str):
            paths[key] = paths[key].replace("${dataset}", dataset)
    
    root_dir = paths.get("root_dir", ".")
    for key in list(paths.keys()):
        if isinstance(paths[key], str):
            paths[key] = paths[key].replace("${paths.root_dir}", root_dir)
    
    data_dir = paths.get("data_dir", os.path.join(root_dir, "data", dataset))
    for key in list(paths.keys()):
        if isinstance(paths[key], str):
            paths[key] = paths[key].replace("${paths.data_dir}", data_dir)
    
    npz_dir = paths.get("npz_dir", os.path.join(data_dir, "recipe_npz"))
    for key in list(paths.keys()):
        if isinstance(paths[key], str):
            paths[key] = paths[key].replace("${paths.npz_dir}", npz_dir)
    
    statistics_dir = paths.get("statistics_dir", os.path.join(data_dir, "statistics"))
    for key in list(paths.keys()):
        if isinstance(paths[key], str):
            paths[key] = paths[key].replace("${paths.statistics_dir}", statistics_dir)
    
    output_dir = paths.get("output_dir", os.path.join(root_dir, "outputs", dataset, "stage2b"))
    for key in list(paths.keys()):
        if isinstance(paths[key], str):
            paths[key] = paths[key].replace("${paths.output_dir}", output_dir)
    
    checkpoint_dir = paths.get("checkpoint_dir", os.path.join(output_dir, "checkpoints"))
    for key in list(paths.keys()):
        if isinstance(paths[key], str):
            paths[key] = paths[key].replace("${paths.checkpoint_dir}", checkpoint_dir)
    
    config["dataset"] = dataset
    
    return config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_generative_metrics_ablation_style(
    all_samples: np.ndarray,
    val_target: np.ndarray,
    K: int,
) -> Dict[str, Any]:
    N = val_target.shape[0]
    n_samples = all_samples.shape[0]
    
    sample_means = all_samples.mean(axis=0)
    
    wasserstein_per_k = []
    vr_per_k = []
    coverage_per_k = []
    mean_error_per_k = []
    std_error_per_k = []
    
    for k in range(K):
        gt_k = val_target[:, k]
        pred_mean_k = sample_means[:, k]
        all_samples_k = all_samples[:, :, k].flatten()
        
        marginal_vr = np.std(all_samples_k) / (np.std(gt_k) + 1e-10)
        vr_per_k.append(marginal_vr)
        
        p5 = np.percentile(all_samples[:, :, k], 5, axis=0)
        p95 = np.percentile(all_samples[:, :, k], 95, axis=0)
        coverage = np.mean((gt_k >= p5) & (gt_k <= p95))
        coverage_per_k.append(coverage)
        
        w_dist = wasserstein_distance(gt_k, all_samples_k)
        wasserstein_per_k.append(w_dist)
        
        gt_mean, gen_mean = np.mean(gt_k), np.mean(all_samples_k)
        gt_std, gen_std = np.std(gt_k), np.std(all_samples_k)
        
        mean_err = abs(gen_mean - gt_mean) / (abs(gt_mean) + 1e-10)
        std_err = abs(gen_std - gt_std) / (gt_std + 1e-10)
        
        mean_error_per_k.append(mean_err)
        std_error_per_k.append(std_err)
    
    results = {
        "wasserstein_per_k": np.array(wasserstein_per_k),
        "vr_per_k": np.array(vr_per_k),
        "coverage_per_k": np.array(coverage_per_k),
        "mean_error_per_k": np.array(mean_error_per_k),
        "std_error_per_k": np.array(std_error_per_k),
        "avg_wasserstein": np.mean(wasserstein_per_k),
        "avg_vr": np.mean(vr_per_k),
        "avg_coverage": np.mean(coverage_per_k),
        "avg_mean_error": np.mean(mean_error_per_k),
        "avg_std_error": np.mean(std_error_per_k),
    }
    
    return results


def estimate_kappa(theta_samples: np.ndarray) -> float:
    cos_mean = np.mean(np.cos(theta_samples))
    sin_mean = np.mean(np.sin(theta_samples))
    R_bar = np.sqrt(cos_mean**2 + sin_mean**2)
    
    if R_bar < 0.53:
        kappa = 2 * R_bar + R_bar**3 + 5 * R_bar**5 / 6
    elif R_bar < 0.85:
        kappa = -0.4 + 1.39 * R_bar + 0.43 / (1 - R_bar)
    else:
        kappa = 1 / (R_bar**3 - 4 * R_bar**2 + 3 * R_bar + 1e-8)
    
    return max(0, kappa)


class Phase1Trainer:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(
            config["training"].get("device", "cuda") 
            if torch.cuda.is_available() else "cpu"
        )
        
        self.output_dir = Path(config["paths"]["output_dir"])
        self.checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[Phase 1] Output dir: {self.output_dir}")
        print(f"[Phase 1] Checkpoint dir: {self.checkpoint_dir}")
        
        log_dir = self.output_dir / "tensorboard" / f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=str(log_dir))
        print(f"[Phase 1] TensorBoard log dir: {log_dir}")
        
        self._setup_data()
        self._setup_model()
        self._setup_training()
        
        self.history = {
            "train_loss": [], 
            "val_loss": [], 
            "wasserstein": [],
            "var_ratio": [],
            "coverage_95": [],
        }
        self.best_wasserstein = float("inf")
        self.best_vr = 0.0
        self.best_coverage = 0.0
        self.patience_counter = 0
    
    def _setup_data(self):
        print("\n[Phase 1] Setting up data...")
        
        npz_dir = self.config["paths"]["npz_dir"]
        train_split = self.config["paths"]["train_split"]
        val_split = self.config["paths"]["val_split"]
        
        if not os.path.exists(npz_dir):
            raise FileNotFoundError(f"NPZ directory not found: {npz_dir}")
        if not os.path.exists(train_split):
            raise FileNotFoundError(f"Train split file not found: {train_split}")
        if not os.path.exists(val_split):
            raise FileNotFoundError(f"Val split file not found: {val_split}")
        
        print(f"  NPZ dir: {npz_dir}")
        print(f"  Train split: {train_split}")
        print(f"  Val split: {val_split}")
        
        self.train_dataset = PolarDataset(npz_dir, train_split)
        self.val_dataset = PolarDataset(npz_dir, val_split)
        
        diff_cfg = self.config.get("magnitude_diffusion", self.config.get("diffusion", {}))
        train_cfg = self.config["training"]
        
        batch_size = diff_cfg.get("batch_size", 256)
        
        use_stratified = diff_cfg.get("stratified_sampling", False)

        if use_stratified:
            gt_stats_path = self.config.get("paths", {}).get("gt_statistics", None)

            if not gt_stats_path:
                raise ValueError(
                    "paths.gt_statistics is required for stratified_sampling.\n"
                    "Add to config: paths.gt_statistics: 'path/to/gt_statistics.json'"
                )

            stratified_config = load_stratified_config(gt_stats_path)
            print(f"\n  Stratified Sampling (dynamic from JSON)")
            print(f"    Loaded from: {gt_stats_path}")
            
            stratified_sampler = StratifiedBatchSampler(
                dataset=self.train_dataset,
                batch_size=batch_size,
                stratified_config=stratified_config,
                seed=train_cfg.get("seed", 42),
            )
            
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_sampler=stratified_sampler,
                collate_fn=polar_collate_fn,
                num_workers=train_cfg.get("num_workers", 4),
                pin_memory=train_cfg.get("pin_memory", True),
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=polar_collate_fn,
                num_workers=train_cfg.get("num_workers", 4),
                pin_memory=train_cfg.get("pin_memory", True),
            )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=polar_collate_fn,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=train_cfg.get("pin_memory", True),
        )

        print(f"\n  Train samples: {len(self.train_dataset):,}")
        print(f"  Val samples: {len(self.val_dataset):,}")
        print(f"  Batch size: {batch_size}")
        print(f"  K: {CHEBYSHEV_K}")
        print(f"  Stratified Sampling: {'ON' if use_stratified else 'OFF'}")

        self._fit_preprocessors()

        sample_batch = next(iter(self.val_loader))
        actual_cond_dim = sample_batch['condition'].shape[1]
        for section in ['magnitude_diffusion', 'diffusion']:
            if section in self.config:
                self.config[section]['cond_dim'] = actual_cond_dim
        print(f"  [Auto] cond_dim = {actual_cond_dim} "
              f"(hints_per_segment = {(actual_cond_dim - 5) // 2})")

    def _fit_preprocessors(self):
        print("\n[Phase 1] Fitting preprocessors...")
        
        diff_cfg = self.config.get("magnitude_diffusion", self.config.get("diffusion", {}))
        
        unbiased_loader = DataLoader(
            self.train_dataset,
            batch_size=256,
            shuffle=False,
            collate_fn=polar_collate_fn,
            num_workers=0,
        )
        stats = compute_polar_statistics(unbiased_loader)

        if diff_cfg.get("normalize_condition", True):
            self.condition_scaler = StandardScaler()
            self.condition_scaler.fit(stats["condition_all"])
            print(f"  [Condition Scaler] StandardScaler fitted")
            print(f"    condition_dim: {stats['condition_all'].shape[1]}")
        else:
            self.condition_scaler = None
            print("  [Condition Scaler] Disabled")
        
        self.r_mean = stats["polar_r_mean"]
        self.r_std = stats["polar_r_std"] + 1e-6
        
        self.r_mean_tensor = torch.tensor(self.r_mean, dtype=torch.float32, device=self.device)
        self.r_std_tensor = torch.tensor(self.r_std, dtype=torch.float32, device=self.device)

        print(f"  [Target r Scaler] Per-k normalization fitted")
        print(f"    r_mean[:4]: {self.r_mean[:4]}")
        print(f"    r_std[:4]:  {self.r_std[:4]}")
        print(f"  Target normalization active: (r - mean) / std")
    
    def _setup_model(self):
        print("\n[Phase 1] Setting up MagnitudeDiffusion model...")
        
        self.model = MagnitudeDiffusion.from_config(self.config)
        self.model = self.model.to(self.device)
        
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model parameters: {n_params:,}")
        print(f"  Output dim (r): {self.model.output_dim}")
        print(f"  Cond dim: {self.model.cond_dim}")
        print(f"  Hidden dims: {self.model.hidden_dims}")
        print(f"  Dropout: {self.model.dropout_rate}")
        print(f"  T: {self.model.T}")
        print(f"  Loss type: {self.model.loss_type}")
        if self.model.loss_type == 'huber':
            print(f"  Huber delta: {self.model.huber_delta}")
        print(f"  Use FiLM: {self.model.use_film}")
        if self.model.use_film:
            print(f"  FiLM t_dim: {self.model.film_t_dim}, c_dim: {self.model.film_c_dim}")
        print(f"  Device: {self.device}")
    
    def _setup_training(self):
        print("\n[Phase 1] Setting up training...")

        diff_cfg = self.config.get("magnitude_diffusion", self.config.get("diffusion", {}))

        lr = diff_cfg.get("lr", 1e-3)
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=diff_cfg.get("weight_decay", 0.0001),
        )
        
        epochs = diff_cfg.get("epochs", 200)
        warmup_epochs = diff_cfg.get("warmup_epochs", 0)
        
        if warmup_epochs > 0 and diff_cfg.get("lr_scheduler") == "cosine":
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=epochs - warmup_epochs,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, epochs)
        
        print(f"  Optimizer: AdamW (lr={lr})")
        print(f"  Scheduler: CosineAnnealingLR")
        print(f"  Epochs: {epochs}")
    
    def _transform_condition(self, condition: torch.Tensor) -> torch.Tensor:
        if self.condition_scaler is not None:
            condition_np = condition.cpu().numpy()
            condition_scaled = self.condition_scaler.transform(condition_np)
            return torch.tensor(condition_scaled, dtype=torch.float32, device=self.device)
        return condition.to(self.device)

    def _normalize_target(self, r: torch.Tensor) -> torch.Tensor:
        return (r - self.r_mean_tensor) / self.r_std_tensor

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            condition = self._transform_condition(batch["condition"])
            r_target = batch["polar_r"].to(self.device)

            r_target_normalized = self._normalize_target(r_target)
            
            self.optimizer.zero_grad()
            loss = self.model.training_loss(r_target_normalized, condition)
            
            loss.backward()
            
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float, float]:
        self.model.eval()
        
        diff_cfg = self.config.get("magnitude_diffusion", self.config.get("diffusion", {}))
        n_samples = diff_cfg.get("n_samples_eval", 30)
        
        total_loss = 0.0
        n_batches = 0
        
        all_samples_list = []
        all_target_list = []
        
        for batch in tqdm(self.val_loader, desc="  Validating", leave=False):
            condition = self._transform_condition(batch["condition"])
            r_target = batch["polar_r"].to(self.device)
            
            r_target_normalized = self._normalize_target(r_target)
            loss = self.model.training_loss(r_target_normalized, condition)
            total_loss += loss.item()
            n_batches += 1

            samples_normalized = self.model.sample(condition, n=n_samples)

            samples = samples_normalized * self.r_std_tensor + self.r_mean_tensor
            
            all_samples_list.append(samples.cpu().numpy())
            all_target_list.append(r_target.cpu().numpy())
        
        val_loss = total_loss / n_batches
        
        all_samples = np.concatenate(all_samples_list, axis=1)
        val_target = np.concatenate(all_target_list, axis=0)

        metrics = compute_generative_metrics_ablation_style(all_samples, val_target, CHEBYSHEV_K)
        
        avg_wasserstein = metrics["avg_wasserstein"]
        avg_vr = metrics["avg_vr"]
        avg_coverage = metrics["avg_coverage"]
        
        print(f"\n  [Generative Metrics per k (first 6)] - n_samples={n_samples}")
        print(f"  {'k':>3} {'Wasserstein':>12} {'VarRatio':>10} {'Cov95%':>10}")
        print(f"  {'-'*40}")
        for k in range(min(6, CHEBYSHEV_K)):
            print(f"  {k:>3} {metrics['wasserstein_per_k'][k]:>12.4f} "
                  f"{metrics['vr_per_k'][k]:>10.3f} {metrics['coverage_per_k'][k]:>10.3f}")
        print(f"  {'-'*40}")
        print(f"  {'Avg':>3} {avg_wasserstein:>12.4f} {avg_vr:>10.3f} {avg_coverage:>10.3f}")
        
        return val_loss, avg_wasserstein, avg_vr, avg_coverage
    
    def train(self):
        diff_cfg = self.config.get("magnitude_diffusion", self.config.get("diffusion", {}))
        train_cfg = self.config["training"]
        epochs = diff_cfg.get("epochs", 200)

        print(f"\n{'='*70}")
        print(f"[Phase 1] Starting MagnitudeDiffusion training...")
        print(f"{'='*70}")
        print(f"  - Target normalization: (r - mean) / std")
        print(f"  - Multi-sampling: n_samples={diff_cfg.get('n_samples_eval', 30)}")
        print(f"  - Hidden dims: {self.model.hidden_dims}")
        print(f"  - Dropout: {self.model.dropout_rate}")
        print(f"  - LR: {self.optimizer.param_groups[0]['lr']}")
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            self.history["train_loss"].append(train_loss)
            
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("LR", lr, epoch)
            
            if epoch % train_cfg.get("val_every_n_epochs", 10) == 0 or epoch == 1 or epoch == epochs:
                val_loss, avg_wasserstein, avg_vr, avg_coverage = self.validate()
                
                self.history["val_loss"].append(val_loss)
                self.history["wasserstein"].append(avg_wasserstein)
                self.history["var_ratio"].append(avg_vr)
                self.history["coverage_95"].append(avg_coverage)
                
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Metrics/wasserstein", avg_wasserstein, epoch)
                self.writer.add_scalar("Metrics/var_ratio", avg_vr, epoch)
                self.writer.add_scalar("Metrics/coverage_95", avg_coverage, epoch)
                
                print(f"\nEpoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | LR: {lr:.2e}")
                print(f"  Wasserstein: {avg_wasserstein:.4f} | "
                      f"VR: {avg_vr:.3f} | Coverage95: {avg_coverage:.3f}")
                
                if avg_wasserstein < self.best_wasserstein:
                    self.best_wasserstein = avg_wasserstein
                    self.best_vr = avg_vr
                    self.best_coverage = avg_coverage
                    self.patience_counter = 0
                    self._save_checkpoint("best_magnitude.pt", epoch, val_loss, avg_wasserstein, avg_vr, avg_coverage)
                    print(f"  -> New best Wasserstein: {avg_wasserstein:.4f}")
                else:
                    self.patience_counter += 1
            
            elif epoch % train_cfg.get("log_every_n_steps", 10) == 0:
                print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | LR: {lr:.2e}")
            
            if epoch % train_cfg.get("save_every_n_epochs", 50) == 0:
                self._save_checkpoint(f"magnitude_epoch_{epoch}.pt", epoch, train_loss, 0, 0, 0)
            
            if diff_cfg.get("early_stopping", False):
                if self.patience_counter >= diff_cfg.get("patience", 30):
                    print(f"\n[Phase 1] Early stopping at epoch {epoch}")
                    break
        
        self._save_checkpoint("final_magnitude.pt", epochs, train_loss, 0, 0, 0)
        self.writer.close()
        self._print_results()
    
    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float,
                         wasserstein: float, var_ratio: float, coverage: float):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "wasserstein": wasserstein,
            "var_ratio": var_ratio,
            "coverage_95": coverage,
            "config": self.config,
            "history": self.history,
            "r_mean": self.r_mean,
            "r_std": self.r_std,
        }
        
        if self.condition_scaler is not None:
            checkpoint["condition_scaler"] = {
                "mean_": self.condition_scaler.mean_,
                "scale_": self.condition_scaler.scale_,
                "var_": self.condition_scaler.var_,
            }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def _print_results(self):
        print(f"\n{'='*70}")
        print("[Phase 1] Training completed! (MagnitudeDiffusion)")
        print(f"{'='*70}")
        print(f"Best Wasserstein: {self.best_wasserstein:.4f} (target: ~0.085)")
        print(f"Best Var Ratio:   {self.best_vr:.3f} (target: ~0.94)")
        print(f"Best Coverage95:  {self.best_coverage:.3f} (target: ~0.89)")
        print(f"\nCheckpoints saved to: {self.checkpoint_dir}")
        print(f"TensorBoard logs: {self.output_dir / 'tensorboard'}")


class Phase2Trainer:
    
    def __init__(self, config: Dict[str, Any], phase1_checkpoint: Optional[str] = None):
        self.config = config
        self.phase1_checkpoint = phase1_checkpoint
        self.device = torch.device(
            config["training"].get("device", "cuda") 
            if torch.cuda.is_available() else "cpu"
        )
        
        self.output_dir = Path(config["paths"]["output_dir"])
        self.checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        log_dir = self.output_dir / "tensorboard" / f"phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=str(log_dir))
        
        self._setup_data()
        self._setup_model()
        self._setup_training()
        
        self.history = {
            "train_loss": [], 
            "val_loss": [], 
            "val_kappa_error": [],
        }
        self.best_val_loss = float("inf")
        self.best_kappa_error = float("inf")
        self.patience_counter = 0
    
    def _setup_data(self):
        print("\n[Phase 2] Setting up data...")
        
        npz_dir = self.config["paths"]["npz_dir"]
        train_split = self.config["paths"]["train_split"]
        val_split = self.config["paths"]["val_split"]
        
        self.train_dataset = PolarDataset(npz_dir, train_split)
        self.val_dataset = PolarDataset(npz_dir, val_split)
        
        diff_cfg = self.config.get("theta_diffusion", self.config.get("diffusion", {}))
        train_cfg = self.config["training"]
        
        batch_size = diff_cfg.get("batch_size", 256)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=polar_collate_fn,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=train_cfg.get("pin_memory", True),
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=polar_collate_fn,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=train_cfg.get("pin_memory", True),
        )
        
        print(f"  Train samples: {len(self.train_dataset):,}")
        print(f"  Val samples: {len(self.val_dataset):,}")
        
        gt_stats_path = self.config.get("paths", {}).get("gt_statistics", None)
        if not gt_stats_path:
            raise ValueError(
                "paths.gt_statistics is required for Phase 2 validation.\n"
                "Add to config: paths.gt_statistics: 'path/to/gt_statistics.json'"
            )
        self.l_bin_config = load_l_bin_config(gt_stats_path)
        print(f"  L-bin config loaded from: {gt_stats_path}")
        print(f"    Bins: {self.l_bin_config['bin_names']}")

        self._fit_preprocessors()

        sample_batch = next(iter(self.val_loader))
        actual_cond_dim = sample_batch['condition'].shape[1]
        for section in ['theta_diffusion', 'diffusion']:
            if section in self.config:
                self.config[section]['cond_dim'] = actual_cond_dim
        print(f"  [Auto] cond_dim = {actual_cond_dim} "
              f"(hints_per_segment = {(actual_cond_dim - 5) // 2})")

    def _fit_preprocessors(self):
        print("\n[Phase 2] Fitting preprocessors...")
        
        diff_cfg = self.config.get("theta_diffusion", self.config.get("diffusion", {}))
        
        unbiased_loader = DataLoader(
            self.train_dataset,
            batch_size=256,
            shuffle=False,
            collate_fn=polar_collate_fn,
            num_workers=0,
        )
        stats = compute_polar_statistics(unbiased_loader)
        
        if diff_cfg.get("normalize_condition", True):
            self.condition_scaler = StandardScaler()
            self.condition_scaler.fit(stats["condition_all"])
            print(f"  [Condition Scaler] StandardScaler fitted")
        else:
            self.condition_scaler = None
    
    def _setup_model(self):
        print("\n[Phase 2] Setting up ThetaDiffusion model...")
        
        self.diffusion = ThetaDiffusion.from_config(self.config)
        self.diffusion = self.diffusion.to(self.device)
        
        n_params = sum(p.numel() for p in self.diffusion.parameters())
        print(f"  Model parameters: {n_params:,}")
        print(f"  Output dim: {self.diffusion.output_dim}")
        print(f"  T: {self.diffusion.T}")
    
    def _setup_training(self):
        print("\n[Phase 2] Setting up training...")
        
        diff_cfg = self.config.get("theta_diffusion", self.config.get("diffusion", {}))
        lr = diff_cfg.get("lr", 1e-3)
        
        self.optimizer = AdamW(
            self.diffusion.parameters(),
            lr=lr,
            weight_decay=diff_cfg.get("weight_decay", 0.0001),
        )
        
        epochs = diff_cfg.get("epochs", 200)
        self.scheduler = CosineAnnealingLR(self.optimizer, epochs)
        
        print(f"  Optimizer: AdamW (lr={lr})")
        print(f"  Scheduler: CosineAnnealingLR")
        print(f"  Epochs: {epochs}")
    
    def _transform_condition(self, condition: torch.Tensor) -> torch.Tensor:
        if self.condition_scaler is not None:
            condition_np = condition.cpu().numpy()
            condition_scaled = self.condition_scaler.transform(condition_np)
            return torch.tensor(condition_scaled, dtype=torch.float32, device=self.device)
        return condition.to(self.device)

    def train_epoch(self) -> float:
        self.diffusion.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in self.train_loader:
            condition = self._transform_condition(batch["condition"])
            theta_0 = batch["polar_theta"][:, 0].to(self.device)
            
            self.optimizer.zero_grad()
            loss = self.diffusion.training_loss(theta_0, condition)
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.diffusion.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, Dict]:
        self.diffusion.eval()
        
        total_loss = 0.0
        n_batches = 0
        bin_names = self.l_bin_config["bin_names"]
        theta_by_bin = {name: [] for name in bin_names}
        
        for batch in self.val_loader:
            condition = self._transform_condition(batch["condition"])
            theta_0 = batch["polar_theta"][:, 0].to(self.device)
            L_straight = batch["L_straight"].numpy()
            
            loss = self.diffusion.training_loss(theta_0, condition)
            total_loss += loss.item()
            n_batches += 1
            
            theta_0_np = theta_0.cpu().numpy()
            for i, L in enumerate(L_straight):
                bin_name = get_l_bin_dynamic(L, self.l_bin_config)
                theta_by_bin[bin_name].append(theta_0_np[i])
        
        val_loss = total_loss / n_batches
        
        kappa_per_bin = {}
        kappa_errors = []
        
        print("\n  [kappa Evaluation per L-bin]")
        print(f"  {'L-bin':<12} {'GT kappa':>8} {'Gen kappa':>8} {'Error':>8}")
        print(f"  {'-'*40}")

        for bin_name in bin_names:
            gt_thetas = theta_by_bin[bin_name]
            if len(gt_thetas) < 10:
                continue
            
            gt_kappa = estimate_kappa(np.array(gt_thetas))
            n_samples = min(len(gt_thetas), 500)
            
            bin_conditions = []
            for batch in self.val_loader:
                L_straight = batch["L_straight"].numpy()
                condition = batch["condition"]
                for i, L in enumerate(L_straight):
                    if get_l_bin_dynamic(L, self.l_bin_config) == bin_name and len(bin_conditions) < n_samples:
                        bin_conditions.append(condition[i])
                if len(bin_conditions) >= n_samples:
                    break
            
            if len(bin_conditions) < 10:
                continue
            
            bin_conditions = torch.stack(bin_conditions)
            bin_conditions_scaled = self._transform_condition(bin_conditions)
            
            gen_theta = self.diffusion.sample(bin_conditions_scaled, n=1).cpu().numpy()
            
            gen_kappa = estimate_kappa(gen_theta)
            error = abs(gen_kappa - gt_kappa)
            kappa_errors.append(error)
            
            kappa_per_bin[bin_name] = {
                "gt_kappa": gt_kappa,
                "gen_kappa": gen_kappa,
                "error": error,
            }
            
            print(f"  {bin_name:<12} {gt_kappa:>8.3f} {gen_kappa:>8.3f} {error:>8.3f}")
        
        avg_kappa_error = np.mean(kappa_errors) if kappa_errors else 0.0
        print(f"  {'-'*40}")
        print(f"  {'Average':<12} {'':<8} {'':<8} {avg_kappa_error:>8.3f}")
        
        return val_loss, avg_kappa_error, kappa_per_bin
    
    def train(self):
        diff_cfg = self.config.get("theta_diffusion", self.config.get("diffusion", {}))
        train_cfg = self.config["training"]
        epochs = diff_cfg.get("epochs", 200)

        print(f"\n{'='*70}")
        print(f"[Phase 2] Starting ThetaDiffusion training...")
        print(f"{'='*70}")
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            self.history["train_loss"].append(train_loss)
            
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("LR", lr, epoch)
            
            if epoch % train_cfg.get("log_every_n_steps", 10) == 0 or epoch == 1:
                print(f"\nEpoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | LR: {lr:.2e}")
            
            if epoch % train_cfg.get("val_every_n_epochs", 20) == 0 or epoch == epochs:
                val_loss, avg_kappa_error, kappa_per_bin = self.validate()
                
                self.history["val_loss"].append(val_loss)
                self.history["val_kappa_error"].append(avg_kappa_error)
                
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Kappa/avg_error", avg_kappa_error, epoch)
                
                for bin_name, stats in kappa_per_bin.items():
                    safe_name = bin_name.replace("<", "lt").replace(">", "gt").replace("-", "_")
                    self.writer.add_scalar(f"Kappa_GT/{safe_name}", stats["gt_kappa"], epoch)
                    self.writer.add_scalar(f"Kappa_Gen/{safe_name}", stats["gen_kappa"], epoch)
                    self.writer.add_scalar(f"Kappa_Error/{safe_name}", stats["error"], epoch)
                
                print(f"\n  Val Loss: {val_loss:.4f} | Avg kappa Error: {avg_kappa_error:.4f}")
                
                if avg_kappa_error < self.best_kappa_error:
                    self.best_kappa_error = avg_kappa_error
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint("best_theta.pt", epoch, val_loss, avg_kappa_error, kappa_per_bin)
                    print(f"  -> New best kappa Error: {avg_kappa_error:.4f}")
                else:
                    self.patience_counter += 1
            
            if epoch % train_cfg.get("save_every_n_epochs", 50) == 0:
                self._save_checkpoint(f"theta_epoch_{epoch}.pt", epoch, train_loss, 0, {})
            
            if diff_cfg.get("early_stopping", False):
                if self.patience_counter >= diff_cfg.get("patience", 20):
                    print(f"\n[Phase 2] Early stopping at epoch {epoch}")
                    break
        
        self._save_checkpoint("final_theta.pt", epochs, train_loss, 0, {})
        self.writer.close()
        self._print_results()
    
    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float,
                         kappa_error: float, kappa_per_bin: Dict):
        checkpoint = {
            "epoch": epoch,
            "diffusion_state_dict": self.diffusion.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "kappa_error": kappa_error,
            "kappa_per_bin": kappa_per_bin,
            "config": self.config,
            "history": self.history,
            "phase1_checkpoint": str(self.phase1_checkpoint) if self.phase1_checkpoint else None,
        }
        
        if self.condition_scaler is not None:
            checkpoint["condition_scaler"] = {
                "mean_": self.condition_scaler.mean_,
                "scale_": self.condition_scaler.scale_,
                "var_": self.condition_scaler.var_,
            }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def _print_results(self):
        print(f"\n{'='*70}")
        print("[Phase 2] Training completed!")
        print(f"{'='*70}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best kappa Error:  {self.best_kappa_error:.4f}")
        print(f"\nCheckpoints saved to: {self.checkpoint_dir}")
        print(f"TensorBoard logs: {self.output_dir / 'tensorboard'}")
        print(f"\nNext steps:")
        print(f"  1. Load best_magnitude.pt for Phase 1 (MagnitudeDiffusion)")
        print(f"  2. Load best_theta.pt for Phase 2 (ThetaDiffusion)")
        print(f"  3. Use ThetaSampler for theta[1:{CHEBYSHEV_K-1}]")
        print(f"  4. Combine r (Phase 1) + theta (Phase 2 + Sampler) -> N, B coefficients")


class Phase3Trainer:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(
            config["training"].get("device", "cuda") 
            if torch.cuda.is_available() else "cpu"
        )
        
        self.output_dir = Path(config["paths"]["output_dir"])
        self.checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[Phase 3] Output dir: {self.output_dir}")
        print(f"[Phase 3] Checkpoint dir: {self.checkpoint_dir}")
        
        log_dir = self.output_dir / "tensorboard" / f"phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=str(log_dir))
        print(f"[Phase 3] TensorBoard log dir: {log_dir}")
        
        self._setup_data()
        self._setup_model()
        self._setup_training()
        
        self.history = {
            "train_loss": [], 
            "val_loss": [], 
            "per_k_corr": [],
        }
        self.best_val_loss = float("inf")
        self.patience_counter = 0
    
    def _setup_data(self):
        print("\n[Phase 3] Setting up data...")
        
        npz_dir = self.config["paths"]["npz_dir"]
        train_split = self.config["paths"]["train_split"]
        val_split = self.config["paths"]["val_split"]
        
        if not os.path.exists(npz_dir):
            raise FileNotFoundError(f"NPZ directory not found: {npz_dir}")
        if not os.path.exists(train_split):
            raise FileNotFoundError(f"Train split file not found: {train_split}")
        if not os.path.exists(val_split):
            raise FileNotFoundError(f"Val split file not found: {val_split}")
        
        print(f"  NPZ dir: {npz_dir}")
        print(f"  Train split: {train_split}")
        print(f"  Val split: {val_split}")
        
        self.train_dataset = PolarDataset(npz_dir, train_split)
        self.val_dataset = PolarDataset(npz_dir, val_split)
        
        diff_cfg = self.config.get("radius_diffusion", {})
        train_cfg = self.config["training"]
        
        batch_size = diff_cfg.get("batch_size", 128)
        
        use_stratified = diff_cfg.get("stratified_sampling", True)

        if use_stratified:
            gt_stats_path = self.config.get("paths", {}).get("gt_statistics", None)

            if not gt_stats_path:
                raise ValueError(
                    "paths.gt_statistics is required for stratified_sampling.\n"
                    "Add to config: paths.gt_statistics: 'path/to/gt_statistics.json'"
                )

            stratified_config = load_stratified_config(gt_stats_path)
            print(f"\n  Stratified Sampling (dynamic from JSON)")
            print(f"    Loaded from: {gt_stats_path}")
            
            stratified_sampler = StratifiedBatchSampler(
                dataset=self.train_dataset,
                batch_size=batch_size,
                stratified_config=stratified_config,
                seed=train_cfg.get("seed", 42),
            )
            
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_sampler=stratified_sampler,
                collate_fn=polar_collate_fn,
                num_workers=train_cfg.get("num_workers", 4),
                pin_memory=train_cfg.get("pin_memory", True),
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=polar_collate_fn,
                num_workers=train_cfg.get("num_workers", 4),
                pin_memory=train_cfg.get("pin_memory", True),
            )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=polar_collate_fn,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=train_cfg.get("pin_memory", True),
        )

        print(f"\n  Train samples: {len(self.train_dataset):,}")
        print(f"  Val samples: {len(self.val_dataset):,}")
        print(f"  Batch size: {batch_size}")
        print(f"  K (radius coeffs): {CHEBYSHEV_K_RADIUS}")
        print(f"  Stratified Sampling: {'ON' if use_stratified else 'OFF'}")

        self._fit_preprocessors()

        sample_batch = next(iter(self.val_loader))
        actual_cond_dim = sample_batch['condition'].shape[1]
        if 'radius_diffusion' in self.config:
            self.config['radius_diffusion']['cond_dim'] = actual_cond_dim
        print(f"  [Auto] cond_dim = {actual_cond_dim} "
              f"(hints_per_segment = {(actual_cond_dim - 5) // 2})")

    def _fit_preprocessors(self):
        print("\n[Phase 3] Fitting preprocessors...")
        
        diff_cfg = self.config.get("radius_diffusion", {})
        
        unbiased_loader = DataLoader(
            self.train_dataset,
            batch_size=256,
            shuffle=False,
            collate_fn=polar_collate_fn,
            num_workers=0,
        )
        stats = compute_polar_statistics(unbiased_loader)
        
        if diff_cfg.get("normalize_condition", True):
            self.condition_scaler = StandardScaler()
            self.condition_scaler.fit(stats["condition_all"])
            print(f"  [Condition Scaler] StandardScaler fitted")
            print(f"    condition_dim: {stats['condition_all'].shape[1]}")
        else:
            self.condition_scaler = None
            print("  [Condition Scaler] Disabled")
        
        self.coeffs_r_mean = stats["coeffs_r_mean"]
        self.coeffs_r_std = stats["coeffs_r_std"] + 1e-6
        
        self.coeffs_r_mean_tensor = torch.tensor(
            self.coeffs_r_mean, dtype=torch.float32, device=self.device
        )
        self.coeffs_r_std_tensor = torch.tensor(
            self.coeffs_r_std, dtype=torch.float32, device=self.device
        )
        
        print(f"  [Target coeffs_r Scaler] Per-k Z-score normalization fitted")
        print(f"    coeffs_r_mean: {self.coeffs_r_mean}")
        print(f"    coeffs_r_std:  {self.coeffs_r_std}")
        print(f"  Per-k normalization active: (coeffs_r - mean) / std")
    
    def _setup_model(self):
        print("\n[Phase 3] Setting up RadiusDiffusion model...")
        diff_cfg = self.config.get("radius_diffusion", {})
        use_weighted = diff_cfg.get("use_weighted_loss", True)
        current_weights = diff_cfg.get("loss_weights", None)
        
        if use_weighted and current_weights is None:
            gt_stats_path = self.config.get("paths", {}).get("gt_statistics")
            if gt_stats_path and os.path.exists(gt_stats_path):
                import json as json_module
                with open(gt_stats_path, 'r') as f:
                    gt_stats = json_module.load(f)
                loss_weights = gt_stats.get("coeffs_r_inverse_variance_weights")
                if loss_weights:
                    if "radius_diffusion" not in self.config:
                        self.config["radius_diffusion"] = {}
                    self.config["radius_diffusion"]["loss_weights"] = loss_weights
                    print(f"  Auto-loaded loss_weights from: {gt_stats_path}")
                    print(f"    loss_weights: {[f'{w:.4f}' for w in loss_weights]}")
                else:
                    print(f"  [WARNING] coeffs_r_inverse_variance_weights not found in {gt_stats_path}")
            else:
                print(f"  [WARNING] gt_statistics not found at: {gt_stats_path}")
        
        self.diffusion = RadiusDiffusion.from_config(self.config)
        self.diffusion = self.diffusion.to(self.device)
        
        n_params = sum(p.numel() for p in self.diffusion.parameters())
        print(f"  Model parameters: {n_params:,}")
        print(f"  Output dim (coeffs_r): {self.diffusion.output_dim}")
        print(f"  Cond dim: {self.diffusion.cond_dim}")
        print(f"  Hidden dims: {self.diffusion.hidden_dims}")
        print(f"  Dropout: {self.diffusion.dropout_rate}")
        print(f"  T: {self.diffusion.T}")
        print(f"  Use FiLM: {self.diffusion.use_film}")
        print(f"  Use Weighted Loss: {self.diffusion.use_weighted_loss}")
        if self.diffusion.use_weighted_loss:
            print(f"  Loss weights: {self.diffusion.loss_weights.tolist()}")
        print(f"  Device: {self.device}")
    
    def _setup_training(self):
        print("\n[Phase 3] Setting up training...")
        
        diff_cfg = self.config.get("radius_diffusion", {})
        
        lr = diff_cfg.get("lr", 0.001)
        
        self.optimizer = AdamW(
            self.diffusion.parameters(),
            lr=lr,
            weight_decay=diff_cfg.get("weight_decay", 0.0001),
        )
        
        epochs = diff_cfg.get("epochs", 200)
        self.scheduler = CosineAnnealingLR(self.optimizer, epochs)
        
        print(f"  Optimizer: AdamW (lr={lr})")
        print(f"  Scheduler: CosineAnnealingLR")
        print(f"  Epochs: {epochs}")
    
    def _transform_condition(self, condition: torch.Tensor) -> torch.Tensor:
        if self.condition_scaler is not None:
            condition_np = condition.cpu().numpy()
            condition_scaled = self.condition_scaler.transform(condition_np)
            return torch.tensor(condition_scaled, dtype=torch.float32, device=self.device)
        return condition.to(self.device)

    def _normalize_target(self, coeffs_r: torch.Tensor) -> torch.Tensor:
        return (coeffs_r - self.coeffs_r_mean_tensor) / self.coeffs_r_std_tensor

    def train_epoch(self) -> float:
        self.diffusion.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            condition = self._transform_condition(batch["condition"])
            coeffs_r = batch["coeffs_r"].to(self.device)

            coeffs_r_normalized = self._normalize_target(coeffs_r)
            
            self.optimizer.zero_grad()
            loss = self.diffusion.training_loss(coeffs_r_normalized, condition)
            
            loss.backward()
            
            nn.utils.clip_grad_norm_(self.diffusion.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        self.diffusion.eval()
        
        diff_cfg = self.config.get("radius_diffusion", {})
        n_samples = diff_cfg.get("n_samples_eval", 30)
        
        total_loss = 0.0
        n_batches = 0
        
        all_samples_list = []
        all_target_list = []
        
        for batch in tqdm(self.val_loader, desc="  Validating", leave=False):
            condition = self._transform_condition(batch["condition"])
            coeffs_r = batch["coeffs_r"].to(self.device)
            
            coeffs_r_normalized = self._normalize_target(coeffs_r)
            loss = self.diffusion.training_loss(coeffs_r_normalized, condition)
            total_loss += loss.item()
            n_batches += 1

            samples_normalized = self.diffusion.sample(condition, n=n_samples)
            
            samples = samples_normalized * self.coeffs_r_std_tensor + self.coeffs_r_mean_tensor
            
            all_samples_list.append(samples.cpu().numpy())
            all_target_list.append(coeffs_r.cpu().numpy())
        
        val_loss = total_loss / n_batches
        
        all_samples = np.concatenate(all_samples_list, axis=1)
        val_target = np.concatenate(all_target_list, axis=0)
        
        metrics = self._compute_per_k_metrics(all_samples, val_target)
        
        print(f"\n  [Phase 3 Generative Metrics] n_samples={n_samples}")
        print(f"  {'k':>3} {'GT Mean':>10} {'Gen Mean':>10} {'Corr':>10}")
        print(f"  {'-'*40}")
        for k in range(CHEBYSHEV_K_RADIUS):
            gt_mean = val_target[:, k].mean()
            gen_mean = all_samples[:, :, k].mean()
            corr = metrics[f'corr_k{k}']
            print(f"  {k:>3} {gt_mean:>10.4f} {gen_mean:>10.4f} {corr:>10.4f}")
        print(f"  {'-'*40}")
        print(f"  {'Avg':>3} {'':>10} {'':>10} {metrics['avg_corr']:>10.4f}")
        
        return val_loss, metrics
    
    def _compute_per_k_metrics(
        self, 
        all_samples: np.ndarray,
        val_target: np.ndarray,
    ) -> Dict[str, float]:
        metrics = {}
        correlations = []
        
        for k in range(CHEBYSHEV_K_RADIUS):
            gt_k = val_target[:, k]
            gen_k_mean = all_samples[:, :, k].mean(axis=0)
            
            if np.std(gt_k) > 1e-10 and np.std(gen_k_mean) > 1e-10:
                corr = np.corrcoef(gt_k, gen_k_mean)[0, 1]
            else:
                corr = 0.0
            
            metrics[f'corr_k{k}'] = corr
            correlations.append(corr)
        
        metrics['avg_corr'] = np.mean(correlations)
        
        return metrics
    
    def train(self):
        diff_cfg = self.config.get("radius_diffusion", {})
        train_cfg = self.config["training"]
        epochs = diff_cfg.get("epochs", 200)

        print(f"\n{'='*70}")
        print(f"[Phase 3] Starting RadiusDiffusion training...")
        print(f"{'='*70}")
        print(f"  6D direct learning")
        print(f"  - Per-k normalization: (coeffs_r - mean) / std")
        print(f"  - Multi-sampling: n_samples={diff_cfg.get('n_samples_eval', 30)}")
        print(f"  - Hidden dims: {self.diffusion.hidden_dims}")
        print(f"  - T: {self.diffusion.T}")
        print(f"  - LR: {self.optimizer.param_groups[0]['lr']}")
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            self.history["train_loss"].append(train_loss)
            
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("LR", lr, epoch)
            
            if epoch % train_cfg.get("val_every_n_epochs", 10) == 0 or epoch == 1 or epoch == epochs:
                val_loss, metrics = self.validate()
                
                self.history["val_loss"].append(val_loss)
                self.history["per_k_corr"].append(metrics['avg_corr'])
                
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Metrics/avg_corr", metrics['avg_corr'], epoch)
                
                for k in range(CHEBYSHEV_K_RADIUS):
                    self.writer.add_scalar(f"Metrics/corr_k{k}", metrics[f'corr_k{k}'], epoch)
                
                print(f"\nEpoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | LR: {lr:.2e}")
                print(f"  Avg Corr: {metrics['avg_corr']:.4f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint("best_radius.pt", epoch, val_loss, metrics)
                    print(f"  -> New best Val Loss: {val_loss:.6f}")
                else:
                    self.patience_counter += 1
            
            elif epoch % train_cfg.get("log_every_n_steps", 10) == 0:
                print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.6f} | LR: {lr:.2e}")
            
            if epoch % train_cfg.get("save_every_n_epochs", 50) == 0:
                self._save_checkpoint(f"radius_epoch_{epoch}.pt", epoch, train_loss, {})
            
            if diff_cfg.get("early_stopping", True):
                if self.patience_counter >= diff_cfg.get("patience", 30):
                    print(f"\n[Phase 3] Early stopping at epoch {epoch}")
                    break
        
        self._save_checkpoint("final_radius.pt", epochs, train_loss, {})
        self.writer.close()
        self._print_results()
    
    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float, metrics: Dict):
        checkpoint = {
            "epoch": epoch,
            "diffusion_state_dict": self.diffusion.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "metrics": metrics,
            "config": self.config,
            "history": self.history,
            "coeffs_r_mean": self.coeffs_r_mean,
            "coeffs_r_std": self.coeffs_r_std,
        }
        
        if self.condition_scaler is not None:
            checkpoint["condition_scaler"] = {
                "mean_": self.condition_scaler.mean_,
                "scale_": self.condition_scaler.scale_,
                "var_": self.condition_scaler.var_,
            }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"  Saved: {path}")
    
    def _print_results(self):
        print(f"\n{'='*70}")
        print("[Phase 3] Training completed! (RadiusDiffusion)")
        print(f"{'='*70}")
        print(f"Best Val Loss: {self.best_val_loss:.6f}")
        print(f"\nCheckpoints saved to: {self.checkpoint_dir}")
        print(f"TensorBoard logs: {self.output_dir / 'tensorboard'}")
        print(f"\nNext steps:")
        print(f"  1. Load best_magnitude.pt for Phase 1 (MagnitudeDiffusion)")
        print(f"  2. Load best_theta.pt for Phase 2 (ThetaDiffusion)")
        print(f"  3. Load best_radius.pt for Phase 3 (RadiusDiffusion)")
        print(f"  4. Use ThetaSampler for theta[1:{CHEBYSHEV_K-1}]")
        print(f"  5. Combine all phases for full vessel reconstruction")


def main():
    parser = argparse.ArgumentParser(description="Stage-2B Training")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3], help="Training phase (1=Magnitude, 2=Theta, 3=Radius)")
    parser.add_argument("--phase1_ckpt", type=str, default=None, help="Phase 1 checkpoint path (for Phase 2)")
    args = parser.parse_args()

    print("=" * 70)
    print("Stage-2B Training Script")
    print("=" * 70)
    print(f"\n  All L-bin settings dynamically loaded from gt_statistics.json")
    print(f"\n  Key features:")
    print(f"  - FiLM Conditioning: gamma*h + beta condition injection")
    print(f"  - Stratified Sampling: dynamic from gt_statistics.json")
    print(f"  - Huber Loss: robust to outliers")
    print(f"  - Phase 3: RadiusDiffusion (6D direct learning)")
    
    config = load_config(args.config)
    
    print("\n[Config] Resolved paths:")
    for key, value in config["paths"].items():
        print(f"  {key}: {value}")
    
    print(f"\n[Config] K (curve): {CHEBYSHEV_K}")
    print(f"[Config] K (radius): {CHEBYSHEV_K_RADIUS}")
    
    set_seed(config["training"].get("seed", 42))
    
    if args.phase == 1:
        trainer = Phase1Trainer(config)
        trainer.train()
    elif args.phase == 2:
        trainer = Phase2Trainer(config, args.phase1_ckpt)
        trainer.train()
    elif args.phase == 3:
        trainer = Phase3Trainer(config)
        trainer.train()


if __name__ == "__main__":
    main()
