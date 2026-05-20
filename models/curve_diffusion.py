
import math
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


CHEBYSHEV_K = 48
CHEBYSHEV_K_RADIUS = 6
CONDITION_DIM = 9

DEFAULT_HIDDEN_DIMS = [256, 512, 512, 256]
DEFAULT_TIME_EMB_DIM = 64
DEFAULT_DROPOUT = 0.1

RADIUS_HIDDEN_DIMS = [128, 256, 256, 128]


def _missing_loss_weights_error() -> str:
    return """
================================================================================
[ERROR] loss_weights is required for RadiusDiffusion
================================================================================

The loss_weights parameter must be explicitly provided.
These weights are dataset-specific and computed from coefficient variances.

To get loss_weights:

  1. From gt_statistics.json (recommended):
     stats = load_gt_statistics("data/<dataset>/statistics/gt_statistics.json")
     loss_weights = stats["coeffs_r_inverse_variance_weights"]

  2. From config (if specified):
     loss_weights = config["radius_diffusion"]["loss_weights"]

  3. From checkpoint (if loading pre-trained model):
     ckpt = torch.load("checkpoint.pt")
     loss_weights = ckpt["config"]["radius_diffusion"]["loss_weights"]

All hardcoded magic numbers have been removed.
================================================================================
"""


class SinusoidalPE(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class FiLMLayer(nn.Module):
    
    def __init__(self, t_dim: int, c_dim: int, hidden_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(c_dim, hidden_dim)
        self.beta_proj = nn.Linear(t_dim + c_dim, hidden_dim)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
    
    def forward(self, h: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(c_emb)
        combined = torch.cat([t_emb, c_emb], dim=-1)
        beta = self.beta_proj(combined)
        return (1 + gamma) * h + beta


class MagnitudeDiffusion(nn.Module):
    
    def __init__(
        self,
        output_dim: int,
        cond_dim: int,
        hidden_dims: List[int] = None,
        time_emb_dim: int = DEFAULT_TIME_EMB_DIM,
        dropout: float = DEFAULT_DROPOUT,
        T: int = 500,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        loss_type: str = 'mse',
        huber_delta: float = 1.0,
        use_film: bool = False,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = DEFAULT_HIDDEN_DIMS.copy()
        
        self.output_dim = output_dim
        self.cond_dim = cond_dim
        self.hidden_dims = hidden_dims
        self.time_emb_dim = time_emb_dim
        self.dropout_rate = dropout
        self.T = T
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.use_film = use_film
        self.input_dim = output_dim
        
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance',
                             betas * (1 - F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)) / (1 - alphas_cumprod))
        
        self.time_mlp = nn.Sequential(
            SinusoidalPE(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )
        
        if use_film:
            self._build_film_network(hidden_dims, dropout)
        else:
            self._build_concat_network(hidden_dims, dropout)
    
    def _build_concat_network(self, hidden_dims: List[int], dropout: float):
        layers = []
        prev_dim = self.output_dim + self.time_emb_dim + hidden_dims[0]
        
        for i, hd in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.LayerNorm(hd))
            layers.append(nn.GELU())
            if dropout > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = hd
        
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.net = nn.Sequential(*layers)
    
    def _build_film_network(self, hidden_dims: List[int], dropout: float):
        self.film_t_dim = self.time_emb_dim
        self.film_c_dim = hidden_dims[0]
        
        self.input_proj = nn.Linear(self.output_dim, hidden_dims[0])
        
        self.film_norms = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        self.film_linears = nn.ModuleList()
        self.film_dropouts = nn.ModuleList()
        
        for i, hd in enumerate(hidden_dims):
            in_dim = hidden_dims[0] if i == 0 else hidden_dims[i - 1]
            out_dim = hd
            
            self.film_norms.append(nn.LayerNorm(in_dim))
            self.film_layers.append(FiLMLayer(self.film_t_dim, self.film_c_dim, in_dim))
            self.film_linears.append(nn.Linear(in_dim, out_dim))
            
            if dropout > 0 and i < len(hidden_dims) - 1:
                self.film_dropouts.append(nn.Dropout(dropout))
            else:
                self.film_dropouts.append(nn.Identity())
        
        self.output_proj = nn.Linear(hidden_dims[-1], self.output_dim)
    
    def forward(self, r_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.use_film:
            return self._forward_film(r_t, t, c)
        else:
            return self._forward_concat(r_t, t, c)
    
    def _forward_concat(self, r_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([r_t, self.time_mlp(t.float()), self.cond_mlp(c)], dim=-1))
    
    def _forward_film(self, r_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t.float())
        c_emb = self.cond_mlp(c)
        
        h = self.input_proj(r_t)
        
        for norm, film, linear, drop in zip(
            self.film_norms, self.film_layers, self.film_linears, self.film_dropouts
        ):
            residual = h
            h = norm(h)
            h = film(h, t_emb, c_emb)
            h = linear(h)
            h = F.gelu(h)
            h = drop(h)
            
            if h.shape == residual.shape:
                h = h + residual
        
        return self.output_proj(h)
    
    def q_sample(self, r_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(r_0)
        return (self.sqrt_alphas_cumprod[t][:, None] * r_0 +
                self.sqrt_one_minus_alphas_cumprod[t][:, None] * noise)
    
    def training_loss(self, r_0: torch.Tensor, condition: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = r_0.shape[0]
        device = r_0.device
        
        t = torch.randint(0, self.T, (batch_size,), device=device, dtype=torch.long)
        
        if noise is None:
            noise = torch.randn_like(r_0)
        r_t = self.q_sample(r_0, t, noise)
        
        noise_pred = self(r_t, t, condition)
        
        if self.loss_type == 'huber':
            return F.huber_loss(noise_pred, noise, delta=self.huber_delta)
        return F.mse_loss(noise_pred, noise)
    
    @torch.no_grad()
    def p_sample(self, r_t: torch.Tensor, t_int: int, c: torch.Tensor) -> torch.Tensor:
        B = r_t.size(0)
        t = torch.full((B,), t_int, device=r_t.device, dtype=torch.long)
        
        noise_pred = self(r_t, t, c)
        mean = self.sqrt_recip_alphas[t_int] * (
            r_t - self.betas[t_int] / self.sqrt_one_minus_alphas_cumprod[t_int] * noise_pred
        )
        
        if t_int == 0:
            return mean
        return mean + torch.sqrt(self.posterior_variance[t_int]) * torch.randn_like(r_t)
    
    @torch.no_grad()
    def sample(self, c: torch.Tensor, n: int = 1, temperature: float = 1.0) -> torch.Tensor:
        B = c.size(0)
        device = c.device
        
        samples = []
        for _ in range(n):
            r_t = torch.randn(B, self.output_dim, device=device) * temperature
            for t in reversed(range(self.T)):
                r_t = self.p_sample(r_t, t, c)
            samples.append(r_t)
        
        if n == 1:
            return samples[0]
        return torch.stack(samples)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MagnitudeDiffusion":
        diff_cfg = config.get("magnitude_diffusion", config.get("diffusion", config))
        hidden_dims = diff_cfg.get("hidden_dims", DEFAULT_HIDDEN_DIMS.copy())
        
        return cls(
            output_dim=diff_cfg.get("input_dim", CHEBYSHEV_K),
            cond_dim=diff_cfg.get("cond_dim", CONDITION_DIM),
            hidden_dims=hidden_dims,
            time_emb_dim=diff_cfg.get("time_emb_dim", DEFAULT_TIME_EMB_DIM),
            dropout=diff_cfg.get("dropout", DEFAULT_DROPOUT),
            T=diff_cfg.get("T", 500),
            beta_start=diff_cfg.get("beta_start", 1e-4),
            beta_end=diff_cfg.get("beta_end", 0.02),
            loss_type=diff_cfg.get("loss_type", 'mse'),
            huber_delta=diff_cfg.get("huber_delta", 1.0),
            use_film=diff_cfg.get("use_film", False),
        )


class ThetaDiffusion(nn.Module):
    
    def __init__(
        self,
        cond_dim: int = CONDITION_DIM,
        hidden_dims: List[int] = None,
        time_emb_dim: int = DEFAULT_TIME_EMB_DIM,
        dropout: float = DEFAULT_DROPOUT,
        T: int = 500,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 256, 128]
        
        self.output_dim = 2
        self.input_dim = 2
        self.cond_dim = cond_dim
        self.hidden_dims = hidden_dims
        self.time_emb_dim = time_emb_dim
        self.dropout_rate = dropout
        self.T = T
        
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance',
                             betas * (1 - F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)) / (1 - alphas_cumprod))
        
        self.time_mlp = nn.Sequential(
            SinusoidalPE(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )
        
        layers = []
        prev_dim = self.output_dim + time_emb_dim + hidden_dims[0]
        
        for i, hd in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.LayerNorm(hd))
            layers.append(nn.GELU())
            if dropout > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = hd
        
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_t, self.time_mlp(t.float()), self.cond_mlp(c)], dim=-1))
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)
        return (self.sqrt_alphas_cumprod[t][:, None] * x_0 +
                self.sqrt_one_minus_alphas_cumprod[t][:, None] * noise)
    
    def training_loss(self, theta_0: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch_size = theta_0.shape[0]
        device = theta_0.device
        
        x_0 = torch.stack([torch.cos(theta_0), torch.sin(theta_0)], dim=-1)
        
        t = torch.randint(0, self.T, (batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        
        noise_pred = self(x_t, t, condition)
        return F.mse_loss(noise_pred, noise)
    
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t_int: int, c: torch.Tensor) -> torch.Tensor:
        B = x_t.size(0)
        t = torch.full((B,), t_int, device=x_t.device, dtype=torch.long)
        
        noise_pred = self(x_t, t, c)
        mean = self.sqrt_recip_alphas[t_int] * (
            x_t - self.betas[t_int] / self.sqrt_one_minus_alphas_cumprod[t_int] * noise_pred
        )
        
        if t_int == 0:
            return mean
        return mean + torch.sqrt(self.posterior_variance[t_int]) * torch.randn_like(x_t)
    
    @torch.no_grad()
    def sample(self, c: torch.Tensor, n: int = 1) -> torch.Tensor:
        B = c.size(0)
        device = c.device
        
        if n == 1:
            x_t = torch.randn(B, self.output_dim, device=device)
            for t in reversed(range(self.T)):
                x_t = self.p_sample(x_t, t, c)
            return torch.atan2(x_t[:, 1], x_t[:, 0])
        
        return torch.stack([self.sample(c, 1) for _ in range(n)])
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ThetaDiffusion":
        diff_cfg = config.get("theta_diffusion", config.get("diffusion", config))
        hidden_dims = diff_cfg.get("hidden_dims", [128, 256, 256, 128])
        
        return cls(
            cond_dim=diff_cfg.get("cond_dim", CONDITION_DIM),
            hidden_dims=hidden_dims,
            time_emb_dim=diff_cfg.get("time_emb_dim", DEFAULT_TIME_EMB_DIM),
            dropout=diff_cfg.get("dropout", DEFAULT_DROPOUT),
            T=diff_cfg.get("T", 500),
            beta_start=diff_cfg.get("beta_start", 1e-4),
            beta_end=diff_cfg.get("beta_end", 0.02),
        )


class RadiusDiffusion(nn.Module):
    
    def __init__(
        self,
        output_dim: int = CHEBYSHEV_K_RADIUS,
        cond_dim: int = CONDITION_DIM,
        hidden_dims: List[int] = None,
        time_emb_dim: int = DEFAULT_TIME_EMB_DIM,
        dropout: float = DEFAULT_DROPOUT,
        T: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        use_film: bool = True,
        use_weighted_loss: bool = True,
        loss_weights: List[float] = None,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = RADIUS_HIDDEN_DIMS.copy()
        
        if use_weighted_loss and loss_weights is None:
            raise ValueError(_missing_loss_weights_error())
        
        self.output_dim = output_dim
        self.cond_dim = cond_dim
        self.hidden_dims = hidden_dims
        self.time_emb_dim = time_emb_dim
        self.dropout_rate = dropout
        self.T = T
        self.use_film = use_film
        self.use_weighted_loss = use_weighted_loss
        
        if loss_weights is not None:
            self.register_buffer('loss_weights', torch.tensor(loss_weights, dtype=torch.float32))
        else:
            self.register_buffer('loss_weights', torch.ones(output_dim, dtype=torch.float32))
        
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance',
                             betas * (1 - F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)) / (1 - alphas_cumprod))
        
        self.time_mlp = nn.Sequential(
            SinusoidalPE(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )
        
        if use_film:
            self._build_film_network(hidden_dims, dropout)
        else:
            self._build_concat_network(hidden_dims, dropout)
    
    def _build_film_network(self, hidden_dims: List[int], dropout: float):
        self.film_t_dim = self.time_emb_dim
        self.film_c_dim = hidden_dims[0]
        
        self.input_proj = nn.Linear(self.output_dim, hidden_dims[0])
        
        self.film_layers = nn.ModuleList()
        self.main_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(len(hidden_dims)):
            in_dim = hidden_dims[0] if i == 0 else hidden_dims[i - 1]
            out_dim = hidden_dims[i]
            
            self.norms.append(nn.LayerNorm(in_dim))
            self.film_layers.append(FiLMLayer(self.film_t_dim, self.film_c_dim, in_dim))
            self.main_layers.append(nn.Linear(in_dim, out_dim))
            
            if dropout > 0 and i < len(hidden_dims) - 1:
                self.dropouts.append(nn.Dropout(dropout))
            else:
                self.dropouts.append(nn.Identity())
        
        self.output_proj = nn.Linear(hidden_dims[-1], self.output_dim)
    
    def _build_concat_network(self, hidden_dims: List[int], dropout: float):
        layers = []
        prev_dim = self.output_dim + self.time_emb_dim + hidden_dims[0]
        
        for i, hd in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.LayerNorm(hd))
            layers.append(nn.GELU())
            if dropout > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = hd
        
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.use_film:
            return self._forward_film(x_t, t, c)
        return self._forward_concat(x_t, t, c)
    
    def _forward_film(self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t.float())
        c_emb = self.cond_mlp(c)
        
        h = self.input_proj(x_t)
        
        for norm, film, main, drop in zip(self.norms, self.film_layers, self.main_layers, self.dropouts):
            residual = h
            h = norm(h)
            h = film(h, t_emb, c_emb)
            h = main(h)
            h = F.gelu(h)
            h = drop(h)
            
            if h.shape == residual.shape:
                h = h + residual
        
        return self.output_proj(h)
    
    def _forward_concat(self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t.float())
        c_emb = self.cond_mlp(c)
        return self.net(torch.cat([x_t, t_emb, c_emb], dim=-1))
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def training_loss(self, x_0: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch_size = x_0.shape[0]
        device = x_0.device
        
        t = torch.randint(0, self.T, (batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        noise_pred = self(x_t, t, condition)
        
        if self.use_weighted_loss:
            per_k_mse = (noise_pred - noise) ** 2
            weighted_mse = per_k_mse * self.loss_weights[None, :]
            return weighted_mse.mean()
        return F.mse_loss(noise_pred, noise)
    
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t_int: int, c: torch.Tensor) -> torch.Tensor:
        B = x_t.size(0)
        t = torch.full((B,), t_int, device=x_t.device, dtype=torch.long)
        
        noise_pred = self(x_t, t, c)
        mean = self.sqrt_recip_alphas[t_int] * (
            x_t - self.betas[t_int] / self.sqrt_one_minus_alphas_cumprod[t_int] * noise_pred
        )
        
        if t_int == 0:
            return mean
        return mean + torch.sqrt(self.posterior_variance[t_int]) * torch.randn_like(x_t)
    
    @torch.no_grad()
    def sample(self, c: torch.Tensor, n: int = 1, temperature: float = 1.0) -> torch.Tensor:
        B = c.size(0)
        device = c.device
        
        samples = []
        for _ in range(n):
            x_t = torch.randn(B, self.output_dim, device=device) * temperature
            for t in reversed(range(self.T)):
                x_t = self.p_sample(x_t, t, c)
            samples.append(x_t)
        
        return torch.stack(samples)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RadiusDiffusion":
        diff_cfg = config.get("radius_diffusion", config.get("diffusion", config))
        hidden_dims = diff_cfg.get("hidden_dims", RADIUS_HIDDEN_DIMS.copy())
        use_weighted_loss = diff_cfg.get("use_weighted_loss", True)
        
        loss_weights = diff_cfg.get("loss_weights", None)
        if use_weighted_loss and loss_weights is None:
            raise ValueError(_missing_loss_weights_error())
        
        return cls(
            output_dim=diff_cfg.get("output_dim", CHEBYSHEV_K_RADIUS),
            cond_dim=diff_cfg.get("cond_dim", CONDITION_DIM),
            hidden_dims=hidden_dims,
            time_emb_dim=diff_cfg.get("time_emb_dim", DEFAULT_TIME_EMB_DIM),
            dropout=diff_cfg.get("dropout", DEFAULT_DROPOUT),
            T=diff_cfg.get("T", 200),
            beta_start=diff_cfg.get("beta_start", 1e-4),
            beta_end=diff_cfg.get("beta_end", 0.02),
            use_film=diff_cfg.get("use_film", True),
            use_weighted_loss=use_weighted_loss,
            loss_weights=loss_weights,
        )


if __name__ == "__main__":
    print("=" * 70)
    print("curve_diffusion.py test")
    print("=" * 70)
    
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n[1] MagnitudeDiffusion Test")
    config_concat = {"magnitude_diffusion": {"input_dim": 48, "cond_dim": 9, "use_film": False}}
    model_concat = MagnitudeDiffusion.from_config(config_concat).to(device)
    print(f"  Concat model: {sum(p.numel() for p in model_concat.parameters()):,} params")
    
    config_film = {"magnitude_diffusion": {"input_dim": 48, "cond_dim": 9, "use_film": True}}
    model_film = MagnitudeDiffusion.from_config(config_film).to(device)
    print(f"  FiLM model: {sum(p.numel() for p in model_film.parameters()):,} params")
    
    condition = torch.randn(batch_size, 9, device=device)
    r_0 = torch.rand(batch_size, 48, device=device)
    
    loss_concat = model_concat.training_loss(r_0, condition)
    loss_film = model_film.training_loss(r_0, condition)
    print(f"  Concat loss: {loss_concat.item():.4f}")
    print(f"  FiLM loss: {loss_film.item():.4f}")
    
    print("\n[2] RadiusDiffusion Test (loss_weights required)")

    try:
        config_no_weights = {"radius_diffusion": {"output_dim": 6, "use_weighted_loss": True}}
        model_fail = RadiusDiffusion.from_config(config_no_weights)
        print("  ERROR: Should have raised ValueError!")
    except ValueError as e:
        print("  Correctly raised ValueError for missing loss_weights")

    config_with_weights = {
        "radius_diffusion": {
            "output_dim": 6,
            "use_weighted_loss": True,
            "loss_weights": [2.5, 0.1, 0.3, 0.6, 0.9, 1.5]
        }
    }
    model_radius = RadiusDiffusion.from_config(config_with_weights).to(device)
    print(f"  RadiusDiffusion created: {sum(p.numel() for p in model_radius.parameters()):,} params")
    print(f"  loss_weights: {model_radius.loss_weights.tolist()}")
    
    coeffs_r = torch.randn(batch_size, 6, device=device)
    loss_radius = model_radius.training_loss(coeffs_r, condition)
    print(f"  RadiusDiffusion loss: {loss_radius.item():.4f}")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
