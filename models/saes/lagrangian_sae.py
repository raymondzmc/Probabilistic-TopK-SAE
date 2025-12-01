# lagrangian_sae.py
"""
Lagrangian Sparse Autoencoder using dual ascent to learn sparsity.

This SAE uses a Lagrangian multiplier (alpha) to adaptively control the sparsity penalty,
targeting a maximum L0 sparsity level. Unlike TopK/BatchTopK which enforce exact
sparsity, this approach learns to achieve a target sparsity through optimization.

Key design choices:
- Only penalizes when L0 > target_l0 (inequality constraint: L0 ≤ target)
- When L0 < target, focus purely on reconstruction (which naturally increases L0)
- Uses running mean (EMA) of L0 for stable constraint evaluation
- Uses per-feature learned thresholds (like JumpReLU) for flexible sparsity control
- Uses differentiable L0 approximation for gradient computation
"""
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any
from pydantic import Field, model_validator
from jaxtyping import Float

from models.saes.base import BaseSAE, SAELoss, SAEOutput, SAEConfig
from models.saes.utils import (
    init_decoder_orthogonal_cuda,
    update_dead_feature_stats,
    maybe_compute_auxk_features,
    compute_aux_loss_with_logging,
)
from models.saes.activations import StepFunction, JumpReLU
from utils.enums import SAEType


class LagrangianSAEConfig(SAEConfig):
    """
    Config for Lagrangian SAE.

    Notes:
    - Uses JumpReLU activation with per-feature learned thresholds
    - Differentiable L0 approximation for gradient computation  
    - Dual ascent on alpha to maintain L0 ≤ target_l0
    - Only penalizes when L0 > target (reconstruction-only when L0 < target)
    - Uses running mean (EMA) of L0 for stable constraint evaluation
    - Comparable to TopK/BatchTopK for analyzing global expected L0
    """
    sae_type: SAEType = Field(default=SAEType.LAGRANGIAN, description="Type of SAE (automatically set to lagrangian)")
    target_l0: float = Field(..., description="Target maximum L0 sparsity (number of active features per sample)")
    initial_alpha: float = Field(0.0, description="Initial alpha for Lagrangian multiplier (>= 0)")
    alpha_lr: float = Field(1e-2, description="Learning rate for alpha dual ascent")
    alpha_max: float = Field(100.0, description="Maximum value for alpha to prevent instability")
    tied_encoder_init: bool = Field(True, description="Initialize encoder as decoder.T")
    use_pre_enc_bias: bool = Field(False, description="Subtract decoder bias before encoding")
    
    # Running mean (EMA) parameters for stable L0 estimation
    l0_ema_momentum: float = Field(0.99, description="Momentum for running mean of L0 (higher = smoother)")
    
    # Per-feature threshold parameters (like JumpReLU)
    # NOTE: initial_threshold should be high enough that initial L0 is near target (0.1-1.0 works well)
    initial_threshold: float = Field(0.5, description="Initial per-feature threshold value (stored in log-space)")
    bandwidth: float = Field(0.1, description="Bandwidth for step function gradient approximation")
    
    # Dead feature tracking and auxiliary loss (optional)
    dead_toks_threshold: int | None = Field(None, description="Threshold for considering a feature as dead (number of tokens)")
    aux_k: int | None = Field(None, description="Auxiliary K for dead-feature loss (select top aux_k from the inactive set)")
    aux_coeff: float | None = Field(None, description="Coefficient for the auxiliary reconstruction loss")
    
    # Quadratic penalty coefficient for constraint violation (augmented Lagrangian)
    rho_quadratic: float = Field(0.1, description="Coefficient for quadratic penalty term in augmented Lagrangian")

    @model_validator(mode="before")
    @classmethod
    def set_sae_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        if isinstance(values, dict):
            values["sae_type"] = SAEType.LAGRANGIAN
        return values


class LagrangianSAEOutput(SAEOutput):
    """
    Lagrangian SAE output extending SAEOutput with useful intermediates.
    """
    preacts: Float[torch.Tensor, "... c"]  # encoder linear outputs (before thresholding)
    l0: Float[torch.Tensor, "..."]  # L0 norm per sample (actual count of non-zero activations)
    l0_differentiable: Float[torch.Tensor, "..."]  # Differentiable L0 approximation for gradient
    alpha: torch.Tensor  # current Lagrangian multiplier value
    auxk_indices: torch.Tensor | None = None  # auxiliary top-k indices for dead latents (shape: ... x aux_k)
    auxk_values: torch.Tensor | None = None   # auxiliary top-k values for dead latents (shape: ... x aux_k)


class LagrangianSAE(BaseSAE):
    """
    Lagrangian Sparse Autoencoder:
      - Linear encoder/decoder with bias
      - JumpReLU activation with per-feature learned thresholds
      - Differentiable L0 approximation for gradient computation
      - Dual ascent on alpha to achieve L0 ≤ target_l0
      - MSE reconstruction loss
      
    The optimization problem is:
        min_θ E[MSE(x, x̂)]
        s.t. E[L0(c)] ≤ target_l0
        
    We use augmented Lagrangian with dual ascent (inequality constraint):
        L = MSE + α · max(0, L0_diff - target) + ρ/2 · max(0, L0 - target)²
        α ← max(0, α + α_lr · (L0 - target_l0))
        
    When L0 < target, only MSE is optimized, allowing reconstruction to naturally
    increase L0 toward the target.
    
    Unlike fixed-threshold approaches, this SAE learns per-feature thresholds,
    allowing each dictionary element to find its optimal activation threshold.
    """

    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        target_l0: float,
        initial_alpha: float = 0.0,
        alpha_lr: float = 1e-2,
        alpha_max: float = 100.0,
        rho_quadratic: float = 0.1,
        l0_ema_momentum: float = 0.99,
        initial_threshold: float = 0.5,
        bandwidth: float = 0.1,
        sparsity_coeff: float | None = None,  # unused; kept for API parity
        mse_coeff: float | None = None,
        dead_toks_threshold: int | None = None,
        aux_k: int | None = None,
        aux_coeff: float | None = None,
        init_decoder_orthogonal: bool = True,
        tied_encoder_init: bool = True,
        use_pre_enc_bias: bool = False,
    ):
        """
        Args:
            input_size: Dimensionality of inputs (e.g., residual stream width).
            n_dict_components: Number of dictionary features (latent size).
            target_l0: Target maximum number of active features per sample.
            initial_alpha: Initial value for Lagrangian multiplier (>= 0).
            alpha_lr: Learning rate for alpha dual ascent.
            alpha_max: Maximum value for alpha.
            rho_quadratic: Coefficient for quadratic penalty in augmented Lagrangian.
            l0_ema_momentum: Momentum for running mean of L0 (0.99 = smooth, 0.0 = no smoothing).
            initial_threshold: Initial per-feature threshold value (stored in log-space).
            bandwidth: Bandwidth for step function gradient approximation.
            sparsity_coeff: Unused for Lagrangian (present for interface compatibility).
            mse_coeff: Coefficient on MSE reconstruction loss (default 1.0).
            dead_toks_threshold: Threshold for considering a feature as dead.
            aux_k: If provided (>0), number of auxiliary features from the inactive set.
            aux_coeff: Coefficient on the auxiliary reconstruction loss (default 0.0 if aux_k is None).
            init_decoder_orthogonal: Initialize decoder weight columns to be orthonormal.
            tied_encoder_init: Initialize encoder.weight = decoder.weight.T.
            use_pre_enc_bias: Whether to subtract decoder bias before encoding.
        """
        super().__init__()
        assert target_l0 > 0, "target_l0 must be positive"
        assert n_dict_components > 0 and input_size > 0
        assert bandwidth > 0, "bandwidth must be positive"
        assert 0.0 <= l0_ema_momentum < 1.0, "l0_ema_momentum must be in [0, 1)"
        assert initial_alpha >= 0, "initial_alpha must be non-negative"
        assert initial_threshold > 0, "initial_threshold must be positive"

        self.input_size = input_size
        self.n_dict_components = n_dict_components
        self.target_l0 = target_l0
        self.alpha_lr = alpha_lr
        self.alpha_max = alpha_max
        self.rho_quadratic = rho_quadratic
        self.l0_ema_momentum = l0_ema_momentum
        self.bandwidth = bandwidth
        self.use_pre_enc_bias = use_pre_enc_bias

        # Loss coefficients
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 0.0  # not used directly
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0

        self.dead_toks_threshold = int(dead_toks_threshold) if dead_toks_threshold is not None else None
        
        # Auxiliary loss for dead features (same as TopK/BatchTopK)
        self.aux_k = int(aux_k) if aux_k is not None and aux_k > 0 else 0
        self.aux_coeff = (aux_coeff if aux_coeff is not None else 0.0) if self.aux_k > 0 else 0.0

        # Lagrangian multiplier (alpha) as a buffer (not a parameter - updated via dual ascent)
        # Alpha >= 0 always (inequality constraint)
        self.register_buffer("alpha", torch.tensor(max(0.0, initial_alpha), dtype=torch.float32))
        
        # Running mean of L0 for stable constraint evaluation (EMA)
        # Initialize to target_l0 so the constraint starts satisfied
        self.register_buffer("running_l0", torch.tensor(target_l0, dtype=torch.float32))

        # Biases
        self.decoder_bias = nn.Parameter(torch.zeros(input_size))
        self.encoder_bias = nn.Parameter(torch.zeros(n_dict_components))

        # Linear maps (no bias in linear layers - biases are separate)
        self.encoder = nn.Linear(input_size, n_dict_components, bias=False)
        self.decoder = nn.Linear(n_dict_components, input_size, bias=False)
        
        # JumpReLU activation with per-feature learned thresholds
        self.jumprelu = JumpReLU(
            feature_size=n_dict_components,
            bandwidth=bandwidth,
            initial_threshold=initial_threshold,
        )

        # Initialize decoder, then (optionally) tie encoder init to decoder^T
        if init_decoder_orthogonal:
            self.decoder.weight.data = init_decoder_orthogonal_cuda(self.decoder.weight)
        else:
            # Random unit-norm columns
            dec_w = torch.randn_like(self.decoder.weight)
            dec_w = F.normalize(dec_w, dim=0)
            self.decoder.weight.data.copy_(dec_w)

        if tied_encoder_init:
            self.encoder.weight.data.copy_(self.decoder.weight.data.T)

        # Dead latent tracking - counts tokens since last activation
        self.register_buffer("stats_last_nonzero", torch.zeros(n_dict_components, dtype=torch.long))

    def forward(self, x: Float[torch.Tensor, "... dim"]) -> LagrangianSAEOutput:
        """
        Forward pass (supports arbitrary leading batch dims; last dim == input_size).
        """
        # Optional: subtract decoder bias before encoding
        if self.use_pre_enc_bias:
            x_enc = x - self.decoder_bias
        else:
            x_enc = x

        # Encoder with ReLU pre-activation
        preacts = F.relu(self.encoder(x_enc) + self.encoder_bias)
        
        # Apply JumpReLU with per-feature learned thresholds
        c = self.jumprelu(preacts)

        # Update dead latent statistics if training
        update_dead_feature_stats(
            activations=c,
            stats_last_nonzero=self.stats_last_nonzero,
            training=self.training,
            dead_toks_threshold=self.dead_toks_threshold,
        )

        # Compute auxiliary top-k indices and values for dead latents
        auxk_values, auxk_indices = maybe_compute_auxk_features(
            preacts=preacts,
            stats_last_nonzero=self.stats_last_nonzero,
            aux_k=self.aux_k,
            aux_coeff=self.aux_coeff,
            dead_toks_threshold=self.dead_toks_threshold,
        )

        # Decode using normalized dictionary elements + add bias back
        x_hat = F.linear(c, self.dict_elements, bias=self.decoder_bias)

        # Compute true L0 (for constraint evaluation and logging)
        l0 = (c > 0).float().sum(dim=-1)
        
        # Compute differentiable L0 using StepFunction (gradient flows to learned thresholds)
        l0_differentiable = StepFunction.apply(
            preacts, 
            self.jumprelu.log_threshold, 
            self.bandwidth
        ).sum(dim=-1)

        return LagrangianSAEOutput(
            input=x,
            c=c,
            output=x_hat,
            logits=None,
            preacts=preacts,
            l0=l0,
            l0_differentiable=l0_differentiable,
            alpha=self.alpha,
            auxk_indices=auxk_indices,
            auxk_values=auxk_values,
        )

    def compute_loss(self, output: LagrangianSAEOutput) -> SAELoss:
        """
        Computes the augmented Lagrangian loss for sparsity control.
        
        Inequality constraint (L0 ≤ target):
            Loss = MSE + α * max(0, L0_diff - target) + ρ/2 * max(0, L0_diff - target)²
            
        When L0 < target: Only MSE is optimized, reconstruction naturally increases L0.
        When L0 > target: Sparsity penalty is applied to reduce L0.
        
        Uses:
        - L0_diff: Differentiable L0 approximation (provides gradient to learned thresholds)
        - L0: True L0 (used for constraint evaluation and dual ascent)
        - running_l0: Exponential moving average of L0 for stable dual ascent
        
        Note: Dual ascent on alpha is performed via update_alpha() which should be
        called after optimizer.step() to avoid autograd conflicts.
        """
        # Reconstruction loss
        mse_loss = F.mse_loss(output.output, output.input)
        
        # Compute mean L0 across the batch (true L0 for monitoring)
        mean_l0 = output.l0.mean()
        batch_constraint_violation = mean_l0 - self.target_l0
        
        # Update running L0 (EMA) for stable constraint evaluation
        if self.training:
            with torch.no_grad():
                # EMA update: running_l0 = momentum * running_l0 + (1 - momentum) * batch_l0
                self.running_l0.mul_(self.l0_ema_momentum).add_(
                    (1.0 - self.l0_ema_momentum) * mean_l0.detach()
                )
        
        # Use RUNNING L0 for dual ascent (smoother updates)
        running_constraint_violation = self.running_l0 - self.target_l0
        
        # Store RAW constraint violation for dual ascent (can be negative)
        # The clamp to >= 0 happens in update_alpha() AFTER the update
        self._last_constraint_violation = running_constraint_violation.detach().clone()
        
        # Differentiable L0 for gradient computation
        mean_l0_diff = output.l0_differentiable.mean()
        differentiable_violation = mean_l0_diff - self.target_l0
        
        # Get current alpha value (detached to avoid gradient through alpha)
        alpha_value = self.alpha.detach().clone()
        
        # Inequality constraint: only penalize when L0 > target
        # Use ReLU to get max(0, L0_diff - target)
        positive_violation = F.relu(differentiable_violation)
        
        # Augmented Lagrangian formulation:
        # L = MSE + α * max(0, L0_diff - target) + ρ/2 * max(0, L0_diff - target)²
        sparsity_loss = alpha_value * positive_violation / self.n_dict_components
        quadratic_penalty = self.rho_quadratic * (positive_violation ** 2) / self.n_dict_components
        
        total_loss = self.mse_coeff * mse_loss + sparsity_loss + quadratic_penalty

        # Compute number of dead features for logging
        num_dead_features = torch.tensor(0.0, device=output.input.device)
        if self.dead_toks_threshold is not None:
            num_dead_features = (self.stats_last_nonzero > self.dead_toks_threshold).sum().float()

        # Compute mean threshold for logging
        mean_threshold = self.jumprelu.threshold.mean()

        loss_dict: dict[str, torch.Tensor] = {
            "mse_loss": mse_loss.detach().clone(),
            "running_l0": self.running_l0.detach().clone(),  # Running mean L0 (expected L0)
            "alpha": alpha_value,
            "quadratic_penalty": quadratic_penalty.detach().clone(),
            "mean_threshold": mean_threshold.detach().clone(),  # Track learned thresholds
        }

        # Optional auxiliary dead-feature loss using residual reconstruction
        weighted_aux_loss, aux_loss_for_logging = compute_aux_loss_with_logging(
            auxk_indices=output.auxk_indices,
            auxk_values=output.auxk_values,
            input_tensor=output.input,
            output_tensor=output.output,
            decoder_bias=self.decoder_bias,
            dict_elements=self.dict_elements,
            n_dict_components=self.n_dict_components,
            input_size=self.input_size,
            aux_k=self.aux_k,
            aux_coeff=self.aux_coeff,
        )
        total_loss = total_loss + weighted_aux_loss
        loss_dict["aux_loss"] = aux_loss_for_logging

        return SAELoss(loss=total_loss, loss_dict=loss_dict)
    
    @torch.no_grad()
    def update_alpha(self) -> None:
        """
        Perform dual ascent update on alpha for inequality constraint.
        
        Should be called after optimizer.step() to avoid autograd conflicts.
        
        Update rule: α ← max(0, α + α_lr * (L0 - target_l0))
        
        - When L0 > target: α increases (stronger sparsity penalty)
        - When L0 < target: α decreases (weaker penalty, allows more features)
        - Alpha is clamped to [0, alpha_max] after update
        """
        if hasattr(self, '_last_constraint_violation'):
            # Update alpha with raw constraint violation (can be negative)
            self.alpha.add_(self.alpha_lr * self._last_constraint_violation)
            # Clamp alpha to valid range AFTER update
            self.alpha.clamp_(min=0.0, max=self.alpha_max)

    @property
    def dict_elements(self) -> torch.Tensor:
        """
        Column-wise unit-norm decoder (dictionary) – normalized every forward.
        This mirrors common SAE practice and avoids degenerate scaling solutions.
        """
        return F.normalize(self.decoder.weight, dim=0)

    @property
    def device(self):
        return next(self.parameters()).device
