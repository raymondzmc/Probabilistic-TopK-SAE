# hc_batch_topk_sae.py
"""
Hard Concrete Batch Top-K SAE: Combines Hard Concrete distribution sampling with
batch-level Top-K selection for sparse autoencoders.

Key features:
- Hard Concrete distribution for differentiable gate sampling
- TopK selection applied across the entire batch (flattened) rather than per-sample
- Beta annealing for temperature scheduling
- Magnitude-based scoring with optional normalization
- Auxiliary loss for dead feature mitigation
"""
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any
from pydantic import Field, model_validator
from jaxtyping import Float

from models.saes.base import BaseSAE, SAELoss, SAEOutput, SAEConfig
from models.saes.utils import init_decoder_orthogonal_cuda
from utils.enums import SAEType


class HardConcreteBatchTopKSAEConfig(SAEConfig):
    """
    Config for Hard Concrete Batch Top-K SAE.

    Notes:
    - Applies Top-K selection across the entire batch (flattened) using Hard Concrete sampled scores
    - Supports auxiliary loss for dead feature mitigation
    - Beta annealing for temperature scheduling
    """
    sae_type: SAEType = Field(default=SAEType.HC_BATCH_TOPK, description="Type of SAE")
    k: int = Field(..., description="Number of active features to keep per sample")
    tied_encoder_init: bool = Field(True, description="Initialize encoder as decoder.T")
    
    # Optional: dead-feature mitigation via auxiliary Top-K
    aux_k: int | None = Field(None, description="Auxiliary K for dead-feature loss")
    aux_coeff: float | None = Field(None, description="Coefficient for the auxiliary reconstruction loss")
    dead_toks_threshold: int | None = Field(None, description="Threshold for considering a feature as dead (number of tokens)")
    
    # Hard Concrete distribution parameters
    initial_beta: float = Field(5.0, description="Initial beta for hard concrete sampling")
    final_beta: float | None = Field(None, description="Final beta for hard concrete sampling")
    anneal_ratio: float | None = Field(None, description="Ratio of training steps before annealing beta")
    
    # Scoring parameters
    use_magnitude: bool = Field(True, description="Use magnitude in the score for the Top-K selection")
    magnitude_scale: float = Field(0.01, description="Scale for the magnitude in scoring")
    normalize_scores: bool = Field(True, description="Normalize scores to have mean 0 and std 1")
    normalize_magnitude: bool = Field(False, description="Normalize magnitude to have mean 0 and std 1")
    
    # Architecture options
    straight_through: bool = Field(False, description="Use straight-through gradient estimator")
    tau: float | None = Field(None, description="Temperature for straight-through soft mask")
    z_scale: float | None = Field(None, description="Scale for the hard concrete samples in activation")
    detach_decoder_bias: bool = Field(False, description="Detach the decoder bias from the gradient")
    use_hard_concrete: bool = Field(True, description="Use hard concrete sampling (if False, uses sigmoid)")
    use_layer_norm: bool = Field(True, description="Use layer norm on the gate logits")
    
    # Running threshold for inference
    threshold_momentum: float = Field(0.99, description="Momentum for running threshold EMA update")

    @model_validator(mode="before")
    @classmethod
    def set_sae_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        if isinstance(values, dict):
            values["sae_type"] = SAEType.HC_BATCH_TOPK
        return values


class HardConcreteBatchTopKSAEOutput(SAEOutput):
    """
    HardConcreteBatchTopK SAE output extending SAEOutput with useful intermediates.
    """
    preacts: Float[torch.Tensor, "... c"]  # encoder linear outputs (after centering)
    mask: Float[torch.Tensor, "... c"]     # binary mask of selected Top-K indices
    scores: Float[torch.Tensor, "... c"]   # scores used for Top-K selection
    z: Float[torch.Tensor, "... c"]        # hard concrete samples
    gate_logits: Float[torch.Tensor, "... c"] | None = None  # gate logits after layer norm
    auxk_indices: torch.Tensor | None = None  # auxiliary top-k indices for dead latents
    auxk_values: torch.Tensor | None = None   # auxiliary top-k values for dead latents


class HardConcreteBatchTopKSAE(BaseSAE):
    """
    Hard Concrete Batch Top-K Sparse Autoencoder:
      - Linear encoder/decoder (no bias on the linear layers)
      - Single learned decoder_bias used to center input and add back after decode
      - Hard Concrete distribution for differentiable gate sampling
      - Top-K selection applied across the entire batch (flattened activations)
      - MSE reconstruction loss with optional auxiliary loss for dead features
    """

    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        k: int,
        sparsity_coeff: float | None = None,  # unused; kept for API parity
        mse_coeff: float | None = None,
        aux_k: int | None = None,
        aux_coeff: float | None = None,
        dead_toks_threshold: int | None = None,
        init_decoder_orthogonal: bool = True,
        tied_encoder_init: bool = True,
        initial_beta: float = 5.0,
        final_beta: float | None = None,
        anneal_ratio: float | None = None,
        use_magnitude: bool = True,
        magnitude_scale: float = 0.01,
        normalize_scores: bool = True,
        normalize_magnitude: bool = False,
        straight_through: bool = False,
        tau: float | None = None,
        z_scale: float | None = None,
        detach_decoder_bias: bool = False,
        use_hard_concrete: bool = True,
        use_layer_norm: bool = True,
        threshold_momentum: float = 0.99,
    ):
        """
        Args:
            input_size: Dimensionality of inputs (e.g., residual stream width).
            n_dict_components: Number of dictionary features (latent size).
            k: Number of active features to keep per sample (Top-K).
            sparsity_coeff: Unused for Top-K (present for interface compatibility).
            mse_coeff: Coefficient on MSE reconstruction loss (default 1.0).
            aux_k: If provided (>0), number of auxiliary features for dead latents.
            aux_coeff: Coefficient on the auxiliary reconstruction loss.
            dead_toks_threshold: Threshold for considering a feature as dead.
            init_decoder_orthogonal: Initialize decoder weight columns to be orthonormal.
            tied_encoder_init: Initialize encoder.weight = decoder.weight.T.
            initial_beta: Initial beta for hard concrete sampling.
            final_beta: Final beta for hard concrete sampling.
            anneal_ratio: Ratio of training before beta annealing starts.
            use_magnitude: Use magnitude in the score for Top-K selection.
            magnitude_scale: Scale for the magnitude in scoring.
            normalize_scores: Normalize scores before Top-K selection.
            normalize_magnitude: Normalize magnitude before adding to scores.
            straight_through: Use straight-through gradient estimator.
            tau: Temperature for straight-through soft mask.
            z_scale: Scale for hard concrete samples in activation.
            detach_decoder_bias: Detach the decoder bias from the gradient.
            use_hard_concrete: Use hard concrete sampling (if False, uses sigmoid).
            use_layer_norm: Use layer norm on the gate logits.
            threshold_momentum: Momentum for running threshold EMA update.
        """
        super().__init__()
        assert k >= 0, "k must be non-negative"
        assert n_dict_components > 0 and input_size > 0

        self.input_size = input_size
        self.n_dict_components = n_dict_components
        self.k = int(k)
        assert self.k > 0 and self.k <= n_dict_components, \
            "k must be greater than 0 and less than or equal to n_dict_components"

        # Loss coefficients
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 0.0
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0

        self.aux_k = int(aux_k) if aux_k is not None and aux_k > 0 else 0
        self.aux_coeff = (aux_coeff if aux_coeff is not None else 0.0) if self.aux_k > 0 else 0.0
        self.dead_toks_threshold = int(dead_toks_threshold) if dead_toks_threshold is not None else None

        # Bias used for input centering and added back on decode
        self.decoder_bias = nn.Parameter(torch.zeros(input_size))

        # Linear maps (no bias)
        self.encoder = nn.Linear(input_size, n_dict_components, bias=False)
        self.decoder = nn.Linear(n_dict_components, input_size, bias=False)

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

        # Layer norm for gate logits
        self.gate_ln = nn.LayerNorm(n_dict_components, elementwise_affine=True)

        # Beta scheduling for Hard Concrete
        self.register_buffer("train_progress", torch.tensor(0.0))
        self.register_buffer("beta", torch.tensor(initial_beta, dtype=torch.float32))
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        assert self.initial_beta > 0.0, "initial_beta must be positive"
        assert self.final_beta is None or (self.final_beta > 0.0 and self.initial_beta >= self.final_beta), \
            "final_beta must be positive and less than or equal to initial_beta"
        assert anneal_ratio is None or (anneal_ratio >= 0.0 and anneal_ratio < 1.0), \
            "anneal_ratio must be between 0.0 and 1.0 (exclusive)"
        self.beta_anneal = self.final_beta is not None
        self.anneal_ratio = anneal_ratio if anneal_ratio is not None else 0.0

        # Scoring parameters
        self.use_magnitude = use_magnitude
        self.magnitude_scale = magnitude_scale
        self.normalize_scores = normalize_scores
        self.normalize_magnitude = normalize_magnitude

        # Architecture options
        self.straight_through = straight_through
        self.tau = 20.0 if straight_through and tau is None else tau
        self.z_scale = z_scale
        self.detach_decoder_bias = detach_decoder_bias
        self.use_hard_concrete = use_hard_concrete
        self.use_layer_norm = use_layer_norm
        
        # Running threshold for inference (use double precision for numerical stability, as in SAELens)
        self.threshold_momentum = threshold_momentum
        self.register_buffer("running_threshold", torch.tensor(0.0, dtype=torch.double))

        # Dead latent tracking - counts tokens since last activation
        self.register_buffer("stats_last_nonzero", torch.zeros(n_dict_components, dtype=torch.long))

        # Create auxk_mask_fn for masking alive latents
        def auxk_mask_fn(x: torch.Tensor) -> torch.Tensor:
            """Mask out alive latents by zeroing those that have been active recently."""
            if self.dead_toks_threshold is None:
                return x
            dead_mask = self.stats_last_nonzero > self.dead_toks_threshold
            # Expand dead_mask to match x dimensions
            dead_mask = dead_mask.view(1, -1).expand_as(x)
            return x * dead_mask.to(x.dtype)

        self.auxk_mask_fn = auxk_mask_fn

    def sample_hard_concrete(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from Hard Concrete distribution."""
        if self.training:
            u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
            z = torch.sigmoid((logits + torch.log(u) - torch.log(1 - u)) / self.beta)
        else:
            z = torch.sigmoid(logits / self.beta)
        return z

    def _refresh_beta(self):
        """Update beta according to annealing schedule."""
        t = float(self.train_progress.item())
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        # Start annealing only after anneal_ratio
        if not self.beta_anneal or t < self.anneal_ratio:
            return

        # Scale t to [0, 1] for the annealing phase
        t_annealed = (t - self.anneal_ratio) / (1.0 - self.anneal_ratio)
        t_annealed = 0.0 if t_annealed < 0.0 else (1.0 if t_annealed > 1.0 else t_annealed)

        # Geometric interpolation: beta = beta0 * (beta1/beta0)^t
        ratio = self.final_beta / self.initial_beta
        new_beta = self.initial_beta * (ratio ** t_annealed)
        new_beta = float(max(1e-3, new_beta))
        self.beta.fill_(new_beta)

    def _apply_batch_topk(
        self, 
        scores: torch.Tensor, 
        preacts: torch.Tensor,
        batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Top-K selection across the entire batch (flattened).
        
        Args:
            scores: Tensor of scores for Top-K selection (batch*seq, n_dict_components)
            preacts: Tensor of pre-activations (batch*seq, n_dict_components)
            batch_size: Number of samples in the batch
            
        Returns:
            acts_topk: sparse activations after masking to Top-K
            mask: binary mask (same shape as scores) with ones at Top-K indices
        """
        # Flatten all scores across the batch
        scores_flat = scores.flatten()

        # Select top-k across the entire batch
        total_k = self.k * batch_size
        _, topk_indices = torch.topk(scores_flat, k=total_k, dim=-1)

        # Create mask
        mask_flat = torch.zeros_like(scores_flat)
        mask_flat.scatter_(-1, topk_indices, 1.0)

        # Reshape back to original shape
        mask = mask_flat.reshape(scores.shape)

        return mask

    @torch.no_grad()
    def _update_running_threshold(self, acts: torch.Tensor) -> None:
        """Update running threshold based on minimum positive activation after batch top-k.
        
        This threshold is used during inference when batch top-k isn't applicable.
        Following SAELens implementation for numerical stability.
        
        Args:
            acts: Activations after batch top-k (sparse tensor with selected values)
        """
        positive_mask = acts > 0
        
        # Disable autocast to prevent numerical issues (following SAELens)
        with torch.autocast(self.running_threshold.device.type, enabled=False):
            if positive_mask.any():
                min_positive = acts[positive_mask].min().to(self.running_threshold.dtype)
                lr = 1 - self.threshold_momentum  # Convert momentum to learning rate
                self.running_threshold = (1 - lr) * self.running_threshold + lr * min_positive

    def _apply_threshold(self, preacts: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply learned threshold during inference (JumpReLU-style).
        
        Args:
            preacts: Pre-activations from encoder
            z: Hard concrete samples or sigmoid outputs
            
        Returns:
            c: Activations with values below threshold zeroed out
            mask: Binary mask of active features
        """
        # Convert threshold to preacts dtype for comparison
        threshold = self.running_threshold.to(preacts.dtype)
        
        # Use the same scoring mechanism as training for threshold comparison
        if self.use_magnitude:
            magnitude = preacts.abs()
            if self.normalize_magnitude:
                magnitude = (magnitude - magnitude.mean(dim=-1, keepdim=True)) / (magnitude.std(dim=-1, keepdim=True) + 1e-8)
            scores = z + self.magnitude_scale * magnitude
        else:
            scores = z
        
        mask = (scores > threshold).float()
        
        # Compute activations with optional z_scale (same as training)
        if self.z_scale is not None:
            c = (preacts * mask) * (self.z_scale + z)
        else:
            c = preacts * mask
            
        return c, mask

    def forward(self, x: Float[torch.Tensor, "... dim"]) -> HardConcreteBatchTopKSAEOutput:
        """
        Forward pass (supports arbitrary leading batch dims; last dim == input_size).
        """
        if self.training:
            self._refresh_beta()

        # Store original shape and flatten batch dimensions
        original_shape = x.shape
        x = x.reshape(-1, self.input_size)
        batch_size = x.shape[0]

        # Center input
        if self.detach_decoder_bias:
            x_centered = x - self.decoder_bias.detach()
        else:
            x_centered = x - self.decoder_bias

        # Encoder pre-activations
        preacts = self.encoder(x_centered)

        # Sample Hard Concrete or use sigmoid
        gate_logits = None
        if self.use_hard_concrete:
            if self.use_layer_norm:
                gate_logits = self.gate_ln(preacts)
                z = self.sample_hard_concrete(gate_logits)
            else:
                z = self.sample_hard_concrete(preacts)
        else:
            z = torch.sigmoid(preacts)

        # Compute scores for Top-K selection
        if self.use_magnitude:
            magnitude = preacts.abs()
            if self.normalize_magnitude:
                magnitude = magnitude.detach()
                magnitude = (magnitude - magnitude.mean(dim=-1, keepdim=True)) / (magnitude.std(dim=-1, keepdim=True) + 1e-8)
            scores = z + self.magnitude_scale * magnitude
        else:
            scores = z

        # Apply sparsification: batch top-k during training, threshold during inference
        if self.training:
            # Apply Batch-level Top-K selection
            mask = self._apply_batch_topk(scores, preacts, batch_size)

            # Add straight-through soft mask for gradient
            if self.straight_through:
                soft = torch.softmax(scores / self.tau, dim=-1)
                soft_k = soft * (self.k / (soft.sum(dim=-1, keepdim=True) + 1e-8)).clamp(max=1.0)
                mask = mask + soft_k - soft_k.detach()

            # Compute activations with optional z_scale
            if self.z_scale is not None:
                c = (preacts * mask) * (self.z_scale + z)
            else:
                c = preacts * mask
            
            # Update running threshold for inference
            self._update_running_threshold(c)
        else:
            # Use learned threshold during inference
            c, mask = self._apply_threshold(preacts, z)

        # Update dead latent statistics if training
        if self.training and self.dead_toks_threshold is not None:
            with torch.no_grad():
                n_tokens = batch_size
                activated_mask = (c.abs() > 1e-3).any(dim=0)
                self.stats_last_nonzero *= (~activated_mask).long()
                self.stats_last_nonzero += n_tokens

        # Compute auxiliary top-k indices and values for dead latents
        auxk_indices = None
        auxk_values = None

        if self.aux_k > 0 and self.aux_coeff > 0.0 and self.dead_toks_threshold is not None:
            masked_preacts = self.auxk_mask_fn(preacts)
            if masked_preacts.abs().max() > 0:
                auxk_values, auxk_indices = torch.topk(
                    masked_preacts, k=min(self.aux_k, masked_preacts.shape[-1]), dim=-1
                )

        # Decode using normalized dictionary elements + add bias back
        x_hat = F.linear(c, self.dict_elements, bias=self.decoder_bias)

        # Reshape outputs back to original shape
        x_hat = x_hat.reshape(original_shape)
        c = c.reshape(*original_shape[:-1], self.n_dict_components)
        preacts = preacts.reshape(*original_shape[:-1], self.n_dict_components)
        mask = mask.reshape(*original_shape[:-1], self.n_dict_components)
        scores = scores.reshape(*original_shape[:-1], self.n_dict_components)
        z = z.reshape(*original_shape[:-1], self.n_dict_components)
        if gate_logits is not None:
            gate_logits = gate_logits.reshape(*original_shape[:-1], self.n_dict_components)

        return HardConcreteBatchTopKSAEOutput(
            input=x.reshape(original_shape),
            c=c,
            output=x_hat,
            logits=None,
            preacts=preacts,
            mask=mask,
            scores=scores,
            z=z,
            gate_logits=gate_logits,
            auxk_indices=auxk_indices,
            auxk_values=auxk_values,
        )

    def compute_loss(self, output: HardConcreteBatchTopKSAEOutput) -> SAELoss:
        """
        Loss = mse_coeff * MSE + aux_coeff * AuxK (optional)

        - No explicit L1 sparsity term (sparsity enforced by batch Top-K).
        - AuxK: Reconstruct the residual error using dead features.
        """
        mse_loss = F.mse_loss(output.output, output.input)
        total_loss = self.mse_coeff * mse_loss
        loss_dict: dict[str, torch.Tensor] = {
            "mse_loss": mse_loss.detach().clone(),
            "preacts_mean": output.preacts.mean().detach().clone(),
            "preacts_std": output.preacts.std().detach().clone(),
            "z_mean": output.z.mean().detach().clone(),
            "z_std": output.z.std().detach().clone(),
            "topk_threshold": self.running_threshold.detach().clone().float(),
        }

        # Optional auxiliary dead-feature loss using residual reconstruction
        if (self.aux_k > 0 and self.aux_coeff > 0.0 and
            output.auxk_indices is not None and output.auxk_values is not None):

            # Create sparse representation for auxiliary latents
            aux_c = torch.zeros_like(output.preacts)
            aux_c.scatter_(-1, output.auxk_indices, output.auxk_values)

            # Decode auxiliary latents (no bias, as we're reconstructing residual)
            x_hat_aux = F.linear(aux_c, self.dict_elements)

            # Compute residual target
            residual_target = output.input - output.output.detach() + self.decoder_bias.detach()

            # Normalized MSE for auxiliary loss
            residual_norm = torch.norm(residual_target, p=2, dim=-1, keepdim=True)
            aux_recon_norm = torch.norm(x_hat_aux, p=2, dim=-1, keepdim=True)

            normalized_aux_loss = F.mse_loss(
                x_hat_aux / (aux_recon_norm + 1e-8),
                residual_target / (residual_norm + 1e-8)
            )

            # Safety: Replace NaN with 0
            aux_loss = normalized_aux_loss.nan_to_num(0.0)

            if torch.isnan(aux_loss).any() or torch.isinf(aux_loss).any():
                aux_loss = torch.zeros_like(aux_loss)

            total_loss = total_loss + self.aux_coeff * aux_loss
            loss_dict["aux_loss"] = aux_loss.detach().clone()
        else:
            loss_dict["aux_loss"] = torch.zeros((), device=output.input.device)

        return SAELoss(loss=total_loss, loss_dict=loss_dict)

    @property
    def dict_elements(self) -> torch.Tensor:
        """
        Column-wise unit-norm decoder (dictionary) – normalized every forward.
        """
        return F.normalize(self.decoder.weight, dim=0)

    @property
    def device(self):
        return next(self.parameters()).device

