# topk_sae.py
import torch
import torch.nn.functional as F
from typing import Any
from pydantic import Field, model_validator
from jaxtyping import Float
from utils.enums import SAEType
from models.saes.base import BaseSAE, SAELoss, SAEOutput, SAEConfig
from models.saes.utils import init_decoder_orthogonal_cuda


class HardConcreteTopKSAEConfig(SAEConfig):
    sae_type: SAEType = Field(default=SAEType.HARD_CONCRETE_TOPK, description="Type of SAE (automatically set to hard_concrete_topk)")
    k: int = Field(..., description="Number of active features to keep per sample")
    tied_encoder_init: bool = Field(True, description="Initialize encoder as decoder.T")

    # Optional: dead-feature mitigation via auxiliary Top-K on the *inactive* set
    aux_k: int | None = Field(None, description="Auxiliary K for dead-feature loss (select top aux_k from the inactive set)")
    aux_coeff: float | None = Field(None, description="Coefficient for the auxiliary reconstruction loss")
    dead_toks_threshold: int | None = Field(None, description="Threshold for considering a feature as dead (number of tokens)")
    
    initial_beta: float = Field(5.0, description="Initial beta for hard concrete sampling")
    final_beta: float | None = Field(None, description="Final beta for hard concrete sampling")

    use_magnitude: bool = Field(True, description="Use magnitude in the score for the Top-K selection")
    magnitude_scale: float = Field(0.01, description="Scale for the magnitude")
    straight_through: bool = Field(False, description="Use straight-through Top-K")
    tau: float | None = Field(None, description="Temperature for straight-through Top-K")
    anneal_ratio: float | None = Field(None, description="Ratio of training steps before annealing beta")
    normalize_scores: bool = Field(True, description="Normalize scores to have mean 0 and std 1")
    normalize_magnitude: bool = Field(False, description="Normalize magnitude to have mean 0 and std 1")
    add_magnitude_to_scores: bool = Field(False, description="Add magnitude to scores")
    z_scale: float | None = Field(None, description="Scale for the hard concrete samples")
    detach_decoder_bias: bool = Field(False, description="Detach the decoder bias from the gradient")
    use_hard_concrete: bool = Field(True, description="Use hard concrete sampling")
    use_layer_norm: bool = Field(True, description="Use layer norm on the gate logits")

    @model_validator(mode="before")
    @classmethod
    def set_sae_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        if isinstance(values, dict):
            values["sae_type"] = SAEType.HARD_CONCRETE_TOPK
        return values


class HardConcreteTopKSAEOutput(SAEOutput):
    """
    TopK SAE output extending SAEOutput with useful intermediates for loss/analysis.
    """
    preacts: Float[torch.Tensor, "... c"]  # encoder linear outputs (after centering)
    mask: Float[torch.Tensor, "... c"]     # binary mask of selected Top-K indices
    scores: Float[torch.Tensor, "... c"]   # scores of the selected Top-K indices
    z: Float[torch.Tensor, "... c"]        # hard concrete samples
    gate_logits: Float[torch.Tensor, "... c"] | None = None # gate logits
    auxk_indices: torch.Tensor | None = None  # auxiliary top-k indices for dead latents (shape: ... x aux_k)
    auxk_values: torch.Tensor | None = None   # auxiliary top-k values for dead latents (shape: ... x aux_k)


class HardConcreteTopKSAE(BaseSAE):
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
        use_magnitude: bool = True,
        magnitude_scale: float = 0.01,
        straight_through: bool = False,
        tau: float | None = None,
        anneal_ratio: float | None = None,
        normalize_scores: bool = False,
        normalize_magnitude: bool = False,
        add_magnitude_to_scores: bool = True,
        z_scale: float | None = None,
        detach_decoder_bias: bool = False,
        use_hard_concrete: bool = True,
        use_layer_norm: bool = True,
    ):
        """
        Args:
            input_size: Dimensionality of inputs (e.g., residual stream width).
            n_dict_components: Number of dictionary features (latent size).
            k: Number of active features to keep per sample (Top-K).
            sparsity_coeff: Unused for Top-K (present for interface compatibility).
            mse_coeff: Coefficient on MSE reconstruction loss (default 1.0).
            aux_k: If provided (>0), number of auxiliary features from the inactive set.
            aux_coeff: Coefficient on the auxiliary reconstruction loss (default 0.0 if aux_k is None).
            dead_toks_threshold: Threshold for considering a feature as dead (number of tokens).
            init_decoder_orthogonal: Initialize decoder weight columns to be orthonormal.
            tied_encoder_init: Initialize encoder.weight = decoder.weight.T.
            initial_beta: Initial beta for hard concrete sampling.
            final_beta: Final beta for hard concrete sampling.
            use_magnitude: Use magnitude in the score for the Top-K selection.
            straight_through: Use straight-through Top-K.
            tau: Temperature for straight-through Top-K.
            anneal_ratio: Ratio of training steps before annealing beta.
            z_scale: Scale for the hard concrete samples.
            detach_decoder_bias: Detach the decoder bias from the gradient.
        """
        super().__init__()
        assert k >= 0, "k must be non-negative"
        assert n_dict_components > 0 and input_size > 0

        self.input_size = input_size
        self.n_dict_components = n_dict_components
        self.k = int(k)
        assert self.k > 0 and self.k <= n_dict_components, "k must be greater than 0 and less than or equal to n_dict_components"

        # Loss coefficients
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 0.0  # not used, but kept for logs
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0

        self.aux_k = int(aux_k) if aux_k is not None and aux_k > 0 else 0
        self.aux_coeff = (aux_coeff if aux_coeff is not None else 0.0) if self.aux_k > 0 else 0.0
        self.dead_toks_threshold = int(dead_toks_threshold) if dead_toks_threshold is not None else None

        # Bias used for input centering and added back on decode
        self.decoder_bias = torch.nn.Parameter(torch.zeros(input_size))

        # Linear maps (no bias)
        self.encoder = torch.nn.Linear(input_size, n_dict_components, bias=False)
        self.decoder = torch.nn.Linear(n_dict_components, input_size, bias=False)

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

        self.gate_ln = torch.nn.LayerNorm(n_dict_components, elementwise_affine=True)
        
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

        self.use_magnitude = use_magnitude
        self.magnitude_scale = magnitude_scale

        self.straight_through = straight_through
        self.tau = 20.0 if straight_through and tau is None else tau
        self.normalize_scores = normalize_scores
        self.normalize_magnitude = normalize_magnitude
        self.add_magnitude_to_scores = add_magnitude_to_scores
        self.z_scale = z_scale
        self.detach_decoder_bias = detach_decoder_bias
        self.use_hard_concrete = use_hard_concrete
        self.use_layer_norm = use_layer_norm
        
        # Dead latent tracking
        self.register_buffer("stats_last_nonzero", torch.zeros(n_dict_components, dtype=torch.long))
        
        # Create auxk_mask_fn for masking alive latents
        def auxk_mask_fn(x: torch.Tensor) -> torch.Tensor:
            """Mask out alive latents by zeroing those that have been active recently."""
            if self.dead_toks_threshold is None:
                return x
            dead_mask = self.stats_last_nonzero > self.dead_toks_threshold
            # Expand dead_mask to match x dimensions
            dead_mask = dead_mask.view(1, -1).expand_as(x)
            # Return masked tensor (creating new tensor to avoid mutation)
            return x * dead_mask.to(x.dtype)
        
        self.auxk_mask_fn = auxk_mask_fn

    def sample_hard_concrete(self, logits: torch.Tensor):
        if self.training:
            u = torch.rand_like(logits).clamp_(1e-6, 1-1e-6)
            z = torch.sigmoid((logits + torch.log(u) - torch.log(1 - u)) / self.beta)
        else:
            z = torch.sigmoid(logits / self.beta)
        return z
    
    def _refresh_beta(self):
        t = float(self.train_progress.item())
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        # Start annealing only after anneal_ratio
        if not self.beta_anneal or t < self.anneal_ratio:
            return
        
        # Scale t to [0, 1] for the annealing phase
        t_annealed = (t - self.anneal_ratio) / (1.0 - self.anneal_ratio)
        t_annealed = 0.0 if t_annealed < 0.0 else (1.0 if t_annealed > 1.0 else t_annealed)
        
        # geometric interpolation: beta = beta0 * (beta1/beta0)^t
        ratio = self.final_beta / self.initial_beta
        new_beta = self.initial_beta * (ratio ** t_annealed)
        # small safety clamp
        new_beta = float(max(1e-3, new_beta))
        self.beta.fill_(new_beta)

    def forward(self, x: Float[torch.Tensor, "... dim"]) -> HardConcreteTopKSAEOutput:
        """
        Forward pass (supports arbitrary leading batch dims; last dim == input_size).
        """

        if self.training:
            self._refresh_beta()

        # Center input
        if self.detach_decoder_bias:
            x_centered = x - self.decoder_bias.detach()
        else:
            x_centered = x - self.decoder_bias
        preacts = self.encoder(x_centered)

        gate_logits = None
        if self.use_hard_concrete:
            if self.use_layer_norm:
                gate_logits = self.gate_ln(preacts)
                z = self.sample_hard_concrete(gate_logits)
            else:
                z = self.sample_hard_concrete(preacts)
        else:
            z = torch.sigmoid(preacts)

        if self.use_magnitude:
            magnitude = preacts.abs()
            if self.normalize_magnitude:
                magnitude = magnitude.detach()
                magnitude = (magnitude - magnitude.mean(dim=-1, keepdim=True)) / (magnitude.std(dim=-1, keepdim=True) + 1e-8)
            scores = z + self.magnitude_scale * magnitude
        else:
            scores = z

        # Select top-k indices
        topk_idx = torch.topk(scores, k=self.k, dim=-1)[1]
        mask = torch.zeros_like(preacts)
        mask.scatter_(-1, topk_idx, 1.0)

        # Add a straight-through soft mask
        if self.straight_through and self.training:
            soft = torch.softmax(scores / self.tau, dim=-1)
            soft_k = soft * (self.k / (soft.sum(dim=-1, keepdim=True) + 1e-8)).clamp(max=1.0)
            mask = mask + soft_k - soft_k.detach()

        # c = preacts * mask
        if self.z_scale is not None:
            c = (preacts * mask) * (self.z_scale + z)
        else:
            c = preacts * mask
        
        # Update dead latent statistics if training
        if self.training and self.dead_toks_threshold is not None:
            with torch.no_grad():
                # Flatten batch dimensions for statistics update
                flat_c = c.reshape(-1, c.shape[-1])
                
                # Create a tensor to track which latents were activated (> 1e-3)
                tmp = torch.zeros_like(self.stats_last_nonzero)
                activated_mask = (flat_c.abs() > 1e-3).any(dim=0)
                
                # Reset counter for activated latents
                self.stats_last_nonzero *= (~activated_mask).long()
                # Increment counter for all latents
                self.stats_last_nonzero += 1
        
        # Compute auxiliary top-k indices and values for dead latents
        auxk_indices = None
        auxk_values = None
        
        if self.aux_k > 0 and self.aux_coeff > 0.0 and self.dead_toks_threshold is not None:
            # Apply mask to get only dead latents
            masked_preacts = self.auxk_mask_fn(preacts)
            
            # Get top-k among dead latents
            if masked_preacts.abs().max() > 0:  # Only if there are dead latents
                auxk_values, auxk_indices = torch.topk(masked_preacts, k=min(self.aux_k, masked_preacts.shape[-1]), dim=-1)

        x_hat = F.linear(c, self.dict_elements, bias=self.decoder_bias)
        return HardConcreteTopKSAEOutput(
            input=x, 
            c=c, 
            output=x_hat, 
            z=z, 
            preacts=preacts, 
            mask=mask, 
            scores=scores,
            gate_logits=gate_logits,
            auxk_indices=auxk_indices,
            auxk_values=auxk_values
        )


    def compute_loss(self, output: HardConcreteTopKSAEOutput) -> SAELoss:
        """
        Loss = mse_coeff * MSE + aux_coeff * AuxK (optional)

        - No explicit L1 sparsity term (sparsity enforced by Top-K).
        - AuxK: Reconstruct the residual error (input - main_reconstruction) using dead latents
          to provide gradient signal to features that haven't been active recently.
        """
        mse_loss = F.mse_loss(output.output, output.input)
        total_loss = self.mse_coeff * mse_loss
        loss_dict: dict[str, torch.Tensor] = {
            "mse_loss": mse_loss.detach().clone(),
            "preacts_mean": output.preacts.mean().detach().clone(),
            "preacts_std": output.preacts.std().detach().clone(),
            "z_mean": output.z.mean().detach().clone(),
            "z_std": output.z.std().detach().clone(),
        }
        
        # Optional auxiliary dead-feature loss using residual reconstruction
        if (self.aux_k > 0 and self.aux_coeff > 0.0 and 
            output.auxk_indices is not None and output.auxk_values is not None):
            
            # Decode auxiliary latents
            # Create sparse representation for auxiliary latents
            aux_c = torch.zeros_like(output.preacts)
            aux_c.scatter_(-1, output.auxk_indices, output.auxk_values)
            
            # Decode auxiliary latents (no bias, as we're reconstructing residual)
            x_hat_aux = F.linear(aux_c, self.dict_elements)
            
            # Compute residual target: input - main_reconstruction + bias
            # The bias is added back because we want auxiliary latents to help reconstruct
            # the part not captured by main latents
            residual_target = output.input - output.output.detach() + self.decoder_bias.detach()
            
            # Normalized MSE for auxiliary loss
            # Avoid division by zero with small epsilon
            residual_norm = torch.norm(residual_target, p=2, dim=-1, keepdim=True)
            aux_recon_norm = torch.norm(x_hat_aux, p=2, dim=-1, keepdim=True)
            
            normalized_aux_loss = F.mse_loss(
                x_hat_aux / (aux_recon_norm + 1e-8),
                residual_target / (residual_norm + 1e-8)
            )
            
            aux_loss = normalized_aux_loss.nan_to_num(0.0)
            total_loss = total_loss + self.aux_coeff * aux_loss
            loss_dict["aux_loss"] = aux_loss.detach().clone()
        else:
            # No auxiliary loss
            loss_dict["aux_loss"] = torch.zeros((), device=output.input.device)
        
        return SAELoss(loss=total_loss, loss_dict=loss_dict)

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
