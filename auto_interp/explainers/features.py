from dataclasses import dataclass
import random
import torch
import numpy as np
from typing import Optional
from auto_interp.explainers.sampler import stratified_sample_by_max_activation


@dataclass
class Example:
    """
    A single example of feature data.

    Attributes:
        tokens (TensorType["seq"]): Tokenized input sequence.
        activations (TensorType["seq"]): Activation values for the input sequence.
        normalized_activations (TensorType["seq"]): Normalized activation values.
    """
    tokens: list[str]
    activations: torch.Tensor
    normalized_activations: Optional[torch.Tensor] = None
    
    @property
    def max_activation(self):
        """
        Get the maximum activation value.

        Returns:
            float: The maximum activation value.
        """
        return self.activations.max().item()


@dataclass
class Feature:
    """
    A feature extracted from a model's activations.

    Attributes:
        module_name (str): The module name associated with the feature.
        feature_index (int): The index of the feature within the module.
        sae_pos (str): Optional SAE position (alternative to module_name).
        neuron_idx (int): Optional neuron index (alternative to feature_index).
    """
    module_name: Optional[str] = None
    feature_index: Optional[int] = None
    sae_pos: Optional[str] = None
    neuron_idx: Optional[int] = None
    
    def __post_init__(self):
        """Initialize module_name and feature_index from sae_pos and neuron_idx if provided."""
        if self.sae_pos is not None and self.neuron_idx is not None:
            self.module_name = f"{self.sae_pos}_neuron_{self.neuron_idx}"
            self.feature_index = self.neuron_idx
        elif self.module_name is None or self.feature_index is None:
            raise ValueError("Must provide either (module_name, feature_index) or (sae_pos, neuron_idx)")

    def __repr__(self) -> str:
        """
        Return a string representation of the feature.

        Returns:
            str: A string representation of the feature.
        """
        return f"{self.module_name}_feature{self.feature_index}"


class FeatureRecord:
    """
    A record of feature data.

    Attributes:
        feature (Feature): The feature associated with the record.
    """
    explanation_examples: list[Example] = []
    positive_examples: list[Example] = []
    negative_examples: list[Example] = []
    max_activation: float = 0.0

    def __init__(
        self,
        feature: Feature,
    ):
        """
        Initialize the feature record.

        Args:
            feature (Feature): The feature associated with the record.
        """
        self.feature: Feature = feature
        
    @property
    def examples(self) -> list[Example]:
        """Backward compatibility: returns explanation_examples."""
        return self.explanation_examples
    
    @property
    def test(self) -> list[list[Example]]:
        """Backward compatibility: returns positive_examples as a single-element list for scorers."""
        return [self.positive_examples] if self.positive_examples else []
    
    @property
    def random_examples(self) -> list[Example]:
        """Backward compatibility: returns negative_examples for detection scorer."""
        return self.negative_examples
    
    @property
    def extra_examples(self) -> list[Example]:
        """Backward compatibility: returns negative_examples for fuzz scorer."""
        return self.negative_examples

    @classmethod
    def from_data(
        cls, 
        data: dict[str, torch.Tensor],
        feature: Feature,
        all_token_ids: list[list[str]],
        neuron_idx: int,
        num_explanation_examples: int = 10,
        num_positive_examples: int = 100,
        num_negative_examples: int = 100,
        stratified_quantiles: int | None = None,
        min_examples_required: int = 10,
        seed: int = 42,
    ) -> Optional["FeatureRecord"]:
        """
        Construct a feature record from data by sampling examples.
        
        Args:
            data: Dictionary containing 'nonzero_activations', 'data_indices', and 'neuron_indices'
            feature: The feature associated with the record
            all_token_ids: List of tokenized sequences
            neuron_idx: Index of the neuron to extract examples for
            num_explanation_examples: Number of examples to use for explanation generation
            num_positive_examples: Number of positive examples (where neuron is active) for scoring
            num_negative_examples: Number of negative examples (where neuron is inactive) for scoring
            stratified_quantiles: Number of quantiles for stratified sampling
            min_examples_required: Minimum number of examples required to create a record
            seed: Random seed for reproducibility
            
        Returns:
            FeatureRecord with sampled examples, or None if insufficient data
        """

        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Get all data for this specific neuron (positive examples)
        neuron_mask = data['neuron_indices'] == neuron_idx
        neuron_data_indices = data['data_indices'][neuron_mask]  # (n_examples,)
        neuron_activations = data['nonzero_activations'][neuron_mask]  # (n_examples, seq_len)
        
        # Check if we have enough positive examples
        if len(neuron_data_indices) < min_examples_required:
            return None
            
        record = cls(feature=feature)
        
        record.max_activation = neuron_activations.max().item()
        
        # Helper function to create Example objects
        def create_example(data_idx: int, activations: torch.Tensor) -> Example:
            # Clean up GPT-2 byte-level BPE tokens
            tokens = []
            for token in all_token_ids[data_idx]:
                # Common GPT-2 token replacements
                clean_token = token.replace("Ġ", " ")  # Space at beginning of word
                clean_token = clean_token.replace("Ċ", "\n")  # Newline
                clean_token = clean_token.replace("ĉ", "\t")  # Tab
                clean_token = clean_token.replace("Ñ", "–")  # En dash
                clean_token = clean_token.replace("ñ", "—")  # Em dash
                clean_token = clean_token.replace("Ģ", "′")  # Prime
                clean_token = clean_token.replace("ģ", "″")  # Double prime
                clean_token = clean_token.replace("Ġ", " ")  # Non-breaking space
                
                # Handle special quote marks and punctuation
                clean_token = clean_token.replace("âĢĻ", "'")  # Opening single quote
                clean_token = clean_token.replace("âĢĿ", "'")  # Closing single quote  
                clean_token = clean_token.replace("âĢľ", '"')  # Opening double quote
                clean_token = clean_token.replace("âĢĶ", '"')  # Closing double quote
                clean_token = clean_token.replace("âĢĵ", "-")  # Hyphen/dash
                clean_token = clean_token.replace("âĢ", "")  # Remove any remaining âĢ sequences
                
                tokens.append(clean_token)
            # Use the feature's global max_activation for normalization
            return Example(
                tokens=tokens,
                activations=activations.float(),
                normalized_activations=(activations.float() * 10 / record.max_activation).floor() if record.max_activation > 0 else torch.zeros_like(activations),
            )
        
        # 1. Stratified sampling for explanation and positive examples
        total_needed = min(num_explanation_examples + num_positive_examples, len(neuron_data_indices))
        
        if len(neuron_data_indices) <= total_needed:
            sampled_indices = torch.arange(len(neuron_data_indices))
        else:
            if stratified_quantiles is None:
                max_activations_per_example = neuron_activations.max(dim=1).values
                sampled_indices = torch.argsort(max_activations_per_example, descending=True)[:total_needed]
            else:
                sampled_indices = stratified_sample_by_max_activation(
                    neuron_activations=neuron_activations.float(),
                    n_samples=total_needed,
                    n_quantiles=stratified_quantiles,
                    seed=seed,
                )

        # Split the stratified samples into explanation and positive examples
        if len(sampled_indices) <= num_explanation_examples:
            explanation_indices = sampled_indices
            positive_indices = torch.tensor([], dtype=torch.long)
        else:
            explanation_indices = sampled_indices[:num_explanation_examples]
            positive_indices = sampled_indices[num_explanation_examples:]

        # Create explanation examples
        record.explanation_examples = [
            create_example(neuron_data_indices[idx].item(), neuron_activations[idx])
            for idx in explanation_indices
        ]
        # Create positive examples
        if len(positive_indices) > 0:
            record.positive_examples = [
                create_example(neuron_data_indices[idx].item(), neuron_activations[idx])
                for idx in positive_indices
            ]
        
        # 2. Sample negative examples (where neuron is NOT active)
        # Get all data indices where this neuron is NOT present
        all_possible_indices = set(range(len(all_token_ids)))
        positive_data_indices = set(neuron_data_indices.tolist())
        negative_data_indices = list(all_possible_indices - positive_data_indices)
        
        if len(negative_data_indices) > 0:
            n_negative_to_sample = min(num_negative_examples, len(negative_data_indices))
            negative_sample_indices = random.sample(negative_data_indices, n_negative_to_sample)
            
            # For negative examples, activations are all zeros
            seq_len = neuron_activations.shape[1] if len(neuron_activations) > 0 else 64  # Use default seq_len
            zero_activations = torch.zeros(seq_len)
            
            record.negative_examples = [
                create_example(idx, zero_activations)
                for idx in negative_sample_indices
            ]
        
        return record

    @classmethod
    def from_data_stratified(
        cls,
        data: dict[str, torch.Tensor],
        feature: Feature,
        all_token_ids: list[list[str]],
        neuron_idx: int,
        num_explanation_examples: int = 10,
        num_examples_per_bucket: int = 20,
        num_buckets: int = 5,
        min_examples_required: int = 10,
        seed: int = 42,
    ) -> Optional[tuple["FeatureRecord", dict[int, list[Example]]]]:
        """
        Create a feature record with stratified evaluation sets.
        
        Explanations are generated from the top activation bucket only.
        Evaluation examples are sampled from each percentile bucket separately.
        
        Args:
            data: Dictionary containing 'nonzero_activations', 'data_indices', and 'neuron_indices'
            feature: The feature associated with the record
            all_token_ids: List of tokenized sequences
            neuron_idx: Index of the neuron to extract examples for
            num_explanation_examples: Number of examples from top bucket for explanation
            num_examples_per_bucket: Number of evaluation examples per bucket
            num_buckets: Number of percentile buckets to create (default 5 = quintiles)
            min_examples_required: Minimum number of examples required to create a record
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (FeatureRecord, bucket_examples_dict) where bucket_examples_dict
            maps bucket index to list of Examples, or None if insufficient data
        """
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Get all data for this specific neuron
        neuron_mask = data['neuron_indices'] == neuron_idx
        neuron_data_indices = data['data_indices'][neuron_mask]
        neuron_activations = data['nonzero_activations'][neuron_mask]
        
        if len(neuron_data_indices) < min_examples_required:
            return None
            
        # Compute max activation for each example
        max_activations_per_example = neuron_activations.max(dim=1).values
        
        # Create percentile buckets
        percentiles = torch.linspace(0, 100, num_buckets + 1)
        # Convert to float32 for quantile computation (data may be float16)
        bucket_boundaries = torch.quantile(max_activations_per_example.float(), percentiles / 100.0)
        
        # Assign each example to a bucket based on its max activation
        bucket_assignments = torch.zeros(len(max_activations_per_example), dtype=torch.long)
        for i in range(num_buckets):
            if i < num_buckets - 1:
                mask = (max_activations_per_example >= bucket_boundaries[i]) & (max_activations_per_example < bucket_boundaries[i+1])
            else:
                # Last bucket includes the maximum value
                mask = max_activations_per_example >= bucket_boundaries[i]
            bucket_assignments[mask] = i
        
        # Create FeatureRecord with explanations from top bucket only
        record = cls(feature=feature)
        record.max_activation = neuron_activations.max().item()
        
        # Helper function to create Example objects
        def create_example(data_idx: int, activations: torch.Tensor) -> Example:
            # Clean up GPT-2 byte-level BPE tokens
            tokens = []
            for token in all_token_ids[data_idx]:
                # Common GPT-2 token replacements
                clean_token = token.replace("Ġ", " ")  # Space at beginning of word
                clean_token = clean_token.replace("Ċ", "\n")  # Newline
                clean_token = clean_token.replace("ĉ", "\t")  # Tab
                clean_token = clean_token.replace("Ñ", "–")  # En dash
                clean_token = clean_token.replace("ñ", "—")  # Em dash
                clean_token = clean_token.replace("Ģ", "′")  # Prime
                clean_token = clean_token.replace("ģ", "″")  # Double prime
                clean_token = clean_token.replace("Ġ", " ")  # Non-breaking space
                
                # Handle special quote marks and punctuation
                clean_token = clean_token.replace("âĢĻ", "'")  # Opening single quote
                clean_token = clean_token.replace("âĢĿ", "'")  # Closing single quote  
                clean_token = clean_token.replace("âĢľ", '"')  # Opening double quote
                clean_token = clean_token.replace("âĢĶ", '"')  # Closing double quote
                clean_token = clean_token.replace("âĢĵ", "-")  # Hyphen/dash
                clean_token = clean_token.replace("âĢ", "")  # Remove any remaining âĢ sequences
                
                tokens.append(clean_token)
            
            return Example(
                tokens=tokens,
                activations=activations.float(),
                normalized_activations=(activations.float() * 10 / record.max_activation).floor() if record.max_activation > 0 else torch.zeros_like(activations),
            )
        
        # Get examples from top bucket for explanation
        top_bucket_mask = bucket_assignments == (num_buckets - 1)
        top_bucket_indices = torch.where(top_bucket_mask)[0]
        
        if len(top_bucket_indices) < num_explanation_examples:
            # If not enough in top bucket, take what we can get
            explanation_indices = top_bucket_indices
        else:
            # Sample from top bucket
            perm = torch.randperm(len(top_bucket_indices))[:num_explanation_examples]
            explanation_indices = top_bucket_indices[perm]
        
        record.explanation_examples = [
            create_example(neuron_data_indices[idx].item(), neuron_activations[idx])
            for idx in explanation_indices
        ]
        
        # Create evaluation examples for each bucket
        bucket_examples = {}
        
        for bucket_idx in range(num_buckets):
            bucket_mask = bucket_assignments == bucket_idx
            bucket_indices = torch.where(bucket_mask)[0]
            
            if len(bucket_indices) == 0:
                bucket_examples[bucket_idx] = []
                continue
                
            # Sample from this bucket
            n_to_sample = min(num_examples_per_bucket, len(bucket_indices))
            if n_to_sample == len(bucket_indices):
                sampled_indices = bucket_indices
            else:
                perm = torch.randperm(len(bucket_indices))[:n_to_sample]
                sampled_indices = bucket_indices[perm]
            
            bucket_examples[bucket_idx] = [
                create_example(neuron_data_indices[idx].item(), neuron_activations[idx])
                for idx in sampled_indices
            ]
        
        # Also sample negative examples (where neuron is NOT active)
        all_possible_indices = set(range(len(all_token_ids)))
        positive_data_indices = set(neuron_data_indices.tolist())
        negative_data_indices = list(all_possible_indices - positive_data_indices)
        
        if len(negative_data_indices) > 0:
            n_negative_to_sample = min(num_examples_per_bucket * num_buckets, len(negative_data_indices))
            negative_sample_indices = random.sample(negative_data_indices, n_negative_to_sample)
            
            seq_len = neuron_activations.shape[1] if len(neuron_activations) > 0 else 64
            zero_activations = torch.zeros(seq_len)
            
            record.negative_examples = [
                create_example(idx, zero_activations)
                for idx in negative_sample_indices
            ]
        
        # Store bucket boundaries for later analysis
        record.bucket_boundaries = bucket_boundaries.tolist()
        record.bucket_percentiles = percentiles.tolist()
        
        return record, bucket_examples
