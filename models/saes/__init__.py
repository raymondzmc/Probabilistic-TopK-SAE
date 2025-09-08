from models.saes.base import SAEConfig, SAEOutput, BaseSAE, SAELoss
from models.saes.relu_sae import ReluSAE, ReLUSAEConfig
from models.saes.hc_sae import HardConcreteSAE, HardConcreteSAEConfig, HardConcreteSAEOutput
from models.saes.lagrangian_hc_sae import LagrangianHardConcreteSAE, LagrangianHardConcreteSAEConfig, LagrangianHardConcreteSAEOutput
from models.saes.gated_sae import (
    GatedSAE, GatedSAEConfig, GatedSAEOutput,
    GatedHardConcreteSAE, GatedHardConcreteSAEConfig, GatedHardConcreteSAEOutput
)
from models.saes.gumbel_topk_sae import GumbelTopKSAE, GumbelTopKSAEConfig, GumbelTopKSAEOutput
from models.saes.topk_sae import TopKSAE, TopKSAEConfig, TopKSAEOutput
from models.saes.vi_sae import VITopKSAE, VITopKSAEConfig, VITopKSAEOutput
from models.saes.hc_topk_sae import HardConcreteTopKSAE, HardConcreteTopKSAEConfig, HardConcreteTopKSAEOutput
from utils.enums import SAEType
from typing import Any, Union
import inspect


ALL_SAE_CONFIGS = [
    cls for name, cls in globals().items() 
    if inspect.isclass(cls) and issubclass(cls, SAEConfig) and cls is not SAEConfig
]

# Union type for type annotations
AllSAEConfigs = Union[*ALL_SAE_CONFIGS]

SAE_TYPE_TO_CONFIG = {
    SAEType.HARD_CONCRETE: HardConcreteSAEConfig,
    SAEType.LAGRANGIAN_HARD_CONCRETE: LagrangianHardConcreteSAEConfig,
    SAEType.RELU: ReLUSAEConfig,
    SAEType.GATED: GatedSAEConfig,
    SAEType.GATED_HARD_CONCRETE: GatedHardConcreteSAEConfig,
    SAEType.TOPK: TopKSAEConfig,
    SAEType.GUMBEL_TOPK: GumbelTopKSAEConfig,
    SAEType.VI_TOPK: VITopKSAEConfig,
    SAEType.HARD_CONCRETE_TOPK: HardConcreteTopKSAEConfig,
}


SAE_TYPE_TO_CLS = {
    SAEType.HARD_CONCRETE: HardConcreteSAE,
    SAEType.LAGRANGIAN_HARD_CONCRETE: LagrangianHardConcreteSAE,
    SAEType.RELU: ReluSAE,
    SAEType.GATED: GatedSAE,
    SAEType.GATED_HARD_CONCRETE: GatedHardConcreteSAE,
    SAEType.TOPK: TopKSAE,
    SAEType.GUMBEL_TOPK: GumbelTopKSAE,
    SAEType.VI_TOPK: VITopKSAE,
    SAEType.HARD_CONCRETE_TOPK: HardConcreteTopKSAE,
}

assert set(SAE_TYPE_TO_CONFIG.keys()) == set(SAE_TYPE_TO_CLS.keys()), f"SAE_TYPE_TO_CONFIG.keys(): {SAE_TYPE_TO_CONFIG.keys()} != SAE_TYPE_TO_CLS.keys(): {SAE_TYPE_TO_CLS.keys()}"
assert set(ALL_SAE_CONFIGS) == set(SAE_TYPE_TO_CONFIG.values()), f"ALL_SAE_CONFIGS: {ALL_SAE_CONFIGS} != SAE_TYPE_TO_CONFIG.values(): {SAE_TYPE_TO_CONFIG.values()}"


def create_sae_config(config_dict: dict[str, Any]) -> SAEConfig:
    """Factory function to create the appropriate SAE config based on sae_type.
    
    Args:
        config_dict: Dictionary containing SAE configuration parameters
        
    Returns:
        Appropriate SAEConfig subclass instance
        
    Raises:
        NotImplementedError: If sae_type is not supported
        ValueError: If sae_type is missing from config_dict
    """
    if "sae_type" not in config_dict:
        raise ValueError("sae_type must be specified in SAE config")
    
    try:
        sae_type = SAEType(config_dict["sae_type"])
    except ValueError:
        raise ValueError(f"Invalid sae_type: {config_dict['sae_type']}")
    
    return SAE_TYPE_TO_CONFIG[sae_type].model_validate(config_dict)


__all__ = [
    "BaseSAE",
    "SAEConfig", 
    "SAELoss",
    "SAEOutput",
    "ReluSAE",
    "ReLUSAEConfig",
    "HardConcreteSAE", 
    "HardConcreteSAEConfig", 
    "HardConcreteSAEOutput",
    "LagrangianHardConcreteSAE",
    "LagrangianHardConcreteSAEConfig",
    "LagrangianHardConcreteSAEOutput",
    "GatedSAE",
    "GatedSAEConfig",
    "GatedSAEOutput",
    "GatedHardConcreteSAE",
    "GatedHardConcreteSAEConfig",
    "GatedHardConcreteSAEOutput",
    "TopKSAE",
    "TopKSAEConfig",
    "TopKSAEOutput",
    "GumbelTopKSAE",
    "GumbelTopKSAEConfig",
    "GumbelTopKSAEOutput",
    "VITopKSAE",
    "VITopKSAEConfig", 
    "VITopKSAEOutput",
    "HardConcreteTopKSAE",
    "HardConcreteTopKSAEConfig",
    "HardConcreteTopKSAEOutput",
    "create_sae_config",
    "AllSAEConfigs",
]