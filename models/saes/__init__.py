from models.saes.base import SAEConfig, SAEOutput, BaseSAE, SAELoss
from models.saes.relu_sae import ReluSAE, ReLUSAEConfig
from models.saes.hc_sae import HardConcreteSAE, HardConcreteSAEConfig, HardConcreteSAEOutput
from models.saes.lagrangian_sae import LagrangianSAE, LagrangianSAEConfig, LagrangianSAEOutput
from models.saes.gated_sae import GatedSAE, GatedSAEConfig, GatedSAEOutput
from models.saes.topk_sae import TopKSAE, TopKSAEConfig, TopKSAEOutput
from models.saes.hc_topk_sae import HardConcreteTopKSAE, HardConcreteTopKSAEConfig, HardConcreteTopKSAEOutput
from models.saes.batch_topk_sae import BatchTopKSAE, BatchTopKSAEConfig, BatchTopKSAEOutput
from models.saes.hc_batch_topk_sae import HardConcreteBatchTopKSAE, HardConcreteBatchTopKSAEConfig, HardConcreteBatchTopKSAEOutput
from models.saes.jumprelu_sae import JumpReLUSAE, JumpReLUSAEConfig, JumpReLUSAEOutput
from utils.enums import SAEType
from typing import Any, Union
import inspect


ALL_SAE_CONFIGS = [
    cls for name, cls in globals().items() 
    if inspect.isclass(cls) and issubclass(cls, SAEConfig) and cls is not SAEConfig
]

# Union type for type annotations (Python 3.9 compatible)
AllSAEConfigs = Union[
    ReLUSAEConfig,
    HardConcreteSAEConfig,
    LagrangianSAEConfig,
    GatedSAEConfig,
    TopKSAEConfig,
    HardConcreteTopKSAEConfig,
    BatchTopKSAEConfig,
    HardConcreteBatchTopKSAEConfig,
    JumpReLUSAEConfig,
]

SAE_TYPE_TO_CONFIG = {
    SAEType.HARD_CONCRETE: HardConcreteSAEConfig,
    SAEType.LAGRANGIAN: LagrangianSAEConfig,
    SAEType.RELU: ReLUSAEConfig,
    SAEType.GATED: GatedSAEConfig,
    SAEType.TOPK: TopKSAEConfig,
    SAEType.HARD_CONCRETE_TOPK: HardConcreteTopKSAEConfig,
    SAEType.BATCH_TOPK: BatchTopKSAEConfig,
    SAEType.HC_BATCH_TOPK: HardConcreteBatchTopKSAEConfig,
    SAEType.JUMP_RELU: JumpReLUSAEConfig,
}


SAE_TYPE_TO_CLS = {
    SAEType.HARD_CONCRETE: HardConcreteSAE,
    SAEType.LAGRANGIAN: LagrangianSAE,
    SAEType.RELU: ReluSAE,
    SAEType.GATED: GatedSAE,
    SAEType.TOPK: TopKSAE,
    SAEType.HARD_CONCRETE_TOPK: HardConcreteTopKSAE,
    SAEType.BATCH_TOPK: BatchTopKSAE,
    SAEType.HC_BATCH_TOPK: HardConcreteBatchTopKSAE,
    SAEType.JUMP_RELU: JumpReLUSAE,
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
    "LagrangianSAE",
    "LagrangianSAEConfig",
    "LagrangianSAEOutput",
    "GatedSAE",
    "GatedSAEConfig",
    "GatedSAEOutput",
    "TopKSAE",
    "TopKSAEConfig",
    "TopKSAEOutput",
    "HardConcreteTopKSAE",
    "HardConcreteTopKSAEConfig",
    "HardConcreteTopKSAEOutput",
    "BatchTopKSAE",
    "BatchTopKSAEConfig",
    "BatchTopKSAEOutput",
    "HardConcreteBatchTopKSAE",
    "HardConcreteBatchTopKSAEConfig",
    "HardConcreteBatchTopKSAEOutput",
    "JumpReLUSAE",
    "JumpReLUSAEConfig",
    "JumpReLUSAEOutput",
    "create_sae_config",
    "AllSAEConfigs",
]
