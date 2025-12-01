from enum import Enum


class SAEType(str, Enum):
    """Enum for different SAE types."""
    RELU = "relu"
    HARD_CONCRETE = "hard_concrete"
    LAGRANGIAN = "lagrangian"
    GATED = "gated"
    TOPK = "topk"
    HARD_CONCRETE_TOPK = "hard_concrete_topk"
    BATCH_TOPK = "batch_topk"
    HC_BATCH_TOPK = "hc_batch_topk"
    JUMP_RELU = "jump_relu"


class EncoderType(str, Enum):
    """Enum for different encoder types."""
    SCALE = "scale"
    SEPARATE = "separate"
    DECODER_TRANSPOSE = "decoder_transpose"
    NONE = "none"
