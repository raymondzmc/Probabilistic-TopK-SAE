import torch

def init_decoder_orthogonal_cuda(decoder_weight: torch.Tensor) -> torch.Tensor:
    """
    Initialize decoder weights to be orthogonal using CUDA if available, otherwise CPU.
    
    This is equivalent to: torch.nn.init.orthogonal_(decoder_weight.data.T).T
    but performs the initialization on CUDA to potentially speed up the computation
    for large matrices, then moves the result back to the original device.
    
    Args:
        decoder_weight: Decoder weight tensor to initialize (shape: input_size x n_dict_components)
        
    Returns:
        Orthogonally initialized decoder weight tensor on the original device
    """
    original_device = decoder_weight.device
    
    # Use CUDA if available, otherwise use the original device
    if torch.cuda.is_available():
        W = decoder_weight.data.clone().to("cuda")
        W = torch.nn.init.orthogonal_(W.T).T
        return W.to(original_device).clone()
    else:
        # If CUDA is not available, perform initialization on the original device
        W = decoder_weight.data.clone()
        W = torch.nn.init.orthogonal_(W.T).T
        return W