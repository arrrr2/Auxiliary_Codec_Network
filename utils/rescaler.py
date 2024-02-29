import torch
import numpy as np
from torch.nn.functional import interpolate

def image_rescaler(source:torch.Tensor | np.ndarray, input_range=1.0, output_range=1.0, 
             mode='bicubic', scale_factor=1.0, antialias=True, 
             output_format='tensor', dtype='float32', layout='chw', dim=3, 
             dispersed=False, device='cpu') -> torch.Tensor | np.ndarray:
    """
    Rescale images using specified interpolation method.
    Parameters:
    - source: Input image data (tensor or array).
    - mode: Interpolation mode ('bicubic' or 'bilinear').
    - scale_factor: Scaling factor.
    - antialias: Whether to use antialiasing.
    - output_format: Output format ('tensor' or 'array').
    - dtype: Output data type ('float32' or 'uint8').
    - layout: Channel layout in the output ('chw' or 'hwc').
    - dim: Dimensionality of the output (3 or 4).

    Returns:
    - Rescaled image in the specified output format and layout.
    """

    # Ensure input is a float32 tensor if needed
    if isinstance(source, np.ndarray):
        source = torch.tensor(source)
    
    source = source.to('cpu')
    
    if dispersed == False:
        if source.dtype != torch.float32:
            source = source.to(torch.float32)

    # Adjust input to NxCxHxW format
    if source.dim() == 3:
        source = source.unsqueeze(0)  # Add batch dimension if needed
    if source.shape[-1] in (1, 3):
        source = source.permute(0, 3, 1, 2)  # Convert to CHW

    # Perform interpolation
    if scale_factor != 1.0:
        if not dispersed:
            rescaled: torch.Tensor = interpolate(source, scale_factor=scale_factor,
                                            mode=mode, antialias=antialias)
        else: 
            if input_range != 1.0:
                source = source.to(torch.uint8)
                rescaled: torch.Tensor = interpolate(source, scale_factor=scale_factor,
                                                mode=mode, antialias=antialias)
            else: 
                source = (source * 255. + 0.5).clamp(0, 255).to(torch.uint8)
                rescaled: torch.Tensor = interpolate(source, scale_factor=scale_factor,
                                                mode=mode, antialias=antialias)
                rescaled = rescaled.to(torch.float32) / 255.
    
    # Convert to desired output format
    if layout == 'hwc':
        rescaled = rescaled.permute(0, 2, 3, 1)  # Convert back to HWC
    if dim == 3:
        rescaled = rescaled.squeeze(0)  # Remove batch dimension

    fact = output_range / input_range

    # Adjust data type
    if fact != 1. : rescaled = (rescaled * fact).clamp(0, output_range)
    if dtype == 'uint8':
        rescaled = rescaled.to(torch.uint8)
    elif dtype == 'float32':
        if dispersed and fact > 1: rescaled = rescaled + .5.to(torch.uint8)
        rescaled = rescaled.to(torch.float32)
        
    rescaled = rescaled.to(device)

    # Convert to desired output type
    if output_format == 'array':
        rescaled = rescaled.numpy()

    return rescaled


if __name__=='__main__':
    tester = (torch.ones([16, 3, 128, 128])).to(torch.uint8)
    print(1 == 1.0)
    result = image_rescaler(tester, 255., 255., 'bicubic', 2., True, 'array', 'uint8', 'hwc', 3, True)

    print(result)
    print('Done')
    