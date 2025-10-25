""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from models.components.ResUNet import ResUNet
from models.components.UNet import UNet

import torch
import torch.nn as nn
import numpy as np
import cv2

def generate_gabor_kernels(num_orientations, ksize, sigma, lambd, gamma=0.5):
    """
    Generates a stack of Gabor filter kernels.

    Args:
        num_orientations (int): Number of orientations (e.g., 90).
        ksize (int): The size of the Gabor kernel (e.g., 31).
        sigma (float): Standard deviation of the Gaussian envelope.
        lambd (float): Wavelength of the sinusoidal factor.
        gamma (float): Spatial aspect ratio.

    Returns:
        torch.Tensor: A tensor of shape (num_orientations, 1, ksize, ksize)
                      containing the Gabor kernels.
    """
    kernels = []
    # Orientations from 0 to 178 degrees, matching your U-Net output
    for i in range(num_orientations):
        theta = ((90 + 2*i)/180) * np.pi  # Angle in radians
        theta = np.arctan(-np.sin(theta)/np.cos(theta))
        kernel = cv2.getGaborKernel(
            (ksize, ksize), 
            sigma, 
            theta, 
            lambd, 
            gamma, 
            psi=0, # Phase offset, 0 and pi/2 are common
            ktype=cv2.CV_32F
        )
        # Add a channel dimension for PyTorch compatibility
        kernels.append(kernel)
    
    # Stack kernels into a single tensor
    gabor_kernels = np.stack(kernels, axis=0)
    # Add the 'in_channels' dimension
    gabor_kernels = torch.from_numpy(gabor_kernels).unsqueeze(1)
    
    return gabor_kernels

class GaborConvLayer(nn.Module):
    def __init__(self, num_orientations=90, ksize=31, sigma=4.0, lambd=10.0):
        super(GaborConvLayer, self).__init__()
        
        # Generate the fixed Gabor kernels
        gabor_weights = generate_gabor_kernels(num_orientations, ksize, sigma, lambd)
        
        # Create a non-trainable Conv2d layer
        self.conv = nn.Conv2d(
            in_channels=1, 
            out_channels=num_orientations, 
            kernel_size=ksize, 
            padding='same', # Preserves input spatial dimensions
            bias=False
        )
        
        # Assign the fixed Gabor weights and make them non-trainable
        self.conv.weight = nn.Parameter(gabor_weights, requires_grad=False)

    def forward(self, x):
        # Apply the convolution
        return self.conv(x)

class MultiTaskDirMapEnh(nn.Module):
    def __init__(self, in_ch=1, out_ch=90, ndim=2, chs: tuple[int, ...] = (64, 128, 256, 512, 1024)):
        super(MultiTaskDirMapEnh, self).__init__()
        
        self.dirmap_net = UNet(in_ch=1, out_ch=90, ndim=ndim, chs=chs)
        self.enhancer_net = ResUNet(in_ch=1, out_ch=2, ndim=ndim)

        self.gabor_layer = GaborConvLayer(
            num_orientations=90, 
            ksize=31, 
            sigma=4.0, 
            lambd=10.0
        )

    def forward(self, x):
        
        # enh_input = torch.concat([out_dirmap, x], axis=1)
        # out_enh = self.enhancer_net(x)
        # out_dirmap = self.dirmap_net(x)


        # --- Step 1: Get orientation map from U-Net ---
        # orientation_map shape: (B, 90, H, W)
        out_dirmap = self.dirmap_net(x)
        
        # Apply softmax to get probabilities for each orientation at each pixel
        orientation_probs = torch.softmax(out_dirmap, dim=1)

        # --- Step 2: Get Gabor filter responses ---
        # gabor_responses shape: (B, 90, H, W)
        gabor_responses = self.gabor_layer(x)
        
        # --- Step 3: Weight Gabor responses by orientation probabilities ---
        # Element-wise multiplication
        weighted_responses = orientation_probs * gabor_responses
        
        # --- Step 4: Sum the weighted responses to get a single feature map ---
        # Sum across the 90 channels
        # combined_feature_map shape: (B, 1, H, W)
        combined_feature_map = torch.sum(weighted_responses, dim=1, keepdim=True)
        
        # --- Step 5: Feed the enhanced map to the segmentation network ---
        out_enh = self.enhancer_net(combined_feature_map)

        return out_dirmap, out_enh




if __name__ == '__main__':
    model         =  MultiTaskDirMapEnh()

    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model         = model.to(device)

    summary(model, (1, 256, 256))