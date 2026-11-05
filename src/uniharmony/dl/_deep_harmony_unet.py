"""Provide the U-Net used by DeepHarmony."""

# Architecture from Fig. 2 of:
# Dewey, B. E., et al. (2019). DeepHarmony: A deep learning approach to contrast
# harmonization across scanner changes. Magnetic Resonance Imaging, 64, 160-170.
# https://doi.org/10.1016/j.mri.2019.05.041

from uniharmony.dl._torch import nn, torch


__all__ = ["DeepHarmonyUNet"]

# Number of feature maps at each resolution level (Fig. 2)
_FEATURES = (16, 32, 64, 128)
_BOTTLENECK_FEATURES = 256

#: Spatial size of inputs must be a multiple of this (4 stride-2 downsamplings).
SIZE_MULTIPLE = 2 ** len(_FEATURES)


def _conv_relu_bn(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> nn.Sequential:
    """Convolution followed by ReLU and batch normalization (in that order, as in Fig. 2)."""
    padding = 1  # 3x3 / stride 1 keeps the size; 4x4 / stride 2 halves it exactly
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
        # Keras defaults (the original implementation): momentum 0.99 (= 0.01 in PyTorch), epsilon 1e-3
        nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
    )


def _upconv_relu_bn(in_channels: int, out_channels: int) -> nn.Sequential:
    """4x4 transposed convolution with stride 1/2 (doubles the size), ReLU and batch normalization."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
    )


class DeepHarmonyUNet(nn.Module):
    """2D U-Net of DeepHarmony (Dewey et al., 2019, Fig. 2).

    Differences to the original U-Net, as described in the paper:

    * Downsampling with 4x4 convolutions of stride 2 and upsampling with 4x4
      transposed convolutions of stride 1/2 (no pooling or interpolation).
    * Fewer feature maps: 16, 32, 64 and 128 per level and 256 at the bottleneck.
    * The input contrasts are concatenated to the last feature map, so the
      final 1x1 convolution only *augments* the inputs.
    * Every convolution is followed by ReLU and batch normalization, except the
      final 1x1 convolution, which is followed by ReLU only.

    Weights are initialized like Keras (Glorot-uniform kernels, zero biases),
    the framework of the original implementation.

    Parameters
    ----------
    n_input_contrasts : int
        Number of input contrasts (channels), :math:`C_I` in the paper.
    n_output_contrasts : int
        Number of output contrasts (channels), :math:`C_O` in the paper.

    Notes
    -----
    Height and width of the input must be multiples of 16.

    """

    def __init__(self, n_input_contrasts: int, n_output_contrasts: int) -> None:
        super().__init__()
        if n_input_contrasts < 1 or n_output_contrasts < 1:
            raise ValueError("n_input_contrasts and n_output_contrasts must be >= 1")
        self.n_input_contrasts = n_input_contrasts
        self.n_output_contrasts = n_output_contrasts

        f0, f1, f2, f3 = _FEATURES
        # Encoder: 3x3 conv to the level's features, then 4x4 stride-2 conv keeping the features
        self.enc0 = _conv_relu_bn(n_input_contrasts, f0, kernel_size=3)  # 128x128
        self.down1 = _conv_relu_bn(f0, f0, kernel_size=4, stride=2)  # 64x64
        self.enc1 = _conv_relu_bn(f0, f1, kernel_size=3)
        self.down2 = _conv_relu_bn(f1, f1, kernel_size=4, stride=2)  # 32x32
        self.enc2 = _conv_relu_bn(f1, f2, kernel_size=3)
        self.down3 = _conv_relu_bn(f2, f2, kernel_size=4, stride=2)  # 16x16
        self.enc3 = _conv_relu_bn(f2, f3, kernel_size=3)
        self.down4 = _conv_relu_bn(f3, f3, kernel_size=4, stride=2)  # 8x8
        self.bottleneck = _conv_relu_bn(f3, _BOTTLENECK_FEATURES, kernel_size=3)
        # Decoder: upsample, concatenate the skip connection, 3x3 conv
        self.up3 = _upconv_relu_bn(_BOTTLENECK_FEATURES, f3)  # 16x16
        self.dec3 = _conv_relu_bn(2 * f3, f3, kernel_size=3)
        self.up2 = _upconv_relu_bn(f3, f2)  # 32x32
        self.dec2 = _conv_relu_bn(2 * f2, f2, kernel_size=3)
        self.up1 = _upconv_relu_bn(f2, f1)  # 64x64
        self.dec1 = _conv_relu_bn(2 * f1, f1, kernel_size=3)
        self.up0 = _upconv_relu_bn(f1, f0)  # 128x128
        self.dec0 = _conv_relu_bn(2 * f0, f0, kernel_size=3)
        # Final 1x1 conv on [features, input contrasts], ReLU, no normalization
        self.out = nn.Sequential(
            nn.Conv2d(f0 + n_input_contrasts, n_output_contrasts, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights like Keras: Glorot-uniform kernels and zero biases."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                module.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map input contrasts to output contrasts.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_input_contrasts, height, width)
            Input images. Height and width must be multiples of 16.

        Returns
        -------
        torch.Tensor, shape (batch, n_output_contrasts, height, width)
            Harmonized images.

        """
        if x.ndim != 4 or x.shape[1] != self.n_input_contrasts:
            raise ValueError(f"Expected input of shape (batch, {self.n_input_contrasts}, H, W), got {tuple(x.shape)}")
        if x.shape[2] % SIZE_MULTIPLE or x.shape[3] % SIZE_MULTIPLE:
            raise ValueError(f"Height and width must be multiples of {SIZE_MULTIPLE}, got {tuple(x.shape[2:])}")
        s0 = self.enc0(x)
        s1 = self.enc1(self.down1(s0))
        s2 = self.enc2(self.down2(s1))
        s3 = self.enc3(self.down3(s2))
        h = self.bottleneck(self.down4(s3))
        h = self.dec3(torch.cat([s3, self.up3(h)], dim=1))
        h = self.dec2(torch.cat([s2, self.up2(h)], dim=1))
        h = self.dec1(torch.cat([s1, self.up1(h)], dim=1))
        h = self.dec0(torch.cat([s0, self.up0(h)], dim=1))
        return self.out(torch.cat([h, x], dim=1))
