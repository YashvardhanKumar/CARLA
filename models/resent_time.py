
import torch
from torch import nn
import torch.nn.functional as F


# class Conv1dSamePadding(nn.Conv1d):
#     """Represents the "Same" padding functionality from Tensorflow.
#     See: https://github.com/pytorch/pytorch/issues/3867
#     Note that the padding argument in the initializer doesn't do anything now
#     """
#     def forward(self, input):
#         return conv1d_same_padding(input, self.weight, self.bias, self.stride,
#                                    self.dilation, self.groups)


# def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
#     # stride and dilation are expected to be tuples.
#     kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
#     l_out = l_in = input.size(2)
#     padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
#     if padding % 2 != 0:
#         input = F.pad(input, [0, 1])

#     return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
#                     padding=padding // 2,
#                     dilation=dilation, groups=groups)

# class ConvBlock(nn.Module):

#     def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
#                  stride: int) -> None:
#         super().__init__()

#         self.layers = nn.Sequential(
#             Conv1dSamePadding(in_channels=in_channels,
#                               out_channels=out_channels,
#                               kernel_size=kernel_size,
#                               stride=stride),
#             nn.BatchNorm1d(num_features=out_channels),
#             nn.ReLU(),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

#         return self.layers(x)

# class ResNetBlock(nn.Module):

#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int
#     ) -> None:
#         super().__init__()

#         channels = [in_channels, out_channels, out_channels, out_channels]
#         kernel_sizes = [8, 5, 3]

#         self.layers = nn.Sequential(*[
#             ConvBlock(
#                 in_channels=channels[i],
#                 out_channels=channels[i + 1],
#                 kernel_size=kernel_sizes[i],
#                 stride=1
#             ) for i in range(len(kernel_sizes))
#         ])

#         self.match_channels = False
#         if in_channels != out_channels:
#             self.match_channels = True
#             self.residual = nn.Sequential(*[
#                 Conv1dSamePadding(
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     kernel_size=1,
#                     stride=1
#                 ),
#                 nn.BatchNorm1d(num_features=out_channels)
#             ])

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.match_channels:
#             return self.layers(x) + self.residual(x)
#         return self.layers(x)

# class ResNetRepresentation(nn.Module):
#     """A PyTorch implementation of the ResNet Baseline
#     Attributes
#     ----------
#     sequence_length:
#         The size of the input sequence
#     mid_channels:
#         The 3 residual blocks will have as output channels:
#         [mid_channels, mid_channels * 2, mid_channels * 2]
#     num_pred_classes:
#         The number of output classes
#     """

#     def __init__(self, in_channels: int, mid_channels: int = 4) -> None:
#         super().__init__()

#         # for easier saving and loading
#         self.input_args = {
#             'in_channels': in_channels,
#         }

#         self.layers = nn.Sequential(*[
#             ResNetBlock(in_channels=in_channels, out_channels=mid_channels),
#             ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
#             ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2),
#         ])

#         #self.avgpool = nn.AdaptiveAvgPool1d(1)

#     def forward(self, x: torch.Tensor):
#         z = self.layers(x)
#         z = z.mean(dim=-1)
#         return z

# def resnet_ts(**kwargs):
#     return {'backbone': ResNetRepresentation(**kwargs), 'dim': kwargs['mid_channels']*2}

# import torch
# from torch import nn
# import torch.nn.functional as F


# class Conv1dSamePadding(nn.Conv1d):
#     """Represents the "Same" padding functionality from Tensorflow."""
#     def forward(self, input):
#         return conv1d_same_padding(input, self.weight, self.bias, self.stride,
#                                    self.dilation, self.groups)


# def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
#     # stride and dilation are expected to be tuples.
#     kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
#     l_out = l_in = input.size(2)
#     padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
#     if padding % 2 != 0:
#         input = F.pad(input, [0, 1])

#     return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
#                    padding=padding // 2,
#                    dilation=dilation, groups=groups)


# class ConvBlock(nn.Module):
#     """Convolutional Block with Conv1d, BatchNorm, and GELU activation."""
#     def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
#                  stride: int) -> None:
#         super().__init__()

#         assert in_channels > 0, "in_channels must be > 0"
#         assert out_channels > 0, "out_channels must be > 0"

#         self.layers = nn.Sequential(
#             Conv1dSamePadding(in_channels=in_channels,
#                               out_channels=out_channels,
#                               kernel_size=kernel_size,
#                               stride=stride),
#             # nn.BatchNorm1d(num_features=out_channels),
#             nn.LayerNorm(out_channels),
#             nn.GELU(),  # ReLU replaced with GELU
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.layers(x)


# class SEBlock(nn.Module):
#     """Squeeze-and-Excitation Block for channel-wise feature recalibration."""
#     def __init__(self, channels: int, reduction: int = 16):
#         super(SEBlock, self).__init__()
#         reduced_channels = max(channels // reduction, 1)  # Prevent zero channels
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             nn.Conv1d(channels, reduced_channels, kernel_size=1),
#             nn.GELU(),
#             nn.Conv1d(reduced_channels, channels, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         se_weight = self.se(x)
#         return x * se_weight


# class ResNetBlock(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int) -> None:
#         super().__init__()
#         assert in_channels > 0, "in_channels must be > 0"
#         assert out_channels > 0, "out_channels must be > 0"

#         channels = [in_channels, out_channels, out_channels, out_channels]
#         kernel_sizes = [8, 5, 3]

#         self.layers = nn.Sequential(*[
#             ConvBlock(
#                 in_channels=channels[i],
#                 out_channels=channels[i + 1],
#                 kernel_size=kernel_sizes[i],
#                 stride=1
#             ) for i in range(len(kernel_sizes))
#         ])

#         self.match_channels = False
#         if in_channels != out_channels:
#             self.match_channels = True
#             self.residual = nn.Sequential(*[
#                 Conv1dSamePadding(
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     kernel_size=1,
#                     stride=1
#                 ),
#                 nn.BatchNorm1d(num_features=out_channels)
#             ])

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.match_channels:
#             return self.layers(x) + self.residual(x)
#         return self.layers(x)


# class ResNetRepresentation(nn.Module):
#     """A PyTorch implementation of the ResNet Baseline."""
#     def __init__(self, in_channels: int, mid_channels: int = 4) -> None:
#         super().__init__()
#         assert in_channels > 0, "in_channels must be > 0"
#         assert mid_channels > 0, "mid_channels must be > 0"

#         self.input_args = {
#             'in_channels': in_channels,
#         }

#         self.layers = nn.Sequential(*[
#             ResNetBlock(in_channels=in_channels, out_channels=mid_channels),
#             ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
#             ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2),
#         ])

#         # Uncomment if needed
#         # self.avgpool = nn.AdaptiveAvgPool1d(1)

#     def forward(self, x: torch.Tensor):
#         z = self.layers(x)
#         z = z.mean(dim=-1)
#         return z

# class BottleneckResNetRepresentation(nn.Module):
#     """Enhanced ResNet Representation with Bottleneck Blocks, SE Blocks, and Self-Attention."""
#     def __init__(self, in_channels: int, mid_channels: int = 4, dropout_p: float = 0.1, num_heads: int = 4) -> None:
#         super().__init__()

#         self.input_args = {
#             'in_channels': in_channels,
#         }

#         self.layers = nn.Sequential(
#             BottleneckResNetBlock(
#                 in_channels=in_channels,
#                 bottleneck_channels=mid_channels,
#                 out_channels=mid_channels * 2,
#                 stride=1,
#                 dropout_p=dropout_p
#             ),
#             BottleneckResNetBlock(
#                 in_channels=mid_channels * 2,
#                 bottleneck_channels=mid_channels * 2,
#                 out_channels=mid_channels * 4,
#                 stride=2,
#                 dropout_p=dropout_p
#             ),
#             BottleneckResNetBlock(
#                 in_channels=mid_channels * 4,
#                 bottleneck_channels=mid_channels * 2,
#                 out_channels=mid_channels * 4,
#                 stride=1,
#                 dropout_p=dropout_p
#             ),
#         )

#         self.attention = nn.MultiheadAttention(embed_dim=mid_channels * 4, num_heads=num_heads, batch_first=True)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)

#     def forward(self, x: torch.Tensor):
#         z = self.layers(x)  # Shape: (batch_size, channels, seq_length)
#         z = z.permute(0, 2, 1)  # Shape: (batch_size, seq_length, channels)
#         attn_output, _ = self.attention(z, z, z)  # Self-attention
#         attn_output = attn_output.permute(0, 2, 1)  # Shape: (batch_size, channels, seq_length)
#         z = self.avgpool(attn_output).squeeze(-1)  # Shape: (batch_size, channels)
#         return z


# def resnet_ts(**kwargs):
#     mid_channels = kwargs.get('mid_channels', 4)
#     assert mid_channels > 0, "mid_channels must be > 0"

#     return {
#         'backbone': ResNetRepresentation(**kwargs),
#         'dim': mid_channels * 2
#     }

import torch
from torch import nn
import torch.nn.functional as F


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow."""
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                   padding=padding // 2,
                   dilation=dilation, groups=groups)


class ConvBlock(nn.Module):
    """Convolutional Block with Conv1d, BatchNorm, and GELU activation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.GELU(),  # ReLU replaced with GELU
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.layers(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel-wise feature recalibration."""
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        # Ensure that the reduced channels are at least 1
        reduced_channels = max(channels // reduction, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, reduced_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        se_weight = self.se(x)
        return x * se_weight


class BottleneckResNetBlock(nn.Module):
    """Bottleneck Residual Block with SE Block and Dropout."""
    def __init__(
            self,
            in_channels: int,
            bottleneck_channels: int,
            out_channels: int,
            stride: int = 1,
            dropout_p: float = 0.0
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            # 1x1 convolution to reduce dimensions
            Conv1dSamePadding(in_channels, bottleneck_channels, kernel_size=1, stride=1),
            nn.BatchNorm1d(bottleneck_channels),
            nn.GELU(),

            # 3x3 convolution (main convolution)
            Conv1dSamePadding(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride),
            nn.BatchNorm1d(bottleneck_channels),
            nn.GELU(),

            # 1x1 convolution to restore dimensions
            Conv1dSamePadding(bottleneck_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_channels),
            SEBlock(out_channels),  # Incorporate SE Block
            nn.Dropout(p=dropout_p)  # Dropout for regularization
        )

        self.match_channels = False
        if in_channels != out_channels or stride != 1:
            self.match_channels = True
            self.residual = nn.Sequential(
                Conv1dSamePadding(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride
                ),
                nn.BatchNorm1d(num_features=out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        if self.match_channels:
            residual = self.residual(x)
        else:
            residual = x
        return F.gelu(out + residual)  # Final activation after addition


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise Separable Convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = Conv1dSamePadding(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, dilation=dilation, groups=in_channels
        )
        self.pointwise = Conv1dSamePadding(
            in_channels, out_channels, kernel_size=1, stride=1
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BottleneckResNetRepresentation(nn.Module):
    """Enhanced ResNet Representation with Bottleneck Blocks and SE Blocks."""
    def __init__(self, in_channels: int, mid_channels: int = 4, dropout_p: float = 0.1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
        }

        self.layers = nn.Sequential(
            # First Bottleneck Block
            BottleneckResNetBlock(
                in_channels=in_channels,
                bottleneck_channels=mid_channels,
                out_channels=mid_channels * 2,
                stride=1,
                dropout_p=dropout_p
            ),
            # Second Bottleneck Block
            BottleneckResNetBlock(
                in_channels=mid_channels * 2,
                bottleneck_channels=mid_channels * 2,
                out_channels=mid_channels * 4,
                stride=2,  # Downsampling
                dropout_p=dropout_p
            ),
            # Third Bottleneck Block
            BottleneckResNetBlock(
                in_channels=mid_channels * 4,
                bottleneck_channels=mid_channels * 2,
                out_channels=mid_channels * 4,
                stride=1,
                dropout_p=dropout_p
            ),
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling

    def forward(self, x: torch.Tensor):
        z = self.layers(x)
        z = self.avgpool(z).squeeze(-1)  # Shape: (batch_size, channels)
        return z


def resnet_ts(**kwargs):
    return {
        'backbone': BottleneckResNetRepresentation(**kwargs),
        'dim': kwargs['mid_channels'] * 4  # Adjusted based on the new architecture
    }
