import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    """
    Processes temporal information for every spatial location.
    It applies a 1D convolution along the time axis (with downsampling if needed),
    then performs batch normalization and ReLU activation.
    A residual connection is added so that if downsampling is applied,
    the input is projected accordingly before being added.
    
    Input: Tensor of shape (B, T, C, H, W)
    Output: Tensor of shape (B, T_out, C, H, W)
    """
    def __init__(self, channels, temporal_stride=1):
        super(TemporalBlock, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=temporal_stride,
            padding=1,
            bias=False
        )
        # self.norm = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Create a downsample layer for the residual if temporal downsampling occurs.
        if temporal_stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=1, stride=temporal_stride, bias=False),
                # nn.BatchNorm1d(channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        # Permute to process the temporal dimension for each spatial location:
        # Rearrange to (B, H, W, C, T) then flatten over (B*H*W)
        x_perm = x.permute(0, 3, 4, 2, 1).contiguous()  # (B, H, W, C, T)
        x_flat = x_perm.view(-1, C, T)  # (B*H*W, C, T)
        
        out = self.conv1d(x_flat)      # (B*H*W, C, T_out)
        # out = self.norm(out)
        
        # Residual connection: if downsampling, project the input accordingly.
        if self.downsample is not None:
            identity = self.downsample(x_flat)
        else:
            identity = x_flat
        
        out = out + identity
        out = self.relu(out)
        
        # Reshape back to (B, T_out, C, H, W)
        T_out = out.shape[-1]
        out = out.view(B, H, W, C, T_out)
        out = out.permute(0, 4, 3, 1, 2).contiguous()  # (B, T_out, C, H, W)
        return out

class VideoBasicBlock(nn.Module):
    """
    A ResNet BasicBlock inflated to process video.
    
    It first applies the standard 2D convolution operations framewise (by merging B and T),
    and a residual connection is added as in the standard ResNet block. The output is then
    reshaped back to video format and fed into the temporal block for processing along time.
    
    Input: Tensor of shape (B, T, in_channels, H, W)
    Output: Tensor of shape (B, T_out, out_channels, H_out, W_out)
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(VideoBasicBlock, self).__init__()
        self.stride = stride
        # Spatial pathway (applied framewise)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample the residual if necessary.
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                # nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

        # Temporal block: use the same stride as the spatial conv to downsample time when needed.
        self.temporal = TemporalBlock(out_channels, temporal_stride=stride)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        # Process spatially by merging B and T dimensions.
        x_reshaped = x.view(B * T, C, H, W)
        out = self.conv1(x_reshaped)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        
        # Residual connection for spatial branch.
        if self.downsample is not None:
            identity = self.downsample(x_reshaped)
        else:
            identity = x_reshaped
        out += identity
        out = self.relu(out)
        
        # Reshape back to video format: (B, T, out_channels, H_out, W_out)
        _, out_channels, H_out, W_out = out.shape
        out = out.view(B, T, out_channels, H_out, W_out)
        
        # Process the temporal dimension using the temporal block.
        out = self.temporal(out)  # (B, T_out, out_channels, H_out, W_out)
        return out

class VideoBottleneck(nn.Module):
    """
    A ResNet Bottleneck block inflated for video.

    It applies three 2D convolution operations framewise (by merging B and T),
    with batch normalization and ReLU activations between layers.
    A residual connection is added similar to the original ResNet Bottleneck block.
    The output is then reshaped back to video format and processed temporally.

    Input: Tensor of shape (B, T, in_channels, H, W)
    Output: Tensor of shape (B, T_out, out_channels * expansion, H_out, W_out)

    Note: The expansion factor is set to 4.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(VideoBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(out_channels * self.expansion)
            )
        else:
            self.downsample = None
        self.temporal = TemporalBlock(out_channels * self.expansion, temporal_stride=stride)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B * T, C, H, W)
        identity = x_reshaped

        out = self.conv1(x_reshaped)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x_reshaped)
        out += identity
        out = self.relu(out)

        # Reshape back to video format and process temporally.
        _, out_channels, H_out, W_out = out.shape
        out = out.view(B, T, out_channels, H_out, W_out)
        out = self.temporal(out)
        return out
    
class VideoResNetDiscriminator(nn.Module):
    """
    The full video discriminator.
    
    The network first processes individual frames with a 2D stem (conv, BN, ReLU, maxpool)
    and then passes the video tensor sequentially through four layers comprised of
    inflated residual blocks. Finally, global average pooling over temporal and spatial dimensions
    is applied and an FC layer outputs the final logits.
    
    Since this module will serve as the GAN discriminator with a hinge loss objective,
    it outputs raw unbounded logits (without a Sigmoid).
    """
    def __init__(self, in_channels=64, block=VideoBasicBlock, layers=[2, 2, 2, 2], num_classes=1):
        super(VideoResNetDiscriminator, self).__init__()
        self.in_channels = in_channels
        # Stem: process each frame with a 7x7 convolution.
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Build the four layers.
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # Global pooling over (T, H, W) followed by FC to produce logits.
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, channels, stride))
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        # Process each frame separately in the stem by merging B and T.
        x = x.view(B * T, C, H, W)     # (B*T, C, H, W)
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)            # (B*T, 64, H1, W1)
        
        # Restore video shape.
        _, C1, H1, W1 = x.shape
        x = x.view(B, T, C1, H1, W1)   # (B, T, 64, H1, W1)
        
        # Each layer processes spatially and then temporally.
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x shape after layer4 is (B, T_final, C_final, H_final, W_final)
        
        # Permute and pool over time and space.
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, C_final, T_final, H_final, W_final)
        x = self.avgpool(x)                         # (B, C_final, 1, 1, 1)
        x = x.view(B, -1)                           # (B, C_final)
        logits = self.fc(x)                         # (B, num_classes); raw logits for Hinge loss.
        return logits

def video_resnet18_discriminator(in_channels=64, num_classes=1):
    """
    Constructs a VideoResNetDiscriminator with 18 layers (2-2-2-2 configuration).

    Args:
        num_classes (int): Number of output classes. Default is 1 for binary discrimination.

    Returns:
        VideoResNetDiscriminator instance.
    """
    return VideoResNetDiscriminator(in_channels=in_channels, block=VideoBasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)

def video_resnet34_discriminator(in_channels=64, num_classes=1):
    """
    Constructs a VideoResNetDiscriminator with 18 layers (3-4-6-3) configuration).

    Args:
        num_classes (int): Number of output classes. Default is 1 for binary discrimination.

    Returns:
        VideoResNetDiscriminator instance.
    """
    return VideoResNetDiscriminator(in_channels=in_channels, block=VideoBasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes)

def video_resnet50_discriminator(in_channels=64, num_classes=1):
    """
    Constructs a VideoResNetDiscriminator with 50 layers (3-4-6-3 configuration)
    using the VideoBottleneck block for the spatial pathway.

    Args:
        in_channels (int): Number of channels in the input frames.
        num_classes (int): Number of output classes. Default is 1 for binary discrimination.

    Returns:
        VideoResNetDiscriminator instance configured as ResNet-50.
    """
    return VideoResNetDiscriminator(in_channels=in_channels, block=VideoBottleneck,
                                    layers=[3, 4, 6, 3], num_classes=num_classes)

if __name__ == "__main__":
    # Example usage
    model = video_resnet50_discriminator()
    print(model)
    
    # Create a dummy input: Batch size 4, 16 time steps, 3 channels, 224x224 resolution.
    dummy_input = torch.randn(4, 16, 64, 256, 256)
    logits = model(dummy_input)
    print("Output shape:", logits.shape)  # Expected output: (4, 1)
