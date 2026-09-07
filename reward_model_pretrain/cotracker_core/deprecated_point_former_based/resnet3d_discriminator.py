import torch
from torch import nn
from torch.nn import functional as F

# inflated Resnet discriminator, based on https://github.com/LMescheder/GAN_stability/blob/master/gan_training/models/resnet2.py
class Discriminator3D(nn.Module):
    def __init__(self, in_channels=64, in_frame_size=256, temporal_size=16, num_classes=1, nfilter=32, **kwargs):
        super().__init__()
        
        # Calculate number of spatial and temporal downsamplings needed
        self.spatial_steps = 5  # Fixed number of spatial downsampling (as before)
        s0 = self.s0 = in_frame_size // (2 ** self.spatial_steps)
        
        # Calculate temporal downsampling steps
        self.temporal_steps = min(self.spatial_steps, (temporal_size - 1).bit_length() - 1)
        self.final_temporal_size = temporal_size // (2 ** self.temporal_steps)
        
        print(f"3D Discriminator arch: Input temporal size: {temporal_size}")
        print(f"3D Discriminator arch: Number of temporal downsampling steps: {self.temporal_steps}")
        print(f"3D Discriminator arch: Final temporal size: {self.final_temporal_size}")
        
        nf = self.nf = nfilter

        # Initial conv (no striding)
        self.conv_img = nn.Conv2d(in_channels, 2*nf, 3, stride=1, padding=1)

        # ResNet blocks (all strides set to 1)
        self.resnet_blocks = nn.ModuleList()
        channels = [
            # (1*nf, 1*nf), (1*nf, 2*nf),  # Layer 0
            (2*nf, 2*nf), (2*nf, 4*nf),  # Layer 1
            (4*nf, 4*nf), (4*nf, 8*nf),  # Layer 2
            (8*nf, 8*nf), (8*nf, 16*nf), # Layer 3
            (16*nf, 16*nf), (16*nf, 16*nf), # Layer 4
            (16*nf, 16*nf), (16*nf, 16*nf)  # Layer 5
        ]
        
        for in_ch, out_ch in channels:
            self.resnet_blocks.append(ResnetBlock3D(in_ch, out_ch, temporal_stride=1))

        self.fc = nn.Linear(16*nf*s0*s0*self.final_temporal_size, num_classes)
    
    def forward(self, x):
        # x: (B, T, C, H, W)
        batch_size = x.size(0)
        
        # Initial conv (process each frame separately)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        out = self.conv_img(x)
        out = out.view(B, T, -1, H, W)

        # Process through ResNet blocks with adaptive pooling
        for i in range(0, len(self.resnet_blocks), 2):
            # First block of the layer
            out = self.resnet_blocks[i](out)
            out = self.resnet_blocks[i+1](out)
            
            # Apply pooling if we're not at the last layer
            if i < 10:  # 5 layers * 2 blocks each = 10
                # Determine if we should do temporal pooling at this step
                temporal_kernel = 2 if i//2 < self.temporal_steps else 1
                temporal_stride = 2 if i//2 < self.temporal_steps else 1
                
                # Apply pooling with adaptive kernel and stride sizes
                out = F.avg_pool3d(
                    out.permute(0,2,1,3,4), 
                    kernel_size=(temporal_kernel,3,3),
                    stride=(temporal_stride,2,2),
                    padding=(0,1,1)
                ).permute(0,2,1,3,4)

        # Global average pooling over remaining spatial dimensions
        _, T, C, H, W = out.shape
        out = out.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)
        out = out.view(batch_size, -1)
        
        # Final classification
        out = self.fc(actvn(out))

        return out

class ResnetBlock3D(nn.Module):
    def __init__(self, fin, fout, temporal_stride=1, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Spatial pathway (2D convolutions)
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

        # Temporal pathway
        self.temporal = TemporalBlock(fout, temporal_stride=temporal_stride)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # Process spatially by merging B and T
        x_reshaped = x.reshape(B * T, C, H, W)
        
        # Spatial residual branch
        x_s = self._shortcut(x_reshaped)
        dx = self.conv_0(actvn(x_reshaped))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        # Reshape back to video format
        _, C_out, H_out, W_out = out.shape
        out = out.view(B, T, C_out, H_out, W_out)
        
        # Process temporally
        out = self.temporal(out)
        return out
    
    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s
    

class TemporalBlock(nn.Module):
    """Similar to the one in video_resnet_discriminator.py"""
    def __init__(self, channels, temporal_stride=1):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=temporal_stride,
            padding=1,
            bias=False
        )
        
        if temporal_stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=1, stride=temporal_stride, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x_perm = x.permute(0, 3, 4, 2, 1).contiguous()  # (B, H, W, C, T)
        x_flat = x_perm.view(-1, C, T)  # (B*H*W, C, T)
        
        out = self.conv1d(x_flat)
        
        if self.downsample is not None:
            identity = self.downsample(x_flat)
        else:
            identity = x_flat
        
        out = out + identity
        out = actvn(out)
        
        T_out = out.shape[-1]
        out = out.view(B, H, W, C, T_out)
        out = out.permute(0, 4, 3, 1, 2).contiguous()  # (B, T_out, C, H, W)
        return out
    
def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out

if __name__ == "__main__":
    # model = TemporalBlock(32, temporal_stride=2)
    # x = torch.randn(1, 16, 32, 32, 32)
    # out = model(x)
    # print(out.shape)

    # model = ResnetBlock3D(32, 64, temporal_stride=2)
    # x = torch.randn(1, 16, 32, 32, 32)
    # out = model(x)
    # print(out.shape)

    model = Discriminator3D(in_channels=64, in_frame_size=256, num_classes=1, nfilter=32)
    x = torch.randn(2, 16, 64, 256, 256)
    out = model(x)
    print(out.shape)