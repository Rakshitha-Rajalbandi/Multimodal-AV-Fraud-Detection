import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

class ImageDeepfakeModel(nn.Module):
    """
    High-Precision Single Image Deepfake & AI-Generation Classifier based on EfficientNet-B0.
    Input shape: (batch_size, 3, height, width)
    Output shape: (batch_size, 2) [Logits for Class 0 = FAKE, Class 1 = REAL]
    """
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class EnhancedImageDeepfakeModel(nn.Module):
    """
    Enhanced deepfake classifier: EfficientNet-B4 backbone + optional FFT frequency branch.
    Input shape: (batch_size, 3, H, W)  — recommended H=W=380 for B4
    Output shape: (batch_size, 2) [logits: class 0=FAKE, class 1=REAL]
    """
    def __init__(self, num_classes=2, dropout=0.4, use_fft_branch=True):
        super().__init__()
        self.use_fft_branch = use_fft_branch

        # EfficientNet-B4 backbone; replace classifier with Identity
        self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        # B4 features[-1] outputs 1792 channels after avgpool
        in_features = 1792

        if use_fft_branch:
            # Lightweight FFT CNN branch: [B,1,H,W] -> [B,128]
            self.fft_branch = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 128),
                nn.SiLU(),
            )
            fused_dim = in_features + 128  # 1920
        else:
            self.fft_branch = None
            fused_dim = in_features  # 1792

        # Classifier head
        self.classifier_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim, 512),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # RGB backbone stream: features -> avgpool -> flatten -> [B, 1792]
        feat = self.backbone.features(x)
        feat = self.backbone.avgpool(feat)
        feat = torch.flatten(feat, 1)  # [B, 1792]

        if self.use_fft_branch and self.fft_branch is not None:
            # Convert to grayscale
            gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]  # [B, H, W]
            # 2D FFT magnitude spectrum
            fft2 = torch.fft.fft2(gray)
            fft_shift = torch.fft.fftshift(fft2, dim=(-2, -1))
            magnitude = torch.log(torch.abs(fft_shift) + 1e-8)  # [B, H, W]
            fft_input = magnitude.unsqueeze(1)  # [B, 1, H, W]
            fft_feat = self.fft_branch(fft_input)  # [B, 128]
            fused = torch.cat([feat, fft_feat], dim=1)  # [B, 1920]
        else:
            fused = feat  # [B, 1792]

        return self.classifier_head(fused)  # [B, 2]
