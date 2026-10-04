import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for EfficientNet-B0.
    Calculates exact spatial feature activations that trigger deepfake classification.
    """

    def __init__(self, model, is_video_model=True):
        self.model = model
        self.is_video_model = is_video_model
        self.model.eval()
        
        # Target the final convolutional feature layer of EfficientNet backbone
        self.target_layer = self.model.backbone.features[-1]

        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, target_class=0):
        """
        Input: input_tensor [1, 3, H, W]
        Target Class: 0 = FAKE, 1 = REAL
        Output: Grad-CAM heatmap array [H, W] normalized to [0.0, 1.0]
        """
        self.model.zero_grad()
        
        if self.is_video_model:
            # Format as 5D sequence tensor [1, 1, C, H, W] for temporal model
            if input_tensor.dim() == 4:
                seq_tensor = input_tensor.unsqueeze(1)
            else:
                seq_tensor = input_tensor
        else:
            # Image model expects standard 4D [B, C, H, W] — no upranking
            seq_tensor = input_tensor

        output = self.model(seq_tensor)  # [1, 2]
        score = output[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback uniform map if hooks didn't capture gradients
            return np.ones((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32) * 0.5

        # Take activations and gradients of single frame [C, H', W']
        activations = self.activations[0]
        gradients = self.gradients[0]

        # Global Average Pooling of Gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam.cpu().numpy()

        if cam.max() > 1e-6:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        # Resize heatmap to match input frame resolution
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam_resized = cv2.resize(cam, (w, h))
        return cam_resized


def overlay_heatmap_on_image(pil_img, cam_mask, alpha=0.50):
    """
    Blends original PIL Image with Grad-CAM heatmap using OpenCV COLORMAP_JET.
    Red/Warm colors = High Deepfake Anomaly Region
    Blue/Cool colors = Authentic Background Region
    """
    rgb_img = np.array(pil_img.convert("RGB"))
    h, w = rgb_img.shape[:2]

    # Resize CAM mask
    cam_resized = cv2.resize(cam_mask, (w, h))
    cam_uint8 = np.uint8(255 * cam_resized)

    # Apply Jet Color Map
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Alpha blending
    blended = np.float32(heatmap_rgb) * alpha + np.float32(rgb_img) * (1.0 - alpha)
    blended = np.uint8(np.clip(blended, 0, 255))
    return Image.fromarray(blended), heatmap_rgb
