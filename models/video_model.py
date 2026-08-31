# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from transformers import CLIPVisionModel, CLIPImageProcessor


# class SpatioTemporalAdapter(nn.Module):
#     def __init__(self, dim=1024, bottleneck=256):
#         super().__init__()

#         self.down_proj = nn.Linear(dim, bottleneck)
#         self.up_proj = nn.Linear(bottleneck, dim)

#         self.layer_norm = nn.LayerNorm(dim)

#         self.temporal_attn = nn.MultiheadAttention(
#             embed_dim=dim,
#             num_heads=8,
#             batch_first=True
#         )

#         self.temporal_norm = nn.LayerNorm(dim)

#     def forward(self, x):
#         """
#         x shape:
#             (batch, frames, 1024)
#         """

#         # Spatial adapter
#         residual = x

#         x = self.layer_norm(x)

#         x = self.down_proj(x)
#         x = F.gelu(x)
#         x = self.up_proj(x)

#         x = x + residual

#         # Temporal attention
#         residual = x

#         x = self.temporal_norm(x)

#         attn_out, _ = self.temporal_attn(
#             x, x, x,
#             need_weights=False
#         )

#         x = attn_out + residual

#         return x


# class VideoDeepfakeModel(nn.Module):

#     def __init__(
#         self,
#         clip_name="openai/clip-vit-large-patch14"
#     ):
#         super().__init__()

#         # CLIP vision backbone
#         self.clip_vision = CLIPVisionModel.from_pretrained(
#             clip_name
#         )

#         # The checkpoint was trained with a frozen CLIP backbone.
#         for param in self.clip_vision.parameters():
#             param.requires_grad = False

#         # Spatio-temporal adapter
#         self.st_adapter = SpatioTemporalAdapter(
#             dim=1024,
#             bottleneck=256
#         )

#         # Classifier
#         self.classifier = nn.Sequential(
#             nn.LayerNorm(1024),
#             nn.Identity(),

#             nn.Linear(1024, 512),
#             nn.ReLU(),

#             nn.Dropout(0.1),

#             nn.Linear(512, 2)
#         )

#     def forward(self, x):
#         """
#         x:
#             (batch, frames, 3, 224, 224)

#         returns:
#             logits: (batch, 2)
#         """

#         batch_size, frames, channels, height, width = x.shape

#         # Process all frames together through CLIP
#         x = x.reshape(
#             batch_size * frames,
#             channels,
#             height,
#             width
#         )

#         outputs = self.clip_vision(
#             pixel_values=x
#         )

#         # CLS representation
#         features = outputs.pooler_output

#         # Back to video sequence
#         features = features.reshape(
#             batch_size,
#             frames,
#             1024
#         )

#         # Temporal adapter
#         features = self.st_adapter(features)

#         # Temporal pooling
#         video_features = features.mean(dim=1)

#         # Classifier
#         logits = self.classifier(video_features)

#         return logits


# # def load_video_model(
# #     checkpoint_path,
# #     device="cpu"
# # ):

# #     device = torch.device(device)

# #     print("Loading CLIP + video model...")

# #     model = VideoDeepfakeModel()

# #     checkpoint = torch.load(
# #         checkpoint_path,
# #         map_location="cpu",
# #         weights_only=False
# #     )

# #     state_dict = checkpoint["model_state_dict"]

# #     # Remove optimizer etc. and load trained weights
# #     missing, unexpected = model.load_state_dict(
# #         state_dict,
# #         strict=False
# #     )

# #     print("\nModel loading report")
# #     print("--------------------")

# #     print("Missing keys:")
# #     for key in missing:
# #         print("  ", key)

# #     print("\nUnexpected keys:")
# #     for key in unexpected:
# #         print("  ", key)

# #     model.to(device)
# #     model.eval()

# #     return model


# def load_video_model(
#     checkpoint_path,
#     device="cpu"
# ):
#     device = torch.device(device)

#     print("Loading CLIP + video model...")

#     model = VideoDeepfakeModel()

#     checkpoint = torch.load(
#         checkpoint_path,
#         map_location="cpu",
#         weights_only=False
#     )

#     state_dict = checkpoint["model_state_dict"]

#     # ---------------------------------------------------------
#     # Fix CLIP checkpoint key names
#     #
#     # Checkpoint:
#     # clip_vision.vision_model.embeddings...
#     #
#     # Current model:
#     # clip_vision.embeddings...
#     # ---------------------------------------------------------

#     corrected_state_dict = {}

#     for key, value in state_dict.items():

#         if key.startswith("clip_vision.vision_model."):
#             new_key = key.replace(
#                 "clip_vision.vision_model.",
#                 "clip_vision.",
#                 1
#             )
#             corrected_state_dict[new_key] = value

#         else:
#             corrected_state_dict[key] = value

#     # Load corrected checkpoint
#     result = model.load_state_dict(
#         corrected_state_dict,
#         strict=False
#     )

#     print("\nModel loading report")
#     print("--------------------")

#     print("Missing keys:", len(result.missing_keys))
#     print("Unexpected keys:", len(result.unexpected_keys))

#     if result.missing_keys:
#         print("\nMissing keys:")
#         for key in result.missing_keys:
#             print("  ", key)

#     if result.unexpected_keys:
#         print("\nUnexpected keys:")
#         for key in result.unexpected_keys:
#             print("  ", key)

#     model.to(device)
#     model.eval()

#     return model


# def calculate_fraud_score(logits):

#     probabilities = torch.softmax(
#         logits,
#         dim=-1
#     )

#     real_probability = float(
#         probabilities[0, 0]
#     )

#     fake_probability = float(
#         probabilities[0, 1]
#     )

#     fraud_score = fake_probability * 100

#     return {
#         "real_probability": real_probability * 100,
#         "fake_probability": fake_probability * 100,
#         "fraud_score": fraud_score
#     }



# import torch
# import torch.nn as nn
# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# class VideoDeepfakeModel(nn.Module):

#     def __init__(self, num_classes=2, hidden_dim=256):
#         super().__init__()

#         # --------------------------------------------------
#         # EfficientNet-B0 backbone
#         # --------------------------------------------------
#         weights = EfficientNet_B0_Weights.DEFAULT

#         self.backbone = efficientnet_b0(weights=weights)

#         # EfficientNet-B0 feature dimension
#         feature_dim = self.backbone.classifier[1].in_features

#         # Remove original ImageNet classifier
#         self.backbone.classifier = nn.Identity()

#         # Initially freeze EfficientNet
#         for param in self.backbone.parameters():
#             param.requires_grad = False

#         # --------------------------------------------------
#         # Temporal GRU
#         # --------------------------------------------------
#         self.temporal_gru = nn.GRU(
#             input_size=feature_dim,
#             hidden_size=hidden_dim,
#             num_layers=1,
#             batch_first=True
#         )

#         # --------------------------------------------------
#         # Classifier
#         # --------------------------------------------------
#         self.classifier = nn.Sequential(
#             nn.LayerNorm(hidden_dim),

#             nn.Linear(hidden_dim, 128),
#             nn.ReLU(),

#             nn.Dropout(0.2),

#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         """
#         Input:
#             x = (batch, frames, 3, 160, 160)

#         Output:
#             logits = (batch, 2)

#         Class mapping:
#             0 = Fake
#             1 = Real
#         """

#         batch_size, frames, channels, height, width = x.shape

#         # --------------------------------------------------
#         # Process all frames through EfficientNet together
#         # --------------------------------------------------

#         x = x.reshape(
#             batch_size * frames,
#             channels,
#             height,
#             width
#         )

#         features = self.backbone(x)

#         # EfficientNet output:
#         # (batch * frames, 1280)

#         features = features.reshape(
#             batch_size,
#             frames,
#             -1
#         )

#         # --------------------------------------------------
#         # Temporal modelling
#         # --------------------------------------------------

#         temporal_output, hidden = self.temporal_gru(features)

#         # Take representation from final time step
#         video_features = temporal_output[:, -1, :]

#         # --------------------------------------------------
#         # Classification
#         # --------------------------------------------------

#         logits = self.classifier(video_features)

#         return logits


# def load_video_model(
#     checkpoint_path=None,
#     device="cpu"
# ):

#     device = torch.device(device)

#     print("Loading EfficientNet-B0 video model...")

#     model = VideoDeepfakeModel()

#     # --------------------------------------------------
#     # Load trained checkpoint if provided
#     # --------------------------------------------------

#     if checkpoint_path is not None:

#         print("Loading checkpoint:")
#         print(checkpoint_path)

#         checkpoint = torch.load(
#             checkpoint_path,
#             map_location="cpu",
#             weights_only=False
#         )

#         # Support both:
#         # checkpoint["model_state_dict"]
#         # and direct state_dict

#         if "model_state_dict" in checkpoint:
#             state_dict = checkpoint["model_state_dict"]
#         else:
#             state_dict = checkpoint

#         result = model.load_state_dict(
#             state_dict,
#             strict=True
#         )

#         print("Checkpoint loaded successfully.")

#     else:

#         print("No trained checkpoint supplied.")
#         print("Using ImageNet-pretrained EfficientNet-B0.")
#         print("The classifier is NOT yet trained for deepfake detection.")

#     model.to(device)
#     model.eval()

#     return model


# def calculate_fraud_score(logits):

#     probabilities = torch.softmax(
#         logits,
#         dim=-1
#     )

#     fake_probability = float(
#         probabilities[0, 0]
#     )

#     real_probability = float(
#         probabilities[0, 1]
#     )

#     fraud_score = fake_probability * 100

#     return {
#         "real_probability": real_probability * 100,
#         "fake_probability": fake_probability * 100,
#         "fraud_score": fraud_score
#     }



import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class VideoDeepfakeModel(nn.Module):

    def __init__(self, num_classes=2, hidden_dim=256):

        super().__init__()

        # ====================================================
        # EfficientNet-B0
        # ====================================================

        weights = EfficientNet_B0_Weights.DEFAULT

        self.backbone = efficientnet_b0(weights=weights)

        feature_dim = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Identity()

        # ----------------------------------------------------
        # Freeze most layers
        # ----------------------------------------------------

        for param in self.backbone.parameters():
            param.requires_grad = False

        # ----------------------------------------------------
        # Unfreeze last EfficientNet blocks
        # This allows deepfake-specific learning
        # ----------------------------------------------------

        for param in self.backbone.features[-2:].parameters():
            param.requires_grad = True

        # ====================================================
        # Temporal GRU
        # ====================================================

        self.temporal_gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )

        # Bidirectional GRU => hidden_dim * 2
        gru_output_dim = hidden_dim * 2

        # ====================================================
        # Attention
        # ====================================================

        self.attention = nn.Sequential(
            nn.Linear(gru_output_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # ====================================================
        # Classifier
        # ====================================================

        self.classifier = nn.Sequential(

            nn.LayerNorm(gru_output_dim),

            nn.Linear(
                gru_output_dim,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        """
        Input:
            x = (batch, frames, 3, 160, 160)

        Output:
            logits = (batch, 2)

        Class mapping:
            0 = FAKE
            1 = REAL
        """

        batch_size, frames, channels, height, width = x.shape

        # ====================================================
        # Extract visual features
        # ====================================================

        x = x.reshape(
            batch_size * frames,
            channels,
            height,
            width
        )

        features = self.backbone(x)

        # (batch * frames, 1280)
        features = features.reshape(
            batch_size,
            frames,
            -1
        )

        # ====================================================
        # Temporal modelling
        # ====================================================

        temporal_output, _ = self.temporal_gru(features)

        # Shape:
        # (batch, frames, 512)
        #
        # because hidden_dim = 256
        # and bidirectional = True

        # ====================================================
        # Temporal attention
        # ====================================================

        attention_scores = self.attention(
            temporal_output
        )

        # (batch, frames, 1)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        # ====================================================
        # Weighted temporal representation
        # ====================================================

        video_features = torch.sum(
            temporal_output * attention_weights,
            dim=1
        )

        # ====================================================
        # Classification
        # ====================================================

        logits = self.classifier(
            video_features
        )

        return logits


def load_video_model(
    checkpoint_path=None,
    device="cpu"
):

    device = torch.device(device)

    print("Loading EfficientNet-B0 video model...")

    model = VideoDeepfakeModel()

    if checkpoint_path is not None:

        print("Loading checkpoint:")
        print(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False
        )

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint

        model.load_state_dict(
            state_dict,
            strict=True
        )

        print(
            "Checkpoint loaded successfully."
        )

    else:

        print(
            "No trained checkpoint supplied."
        )

    model.to(device)

    model.eval()

    return model


def calculate_fraud_score(logits):

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    fake_probability = float(
        probabilities[0, 0]
    )

    real_probability = float(
        probabilities[0, 1]
    )

    fraud_score = fake_probability * 100

    return {

        "real_probability":
            real_probability * 100,

        "fake_probability":
            fake_probability * 100,

        "fraud_score":
            fraud_score
    }