"""Encoder-four-scale fusion with one final-decoder residual injection.

This is the patent-aligned integration, not the historical decoder-mixed model
or the encoder-skip-injection model. Its trained performance is not established
by copying either earlier architecture's components or checkpoint metadata.
"""
from __future__ import annotations

import inspect

import segmentation_models_pytorch as smp
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .components import FeatureDiffusion, GlobalFeatureUpsample, PyrmidFusionNet, conv_block


ARCHITECTURE_ID = "efficientnet_b3_encoder4_final_decoder_channel_gate_v1"
RAW_ENCODER_CHANNELS = (32, 48, 136, 384)
ADAPTED_CHANNELS = (64, 128, 256, 512)
ENCODER_SPATIAL_SIZES = ((56, 56), (28, 28), (14, 14), (7, 7))


class TLSFusionBranch(nn.Module):
    """Original fusion mathematics, supplied with four encoder feature maps.

    Layer construction and forward ordering follow ``TLSFusionBranch`` from
    the supplied ``frozen_fresh_model.py``. Diffusion is applied separately by
    ``PatentSegmenter`` to this branch's final 32-channel feature map.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stage_adapters = nn.ModuleList([
            conv_block(native, adapted, 1, 1, 0, bn_act=True)
            for native, adapted in zip(RAW_ENCODER_CHANNELS, ADAPTED_CHANNELS)
        ])
        self.cfgb = nn.Sequential(
            conv_block(512, 512, 3, 2, 1, bn_act=True), nn.Dropout(0.15)
        )
        self.apf1 = PyrmidFusionNet(512, 512, 256, 32, classes=4)
        self.apf2 = PyrmidFusionNet(256, 256, 128, 16, classes=4)
        self.apf3 = PyrmidFusionNet(128, 128, 64, 8, classes=4)
        self.apf4 = PyrmidFusionNet(64, 64, 32, 4, classes=4)
        self.fab = nn.Sequential(
            conv_block(512, 256, 3, 1, 1, group=256, bn_act=True), nn.Dropout(0.15)
        )
        self.gfu4 = GlobalFeatureUpsample(256, 256, 128)
        self.gfu3 = GlobalFeatureUpsample(128, 128, 64)
        self.gfu2 = GlobalFeatureUpsample(64, 64, 32)
        self.gfu1 = GlobalFeatureUpsample(32, 32, 32)

    def forward(self, stages: list[Tensor] | tuple[Tensor, ...]) -> Tensor:
        if len(stages) != 4:
            raise ValueError("Fusion requires exactly four encoder feature maps.")
        adapted = [adapter(feature) for adapter, feature in zip(self.stage_adapters, stages)]
        x2, x3, x4, x5 = adapted
        a1 = self.apf1(self.cfgb(x5), x5)
        a2 = self.apf2(a1, x4)
        a3 = self.apf3(a2, x3)
        a4 = self.apf4(a3, x2)
        d5 = self.gfu4(a1, self.fab(x5))
        d4 = self.gfu3(a2, d5)
        d3 = self.gfu2(a3, d4)
        return self.gfu1(a4, d3)


class PatentSegmenter(nn.Module):
    """Four-class, 224-pixel encoder-fusion segmentation model.

    Encoder weights are initialized without downloading pretrained parameters.
    ``forward`` returns logits, not Softmax probabilities, and a scalar noise
    MSE. In evaluation mode the original FeatureDiffusion implementation uses
    the clean fused feature, fixed index zero and no newly sampled Gaussian
    noise; its zero noise-MSE return value is a placeholder, not a measured
    reconstruction error. Setting ``with_dit=False`` is a fusion-only ablation.
    """

    architecture_id = ARCHITECTURE_ID

    def __init__(self, with_dit: bool = True, timesteps: int = 50) -> None:
        super().__init__()
        if not isinstance(with_dit, bool):
            raise TypeError("with_dit must be a bool.")
        if isinstance(timesteps, bool) or not isinstance(timesteps, int) or timesteps < 1:
            raise ValueError("timesteps must be a positive integer.")
        if smp.__version__ != "0.5.0":
            raise RuntimeError("This architecture requires segmentation-models-pytorch==0.5.0.")
        self.with_dit = with_dit
        self.timesteps = timesteps
        self.network = smp.Unet(
            encoder_name="efficientnet-b3", encoder_weights=None,
            in_channels=3, classes=4,
        )
        parameters = list(inspect.signature(self.network.decoder.forward).parameters.values())
        if len(parameters) != 1 or parameters[0].kind == inspect.Parameter.VAR_POSITIONAL:
            raise RuntimeError("Expected the SMP 0.5 feature-list decoder API.")
        if tuple(self.network.encoder.out_channels) != (3, 40, *RAW_ENCODER_CHANNELS):
            raise RuntimeError("EfficientNet-B3 encoder channel contract differs.")
        self.tls_fusion = TLSFusionBranch()
        self.refinement_projection = nn.Conv2d(32, 16, 1, bias=False)
        self.decoder_gate = nn.Parameter(torch.zeros(1, 16, 1, 1))
        # Keep shared initialization order identical for no-DiT/DiT variants.
        self.feature_diffusion = (
            FeatureDiffusion(channels=32, timesteps=timesteps) if with_dit else None
        )

    @staticmethod
    def _validate_images(images: Tensor) -> None:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
            raise ValueError("Expected normalized RGB images with shape [B,3,224,224].")
        if not images.is_floating_point():
            raise ValueError("Images must be floating-point tensors.")

    def native_logits(self, images: Tensor) -> Tensor:
        """Run the underlying original U-Net; caller controls autograd/mode."""
        self._validate_images(images)
        return self.network(images)

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        self._validate_images(images)
        features = list(self.network.encoder(images))
        stages = features[2:6]
        observed = tuple((f.shape[1], tuple(f.shape[-2:])) for f in stages)
        expected = tuple(zip(RAW_ENCODER_CHANNELS, ENCODER_SPATIAL_SIZES))
        if observed != expected:
            raise RuntimeError(f"Encoder-four-scale feature contract differs: {observed}.")
        # The native decoder executes once. No decoder block is captured, no
        # encoder feature is modified, and no extra encoder-skip gate is used.
        native_final = self.network.decoder(features)
        if tuple(native_final.shape[1:]) != (16, 224, 224):
            raise RuntimeError("Expected a native final decoder feature [B,16,224,224].")
        d2 = self.tls_fusion(stages)
        if tuple(d2.shape[1:]) != (32, 56, 56):
            raise RuntimeError("Expected a fused feature [B,32,56,56].")
        if self.feature_diffusion is not None:
            reconstructed, noise_mse = self.feature_diffusion(d2)
        else:
            reconstructed, noise_mse = d2, d2.new_zeros(())
        projected = self.refinement_projection(reconstructed)
        residual = F.interpolate(
            projected, size=native_final.shape[-2:], mode="bilinear", align_corners=False
        )
        refined_final = native_final + torch.tanh(self.decoder_gate) * residual
        logits = self.network.segmentation_head(refined_final)
        if tuple(logits.shape) != (images.shape[0], 4, 224, 224) or noise_mse.ndim != 0:
            raise RuntimeError("Segmentation output contract differs.")
        return {"logits": logits, "noise_mse": noise_mse}

    def architecture_config(self) -> dict:
        """Return the explicit architecture contract for checkpoint metadata."""
        return {
            "architecture_id": ARCHITECTURE_ID,
            "encoder": "efficientnet-b3",
            "segmentation_models_pytorch": smp.__version__,
            "encoder_weights": None,
            "input_shape": [3, 224, 224],
            "classes": 4,
            "feature_source": "encoder_only",
            "encoder_stage_indices": [2, 3, 4, 5],
            "raw_encoder_channels": list(RAW_ENCODER_CHANNELS),
            "encoder_spatial_sizes": [list(size) for size in ENCODER_SPATIAL_SIZES],
            "adapted_channels": list(ADAPTED_CHANNELS),
            "fusion_output_shape": [32, 56, 56],
            "with_dit": self.with_dit,
            "timesteps": self.timesteps,
            "beta_schedule": {"kind": "linear", "start": 0.0001, "end": 0.02},
            "eval_diffusion": "fixed index 0; no added Gaussian noise; one reconstruction",
            "projection": {"kernel_size": 1, "in_channels": 32, "out_channels": 16},
            "interpolation": {"mode": "bilinear", "align_corners": False},
            "injection": "native final decoder feature before the original segmentation head",
            "native_final_shape": [16, 224, 224],
            "gate": {"shape": [1, 16, 1, 1], "activation": "tanh", "initial_value": 0.0},
            "output_shape": [4, 224, 224],
        }
