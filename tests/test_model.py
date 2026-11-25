"""Behavioral checks for the encoder-four-scale final-gate architecture.

Synthetic inputs establish implementation contracts, not segmentation accuracy
or equivalence to historical experiments on different whole-model variants.
"""
from __future__ import annotations

import io

import pytest
import torch
from torch.nn import functional as F

from tlsseg.models import ARCHITECTURE_ID, PatentSegmenter


@pytest.fixture(autouse=True)
def single_threaded_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(with_dit=True):
    torch.manual_seed(42)
    return PatentSegmenter(with_dit=with_dit)


@pytest.mark.parametrize("with_dit", [False, True])
def test_zero_gate_is_exact_native_unet_and_uses_encoder_four_scales(with_dit):
    model = make_model(with_dit).eval()
    torch.manual_seed(12)
    images = torch.randn(1, 3, 224, 224)
    observed = []

    def capture_inputs(_module, args):
        observed.append([tuple(feature.shape[1:]) for feature in args[0]])

    hook = model.tls_fusion.register_forward_pre_hook(capture_inputs)
    try:
        with torch.inference_mode():
            native = model.native_logits(images)
            result = model(images)
    finally:
        hook.remove()
    assert observed == [[(32, 56, 56), (48, 28, 28), (136, 14, 14), (384, 7, 7)]]
    assert tuple(model.decoder_gate.shape) == (1, 16, 1, 1)
    assert model.decoder_gate.eq(0).all()
    assert torch.equal(result["logits"], native)
    assert result["noise_mse"].shape == ()
    assert result["noise_mse"].item() == 0.0
    assert result["logits"].shape == (1, 4, 224, 224)
    assert model.architecture_config()["architecture_id"] == ARCHITECTURE_ID


def test_nonzero_final_gate_changes_logits_and_eval_is_deterministic():
    model = make_model().eval()
    torch.manual_seed(13)
    images = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        native = model.native_logits(images)
        model.decoder_gate.fill_(0.25)
        first = model(images)
        second = model(images)
    assert torch.isfinite(first["logits"]).all()
    assert (first["logits"] - native).abs().max().item() > 1e-7
    assert torch.equal(first["logits"], second["logits"])
    assert first["noise_mse"].item() == 0.0


def test_two_training_steps_open_gate_and_update_fusion_and_dit():
    model = make_model().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    torch.manual_seed(14)
    images = torch.randn(1, 3, 224, 224)
    labels = torch.randint(0, 4, (1, 224, 224))
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = model(images)
        assert torch.isfinite(result["noise_mse"])
        assert result["noise_mse"].item() > 0.0
        loss = F.cross_entropy(result["logits"], labels) + 0.01 * result["noise_mse"]
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.decoder_gate.grad is not None
        assert model.decoder_gate.grad.abs().sum().item() > 0.0
        optimizer.step()
    assert model.decoder_gate.detach().abs().sum().item() > 0.0
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0.0
               for p in model.tls_fusion.stage_adapters.parameters())
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0.0
               for p in model.feature_diffusion.denoise_net.dit_noise_head.unpatch.parameters())
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0.0
               for p in model.network.segmentation_head.parameters())


def test_checkpoint_roundtrip_preserves_nonzero_gate_predictions():
    model = make_model().eval()
    torch.manual_seed(15)
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        model.decoder_gate.fill_(0.15)
    with torch.inference_mode():
        expected = model(images)["logits"]
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = make_model().eval()
    restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True), strict=True)
    with torch.inference_mode():
        actual = restored(images)["logits"]
    assert torch.equal(actual, expected)
    assert restored.architecture_config() == model.architecture_config()


def test_rejects_incompatible_input_geometry():
    model = make_model(with_dit=False)
    with pytest.raises(ValueError, match="224"):
        model(torch.randn(1, 3, 112, 112))
    with pytest.raises(ValueError, match="floating"):
        model(torch.zeros(1, 3, 224, 224, dtype=torch.uint8))
