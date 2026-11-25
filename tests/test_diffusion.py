"""Known-answer tests for the sampled-noise target and fixed-index inference."""
import torch
from torch import nn

from tlsseg.models.components import FeatureDiffusion


def test_training_predicts_the_same_fresh_noise_at_sampled_strength(monkeypatch):
    diffusion = FeatureDiffusion().train()
    feature = torch.linspace(-1, 1, 2 * 32 * 4 * 4).reshape(2, 32, 4, 4)
    sampled_noise = feature.flip(-1) + 0.3
    times = torch.tensor([7, 20])
    a = diffusion.sqrt_alphas_cumprod[times].reshape(2, 1, 1, 1)
    b = diffusion.sqrt_one_minus_alphas_cumprod[times].reshape(2, 1, 1, 1)

    class ExactNoise(nn.Module):
        def forward(self, noisy, t):
            torch.testing.assert_close(t, times)
            torch.testing.assert_close(noisy, a * feature + b * sampled_noise)
            return sampled_noise

    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: times)
    monkeypatch.setattr(torch, "randn_like", lambda value: sampled_noise)
    diffusion.denoise_net = ExactNoise()
    reconstructed, mse = diffusion(feature)
    assert mse.item() == 0
    torch.testing.assert_close(reconstructed, feature, atol=2e-7, rtol=1e-6)


def test_inference_passes_clean_feature_once_at_zero_without_sampling(monkeypatch):
    diffusion = FeatureDiffusion().eval()
    feature = torch.ones(1, 32, 4, 4)
    calls = []

    class ZeroNoise(nn.Module):
        def forward(self, value, t):
            assert torch.equal(value, feature)
            assert t.eq(0).all()
            calls.append(1)
            return torch.zeros_like(value)

    def forbid_sampling(*args, **kwargs):
        raise AssertionError("Inference must not sample a noise tensor or a time index.")

    monkeypatch.setattr(torch, "randint", forbid_sampling)
    monkeypatch.setattr(torch, "randn_like", forbid_sampling)
    diffusion.denoise_net = ZeroNoise()
    reconstructed, placeholder = diffusion(feature)
    assert len(calls) == 1
    assert placeholder.item() == 0
    assert diffusion.sqrt_one_minus_alphas_cumprod[0].item() > 0
    torch.testing.assert_close(reconstructed, feature / (diffusion.sqrt_alphas_cumprod[0] + 1e-8))
