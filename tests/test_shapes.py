"""
Shape tests — verify all tensor shapes through the forward pass.

These tests run on CPU with random weights (no pretrained weights needed)
to validate the architecture is wired correctly.
"""
import time

import pytest
import torch
import torch.nn as nn

from world_model.config import ModelConfig
from world_model.pos_encoding import SphericalSinusoidal, downsample_pos_map
from world_model.action_mlp import ActionMLP
from world_model.transformer import TransformerBottleneck
from world_model.unet import ThinUNet


# ── Helpers ──────────────────────────────────────────────────────────

def _make_config() -> ModelConfig:
    return ModelConfig()


# ── Positional Encoding ─────────────────────────────────────────────

class TestPosEncoding:

    def test_shape(self):
        B, H, W, d_model = 2, 32, 32, 512
        pos_map = torch.randn(B, H, W, 2)
        enc = SphericalSinusoidal(d_model=d_model)
        out = enc(pos_map)
        assert out.shape == (B, H * W, d_model)
        assert out.dtype == torch.float32

    def test_different_spatial_size(self):
        B, H, W, d_model = 1, 16, 16, 256
        pos_map = torch.randn(B, H, W, 2)
        enc = SphericalSinusoidal(d_model=d_model)
        out = enc(pos_map)
        assert out.shape == (B, 256, d_model)

    def test_deterministic(self):
        """Same input → same output."""
        pos_map = torch.randn(1, 32, 32, 2)
        enc = SphericalSinusoidal(d_model=512)
        a = enc(pos_map)
        b = enc(pos_map)
        assert torch.allclose(a, b)

    def test_downsample_pos_map(self):
        pos_map = torch.randn(2, 256, 256, 2)
        small = downsample_pos_map(pos_map, size=32)
        assert small.shape == (2, 32, 32, 2)


# ── Action MLP ───────────────────────────────────────────────────────

class TestActionMLP:

    def test_shape(self):
        d_model, n_layers = 512, 2
        mlp = ActionMLP(d_model=d_model, n_layers=n_layers)
        action = torch.randn(4, 3)
        params = mlp(action)

        assert len(params) == n_layers
        for scale, shift in params:
            assert scale.shape == (4, d_model)
            assert shift.shape == (4, d_model)

    def test_zero_init(self):
        """Final layer should be zero-initialized → zero output at init."""
        mlp = ActionMLP(d_model=512, n_layers=2)
        action = torch.zeros(1, 3)
        params = mlp(action)

        for scale, shift in params:
            assert torch.allclose(scale, torch.zeros_like(scale), atol=1e-6)
            assert torch.allclose(shift, torch.zeros_like(shift), atol=1e-6)


# ── Transformer Bottleneck ───────────────────────────────────────────

class TestTransformer:

    def test_shape(self):
        B, H, W = 2, 32, 32
        warp_ch, anchor_ch = 512, 4
        d_model, n_layers = 512, 2

        transformer = TransformerBottleneck(
            warp_channels=warp_ch, anchor_channels=anchor_ch,
            d_model=d_model, n_heads=4, n_layers=n_layers,
        )
        z_warp = torch.randn(B, warp_ch, H, W)
        z_anchor = torch.randn(B, anchor_ch, H, W)
        pos_enc = torch.randn(B, H * W, d_model)
        adaln_params = [(torch.randn(B, d_model), torch.randn(B, d_model))
                        for _ in range(n_layers)]

        out = transformer(z_warp, z_anchor, pos_enc, adaln_params)
        assert out.shape == (B, warp_ch, H, W)

    def test_gradient_flows(self):
        """Ensure gradients flow through the transformer."""
        transformer = TransformerBottleneck(warp_channels=128, anchor_channels=4,
                                            d_model=128, n_heads=4, n_layers=1)
        z_warp = torch.randn(1, 128, 8, 8, requires_grad=True)
        z_anchor = torch.randn(1, 4, 8, 8)
        pos_enc = torch.randn(1, 64, 128)
        adaln_params = [(torch.randn(1, 128), torch.randn(1, 128))]

        out = transformer(z_warp, z_anchor, pos_enc, adaln_params)
        out.sum().backward()
        assert z_warp.grad is not None
        assert z_warp.grad.abs().sum() > 0


# ── UNet ─────────────────────────────────────────────────────────────

class TestUNet:

    def test_shape(self):
        B, C, H, W = 2, 4, 32, 32
        config = _make_config()
        unet = ThinUNet(
            latent_channels=C, d_model=config.d_model,
            n_heads=config.n_heads, n_layers=config.n_layers,
        )
        z_warp = torch.randn(B, C, H, W)
        z_anchor = torch.randn(B, C, H, W)
        pos_enc = torch.randn(B, H * W, config.d_model)
        adaln_params = [(torch.randn(B, config.d_model), torch.randn(B, config.d_model))
                        for _ in range(config.n_layers)]

        out = unet(z_warp, z_anchor, pos_enc, adaln_params)
        assert out.shape == (B, C, H, W)

    def test_residual_learning(self):
        """With zero-init weights, output should approximate input (residual = 0)."""
        unet = ThinUNet(latent_channels=4, d_model=128, n_heads=4, n_layers=1)

        # Zero out the final conv so residual path is zero
        nn.init.zeros_(unet.dec_conv_out.weight)
        nn.init.zeros_(unet.dec_conv_out.bias)

        z = torch.randn(1, 4, 32, 32)
        pos = torch.randn(1, 1024, 128)
        adaln = [(torch.zeros(1, 128), torch.zeros(1, 128))]
        out = unet(z, z, pos, adaln)
        # With zeroed output conv, output should be exactly z (the skip)
        assert torch.allclose(out, z, atol=1e-5)


# ── Full Model (without VAE — mock it) ──────────────────────────────

class _MockVAE(nn.Module):
    """Deterministic mock VAE for CPU shape tests."""
    def __init__(self):
        super().__init__()
        self._loaded = True
        self.vae = None

    def encode(self, x):
        B, C, H, W = x.shape
        return x[:, :4, :H//8, :W//8] if C >= 4 else torch.randn(B, 4, H//8, W//8)

    def decode(self, z):
        B, C, H, W = z.shape
        return torch.randn(B, 3, H*8, W*8).clamp(-1, 1)

    def to(self, *args, **kwargs):
        return self


class TestFullModel:

    def _make_model(self):
        from world_model.model import InfiniteRaceWorldModel
        config = ModelConfig()
        model = InfiniteRaceWorldModel(config)
        # Replace real VAE with mock for CPU testing
        model.vae = _MockVAE()
        return model

    def test_forward_shape(self):
        model = self._make_model()
        B = 2
        cue1 = torch.randn(B, 3, 256, 256)
        cue2 = torch.randn(B, 3, 256, 256)
        pos_map = torch.randn(B, 256, 256, 2)
        action = torch.randn(B, 3).clamp(-1, 1)

        out = model(cue1, cue2, pos_map, action)
        assert out.shape == (B, 3, 256, 256)
        assert out.dtype == torch.float32

    def test_output_range(self):
        model = self._make_model()
        cue1 = torch.randn(1, 3, 256, 256)
        cue2 = torch.randn(1, 3, 256, 256)
        pos_map = torch.randn(1, 256, 256, 2)
        action = torch.randn(1, 3).clamp(-1, 1)

        out = model(cue1, cue2, pos_map, action)
        # Mock VAE clamps to [-1, 1]
        assert out.min() >= -1.05
        assert out.max() <= 1.05

    def test_noise_level(self):
        """Forward with noise_level > 0 should still produce valid shapes."""
        model = self._make_model()
        cue1 = torch.randn(1, 3, 256, 256)
        cue2 = torch.randn(1, 3, 256, 256)
        pos_map = torch.randn(1, 256, 256, 2)
        action = torch.randn(1, 3).clamp(-1, 1)

        out = model(cue1, cue2, pos_map, action, noise_level=0.5)
        assert out.shape == (1, 3, 256, 256)

    def test_save_load(self, tmp_path):
        model = self._make_model()
        path = str(tmp_path / "test_ckpt.pt")
        model.save(path)

        loaded = type(model).load(path)
        loaded.vae = _MockVAE()

        cue1 = torch.randn(1, 3, 256, 256)
        cue2 = torch.randn(1, 3, 256, 256)
        pos_map = torch.randn(1, 256, 256, 2)
        action = torch.randn(1, 3).clamp(-1, 1)
        out = loaded(cue1, cue2, pos_map, action)
        assert out.shape == (1, 3, 256, 256)


# ── Inference Speed (GPU only) ──────────────────────────────────────

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available — skipping GPU inference speed test",
)
class TestInferenceSpeed:

    def test_inference_under_100ms(self):
        """LCM 2-step inference should complete under 100ms on GPU."""
        from world_model.model import InfiniteRaceWorldModel

        model = InfiniteRaceWorldModel(ModelConfig()).cuda().half().eval()
        # Replace VAE with mock (real VAE requires download)
        model.vae = _MockVAE().cuda()

        cue1 = torch.randn(1, 3, 256, 256, device='cuda', dtype=torch.float16)
        cue2 = torch.randn(1, 3, 256, 256, device='cuda', dtype=torch.float16)
        pos_map = torch.randn(1, 256, 256, 2, device='cuda', dtype=torch.float16)
        action = torch.randn(1, 3, device='cuda', dtype=torch.float16)

        # Warmup
        for _ in range(3):
            model.infer(cue1, cue2, pos_map, action, steps=2)

        # Time
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            model.infer(cue1, cue2, pos_map, action, steps=2)
        torch.cuda.synchronize()
        ms_per_frame = (time.perf_counter() - t0) / 10 * 1000

        assert ms_per_frame < 100, f"Inference too slow: {ms_per_frame:.1f}ms (target <100ms)"
