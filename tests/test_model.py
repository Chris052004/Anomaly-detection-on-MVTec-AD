import torch

from model_classes.autoencoder_model import ConvAutoencoder


def test_forward_pass_preserves_shape():
    model = ConvAutoencoder(latent_dim=100)
    x = torch.rand(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 3, 256, 256)


def test_forward_pass_output_in_unit_range():
    model = ConvAutoencoder(latent_dim=100)
    x = torch.rand(2, 3, 256, 256)
    out = model(x)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_different_latent_dim_still_produces_correct_shape():
    model = ConvAutoencoder(latent_dim=32)
    x = torch.rand(1, 3, 256, 256)
    out = model(x)
    assert out.shape == (1, 3, 256, 256)
