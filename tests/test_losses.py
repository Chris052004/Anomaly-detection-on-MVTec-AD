import torch
import torch.nn.functional as F

from anomaly_ae.losses import anomaly_map, combined_loss, ssim_map


def test_ssim_map_shape():
    a = torch.rand(2, 3, 64, 64)
    b = torch.rand(2, 3, 64, 64)
    result = ssim_map(a, b)
    assert result.shape == (2, 1, 64, 64)


def test_ssim_map_of_identical_images_is_near_one():
    img = torch.rand(1, 3, 64, 64)
    result = ssim_map(img, img)
    assert result.mean().item() > 0.99


def test_anomaly_map_of_identical_images_is_near_zero():
    img = torch.rand(1, 3, 64, 64)
    result = anomaly_map(img, img)
    assert result.shape == (1, 1, 64, 64)
    assert result.mean().item() < 0.01


def test_combined_loss_mse_only_matches_plain_mse():
    a = torch.rand(2, 3, 64, 64)
    b = torch.rand(2, 3, 64, 64)
    loss = combined_loss(a, b, mode="mse_only", ssim_weight=0.85, mse_weight=0.15)
    expected = F.mse_loss(a, b)
    assert torch.isclose(loss, expected)


def test_combined_loss_ssim_mse_of_identical_images_is_near_zero():
    img = torch.rand(1, 3, 64, 64)
    loss = combined_loss(img, img, mode="ssim_mse", ssim_weight=0.85, mse_weight=0.15)
    assert loss.item() < 0.01


def test_combined_loss_unknown_mode_raises():
    a = torch.rand(1, 3, 64, 64)
    try:
        combined_loss(a, a, mode="not_a_mode", ssim_weight=0.85, mse_weight=0.15)
        assert False, "expected ValueError"
    except ValueError:
        pass
