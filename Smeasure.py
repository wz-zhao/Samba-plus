"""Device-independent S-measure, including empty foreground/background cases."""
import torch


def _object(pred, gt):
    values = pred[gt.bool()]
    if values.numel() == 0:
        return pred.new_zeros(())
    sigma = values.std() if values.numel() > 1 else values.new_zeros(())
    mean = values.mean()
    return 2 * mean / (mean.square() + 1 + sigma + 1e-20)


def S_object(pred, gt):
    pred, gt = pred.squeeze(), gt.squeeze()
    fraction = gt.float().mean()
    return fraction * _object(pred, gt) + (1 - fraction) * _object(1 - pred, 1 - gt)


def _centroid(gt):
    gt = gt.reshape(gt.shape[-2:]).float()
    rows, cols = gt.shape
    total = gt.sum()
    if total == 0:
        return cols // 2, rows // 2
    x = (gt.sum(0) * torch.arange(cols, device=gt.device)).sum() / total
    y = (gt.sum(1) * torch.arange(rows, device=gt.device)).sum() / total
    return int(x.round().item()), int(y.round().item())


def _ssim(pred, gt):
    if pred.numel() == 0:
        return pred.new_zeros(())
    gt = gt.float()
    x, y = pred.mean(), gt.mean()
    denominator = max(pred.numel() - 1, 1)
    vx = (pred - x).square().sum() / denominator
    vy = (gt - y).square().sum() / denominator
    covariance = ((pred - x) * (gt - y)).sum() / denominator
    alpha = 4 * x * y * covariance
    beta = (x.square() + y.square()) * (vx + vy)
    return torch.where(beta == 0, torch.ones_like(beta), alpha / beta.clamp_min(1e-20))


def _dividePrediction(pred, X, Y):
    pred = pred.reshape(pred.shape[-2:])
    return pred[:Y, :X], pred[:Y, X:], pred[Y:, :X], pred[Y:, X:]


def _divideGT(gt, X, Y):
    h, w = gt.shape[-2:]
    pieces = _dividePrediction(gt, X, Y)
    weights = (X * Y / (h * w), (w - X) * Y / (h * w),
               X * (h - Y) / (h * w), (w - X) * (h - Y) / (h * w))
    return (*pieces, *weights)


def S_region(pred, gt):
    x, y = _centroid(gt)
    gt_parts = _divideGT(gt, x, y)
    pred_parts = _dividePrediction(pred, x, y)
    return sum(weight * _ssim(p, g) for p, g, weight in
               zip(pred_parts, gt_parts[:4], gt_parts[4:]))


def s_measure(pred, gt):
    pred = pred.reshape(pred.shape[-2:]).float()
    gt = (gt.reshape(gt.shape[-2:]) >= 0.5).float()
    mean = gt.mean()
    if mean == 0:
        return 1 - pred.mean()
    if mean == 1:
        return pred.mean()
    return (0.5 * S_object(pred, gt) + 0.5 * S_region(pred, gt)).clamp(0, 1)
