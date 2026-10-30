"""Small real-network checks; no datasets or downloaded checkpoints required."""
import torch


def test_real_model_all_stages(monkeypatch):
    monkeypatch.setenv('SAMBA_SCAN_BACKEND', 'torch')
    from models.SalMamba_tri import Model
    from train_macl_ddp import convert_bn_to_gn, forward_losses
    from utils.init_func import group_weight
    torch.set_num_threads(2)
    model = convert_bn_to_gn(Model(rq_probability=1)).train()
    batch = {'image': torch.randn(2, 3, 32, 32),
             'label': (torch.rand(2, 1, 32, 32) > .5).float()}
    for stage in [1, 2, 3]:
        if stage == 2:
            batch['depth'] = torch.randn_like(batch['image'])
        if stage == 3:
            batch['flow'] = torch.randn_like(batch['image'])
        losses = forward_losses(model, batch, torch.device('cpu'), stage)
        assert losses.shape == (2,) and torch.isfinite(losses).all()
        losses.mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.zero_grad(set_to_none=True)
    groups = group_weight([], model, torch.nn.BatchNorm2d, 1e-4)
    assert {id(p) for group in groups for p in group['params']} == {id(p) for p in model.parameters()}


def test_saliency_scan_includes_zero_features():
    from models.encoders.vmamba import CrossScan_saliency, CrossMerge_saliency
    x = torch.tensor([[[[0., 2.], [0., 4.]], [[1., 0.], [3., 0.]]]], requires_grad=True)
    for mask in [torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2),
                 torch.tensor([[[[0., 1.], [1., 0.]]]])]:
        scan, indices, background = CrossScan_saliency(x, mask)
        restored = CrossMerge_saliency(scan.reshape(1, 4, 2, 2, 2), indices, background, mask)
        torch.testing.assert_close(restored.reshape_as(x), 4 * x)
        restored.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 4))
        x.grad = None
