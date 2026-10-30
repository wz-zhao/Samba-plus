import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.tri_vmamba import vssm_small as backbone
from .decoders.MambaDecoder import MambaDecoder


class Model(nn.Module):
    def __init__(self, backbone_pretrained=None, rq_probability=0.1,
                 rq_salient_max_delta=0.05):
        super().__init__()
        self.rq_probability = rq_probability
        self.rq_module_rgb = RandomModule(modality='RGB', fine_grained_region_num=24)
        self.rq_module_depth = RandomModule(modality='Depth', fine_grained_region_num=20,
                                           spacing='log', salient_max_delta=rq_salient_max_delta)
        self.rq_module_flow = RandomModule(modality='Flow', fine_grained_region_num=16,
                                          salient_max_delta=rq_salient_max_delta)
        self.rq_module_t = RandomModule(modality='Thermal', fine_grained_region_num=24,
                                       spacing='log', salient_max_delta=rq_salient_max_delta)
        self.backbone = backbone(pretrained=backbone_pretrained)
        self.channels = [96, 192, 384, 768]
        self.decoder = MambaDecoder(img_size=[352, 352], in_channels=self.channels,
                                    num_classes=1, depths=[4, 4, 4, 4],
                                    embed_dim=96, deep_supervision=False)

    def forward(self, rgb, modal_flow=None, modal_depth=None, modal_thermal=None,
                mode=None, gt=None, task=None):
        # Quantize the tensors actually passed to the shared encoder. Keep RGB
        # intact and bound non-RGB perturbations using the saliency mask.
        if self.training and mode == 'train':
            if gt is None and self.rq_probability > 0:
                raise ValueError('Training RQ requires the saliency ground-truth mask')
            inputs = [modal_flow, modal_depth, modal_thermal]
            modules = [self.rq_module_flow, self.rq_module_depth, self.rq_module_t]
            for index, (value, module) in enumerate(zip(inputs, modules)):
                if value is not None and torch.rand((), device=value.device) < self.rq_probability:
                    inputs[index] = module(value, gt_mask=gt)
            modal_flow, modal_depth, modal_thermal = inputs
        features, saliency, aux_preds = self.backbone(
            rgb, modal_flow, modal_depth, modal_thermal, task=task)
        out = self.decoder(features)
        out = F.interpolate(out, size=rgb.shape[-2:], mode='bilinear', align_corners=False)
        saliency = F.interpolate(saliency, size=rgb.shape[-2:], mode='bilinear', align_corners=False)
        return out, saliency, aux_preds

    def load_pretrain_model(self, model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        return self.load_state_dict(state, strict=False)


class RandomModule(nn.Module):
    """Randomized quantization with an explicit foreground perturbation bound.

    salient_max_delta is a fraction of each sample/channel's dynamic range.
    Log spacing is applied to a shifted positive domain, so standardized depth
    and thermal values may be negative without invalid interval boundaries.
    """
    def __init__(self, region_num=None, fine_grained_region_num=16,
                 coarse_grained_region_num=6, collapse_to_val='inside_random',
                 spacing='random', transforms_like=False,
                 p_random_apply_rand_quant=1, modality='RGB', salient_max_delta=0.05):
        super().__init__()
        self.fine = region_num or fine_grained_region_num
        self.coarse = region_num or coarse_grained_region_num
        self.collapse_to_val = collapse_to_val
        self.spacing = spacing
        self.transforms_like = transforms_like
        self.probability = p_random_apply_rand_quant
        self.modality = modality
        self.salient_max_delta = salient_max_delta
        if min(self.fine, self.coarse) < 1 or not 0 <= salient_max_delta <= 1:
            raise ValueError('Invalid RQ bins or salient perturbation bound')

    def _quantize(self, value, bins):
        lo, hi = value.min(), value.max()
        span = hi - lo
        if span <= torch.finfo(value.dtype).eps:
            return value.clone()
        if self.spacing == 'random':
            cuts = torch.rand(bins - 1, device=value.device).sort().values
        elif self.spacing == 'uniform':
            cuts = torch.linspace(0, 1, bins + 1, device=value.device)[1:-1]
        elif self.spacing == 'log':
            cuts = torch.expm1(torch.linspace(0, 1, bins + 1, device=value.device)[1:-1]
                              * torch.log(value.new_tensor(10.0))) / 9
        else:
            raise ValueError(f'Unknown RQ spacing: {self.spacing}')
        boundaries = torch.cat([value.new_zeros(1), cuts, value.new_ones(1)])
        ids = torch.bucketize(((value - lo) / span).contiguous(), cuts)
        if self.collapse_to_val == 'middle':
            ratio = value.new_full((bins,), 0.5)
        elif self.collapse_to_val == 'inside_random':
            ratio = torch.rand(bins, device=value.device)
        elif self.collapse_to_val == 'all_zeros':
            return torch.zeros_like(value)
        else:
            raise ValueError(f'Unknown RQ representative: {self.collapse_to_val}')
        representatives = boundaries[:-1] + ratio * (boundaries[1:] - boundaries[:-1])
        return lo + span * representatives[ids]

    def forward(self, x, gt_mask=None):
        squeeze = x.ndim == 3
        values = x.unsqueeze(0) if squeeze else x
        if gt_mask is None:
            foreground = torch.zeros_like(values[:, :1], dtype=torch.bool)
        else:
            mask = gt_mask.detach().reshape(-1, 1, *gt_mask.shape[-2:]).to(values.device)
            foreground = F.interpolate(mask.float(), size=values.shape[-2:], mode='nearest') >= 0.5
            if foreground.shape[0] == 1:
                foreground = foreground.expand(values.shape[0], -1, -1, -1)
        output = values.clone()
        for batch_index in range(values.shape[0]):
            if torch.rand((), device=x.device) >= self.probability:
                continue
            for channel in range(values.shape[1]):
                original = values[batch_index, channel]
                fine = self._quantize(original, self.fine)
                coarse = self._quantize(original, self.coarse)
                bound = (original.max() - original.min()) * self.salient_max_delta
                fine = original + (fine - original).clamp(-bound, bound)
                output[batch_index, channel] = torch.where(foreground[batch_index, 0], fine, coarse)
        return output.squeeze(0) if squeeze else output
