import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from ..net_utils import FeatureFusionModule as FFM
from ..net_utils import FeatureRectifyModule as FRM
import math
import time
from engine.logger import get_logger
from models.encoders.vmamba import Backbone_VSSM, SaliencyMambaBlock,ConcatMambaFusionBlock
from models.ERM import ERM_Contour_Integrity,GNNRoIFusion
logger = get_logger()


class RGBXMamba(nn.Module):
    def __init__(self, 
                 num_classes=1000,
                 norm_layer=nn.LayerNorm,
                 depths=[2,2,27,2], # [2,2,27,2] for vmamba small
                 dims=128,
                 pretrained=None,
                 mlp_ratio=0.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[448, 448],
                 patch_size=4,
                 drop_path_rate=0.6,


                 **kwargs):
        super().__init__()
        
        self.ape = ape

        self.vssm_r = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )
     


        dims_list = [0,192,384,768]

        self.erm = ERM_Contour_Integrity(in_channels=768)

        self.saliency_mamba = nn.ModuleList(
            SaliencyMambaBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
                lip = False
            ) for i in range(4)
        )

        aux_in_channels = 768
        self.aux_heads = nn.ModuleDict({
            'depth': nn.Conv2d(aux_in_channels, 1, kernel_size=1),
            'flow': nn.Conv2d(aux_in_channels, 1, kernel_size=1),
            'thermal': nn.Conv2d(aux_in_channels, 1, kernel_size=1)
        })
        
        self.modality_mamba = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            GNNRoIFusion(
                channels=192 * (2 ** 2),
                num_gnn_layers=3,
                heads=4,
                num_conv_layers=1
            )
        ])

        if self.ape:
            self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
            self.absolute_pos_embed = []
            self.absolute_pos_embed_x = []
            for i_layer in range(len(depths)):
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                      self.patches_resolution[1] // (2 ** i_layer))
                dim=int(dims * (2 ** i_layer))
                absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed, std=.02)
                absolute_pos_embed_x = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_x, std=.02)
                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)
        

    def forward_features(self, x_rgb, x_f = None, x_d = None,x_t = None,task = None):
        """
        x_rgb: B x C x H x W
        """
        B = x_rgb.shape[0]
        outs_fused = []
        
        x_rgb = torch.cat([x for x in (x_rgb, x_d, x_f, x_t) if x is not None],dim=0)


        outs_rgb = self.vssm_r(x_rgb)
       
        outs_temp = outs_rgb
        outs_rgb, outs_d, outs_f, outs_t = split_modal_outputs(
            outs_temp,
            x_d=x_d,
            x_f=x_f,
            x_t=x_t)
        
        
        B, C, H, W = outs_rgb[3].shape
        aux_preds = {}
        feature_stage_idx = 3

        #if x_d is not None: 
        #     aux_preds['depth'] = self.aux_heads['depth'](F.interpolate(outs_d[3], size=(448,448), mode='bilinear', align_corners=False))
        #if x_f is not None: 
        #     aux_preds['flow'] = self.aux_heads['flow'](F.interpolate(outs_f[3], size=(448,448), mode='bilinear', align_corners=False)) 
        # if x_t is not None:
        #     aux_preds['thermal'] = self.aux_heads['thermal'](F.interpolate(outs_t[3], size=(448,448), mode='bilinear', align_corners=False))

        id = 3
        out_rgb = outs_rgb[id]
        B, C, H, W = out_rgb.shape

        inputs = [out_rgb]

        if x_d is not None:
            inputs.append(outs_d[id])
        if x_f is not None:
            inputs.append(outs_f[id])
        if x_t is not None:
            inputs.append(outs_t[id])

        out_rgb = self.modality_mamba[id - 1](inputs)
        supervised_mask, final_refined_feature =  self.erm(out_rgb)

        guide_saliency = supervised_mask
        out_rgb = final_refined_feature.permute(0, 2, 3, 1).contiguous()
        B,H,W,C = out_rgb.shape
        resized_gt = F.interpolate(guide_saliency, size=(H, W), mode='bilinear', align_corners=False)
        resized_gt  = (resized_gt >= 0.3).float()
        final_refined_feature = self.saliency_mamba[id](out_rgb, resized_gt)
        final_refined_feature = final_refined_feature.permute(0, 3, 1, 2).contiguous()
            
        for i in range(3):
            out_rgb = outs_rgb[i].contiguous()
            B,C ,H,W = out_rgb.shape
            resized_gt = F.interpolate(guide_saliency, size=(H, W), mode='bilinear', align_corners=False)
            resized_gt  = (resized_gt >= 0.3).float()
            out_rgb = out_rgb.permute(0, 2, 3, 1).contiguous()
            out_rgb = self.saliency_mamba[i](out_rgb, resized_gt)
            out_rgb = out_rgb.permute(0, 3, 1, 2).contiguous()
            outs_fused.append(out_rgb)   
  
        outs_fused.append(final_refined_feature)  

        return outs_fused, guide_saliency, aux_preds

    def forward(self, x_rgb, x_f, x_d, x_t,task):
        out, saliency,aux_preds = self.forward_features(x_rgb, x_f, x_d, x_t, task)
        return out, saliency,aux_preds

class vssm_tiny(RGBXMamba):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_tiny, self).__init__(
            depths=[2, 2, 9, 2], 
            dims=96,
            pretrained='pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
        )

class vssm_small(RGBXMamba):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_small, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained='vssmsmall_dp03_ckpt_epoch_238.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
        )

class vssm_base(RGBXMamba):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_base, self).__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            pretrained='models/pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.6, # VMamba-B with droppath 0.5 + no ema. VMamba-B* represents for VMamba-B with droppath 0.6 + ema
        )




def split_modal_outputs(outs_temp, x_d=None, x_f=None, x_t=None):
    total_modal = 1 
    if x_d is not None: total_modal += 1
    if x_f is not None: total_modal += 1
    if x_t is not None: total_modal += 1

    B = outs_temp[0].shape[0]
    per_modal = B // total_modal
    assert B % total_modal == 0, f"Batch size {B} must be divisible by number of modalities {total_modal}"

    modal_idx = {}
    start = 0
    modal_idx['rgb'] = (start, start + per_modal)
    start += per_modal
    if x_d is not None:
        modal_idx['d'] = (start, start + per_modal)
        start += per_modal
    if x_f is not None:
        modal_idx['f'] = (start, start + per_modal)
        start += per_modal
    if x_t is not None:
        modal_idx['t'] = (start, start + per_modal)
        start += per_modal


    outs_rgb = [feat[modal_idx['rgb'][0]:modal_idx['rgb'][1]] for feat in outs_temp]
    outs_d = [feat[modal_idx['d'][0]:modal_idx['d'][1]] for feat in outs_temp] if x_d is not None else None
    outs_f = [feat[modal_idx['f'][0]:modal_idx['f'][1]] for feat in outs_temp] if x_f is not None else None
    outs_t = [feat[modal_idx['t'][0]:modal_idx['t'][1]] for feat in outs_temp] if x_t is not None else None

    return outs_rgb, outs_d, outs_f, outs_t
