import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Optional


def deform_conv2d_pytorch_grouped(
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    mask: Optional[Tensor] = None,
    groups: int = 1
) -> Tensor:

    bs, _, in_h, in_w = input.shape
    out_channels, in_channels_per_group, kh, kw = weight.shape
    
    out_h = (in_h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1


    grid_y, grid_x = torch.meshgrid(
        torch.arange(out_h, device=input.device, dtype=input.dtype),
        torch.arange(out_w, device=input.device, dtype=input.dtype),
        indexing='ij'
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)
    base_grid[..., 0] = base_grid[..., 0] * stride[1] - padding[1]
    base_grid[..., 1] = base_grid[..., 1] * stride[0] - padding[0]


    kernel_offset_y, kernel_offset_x = torch.meshgrid(
        torch.arange(-(kh - 1) // 2, (kh + 1) // 2, device=input.device, dtype=input.dtype),
        torch.arange(-(kw - 1) // 2, (kw + 1) // 2, device=input.device, dtype=input.dtype),
        indexing='ij'
    )
    kernel_offset_x = kernel_offset_x * dilation[1]
    kernel_offset_y = kernel_offset_y * dilation[0]
    kernel_offset = torch.stack([kernel_offset_x, kernel_offset_y], dim=-1).view(1, kh * kw, 1, 1, 2)


    input_groups = torch.chunk(input, groups, dim=1)
    weight_groups = torch.chunk(weight, groups, dim=0)
    

    offset_groups = torch.chunk(offset, groups, dim=1)

    mask_groups = torch.chunk(mask, groups, dim=1)

    output_groups = []
    for i in range(groups):

        group_offset = offset_groups[i].view(bs, 2, kh * kw, out_h, out_w).permute(0, 2, 3, 4, 1)
        sampling_grid = base_grid + kernel_offset + group_offset


        sampling_grid_x = 2.0 * sampling_grid[..., 0] / (in_w - 1) - 1.0
        sampling_grid_y = 2.0 * sampling_grid[..., 1] / (in_h - 1) - 1.0
        normalized_sampling_grid = torch.stack([sampling_grid_x, sampling_grid_y], dim=-1)


        sampled_features = F.grid_sample(
            input_groups[i],
            normalized_sampling_grid.view(bs, kh * kw * out_h, out_w, 2),
            mode='bilinear', padding_mode='zeros', align_corners=False
        )
        sampled_features = sampled_features.view(bs, in_channels_per_group, kh * kw, out_h, out_w)
        
 
        group_mask = mask_groups[i] # (N, K*K, H, W)
        sampled_features = sampled_features * group_mask.unsqueeze(1)
        

        group_weight = weight_groups[i].view(out_channels // groups, in_channels_per_group, kh * kw)
        out_group = torch.einsum('bifk,oif->bok', 
                                 sampled_features.view(bs, in_channels_per_group, kh * kw, out_h * out_w), 
                                 group_weight)
        output_groups.append(out_group.view(bs, out_channels // groups, out_h, out_w))


    out = torch.cat(output_groups, dim=1)
    if bias is not None:
        out += bias.view(1, -1, 1, 1)
    return out


class DeformConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True):
        super(DeformConv2d, self).__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0, \
            "in_channels and out_channels must be divisible by groups"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.offset_mask_conv = nn.Conv2d(in_channels,
                                          groups * 3 * kernel_size * kernel_size,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
            
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=1)
        self.offset_mask_conv.weight.data.zero_()
        self.offset_mask_conv.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        offset_mask = self.offset_mask_conv(x)
        
        offset_channel = self.groups * 2 * self.kernel_size**2
        offset = offset_mask[:, :offset_channel, ...]
        mask = torch.sigmoid(offset_mask[:, offset_channel:, ...])
        
        return deform_conv2d_pytorch_grouped(
            input=x,
            offset=offset,
            mask=mask,
            weight=self.weight,
            bias=self.bias,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            dilation=(self.dilation, self.dilation),
            groups=self.groups
        )



class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ECABlock(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log(in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).view(b, c, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)



class ERM_Contour_Integrity(nn.Module):


    def __init__(self, in_channels, num_classes=1, scales=[3, 5, 7], 
                 shape_prior_kernel_size=15, dcn_groups=4):
        super(ERM_Contour_Integrity, self).__init__()
        self.scales = scales
        
        self.feature_refinement = nn.Sequential(
            DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=dcn_groups),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        self.temp_pred_head = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        self.multi_scale_boundary_attention = DepthwiseSeparableConv(
            in_channels=len(scales), 
            out_channels=in_channels
        )

        self.reverse_attention_generator = nn.Sequential(
            nn.Conv2d(1, in_channels // 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1, bias=False)
        )
        
        pad = (shape_prior_kernel_size - 1) // 2
        self.shape_prior_head = nn.AvgPool2d(kernel_size=shape_prior_kernel_size, stride=1, padding=pad)

        self.eca = ECABlock(in_channels)
        self.final_pred_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def tensor_dilate(self, bin_img, ksize=3):
        pad = (ksize - 1) // 2
        return F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=pad)

    def tensor_erode(self, bin_img, ksize=3):
        pad = (ksize - 1) // 2
        return -F.max_pool2d(-bin_img, kernel_size=ksize, stride=1, padding=pad)

    def forward(self, x):
        input_feature = x
        refined_feature = self.feature_refinement(x)
        
        temp_mask = self.temp_pred_head(refined_feature)
        saliency_prob = torch.sigmoid(temp_mask)

        boundary_maps = []
        for ksize in self.scales:
            dilated = self.tensor_dilate(saliency_prob, ksize=ksize)
            eroded = self.tensor_erode(saliency_prob, ksize=ksize)
            boundary_maps.append(dilated - eroded)
        multi_scale_boundary_map = torch.cat(boundary_maps, dim=1)
        boundary_attention = self.multi_scale_boundary_attention(multi_scale_boundary_map)
        
        with torch.no_grad():
            shape_prior = self.shape_prior_head(saliency_prob)
        guided_reverse_map = shape_prior * (1 - saliency_prob)
        reverse_attention = self.reverse_attention_generator(guided_reverse_map)

        combined_spatial_attention = boundary_attention + reverse_attention
        
        feature_with_spatial_att = refined_feature * (1 + combined_spatial_attention)
        final_refined_feature = self.eca(feature_with_spatial_att) + input_feature
        
        supervised_mask = self.final_pred_head(final_refined_feature)
        
        return supervised_mask, final_refined_feature



import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):

    def __init__(self, channels: int):
        super(GATLayer, self).__init__()
        self.channels = channels
        self.W = nn.Linear(channels, channels, bias=False)
        self.attn_mlp = nn.Linear(2 * channels, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        num_modals = node_features.size(1)
        h = self.W(node_features)
        h_i = h.unsqueeze(2).expand(-1, -1, num_modals, -1)
        h_j = h.unsqueeze(1).expand(-1, num_modals, -1, -1)
        node_pairs = torch.cat([h_i, h_j], dim=-1)
        e = self.leaky_relu(self.attn_mlp(node_pairs).squeeze(-1))
        attn_weights = F.softmax(e, dim=2)
        updated_nodes = torch.bmm(attn_weights, h)
        return F.relu(updated_nodes)


import torch
import torch.nn as nn

from torch_geometric.nn import GATv2Conv

class GNNRoIFusion(nn.Module):

    def __init__(self, channels: int, num_gnn_layers: int = 2, heads: int = 4, num_conv_layers: int = 1):
        super(GNNRoIFusion, self).__init__()
        self.channels = channels
        self.heads = heads
        
        self.edge_index_cache = {}

        self.fusion_node_initializer = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels)
        )
        

        self.gnn_layers = nn.ModuleList(
            [GATv2Conv(channels, channels // heads, heads=heads) for _ in range(num_gnn_layers)]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(channels) for _ in range(num_gnn_layers)]
        )
        
        convs = []
        if num_conv_layers > 0:
            for _ in range(num_conv_layers):
                convs.extend([
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                ])
        self.spatial_interaction_block = nn.Sequential(*convs)
        
    def _get_or_create_edge_index(self, num_nodes: int, batch_size: int, device: torch.device) -> torch.Tensor:

        cache_key = (num_nodes, batch_size)

        if cache_key in self.edge_index_cache:
            return self.edge_index_cache[cache_key].to(device)

        if num_nodes <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=device)


        adj = torch.ones(num_nodes, num_nodes, dtype=torch.bool, device=device)
        adj.fill_diagonal_(False)  # GATv2Conv adds self-loops by default
        edge_index_template = adj.nonzero(as_tuple=False).t().contiguous()
        num_edges_per_graph = edge_index_template.size(1)
        offsets = torch.arange(0, batch_size, device=device) * num_nodes
        

        edge_index = edge_index_template.unsqueeze(0).expand(batch_size, -1, -1)
        edge_index = edge_index + offsets.view(batch_size, 1, 1)
        
        edge_index = edge_index.reshape(2, -1)

        self.edge_index_cache[cache_key] = edge_index
        
        return edge_index

    def forward(self, modals: list[torch.Tensor]) -> torch.Tensor:
        batch_size, _, height, width = modals[0].shape
        num_modals = len(modals)
        num_nodes_per_graph = num_modals + 1
        batch_of_pixels = batch_size * height * width


        stacked_modals = torch.stack(modals, dim=1)
        modal_nodes = stacked_modals.permute(0, 3, 4, 1, 2).reshape(batch_of_pixels, num_modals, self.channels)
        mean_modal_features = modal_nodes.mean(dim=1, keepdim=True)
        fusion_node = self.fusion_node_initializer(mean_modal_features)
        

        all_nodes_bached = torch.cat([fusion_node, modal_nodes], dim=1)
        x = all_nodes_bached.view(-1, self.channels)
    
        edge_index = self._get_or_create_edge_index(num_nodes_per_graph, batch_of_pixels, x.device)

        for i, layer in enumerate(self.gnn_layers):
            identity = x
            x = layer(x, edge_index)
            x = self.layer_norms[i](x + identity)

        output_nodes = x.view(batch_of_pixels, num_nodes_per_graph, self.channels)
        
        fused_pixel_vectors = output_nodes[:, 0, :]
        
        fused_map = fused_pixel_vectors.view(batch_size, height, width, self.channels)
        fused_map = fused_map.permute(0, 3, 1, 2)
        
 
        if self.spatial_interaction_block:
            identity = fused_map
            fused_map_spatial = self.spatial_interaction_block(fused_map)
            return fused_map_spatial + identity
        
        return fused_map
    
import torch
import torch.nn as nn

class SimpleAverageFusion(nn.Module):
    def __init__(self, channels):
   
        super(SimpleAverageFusion, self).__init__()
   
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, modal_feats):

        if not modal_feats:
            return None

        stacked_feats = torch.stack(modal_feats, dim=1)


        fused_feats = torch.mean(stacked_feats, dim=1)

        out = self.proj(fused_feats)

        return out

