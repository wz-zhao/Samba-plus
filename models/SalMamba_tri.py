import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
from .encoders.tri_vmamba import vssm_small as backbone
from .decoders.MambaDecoder import MambaDecoder


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.p_depth_dropout = 0.2
        self.p_flow_dropout = 1
        # self.p_thermal_dropout = 0.05
        self.p_pure_rgb_scenario = 0
    
        self.rq_module_rgb = RandomModule(modality='RGB',
                                     fine_grained_region_num=24,coarse_grained_region_num=6,
                                     collapse_to_val='inside_random', spacing='random')
        self.rq_module_depth = RandomModule(modality='Depth', fine_grained_region_num=20,
                                                    coarse_grained_region_num=6, collapse_to_val='inside_random',
                                                    spacing='log')
        self.rq_module_flow = RandomModule(modality='Flow', fine_grained_region_num=16,
                                                    coarse_grained_region_num=5, collapse_to_val='inside_random',
                                                    spacing='random') 
        self.rq_module_t = RandomModule(modality='Thermal', fine_grained_region_num=24, 
                                                    coarse_grained_region_num=6,collapse_to_val='inside_random',
                                                    spacing='log')

        self.backbone = backbone()
        self.channels = self.channels = [96, 192, 384, 768]
        self.decoder = MambaDecoder(img_size=[352, 352],
                                    in_channels=self.channels, 
                                    num_classes=1, 
                                    depths=[4, 4, 4, 4],
                                    embed_dim=self.channels[0], 
                                    deep_supervision=False)
    def forward(self, rgb, modal_flow, modal_depth,modal_thermal,mode,gt=None,task=None):
        from torchvision.transforms.functional import to_pil_image
        num = 3
        if mode == 'train':

            _modal_depth, _modal_flow, _modal_thermal = modal_depth, modal_flow, modal_thermal 
           

            if _modal_depth is not None and torch.rand(1).item() <0.1:
                _modal_depth = None
            if _modal_flow is not None and torch.rand(1).item() < 0.1:
               _modal_flow = None  
            if _modal_thermal is not None and torch.rand(1).item() < 0.1:
                _modal_thermal = None
            
            
            rgb = self.rq_module_rgb(rgb) if torch.rand(1).item() < 0.1 else rgb 
            if  modal_thermal is not None :
                modal_thermal = self.rq_module_t(modal_thermal) if torch.rand(1).item() < 0.1 else modal_thermal 
            if modal_depth is not None:
                modal_depth = self.rq_module_depth(modal_depth) if torch.rand(1).item() < 0.1 else modal_depth
            if modal_flow is not None:
                modal_flow = self.rq_module_flow(modal_flow) if torch.rand(1).item() < 0.1 else modal_flow 

        else:
            _modal_depth, _modal_flow, _modal_thermal = modal_depth, modal_flow, modal_thermal
           

        orisize = rgb.shape
        x, saliency,aux_preds = self.backbone(rgb, _modal_flow, _modal_depth, _modal_thermal, task = task)
       
        out = self.decoder.forward(x)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        saliency = F.interpolate(saliency, size=orisize[2:], mode='bilinear', align_corners=False)
      
        return out, saliency, aux_preds
    
    def load_pretrain_model(self, model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

if __name__ == '__main__':
    model = Model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = flow = depth = torch.ones([2, 3, 448, 448]).to(device)
    out = model(image, flow)
    print(model)
    print(out.shape)
        

######################################################################################################################################################


class RandomModule(nn.Module):
    def __init__(self, region_num=None, fine_grained_region_num=16, coarse_grained_region_num=4,
                 collapse_to_val='inside_random', spacing='random', transforms_like=False,
                 p_random_apply_rand_quant=1, modality='RGB'):
        """
        region_num: int; Default region number if not using significance-guided.
        fine_grained_region_num: int; Region number for significant areas (cải tiến 1).
        coarse_grained_region_num: int; Region number for background areas (cải tiến 1).
        collapse_to_val: str; How to represent values in a region ('middle', 'inside_random', 'all_zeros').
        spacing: str; How to space region percentiles ('random', 'uniform', 'log').
        transforms_like: bool; Whether the input is (C, H, W) or (B, C, H, W).
        p_random_apply_rand_quant: float; Probability to apply quantization.
        modality: str; Type of modality ('RGB', 'Flow', 'Depth', 'Thermal'). Used for modal-aware strategy (cải tiến 2).
        """
        super().__init__()
        self.region_num = region_num
        self.fine_grained_region_num = fine_grained_region_num
        self.coarse_grained_region_num = coarse_grained_region_num
        self.collapse_to_val = collapse_to_val
        self.spacing = spacing
        self.transforms_like = transforms_like
        self.p_random_apply_rand_quant = p_random_apply_rand_quant
        
        # Validate and set modality
        allowed_modalities = ['RGB', 'Flow', 'Depth', 'Thermal'] # Added 'Thermal'
        if modality not in allowed_modalities:
            raise ValueError(f"Modality must be one of {allowed_modalities}, but got '{modality}'")
        self.modality = modality

        if self.region_num is None and (self.fine_grained_region_num is None or self.coarse_grained_region_num is None):
            raise ValueError("Either region_num must be set, or both fine_grained_region_num and coarse_grained_region_num must be set.")

    def get_params(self, x):
        """
        x: (C, H, W)
        returns (C), (C)
        """
        C, _, _ = x.size() # one batch img
        # min_val, max_val over spatial dimension for each channel
        min_val, max_val = x.view(C, -1).min(1)[0], x.view(C, -1).max(1)[0]
        return min_val, max_val

    def _get_region_num_map(self, gt_mask, H, W, device):
        """
        Generates a region_num map based on the GT mask for significance-guided quantization.
        gt_mask: (B, 1, H, W) or (1, H, W) or (H, W)
        Returns: (B, 1, H, W) tensor with region numbers for each pixel.
        """
        if gt_mask is None:
            # If no GT mask, use the default region_num or fine_grained_region_num for all
            if self.region_num is not None:
                return torch.full((1, 1, H, W), self.region_num, dtype=torch.int, device=device)
            else:
                return torch.full((1, 1, H, W), self.fine_grained_region_num, dtype=torch.int, device=device)

        # Ensure mask is 4D (B, C, H, W) where C=1
        if gt_mask.dim() == 2:
            gt_mask = gt_mask.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        elif gt_mask.dim() == 3:
            gt_mask = gt_mask.unsqueeze(0) # (1, C, H, W) or (1, 1, H, W) assuming C=1
        
        # Ensure mask is single channel
        if gt_mask.shape[1] != 1:
            gt_mask = gt_mask[:, :1, :, :] # Take only the first channel if multi-channel

        # Resize GT mask to match input feature map size if needed
        if gt_mask.shape[-2:] != (H, W):
            gt_mask = F.interpolate(gt_mask, size=(H, W), mode='bilinear', align_corners=False)

        region_num_map = torch.where(gt_mask > 0.5,
                                     torch.tensor(self.fine_grained_region_num, device=device, dtype=torch.int),
                                     torch.tensor(self.coarse_grained_region_num, device=device, dtype=torch.int))
        return region_num_map

    def _log_spacing_percentiles(self, min_val, max_val, num_regions, device):
        """Generates log-spaced percentiles for a given channel."""
        # Clamp to avoid log(0) or log(negative)
        # Using a small offset for log computation to avoid issues with values exactly at zero.
        # This offset should be very small compared to the typical value range.
        log_offset = 1e-6 
        min_val_log = min_val.clamp(min=log_offset)
        max_val_log = max_val.clamp(min=log_offset)

        # Handle cases where min_val and max_val are very close after clamping
        if torch.isclose(min_val_log, max_val_log):
            return torch.empty(0, device=device) # No percentiles needed if range is zero

        log_min_scaled = torch.log(min_val_log)
        log_max_scaled = torch.log(max_val_log)

        # We need num_regions - 1 percentiles to divide into num_regions regions
        if num_regions - 1 <= 0:
            return torch.empty(0, device=device)

        # Generate points in log space including min_log and max_log, then exclude ends for division points
        log_percentiles = torch.linspace(log_min_scaled, log_max_scaled, num_regions + 1, device=device)
        
        # Exclude the very first (log_min_scaled) and very last (log_max_scaled) points
        # These are the actual boundaries, not the intermediate division points
        return torch.exp(log_percentiles[1:-1])

    def forward(self, x, gt_mask=None):
        """
        x: (B, C, H, W) or (C, H, W)
        gt_mask: (B, 1, H, W) or (1, H, W) or (H, W) Ground truth saliency map or pseudo label (for cải tiến 1).
        """
        # Get machine epsilon for float type for comparisons, ensuring it's tiny enough
        # Using `torch.finfo(x.dtype).eps` is more robust than a fixed constant
        # A small multiplier can be added for robustness in comparisons
        robust_epsilon = torch.finfo(x.dtype).eps * 4 
        
        if self.p_random_apply_rand_quant != 1:
            x_orig = x

        # Determine input shape and flatten if not transforms_like
        if not self.transforms_like:
            B, C_orig, H, W = x.shape
            x_reshaped = x.view(B * C_orig, H, W)
            C_flattened = B * C_orig
        else:
            C_orig, H, W = x.shape
            B = 1 # Assume batch size of 1 for transforms_like
            x_reshaped = x
            C_flattened = C_orig

        min_val_per_channel, max_val_per_channel = self.get_params(x_reshaped) # -> (C_flattened), (C_flattened)

        # --- Cải tiến 1: 基于显著性结构的“显著性引导式随机量化” ---
        region_num_map = self._get_region_num_map(gt_mask, H, W, x.device) # (B or 1, 1, H, W)

        # Expand region_num_map to match the flattened 'channels' of x_reshaped
        if region_num_map.shape[0] == 1 and B > 1:
            region_num_map = region_num_map.repeat(B, 1, 1, 1)
        
        # Now replicate for each original channel if not transforms_like
        if not self.transforms_like:
            region_num_map_expanded = region_num_map.repeat_interleave(C_orig, dim=1).view(C_flattened, 1, H, W)
        else:
            # If transforms_like, x is (C_orig, H, W), so region_num_map needs to be (C_orig, 1, H, W)
            if region_num_map.shape[0] == 1 and C_orig > 1: # Broadcast if mask is for single channel but input is multi-channel
                region_num_map_expanded = region_num_map.repeat(C_orig, 1, 1, 1)
            else:
                 region_num_map_expanded = region_num_map # Already has correct C or 1 and will be handled below

        quantized_x = torch.zeros_like(x_reshaped)

        # Process each "channel" (which could be a batch_channel combination)
        for i in range(C_flattened):
            current_x_channel_data = x_reshaped[i, :, :] # (H, W)
            current_min_val = min_val_per_channel[i]
            current_max_val = max_val_per_channel[i]

            # Get the unique region_num values present in this flattened channel's map
            unique_region_nums = torch.unique(region_num_map_expanded[i, 0, :, :]).int().tolist()

            temp_quantized_channel = torch.zeros_like(current_x_channel_data)

            # Handle cases where min_val and max_val are extremely close, effectively a single value
            if torch.isclose(current_min_val, current_max_val, atol=robust_epsilon):
                temp_quantized_channel.fill_(current_min_val)
                quantized_x[i, :, :] = temp_quantized_channel
                continue

            for rn in unique_region_nums:
                # Create a mask for pixels corresponding to the current region_num 'rn'
                rn_mask = (region_num_map_expanded[i, 0, :, :] == rn) # (H, W) boolean

                if not rn_mask.any(): # Skip if no pixels for this region_num
                    continue

                # Extract pixels that fall into this specific region_num 'rn'
                pixels_to_quantize = current_x_channel_data[rn_mask] # (num_pixels,)

                # Calculate region percentiles based on the current 'rn'
                # These are the internal division points
                # --- Modality-specific spacing for Depth and Thermal ---
                if self.modality in ['Depth', 'Thermal'] and self.spacing == 'log':
                    internal_division_points = self._log_spacing_percentiles(current_min_val, current_max_val, rn, x.device)
                elif self.spacing == "random":
                    internal_division_points = torch.rand(rn - 1, device=x.device) if rn > 1 else torch.empty(0, device=x.device)
                    # Scale to the actual value range
                    internal_division_points = internal_division_points * (current_max_val - current_min_val) + current_min_val
                elif self.spacing == "uniform":
                    internal_division_points = torch.linspace(current_min_val, current_max_val, rn + 1, device=x.device)[1:-1] if rn > 1 else torch.empty(0, device=x.device)
                else:
                    raise NotImplementedError(f"Spacing '{self.spacing}' not implemented for modality '{self.modality}'.")

                # Ensure division points are sorted
                if internal_division_points.numel() > 0:
                    internal_division_points = internal_division_points.sort(0)[0]
                    # Clamp to ensure they are within the min/max range
                    internal_division_points = internal_division_points.clamp(min=current_min_val, max=current_max_val)

                # --- Improved interval boundary generation ---
                # All unique boundaries including min_val and max_val
                all_boundaries = torch.cat([current_min_val.view(1), internal_division_points, current_max_val.view(1)], dim=0)
                
                # left_ends: [b_0, b_1, ..., b_n-1]
                ordered_region_left_ends = all_boundaries[:-1]
                # right_ends: [b_1, b_2, ..., b_n]
                ordered_region_right_ends = all_boundaries[1:]

                # For comparison (is_inside_each_region), use left_ends as inclusive, right_ends as exclusive
                # Except for the very last interval, where right_end is inclusive of max_val.
                # To handle floating point issues and ensure max_val is always included in the last bin,
                # we use a very tiny epsilon to extend the last bin's upper bound for checking.
                # All other bins are [lower, upper)
                ordered_region_right_ends_for_checking = ordered_region_right_ends.clone()
                ordered_region_right_ends_for_checking[-1] = current_max_val + 1e-2  
                # print(f"[Check] current_max_val: {current_max_val.item()}")
                # print(f"[Check] ordered_region_right_ends_for_checking[-1]: {ordered_region_right_ends_for_checking[-1].item()}")
                # print(f"[Check] current_max_val + robust_epsilon: {(current_max_val + robust_epsilon).item()}")
                # print(f"[Check] robust_epsilon: {robust_epsilon}")

                # Associate region id for the current segment of pixels
                # Reshape for broadcasting comparison
                pixels_view = pixels_to_quantize.view(1, -1) # (1, num_pixels)
                left_ends_view = ordered_region_left_ends.view(-1, 1) # (rn, 1)
                right_ends_view = ordered_region_right_ends_for_checking.view(-1, 1) # (rn, 1)

                is_inside_each_region = (pixels_view < right_ends_view) * (pixels_view >= left_ends_view)
                
                # Sanity check: each pixel falls into one sub_range
                # If this fails, it means a pixel is not covered, or covered by multiple ranges.
                sum_regions = is_inside_each_region.sum(0)
                if not (sum_regions == 1).all():
                    # Detailed debug print if assertion fails during development
                    print(f"Assertion failed for channel {i}, rn {rn}")
                    print(f"Pixels to quantize count: {pixels_to_quantize.numel()}")
                    print(f"Min/Max current_x: {pixels_to_quantize.min().item()}/{pixels_to_quantize.max().item()}")
                    print(f"Min/Max channel: {current_min_val.item()}/{current_max_val.item()}")
                    print(f"Robust Epsilon: {robust_epsilon}")
                    print(f"Left Ends: {ordered_region_left_ends.squeeze()}")
                    print(f"Right Ends Check: {ordered_region_right_ends_for_checking.squeeze()}")
                    print(f"Original Right Ends: {ordered_region_right_ends.squeeze()}") # For debugging midpoints
                    
                    # Find problematic pixels
                    pixels_not_covered_mask = (sum_regions == 0)
                    pixels_covered_multiple_times_mask = (sum_regions > 1)
                    
                    if pixels_not_covered_mask.any():
                        print(f"Pixels not covered ({pixels_not_covered_mask.sum()}): {pixels_to_quantize[pixels_not_covered_mask]}")
                    if pixels_covered_multiple_times_mask.any():
                        print(f"Pixels covered multiple times ({pixels_covered_multiple_times_mask.sum()}): {pixels_to_quantize[pixels_covered_multiple_times_mask]}")
                    
                    raise AssertionError("Each pixel must fall into exactly one sub-range.")

                associated_region_id = torch.argmax(is_inside_each_region.int(), dim=0, keepdim=True) # (1, num_pixels)

                if self.collapse_to_val == 'middle':
                    # Calculate middle points using the original (non-epsilon-modified) right ends
                    ordered_region_mid = (ordered_region_right_ends + ordered_region_left_ends) / 2
                    proxy_vals = torch.gather(ordered_region_mid.expand([-1, pixels_to_quantize.numel()]), 0, associated_region_id)[0]
                elif self.collapse_to_val == 'inside_random':
                    # Need rn random points, one for each region
                    proxy_percentiles_per_region = torch.rand(rn, device=x.device)
                    
                    # Use the original (non-epsilon-modified) left/right ends for random point generation
                    region_widths = ordered_region_right_ends - ordered_region_left_ends # (rn,)
                    
                    # Ensure region_widths are non-negative, especially for very small ranges
                    region_widths = region_widths.clamp(min=0) 
                    
                    ordered_region_rand = ordered_region_left_ends + proxy_percentiles_per_region * region_widths
                    
                    proxy_vals = torch.gather(ordered_region_rand.unsqueeze(1).expand(-1, pixels_to_quantize.numel()),0,associated_region_id)[0]

                elif self.collapse_to_val == 'all_zeros':
                    proxy_vals = torch.zeros_like(pixels_to_quantize)
                else:
                    raise NotImplementedError

                # Place the quantized values back into the temporary channel
                temp_quantized_channel[rn_mask] = proxy_vals.type(x.dtype)
            
            quantized_x[i, :, :] = temp_quantized_channel

        if not self.transforms_like:
            x_final = quantized_x.view(B, C_orig, H, W)
        else:
            x_final = quantized_x

        if self.p_random_apply_rand_quant != 1:
            if not self.transforms_like:
                rand_mask = (torch.rand([B, 1, 1, 1], device=x.device) < self.p_random_apply_rand_quant)
                x_output = torch.where(rand_mask, x_final, x_orig)
            else:
                rand_mask = (torch.rand([C_orig, 1, 1], device=x.device) < self.p_random_apply_rand_quant)
                x_output = torch.where(rand_mask, x_final, x_orig)
        else:
            x_output = x_final

        return x_output

