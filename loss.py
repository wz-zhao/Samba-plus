import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import IOU

def total_loss_suite_optimized(pred, mask, lambda_wbce=1.0, lambda_dice=1.0, lambda_focal=1.0, gamma=2.0):

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce_loss = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce_loss = (weit * wbce_loss).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred_sig = torch.sigmoid(pred)
    inter = ((pred_sig * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sig + mask) * weit).sum(dim=(2, 3))
    
    dice_score = (2. * inter + 1e-6) / (union + 1e-6)
    dice_loss = 1. - dice_score

    focal_bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    pt = torch.exp(-focal_bce)
    focal_loss = (((1 - pt)**gamma * focal_bce)).mean()

    total_loss = (lambda_wbce * wbce_loss + lambda_dice * dice_loss).mean() + lambda_focal * focal_loss

    return total_loss

def total_loss_suite(pred, mask, 
                       lambda_structure=1.0, 
                       lambda_focal=1):

    # -----------------------------------------------------------------
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    bce = (weit * bce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    
    # Weighted IoU
    pred_sig = torch.sigmoid(pred)
    inter = ((pred_sig * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sig + mask) * weit).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    
    loss_structure = (bce + iou).mean()
    # -----------------------------------------------------------------

    focal_bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    pt = torch.exp(-focal_bce)
    gamma = 2.0
    loss_focal = ( (1 - pt)**gamma * focal_bce ).mean()
    # -----------------------------------------------------------------
 
    # 最终加权求和
    total_loss = (lambda_structure * loss_structure + 
                  lambda_focal * loss_focal)

    return total_loss



def distillation_loss(student_features, teacher_features):

    loss = 0.0
    mse_loss = nn.MSELoss()
    
    assert len(student_features) == len(teacher_features)
    
    for i in range(len(student_features)):
        loss += mse_loss(student_features[i], teacher_features[i])
        
    return loss / len(student_features)

def manual_gradient_all_reduce(model: nn.Module):

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)


def coarse_loss(pred, target):
    
    pred = torch.sigmoid(pred)
    target = target.float()

    # Flatten
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(1)
    dice = (2. * intersection + 1) / (pred_flat.sum(1) + target_flat.sum(1) + 1)
    dice_loss = 1 - dice

    # Soft IOU
    union = (pred_flat + target_flat - pred_flat * target_flat).sum(1)
    iou = (intersection + 1) / (union + 1)
    iou_loss = 1 - iou

    # Combine
    return (dice_loss + iou_loss).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t)**self.gamma * BCE_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss



def structure_loss_edge(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    bce = (weit * bce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    
    pred_sig = torch.sigmoid(pred)
    inter = ((pred_sig * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sig + mask) * weit).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)

    return (bce + iou).mean()
