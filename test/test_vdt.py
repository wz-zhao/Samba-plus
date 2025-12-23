import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from dataset.vdt_dataset import Data
from models.SalMamba_tri import Model
# ==============================================================================


from scipy.ndimage import convolve, distance_transform_edt as bwdist
from skimage.measure import label

def _prepare_data(pred, gt):
    gt = gt > 0.5
    pred = pred.astype(np.float64)
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt

def calculate_mae(pred, gt):
    pred, gt = _prepare_data(pred, gt)
    return np.mean(np.abs(pred - gt))

def calculate_fmeasure(pred, gt):
    pred, gt = _prepare_data(pred, gt)
    beta2 = 0.3
    
    thresholds = np.linspace(0, 1, 256)
    f_scores = []
    
    for threshold in thresholds:
        pred_binary = pred >= threshold
        
        tp = np.sum((pred_binary == 1) & (gt == 1))
        fp = np.sum((pred_binary == 1) & (gt == 0))
        fn = np.sum((pred_binary == 0) & (gt == 1))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        f_score = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)
        f_scores.append(f_score)
        
    f_scores = np.array(f_scores)
    return np.max(f_scores), np.mean(f_scores)


def calculate_smeasure(pred, gt):
    pred, gt = _prepare_data(pred, gt)
    alpha = 0.5
    y = np.mean(gt)
    if y == 0:
        return 1.0 - np.mean(pred)
    elif y == 1:
        return np.mean(pred)
    else:
        s_object = _object(pred, gt)
        s_region = _region(pred, gt)
        return alpha * s_object + (1 - alpha) * s_region

def _object(pred, gt):
    fg = pred[gt]
    bg = pred[~gt]
    u = np.mean(fg)
    v = np.mean(bg)
    return u - v

def _region(pred, gt):
    x, y = _centroid(gt)
    part_info = _divide_with_gt(pred, gt, x, y)
    w1, w2, w3, w4 = part_info['weight']
    
    pred1, pred2, pred3, pred4 = part_info['pred']
    gt1, gt2, gt3, gt4 = part_info['gt']
    
    s1 = _ssim(pred1, gt1)
    s2 = _ssim(pred2, gt2)
    s3 = _ssim(pred3, gt3)
    s4 = _ssim(pred4, gt4)
    
    return w1 * s1 + w2 * s2 + w3 * s3 + w4 * s4
    
def _centroid(gt):
    h, w = gt.shape
    if np.sum(gt) == 0:
        return h // 2, w // 2
    
    area_object = np.sum(gt)
    row_idx, col_idx = np.where(gt)
    x = np.round(np.mean(row_idx))
    y = np.round(np.mean(col_idx))
    return int(x), int(y)

def _divide_with_gt(pred, gt, x, y):
    h, w = gt.shape
    area = h * w
    
    lt_pred, lt_gt = pred[0:x, 0:y], gt[0:x, 0:y]
    rt_pred, rt_gt = pred[0:x, y:w], gt[0:x, y:w]
    lb_pred, lb_gt = pred[x:h, 0:y], gt[x:h, 0:y]
    rb_pred, rb_gt = pred[x:h, y:w], gt[x:h, y:w]
    
    w1 = (x * y) / area
    w2 = (x * (w - y)) / area
    w3 = ((h - x) * y) / area
    w4 = 1.0 - w1 - w2 - w3
    
    return dict(
        pred=(lt_pred, rt_pred, lb_pred, rb_pred),
        gt=(lt_gt, rt_gt, lb_gt, rb_gt),
        weight=(w1, w2, w3, w4)
    )

def _ssim(pred, gt):
    h, w = pred.shape
    N = h * w
    
    x = np.mean(pred)
    y = np.mean(gt)
    
    sigma_x = np.sum((pred - x)**2) / (N - 1)
    sigma_y = np.sum((gt - y)**2) / (N - 1)
    sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1)
    
    alpha = 2 * x * y + 1e-8
    beta = x**2 + y**2 + 1e-8
    gamma = 2 * sigma_xy + 1e-8
    omega = sigma_x + sigma_y + 1e-8
    
    return (alpha * gamma) / (beta * omega)


def calculate_emeasure(pred, gt):
    pred, gt = _prepare_data(pred, gt)
    thresholds = np.linspace(0, 1, 256)
    e_scores = []

    for threshold in thresholds:
        pred_binary = pred >= threshold
        
        enhanced_matrix = _get_enhanced_matrix(pred_binary, gt)
        score = np.sum(enhanced_matrix) / (enhanced_matrix.size - 1 + 1e-8)
        e_scores.append(score)
        
    e_scores = np.array(e_scores)
    return np.max(e_scores), np.mean(e_scores)

def _get_enhanced_matrix(pred, gt):
    align_matrix = 2 * gt * pred / (gt*gt + pred*pred + 1e-8)
    enhanced_matrix = (align_matrix - np.mean(align_matrix))**2
    return enhanced_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_thread', type=int, default=8)
    parser.add_argument('--model_path', type=str, default='./Mix_training.pth')
    parser.add_argument('--dataset_root', type=str, default='./VDT-2048 dataset/Test/')
    parser.add_argument('--testsavefold', type=str, default='./VDT')
    
    config = parser.parse_args()

    print("PyTorch and CUDA is available: ", torch.cuda.is_available())
    print('------------------------------------------')

    model = Model()
    if config.cuda:
        model = model.cuda()

    assert os.path.exists(config.model_path), f'Pretrained model path does not exist: {config.model_path}'
    print(f'Loading model from {config.model_path}')
    model.load_pretrain_model(config.model_path)
    model.eval()

    out_path = config.testsavefold
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    data = Data(root=config.dataset_root, mode='test')
    loader = DataLoader(data, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_thread)
    img_num = len(loader)

    total_mae, total_max_fm, total_mean_fm = 0.0, 0.0, 0.0
    total_max_em, total_mean_em, total_s_measure = 0.0, 0.0, 0.0

    with torch.no_grad():
        for sample, (H, W), name in tqdm(loader, desc="Testing on VDT dataset"):
            image, depth, thermal, label = sample['image'], sample['depth'], sample['thermal'], sample['label']

            if config.cuda:
                image, depth, thermal = image.cuda(), depth.cuda(), thermal.cuda()

            flow = None
            out, saliency, m = model(image, flow, depth, thermal, mode='test', gt=None, task=None)
  
            score = F.interpolate(out, size=(H.item(), W.item()), mode='bilinear', align_corners=True)
            label = F.interpolate(label, size=(H.item(), W.item()), mode='bilinear', align_corners=True)
            pred = torch.sigmoid(score).squeeze().cpu().numpy()

            save_name = os.path.join(out_path, name[0][:-4] + '.png')
            pred_save = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            cv2.imwrite(save_name, 255 * pred_save)

if __name__ == '__main__':
    main()