#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import argparse
import itertools

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from lpips import LPIPS
from safetensors.torch import load_file

from HLA-AD-D import (
    ImageFolderWithoutTarget,
    ImageFolderWithPath,
    InfiniteDataloader
)
from ModelHLA-AD import (
    vit_base_patch16_224_ours,
    LPIPSOnlyLoss,
    feature_anomaly_score,
    feature_anomaly_scoren
)

# ===========================
# Global Config
# ===========================
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
seed = 3402
image_size = 224

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

output_dirT = "/data0/xmh/code/24code/VITAD/output/HLA-AD/Test"
output_dirM = "/data0/xmh/code/24code/VITAD/output/HLA-AD/Model"

os.makedirs(output_dirT, exist_ok=True)
os.makedirs(output_dirM, exist_ok=True)


# ===========================
# Utils
# ===========================
def denormalize(tensor):
    mean_t = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  device=tensor.device).view(1, -1, 1, 1)
    return tensor * std_t + mean_t


def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def cvt2heatmap(gray):
    return cv2.applyColorMap(gray.astype(np.uint8), cv2.COLORMAP_JET)


def show_cam_on_image(img, heatmap):
    cam = img / 255.0 + heatmap / 255.0
    cam = cam / cam.max()
    return np.uint8(cam * 255)


# ===========================
# Argparse
# ===========================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('--mvtec_ad_path', default='/data0/xmh/data')
    parser.add_argument('--mvtec_loco_path', default='/data0/xmh/data/MVTec_LOCO')
    parser.add_argument('--train_steps', type=int, default=20000)
    return parser.parse_args()


# ===========================
# Transforms
# ===========================
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

augment_transform = transforms.ColorJitter(
    brightness=0.2, contrast=0.2, saturation=0.2
)


def train_transform(img):
    return default_transform(img), default_transform(augment_transform(img))


# ===========================
# Test
# ===========================
@torch.no_grad()
def test(test_set, model, adclass, iteration):
    model.eval()

    y_true, y_score = [], []
    pixel_preds, pixel_labels = [], []

    lpips_global = LPIPS(spatial=False).to(device).eval()
    lpips_spatial = LPIPS(spatial=True).to(device).eval()

    for image, target, path in test_set:
        image = default_transform(image)[None].to(device)

        recon, posterior, feats, feats_o = model(image)
        _, feats_r, feats_o_r = model.forward_features(recon)

        # feature anomaly
        amap = feature_anomaly_score(
            feats_o, feats_o_r, mode='l2', input_size=(224, 224)
        )
        image_score = amap.view(1, -1).max(dim=1)[0].cpu().numpy()

        # LPIPS pixel
        p_map = lpips_spatial(image, recon)
        p_map = F.interpolate(p_map, size=image.shape[2:], mode='bilinear')
        pixel_map = p_map.squeeze().cpu().numpy()

        # save visualization
        img_name = os.path.splitext(os.path.basename(path))[0]
        defect = os.path.basename(os.path.dirname(path))
        cls_name = os.path.basename(adclass)

        save_dir = os.path.join(output_dirT, cls_name, defect)
        os.makedirs(save_dir, exist_ok=True)

        img_denorm = denormalize(image).clamp(0, 1)
        rec_denorm = denormalize(recon).clamp(0, 1)

        save_image(img_denorm, os.path.join(save_dir, f"{img_name}_input.png"))
        save_image(rec_denorm, os.path.join(save_dir, f"{img_name}_recon.png"))

        heat = gaussian_filter(pixel_map, sigma=2)
        heat = cvt2heatmap(min_max_norm(heat) * 255)

        raw = cv2.resize(cv2.imread(path), (224, 224))
        cam = show_cam_on_image(raw, heat)
        cv2.imwrite(os.path.join(save_dir, f"{img_name}_heat.png"), cam)

        y_true.append(0 if defect == 'good' else 1)
        y_score.append(image_score)

        pixel_preds.extend(pixel_map.flatten())
        pixel_labels.extend(target.flatten().numpy())

    auc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    pixel_rocauc = roc_auc_score(
        (np.array(pixel_labels) > 0.5).astype(int),
        pixel_preds
    )

    return pixel_rocauc * 100, auc * 100, auprc


# ===========================
# Main
# ===========================
def main(adclass):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = get_args()
    dataset_root = (
        args.mvtec_ad_path if args.dataset == 'mvtec_ad'
        else args.mvtec_loco_path
    )

    train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_root, adclass, 'train'),
        transform=transforms.Lambda(train_transform)
    )

    test_set = ImageFolderWithPath(
        os.path.join(dataset_root, adclass, 'test')
    )

    train_loader = DataLoader(
        train_set, batch_size=8, shuffle=True,
        num_workers=4, pin_memory=True
    )

    train_loader = InfiniteDataloader(train_loader)

    model = vit_base_patch16_224_ours(
        pretrained=False, drop_path_rate=0.1
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * args.train_steps), gamma=0.1
    )

    loss_fn = LPIPSOnlyLoss(kl_weight=1e-6)

    model.train()
    for step, (img, _), _ in zip(
        tqdm(range(args.train_steps)),
        train_loader,
        itertools.repeat(None)
    ):
        img = img.to(device)

        recon, posterior, feats, feats_o = model(img)
        _, feats_r, feats_o_r = model.forward_features(recon)

        loss, _ = loss_fn(img, recon, posterior, feats_o, feats_o_r)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 1000 == 0:
            torch.save(
                model,
                os.path.join(output_dirM, f"{adclass.replace('/', '_')}_{step}.pth")
            )

            p_auc, auc, auprc = test(
                test_set, model, adclass, step
            )

            print(f"[{step}] AUC={auc:.2f} | PixelAUC={p_auc:.2f} | AUPRC={auprc:.4f}")


# ===========================
# Entry
# ===========================
if __name__ == '__main__':
    classes = [
        'UAD/UAD_OCT17_MVFA_mvtecstyle/oct17',
        'UAD/UAD_BrainMRI_MVFA_mvtecstyle/brainmri',
        'UAD/UAD_APTOS_mvtecstyle/aptos',
        'UAD/UAD_HIS_MVFA_mvtecstyle/his',
        'UAD/UAD_RESC_MVFA_mvtecstyle/resc',
    ]

    for cls in classes:
        print(f"\n=== Training {cls} ===")
        main(cls)
