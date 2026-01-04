
#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from torchvision.datasets import ImageFolder
from PIL import Image
import os
from torchvision import utils,transforms



class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample

# class ImageFolderWithPath(ImageFolder):
#
#     def __getitem__(self, index):
#         path, target = self.samples[index]
#         sample, target = super().__getitem__(index)
#         return sample, target, path

gt_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
class ImageFolderWithPath(ImageFolder):
    """返回 (image_tensor, target, path, gt_tensor, gt_path)
       若路径中包含'good'，则gt_tensor为全零张量
    """

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)

        # 将 test 路径替换成 ground_truth
        gt_path = path.replace('test' , 'ground_truth')
        # print(gt_path)

        # 情况1: "good" 样本 → 生成全零真值
        if "good" in path:
            # 生成与 sample 尺寸一致的全零图像
            gt_tensor = torch.zeros((1, 224, 224), dtype=torch.float32)
            # gt_tensor = gt_transform(gt_tensor)
        else:
            # 情况2: 异常样本 → 读取对应 GT 掩码
            base, ext = os.path.splitext(gt_path)  # 分离扩展名
            gt_path = base + '_mask' + ext  # 例如 CNV-1016042-1_mask.png
            gt_img = Image.open(gt_path).convert('L')

            gt_tensor = gt_transform(gt_img)



        return sample, gt_tensor, path
def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
