import torch
import torch.nn as nn
import pytorch_ssim

class IOU(nn.Module):
    def __init__(self, size_average=True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(b):
            Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
            IoU1 = Iand1 / Ior1
            IoU += (1 - IoU1)
        return IoU / b if self.size_average else IoU

# 初始化損失模組
iou_loss_module = IOU(size_average=True)
bce_loss_module = nn.BCELoss()
ssim_loss_module = pytorch_ssim.SSIM(window_size=11, size_average=True)

# 組合損失函數
def combined_loss(pred, target):
    iou_loss = iou_loss_module(pred, target)
    bce_loss = bce_loss_module(pred, target)
    ssim_loss = 1 - ssim_loss_module(pred, target)
    return iou_loss + bce_loss + ssim_loss  # 調整損失權重
