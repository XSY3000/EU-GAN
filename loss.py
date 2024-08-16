import torch.nn as nn
import torch.nn.functional as F
import torch

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target, mask=None):
        """
        :param pred: tensor of shape (N, H, W) or (N, C, H, W)
        :param target: tensor of shape (N, H, W) or (N, C, H, W)
        :param mask: tensor of shape (N, H, W) or (N, C, H, W)
        :return: dice loss
        """

        if mask is not None:
            pred = pred * mask
            target = target * mask

        # Flatten both tensors
        pred = pred.view(-1)
        target = target.view(-1)

        # Compute intersection and union
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        # Compute Dice score
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1. - dice

        return dice_loss, intersection/target.sum()

def cross_entropy2d(output, target, weight=None, size_average=True):
    """
    2D cross entropu loss
    :param output: generator output
    :param target: ground truth
    :return: loss
    """
    n, c, h, w = output.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        raise RuntimeError("inconsistent dimension of outputs and targets")

    output = output.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        output, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


def gram_matrix(features):
    # 将特征图展平为二维矩阵
    batch_size, num_channels, height, width = features.size()
    features = features.view(batch_size, num_channels, height * width)

    # 计算Gram矩阵
    gram = torch.matmul(features, features.transpose(1, 2))

    # 归一化Gram矩阵
    gram = gram / (num_channels * height * width)

    return gram

def StyleLoss(real_features, fake_features):
    loss = 0
    for i in range(len(real_features)):
        real_feature = real_features[i].detach()
        fake_feature = fake_features[i]
        # 计算输入图像和目标图像的Gram矩阵
        real_gram = gram_matrix(real_feature)
        fake_gram = gram_matrix(fake_feature)

        # 计算风格损失（均方差）
        loss += F.mse_loss(real_gram, fake_gram)

    return loss
