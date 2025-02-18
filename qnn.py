from math import sqrt, log, ceil, pi, inf, nan
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.polys.polyconfig import query


class Flipout(nn.Module):
    def __init__(self, p=0.25):
        super(Flipout, self).__init__()
        assert 0 < p < 0.5, "p should be smaller than 0.5!"
        self.p = p
        self.flip_scale = 1. / (1. - 2. * p)

    def forward(self, x):
        if self.training:
            x[torch.rand_like(x) < self.p] *= -1
            return self.flip_scale * x
        return x


class AdaptiveFocalLoss(nn.Module):
    def __init__(self, num_classes, momentum=0.9, alpha=0.5, beta=0., gamma=1., focal=0., reverse_sample_set=None,
                 label_smoothing=0.):
        super(AdaptiveFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.focal = focal
        self.log_softmax = nn.LogSoftmax(1)
        self.weight = torch.zeros(num_classes)
        self.reverse_sample_set = reverse_sample_set
        self.label_smoothing = label_smoothing
        # self.loss = nn.CrossEntropyLoss(weight=torch.zeros(num_classes))

    def forward(self, y, y_true):
        y_log = self.log_softmax(y)
        y_pred = y_log.max(1).indices.data
        # computing focal_score!
        # from sklearn.metrics import precision_recall_fscore_support
        # p, r = precision_recall_fscore_support(y_true.cpu().numpy(), y_pred.cpu().numpy())[:2]
        _, precision_count = y_pred.unique(return_counts=True)
        _, recall_count = y_true.unique(return_counts=True)
        _, right_count = y_pred[y_pred == y_true].unique(return_counts=True)
        p, r = right_count / precision_count, right_count / recall_count
        focal_score = 1. - p * r / (self.alpha * p + (1 - self.alpha) * r)
        if self.beta > 0.:
            focal_score *= 1. + self.beta * (1. - y_log.exp())
        if self.gamma != 1.:
            focal_score **= self.gamma
        # update loss weights!
        self.weight *= self.momentum
        self.weight += (1. - self.momentum) * focal_score
        # computing label matrix with imbalance label smoothing method!
        label_map = torch.zeros_like(y)
        label_map[torch.arange(label_map.size(0)), y_true] = 1.
        if self.label_smoothing > 0:
            label_smoothing = self.label_smoothing * torch.zeros_like(self.weight)
            if self.reverse_sample_set is not None:
                label_smoothing[self.reverse_sample_set] *= -1
            label_map += label_smoothing
        # computing cross entropy loss with reduction = 'mean'
        label_weight = label_map * self.weight
        return -(label_weight * y_log).sum() / label_weight.sum()


class ResBatchNorm2d(nn.Module):
    def __init__(self, in_channels, norm_scale, bias=False, short_cut=None):
        super(ResBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels, affine=bias) # nn.BatchNorm2d(channels, affine=True) # FullNorm() # SphereNorm2d(channels)  #
        self.norm_scale = norm_scale # nn.Parameter(torch.tensor(norm_scale)) # norm_scale
        self.alpha = nn.Parameter(torch.tensor(1.)) if short_cut is not None else None
        self.shortcut = short_cut if short_cut is not None else None

    def forward(self, x):
        y = self.norm_scale * self.bn(x)
        if self.alpha is not None:
            y += self.shortcut * self.alpha * x
        return y # self.norm_scale * y  # self.norm_scale * (self.norm_momentum * x + self.bn(x)) # self.norm_scale * (self.norm_momentum * x + F.layer_norm(x, x.shape[1:])) #


class ResLayerNorm2d(nn.Module):
    def __init__(self, in_channels, norm_scale, bias=False, short_cut=None):
        super(ResLayerNorm2d, self).__init__()
        # self.ln = nn.LayerNorm(image_size, elementwise_affine=bias) # nn.BatchNorm2d(channels, affine=True) # FullNorm() # SphereNorm2d(channels)  #
        self.norm_scale = norm_scale # nn.Parameter(torch.tensor(norm_scale)) #
        self.alpha = nn.Parameter(torch.tensor(1.)) if short_cut is not None else None
        self.shortcut = short_cut if short_cut is not None else None

    def forward(self, x):
        y = self.norm_scale * F.layer_norm(x, x.shape[1:])
        if self.alpha is not None:
            y += self.shortcut * self.alpha * x
        return y # self.norm_scale * y


class ResReLU(nn.Module):
    def __init__(self, relu_neg_momentum=0., reverse=False):
        super(ResReLU, self).__init__()
        self.relu = nn.ReLU()
        self.relu_neg_momentum = nn.Parameter(torch.tensor(relu_neg_momentum))
        self.reverse = reverse

    def forward(self, x):
        return self.relu_neg_momentum * x + (-self.relu(-x) if self.reverse else self.relu(x))


# class ReLUSplitNorm(nn.Module):
#     def __init__(self, in_channels, norm_scale, bias=False, short_cut=None, relu_neg_momentum=0.):
#         super(ReLUSplitNorm, self).__init__()
#         self.norm1 = ResLayerNorm2d(in_channels, norm_scale, bias, short_cut)
#         self.norm2 = ResLayerNorm2d(in_channels, norm_scale, bias, short_cut)
#         self.relu1 = ResReLU(relu_neg_momentum, reverse=False)
#         self.relu2 = ResReLU(relu_neg_momentum, reverse=True)
#
#     def forward(self, x):
#         y1 = self.norm1(self.relu1(x))
#         y2 = self.norm2(self.relu2(x))
#         return y1, y2


def huber_full_norm(x):
    x = x - x.mean()
    x_abs = x.abs()
    lt1_mask = (x_abs < 0.1).data
    if lt1_mask.all():
        x_std = ((x ** 2).mean() + 1e-5).sqrt()
    elif not lt1_mask.any():
        x_std = x_abs.mean()
    else:
        lt1_mask = lt1_mask.type(torch.int)
        gt1_mask = 1 - lt1_mask
        lt1_num = lt1_mask.sum()
        x_std2 = ((lt1_mask * x) ** 2).sum().sqrt() / sqrt(lt1_num)
        x_std1 = (gt1_mask * x_abs).sum() / (x.numel() - lt1_num)
        x_std = x_std1 + x_std2
    return x / x_std


class HuberLayerNorm(nn.Module):
    def __init__(self, layer_dim=None):
        super(HuberLayerNorm, self).__init__()
        if layer_dim is None:
            layer_dim = -1
        self.layer_dim = layer_dim

    def forward(self, x):
        x = x - x.mean(self.layer_dim, keepdim=True)
        x_abs = x.abs()
        lt1_mask = (x_abs <= 1).data
        if lt1_mask.all():
            x_std = ((x ** 2).mean(self.layer_dim, keepdim=True) + 1e-5).sqrt()
        elif not lt1_mask.any():
            x_std = x_abs.mean(self.layer_dim, keepdim=True)
        else:
            lt1_mask = lt1_mask.type(torch.int)
            gt1_mask = 1 - lt1_mask
            lt1_num = lt1_mask.sum()
            x_std2 = ((lt1_mask * x) ** 2).sum(self.layer_dim, keepdim=True).sqrt() / sqrt(lt1_num)
            x_std1 = (gt1_mask * x_abs).sum(self.layer_dim, keepdim=True) / (x.numel() - lt1_num)
            x_std = x_std1 + x_std2
        return x / x_std


class ReLUSplitNorm(nn.Module):
    def __init__(self, in_channel, norm_scale):
        super(ReLUSplitNorm, self).__init__()
        self.up_avg_momentum = nn.Parameter(torch.tensor(1.))
        self.down_avg_momentum = nn.Parameter(torch.tensor(1.))  # nn.Parameter(torch.ones(in_channel, 1, 1)) # nn.Parameter(torch.tensor(1.))
        self.up_relu_momentum = nn.Parameter(torch.tensor(1.))
        self.down_relu_momentum = nn.Parameter(torch.tensor(1.))
        self.up_norm_momentum = nn.Parameter(torch.tensor(0.)) # nn.Parameter(torch.zeros(in_channel, 1, 1)) #
        self.down_norm_momentum = nn.Parameter(torch.tensor(0.)) # nn.Parameter(torch.zeros(in_channel, 1, 1)) # nn.Parameter(torch.tensor(0.))
        self.momentum = nn.Parameter(torch.tensor(1.))
        self.norm_scale = norm_scale
        self.norm1 = nn.BatchNorm2d(in_channel, affine=False)
        self.norm2 = nn.BatchNorm2d(in_channel, affine=False)

    def forward(self, x):
        avg_dims = list(range(1, x.dim()))
        # 均值之上归一化
        avg0 = x.mean() # [1, 2, 3], keepdim=True
        up_mask = (x > avg0).type(torch.int).data
        x1 = up_mask * avg0 + F.gelu(x - avg0) #self.up_avg_momentum * up_mask * avg0
        # avg1 = x1.sum(avg_dims, keepdim=True) / pos_size
        # x1_norm = self.norm_scale * torch.sqrt(pos_size / total_size) * F.layer_norm(x1 + avg1 * downmask, x1.shape[1:]) #
        x1_norm = self.norm_scale * self.norm1(x1) # F.layer_norm(x1, x1.shape[1:]) #(x1 - x1.mean()) / (x1.std() + 1e-4) #   [0, 2, 3], keepdim=True, unbiased=False / ((x1 - x1.mean()).abs().mean() + 1e-4) # * huber_full_norm(x1) #  self.norm_scale * F.layer_norm(x1, x1.shape[1:])
        # 均值之下归一化
        x2 = (1 - up_mask) * avg0 - F.gelu(avg0 - x) # self.down_avg_momentum * (1 - up_mask) * avg0  -self.down_avg_momentum * avg0 + F.gelu(-x) #  + F.gelu(avg0 - x) # + downmask * x self.down_relu_neg_momentum * x + avg0
        # avg2 = x2.sum(avg_dims, keepdim=True) / neg_size
        # x2_norm = self.norm_scale * torch.sqrt(neg_size / total_size) * F.layer_norm(x2 + avg2 * upmask, x2.shape[1:]) #
        x2_norm = self.norm_scale * self.norm2(x2) # F.layer_norm(x2, x2.shape[1:]) #(x2 - x2.mean()) / (x2.std() + 1e-4) #    / ((x2 - x2.mean()).abs().mean() + 1e-4) # * huber_full_norm(x2) # self.norm_scale * F.layer_norm(x2, x2.shape[1:])
        # x1 = self.up_norm_momentum * x1 + x1_norm
        # x2 = self.down_norm_momentum * x2 + x2_norm
        # x1 = self.up_norm_momentum * x1 + x1_norm
        # x2 = self.down_norm_momentum * x2 + x2_norm
        return x1, x2, self.up_norm_momentum * x1 + x1_norm, self.down_norm_momentum * x2 + x2_norm


class AttnConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, attn_kernels, norm_scale):
        super(AttnConv2d, self).__init__()
        if in_channels != out_channels:
            self.conv_momentum = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.conv_momentum = nn.Parameter(torch.tensor(1.))
        self.key_conv1 = nn.Conv2d(in_channels, in_channels, attn_kernels, padding=attn_kernels // 2, bias=False)
        self.key_conv2 = nn.Conv2d(in_channels, in_channels, attn_kernels, padding=attn_kernels // 2, bias=False)
        self.query_conv1 = nn.Conv2d(in_channels, out_channels, attn_kernels, padding=attn_kernels // 2, bias=False)
        self.query_conv2 = nn.Conv2d(in_channels, out_channels, attn_kernels, padding=attn_kernels // 2, bias=False)
        # self.conv5 = nn.Conv2d(in_channels, out_channels, attn_kernels, padding=attn_kernels // 2, bias=False)
        # self.conv6 = nn.Conv2d(in_channels, out_channels, attn_kernels, padding=attn_kernels // 2, bias=False)
        # self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # self.value_conv1 = nn.Conv2d(in_channels, out_channels, attn_kernels, padding=attn_kernels // 2, bias=False)
        # self.value_conv2 = nn.Conv2d(in_channels, out_channels, attn_kernels, padding=attn_kernels // 2, bias=False)
        self.conv = nn.Conv2d(in_channels, out_channels, attn_kernels, padding=attn_kernels // 2, bias=False)
        self.xor_conv = nn.Conv2d(in_channels, out_channels, attn_kernels, padding=attn_kernels // 2, bias=False)
        self.pos_conv_kernels1 = nn.Parameter(nn.Conv2d(in_channels, out_channels, attn_kernels).weight.data.flatten(1))
        self.pos_conv_kernels2 = nn.Parameter(nn.Conv2d(in_channels, out_channels, attn_kernels).weight.data.flatten(1))
        self.pos_conv_kernels3 = nn.Parameter(nn.Conv2d(out_channels, out_channels, attn_kernels).weight.data.flatten(1))
        # self.unfold1 = nn.Unfold(attn_kernels, stride=attn_kernels)
        # self.unfold2 = nn.Unfold(attn_kernels, stride=attn_kernels)
        self.unfold = nn.Unfold(attn_kernels, padding=attn_kernels // 2)
        self.attn_kernel_size = attn_kernels
        self.in_channels = in_channels
        self.softmax = nn.Softmax(2)
        self.norm_scale = norm_scale
        self.kernel_momentum = nn.Parameter(torch.tensor(0.))
        # self.channel_attn = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False),
        #     nn.Softmax(1),
        # )
        # self.pos_code = nn.Parameter(torch.randn([in_channels] + image_size))
        # self.attn_weight = nn.Parameter(
        #     nn.Conv2d(in_channels, out_channels, attn_kernels, bias=False).weight.data.flatten(1))

    # def generate_attn_kernels(self, x1, x2, key_conv, query_conv, attn_kernels, pos_conv_kernels=None, softmax=False):
    #     flat_kernel_size = attn_kernels * attn_kernels
    #     key = F.unfold(key_conv(x1), attn_kernels, stride=attn_kernels)
    #     key = key.unflatten(1, (-1, flat_kernel_size)).transpose(1, 2)
    #     query = F.unfold(query_conv(x2), attn_kernels, stride=attn_kernels)
    #     query = query.unflatten(1, (-1, flat_kernel_size)).permute([0, 2, 3, 1])
    #     attn_kernls = key.matmul(query).transpose(1, 3).flatten(2) # / sqrt(self.in_channels * flat_kernel_size)
    #     attn_kernls = self.norm_scale * F.layer_norm(attn_kernls, attn_kernls.shape[1:])
    #     if softmax:
    #         attn_kernls = self.softmax(attn_kernls)  # attn_kernls.transpose(1, 2).flatten(1, 2)
    #     if pos_conv_kernels is not None:
    #         attn_kernls = attn_kernls * pos_conv_kernels
    #     attn_value = attn_kernls.matmul(self.unfold(x1))
    #     attn_value = attn_value.unflatten(2, (int(sqrt(attn_value.size(2))), -1))
    #     attn_value = self.norm_scale * F.layer_norm(attn_value, attn_value.shape[1:])
    #     return attn_value

    def forward(self, x1, x2):
        flat_kernel_size = self.attn_kernel_size * self.attn_kernel_size
        key1 = F.unfold(self.key_conv1(x1), self.attn_kernel_size, stride=self.attn_kernel_size)
        key1 = key1.unflatten(1, (-1, flat_kernel_size)).transpose(1, 2)
        # key2 = F.unfold(self.key_conv1(x1), self.attn_kernel_size, stride=self.attn_kernel_size)
        # key2 = key2.unflatten(1, (-1, flat_kernel_size)).transpose(1, 2)
        query1 = F.unfold(self.query_conv1(x1), self.attn_kernel_size, stride=self.attn_kernel_size).unflatten(1, (-1, flat_kernel_size)).permute([0, 2, 3, 1])
        query2 = F.unfold(self.query_conv2(x2), self.attn_kernel_size, stride=self.attn_kernel_size).unflatten(1, (-1, flat_kernel_size)).permute([0, 2, 3, 1])
        attn_kernls1 = key1.matmul(query1).transpose(1, 3).flatten(2) #  / sqrt(self.in_channels * flat_kernel_size)
        attn_kernls2 = key1.matmul(query2).transpose(1, 3).flatten(2)
        attn_kernls = (attn_kernls1 + self.pos_conv_kernels1) * (attn_kernls2 + self.pos_conv_kernels2)
        attn_kernls = self.kernel_momentum * attn_kernls + self.norm_scale * (attn_kernls - attn_kernls.mean()) / (attn_kernls.std() + 1e-4) # self.norm_scale * F.layer_norm(attn_kernls, attn_kernls.shape[1:]) #
        # if self.softmax:
        #     attn_kernls = self.softmax(attn_kernls)  # attn_kernls.transpose(1, 2).flatten(1, 2)
        # if self.pos_conv_kernels is not None:
        #     attn_kernls = attn_kernls * self.pos_conv_kernels
        attn_value = attn_kernls.matmul(self.unfold(x1))
        attn_value = attn_value.unflatten(2, (int(sqrt(attn_value.size(2))), -1))
        # attn_value = self.norm_scale * F.layer_norm(attn_value, attn_value.shape[1:])
        return attn_value


class ButterflyAttention(nn.Module):
    def __init__(self, in_channels, out_channels, norm_scale, kernels):
        super(ButterflyAttention, self).__init__()
        self.attn_conv1 = AttnConv2d(in_channels, out_channels, kernels, norm_scale)
        # self.attn_conv2 = AttnConv2d(in_channels, out_channels, kernels, norm_scale)
        self.norm_scale = norm_scale

    def forward(self, x1, x2):
        y1 = self.attn_conv1(x1, x2)
        y1 = self.norm_scale * ((y1 - y1.mean()) / (y1.std() + 1e-4))
        return y1


class ButterflyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_scale, kernels, cross_xor=False):
        super(ButterflyConv2d, self).__init__()
        self.xor_conv1 = nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False)
        self.xor_conv2 = nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False)
        self.xor_conv3 = nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False)
        self.xor_conv4 = nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False)
        self.proj1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.proj2 = nn.Conv2d(in_channels, out_channels, 1, bias=False) # nn.Linear(in_channels, out_channels, bias=False)
        self.affine_conv = nn.Parameter(nn.Conv2d(in_channels, out_channels, kernels).weight)
        self.affine_bias = nn.Parameter(nn.Conv2d(in_channels, out_channels, kernels).weight)
        self.kernel_momentum = nn.Parameter(torch.tensor(0.))
        self.norm = HuberLayerNorm([1, 2, 3])
        self.norm_scale = norm_scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernels = kernels
        self.cross_xor = cross_xor

    def batch_norm(self, x):
        # x = x.transpose(0, 1)
        x = self.norm_scale * F.layer_norm(x, x.shape[1:])
        # x = x.transpose(0, 1)
        return x

    def cross_attn(self, x1, x2):
        attn1 = self.proj1(x1)
        attn2 = self.proj2(x2)
        # attn2 = self.proj2(attn2.transpose(1, 2)).unsqueeze(-2)
        attn_kernels = attn1.unsqueeze(1) * attn2.unsqueeze(2)
        pad_num = attn_kernels.size(-1) % self.kernels
        if pad_num != 0:
            pad_num = self.kernels - pad_num
            attn_kernels = F.pad(attn_kernels, [0, pad_num, 0, pad_num])
        attn_kernels = attn_kernels.view(attn_kernels.shape[:3] + (self.kernels, self.kernels, -1)).mean([0, -1])
        attn_kernels = self.kernel_momentum * attn_kernels + self.norm_scale * (attn_kernels - attn_kernels.mean()) / (attn_kernels.std() + 1e-4)
        attn_kernels = self.affine_conv * attn_kernels + self.affine_bias
        attn_value = F.conv2d(x1, attn_kernels, padding=self.kernels // 2)
        return attn_value

    def forward(self, x1, x2):
        attn_value = self.cross_attn(x1, x2)
        attn_value = self.norm_scale * ((attn_value - attn_value.mean()) / (attn_value.std() + 1e-4))
        y1 = self.xor_conv1(x1)
        y2 = self.xor_conv2(x1)
        y1 = self.norm_scale * F.layer_norm(y1, y1.shape[1:])
        y2 = self.norm_scale * F.layer_norm(y2, y2.shape[1:])
        y = attn_value + y1 * y2
        if self.cross_xor:
            y3 = self.xor_conv3(x1)
            y4 = self.xor_conv4(x2)
            y3 = self.norm_scale * F.layer_norm(y3, y3.shape[1:])
            y4 = self.norm_scale * F.layer_norm(y4, y4.shape[1:])
            y = y + y3 * y4
        return y


class XorConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_scale, kernels, cross):
        super(XorConv2d, self).__init__()
        self.xor_conv1 = nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False)
        self.xor_conv2 = nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False)
        self.xor_conv3 = nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False)
        self.xor_conv4 = nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False)
        self.xor_conv5 = nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False)
        self.xor_conv6 = nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False)
        self.norm = HuberLayerNorm([1, 2, 3])
        self.norm_scale = norm_scale
        self.kernels = kernels
        self.conv_momentum1 = nn.Parameter(torch.tensor(1.))
        self.conv_momentum2 = nn.Parameter(torch.tensor(1.))
        self.xor_conv_momentum1 = nn.Parameter(torch.tensor(1.))
        self.xor_conv_momentum2 = nn.Parameter(torch.tensor(1.))
        self.xor_conv_momentum3 = nn.Parameter(torch.tensor(1.))
        self.xor_conv_momentum4 = nn.Parameter(torch.tensor(1.))
        self.xor_conv_momentum5 = nn.Parameter(torch.tensor(1.))
        self.xor_conv_momentum6 = nn.Parameter(torch.tensor(1.))
        self.out_channels = out_channels
        self.kernels = kernels
        self.norm1 = nn.BatchNorm2d(out_channels, affine=False)
        self.norm2 = nn.BatchNorm2d(out_channels, affine=False)
        self.xor_norm1 = nn.BatchNorm2d(out_channels, affine=False)
        self.xor_norm2 = nn.BatchNorm2d(out_channels, affine=False)
        self.xor_norm3 = nn.BatchNorm2d(out_channels, affine=False)
        self.xor_norm4 = nn.BatchNorm2d(out_channels, affine=False)
        self.xor_norm5 = nn.BatchNorm2d(out_channels, affine=False)
        self.xor_norm6 = nn.BatchNorm2d(out_channels, affine=False)
        self.cross = cross

    def full_norm(self, x, conv_momentum):
        x = 0.5 * conv_momentum * x + self.norm_scale * (x - x.mean()) / (x.std() + 1e-4) # conv_momentum * x +  conv_momentum * x +
        return x

    def res_layer_norm(self, x, conv_momentum):
        # x = x.transpose(0, 1)
        x = conv_momentum * x + self.norm_scale * F.layer_norm(x, x.shape[1:]) # F.layer_norm(x, x.shape[1:])) # conv_momentum * x +  (1 / sqrt(self.out_channels) / self.kernels) * conv_momentum
        # x = x.transpose(0, 1)
        return x

    def forward(self, x1, x2):
        single_y1 = self.conv1(x1)
        single_y2 = self.conv2(x2)
        single_y1 = self.res_layer_norm(single_y1, self.conv_momentum1)
        single_y2 = self.res_layer_norm(single_y2, self.conv_momentum1)
        y1 = self.xor_conv1(x1)
        y2 = self.xor_conv2(x1)
        y3 = self.xor_conv3(x2)
        y4 = self.xor_conv4(x2)
        y1_xor = y1 * y2
        y2_xor = y3 * y4
        y1_xor = self.res_layer_norm(y1_xor, self.conv_momentum1)
        y2_xor = self.res_layer_norm(y2_xor, self.conv_momentum1)
        y5 = self.xor_conv5(x2)
        y6 = self.xor_conv6(x1)
        y3_xor = y5 * y6
        y3_xor = self.res_layer_norm(y3_xor, self.conv_momentum1)
        return single_y1 + single_y2 + y1_xor + y2_xor + y3_xor


class MetaResConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_scale, kernels, cross):
        super(MetaResConv2d, self).__init__()
        if in_channels != out_channels:
            self.conv_momentum1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.conv_momentum2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.conv_momentum1 = nn.Parameter(torch.tensor(1.))
            self.conv_momentum2 = nn.Parameter(torch.tensor(1.))
        self.mean_split = ReLUSplitNorm(in_channels, norm_scale)
        self.conv1 = XorConv2d(in_channels, out_channels, norm_scale, kernels, cross)
        # self.conv2 = ButterflyConv2d(in_channels, out_channels, norm_scale, kernels)
        self.norm = HuberLayerNorm([1, 2, 3])
        self.norm_scale = norm_scale

    def forward(self, x):
        x1, x2, x1_norm, x2_norm = self.mean_split(x)
        y = self.conv1(x1_norm, x2_norm)
        if type(self.conv_momentum1) is nn.parameter.Parameter:
            x1 = self.conv_momentum1 * x1
            x2 = self.conv_momentum2 * x2
        else:
            x1 = self.conv_momentum1(x1)
            x2 = self.conv_momentum2(x2)
        # if type(self.conv_momentum1) is not nn.parameter.Parameter:
        #     x3 = self.conv_momentum1(x3)
        return x1 + x2 + y


class MetaResNet(nn.Module):
    def __init__(self, num_layers, init_channels, kernel_size=3, norm_scale=0.1773, image_sizes=None, dropout=True):
        super(MetaResNet, self).__init__()
        num_hidden_layers = int(torch.tensor([i[1] for i in num_layers]).sum().item())
        dropouts = [0.5] * (len(num_layers) - 1) + [0.5] if dropout else None
        alpha = 1. * torch.ones(num_hidden_layers)  # torch.sort(torch.cat([torch.full((1,), 1.), 0.905 + 0.09 * torch.rand(num_hidden_layers - 2), torch.full((1,), 0.9)]), descending=True).values #
        beta = 0. * torch.ones(num_hidden_layers + len(num_layers)) # torch.sort(torch.cat([torch.full((1,), 1.), 0.905 + 0.09 * torch.rand(num_hidden_layers + len(num_layers) - 3), torch.full((1,), 0.9)]), descending=True).values #
        layers = [MetaResConv2d(init_channels, num_layers[0][0], norm_scale, kernel_size, False)]
        xor = True
        k = -1
        alpha_id, beta_id = 0, 1
        cross = True
        # prob = True
        for n, (i, j) in enumerate(num_layers):
            if k > 0 and k != i:
                layers.append(MetaResConv2d(k, i, norm_scale, kernel_size, cross))
                beta_id += 1
                cross = ~cross
            for id in range(j):
                layers.append(MetaResConv2d(i, i, norm_scale, kernel_size, cross)) # id % 2 == 1
                cross = ~cross
                xor = ~xor
                alpha_id += 1
                beta_id += 1
            layers.append(nn.AvgPool2d(2))
            if image_sizes is not None:
                image_sizes[0] //= 2
                image_sizes[1] //= 2
            if dropouts is not None:
                layers.append(nn.Dropout(dropouts[n]))
            k = i
            # prob = not prob
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



