from math import sqrt, log, ceil, pi, inf, nan
import torch
import torch.nn as nn
import torch.nn.functional as F


def softxor(x, y, eps=1e-3):
    y = x.tanh() * y.tanh()
    return 0.5 * torch.log((1 + eps + y) / (1 + eps - y))


def quadratic_softxor(x, y, eps=1e-5):
    x /= (1 + x ** 2).sqrt()
    y /= (1 + y ** 2).sqrt()
    y *= x
    return y / (1 + eps - y ** 2).sqrt()


class Flipout(nn.Module):
    def __init__(self, p=0.25):
        super(Flipout, self).__init__()
        assert 0 < p < 0.1816, "p should be smaller than 0.1816!"
        self.p = p
        self.flip_scale = 1. / (1. - 2. * p)

    def forward(self, x):
        if self.training:
            x[torch.rand_like(x) < self.p] *= -1
            return self.flip_scale * x
        return x


class ResAttention2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResAttention2d, self).__init__()
        self.channel_attn = nn.Sequential(
            ResBatchNorm2d(in_channels, 0.1816, bias=False, short_cut=False),
            nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False),
            nn.Softmax(1),
        )
        self.bn = nn.Sequential(
            ResBatchNorm2d(in_channels, 0.1816, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, affine=False),
        )
        if in_channels == out_channels:
            self.short_cut = nn.Parameter(torch.tensor(1.))
        else:
            self.short_cut = None # nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        # x = torch.roll(x, (x.size(-2) // 2, x.size(-1) // 2), (-2, -1))
        y = self.channel_attn(x) * self.bn(x)
        if self.short_cut is not None:
            y += 0.1816 * self.short_cut * x # if isinstance(self.short_cut, torch.Tensor) else self.short_cut(x)
        return y


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
    def __init__(self, in_channels, norm_scale, bias=False, short_cut=False):
        super(ResBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels, affine=bias) # nn.BatchNorm2d(channels, affine=True) # FullNorm() # SphereNorm2d(channels)  #
        self.norm_scale = norm_scale # nn.Parameter(torch.tensor(norm_scale)) # norm_scale
        self.alpha = nn.Parameter(torch.tensor(1.)) if short_cut else None

    def forward(self, x):
        y = self.norm_scale * self.bn(x)
        if self.alpha is not None:
            y += .5 * self.alpha * x
        return y # self.norm_scale * y  # self.norm_scale * (self.norm_momentum * x + self.bn(x)) # self.norm_scale * (self.norm_momentum * x + F.layer_norm(x, x.shape[1:])) #


class ResLayerNorm2d(nn.Module):
    def __init__(self, image_size, norm_scale, bias=False, short_cut=False):
        super(ResLayerNorm2d, self).__init__()
        self.ln = nn.LayerNorm(image_size, elementwise_affine=bias) # nn.BatchNorm2d(channels, affine=True) # FullNorm() # SphereNorm2d(channels)  #
        self.norm_scale = norm_scale # nn.Parameter(torch.tensor(norm_scale)) #
        self.alpha = nn.Parameter(torch.tensor(1.)) if short_cut else None

    def forward(self, x):
        y = self.norm_scale * self.ln(x)
        if self.alpha is not None:
            y += .5 * self.alpha * x
        return y # self.norm_scale * y


class ResConv2d(nn.Module):
    def __init__(self, image_size, in_channels, out_channels, kernels=3, conv_momentum=None, num_heads=1, batch_norm=True, xor=True):
        super(ResConv2d, self).__init__()
        if in_channels != out_channels:
            self.conv_momentum = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.conv_momentum = nn.Parameter(torch.tensor(1. if conv_momentum is None else conv_momentum))
        if '__getitem__' in dir(num_heads):
            self.num_heads = num_heads
        else:
            self.num_heads = (num_heads, num_heads)
        # self.alpha = nn.Parameter(torch.tensor(0.))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False)
        self.conv2 = nn.Conv2d(in_channels // 2 if in_channels >= 10 else in_channels, out_channels, kernels, padding=kernels // 2, bias=False)
        self.conv3 = nn.Conv2d(in_channels // 2 if in_channels >= 10 else in_channels, out_channels, kernels, padding=kernels // 2, bias=False)
        if batch_norm:
            self.norm1 = ResBatchNorm2d(out_channels, 0.1816, short_cut=False)
            self.norm2 = ResBatchNorm2d(out_channels, 0.1816, short_cut=False)
            self.norm3 = ResBatchNorm2d(out_channels, 0.1816, short_cut=False)
        else:
            self.norm1 = ResLayerNorm2d(image_size, 0.1816, short_cut=False)
            self.norm2 = ResLayerNorm2d(image_size, 0.1816, short_cut=False)
            self.norm3 = ResLayerNorm2d(image_size, 0.1816, short_cut=False)
        self.batch_norm = batch_norm
        self.xor = xor
        self.dropout = nn.Dropout()
        if self.num_heads[0] * self.num_heads[1] == 1:
            self.multi_head_downsampling = None
        else:
            self.multi_head_downsampling = nn.Unfold(self.num_heads, stride=self.num_heads) # nn.AvgPool2d(self.num_heads)
        # self.conv.weight.data /= 10

    def feature_map_stack(self, y):
        if self.multi_head_downsampling is not None:
            h, w = y.shape[-2:]
            y = self.multi_head_downsampling(y)
            y = y.unflatten(1, (-1, self.num_heads[0] * self.num_heads[1], self.num_heads[0] * self.num_heads[1]))
            y = y.transpose(2, 3).contiguous()
            y = y.view((y.size(0), -1) + self.num_heads + (int(sqrt(y.size(-1))), int(sqrt(y.size(-1)))))
        # if self.multi_head_downsampling is not None:
        #     h, w = y.shape[-2:]
        #     y = self.multi_head_downsampling(y).unflatten(1, (-1,) + self.num_heads)
            y = y.transpose(3, 4).contiguous()
            y = y.view(y.shape[:2] + (y.shape[2] * y.shape[3], -1))
            if y.size(2) * y.size(3) < h * w:
                h1, w1 = (h - y.size(2)) // 2, (w - y.size(3)) // 2
                h2, w2 = h - y.size(2) - h1, w - y.size(3) - w1
                y = F.pad(y, (w1, w2, h1, h2))
        return y

    def forward(self, x):
        # y = self.conv2(x)
        # if self.dropout is not None:
        #     y = self.dropout(y)
        y1 = self.conv1(x)
        if x.size(1) >= 10:
            x1, x2 = x.chunk(2, 1)
            y2 = self.conv2(x1)
            y3 = self.conv3(x1)
        else:
            y2 = self.conv2(x)
            y3 = self.conv3(x)
        # y4 = self.conv4(x)
        x = self.conv_momentum * x if type(self.conv_momentum) is nn.parameter.Parameter else self.conv_momentum(x)
        y = self.norm1(self.feature_map_stack(y1)) + self.norm2(self.feature_map_stack(y2)) * self.norm3(self.feature_map_stack(y3)) if self.xor else self.norm1(self.feature_map_stack(y1)) # softxor(self.bn1(y2), self.bn2(y3))
        return x + y


class AttnConv2d(nn.Module):
    def __init__(self, image_size, in_channels, out_channels, attn_kernels=3, kernels=3, conv_momentum=None,
                 num_heads=1, batch_norm=True):
        super(AttnConv2d, self).__init__()
        if in_channels == out_channels:
            self.conv_momentum = nn.Parameter(torch.tensor(1. if conv_momentum is None else conv_momentum))
        else:
            self.conv_momentum = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if '__getitem__' in dir(num_heads):
            self.num_heads = num_heads
        else:
            self.num_heads = (num_heads, num_heads)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernels, padding=kernels // 2, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernels, padding=kernels // 2, bias=False)
        self.unfold1 = nn.Unfold(attn_kernels, stride=attn_kernels)
        self.unfold2 = nn.Unfold(attn_kernels, stride=attn_kernels)
        self.unfold3 = nn.Unfold(attn_kernels, padding=attn_kernels // 2)
        self.norm = ResBatchNorm2d(out_channels, 0.1816, short_cut=False) if batch_norm else ResLayerNorm2d(image_size, 0.1816, short_cut=False)
        self.attn_kernels = attn_kernels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.softmax = nn.Softmax(2)
        # self.attn_value_kernel = nn.Parameter(0.1816 * torch.randn(out_channels, in_channels, attn_kernels, attn_kernels))
        self.dropout = nn.Dropout()
        if self.num_heads[0] * self.num_heads[1] == 1:
            self.multi_head_downsampling = None
        else:
            self.multi_head_downsampling = nn.Unfold(self.num_heads, stride=self.num_heads) # nn.AvgPool2d(self.num_heads)

    def feature_map_stack(self, attn_value):
        if self.multi_head_downsampling is not None:
            h, w = attn_value.shape[-2:]
            attn_value = self.multi_head_downsampling(attn_value).unflatten(1, (
            -1, self.num_heads[0] * self.num_heads[1], self.num_heads[0] * self.num_heads[1]))
            attn_value = attn_value.transpose(2, 3).contiguous()
            attn_value = attn_value.view(attn_value.size(0), -1, self.num_heads[0], self.num_heads[1],
                                         int(sqrt(attn_value.size(-1))), int(sqrt(attn_value.size(-1))))
            # if self.multi_head_downsampling is not None:
            #     h, w = attn_value.shape[-2:]
            #     attn_value = self.multi_head_downsampling(attn_value).unflatten(1, (-1,) + self.num_heads)
            attn_value = attn_value.transpose(3, 4).contiguous()
            attn_value = attn_value.view(attn_value.shape[:2] + (attn_value.shape[2] * attn_value.shape[3], -1))
            if attn_value.size(2) * attn_value.size(3) < h * w:
                h1, w1 = (h - attn_value.size(2)) // 2, (w - attn_value.size(3)) // 2
                h2, w2 = h - attn_value.size(2) - h1, w - attn_value.size(3) - w1
                attn_value = F.pad(attn_value, (w1, w2, h1, h2))
        return attn_value

    def forward(self, x):
        y1 = self.unfold1(self.conv1(x))
        y1 = y1.unflatten(1, (-1, self.attn_kernels * self.attn_kernels)).transpose(1, 2)
        # y1 = y1.unflatten(1, (self.attn_kernels * self.attn_kernels, -1))
        y2 = self.unfold2(self.conv2(x))
        y2 = y2.unflatten(1, (-1, self.attn_kernels * self.attn_kernels)).permute([0, 2, 3, 1])
        # y2 = y2.unflatten(1, (self.attn_kernels * self.attn_kernels, -1))
        attn_kernls = y1.matmul(y2)
        # attn_kernls = y1.matmul(y2.transpose(2, 3))
        attn_kernls = self.softmax(attn_kernls.transpose(1, 3).flatten(2) / sqrt(self.in_channels * self.attn_kernels * self.attn_kernels)) # attn_kernls.transpose(1, 2).flatten(1, 2)
        attn_value = attn_kernls.matmul(self.unfold3(self.conv3(x)))
        # attn_value = self.unfold3(x).transpose(1, 2).matmul(attn_kernls).transpose(1, 2)
        attn_value = attn_value.unflatten(2, (int(sqrt(attn_value.size(2))), -1))
        # x = self.conv_momentum * x if type(self.conv_momentum) is nn.parameter.Parameter else self.conv_momentum(x)
        if x.size() == attn_value.size():
            return self.conv_momentum * x + self.norm(self.feature_map_stack(attn_value))
        else:
            return self.conv_momentum(x) + self.norm(self.feature_map_stack(attn_value))


# class AttnConv2d(nn.Module):
#     def __init__(self, image_size, in_channels, out_channels, attn_kernels=3, kernels=3, conv_momentum=None, batch_norm=True):
#         super(AttnConv2d, self).__init__()
#         if in_channels != out_channels:
#             self.conv_momentum = nn.Conv2d(in_channels, out_channels, 1, bias=False)
#         else:
#             self.conv_momentum = nn.Parameter(torch.tensor(1. if conv_momentum is None else conv_momentum))
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernels, padding=kernels // 2, bias=False)
#         self.conv2 = nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False)
#         self.unfold1 = nn.Unfold(attn_kernels, stride=attn_kernels)
#         self.unfold2 = nn.Unfold(attn_kernels, stride=attn_kernels)
#         # self.prenorm1 = ResLayerNorm2d(image_size, 0.1816, bias=False, short_cut=True)
#         # self.prenorm2 = ResLayerNorm2d(image_size, 0.1816, bias=False, short_cut=True)
#         # self.prenorm3 = ResLayerNorm2d(image_size, 0.1816, bias=False, short_cut=True)
#         # if batch_norm:
#         #     self.norm1 = ResBatchNorm2d(in_channels, 0.1816, short_cut=False)
#         #     self.norm2 = ResBatchNorm2d(out_channels, 0.1816, short_cut=False)
#         #     self.norm3 = ResBatchNorm2d(out_channels, 0.1816, short_cut=True)
#         # else:
#         #     self.norm1 = ResLayerNorm2d(image_size, 0.1816, short_cut=False)
#         #     self.norm2 = ResLayerNorm2d(image_size, 0.1816, short_cut=False)
#         #     self.norm3 = ResLayerNorm2d(image_size, 0.1816, short_cut=True)
#         self.norm = ResLayerNorm2d(image_size, 0.1816, short_cut=False)
#         self.batch_norm = batch_norm
#         self.attn_kernels = attn_kernels
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.softmax = nn.Softmax(0)
#         self.attn_value_kernel = nn.Parameter(0.1816 * torch.randn(out_channels, in_channels, attn_kernels, attn_kernels))
#         self.dropout = nn.Dropout()
#
#     def forward(self, x):
#         y1 = self.unfold1(self.conv1(x))
#         y1 = y1.unflatten(1, (-1, self.attn_kernels * self.attn_kernels)).transpose(0, 2).flatten(2) #.flatten(0, 1)
#         # y1 = y1.view(y1.size(0), -1, y1.size(2) * self.attn_kernels).transpose(0, 1).flatten(1)
#         y2 = self.unfold2(self.conv2(x))
#         y2 = y2.unflatten(1, (-1, self.attn_kernels * self.attn_kernels)).transpose(0, 2).flatten(2) #.flatten(0, 1)
#         # y2 = y2.view(y2.size(0), -1, y2.size(2) * self.attn_kernels).transpose(0, 1).flatten(1)
#         attn_kernls = y1.matmul(y2.transpose(-2, -1))
#         # attn_kernls = attn_kernls.mean(0)
#         # attn_kernls = self.softmax(attn_kernls.view(-1, self.attn_kernels * self.attn_kernels * self.in_channels))
#         # attn_kernls = self.softmax(attn_kernls.transpose(0, 1).flatten(1))
#
#         # attn_kernls = self.softmax(attn_kernls.transpose(0, 1).flatten(1))
#         # attn_kernls = self.softmax(attn_kernls.transpose(0, 1) / sqrt(self.in_channels)) # .flatten(0, 1) * self.attn_kernels * self.attn_kernels
#         attn_kernls = self.softmax(attn_kernls.flatten(0, 1) / sqrt(self.in_channels * self.attn_kernels * self.attn_kernels)) # .transpose(0, 1) * self.attn_kernels * self.attn_kernels
#         # attn_kernls = attn_kernls.permute([2, 0, 1]).unflatten(-1, (self.attn_kernels, -1))
#         attn_kernls = attn_kernls.unflatten(0, (self.attn_kernels * self.attn_kernels, -1)).transpose(0, 2).unflatten(-1, (self.attn_kernels, -1))#  .transpose(0, 1) # contiguous()..view(-1, self.in_channels, self.attn_kernels, self.attn_kernels)
#         # attn_kernls *= self.attn_value_kernel
#         # x = self.prenorm3(x)
#         attn_value = F.conv2d(x, attn_kernls, padding=self.attn_kernels // 2)
#         x = self.conv_momentum * x if type(self.conv_momentum) is nn.parameter.Parameter else self.conv_momentum(x)
#         return x + self.norm(attn_value)


# class XorAttnConv2d(nn.Module):
#     def __init__(self, image_size, in_channels, out_channels, attn_kernels=3, kernels=3, conv_momentum=None, batch_norm=True):
#         super(XorAttnConv2d, self).__init__()
#         if in_channels != out_channels:
#             self.conv_momentum = nn.Conv2d(in_channels, out_channels, 1, bias=False)
#         else:
#             self.conv_momentum = nn.Parameter(torch.tensor(1. if conv_momentum is None else conv_momentum))
#         self.attn_conv1 = AttnConv2d(image_size, in_channels, out_channels, attn_kernels, kernels, conv_momentum, batch_norm)
#         self.attn_conv2 = AttnConv2d(image_size, in_channels, out_channels, attn_kernels, kernels, conv_momentum,
#                                      batch_norm)
#         self.attn_conv3 = AttnConv2d(image_size, in_channels, out_channels, attn_kernels, kernels, conv_momentum,
#                                      batch_norm)
#         self.norm1 = ResLayerNorm2d(image_size, 0.1816, short_cut=False)
#         self.norm2 = ResLayerNorm2d(image_size, 0.1816, short_cut=False)
#         self.norm3 = ResLayerNorm2d(image_size, 0.1816, short_cut=False)
#
#     def forward(self, x):
#         y1 = self.attn_conv1(x)
#         y2 = self.attn_conv2(x)
#         y3 = self.attn_conv3(x)
#         x = self.conv_momentum * x if type(self.conv_momentum) is nn.parameter.Parameter else self.conv_momentum(x)
#         return x + self.norm1(y1) + self.norm2(y2) * self.norm3(y3)


class ResReLU(nn.Module):
    def __init__(self, relu_neg_momentum=0.01, reverse=False):
        super(ResReLU, self).__init__()
        # torch.manual_seed(torch.randint(0, 32767, (1,)).item())
        self.relu = nn.ReLU()
        self.relu_neg_momentum = nn.Parameter(torch.tensor(relu_neg_momentum))
        self.reverse = reverse

    def forward(self, x):
        # if self.alpha < 0:
        #     self.alpha.data = torch.tensor(0.).cuda()
        # if self.beta < 0:
        #     self.beta.data = torch.tensor(0.).cuda()
        # if self.beta > 2:
        #     self.beta.data = torch.tensor(2.).cuda()
        return self.relu_neg_momentum * x + (-self.relu(-x) if self.reverse else self.relu(x))


class MetaResConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_scale, kernel_size, relu_neg_momentum, image_size, conv_momentum=None, has_attn=False):
        super(MetaResConv2d, self).__init__()
        # self.layer0 = nn.Sequential(
        #     # ResBatchNorm2d(in_channels, norm_scale, short_cut=True),
        #     ResLayerNorm2d(image_size, norm_scale, bias=False, short_cut=True),
        #     ResConv2d(image_size, in_channels, out_channels, kernel_size, conv_momentum, batch_norm=False, xor=True),
        #     ResReLU(relu_neg_momentum, reverse=False)
        # )
        self.layer1 = nn.Sequential(
            ResBatchNorm2d(in_channels, norm_scale, bias=False, short_cut=True),
            # ResLayerNorm2d(image_size, norm_scale, bias=False, short_cut=True),
            ResConv2d(image_size, in_channels, out_channels, kernel_size, conv_momentum, num_heads=2, batch_norm=True, xor=True),
            ResBatchNorm2d(out_channels, norm_scale, bias=False, short_cut=True),
            ResConv2d(image_size, out_channels, out_channels, kernel_size, conv_momentum, num_heads=1, batch_norm=True,
                      xor=True),
            ResReLU(relu_neg_momentum, reverse=False),
            # ResBatchNorm2d(out_channels, norm_scale, short_cut=True),
            # # ResLayerNorm2d(image_size, norm_scale, bias=False, short_cut=True),
            # ResConv2d(image_size, out_channels, out_channels, kernel_size, conv_momentum, batch_norm=True, xor=False),
            # ResReLU(relu_neg_momentum, reverse=False),
        )
        self.layer2 = nn.Sequential(
            # ResLayerNorm2d(image_size, norm_scale, bias=False, short_cut=True),
            # ResConv2d(image_size, in_channels, out_channels, kernel_size, conv_momentum, batch_norm=False),
            # ResReLU(relu_neg_momentum, reverse=True)
            ResBatchNorm2d(in_channels, norm_scale, bias=False, short_cut=True),
            # ResLayerNorm2d(image_size, norm_scale, bias=False, short_cut=True),
            AttnConv2d(image_size, in_channels, out_channels, kernel_size, kernel_size, conv_momentum, num_heads=2,
                       batch_norm=True),
            ResBatchNorm2d(out_channels, norm_scale, bias=False, short_cut=True),
            ResConv2d(image_size, out_channels, out_channels, kernel_size, conv_momentum, num_heads=1, batch_norm=True,
                      xor=False),
            ResReLU(relu_neg_momentum, reverse=True),
            # ResBatchNorm2d(out_channels, norm_scale, short_cut=True),
            # # ResLayerNorm2d(image_size, norm_scale, bias=False, short_cut=True),
            # AttnConv2d(image_size, out_channels, out_channels, kernel_size, kernel_size, conv_momentum,
            #            batch_norm=True),
            # ResReLU(relu_neg_momentum, reverse=True)
        )
        # self.relu = ResReLU(relu_neg_momentum)
        # self.drop_out = nn.Dropout()

    def forward(self, x):
        return self.layer1(x) + self.layer2(x)


class MetaResNet(nn.Module):
    def __init__(self, num_layers, init_channels, kernel_size=3, norm_scale=0.1816, image_sizes=None, dropout=True):
        super(MetaResNet, self).__init__()
        num_hidden_layers = int(torch.tensor([i[1] for i in num_layers]).sum().item())
        dropouts = [0.1816] * (len(num_layers) - 1) + [0.1816] if dropout else None
        alpha = 1. * torch.ones(num_hidden_layers)  # torch.sort(torch.cat([torch.full((1,), 1.), 0.905 + 0.09 * torch.rand(num_hidden_layers - 2), torch.full((1,), 0.9)]), descending=True).values #
        beta = 0. * torch.ones(num_hidden_layers + len(num_layers)) # torch.sort(torch.cat([torch.full((1,), 1.), 0.905 + 0.09 * torch.rand(num_hidden_layers + len(num_layers) - 3), torch.full((1,), 0.9)]), descending=True).values #
        layers = [MetaResConv2d(init_channels, num_layers[0][0], norm_scale, kernel_size, beta[0].item(), image_size=image_sizes)] # has_attn=False
        xor = True
        k = -1
        alpha_id, beta_id = 0, 1
        # prob = True
        for n, (i, j) in enumerate(num_layers):
            if k > 0 and k != i:
                layers.append(MetaResConv2d(k, i, norm_scale, kernel_size, beta[beta_id].item(), image_size=image_sizes, has_attn=xor)) # , has_attn=n % 2 == 0
                xor = ~xor
                beta_id += 1
            for id in range(j):
                layers.append(MetaResConv2d(i, i, norm_scale, kernel_size, beta[beta_id].item(), image_sizes, alpha[alpha_id].item(), has_attn=xor)) # id % 2 == 1
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



