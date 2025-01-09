from math import sqrt, log, ceil, pi, inf, nan
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, image_size, norm_scale, bias=False, short_cut=None):
        super(ResLayerNorm2d, self).__init__()
        self.ln = nn.LayerNorm(image_size, elementwise_affine=bias) # nn.BatchNorm2d(channels, affine=True) # FullNorm() # SphereNorm2d(channels)  #
        self.norm_scale = norm_scale # nn.Parameter(torch.tensor(norm_scale)) #
        self.alpha = nn.Parameter(torch.tensor(1.)) if short_cut is not None else None
        self.shortcut = short_cut if short_cut is not None else None

    def forward(self, x):
        y = self.norm_scale * self.bn(x)
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


class PreReLUResBatchNorm2d(nn.Module):
    def __init__(self, in_channels, norm_scale, bias=False, short_cut=None, relu_neg_momentum=0.):
        super(PreReLUResBatchNorm2d, self).__init__()
        self.norm1 = ResBatchNorm2d(in_channels, norm_scale, bias, short_cut)
        self.norm2 = ResBatchNorm2d(in_channels, norm_scale, bias, short_cut)
        self.relu1 = ResReLU(relu_neg_momentum, reverse=False)
        self.relu2 = ResReLU(relu_neg_momentum, reverse=True)
    def forward(self, x):
        y1 = self.norm1(self.relu1(x))
        y2 = self.norm2(self.relu2(x))
        return y1, y2


class AttnConv2d(nn.Module):
    def __init__(self, image_size, in_channels, out_channels, hidden_kernels, attn_kernels):
        super(AttnConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, hidden_kernels, padding=hidden_kernels // 2, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, hidden_kernels, padding=hidden_kernels // 2, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, hidden_kernels, padding=hidden_kernels // 2, bias=False)
        self.unfold1 = nn.Unfold(attn_kernels, stride=attn_kernels)
        self.unfold2 = nn.Unfold(attn_kernels, stride=attn_kernels)
        self.unfold3 = nn.Unfold(attn_kernels, padding=attn_kernels // 2)
        self.attn_kernels = attn_kernels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.softmax = nn.Softmax(2)
        # self.pos_code = nn.Parameter(torch.randn([in_channels] + image_size))
        # self.attn_weight = nn.Parameter(
        #     nn.Conv2d(in_channels, out_channels, attn_kernels, bias=False).weight.data.flatten(1))

    def forward(self, x):
        # print(x.shape, self.conv1(x).shape, self.unfold1.kernel_size, self.unfold1.stride)
        # x_pos = x + self.pos_code
        y1 = self.unfold1(self.conv1(x))
        # print(y1.shape)
        y1 = y1.unflatten(1, (-1, self.attn_kernels * self.attn_kernels)).transpose(1, 2)
        # print(y1.shape)
        # y1 = y1.unflatten(1, (self.attn_kernels * self.attn_kernels, -1))
        # print(x.shape, self.conv1(x).shape)
        y2 = self.unfold2(self.conv2(x))
        # print(y2.shape)
        y2 = y2.unflatten(1, (-1, self.attn_kernels * self.attn_kernels)).permute([0, 2, 3, 1])
        # print(y2.shape)
        # y2 = y2.unflatten(1, (self.attn_kernels * self.attn_kernels, -1))
        attn_kernls = y1.matmul(y2).transpose(1, 3).flatten(2) / sqrt(
            self.out_channels * self.attn_kernels * self.attn_kernels)  # + self.attn_weight / self.attn_kernels
        # print(attn_kernls.shape)
        attn_kernls = self.softmax(attn_kernls)  # attn_kernls.transpose(1, 2).flatten(1, 2)
            # attn_kernls = attn_kernls.unflatten(2, (-1, self.attn_kernels * self.attn_kernels))
        # attn_kernls = attn_kernls * self.attn_weight
        attn_value = attn_kernls.matmul(self.unfold3(self.conv3(x)))
        attn_value = attn_value.unflatten(2, (int(sqrt(attn_value.size(2))), -1))
        return attn_value


class ButterflyGatingUnit(nn.Module):
    def __init__(self, image_size, in_channels, norm_scale, hidden_kernels, kernels):
        super(ButterflyGatingUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernels, padding=kernels // 2, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernels, padding=kernels // 2, bias=False)
        self.conv3 = AttnConv2d(image_size, in_channels * 2, in_channels, hidden_kernels, kernels)
        self.norm1 = ResBatchNorm2d(in_channels, norm_scale, short_cut=None)
        self.norm2 = ResBatchNorm2d(in_channels, norm_scale, short_cut=None)
        self.norm3 = ResBatchNorm2d(in_channels, norm_scale, short_cut=None)
        self.conv_momentum = nn.Parameter(torch.tensor(1.))

    def forward(self, x1, x2):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        x = torch.cat([x1, x2], 1)
        y3 = self.conv3(x)
        y = torch.cat([self.norm1(y1) * self.norm2(y2), self.norm3(y3)], 1)
        return x + y, x # self.conv_momentum * x +


class ResConv2d(nn.Module):
    def __init__(self, image_size, in_channels, out_channels, norm_scale, hidden_kernels=3, kernels=3,
                     conv_momentum=None,
                     num_heads=1, batch_norm=True, xor=True):
        super(ResConv2d, self).__init__()
        if in_channels != out_channels:
            self.conv_momentum1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.conv_momentum1 = nn.Conv2d(in_channels, out_channels, 1, bias=False) # nn.Parameter(torch.tensor(1.)) # nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if in_channels * 2 != out_channels:
            self.conv_momentum2 = nn.Conv2d(in_channels * 2, out_channels, 1, bias=False)
        else:
            self.conv_momentum2 = nn.Parameter(torch.tensor(1.)) # nn.Conv2d(in_channels * 2, out_channels, 1, bias=False) #
        if '__getitem__' in dir(num_heads):
            self.num_heads = num_heads
        else:
            self.num_heads = (num_heads, num_heads)
        self.pre_relu_norm = PreReLUResBatchNorm2d(in_channels, norm_scale, False, 0.5, 0.)
        self.conv = ButterflyGatingUnit(image_size, in_channels, norm_scale, hidden_kernels, kernels)
        self.full_conv = nn.Conv2d(in_channels * 2, out_channels, kernels, padding=kernels // 2, bias=False)
        self.norm_scale = norm_scale
        self.batch_norm = batch_norm
        self.xor = xor
        self.dropout = nn.Dropout()
        self.pre_relu = ResReLU(reverse=False)
        self.in_channels = in_channels
        if self.num_heads[0] * self.num_heads[1] == 1:
            self.multi_head_downsampling = None
        else:
            self.multi_head_downsampling = nn.Unfold(self.num_heads, stride=self.num_heads) # nn.AvgPool2d(self.num_heads)

    # def feature_map_stack(self, y):
    #     if self.multi_head_downsampling is not None:
    #         h, w = y.shape[-2:]
    #         y = self.multi_head_downsampling(y)
    #         y = y.unflatten(1, (-1, self.num_heads[0] * self.num_heads[1], self.num_heads[0] * self.num_heads[1]))
    #         y = y.transpose(2, 3).contiguous()
    #         y = y.view((y.size(0), -1) + self.num_heads + (int(sqrt(y.size(-1))), int(sqrt(y.size(-1)))))
    #     # if self.multi_head_downsampling is not None:
    #     #     h, w = y.shape[-2:]
    #     #     y = self.multi_head_downsampling(y).unflatten(1, (-1,) + self.num_heads)
    #         y = y.transpose(3, 4).contiguous()
    #         y = y.view(y.shape[:2] + (y.shape[2] * y.shape[3], -1))
    #         if y.size(2) * y.size(3) < h * w:
    #             h1, w1 = (h - y.size(2)) // 2, (w - y.size(3)) // 2
    #             h2, w2 = h - y.size(2) - h1, w - y.size(3) - w1
    #             y = F.pad(y, (w1, w2, h1, h2))
    #     return y

    def forward(self, x):
        y1, y2 = self.pre_relu_norm(x)
        y, _ = self.conv(y1, y2) #  + self.pos_code
        y_sum = y1 + y2
        x1 = self.conv_momentum1(y_sum) # 0.618034 * self.conv_momentum1 * y_sum if type(self.conv_momentum1) is nn.parameter.Parameter else
        # x2 = y if type(self.conv_momentum2) is nn.parameter.Parameter else self.conv_momentum2(y) # self.conv_momentum2 *
        return x1 + self.full_conv(y) # self.feature_map_stack(y)


# class MetaResConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, norm_scale, kernel_size, relu_neg_momentum, conv_momentum=1., has_attn=False, image_size=None):
#         super(MetaResConv2d, self).__init__()
#         # if in_channels != out_channels:
#         #     self.conv_momentum = nn.Conv2d(in_channels, out_channels, 1, bias=False)
#         # else:
#         #     self.conv_momentum = nn.Parameter(torch.tensor(1.))
#         # self.alpha = conv_momentum
#         self.layer1 = nn.Sequential(
#             # ResBatchNorm2d(in_channels, norm_scale, bias=False, short_cut=True),
#             # ResLayerNorm2d(image_size, norm_scale, bias=False, short_cut=True),
#             # ResReLU(relu_neg_momentum, reverse=False),
#             # ResConv2d(image_size, in_channels, out_channels, norm_scale, kernel_size, conv_momentum, num_heads=2, pre_norm=True, batch_norm=True, xor=True),
#             # ResBatchNorm2d(out_channels, norm_scale, bias=False, short_cut=True),
#             # ResConv2d(image_size, out_channels, out_channels, norm_scale, kernel_size, conv_momentum, num_heads=1, pre_norm=True, batch_norm=True,
#             #          xor=True),
#             # PreReLUResBatchNorm2d(in_channels, norm_scale, False, 0.618034, 0.),
#             ResConv2d(image_size, in_channels, out_channels, norm_scale, kernel_size, kernel_size, conv_momentum, num_heads=1, batch_norm=True, xor=True),
#         )
#         # self.layer2 = nn.Sequential(
#         #     # ResBatchNorm2d(in_channels, norm_scale, bias=False, short_cut=True),
#         #     # ResLayerNorm2d(image_size, norm_scale, bias=False, short_cut=True),
#         #     # ResReLU(relu_neg_momentum, reverse=True),
#         #     ResAttnConv2d(image_size, in_channels, out_channels, norm_scale, kernel_size, kernel_size, conv_momentum, num_heads=2,
#         #                batch_norm=True, xor=True),
#         #     # ResBatchNorm2d(out_channels, norm_scale, bias=False, short_cut=True),
#         #     ResConv2d(image_size, out_channels, out_channels, norm_scale, kernel_size, conv_momentum, num_heads=1, pre_norm=True, batch_norm=True,
#         #               xor=True),
#         #     ResReLU(relu_neg_momentum, reverse=True),
#         # )
#         # self.relu = ResReLU(relu_neg_momentum)
#         # self.drop_out = nn.Dropout()
#
#     def forward(self, x):
#         # y = self.layer1(x) + self.layer2(x)
#         # x = self.alpha * self.conv_momentum * x if type(self.conv_momentum) is nn.parameter.Parameter else self.conv_momentum(x)
#         return self.layer1(x)


class MetaResNet(nn.Module):
    def __init__(self, num_layers, init_channels, kernel_size=3, norm_scale=0.1816, image_sizes=None, dropout=True):
        super(MetaResNet, self).__init__()
        num_hidden_layers = int(torch.tensor([i[1] for i in num_layers]).sum().item())
        dropouts = [0.1816] * (len(num_layers) - 1) + [0.1816] if dropout else None
        alpha = 1. * torch.ones(num_hidden_layers)  # torch.sort(torch.cat([torch.full((1,), 1.), 0.905 + 0.09 * torch.rand(num_hidden_layers - 2), torch.full((1,), 0.9)]), descending=True).values #
        beta = 0. * torch.ones(num_hidden_layers + len(num_layers)) # torch.sort(torch.cat([torch.full((1,), 1.), 0.905 + 0.09 * torch.rand(num_hidden_layers + len(num_layers) - 3), torch.full((1,), 0.9)]), descending=True).values #
        layers = [ResConv2d(None, init_channels, num_layers[0][0], norm_scale, kernel_size, kernel_size,
                     conv_momentum=None,
                     num_heads=1, batch_norm=True, xor=True)] # has_attn=False
        xor = True
        k = -1
        alpha_id, beta_id = 0, 1
        # prob = True
        for n, (i, j) in enumerate(num_layers):
            if k > 0 and k != i:
                layers.append(ResConv2d(None, k, i, norm_scale, kernel_size, kernel_size,
                     conv_momentum=None, num_heads=1, batch_norm=True, xor=True))
                beta_id += 1
            for id in range(j):
                layers.append(ResConv2d(None, i, i, norm_scale, kernel_size, kernel_size,
                     conv_momentum=None, num_heads=1, batch_norm=True, xor=True)) # id % 2 == 1
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



