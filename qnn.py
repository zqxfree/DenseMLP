from math import sqrt, log, ceil, pi, inf, nan, pow
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.polys.polyconfig import query


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


def multi_kmeans_1d(x: torch.Tensor, n: int, max_iter: int = 50, epsilon: float = 1e-3):
    with torch.no_grad():
        if n <= 0:
            raise ValueError("聚类数必须大于0")
        if x.dim() < 2:
            raise ValueError("输入必须是多维张量")
        if len(x) < n:
            raise ValueError("聚类数不能超过数据点数")

        # 对输入数据进行排序
        centers = torch.quantile(x, torch.linspace(1 / (n + 1), 1 - 1 / (n + 1), n, device=x.device), -1).permute(
            list(range(1, x.dim())) + [0])

        for _ in range(max_iter):
            # 计算分界点
            boundaries = (centers[..., :-1] + centers[..., 1:]) / 2

            # 向量化计算簇归属
            compare = x[..., None] > boundaries[..., None, :]
            cluster_indices = compare.sum(-1)

            # 计算新中心点
            outputs = []
            new_centers = []

            for i in range(n):
                mask = cluster_indices == i
                y = mask * x
                y_n = mask.sum(-1)
                new_centers.append(torch.where(y_n > 0, y.sum(-1) / y_n, centers[..., i]))
                outputs.append(y)

            # 检查收敛
            new_centers = torch.stack(new_centers, -1)
            if torch.all(torch.abs(new_centers - centers).max(
                    -1).values <= epsilon):  # torch.allclose(centers, new_centers, atol=1e-6):
                outputs = torch.stack(outputs, 0)
                break
            centers = new_centers

        return outputs


def flat_kmeans_1d(x: torch.Tensor, n: int, max_iter: int = 0, epsilon: float = 1e-2, equal_dist_centers=True):
    with torch.no_grad():
        if n <= 0:
            raise ValueError("聚类数必须大于0")
        if x.dim() != 1:
            raise ValueError("输入必须是一维张量")
        if len(x) < n:
            raise ValueError("聚类数不能超过数据点数")
        print(x.min(), x.max())
        # 对输入数据进行排序
        if equal_dist_centers:
            centers = torch.linspace(x.min(), x.max(), n + 2, device=x.device)
        else:
            centers = torch.quantile(x, torch.linspace(0., 1., n + 2, device=x.device))



        x = x[:, None]
        cur_iter = 0
        while True:
            # 计算分界点
            boundaries = (centers[:-1] + centers[1:]) / 2

            # 向量化计算簇归属
            cluster_indices = (x > boundaries).sum(1, keepdim=True)

            # 计算新中心点
            outputs = []
            new_centers = []

            for i in range(n + 1):
                mask = cluster_indices == i
                y = mask * x
                y_n = mask.sum()
                new_centers.append(y.sum() / y_n if y_n > 0 else centers[i])
                outputs.append(y)

            # 检查收敛
            new_centers = torch.stack(new_centers)
            if cur_iter == max_iter or torch.abs(new_centers - centers).max() <= epsilon:  # torch.allclose(centers, new_centers, atol=1e-6):
                outputs = torch.cat(outputs, 1).t()
                break

            centers = new_centers
            cur_iter += 1

        return outputs


class ReLUSplitNorm(nn.Module):
    def __init__(self, in_channel, norm_scale, split_num=4):
        super(ReLUSplitNorm, self).__init__()
        self.up_avg_momentum = nn.Parameter(torch.tensor(1.))
        self.down_avg_momentum = nn.Parameter(torch.tensor(1.))  # nn.Parameter(torch.ones(in_channel, 1, 1)) # nn.Parameter(torch.tensor(1.))
        self.up_relu_momentum = nn.Parameter(torch.tensor(1.))
        self.down_relu_momentum = nn.Parameter(torch.tensor(1.))
        self.norm_momentum1 = nn.Parameter(torch.tensor(0.))
        self.norm_momentum2 = nn.Parameter(torch.tensor(0.))
        self.norm_momentum3 = nn.Parameter(torch.tensor(0.))
        self.norm_momentum4 = nn.Parameter(torch.tensor(0.))
        self.split_momentum = nn.Parameter(torch.ones(split_num))
        self.norm_scale = norm_scale
        # self.avg_norm = nn.BatchNorm2d(in_channel, affine=False)
        self.norm1 = nn.BatchNorm2d(in_channel, affine=False)
        self.norm2 = nn.BatchNorm2d(in_channel, affine=False)
        self.norm3 = nn.BatchNorm2d(in_channel, affine=False)
        self.norm4 = nn.BatchNorm2d(in_channel, affine=False)
        self.norm_momentum = nn.Parameter(torch.zeros(split_num))
        self.norm = nn.ModuleList([nn.BatchNorm2d(in_channel, affine=False) for _ in range(split_num)])
        self.split_num = split_num
        self.init_center_mode = 1

    def init_split_centers_1d(self, x):
        if self.init_center_mode == 0:
            centers = torch.linspace(x.min(), x.max() + 1e-3, self.split_num, device=x.device)
        else:
            centers = torch.quantile(x, torch.linspace(0., 1., self.split_num, device=x.device))
        return centers

    def mean_split(self, x, centers, max_iter=1, center_mode="any"):
        with torch.no_grad():
            y = x.unsqueeze(-1)
            if center_mode == 'avg':
                for _ in range(max_iter):
                    avg = torch.mean(torch.diff(y < centers) * y, 0)  # list(range(x.dim()))
                    centers[1:-1] = torch.mean(torch.diff(y < avg) * y, 0)
            elif center_mode == 'median':
                for _ in range(max_iter):
                    avg = torch.mean(torch.diff(y < centers) * y, 0)
                    centers[1:-1] = torch.median(torch.diff(y < avg) * y, 0).values
            else:
                for _ in range(max_iter):
                    avg = torch.mean(torch.diff(y < centers) * y, 0)
                    centers[1:-1] = (avg[:-1] + avg[1:]) / 2
            avg = torch.mean(torch.diff(y < centers) * y, 0)
            return avg # .detach() # z, y - z


    def forward(self, x):
        # avg1 = x.mean() # x.mean() # [1, 2, 3], keepdim=True
        # up_mask = (x > avg1).type(torch.int).data
        # x_up = up_mask * x
        # n_up = up_mask.sum()
        # avg2 = x_up.sum() / n_up
        # avg0 = (x - x_up).sum() / (x.numel() - n_up)
        centers = self.init_split_centers_1d(x.view(-1))
        avg0, avg1, avg2 = self.mean_split(x.view(-1), centers)
        x1 = F.elu(x - avg2) # top_mask * x # (1 - up_mask) * avg0 + (up_mask - top_mask) * avg1 + top_mask * x
        x2 = F.elu(avg2 - avg1 - F.elu(avg2 - x)) # (up_mask - top_mask) * x # (1 - up_mask) * avg0 + (up_mask - top_mask) * x + top_mask * avg1
        x3 = F.elu(avg1 - avg0 - F.elu(avg1 - x)) #(1 - up_mask - bottom_mask) * x # up_mask * avg0 + (1 - up_mask - bottom_mask) * x + bottom_mask * avg2
        x4 = -F.elu(avg0 - x) #bottom_mask * x # up_mask * avg0 + (1 - up_mask - bottom_mask) * avg2 + bottom_mask * x

        # y1 = F.relu(x - avg2)  # top_mask * x # (1 - up_mask) * avg0 + (up_mask - top_mask) * avg1 + top_mask * x
        # y2 = F.relu(avg2 - avg1 - F.relu(avg2 - x))  # (up_mask - top_mask) * x # (1 - up_mask) * avg0 + (up_mask - top_mask) * x + top_mask * avg1
        # y3 = F.relu(avg1 - avg0 - F.relu(avg1 - x))  # (1 - up_mask - bottom_mask) * x # up_mask * avg0 + (1 - up_mask - bottom_mask) * x + bottom_mask * avg2
        # y4 = -F.relu(avg0 - x)  # bottom_mask * x # up_mask * avg0 + (1 - up_mask - bottom_mask) * avg2 + bottom_mask * x

        x1_norm = self.norm_momentum1 * x1 + self.norm_scale * self.norm1(x1)
        x2_norm = self.norm_momentum2 * x2 + self.norm_scale * self.norm2(x2)
        x3_norm = self.norm_momentum3 * x3 + self.norm_scale * self.norm3(x3)
        x4_norm = self.norm_momentum4 * x4 + self.norm_scale * self.norm4(x4)
        # x_merge = self.split_momentum[0] * y1 + self.split_momentum[1] * y2  + self.split_momentum[2] * y3 + self.split_momentum[3] * y4

        # centers = self.init_split_centers_1d(x.view(-1))
        # x_split, x_compl = self.mean_split(x, centers)
        # x_split = x_split + x_compl.sign() * F.elu(-x_compl.abs())
        # x_split = (self.norm_momentum * x_split).permute([-1] + list(range(x_split.dim() - 1))) + self.norm_scale * torch.stack([norm(x_split[..., i]) for i, norm in enumerate(self.norm)])
        # x_split = x_split.permute(list(range(1, x_split.dim())) + [0])
        return torch.stack([x1_norm, x2_norm, x3_norm, x4_norm]) # x_merge # x1, x2, x3, x4 x_split #


class AttnConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_scale, attn_kernel_size, attn_hidden_size, image_size, kerel_split_thres=16):
        super(AttnConv2d, self).__init__()
        if image_size >= kerel_split_thres:
            in_features = (image_size // attn_kernel_size) ** 2
            self.key_unfold = nn.Unfold(attn_kernel_size, stride=attn_kernel_size)
            self.query_unfold = nn.Unfold(attn_kernel_size, stride=attn_kernel_size)
            self.key_linear = nn.Linear(in_features, attn_hidden_size, bias=False)
            self.query_linear = nn.Linear(in_features, attn_hidden_size, bias=False)
        else:
            in_features = image_size ** 2
            self.key_unfold = None
            self.query_unfold = None
            self.key_linear = nn.Linear(in_features, attn_hidden_size * attn_kernel_size * attn_kernel_size, bias=False)
            self.query_linear = nn.Linear(in_features, attn_hidden_size * attn_kernel_size * attn_kernel_size, bias=False)
        bound = pow(3. / (attn_hidden_size * in_channels * in_features * in_features * attn_kernel_size * attn_kernel_size), 0.25)
        nn.init.uniform_(self.key_linear.weight, -bound, bound)
        nn.init.uniform_(self.query_linear.weight, -bound, bound)
        self.value_unfold = nn.Unfold(attn_kernel_size, padding=attn_kernel_size // 2)
        self.value_conv = nn.Conv2d(in_channels, in_channels, attn_kernel_size, padding=attn_kernel_size // 2, groups=in_channels, bias=False)
        self.query_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pos_conv_kernels = nn.Parameter(nn.Conv2d(in_channels, in_channels, attn_kernel_size).weight.data)
        self.norm_scale = norm_scale
        self.kernel_momentum = nn.Parameter(torch.tensor(1.))
        self.kernel_norm = nn.LayerNorm(in_channels * attn_kernel_size * attn_kernel_size, elementwise_affine=False)
        self.attn_hidden_size = attn_hidden_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_kernel_size = attn_kernel_size

    def patch_regroup(self, x):
        channels = x.size(1)
        patch_size = x.size(-1) // self.multi_heads
        x = F.unfold(x, kernel_size=patch_size, stride=patch_size)
        x = x.unflatten(1, (channels, -1)).transpose(2, 3).contiguous().view(x.size(0), -1, patch_size, patch_size) # (N*multi_heads*multi_heads, channels, patch_size, patch_size)
        # x = x.transpose(1, 2).flatten(0, 1).unflatten(1, (-1, patch_size, patch_size)) # (N*multi_heads*multi_heads, channels, patch_size, patch_size)
        return x

    def full_norm(self, x):
        x = self.norm_scale * (x - x.mean()) / (x.std() + 1e-4)  # (x - x.mean()) # / (x.std() + 1e-4) # conv_momentum * x +  conv_momentum * x +
        return x

    def forward(self, x1, x2):
        x2 = self.query_conv(x2)
        if self.key_unfold is not None:
            key = self.key_linear(self.key_unfold(x1)).unflatten(1, (self.in_channels, -1))
            query = self.query_linear(self.query_unfold(x2)).unflatten(1, (self.out_channels, -1))
        else:
            key = self.key_linear(x1.flatten(2)).unflatten(2, (-1, self.attn_hidden_size))
            query = self.query_linear(x2.flatten(2)).unflatten(2, (-1, self.attn_hidden_size))
        attn_kernel = key.transpose(1, 2).matmul(query.permute([0, 2, 3, 1])).transpose(1, 3).mean(0) #.flatten(2)
        # attn_kernel = self.full_norm(attn_kernel)
        attn_kernel = self.norm_scale * attn_kernel  # * self.pos_conv_kernels
        attn_kernel = (attn_kernel - attn_kernel.detach().max()).exp()
        attn_kernel = attn_kernel / attn_kernel.sum()
        # attn_kernel = self.kernel_momentum * attn_kernel + attn_softmax
        attn_kernel = attn_kernel.unflatten(-1, (-1, self.attn_kernel_size)) # F.softmax(attn_kernel, -1 ).
        value = F.conv2d(self.value_conv(x1), attn_kernel, padding=self.attn_kernel_size // 2) # self.value_unfold(x1)
        # value = attn_kernel.matmul(value).unflatten(2, (int(sqrt(value.size(-1))), -1))
        # value = self.full_conv(value)
        return value


class ButterflyAttnConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_scale, attn_kernel_size, attn_hidden_size, image_size, kerel_split_thres=16):
        super(ButterflyAttnConv2d, self).__init__()
        self.xor_conv1 = AttnConv2d(in_channels, out_channels, norm_scale, attn_kernel_size, attn_hidden_size, image_size, kerel_split_thres)
        self.xor_conv2 = AttnConv2d(in_channels, out_channels, norm_scale, attn_kernel_size, attn_hidden_size, image_size, kerel_split_thres)
        self.xor_conv3 = AttnConv2d(in_channels, out_channels, norm_scale, attn_kernel_size, attn_hidden_size, image_size, kerel_split_thres)
        self.xor_conv4 = AttnConv2d(in_channels, out_channels, norm_scale, attn_kernel_size, attn_hidden_size, image_size, kerel_split_thres)
        self.conv1 = nn.Conv2d(in_channels, out_channels, attn_kernel_size, padding=attn_kernel_size // 2, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, attn_kernel_size, padding=attn_kernel_size // 2, bias=False)
        self.norm_scale = norm_scale
        self.conv_momentum = nn.Parameter(torch.tensor(1.))
        self.out_channels = out_channels
        self.norm1 = nn.BatchNorm2d(out_channels, affine=False)
        self.norm2 = nn.BatchNorm2d(out_channels, affine=False)
        self.xor_norm1 = nn.BatchNorm2d(out_channels, affine=False)
        self.xor_norm2 = nn.BatchNorm2d(out_channels, affine=False)
        self.xor_norm3 = nn.BatchNorm2d(out_channels, affine=False)
        self.xor_norm4 = nn.BatchNorm2d(out_channels, affine=False)
        self.xor_norm5 = nn.BatchNorm2d(out_channels, affine=False)
        self.xor_norm6 = nn.BatchNorm2d(out_channels, affine=False)

    def full_norm(self, x, conv_momentum):
        x = 0.5 * conv_momentum * x + self.norm_scale * (x - x.mean()) / (x.std() + 1e-4) # conv_momentum * x +  conv_momentum * x +
        return x

    def res_layer_norm(self, x):
        # x = x.transpose(0, 1)
        x = self.conv_momentum * x + self.norm_scale * F.layer_norm(x, x.shape[1:]) # F.layer_norm(x, x.shape[1:])) # conv_momentum * x +  (1 / sqrt(self.out_channels) / self.kernels) * conv_momentum
        # x = x.transpose(0, 1)
        return x

    def forward(self, x1, x2):
        # single_y1 = self.conv1(x1)
        # single_y2 = self.conv2(x2)
        y1 = self.xor_conv1(x1, x1)
        y2 = self.xor_conv2(x1, x1)
        y3 = self.xor_conv3(x2, x2)
        y4 = self.xor_conv4(x2, x2)
        # y5 = self.xor_conv2(x1, x1)
        # y6 = self.xor_conv3(x2, x2)
        # single_y1 = self.res_layer_norm(single_y1)
        # single_y2 = self.res_layer_norm(single_y2)
        y1 = self.res_layer_norm(y1)
        y2 = self.res_layer_norm(y2)
        y3 = self.res_layer_norm(y3)
        y4 = self.res_layer_norm(y4)
        y = y1 + y2 + y3 + y4 # single_y1 + single_y2 +
        return y


class XorConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_scale, kernels, multi_head=1):
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
        self.norm_scale = norm_scale
        self.kernels = kernels
        self.conv_momentum1 = nn.Parameter(torch.tensor(1.))
        # self.conv_momentum2 = nn.Parameter(torch.tensor(1.))
        # self.xor_conv_momentum1 = nn.Parameter(torch.tensor(1.))
        # self.xor_conv_momentum2 = nn.Parameter(torch.tensor(1.))
        # self.xor_conv_momentum3 = nn.Parameter(torch.tensor(1.))
        # self.xor_conv_momentum4 = nn.Parameter(torch.tensor(1.))
        # self.xor_conv_momentum5 = nn.Parameter(torch.tensor(1.))
        # self.xor_conv_momentum6 = nn.Parameter(torch.tensor(1.))
        self.out_channels = out_channels
        self.kernels = kernels
        self.batch_norm1 = nn.BatchNorm2d(out_channels, affine=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels, affine=False)
        self.xor_norm1 = nn.BatchNorm2d(out_channels, affine=False)
        self.xor_norm2 = nn.BatchNorm2d(out_channels, affine=False)
        self.xor_norm3 = nn.BatchNorm2d(out_channels, affine=False)
        self.xor_norm4 = nn.BatchNorm2d(out_channels, affine=False)
        self.xor_norm5 = nn.BatchNorm2d(out_channels, affine=False)
        self.xor_norm6 = nn.BatchNorm2d(out_channels, affine=False)
        self.multi_head = multi_head

    def patch_group(self, x):
        channels = x.size(1)
        patch_size = x.size(-1) // self.multi_head
        x = F.unfold(x, kernel_size=patch_size, stride=patch_size)
        x = x.unflatten(1, (channels, -1)).permute([0, 3, 1, 2]).contiguous().view(x.size(0), -1, patch_size, patch_size) # (N*multi_heads*multi_heads, channels, patch_size, patch_size)
        # x = x.transpose(1, 2).flatten(0, 1).unflatten(1, (-1, patch_size, patch_size)) # (N*multi_heads*multi_heads, channels, patch_size, patch_size)
        return x

    def inv_patch_group(self, x, origin_sizes):
        patch_size = x.size(-1)
        x = x.flatten(2).unflatten(1, (self.multi_head * self.multi_head, -1)).flatten(2, 3).transpose(1, 2)
        # x = x.flatten(1).unflatten(0, (-1, self.multi_head * self.multi_head)).transpose(1, 2)
        x = F.fold(x, origin_sizes, patch_size, stride=patch_size)
        return x

    def full_norm(self, x, conv_momentum):
        x = 0.5 * conv_momentum * x + self.norm_scale * (x - x.mean()) / (x.std() + 1e-4) # conv_momentum * x +  conv_momentum * x +
        return x

    def res_layer_norm(self, x, has_momentum=True):
        y = self.conv_momentum1 * x + self.norm_scale * F.layer_norm(x, x.shape[1:]) # # self.conv_momentum1 * x self.norm_scale * F.softmax(x, 1) # (x - x.mean()) / (x.std() + 1e-4) # F.layer_norm(x, x.shape[1:])) # conv_momentum * x +  (1 / sqrt(self.out_channels) / self.kernels) * conv_momentum
        # if has_momentum:
        #     y = y + self.conv_momentum1 * x
        return y

    def batch_norm(self, x, norm):
        # x = x.transpose(0, 1)
        x = self.conv_momentum2 * x + self.norm_scale * norm(x) # F.layer_norm(x, x.shape[1:])) # conv_momentum * x +  (1 / sqrt(self.out_channels) / self.kernels) * conv_momentum
        # x = x.transpose(0, 1)
        return x

    def forward(self, x1, x2):
        single_y1 = self.conv1(x1)
        single_y2 = self.conv2(x2)
        y1 = self.xor_conv1(x1)
        y2 = self.xor_conv2(x1)
        y3 = self.xor_conv3(x2)
        y4 = self.xor_conv4(x2)
        y5 = self.xor_conv5(x2)
        y6 = self.xor_conv6(x1)
        # y7 = self.xor_conv5(x1)
        # y8 = self.xor_conv6(x2)
        single_y1 = self.res_layer_norm(single_y1, True)
        single_y2 = self.res_layer_norm(single_y2, True)
        y1_xor = y1 * y2
        y2_xor = y3 * y4
        y3_xor = y5 * y6
        # y4_xor = y7 * y8
        y1_xor = self.res_layer_norm(y1_xor)
        y2_xor = self.res_layer_norm(y2_xor)
        y3_xor = self.res_layer_norm(y3_xor)
        # y4_xor = self.res_layer_norm(y4_xor)
        y = single_y1 + single_y2 + y1_xor + y2_xor + y3_xor # + y4_xor #
        return y


class SplitConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_scale, kernels, split_num=4):
        super(SplitConv2d, self).__init__()
        self.xor_conv1 = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False) for _ in range(split_num)])
        # xor_conv2 = []
        # for _ in range(split_num):
        #     xor_conv2.append(nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=True))
        #     nn.init.ones_(xor_conv2[-1].bias)
        # self.xor_conv2 = nn.ModuleList(xor_conv2)
        # xor_conv3 = []
        # for _ in range(split_num):
        #     xor_conv3.append(nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=True))
        #     nn.init.ones_(xor_conv3[-1].bias)
        # self.xor_conv3 = nn.ModuleList(xor_conv3)

        if True: #in_channels != out_channels:
            self.xor_conv2 = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False) for _ in
                 range(split_num)])  # (split_num - 1) // 2 + 1
        else:
            self.xor_conv2 = None
        self.xor_conv3 = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False) for _ in
             range(split_num)])  # (split_num - 1) // 2 + 1
        self.xor_conv4 = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False) for _ in range(split_num)])
        self.norm_scale = norm_scale
        self.kernels = kernels
        self.conv_momentum = nn.Parameter(torch.tensor(1.)) #nn.Parameter(torch.ones(out_channels, 1, 1)) #
        self.out_channels = out_channels
        self.kernels = kernels
        self.split_num = split_num
        self.cross_monentum = nn.Parameter(torch.ones(4))
        self.alpha1 = nn.Parameter(torch.tensor(1.)) # nn.Parameter(torch.ones(out_channels, 1, 1))
        self.alpha2 = nn.Parameter(torch.ones(out_channels, 1, 1)) # nn.Parameter(torch.tensor(1.)) #

    def res_layer_norm(self, x):
        return self.conv_momentum * x + self.norm_scale * F.layer_norm(x, x.shape[1:])

    def forward(self, x):
        y = 0.
        z = x.sum(0) # x[0] + x[1] + x[2] + x[3] # self.cross_monentum[0] * x[0] + self.cross_monentum[1] * x[1] + self.cross_monentum[2] * x[2] + self.cross_monentum[3] * x[3] #
        if self.xor_conv2 is not None:
            for xi, conv1, conv2, conv3, conv4 in zip(x, self.xor_conv1, self.xor_conv2, self.xor_conv3, self.xor_conv4):
                y = y + self.res_layer_norm(conv1(xi) * (self.alpha1 * conv2(xi) + self.alpha2)) + self.res_layer_norm(conv3(xi) * conv4(self.cross_monentum[0] * (z - xi) / 3)) #
        # else:
        #     for xi, conv1, conv3, conv4 in zip(x, self.xor_conv1, self.xor_conv3, self.xor_conv4):
        #         y = y + self.res_layer_norm(conv1(xi) * (self.alpha1 * xi + self.alpha2)) + self.res_layer_norm(conv3(xi) * conv4(self.cross_monentum[0] * (z - xi) / 3))
        # k = 0
        # for i in range(self.split_num):
        #     for j in range(i + 1, self.split_num):
        #         y = y + self.res_layer_norm(self.xor_conv3[k](x[i]) * self.xor_conv4[k](x[j]))
        #         k += 1
        # k = 0
        # for i in range(self.split_num):
        #     z = self.xor_conv3[k](x[i])
        #     k += 1
        #     u = 0.
        #     for j in range(i + 1, self.split_num):
        #         u = u + self.xor_conv3[k](x[j])
        #         k += 1
        #     y = y + self.res_layer_norm(z * u)
        # for i in range(self.split_num):
        #     z = self.xor_conv1[i](x[i])
        #     for j in range(self.split_num):
        #         y = y + self.res_layer_norm(z * self.xor_conv2[i * self.split_num + j](x[j]))
        return y


class MetaResConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_scale, kernels, image_size=32):
        super(MetaResConv2d, self).__init__()
        # if in_channels != out_channels:
        #     self.res_momentum = nn.Conv2d(in_channels, out_channels, 1, bias=False) # SplitConv2d(in_channels, out_channels, norm_scale, 1, 4)
        # else:
        #     self.res_momentum = nn.Parameter(torch.ones(4))
        self.res_momentum = nn.Parameter(torch.ones(4))
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else None
        self.mean_split = ReLUSplitNorm(in_channels, norm_scale)
        self.conv = SplitConv2d(in_channels, out_channels, norm_scale, kernels, 4) #XorConv2d(in_channels, out_channels, norm_scale, kernels, 1)
        # self.conv = ButterflyAttnConv2d(in_channels, out_channels, norm_scale, kernels, 64, image_size, 16)
        self.norm_scale = norm_scale

    def forward(self, x):
        x_split = self.mean_split(x)
        y = self.conv(x_split)
        # x = self.res_momentum[0] * x #1 + self.res_momentum[1] * x2 + self.res_momentum[2] * x3 + self.res_momentum[3] * x4
        if self.res_conv is not None:
            x = self.res_conv(x)
        else:
            x = self.res_momentum[0] * x
        # if type(self.res_momentum) is not nn.parameter.Parameter:
        #     x = self.res_momentum[0] * x1 + self.res_momentum[1] * x2 + self.res_momentum[2] * x3 + self.res_momentum[3] * x4
        # else:
        #     x = self.res_momentum((x1 + x2 + x3 + x4) / 2)
        # if type(self.conv_momentum1) is not nn.parameter.Parameter:
        #     x3 = self.conv_momentum1(x3)
        return x + y


class MetaResNet(nn.Module):
    def __init__(self, num_layers, init_channels, kernel_size=3, norm_scale=0.1773, image_size=None, dropout=True):
        super(MetaResNet, self).__init__()
        num_hidden_layers = int(torch.tensor([i[1] for i in num_layers]).sum().item())
        dropouts = [0.5] * (len(num_layers) - 1) + [0.5] if dropout else None
        alpha = 1. * torch.ones(num_hidden_layers)  # torch.sort(torch.cat([torch.full((1,), 1.), 0.905 + 0.09 * torch.rand(num_hidden_layers - 2), torch.full((1,), 0.9)]), descending=True).values #
        beta = 0. * torch.ones(num_hidden_layers + len(num_layers)) # torch.sort(torch.cat([torch.full((1,), 1.), 0.905 + 0.09 * torch.rand(num_hidden_layers + len(num_layers) - 3), torch.full((1,), 0.9)]), descending=True).values #
        layers = [MetaResConv2d(init_channels, num_layers[0][0], norm_scale, kernel_size, image_size)]
        xor = True
        k = -1
        alpha_id, beta_id = 0, 1
        cross = True
        # prob = True
        for n, (i, j) in enumerate(num_layers):
            if k > 0 and k != i:
                layers.append(MetaResConv2d(k, i, norm_scale, kernel_size, image_size))
                beta_id += 1
                cross = ~cross
            for id in range(j):
                layers.append(MetaResConv2d(i, i, norm_scale, kernel_size, image_size)) # id % 2 == 1
                cross = ~cross
                xor = ~xor
                alpha_id += 1
                beta_id += 1
            layers.append(nn.AvgPool2d(2))
            if image_size is not None:
                image_size //= 2
            if dropouts is not None:
                layers.append(nn.Dropout(dropouts[n]))
            k = i
            # prob = not prob
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



