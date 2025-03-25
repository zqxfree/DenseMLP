import math
from math import sqrt, log, ceil, pi, inf, nan
import torch
import torch.nn as nn
import torch.nn.functional as F
from qnn import *


class PMAXPool1d(nn.Module):
    def __init__(self, kernel_size=2, p=0.9):
        super(PMAXPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.p0 = kernel_size * (1 - p) / (kernel_size - 1)
        self.p1 = kernel_size * p - self.p0
        self.max_pool = nn.MaxPool1d(kernel_size, return_indices=True)
        self.avg_pool = nn.AvgPool1d(kernel_size)

    def forward(self, x):
        y, max_indices = self.max_pool(x)
        y = self.p0 * x + self.p1 * F.max_unpool1d(y, max_indices, self.kernel_size, output_size=x.size())
        return self.avg_pool(y)


class PMAXPool2d(nn.Module):
    def __init__(self, kernel_size=2, p=0.7):
        super(PMAXPool2d, self).__init__()
        if '__getitem__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.p0 = kernel_size[0] * kernel_size[1] * (1 - p) / (kernel_size[0] * kernel_size[1] - 1)
        self.p1 = kernel_size[0] * kernel_size[1] * p - self.p0
        self.max_pool = nn.MaxPool2d(kernel_size, return_indices=True)
        self.avg_pool = nn.AvgPool2d(kernel_size)

    def forward(self, x):
        y, max_indices = self.max_pool(x)
        y = self.p0 * x + self.p1 * F.max_unpool2d(y, max_indices, self.kernel_size, output_size=x.size())
        return self.avg_pool(y)


class SpatialMaxout2d(nn.Module):
    def __init__(self, kernel_size, bottom_right_pad=True, spatial_scaling=True):
        super(SpatialMaxout2d, self).__init__()
        if '__getitem__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.bottom_right_pad = bottom_right_pad
        self.pool = nn.MaxPool2d(kernel_size, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size)
        if spatial_scaling:
            self.alpha = nn.Parameter(torch.zeros(kernel_size))
        else:
            self.register_parameter('alpha', None)

    def forward(self, x):
        row_pads, col_pads = self.kernel_size[0] - x.size(-2) % self.kernel_size[0], self.kernel_size[1] - x.size(-1) % self.kernel_size[1]
        if self.bottom_right_pad:
            pad_shape = (0, row_pads, 0, col_pads)
            inv_pad_shape = (0, -row_pads, 0, -col_pads)
        else:
            pad_shape = (row_pads, 0, col_pads, 0)
            inv_pad_shape = (-row_pads, 0, -col_pads, 0)
        if max(pad_shape) > 0:
            x = F.pad(x, pad_shape)
        x_smo, indices = self.pool(x)
        x_smo = self.unpool(x_smo, indices.detach())
        if self.alpha is not None:
            row_patches, col_patches = x.size(-2) // self.kernel_size[0], x.size(-1) // self.kernel_size[1]
            alpha = self.alpha.repeat(row_patches, col_patches)
            x_smo = (1 - alpha) * x_smo + alpha * x
        if max(pad_shape) > 0:
            x_smo = F.pad(x_smo, inv_pad_shape)
        return x_smo


class SpatialMaxout1d(nn.Module):
    def __init__(self, kernel_size, spatial_scaling=True):
        super(SpatialMaxout1d, self).__init__()
        self.kernel_size = kernel_size
        self.pool = nn.MaxPool1d(kernel_size, return_indices=True)
        self.unpool = nn.MaxUnpool1d(kernel_size)
        if spatial_scaling:
            self.alpha = nn.Parameter(torch.zeros(kernel_size))
        else:
            self.register_parameter('alpha', None)

    def forward(self, x):
        pads = self.kernel_size - x.size(-1) % self.kernel_size
        pad_shape = (0, pads)
        inv_pad_shape = (0, -pads)
        if max(pad_shape) > 0:
            x = F.pad(x, pad_shape)
        x_smo, indices = self.pool(x)
        x_smo = self.unpool(x_smo, indices.detach())
        if self.alpha is not None:
            patches = x.size(-1) // self.kernel_size
            alpha = self.alpha.repeat(patches)
            x_smo = (1 - alpha) * x_smo + alpha * x
        if max(pad_shape) > 0:
            x_smo = F.pad(x_smo, inv_pad_shape)
        return x_smo


class BatchLinear(nn.Module):
    def __init__(self, batch_size, in_features, out_features):
        super(BatchLinear, self).__init__()
        self.weight = nn.Parameter(torch.empty(batch_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(batch_size * out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.uniform_(self.bias, -1 / sqrt(in_features), 1 / sqrt(in_features))

    def forward(self, x):
        x = x.transpose(0, 1).bmm(self.weight).transpose(0, 1)
        x = x.contiguous().view(x.size(0), 1, -1)
        x += self.bias
        return x


class LossActivation(nn.Module):
    def __init__(self, momentum_shape=None, momentum=0.1, flip=False):
        super(LossActivation, self).__init__()
        if momentum_shape is not None:
            self.momentum = nn.Parameter(torch.empty(momentum_shape))
            nn.init.constant_(self.momentum, momentum)
        else:
            self.register_parameter('momentum', None)
        self.flip = flip

    def forward(self, x):
        if self.flip:
            if self.training:
                mask = torch.randint_like(x, 0, 2).detach()
                y = mask * x + (1 - mask) * x.abs()
                # y *= y.tanh()
            else:
                y = F.relu(x)
        else:
            y = x.abs()
        if self.momentum is not None:
            return self.momentum * x + (1 - self.momentum) * y
        return y


class SoftTanh(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(1 + x * x)


class Abs(nn.Module):
    def forward(self, x):
        return F.relu(x - 0.5) - F.relu(-x - 0.5)


class SwapAxes(nn.Module):
    def __init__(self, dim0, dim1, flat=False):
        super(SwapAxes, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        self.flat = flat

    def forward(self, x):
        # if isinstance(x, tuple):
        #     x, loss = x
        # else:
        #     loss = 0.
        return x.transpose(self.dim0, self.dim1)  # if not self.flat else x.flatten(1), loss


class Scale(nn.Module):
    def __init__(self, scale_size):
        super(Scale, self).__init__()
        self.scale_size = scale_size

    def forward(self, x):
        if x.size(-2) % self.scale_size != 0:
            x = x[:, :-(x.size(-2) % self.scale_size), :]
        return x.contiguous().view(-1, x.size(-2) // self.scale_size, x.size(-1) * self.scale_size)


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class TernaryTransformerBlock(nn.Module):
    def __init__(self, seq_len, num_features):
        super(TernaryTransformerBlock, self).__init__()
        self.value = TernaryMLP(num_features, num_features, residual=False)
        self.left_attn_layer1 = nn.Sequential(
            nn.Linear(num_features, 1),
            SwapAxes(-2, -1),
            nn.LayerNorm(seq_len, elementwise_affine=False),
            nn.ReLU(),
            SwapAxes(-2, -1),
        )
        self.right_attn_layer1 = nn.Sequential(
            SwapAxes(-2, -1),
            nn.Linear(seq_len, 1),
            SwapAxes(-2, -1),
            nn.LayerNorm(num_features, elementwise_affine=False),
            nn.ReLU(),
        )
        self.alpha = nn.Parameter(torch.rand(1))
        # self.alpha2 = nn.Parameter(torch.tensor(0.))
        # self.alpha3 = nn.Parameter(torch.tensor(0.))

    def get_attn(self, x, attn_layer):
        attn = attn_layer(x)
        attn = attn / (attn.sum(-1, keepdim=True) + 1e-6)
        return attn

    def forward(self, x, seq_pos_code, feature_pos_code):
        v = self.value(x)
        left_attn = self.get_attn(x, self.left_attn_layer1)
        right_attn = self.get_attn(x, self.right_attn_layer1)
        x = (1 - self.alpha) * x + self.alpha * left_attn * v * right_attn
        return x


class TernaryTransformer(nn.Module):
    def __init__(self, seq_len, num_features, depth):
        super(TernaryTransformer, self).__init__()
        seq_len, seq_dist = self.attn_pos_encoding(seq_len)
        num_features, feature_dist = self.attn_pos_encoding(num_features)
        self.seq_pos_code1 = nn.Parameter(seq_dist)
        self.seq_pos_code2 = nn.Parameter(seq_dist.clone())
        self.feature_pos_code1 = nn.Parameter(feature_dist)
        self.feature_pos_code2 = nn.Parameter(feature_dist.clone())
        self.attn_layers = nn.ModuleList([TernaryTransformerBlock(seq_len, num_features) for _ in range(depth)])
        self.mlp_layers = nn.ModuleList([TernaryMLP(num_features, num_features, residual=True) for _ in range(depth)])

    def attn_pos_encoding(self, seq_len):
        if '__getitem__' in dir(seq_len):
            dist = torch.sqrt(torch.stack([torch.flatten(
                torch.pow(torch.arange(seq_len[0]) - i, 2)[:, None] + torch.pow(torch.arange(seq_len[1]) - j, 2)) for i
                                           in range(seq_len[0]) for j in range(seq_len[1])]))
            dist /= sqrt(seq_len[0] * seq_len[0] + seq_len[1] * seq_len[1])
            seq_len = seq_len[0] * seq_len[1]
        else:
            dist = torch.abs(torch.arange(seq_len)[:, None] - torch.arange(seq_len)) / seq_len
        return seq_len, dist

    def pos_norm(self, attn_pos_code):
        attn_pos_min = attn_pos_code.min(1, keepdim=True)[0].detach()
        attn_pos_max = attn_pos_code.max(1, keepdim=True)[0].detach()
        return torch.cos(pi / 2 * (attn_pos_code - attn_pos_min) / (attn_pos_max - attn_pos_min))

    def forward(self, x):
        seq_pos_code = (self.pos_norm(self.seq_pos_code1), self.pos_norm(self.seq_pos_code2))
        feature_pos_code = (self.pos_norm(self.feature_pos_code1), self.pos_norm(self.feature_pos_code2))
        for layer1, layer2 in zip(self.attn_layers, self.mlp_layers):
            x = layer1(x, seq_pos_code, feature_pos_code)
            x = layer2(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AutoEncoder, self).__init__()
        self.encode_residuals = nn.Parameter(torch.randn(4))
        self.decode_residuals = nn.Parameter(torch.randn(4))
        self.encode = nn.Sequential(
            TernaryMLP(input_size, hidden_size),
            TernaryMLP(hidden_size, hidden_size),
            TernaryMLP(hidden_size, hidden_size),
            TernaryMLP(hidden_size, output_size),
            # TernaryLinear(input_size, hidden_size),
            # nn.LayerNorm(hidden_size),
            # nn.ReLU(),
            # TernaryLinear(hidden_size, hidden_size),
            # nn.LayerNorm(hidden_size),
            # nn.ReLU(),
            # TernaryLinear(hidden_size, output_size),
            # nn.LayerNorm(output_size),
            nn.ReLU(),
        )
        self.decode = nn.Sequential(
            TernaryMLP(output_size, hidden_size),
            TernaryMLP(hidden_size, hidden_size),
            TernaryMLP(hidden_size, hidden_size),
            TernaryMLP(hidden_size, input_size),
            # TernaryLinear(output_size, hidden_size),
            # nn.LayerNorm(hidden_size),
            # nn.ReLU(),
            # TernaryLinear(hidden_size, hidden_size),
            # nn.LayerNorm(hidden_size),
            # nn.ReLU(),
            # TernaryLinear(hidden_size, input_size),
            # nn.LayerNorm(input_size),
        )

    def forward(self, x):
        x = self.encode(x)
        sparse_loss = x.exp().mean()
        z = self.decode(x)
        return x, z, sparse_loss


class DenseMLP(nn.Module):
    def __init__(self, dense_hidden_size, num_layers=None):
        super(DenseMLP, self).__init__()
        if num_layers is None:
            num_layers = dense_hidden_size
        self.dense_hidden_size = dense_hidden_size
        self.relu = nn.ReLU()
        self.dense_layers = nn.ModuleList(
            [BatchLinear(i, dense_hidden_size, dense_hidden_size // i) for i in range(1, num_layers)])
        self.norm = LayerMinMaxNorm()

    def forward(self, x):
        dense_input = x.unsqueeze(1)
        for layer in self.dense_layers:
            dense_output = layer(self.norm(dense_input))
            dense_output = self.relu(dense_output)
            if dense_output.size(-1) < self.dense_hidden_size:
                dense_output = torch.cat([dense_output, torch.zeros(dense_output.size(0), 1,
                                                                    self.dense_hidden_size - dense_output.size(-1)).to(
                    dense_output.device)], -1)
            dense_input = torch.cat([dense_input, dense_output], 1)
        return dense_output.squeeze(1)


class QuantumAttention(nn.Module):
    def __init__(self, seq_len, num_features, dense_layers=None):
        super(QuantumAttention, self).__init__()
        self.seq_len = seq_len
        self.value = DenseMLP(num_features, dense_layers)
        self.query = nn.Linear(num_features, seq_len)
        self.lin = nn.Linear(num_features, num_features)

    def forward(self, x):
        v = self.value(x.flatten(0, 1))
        v = v.unflatten(0, (-1, self.seq_len))
        attn = self.query(x).transpose(-2, -1)
        attn = attn / (attn.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6)
        return F.layer_norm(x + attn.matmul(v), (v.shape[-1],))


class DenseTransformer(nn.Module):
    def __init__(self, depth, seq_len, num_features, dense_layers=None, image_patches=None):
        super(DenseTransformer, self).__init__()
        self.depth = depth
        if image_patches is not None:
            self.unfold = nn.Unfold(image_patches[:2], stride=image_patches[2:4])
            seq_len0 = (seq_len - image_patches[0]) // image_patches[2] + 1
            seq_len1 = (num_features - image_patches[1]) // image_patches[3] + 1
            seq_len = seq_len0 * seq_len1
            num_features = image_patches[0] * image_patches[1] * image_patches[4]
            dist = torch.sqrt(torch.stack([torch.flatten(
                torch.pow(torch.arange(seq_len0) - i, 2)[:, None] + torch.pow(torch.arange(seq_len1) - j, 2)) for i in
                range(seq_len0) for j in range(seq_len1)]))
        else:
            self.unfold = None
            dist = torch.stack([torch.abs(torch.arange(seq_len) - i) for i in range(seq_len)])
        # indices = dist >= 10
        # dist[indices] = inf
        self.pos_weight = nn.Parameter(torch.pow(2, -dist), requires_grad=False)
        self.attentions = nn.ModuleList(
            [QuantumAttention(seq_len, num_features, dense_layers) for _ in range(depth)])
        self.fc = nn.Linear(seq_len * num_features, 10)

    def forward(self, x):
        if self.unfold is not None:
            x = self.unfold(x).transpose(1, 2)
        # x = self.pos_weight.matmul(x)
        for attn in self.attentions:
            x = attn(x)
        return self.fc(x.flatten(1))


class ChannelLSTM(nn.Module):
    def __init__(self, channel):
        super(ChannelLSTM, self).__init__()
        self.channel = channel
        self.lstm1 = nn.LSTM(channel, channel, batch_first=True)
        self.lstm2 = nn.LSTM(channel, channel, batch_first=True)

    def forward(self, x):
        batch = x.size(0)
        x = x.transpose(1, 3).flatten(0, 1)
        x = self.lstm1(x)[1][0]
        x = x.view(batch, -1, self.channel)
        x = self.lstm2(x)[1][0].squeeze(0)
        return x


class FRNN(nn.Module):
    def __init__(self, num_recurrent_layers, encoder, decoder=None, fc_layer=None):
        super(FRNN, self).__init__()
        self.num_recurrent_layers = num_recurrent_layers
        self.encoder = encoder
        self.decoder = decoder
        self.fc_layer = fc_layer

    def forward(self, x):
        outputs = []
        for _ in range(self.num_recurrent_layers):
            y = self.encoder(x)
            z = self.decoder(y) if self.decoder is not None else y
            x = torch.cat([x[:, z.size(1):, :], z], dim=1)
            outputs.append(self.fc_layer(y) if self.fc_layer is not None else y)
        return torch.cat(outputs, dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=3):
        super(ResBlock, self).__init__()
        self.short_cut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernels, padding=kernels // 2),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.short_cut(x) + self.conv_layer(x))


class RawDense2d(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=64, out_channels=64, num_dense_layers=4, kernels=3, relu=True):
        super(RawDense2d, self).__init__()
        self.num_dense_layers = num_dense_layers - 1
        self.pure_res_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // num_dense_layers, kernels, padding=kernels // 2),
            nn.BatchNorm2d(out_channels // num_dense_layers))
        self.out_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(hidden_channels, out_channels // num_dense_layers,
                                                                kernels, padding=kernels // 2),
                                                      nn.BatchNorm2d(out_channels // num_dense_layers)) for i
                                        in range(self.num_dense_layers)])
        self.dense_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else hidden_channels, hidden_channels, kernels, padding=kernels // 2),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU()
            ) for i in range(self.num_dense_layers)])
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        y = [self.pure_res_conv(x)]
        for i in range(self.num_dense_layers):
            x = self.dense_layers[i](x)
            y.append(self.out_convs[i](x))
        y = torch.cat(y, 1)
        if self.relu is not None:
            y = self.relu(y)
        return y


class RatioDense2d(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, kernels=3):
        super(RatioDense2d, self).__init__()
        out_channels = int(pow(2, num_layers))  # num_layers * (num_layers + 1) // 2 #
        self.num_layers = num_layers - 1
        self.out_convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_channels if i == out_channels else hidden_channels, out_channels // i,
                                     kernels, padding=kernels // 2),
                           nn.BatchNorm2d(out_channels // i),
                           ) for i in 2 ** torch.arange(num_layers, 0, -1)]
        )
        # self.out_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels if i == 1 else hidden_channels, i*factor,
        #                                                         kernels, padding=kernels // 2),
        #                                               nn.BatchNorm2d(i*factor)) for i in range(1, num_layers + 1)])
        self.dense_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else hidden_channels, hidden_channels, kernels, padding=kernels // 2),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU()
            ) for i in range(num_layers - 1)])
        self.relu = nn.ReLU()

    def forward(self, x):
        y = [self.out_convs[0](x)]
        for i in range(self.num_layers):
            x = self.dense_layers[i](x)
            y.append(self.out_convs[i + 1](x))
        return self.relu(torch.cat(y, 1))

class MomentumLayerNorm(nn.Module):
    def __init__(self, normalized_shape, momentum=(0.1816, 0.9)):
        super(MomentumLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape[-1], elementwise_affine=False)
        self.alpha1 = nn.Parameter(
            torch.full(normalized_shape[:-1] + [1], momentum[0]))  # torch.full([channels, 1, 1], 0.1816)
        self.alpha2 = nn.Parameter(torch.full(normalized_shape[:-1] + [1], momentum[1]))
        self.beta = nn.Parameter(torch.ones(normalized_shape[:-1] + [1]))
        self.gamma = nn.Parameter(torch.zeros(normalized_shape[:-1] + [1]))

    def forward(self, x):
        # print(x.min(),x.max())
        y = self.alpha1 * x + self.alpha2 * (self.beta * self.ln(x) + self.gamma)
        # print(y.min(), y.max())
        return y


class TripletDropout(nn.Module):
    def __init__(self, p1=0.1816, p2=0.1816):
        super(TripletDropout, self).__init__()
        assert 0 <= p1 <= 1 and 0 <= p1 <= 1 and p1 + p2 <= 1, "p1 or p2 are nor proper prbablities!"
        assert p1 + 2 * p2 == 1, "p1+2*p2 should not equal 1."
        self.p1 = p1
        self.p2 = p2

    def forward(self, x):
        if self.training:
            drop_mask = torch.rand_like(x)
            if self.p1 > 0:
                x[drop_mask < self.p1] = 0.
            if self.p2 > 0:
                x[drop_mask > 1 - self.p2] *= -1.
            return x / (1 - self.p1 - 2 * self.p2)
        return x


class ResInstanceNorm2d(nn.Module):
    def __init__(self, channels):
        super(ResInstanceNorm2d, self).__init__()
        self.bn = nn.InstanceNorm2d(channels)
        self.alpha = nn.Parameter(torch.tensor(0.1773))
        # self.beta = norm_decay
        # self.beta = nn.Parameter(torch.tensor(0.1816 if first_norm else 0.175))

    def forward(self, x):
        # x_max = x.detach().abs().transpose(0, 1).flatten(1).max(1).values[:, None, None]
        # y = self.beta * self.bn(x)
        # if self.alpha is not None:
        #     y += self.alpha * x
        return self.alpha * x + 0.1773 * self.bn(x) # self.alpha * x + (1 - self.alpha) * self.beta * self.bn(x)  # 0.175 * x + 0.175 * self.bn(x) # 0.1816 * (x + self.bn(x))  #


class PreMinMax(nn.Module):
    def forward(self, x):
        min_max = x.detach().transpose(0, 1).flatten(1)
        x_min, x_max = min_max.min(1).values[:, None, None], min_max.max(1).values[:, None, None]
        return (2 * x - x_max - x_min) / (x_max - x_min)


class ReLUBatchNorm2d(nn.Module):
    def __init__(self, channels, norm_decay=0.1773, pool=False):
        super(ReLUBatchNorm2d, self).__init__()
        self.alpha1 = nn.Parameter(torch.tensor(0.01))
        self.alpha2 = nn.Parameter(torch.tensor(0.01))
        self.beta1 = nn.Parameter(torch.tensor(0.99))
        self.beta2 = nn.Parameter(torch.tensor(0.99))
        self.bn = nn.BatchNorm2d(channels, affine=True)
        self.gamma = norm_decay
        self.pool = pool

    def forward(self, x):
        x1 = self.alpha1 * x + self.beta1 * F.relu(x)
        x2 = self.alpha2 * x + self.beta2 * F.relu(x)
        if self.pool:
            x1 = F.max_pool2d(x1, 2)
            x2 = F.max_pool2d(x2, 2)
        return self.gamma * (x1 + self.bn(x2))


class SeparateReLU(nn.Module):
    def __init__(self, channels):
        super(SeparateReLU, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, x):
        x1 = self.relu(x)
        x2 = self.relu(-x)
        return self.conv(torch.cat([x1, -x2], 1))


class PReLU(nn.Module):
    def __init__(self, init=0.01, reverse=False):
        super(PReLU, self).__init__()
        self.reverse = reverse
        self.relu = nn.ReLU()
        self.alpha = nn.Parameter(torch.tensor(init))  # nn.Parameter(torch.full([channels, 1, 1], init)) #

    def forward(self, x):
        # if self.alpha < 0:
        #     self.alpha.data = torch.tensor(0.).cuda()
        # elif self.alpha > 1:
        #     self.alpha.data = torch.tensor(1.).cuda()
        return self.alpha * x + (1 - self.alpha) * (-self.relu(-x) if self.reverse else self.relu(x))

class SphereNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-5, momentum=0.1816, affine=False):
        super(SphereNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_norm', torch.ones(channels, 1, 1) if momentum > 0. else None)
        self.affine = nn.Parameter(torch.ones(channels, 1, 1)) if affine else None

    def forward(self, x):
        if self.running_norm is not None and not self.training:
            norm = self.running_norm
        else:
            norm = x.pow(2).mean([0, 2, 3], keepdim=True)
            if self.running_norm is not None:
                with torch.no_grad():
                    self.running_norm = self.momentum * norm + (1 - self.momentum) * self.running_norm
        x = x / torch.sqrt(norm + self.eps)
        if self.affine is not None:
            x = x * self.affine
        return x


class FullNorm(nn.Module):
    def __init__(self):
        super(FullNorm, self).__init__()
        # self.alpha = nn.Parameter(torch.tensor(1.))
        # self.beta = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return (x - x.mean()) / (x.std() + 1e-5)


class RawResNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm_scale=None, short_cut_mode=1):
        super(RawResNet, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels, affine=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm_scale = norm_scale
        self.short_cut_mode = short_cut_mode
        self.relu = nn.ReLU()
        self.short_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Parameter(torch.tensor(1.))
        self.short = in_channels == out_channels

    def forward(self, x):
        y = self.bn(x)
        if self.norm_scale is not None:
            y *= self.norm_scale
        y = self.conv(y)
        if self.short_cut_mode == 0:
            y = self.relu((self.short_conv * x if self.short else self.short_conv(x)) + y)
        elif self.short_cut_mode == 1:
            y = (self.short_conv * x if self.short else self.short_conv(x)) + self.relu(y)
        else:
            y = self.relu(y)
        return y


# class ResAttention2d(nn.Module):
#     def __init__(self, image_sizes, channels, k=8):
#         super(ResAttention2d, self).__init__()
#         self.row_attn = nn.Sequential(
#             ResBatchNorm2d(channels, 0.1816),
#             nn.Conv2d(channels, k * channels, (image_sizes[0], 1)),
#             nn.Softmax(-1),
#         )
#         self.col_attn = nn.Sequential(
#             ResBatchNorm2d(channels, 0.1816),
#             nn.Conv2d(channels, k * channels, (1, image_sizes[1])),
#             nn.Softmax(-2),
#         )
#         self.norm_momentum = nn.Parameter(torch.tensor(0.1816))
#         self.bn = nn.BatchNorm2d(channels)
#         self.channels = channels
#         self.k = k
#
#     def forward(self, x):
#         return self.col_attn(x).view(x.size(0), self.channels, self.k, -1).transpose(-2, -1).matmul(self.row_attn(x).view(x.size(0), self.channels, self.k, -1)) * self.bn(x)


class GerneralNorm(nn.Module):
    def __init__(self, norm_dims, affine_shape=None, affine_dim_indices=None, scale=False, bias=False):
        super(GerneralNorm, self).__init__()
        self.norm_dims = norm_dims
        if affine_dim_indices is not None and affine_shape is not None:
            if '__getitem__' not in dir(affine_dim_indices):
                affine_dims = [affine_dim_indices]
            if '__getitem__' not in dir(affine_shape):
                affine_shape = [affine_shape]
            a = torch.tensor(affine_dim_indices).sort().values
            a -= a[0]

    def forward(self, x):
        return self.layer(x)


class TrueBatchNorm2d(nn.Module):
    def __init__(self, channels, momentum=0.1816, quantile_num=20):
        super(TrueBatchNorm2d, self).__init__()
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))
        self.register_buffer('running_mean', torch.zeros(1, channels, 1, 1))
        self.register_buffer('running_var', torch.zeros(1, channels, 1, 1))
        self.register_buffer('quantiles', torch.linspace(1 / quantile_num, 1 - 1 / quantile_num, quantile_num - 1) if quantile_num > 0 else None)

    def forward(self, x):
        if self.training:
            x_detach = x.detach().clone()
            if self.quantiles is None:
                avg = x_detach.mean([0, 2, 3], keepdim=True)
            else:
                avg = x_detach.transpose(0, 1).flatten(1).quantile(self.quantiles, 1, keepdims=True).mean(0, keepdims=True).unsqueeze(-1)
            var = ((x - avg) ** 2).mean([0, 2, 3], keepdim=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * avg
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            avg = self.running_mean
            var = self.running_var
        return self.weight * (x - avg) / (var + 1e-5).sqrt() + self.bias


class CentroidBatchNorm2d(nn.Module):
    def __init__(self, channels, momentum=0.1816):
        super(CentroidBatchNorm2d, self).__init__()
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))
        self.register_buffer('running_mean', torch.zeros(channels, 1, 1))
        self.register_buffer('running_var', torch.zeros(channels, 1, 1))

    def forward(self, x):
        if self.training:
            y = x.transpose(0, 1).flatten(1).sort().values  # (C, N*H*W)
            z = y[:, 1:-1]  # (C, N*H*W-2)
            y_diff = y.detach().diff()  # (C, N*H*W-1)
            y_coef = torch.exp(-y_diff[:, :-1] - y_diff[:, 1:])  # (C, N*H*W-2)
            y_coef /= y_coef.sum(1, keepdim=True)  # (C, N*H*W-2)
            avg = (y_coef * z.detach()).sum(1, keepdim=True)  # (C, 1)
            var = ((z - avg) ** 2 * y_coef).mean(1, keepdim=True).unsqueeze(-1)  # (C, 1, 1)
            avg = avg.unsqueeze(-1)  # (C, 1, 1)
            self.running_mean *= 1 - self.momentum
            self.running_mean += self.momentum * avg
            self.running_var *= 1 - self.momentum
            self.running_var += self.momentum * var
        else:
            avg = self.running_mean
            var = self.running_var
        return self.weight * (x - avg) / (var + 1e-5).sqrt() + self.bias


class ChannelSoftmax(nn.Module):
    def __init__(self, dims):
        super(ChannelSoftmax, self).__init__()
        self.dims = dims

    def forward(self, x):
        y = -F.relu(x.detach().max(1, keepdims=True).values)
        z = (x + y).exp()
        return z / (y.exp() + z.sum(1, keepdims=True))


class NeuronShortcut(nn.Module):
    def __init__(self, seq_len, in_channels, out_channels, in_features, out_features):
        super(NeuronShortcut, self).__init__()
        if in_channels != out_channels:
            self.left_shortcut = nn.Parameter(sqrt(3 / sqrt(out_channels * in_channels)) * (2 * torch.rand(seq_len, out_channels, in_channels) - 1))
        else:
            self.register_parameter('left_shortcut', None)
        if in_features != out_features:
            self.right_shortcut = nn.Parameter(sqrt(3 / sqrt(out_features * in_features)) * (2 * torch.rand(seq_len, in_features, out_features) - 1))
        else:
            self.register_parameter('right_shortcut', None)
        if self.left_shortcut is None and self.right_shortcut is None:
            self.momentum = nn.Parameter(torch.tensor(1.))
        else:
            self.register_parameter('momentum', None)

    def forward(self, x):
        # computing shortcut
        if self.momentum is not None:
            x = self.momentum * x
        else:
            if self.left_shortcut is not None:
                x = self.left_shortcut.matmul(x)
            if self.right_shortcut is not None:
                x = x.matmul(self.right_shortcut)
        return x


class TrueNeuron(nn.Module):
    def __init__(self, seq_len, in_channels, out_channels, in_features, out_features, norm_decay, last_layer=False):
        super(TrueNeuron, self).__init__()
        self.last_layer = last_layer
        self.norm_decay = norm_decay
        self.pos_bias1 = nn.Parameter(torch.zeros(seq_len, in_features))
        self.pos_bias2 = nn.Parameter(torch.zeros(seq_len, in_features))
        self.pos_bias3 = nn.Parameter(torch.zeros(seq_len, in_features))
        self.pre_norm = nn.LayerNorm([seq_len, in_features], elementwise_affine=False)
        self.query1 = nn.Parameter(sqrt(3 / in_features) * (2 * torch.rand(seq_len, in_channels, in_features) - 1))
        self.query2 = nn.Parameter(sqrt(3 / sqrt(out_channels * in_channels)) * (2 * torch.rand(seq_len, out_channels, in_channels) - 1))
        self.key = nn.Parameter(sqrt(3 / in_features) * (2 * torch.rand(seq_len, in_channels, in_features) - 1))
        self.value = nn.Parameter(sqrt(3 / sqrt(out_features * in_features)) * (2 * torch.rand(in_channels, in_features, out_features) - 1))
        self.query_weight = nn.Parameter(sqrt(3 / out_channels) * (2 * torch.rand(seq_len, out_channels, 1) - 1))
        self.key_weight = nn.Parameter(sqrt(3 / in_channels) * (2 * torch.rand(seq_len, 1, in_channels) - 1))
        self.short_cut = NeuronShortcut(seq_len, in_channels, out_channels, in_features, out_features)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(-2)

    def forward(self, x):
        # computing attetion matrix
        # x.shape=(N, C_in, L, d_in)
        if isinstance(x, torch.Tensor):
            x_shortcut = x
        else:
            x_shortcut = x[0]
            x = x[1]
        x_shortcut = (self.pre_norm(x_shortcut) + self.pos_bias1) / self.norm_decay  # (N, C_in, L, d_in)
        x_shortcut = self.short_cut(x_shortcut.transpose(1, 2))  # (N, L, C_out, d_out)
        x = self.pre_norm(x)
        x_value = (x + self.pos_bias2) / self.norm_decay  # (N, C_in, L, d_in)
        value = self.relu(x_value.matmul(self.value)).transpose(1, 2)  # (N, L, C_in, d_out)
        # computing attention matrix
        x_attn = (x + self.pos_bias3) / self.norm_decay  # (N, C_in, L, d_in)
        q = x_attn.transpose(1, 2).mul(self.query1).sum(-1, keepdims=True)  # (N, L, C_in, 1)
        q = F.softmax(self.query2.matmul(q), -2) * self.query_weight # (N, L, C_out, 1)
        k = F.softmax(x_attn.transpose(1, 2).mul(self.key).sum(-1).unsqueeze(-2), -1) * self.key_weight # (N, L, 1, C_in)
        attn = q.matmul(k.matmul(value)) # (N, L, C_out, d_out)
        return self.relu(x_shortcut) + attn if self.last_layer else (x_shortcut, self.relu(x_shortcut) + attn) # (N, L, C_out, d_out)


# class TrueNeuron(nn.Module):
#     def __init__(self, seq_len, in_channels, out_channels, in_features, out_features, max_attn_weight_value, max_norm_values):
#         super(TrueNeuron, self).__init__()
#         self.max_norm_values = max_norm_values
#         self.query1 = nn.Parameter(sqrt(3 / in_features) * (2 * torch.rand(seq_len, in_channels, in_features) - 1))
#         self.query2 = nn.Parameter(sqrt(3 / in_channels) * (2 * torch.rand(seq_len, in_channels, out_channels) - 1))
#         self.key = nn.Parameter(sqrt(3 / in_features) * (2 * torch.rand(seq_len, in_channels, in_features) - 1))
#         self.value = nn.Parameter(sqrt(3 / in_features) * (2 * torch.rand(in_channels, in_features, out_features) - 1))
#         self.attn_weight = nn.Parameter(torch.randn(seq_len, out_channels, in_channels) / max_attn_weight_value)
#         self.attn_norm = nn.LayerNorm([seq_len, out_channels], elementwise_affine=False)
#         self.value_norm = nn.LayerNorm([seq_len, out_features], elementwise_affine=False)
#         self.attn_position_bias = nn.Parameter(torch.zeros(seq_len, out_channels))
#         self.value_position_bias = nn.Parameter(torch.zeros(seq_len, out_features))
#         self.short_cut = NeuronShortcut(seq_len, in_channels, out_channels, in_features, out_features)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(-1)
#
#     def forward(self, x):
#         # computing attetion matrix
#         # x.shape=(N, L, C_in, d_in)
#         q = x.mul(self.query1).sum(-1).unsqueeze(-2).matmul(self.query2) # (N, L, 1, C_out) .view(x.shape[:-2] + (1, -1))
#         k = x.mul(self.key).sum(-1, keepdims=True) # (N, L, C_in, 1)
#         attn = k.matmul(q).transpose(1, 2) # (N, C_in, L, C_out)
#         attn = self.softmax((self.attn_norm(attn)  + self.attn_position_bias) / self.max_norm_values[0])  # (N, C_in, L, C_out)
#         attn = attn.permute([0, 2, 3, 1])  # (N, L, C_out, C_in)
#         attn *= self.attn_weight  # (N, L, C_out, C_in)
#         # computing value matrix
#         value = x.transpose(1, 2).matmul(self.value)  # (N, C_in, L, d_out)
#         value = self.relu((self.value_norm(value) + self.value_position_bias) / self.max_norm_values[1]).transpose(1, 2)  # (N, L, C_in, d_out)
#         return self.short_cut(x) + attn.matmul(value) # (N, L, C_out, d_out)


class DropoutAll(nn.Module):
    def __init__(self, p=0.1816):
        super(DropoutAll, self).__init__()
        assert 0 < p < 1, "p should be a proper probablity!"
        self.p = p
        self.drop_scale = 1. / (1. - p)

    def forward(self, x):
        if self.training:
            return torch.zeros_like(x) if torch.rand(1) < self.p else self.drop_scale * x
        return x


class MetaDenseNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, kernels=3):
        super(MetaDenseNet, self).__init__()
        out_channels = int(pow(2, num_layers))  # num_layers * (num_layers + 1) // 2 #
        self.num_layers = num_layers - 1
        self.out_convs = nn.ModuleList(
            [nn.Sequential(ResBatchNorm2d(in_channels if i == out_channels else hidden_channels, 0.1773),
                           ResConv2d(in_channels if i == out_channels else hidden_channels, out_channels // i, kernels, 1.),
                           ) for i in 2 ** torch.arange(num_layers, 0, -1)]
        )
        # self.out_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels if i == 1 else hidden_channels, i*factor,
        #                                                         kernels, padding=kernels // 2),
        #                                               nn.BatchNorm2d(i*factor)) for i in range(1, num_layers + 1)])
        self.dense_layers = nn.ModuleList(
            [nn.Sequential(
                ResBatchNorm2d(in_channels if i == 0 else hidden_channels, 0.1773),
                ResConv2d(in_channels if i == 0 else hidden_channels, hidden_channels, kernels, 1.),
                ResReLU(1.1),
            ) for i in range(num_layers - 1)])
        self.relu = ResReLU(1.1)

    def forward(self, x):
        y = [self.out_convs[0](x)]
        for i in range(self.num_layers):
            x = self.dense_layers[i](x)
            y.append(self.out_convs[i + 1](x))
        return self.relu(torch.cat(y, 1))


class MetaPreReLU(nn.Module):
    def __init__(self, in_channels, out_channels, norm_decay, kernel_size, prob, pool=False, relu=True):
        super(MetaPreReLU, self).__init__()
        self.layer = nn.Sequential(
            ReLUBatchNorm2d(in_channels, norm_decay, pool) if relu else ResBatchNorm2d(in_channels, norm_decay),
            ResConv2d(in_channels, out_channels, kernel_size, prob),
        )

    def forward(self, x):
        return self.layer(x)


class MetaPreReLUNet(nn.Module):
    def __init__(self, num_layers, init_channels, kernel_size=3, norm_decay=0.1773):
        super(MetaPreReLUNet, self).__init__()
        layers = [MetaPreReLU(init_channels, num_layers[0][0], norm_decay, kernel_size, False, False, False)]
        k = -1
        for i, j in num_layers:
            if k > 0:
                layers.append(MetaPreReLU(k, i, norm_decay, kernel_size, False, True))
            for id in range(j):
                layers.append(MetaPreReLU(i, i, norm_decay, kernel_size, id % 2 == 1)) # id % 2 == 1
            k = i
        layers.append(ReLUBatchNorm2d(num_layers[-1][0], norm_decay, True))
        # layers.append(ResReLU())
        # layers.append(nn.MaxPool2d(2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# class MetaSeparateNet(nn.Module):
#     def __init__(self, num_layers, init_channels, expand, kernel_size=3, norm_decay=0.1773):
#         super(MetaSeparateNet, self).__init__()
#         layers = [MetaResConv2d(init_channels[0], init_channels[1], norm_decay, kernel_size, False), SeparateReLU()]
#         k = init_channels[1]
#         for i, j in enumerate(num_layers):
#             if i > 0:
#                 layers.append(MetaSeparateConv2d(k, norm_decay, kernel_size, expand[i - 1], True))
#             if i == 0 or expand[i - 1]:
#                 k *= 2
#             for id in range(j):
#                 layers.append(MetaSeparateConv2d(k, norm_decay, kernel_size, False, i == len(num_layers) - 1 and id == j - 1))
#         self.layers = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.layers(x)


class DeepMergeConv2d(nn.Module):
    def __init__(self, num_layers, init_channels, kernel_size=3, norm_decay=0.1773):
        super(DeepMergeConv2d, self).__init__()
        layers = [REConv2d(init_channels, num_layers[0][0], norm_decay, kernel_size)]
        k = -1
        for i, j in num_layers:
            if k > 0:
                layers.append(ResBatchNorm2d(k, norm_decay))
                layers.append(nn.Conv2d(k, i, 2, 2))
                layers.append(ResReLU())
            for _ in range(j):
                layers.append(REConv2d(i, i, norm_decay, kernel_size))
            k = i
        layers.append(ResBatchNorm2d(num_layers[-1][0], norm_decay))
        layers.append(nn.Conv2d(num_layers[-1][0], num_layers[-1][0], 2, 2))
        layers.append(ResReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DenseRatio2d(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, kernels=3):
        super(DenseRatio2d, self).__init__()
        out_channels = int(pow(2, num_layers))  # num_layers * (num_layers + 1) // 2 #
        self.num_layers = num_layers - 1
        self.out_convs = nn.ModuleList(
            [nn.Sequential(# ResBatchNorm2d(in_channels if i == out_channels else hidden_channels),
                           ResLayerhNorm(),
                           ResConv2d(in_channels if i == out_channels else hidden_channels, out_channels // i, kernels),
                           ) for i in 2 ** torch.arange(num_layers, 0, -1)])
        # self.out_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels if i == 1 else hidden_channels, i*factor,
        #                                                         kernels, padding=kernels // 2),
        #                                               nn.BatchNorm2d(i*factor)) for i in range(1, num_layers + 1)])
        self.dense_layers = nn.ModuleList(
            [nn.Sequential(
                ResLayerhNorm(),
                # ResBatchNorm2d(in_channels if i == 0 else hidden_channels),
                ResConv2d(in_channels if i == 0 else hidden_channels, hidden_channels, kernels),
                ResReLU(),  # nn.PReLU(init=0.),  # nnn.PReLU(num_parameters=channels, init=0.),
            ) for i in range(num_layers - 1)])
        self.relu = ResReLU()

    def forward(self, x):
        y = [self.out_convs[0](x)]
        for i in range(self.num_layers):
            x = self.dense_layers[i](x)
            y.append(self.out_convs[i + 1](x))
        return self.relu(torch.cat(y, 1))


class DenseRes2d(nn.Module):
    def __init__(self, num_res_layers, in_channels, out_channels=None):
        super(DenseRes2d, self).__init__()
        self.num_res_layers = num_res_layers - 1
        if out_channels is None:
            out_channels = in_channels
        self.first_layer = nn.Sequential(
                ResBatchNorm2d(in_channels),
                ResConv2d(in_channels, out_channels, 3),
                ResReLU(),  # nn.PReLU(init=0.),  # nnn.PReLU(num_parameters=channels, init=0.),
            )
        self.res_conv = nn.ModuleList(
            [nn.Sequential(
                ResBatchNorm2d(out_channels),
                ResConv2d(out_channels, out_channels, 3),
            ) for i in range(num_res_layers - 1)])
        self.short_cut = nn.ModuleList(
            [nn.Sequential(
                ResBatchNorm2d(in_channels),
                ResConv2d(in_channels, out_channels, 1),
                # nn.Conv2d(in_channels, out_channels, 3),
            ) for _ in range(num_res_layers - 1)])
        self.relu_layers = nn.ModuleList([ResReLU() for _ in range(num_res_layers - 1)])
        self.head_affine = nn.Parameter(torch.ones(num_res_layers - 1))
        self.tail_affine = nn.Parameter(torch.ones(num_res_layers - 1))

    def forward(self, x):
        y = self.first_layer(x)
        z = 0.
        for i in range(self.num_res_layers):
            y = self.relu_layers[i](self.short_cut[i](x) + self.res_conv[i](y))
            z += self.tail_affine[i] * y
        return z  # self.relu(z)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super(TransformerBlock, self).__init__()
        self.norm_coef = sqrt(d_model // num_heads)
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(d_model)
        # self_attention params
        self.query = nn.Linear(d_model, d_model, False)
        self.key = nn.Linear(d_model, d_model, False)
        self.value = nn.Linear(d_model, d_model, False)
        self.concat_lin = nn.Linear(d_model, d_model, False)
        self.softmax = nn.Softmax(-1)
        # mlp params
        self.mlp_lin1 = nn.Linear(d_model, d_model * 4)
        self.mlp_lin2 = nn.Linear(d_model * 4, d_model)
        self.relu = nn.ReLU()
        self.norm2 = nn.LayerNorm(d_model)

    def self_attn(self, x):
        query = self.query(x).chunk(self.num_heads, -1)
        key = self.key(x).transpose(-2, -1).chunk(self.num_heads, -2)
        value = self.value(x).chunk(self.num_heads, -1)
        return self.norm1(x + self.concat_lin(
            torch.cat([self.softmax(q.matmul(k) / self.norm_coef).matmul(v) for q, k, v in zip(query, key, value)],
                      -1)))

    def mlp(self, x):
        return self.norm2(x + self.mlp_lin2(self.relu(self.mlp_lin1(x))))

    def forward(self, x):
        x = self.self_attn(x)
        return self.mlp(x)


class PositionEncoding(nn.Module):
    def __init__(self, seq_len=16, d_model=128, base=10000):
        super(PositionEncoding, self).__init__()
        pos_vector = torch.arange(seq_len, dtype=torch.float32)[:, None]
        model_vector = torch.pow(base, torch.arange(d_model // 2).repeat_interleave(2) * 2 / d_model)
        pos_code = torch.sin(pos_vector / model_vector + (pi / 2) * torch.tensor([i % 2 for i in range(d_model)]))
        self.register_buffer('pos_code', pos_code)

    def forward(self, x):
        return x + self.pos_code



def softxor(x, y, eps=1e-3):
    y = x.tanh() * y.tanh()
    return 0.5 * torch.log((1 + eps + y) / (1 + eps - y))


def quadratic_softxor(x, y, eps=1e-5):
    x /= (1 + x ** 2).sqrt()
    y /= (1 + y ** 2).sqrt()
    y *= x
    return y / (1 + eps - y ** 2).sqrt()


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


class AttnConv2d(nn.Module):
    def __init__(self, image_size, in_channels, out_channels, attn_kernels=3, kernels=3, conv_momentum=None, batch_norm=True):
        super(AttnConv2d, self).__init__()
        if in_channels != out_channels:
            self.conv_momentum = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.conv_momentum = nn.Parameter(torch.tensor(1. if conv_momentum is None else conv_momentum))
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernels, padding=kernels // 2, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernels, padding=kernels // 2, bias=False)
        self.unfold1 = nn.Unfold(attn_kernels, stride=attn_kernels)
        self.unfold2 = nn.Unfold(attn_kernels, stride=attn_kernels)
        # self.prenorm1 = ResLayerNorm2d(image_size, 0.1816, bias=False, short_cut=True)
        # self.prenorm2 = ResLayerNorm2d(image_size, 0.1816, bias=False, short_cut=True)
        # self.prenorm3 = ResLayerNorm2d(image_size, 0.1816, bias=False, short_cut=True)
        # if batch_norm:
        #     self.norm1 = ResBatchNorm2d(in_channels, 0.1816, short_cut=False)
        #     self.norm2 = ResBatchNorm2d(out_channels, 0.1816, short_cut=False)
        #     self.norm3 = ResBatchNorm2d(out_channels, 0.1816, short_cut=True)
        # else:
        #     self.norm1 = ResLayerNorm2d(image_size, 0.1816, short_cut=False)
        #     self.norm2 = ResLayerNorm2d(image_size, 0.1816, short_cut=False)
        #     self.norm3 = ResLayerNorm2d(image_size, 0.1816, short_cut=True)
        self.norm = ResLayerNorm2d(image_size, 0.1816, short_cut=False)
        self.batch_norm = batch_norm
        self.attn_kernels = attn_kernels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.softmax = nn.Softmax(0)
        self.attn_value_kernel = nn.Parameter(0.1816 * torch.randn(out_channels, in_channels, attn_kernels, attn_kernels))
        self.dropout = nn.Dropout()

    def forward(self, x):
        y1 = self.unfold1(self.conv1(x))
        y1 = y1.unflatten(1, (-1, self.attn_kernels * self.attn_kernels)).transpose(0, 2).flatten(2) #.flatten(0, 1)
        # y1 = y1.view(y1.size(0), -1, y1.size(2) * self.attn_kernels).transpose(0, 1).flatten(1)
        y2 = self.unfold2(self.conv2(x))
        y2 = y2.unflatten(1, (-1, self.attn_kernels * self.attn_kernels)).transpose(0, 2).flatten(2) #.flatten(0, 1)
        # y2 = y2.view(y2.size(0), -1, y2.size(2) * self.attn_kernels).transpose(0, 1).flatten(1)
        attn_kernls = y1.matmul(y2.transpose(-2, -1))
        # attn_kernls = attn_kernls.mean(0)
        # attn_kernls = self.softmax(attn_kernls.view(-1, self.attn_kernels * self.attn_kernels * self.in_channels))
        # attn_kernls = self.softmax(attn_kernls.transpose(0, 1).flatten(1))

        # attn_kernls = self.softmax(attn_kernls.transpose(0, 1).flatten(1))
        # attn_kernls = self.softmax(attn_kernls.transpose(0, 1) / sqrt(self.in_channels)) # .flatten(0, 1) * self.attn_kernels * self.attn_kernels
        attn_kernls = self.softmax(attn_kernls.flatten(0, 1) / sqrt(self.in_channels * self.attn_kernels * self.attn_kernels)) # .transpose(0, 1) * self.attn_kernels * self.attn_kernels
        # attn_kernls = attn_kernls.permute([2, 0, 1]).unflatten(-1, (self.attn_kernels, -1))
        attn_kernls = attn_kernls.unflatten(0, (self.attn_kernels * self.attn_kernels, -1)).transpose(0, 2).unflatten(-1, (self.attn_kernels, -1))#  .transpose(0, 1) # contiguous()..view(-1, self.in_channels, self.attn_kernels, self.attn_kernels)
        # attn_kernls *= self.attn_value_kernel
        # x = self.prenorm3(x)
        attn_value = F.conv2d(x, attn_kernls, padding=self.attn_kernels // 2)
        x = self.conv_momentum * x if type(self.conv_momentum) is nn.parameter.Parameter else self.conv_momentum(x)
        return x + self.norm(attn_value)


class XorAttnConv2d(nn.Module):
    def __init__(self, image_size, in_channels, out_channels, attn_kernels=3, kernels=3, conv_momentum=None, batch_norm=True):
        super(XorAttnConv2d, self).__init__()
        if in_channels != out_channels:
            self.conv_momentum = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.conv_momentum = nn.Parameter(torch.tensor(1. if conv_momentum is None else conv_momentum))
        self.attn_conv1 = AttnConv2d(image_size, in_channels, out_channels, attn_kernels, kernels, conv_momentum, batch_norm)
        self.attn_conv2 = AttnConv2d(image_size, in_channels, out_channels, attn_kernels, kernels, conv_momentum,
                                     batch_norm)
        self.attn_conv3 = AttnConv2d(image_size, in_channels, out_channels, attn_kernels, kernels, conv_momentum,
                                     batch_norm)
        self.norm1 = ResLayerNorm2d(image_size, 0.1816, short_cut=False)
        self.norm2 = ResLayerNorm2d(image_size, 0.1816, short_cut=False)
        self.norm3 = ResLayerNorm2d(image_size, 0.1816, short_cut=False)

    def forward(self, x):
        y1 = self.attn_conv1(x)
        y2 = self.attn_conv2(x)
        y3 = self.attn_conv3(x)
        x = self.conv_momentum * x if type(self.conv_momentum) is nn.parameter.Parameter else self.conv_momentum(x)
        return x + self.norm1(y1) + self.norm2(y2) * self.norm3(y3)

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
        # y = self.norm1(self.feature_map_stack(y1)) + self.norm2(self.feature_map_stack(y2)) * self.norm3(self.feature_map_stack(y3)) if self.xor else self.norm1(self.feature_map_stack(y1)) # softxor(self.bn1(y2), self.bn2(y3))
        y = self.norm1(y1) + self.norm2(y2) * self.norm3(y3) if self.xor else self.norm1(y1)
        return x + self.feature_map_stack(y)


class ResConv2d(nn.Module):
    def __init__(self, image_size, in_channels, out_channels, norm_scale, kernels=3, conv_momentum=None, num_heads=1, pre_norm=False, batch_norm=True, xor=True):
        super(ResConv2d, self).__init__()
        if in_channels != out_channels:
            self.conv_momentum = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.conv_momentum = nn.Parameter(torch.tensor(1.))
        if '__getitem__' in dir(num_heads):
            self.num_heads = num_heads
        else:
            self.num_heads = (num_heads, num_heads)
        if batch_norm:
            self.conv1 = nn.Sequential(
                ResBatchNorm2d(in_channels // 2 if in_channels >= 10 else in_channels, norm_scale, short_cut=0.01),
                nn.Conv2d(in_channels // 2 if in_channels >= 10 else in_channels, out_channels // 2, kernels,
                          padding=kernels // 2, bias=False),
                ResBatchNorm2d(out_channels // 2, norm_scale, short_cut=0.6)
            )
            self.conv2 = nn.Sequential(
                ResBatchNorm2d(in_channels // 2 if in_channels >= 10 else in_channels, norm_scale, short_cut=0.01),
                nn.Conv2d(in_channels // 2 if in_channels >= 10 else in_channels, out_channels // 2, kernels,
                          padding=kernels // 2, bias=False),
                ResBatchNorm2d(out_channels // 2, norm_scale, short_cut=0.6)
            )
            self.conv3 = nn.Sequential(
                ResBatchNorm2d(in_channels // 2 if in_channels >= 10 else in_channels, norm_scale, short_cut=0.01),
                nn.Conv2d(in_channels // 2 if in_channels >= 10 else in_channels, out_channels // 2, kernels,
                          padding=kernels // 2, bias=False),
                ResBatchNorm2d(out_channels // 2, norm_scale, short_cut=0.6)
            )
            self.conv4 = nn.Sequential(
                ResBatchNorm2d(in_channels // 2 if in_channels >= 10 else in_channels, norm_scale, short_cut=0.01),
                nn.Conv2d(in_channels // 2 if in_channels >= 10 else in_channels, out_channels // 2, kernels,
                          padding=kernels // 2, bias=False),
                ResBatchNorm2d(out_channels // 2, norm_scale, short_cut=0.6)
            )
        # else:
        #     self.conv1 = nn.Sequential(
        #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.01),
        #         nn.Conv2d(in_channels // 2 if in_channels >= 10 else in_channels, out_channels, kernels,
        #                   padding=kernels // 2, bias=False),
        #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.6),
        #     )
        #     self.conv2 = nn.Sequential(
        #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.01),
        #         nn.Conv2d(in_channels // 2 if in_channels >= 10 else in_channels, out_channels, kernels,
        #                   padding=kernels // 2, bias=False),
        #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.6),
        #     )
        #     self.conv3 = nn.Sequential(
        #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.01),
        #         nn.Conv2d(in_channels // 2 if in_channels >= 10 else in_channels, out_channels, kernels,
        #                   padding=kernels // 2, bias=False),
        #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.6),
        #     )
        #     self.conv4 = nn.Sequential(
        #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.01),
        #         nn.Conv2d(in_channels // 2 if in_channels >= 10 else in_channels, out_channels, kernels,
        #                   padding=kernels // 2, bias=False),
        #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.6),
        #     )
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

    def channel_split(self, x):
        if x.size(1) >= 10:
            x1, x2 = x.chunk(2, 1)
            # x1, x2 = x.roll(self.in_channels // 4, 1).chunk(2, 1)
        else:
            # x1 = x.clone()
            # x2 = x.clone()
            # x3 = x.clone()
            # x4 = x.clone()
            # x1 = x2 = x3 = x4 =x
            x1 = x2 = x
        return x1, x2 #, x3, x4

    def forward(self, x):
        # x_norm = self.norm_scale * self.pre_norm(x)
        x1, x2 = self.channel_split(x) #  + self.pos_code
        y1 = self.conv1(x1)
        y2 = self.conv2(x1)
        y3 = self.conv3(x2)
        y4 = self.conv4(x2)
        x = 0.99 * self.conv_momentum * x if type(self.conv_momentum) is nn.parameter.Parameter else self.conv_momentum(x)
        y = torch.cat([(y1 + y3) / sqrt(2), y2 * y4], 1) # / sqrt(3)
        return x + self.feature_map_stack(y)

    class AttnConv2d(nn.Module):
        def __init__(self, image_size, in_channels, out_channels, hidden_kernels=3, attn_kernels=3, softmax=True):
            super(AttnConv2d, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, in_channels, hidden_kernels, padding=hidden_kernels // 2, bias=False)
            self.conv2 = nn.Conv2d(in_channels, out_channels, hidden_kernels, padding=hidden_kernels // 2, bias=False)
            # self.conv3 = nn.Conv2d(in_channels, in_channels, hidden_kernels, padding=hidden_kernels // 2, bias=False)
            self.unfold1 = nn.Unfold(attn_kernels, stride=attn_kernels)
            self.unfold2 = nn.Unfold(attn_kernels, stride=attn_kernels)
            self.unfold3 = nn.Unfold(attn_kernels, padding=attn_kernels // 2)
            self.attn_kernels = attn_kernels
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.softmax = nn.Softmax(2) if softmax else None
            # self.pos_code = nn.Parameter(torch.randn([in_channels] + image_size))
            self.attn_weight = nn.Parameter(
                nn.Conv2d(in_channels, out_channels, attn_kernels, bias=False).weight.data.flatten(1))

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
                self.in_channels * self.attn_kernels * self.attn_kernels)  # + self.attn_weight / self.attn_kernels
            # print(attn_kernls.shape)
            if self.softmax is not None:
                attn_kernls = self.softmax(attn_kernls)  # attn_kernls.transpose(1, 2).flatten(1, 2)
                # attn_kernls = attn_kernls.unflatten(2, (-1, self.attn_kernels * self.attn_kernels))
            attn_kernls = attn_kernls * self.attn_weight
            attn_value = attn_kernls.matmul(self.unfold3(x))
            attn_value = attn_value.unflatten(2, (int(sqrt(attn_value.size(2))), -1))
            return attn_value

    # class AttnConv2d(nn.Module):
    #     def __init__(self, image_size, in_channels, out_channels, hidden_kernels=3, conv_kernels=3, attn_kernels=3):
    #         super(AttnConv2d, self).__init__()
    #         self.kernel1 = AttnConvKernel(image_size, in_channels, out_channels, hidden_kernels, conv_kernels, softmax=False)
    #         self.kernel2 = AttnConvKernel(image_size, in_channels, out_channels, hidden_kernels, attn_kernels, softmax=True)
    #         self.unfold = nn.Unfold(conv_kernels, padding=conv_kernels // 2)
    #         self.conv = nn.Conv2d(in_channels, in_channels, conv_kernels, padding=conv_kernels // 2, bias=False)
    #         self.attn_kernels = attn_kernels
    #
    #     def forward(self, x):
    #         x = x + self.pos_code
    #         kernel1 = self.kernel1(x)
    #         kernel2 = self.kernel2(x)
    #         attn_kernls = kernel2 * kernel1 # kernel2 * (kernel1 + self.attn_weight / self.attn_kernels) #
    #         # print(x.shape, self.conv(x).shape, self.unfold.kernel_size, self.unfold.stride, self.unfold(self.conv(x)).shape, attn_kernls.shape)
    #         attn_value = attn_kernls.flatten(2).matmul(self.unfold(self.conv(x)))
    #         # attn_value = self.unfold3(x).transpose(1, 2).matmul(attn_kernls).transpose(1, 2)
    #         attn_value = attn_value.unflatten(2, (int(sqrt(attn_value.size(2))), -1))
    #         return attn_value

    class ResAttnConv2d(nn.Module):
        def __init__(self, image_size, in_channels, out_channels, norm_scale, hidden_kernels=3, attn_kernels=3,
                     conv_momentum=None,
                     num_heads=1, batch_norm=True, xor=True):
            super(ResAttnConv2d, self).__init__()
            if in_channels != out_channels:
                self.conv_momentum = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            else:
                self.conv_momentum = nn.Parameter(torch.tensor(1.))
            if '__getitem__' in dir(num_heads):
                self.num_heads = num_heads
            else:
                self.num_heads = (num_heads, num_heads)

            if batch_norm:
                self.conv1 = nn.Sequential(
                    ResBatchNorm2d(in_channels // 2 if in_channels >= 10 else in_channels, norm_scale, short_cut=0.01),
                    AttnConv2d(image_size, in_channels // 2 if in_channels >= 10 else in_channels, out_channels // 2,
                               hidden_kernels, attn_kernels, softmax=True),
                    ResBatchNorm2d(out_channels // 2, norm_scale, short_cut=0.6)
                )
                self.conv2 = nn.Sequential(
                    ResBatchNorm2d(in_channels // 2 if in_channels >= 10 else in_channels, norm_scale, short_cut=0.01),
                    AttnConv2d(image_size, in_channels // 2 if in_channels >= 10 else in_channels, out_channels // 2,
                               hidden_kernels, attn_kernels, softmax=True),
                    ResBatchNorm2d(out_channels // 2, norm_scale, short_cut=0.6)
                )
                self.conv3 = nn.Sequential(
                    ResBatchNorm2d(in_channels // 2 if in_channels >= 10 else in_channels, norm_scale, short_cut=0.01),
                    AttnConv2d(image_size, in_channels // 2 if in_channels >= 10 else in_channels, out_channels // 2,
                               hidden_kernels, attn_kernels, softmax=True),
                    ResBatchNorm2d(out_channels // 2, norm_scale, short_cut=0.6)
                )
                self.conv4 = nn.Sequential(
                    ResBatchNorm2d(in_channels // 2 if in_channels >= 10 else in_channels, norm_scale, short_cut=0.01),
                    AttnConv2d(image_size, in_channels // 2 if in_channels >= 10 else in_channels, out_channels // 2,
                               hidden_kernels, attn_kernels, softmax=True),
                    ResBatchNorm2d(out_channels // 2, norm_scale, short_cut=0.6)
                )
            # else:
            #     self.conv1 = nn.Sequential(
            #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.01),
            #         AttnConv2d(image_size, in_channels // 2 if in_channels >= 10 else in_channels, out_channels, 3, kernels, attn_kernels),
            #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.6),
            #     )
            #     self.conv2 = nn.Sequential(
            #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.01),
            #         AttnConv2d(image_size, in_channels // 2 if in_channels >= 10 else in_channels, out_channels, 3, kernels, attn_kernels),
            #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.6),
            #     )
            #     self.conv3 = nn.Sequential(
            #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.01),
            #         AttnConv2d(image_size, in_channels // 2 if in_channels >= 10 else in_channels, out_channels, 3, kernels, attn_kernels),
            #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.6),
            #     )
            #     self.conv4 = nn.Sequential(
            #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.01),
            #         AttnConv2d(image_size, in_channels // 2 if in_channels >= 10 else in_channels, out_channels, 3, kernels, attn_kernels),
            #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.6),
            #     )
            # self.pos_code = nn.Parameter(torch.randn([in_channels] + image_size))
            self.batch_norm = batch_norm
            self.norm_scale = norm_scale
            self.xor = xor
            if self.num_heads[0] * self.num_heads[1] == 1:
                self.multi_head_downsampling = None
            else:
                self.multi_head_downsampling = nn.Unfold(self.num_heads,
                                                         stride=self.num_heads)  # nn.AvgPool2d(self.num_heads)
            self.dropout = nn.Dropout()
            self.pre_relu = ResReLU(reverse=True)
            self.in_channels = in_channels

        def feature_map_stack(self, y):
            if self.multi_head_downsampling is not None:
                h, w = y.shape[-2:]
                y = self.multi_head_downsampling(y)
                y = y.unflatten(1, (-1, self.num_heads[0] * self.num_heads[1], self.num_heads[0] * self.num_heads[1]))
                y = y.transpose(2, 3).contiguous()
                y = y.view((y.size(0), -1) + self.num_heads + (int(sqrt(y.size(-1))), int(sqrt(y.size(-1)))))
                y = y.transpose(3, 4).contiguous()
                y = y.view(y.shape[:2] + (y.shape[2] * y.shape[3], -1))
                if y.size(2) * y.size(3) < h * w:
                    h1, w1 = (h - y.size(2)) // 2, (w - y.size(3)) // 2
                    h2, w2 = h - y.size(2) - h1, w - y.size(3) - w1
                    y = F.pad(y, (w1, w2, h1, h2))
            return y

        def channel_split(self, x):
            if x.size(1) >= 10:
                x1, x2 = x.chunk(2, 1)
                # x1, x2 = x.roll(self.in_channels // 4, 1).chunk(2, 1)
            else:
                # x1 = x.clone()
                # x2 = x.clone()
                # x3 = x.clone()
                # x4 = x.clone()
                # x1 = x2 = x3 = x4 =x
                x1 = x2 = x
            return x1, x2  # , x3, x4

        def forward(self, x):
            # x_norm = self.norm_scale * self.pre_norm(x)
            x1, x2 = self.channel_split(x)  # + self.pos_code
            y1 = self.conv1(x1)
            y2 = self.conv2(x1)
            y3 = self.conv3(x2)
            y4 = self.conv4(x2)
            x = 0.99 * self.conv_momentum * x if type(
                self.conv_momentum) is nn.parameter.Parameter else self.conv_momentum(x)
            y = torch.cat([(y1 + y3) / sqrt(2), y2 * y4], 1)  # / sqrt(3) (y1 + y3 + y2 * y4) / sqrt(3)
            return x + self.feature_map_stack(y)

    class ResAttnConv2d(nn.Module):
        def __init__(self, image_size, in_channels, out_channels, norm_scale, hidden_kernels=3, attn_kernels=3,
                     conv_momentum=None,
                     num_heads=1, batch_norm=True, xor=True):
            super(ResAttnConv2d, self).__init__()
            if in_channels != out_channels:
                self.conv_momentum = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            else:
                self.conv_momentum = nn.Parameter(torch.tensor(1.))
            if '__getitem__' in dir(num_heads):
                self.num_heads = num_heads
            else:
                self.num_heads = (num_heads, num_heads)

            if batch_norm:
                self.conv1 = nn.Sequential(
                    ResBatchNorm2d(in_channels // 2 if in_channels >= 10 else in_channels, norm_scale, short_cut=0.01),
                    AttnConv2d(image_size, in_channels // 2 if in_channels >= 10 else in_channels, out_channels // 2,
                               hidden_kernels, attn_kernels, softmax=True),
                    ResBatchNorm2d(out_channels // 2, norm_scale, short_cut=0.6)
                )
                self.conv2 = nn.Sequential(
                    ResBatchNorm2d(in_channels // 2 if in_channels >= 10 else in_channels, norm_scale, short_cut=0.01),
                    AttnConv2d(image_size, in_channels // 2 if in_channels >= 10 else in_channels, out_channels // 2,
                               hidden_kernels, attn_kernels, softmax=True),
                    ResBatchNorm2d(out_channels // 2, norm_scale, short_cut=0.6)
                )
                self.conv3 = nn.Sequential(
                    ResBatchNorm2d(in_channels // 2 if in_channels >= 10 else in_channels, norm_scale, short_cut=0.01),
                    AttnConv2d(image_size, in_channels // 2 if in_channels >= 10 else in_channels, out_channels // 2,
                               hidden_kernels, attn_kernels, softmax=True),
                    ResBatchNorm2d(out_channels // 2, norm_scale, short_cut=0.6)
                )
                self.conv4 = nn.Sequential(
                    ResBatchNorm2d(in_channels // 2 if in_channels >= 10 else in_channels, norm_scale, short_cut=0.01),
                    AttnConv2d(image_size, in_channels // 2 if in_channels >= 10 else in_channels, out_channels // 2,
                               hidden_kernels, attn_kernels, softmax=True),
                    ResBatchNorm2d(out_channels // 2, norm_scale, short_cut=0.6)
                )
            # else:
            #     self.conv1 = nn.Sequential(
            #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.01),
            #         AttnConv2d(image_size, in_channels // 2 if in_channels >= 10 else in_channels, out_channels, 3, kernels, attn_kernels),
            #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.6),
            #     )
            #     self.conv2 = nn.Sequential(
            #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.01),
            #         AttnConv2d(image_size, in_channels // 2 if in_channels >= 10 else in_channels, out_channels, 3, kernels, attn_kernels),
            #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.6),
            #     )
            #     self.conv3 = nn.Sequential(
            #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.01),
            #         AttnConv2d(image_size, in_channels // 2 if in_channels >= 10 else in_channels, out_channels, 3, kernels, attn_kernels),
            #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.6),
            #     )
            #     self.conv4 = nn.Sequential(
            #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.01),
            #         AttnConv2d(image_size, in_channels // 2 if in_channels >= 10 else in_channels, out_channels, 3, kernels, attn_kernels),
            #         ResLayerNorm2d(image_size, norm_scale, short_cut=0.6),
            #     )
            # self.pos_code = nn.Parameter(torch.randn([in_channels] + image_size))
            self.batch_norm = batch_norm
            self.norm_scale = norm_scale
            self.xor = xor
            if self.num_heads[0] * self.num_heads[1] == 1:
                self.multi_head_downsampling = None
            else:
                self.multi_head_downsampling = nn.Unfold(self.num_heads,
                                                         stride=self.num_heads)  # nn.AvgPool2d(self.num_heads)
            self.dropout = nn.Dropout()
            self.pre_relu = ResReLU(reverse=True)
            self.in_channels = in_channels

        def feature_map_stack(self, y):
            if self.multi_head_downsampling is not None:
                h, w = y.shape[-2:]
                y = self.multi_head_downsampling(y)
                y = y.unflatten(1, (-1, self.num_heads[0] * self.num_heads[1], self.num_heads[0] * self.num_heads[1]))
                y = y.transpose(2, 3).contiguous()
                y = y.view((y.size(0), -1) + self.num_heads + (int(sqrt(y.size(-1))), int(sqrt(y.size(-1)))))
                y = y.transpose(3, 4).contiguous()
                y = y.view(y.shape[:2] + (y.shape[2] * y.shape[3], -1))
                if y.size(2) * y.size(3) < h * w:
                    h1, w1 = (h - y.size(2)) // 2, (w - y.size(3)) // 2
                    h2, w2 = h - y.size(2) - h1, w - y.size(3) - w1
                    y = F.pad(y, (w1, w2, h1, h2))
            return y

        def channel_split(self, x):
            if x.size(1) >= 10:
                x1, x2 = x.chunk(2, 1)
                # x1, x2 = x.roll(self.in_channels // 4, 1).chunk(2, 1)
            else:
                # x1 = x.clone()
                # x2 = x.clone()
                # x3 = x.clone()
                # x4 = x.clone()
                # x1 = x2 = x3 = x4 =x
                x1 = x2 = x
            return x1, x2  # , x3, x4

        def forward(self, x):
            # x_norm = self.norm_scale * self.pre_norm(x)
            x1, x2 = self.channel_split(x)  # + self.pos_code
            y1 = self.conv1(x1)
            y2 = self.conv2(x1)
            y3 = self.conv3(x2)
            y4 = self.conv4(x2)
            x = 0.99 * self.conv_momentum * x if type(
                self.conv_momentum) is nn.parameter.Parameter else self.conv_momentum(x)
            y = torch.cat([(y1 + y3) / sqrt(2), y2 * y4], 1)  # / sqrt(3) (y1 + y3 + y2 * y4) / sqrt(3)
            return x + self.feature_map_stack(y)


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