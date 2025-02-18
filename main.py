import torch
import time
from dense_mlp import *
import numpy as np
from qnn import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier, \
    HistGradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier, VotingClassifier, \
    BaggingClassifier
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.linear_model import LogisticRegression
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import median_abs_deviation, skew, kurtosis, mode
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms
# from lightgbm.sklearn import LGBMClassifier
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Hyper-parameters
batch_size = 100
# learning_rate = 0.1773

# class ResMLP(nn.Module):
#     def __init__(self):
#         super(ResMLP, self).__init__()
#         self.lin = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#         )
#         self.alpha = nn.Parameter(torch.tensor(0.5))
#
#     def forward(self, x):
#         return self.alpha * x + (1 - self.alpha) * self.lin(x)


# model = nn.Sequential(
    # nn.Conv2d(1, 16, 3, padding=1),
    # nn.BatchNorm2d(16),
    # nn.ReLU(),
    # nn.Conv2d(16, 64, 3, padding=1),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.MaxPool2d(2),
    # RawDense2d(),
    # RawDense2d(),
    # nn.MaxPool2d(2),
    # RawDense2d(),
    # RawDense2d(),
    # RawDense2d(),
    # RawDense2d(),
    # RawDense2d(),
    # RawDense2d(),
    # nn.Flatten(1),
    # nn.Linear(64*7*7, 10),

    # RatioDense2d(3, 128, 7),
    # nn.MaxPool2d(2),
    # RatioDense2d(127, 256, 8),
    # nn.MaxPool2d(2),
    # RatioDense2d(255, 512, 9),
    # nn.MaxPool2d(2),
    # RatioDense2d(511, 511, 9),
    # nn.MaxPool2d(2),
    # nn.Flatten(1),
    # nn.Linear(511 * 2 * 2, 10),

    # MetaDenseNet(3, 128, 7),
    # nn.MaxPool2d(2),
    # MetaDenseNet(127, 256, 8),
    # nn.MaxPool2d(2),
    # MetaDenseNet(255, 512, 9),
    # nn.MaxPool2d(2),
    # MetaDenseNet(511, 511, 9),
    # nn.MaxPool2d(2),
    # nn.Flatten(1),
    # nn.Linear(511 * 2 * 2, 10),

    # DenseRatio2d(3, 128, 7),
    # nn.MaxPool2d(2),
    # DenseRatio2d(127, 256, 8),
    # nn.MaxPool2d(2),
    # DenseRatio2d(255, 512, 9),
    # nn.MaxPool2d(2),
    # DenseRatio2d(511, 511, 9),
    # nn.MaxPool2d(2),
    # nn.Flatten(1),
    # nn.Linear(511 * 2 * 2, 10),

    # RawRes2d(3, 128, 128, 3),
    # nn.MaxPool2d(2),
    # RawRes2d(128, 256, 256, 3),
    # nn.MaxPool2d(2),
    # RawRes2d(256, 512, 512, 3),
    # nn.MaxPool2d(2),
    # RawRes2d(512, 512, 512, 9),
    # nn.MaxPool2d(2),
    # nn.Flatten(1),
    # nn.Linear(512 * 2 * 2, 10),

    # DenseRes2d(8, 3, 128),
    # nn.MaxPool2d(2),
    # nn.Dropout(),
    # DenseRes2d(8, 128, 256),
    # nn.MaxPool2d(2),
    # nn.Dropout(),
    # DenseRes2d(8, 256, 512),
    # nn.MaxPool2d(2),
    # nn.Dropout(),
    # DenseRes2d(8, 512, 512),
    # DenseRes2d(8, 512, 512),
    # nn.MaxPool2d(2),
    # nn.Dropout(),
    # nn.Flatten(1),
    # nn.Linear(512 * 2 * 2, 10),

    # DenseRes2d(4, 3, 128),
    # nn.MaxPool2d(2),
    # DenseRes2d(4, 128, 256),
    # nn.MaxPool2d(2),
    # DenseRes2d(4, 256, 512),
    # nn.MaxPool2d(2),
    # DenseRes2d(8, 512, 512),
    # nn.MaxPool2d(2),
    # nn.Flatten(1),
    # nn.Linear(512 * 2 * 2, 10),


    # MetaResNet([[128, 2], [256, 2], [512, 2], [512, 4]], 3, dropout=False),
    # # ResBatchNorm2d(512),
    # # nn.Conv2d(512, 10, 2),
    # nn.Flatten(1),
    # nn.Linear(512 * 2 * 2, 10),

    # MetaResNet([[64, 2], [128, 2], [128, 4]], 1, 5, norm_scale=0.1773, image_sizes=[28, 28], dropout=False),
    # # ResBatchNorm2d(512),
    # # nn.Conv2d(512, 10, 2),
    # nn.Flatten(1),
    # nn.Linear(128 * 3 * 3, 10),

    # MetaResConv2d(1, 64, 0.182, 5, 0.),
    # nn.Unfold(4, stride=4),
    # SwapAxes(1, 2),
    # MetaResConv1d(49, 64, 0.182, 5, 0.),
    # nn.MaxPool1d(2),
    # MetaResConv1d(64, 128, 0.182, 5, 0.),
    # nn.MaxPool1d(2),
    # MetaResConv1d(128, 128, 0.182, 5, 0.),
    # nn.MaxPool1d(2),
    # MetaResConv1d(128, 128, 0.182, 5, 0.),
    # nn.MaxPool1d(2),
    # MetaResConv1d(128, 128, 0.182, 5, 0.),
    # nn.MaxPool1d(2),
    # MetaResConv1d(128, 128, 0.182, 5, 0.),
    # nn.MaxPool1d(2),
    # MetaResConv1d(128, 128, 0.182, 5, 0.),
    # nn.MaxPool1d(2),
    # nn.Flatten(1),
    # nn.Linear(128 * 8, 10),

    # RawResNet(1, 64, 5),
    # RawResNet(64, 64, 5),
    # RawResNet(64, 64, 5),
    # nn.MaxPool2d(2),
    # RawResNet(64, 128, 5),
    # RawResNet(128, 128, 5),
    # RawResNet(128, 128, 5),
    # nn.MaxPool2d(2),
    # RawResNet(128, 128, 5),
    # RawResNet(128, 128, 5),
    # RawResNet(128, 128, 5),
    # RawResNet(128, 128, 5),
    # nn.MaxPool2d(2),
    # nn.Flatten(1),
    # nn.Linear(128 * 3 * 3, 10),

    # (array([4.2826562, 4.119719 , 4.2742929, 3.9594956, 4.0765022, 3.7927299]),
    #  array([4.2826562, 4.119719 , 4.2742929, 3.9594956, 4.0765022, 3.7927299]),
    #  array([4.2826368, 4.1196864, 4.2824851, 3.9595714, 4.1271166, 3.7927807]),
    #  array([4.2826368, 4.1196864, 4.2824851, 3.9595714, 4.1271166, 3.7927807]))
    # nn.Unfold(7, stride=3),
    # SwapAxes(1, 2),
    # nn.Unflatten(1, (64, 1)), # (100, 64, 1, 49)
    # TrueNeuron(64, 1, 32, 49, 32, ), # (100, 64, 32, 32)
    # SwapAxes(1, 2),
    # TrueNeuron(32, 64, 32, 32, 32), # (100, 32, 32, 32),
    # SwapAxes(1, 2),
    # TrueNeuron(32, 32, 64, 32, 32),  # (100, 32, 64, 32),
    # SwapAxes(1, 2),
    # TrueNeuron(64, 32, 16, 32, 16),  # (100, 64, 16, 16),
    # SwapAxes(1, 2),
    # TrueNeuron(16, 64, 128, 16, 16),  # (100, 16, 128, 16),
    # SwapAxes(1, 2),
    # TrueNeuron(128, 16, 8, 16, 8),  # (100, 128, 8, 8),
    # NeuronShortcut(128, 8, 1, 8, 1),
    # nn.Flatten(1),
    # nn.Linear(128, 10),

# (array([3.5561, 3.2482, 3.2483, 3.4418, 2.8269, 3.4417]),
#  array([-3.5559, -3.2482, -3.2482, -3.4418, -2.8269, -3.4417]))
#     nn.Unfold(7, stride=3),
#     SwapAxes(1, 2),
#     nn.Unflatten(1, (1, 64)),  # (100, 1, 64, 49)
#     TrueNeuron(64, 1, 32, 49, 32, 3.5561),  # (100, 64, 32, 32)
#     TrueNeuron(32, 64, 32, 32, 32, 3.2482),  # (100, 32, 32, 32),
#     TrueNeuron(32, 32, 64, 32, 32, 3.2483),  # (100, 32, 64, 32),
#     TrueNeuron(64, 32, 16, 32, 16, 3.4418),  # (100, 64, 16, 16),
#     TrueNeuron(16, 64, 128, 16, 16, 2.8269),  # (100, 16, 128, 16),
#     TrueNeuron(128, 16, 8, 16, 8, 3.4417, last_layer=True),  # (100, 128, 8, 8),
#     NeuronShortcut(128, 8, 1, 8, 1),
#     nn.Flatten(1),
#     nn.Linear(128, 10),


    # MetaResNet([[64, 2], [256, 4]], 1, dropout=True),
    # # ResBatchNorm2d(512),
    # # nn.Conv2d(512, 10, 2),
    # nn.Flatten(1),
    # nn.Linear(256 * 7 * 7, 10),

    # DenseRes2d(4, 1, 64),
    # nn.MaxPool2d(2),
    # DenseRes2d(4, 64, 64),
    # nn.MaxPool2d(2),
    # DenseRes2d(8, 64, 128),
    # nn.MaxPool2d(2),
    # nn.Flatten(1),
    # nn.Linear(128 * 3 * 3, 10),

    # OrthoRes2d(4, 28, 28, 1, 64),
    # nn.MaxPool2d(2),
    # OrthoRes2d(4, 14, 14, 64, 64),
    # nn.MaxPool2d(2),
    # OrthoRes2d(8, 7, 7, 64, 128),
    # nn.MaxPool2d(2),
    # nn.Flatten(1),
    # nn.Linear(128 * 3 * 3, 10),

    # RawDense2d(1, 16, 64, 4, 5),
    # nn.MaxPool2d(2),
    # nn.Dropout(),
    # RawDense2d(64, 64, 64, 4, 5),
    # nn.MaxPool2d(2),
    # nn.Dropout(),
    # RawDense2d(64, 64, 64, 16, 5),
    # nn.MaxPool2d(2),
    # nn.Dropout(),
    # nn.Flatten(1),
    # nn.Linear(64 * 3 * 3, 10),

    # Res2d(3, 64),
    # nn.MaxPool2d(2),
    # Res2d(64, 128),
    # Res2d(128, 128),
    # nn.MaxPool2d(2),
    # Res2d(128, 256),
    # Res2d(256, 256),
    # nn.MaxPool2d(2),
    # Res2d(256, 512),
    # Res2d(512, 512),
    # Res2d(512, 512),
    # Res2d(512, 512),
    # nn.MaxPool2d(2),
    # nn.Flatten(1),
    # nn.Linear(512 * 2 * 2, 10),

    # nn.Conv2d(1, 16, 5, padding=2),
    # nn.BatchNorm2d(16),
    # nn.ReLU(),
    # nn.Conv2d(16, 64, 5, padding=2),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.Conv2d(64, 64, 5, padding=2),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.Conv2d(64, 64, 5, padding=2),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.MaxPool2d(2),
    # nn.Conv2d(64, 64, 5, padding=2),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.Conv2d(64, 64, 5, padding=2),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.Conv2d(64, 64, 5, padding=2),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.MaxPool2d(2),
    # nn.Conv2d(64, 64, 5, padding=2),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.Conv2d(64, 64, 5, padding=2),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.Conv2d(64, 64, 5, padding=2),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.Flatten(1),
    # nn.Linear(64 * 7 * 7, 10),

    # TernaryConv2d(3, 64),
    # TernaryConv2d(64, 64),
    # TernaryConv2d(64, 64),
    # TernaryConv2d(64, 64),
    # nn.MaxPool2d(2),
    # TernaryConv2d(64, 64),
    # TernaryConv2d(64, 64),
    # TernaryConv2d(64, 64),
    # TernaryConv2d(64, 64),
    # TernaryConv2d(64, 64),
    # TernaryConv2d(64, 64),
    # nn.MaxPool2d(2),
    # TernaryConv2d(64, 64),
    # TernaryConv2d(64, 64),
    # TernaryConv2d(64, 64),
    # TernaryConv2d(64, 64),
    # TernaryConv2d(64, 64),
    # TernaryConv2d(64, 64),
    # nn.MaxPool2d(2),
    # nn.Flatten(1),
    # nn.Linear(64 * 4 * 4, 10),

    # nn.Conv2d(3, 64, 3, padding=1),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.Conv2d(64, 64, 3, padding=1),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.Conv2d(64, 64, 3, padding=1),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.MaxPool2d(2),
    # nn.Conv2d(64, 64, 3, padding=1),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.Conv2d(64, 64, 3, padding=1),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.MaxPool2d(2),
    # nn.Conv2d(64, 64, 3, padding=1),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.Conv2d(64, 64, 3, padding=1),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.MaxPool2d(2),
    # nn.Conv2d(64, 64, 3, padding=1),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.Conv2d(64, 64, 3, padding=1),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.MaxPool2d(2),
    # nn.Conv2d(64, 64, 3, padding=1),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.Conv2d(64, 64, 3, padding=1),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.Flatten(1),
    # nn.Linear(64 * 2 * 2, 10),

    # nn.Unfold(7, stride=3),
    # SwapAxes(-2, -1),
    # XorMLPEncoder(49, 49, 3),
    # SwapAxes(-2, -1),
    # XorMLPEncoder(64, 64, 3),
    # SwapAxes(-2, -1),
    # XorMLPEncoder(49, 49, 3),
    # SwapAxes(-2, -1),
    # XorMLPEncoder(64, 64, 3),
    # nn.Flatten(1),
    # nn.Linear(64 * 49, 10),

    # TernaryMLP(28, 28, 28),
    # TernaryMLP(28, 28, 28),
    # SwapAxes(1, 2),
    # TernaryMLP(28, 28, 28),
    # TernaryMLP(28, 28, 28),
    # SwapAxes(1, 2),
    # TernaryMLP(28, 28, 28),
    # TernaryMLP(28, 28, 28),
    # SwapAxes(1, 2),
    # TernaryMLP(28, 28, 28),
    # TernaryMLP(28, 28, 28),
    # SwapAxes(1, 2),
    # TernaryMLP(28, 28, 28),
    # TernaryMLP(28, 28, 28),
    # SwapAxes(1, 2),
    # TernaryMLP(28, 28, 28),
    # TernaryMLP(28, 28, 28),
    # nn.Flatten(1),
    # nn.Linear(28*28, 10),

    # nn.Unfold(7, stride=3),
    # BiTernaryMLP((49, 64), (49, 64), (49, 64)),
    # BiTernaryMLP((49, 64), (49, 64), (49, 64)),
    # BiTernaryMLP((49, 64), (49, 64), (49, 64)),
    # BiTernaryMLP((49, 64), (49, 64), (49, 64)),
    # BiTernaryMLP((49, 64), (49, 64), (49, 64)),
    # BiTernaryMLP((49, 64), (49, 64), (49, 64)),
    # nn.Flatten(1),
    # nn.Linear(64 * 49, 10),
# ).cuda()
model = nn.Sequential(
    # MetaResNet([[64, 2], [128, 2], [128, 4]], 1, 5, norm_scale=0.1773, image_sizes=[28, 28], dropout=False),
    MetaResNet([[32, 2], [128, 2], [128, 4]], 3, 3, norm_scale=0.1773, image_sizes=[28, 28], dropout=False),
    # ResBatchNorm2d(512),
    # nn.Conv2d(512, 10, 2),
    nn.Flatten(1),
    nn.Linear(128 * 4 * 4, 10),
).cuda()
# model = AutoEncoder(128, 128, 128).cuda()
optimizer = torch.optim.Adam(model.parameters())
# optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.5)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True,
#                                                        threshold=0.00005, threshold_mode='rel', cooldown=0, min_lr=0,
#                                                        eps=1e-08)


cifar10_transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])


transform_train1 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

norm_mean = [0.485, 0.456, 0.406]  # 均值
norm_std = [0.229, 0.224, 0.225]  # 方差
transform_train2 = transforms.Compose([transforms.ToTensor(),  # 将PILImage转换为张量
                                      # 将[0.1773]归一化到[-1,1]
                                      transforms.Normalize(norm_mean, norm_std),
                                      transforms.RandomHorizontalFlip(),  # 随机水平镜像
                                      transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
                                      transforms.RandomCrop(32, padding=4)  # 随机中心裁剪
                                      ])

transform_test2 = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(norm_mean, norm_std)])

# transform_train = transforms.Compose([transforms.RandomResizedCrop(32),
#                                      transforms.RandomHorizontalFlip(),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
#
# transform_test =  transforms.Compose([transforms.Resize((32, 32)),  # cannot 224, must (224, 224)
#                                    transforms.ToTensor(),
#                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trans_train = transforms.Compose(
        [transforms.RandomResizedCrop(224),  # 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小；
         # （即先随机采集，然后对裁剪得到的图像缩放为同一大小） 默认scale=(0.08, 1.0)
         transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5；
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])]
)

trans_valid = transforms.Compose(
    [transforms.Resize(256),  # 是按照比例把图像最小的一个边长放缩到256，另一边按照相同比例放缩。
     transforms.CenterCrop(224),  # 依据给定的size从中心裁剪
     transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0.1773]
     # 归一化至[0.1773]是直接除以255，若自己的ndarray数据尺度有变化，则需要自行修改。
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])]
)


# MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root=r'D:/PycharmProjects/dataset',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=False)
#
# test_dataset = torchvision.datasets.MNIST(root='D:/PycharmProjects/dataset',
#                                           train=False,
#                                           transform=transforms.ToTensor(),
#                                           download=False)
train_dataset = torchvision.datasets.CIFAR10(root=r'D:/PycharmProjects/dataset',
                                             train=True,
                                             transform=transform_train2, #transforms.ToTensor(), #  , # cifar10_transform, #  #
                                             download=False)

test_dataset = torchvision.datasets.CIFAR10(root=r'D:/PycharmProjects/dataset',
                                            train=False,
                                            transform=transform_test2 #transforms.ToTensor() #
                                            )
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

train_total_labels = 50000
test_total_labels = 10000
num_epochs = 500
save_model_file = None


def print_res_prob():
    print('Parameters:')
    k = 1
    c = 0
    for i, j in model.named_parameters():
        if 'normddd_momentum' in i and c == 0:
            print('norm_momentum{}: {}'.format(k, round(j.item(), 4)), end=' ')
        # elif 'conv_momentum' in i and c == 0:
        #     if j.data.numel() == 1:
        #         # if j.item() > 5.64:
        #         #     j.data = torch.tensor(10.).cuda()
        #         print('conv_momentum{}: {}'.format(k, round(j.item(), 4)), end=' ')
        #     else:
        #         print('conv_momentum{}: {}'.format(k, round(j.data.abs().max().item(), 4)), end=' ')
        #     c += 1
        # elif 'up_norm_momentum' in i and c == 1:
        #     print('up_norm_momentum{}: {}'.format(k, round(j.item(), 4)), end=' ')
        # elif 'down_norm_momentum' in i and c == 1:
        #     print('down_norm_momentum{}: {}'.format(k, round(j.item(), 4)), end=' ')
        # elif 'up_avg_momentum' in i and c == 1:
        #     print('up_avg_momentum{}: {}'.format(k, round(j.item(), 4)), end=' ')
        # elif 'down_avg_momentum' in i and c == 1:
        #     print('down_avg_momentum{}: {}'.format(k, round(j.item(), 4)), end=' ')
        # elif 'relu_neg_momentum' in i and c == 1:
        #     print('relu_neg_momentum{}: {}'.format(k, round(j.item(), 4)), end=' ')
        #     print()
            k += 1
            c = 0
        # elif 'relu_pos_momentum' in i and c == 2:
        #     print('relu_pos_momentum{}: {}'.format(k, round(j.item(), 4)), end=' ')
        #     print()
        #     k += 1
        #     c = 0
    #     if 'res_prob' in i:
    #         k += 1
    #         print('res_probe{}: {}'.format(k, round(j.item(), 4)), end=' ')
    #         if k % 6 == 0:
    #             print()
    # if k % 6 > 0:
    #     print()


def epoch_train_test(data_loader=None, criterion=nn.CrossEntropyLoss(), num_classes=10, print_period=100, train=False,
                     device=torch.device('cuda:0')):
    total_outputs, tpr = [], []
    total_loss1, total_loss2, cur_num_labels, total_accuracies, epoch_duration = 0, 0, 0, 0, 0
    confusion = torch.zeros(num_classes, num_classes).to(device).detach()
    t0 = time.time()
    total_labels = train_total_labels if train else test_total_labels
    for j, (inputs, true_labels) in enumerate(data_loader, 1):
        # 监督学习
        inputs = inputs.to(device)
        true_labels = true_labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, true_labels)
        # 参数更新
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # for group in optimizer.param_groups:
            #     for p in group['params']:
            #         if p.grad is not None:
            #             p.data = torch.max(p.data, torch.zeros_like(p.data))  # 如果参数小于0，则置为0

        # if true_labels is not None:
        #     true_labels = true_labels.data
        total_loss1 += loss.item()
        # total_loss2 += sparse_loss.item()
        cur_num_labels += outputs.size(0)
        # tpr.append(F.softmax(outputs.data, 1))
        if num_classes > 0:
            output_labels = outputs.data.argmax(1)
            acc_num = output_labels.eq(true_labels).sum().item()
            total_accuracies += acc_num
            total_outputs.append(output_labels)
            mix_labels = torch.stack(
                [true_labels, output_labels]).view(2, -1).detach()
            indices, counts = mix_labels.unique(
                return_counts=True, dim=1)
            confusion[indices.tolist()] += counts
        else:
            total_outputs.append(outputs)
        epoch_duration = time.time() - t0
        # print(j, inputs.shape, outputs.shape)
        if not train and cur_num_labels == total_labels or train and j % print_period == 0:
            print(
                "[{}/{}] Loss: {:.4f}, ".format(cur_num_labels, total_labels, total_loss1, total_loss2), end='')
            if num_classes > 0:
                print("Accuracy: {:.2f} %, ".format(int(total_accuracies / cur_num_labels * 10000) / 100), end='')
            print("Duration: {:.1f} s".format(epoch_duration))
    return epoch_duration, total_loss1, total_accuracies, confusion, None, torch.cat(total_outputs)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
 # make dataset
    x_train = torch.from_numpy(np.array([0])).type(torch.float32).cuda()
    x_test = torch.from_numpy(np.array([0])).type(torch.float32).cuda()
    y_train = torch.from_numpy(np.array([0])).type(torch.float32).cuda()
    y_test = torch.from_numpy(np.array([0])).type(torch.float32).cuda()
    train_losses, train_accuracies, test_losses, test_accuracies = [], [None], [], []
    thres_epochs = 0
    thres_accuracy = 1.
    over_thres = 15
    counts = [0, 0, 0]
    max_test_loss_acc = [inf, 0]


    def print_last_result(train_losses, train_accuracies, test_losses, test_accuracies):
        train_losses = np.array(train_losses)
        train_accuracies = np.array(train_accuracies[1:])
        test_losses = np.array(test_losses)
        test_accuracies = np.array(test_accuracies)
        max_train_acc = train_accuracies.max()
        max_test_acc = test_accuracies.max()
        print("Train Max Accuracy:")
        print("Epoch: ({})  Acc: {:.2f} %".format(
            ','.join(np.arange(1, len(train_accuracies) + 1).astype(str)[train_accuracies == max_train_acc]),
            max_train_acc * 100))
        print("Test Max Accuracy:")
        print("Epoch: ({})  Acc: {:.2f} %".format(
            ','.join(np.arange(len(test_accuracies)).astype(str)[test_accuracies == max_test_acc]),
            max_test_acc * 100))



    def train_test():
        if counts[0] == 0:
            with torch.no_grad():
                print("First Test Stage:")
                epoch_duration, total_loss, total_accuracies, confusion, tpr, outputs = epoch_train_test(test_loader)
                test_losses.append(total_loss)
                if total_accuracies > 0:
                    test_accuracies.append(total_accuracies / test_total_labels)
                print_res_prob()
                print()
            counts[0] += 1
            return False
        print("Epoch {}/{}:".format(counts[0], num_epochs))
        # train
        model.train()
        print("Train Stage:")
        epoch_duration, total_loss, total_accuracies, confusion, tpr, outputs = epoch_train_test(train_loader, train=True)
        train_losses.append(total_loss)
        if total_accuracies > 0:
            train_accuracies.append(total_accuracies / train_total_labels)
        if counts[0] > thres_epochs and len(train_accuracies) > 1 and thres_accuracy <= train_accuracies[-1]:
            counts[1] += 1
            if counts[1] == over_thres + 1:
                return True
        # test
        model.eval()
        with torch.no_grad():
            print("Test Stage:")
            epoch_duration, total_loss, total_accuracies, confusion, tpr, outputs = epoch_train_test(test_loader)
            test_losses.append(total_loss)
            if total_accuracies > 0:
                test_accuracies.append(total_accuracies / test_total_labels)
                # scheduler.step(total_accuracies / test_total_labels)
            if (total_accuracies > 0 and total_accuracies > max_test_loss_acc[1]) or (
                    total_accuracies == 0 and total_loss < max_test_loss_acc[1]):
                max_test_loss_acc[1] = total_accuracies
                max_test_loss_acc[0] = total_loss
                if save_model_file is not None:
                    torch.save(model.state_dict(), save_model_file)
                best_confusion = confusion
                best_tpr = tpr
                best_outputs = outputs
            if counts[0] > thres_epochs and len(test_accuracies) > 0 and thres_accuracy <= test_accuracies[-1]:
                counts[2] += 1
                if counts[2] == over_thres:
                    return True
        print_res_prob()
        print()
        counts[0] += 1
        return False

    for _ in range(num_epochs + 1):
        stop_iter = train_test()
        if stop_iter:
            break

    print()
    print_last_result(train_losses, train_accuracies, test_losses, test_accuracies)

    # 绘制训练-测试准确率动态图
    plt.rcParams["font.sans-serif"] = 'KaiTi'
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots()
    ax.set_title('cifar10数据集分类', fontsize=12)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Acc', fontsize=10)
    ax.set_xlim(-5, len(test_accuracies) + 5)
    ax.set_ylim(0., 1.)
    ln1, = plt.plot([], [], '-', color='orange', linewidth=1, ms=2, mec='orange', mfc='w')
    ln2, = plt.plot([], [], 'm-', linewidth=1, ms=2, mec='m', mfc='w')
    # 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    ax.legend(['训练准确率', '测试准确率'], loc='lower right', prop={'size': 12})

    count = [1]


    def update(i):
        if count[0] > len(test_accuracies):
            return ln1, ln2
        ln1.set_data(list(range(count[0])), train_accuracies[:count[0]])
        ln2.set_data(list(range(count[0])), test_accuracies[:count[0]])
        count[0] += 1
        return ln1, ln2


    # 创建动画
    ani = FuncAnimation(fig, update, frames=np.arange(len(test_accuracies) + 10), blit=True)
    # 将动图保存为gif
    # ani.save("pratice1.1.gif", writer='yy')
    # 图像展示
    plt.show()
