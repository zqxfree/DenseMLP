import torch
import time
import numpy as np
from qnn import *
import torchvision
import torchvision.transforms as transforms
from lightgbm.sklearn import LGBMClassifier
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Hyper-parameters
batch_size = 100
learning_rate = 0.001


model = nn.Sequential(
    DeepConv2d([3] + [128] * 3 + [256] * 3 + [512] * 7, [3] * 16, [2, 5, 8, -1]),
    # nn.Conv2d(512, 10, 2),
    # nn.AvgPool2d(2),
    # ResLayerhNorm([512, 2, 2]),
    # ResBatchNorm2d(512),
    nn.Flatten(1),
    # ResBatchNorm1d(512 * 2 * 2),
    nn.Linear(512 * 2 * 2, 10),
).cuda()

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
                                      # 将[0,1]归一化到[-1,1]
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
     transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
     # 归一化至[0-1]是直接除以255，若自己的ndarray数据尺度有变化，则需要自行修改。
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
                                             transform=transform_train2, # transforms.ToTensor(), # , # cifar10_transform, #  #
                                             download=False)

test_dataset = torchvision.datasets.CIFAR10(root=r'D:/PycharmProjects/dataset',
                                            train=False,
                                            transform=transform_test2 # transforms.ToTensor() #
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
num_epochs = 1000
save_model_file = None


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
    over_thres = 10
    counts = [0, 0, 0]
    max_test_loss_acc = [inf, 0]
    epoch = [0]
    stop_epoch = [False]


    def print_last_result(train_losses, train_accuracies, test_losses, test_accuracies):
        train_losses = np.array(train_losses)
        train_accuracies = np.array(train_accuracies)
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

    def train_test(j):
        if stop_epoch[0]:
            return ln1, ln2
        if counts[0] == 0:
            with torch.no_grad():
                print("First Test Stage:")
                epoch_duration, total_loss, total_accuracies, confusion, tpr, outputs = epoch_train_test(test_loader)
                test_losses.append(total_loss)
                if total_accuracies > 0:
                    test_accuracies.append(total_accuracies / test_total_labels)
                print()
            ln1.set_data(epoch, train_accuracies)
            ln2.set_data(epoch, test_accuracies)
            counts[0] += 1
            return ln1, ln2
        if counts[0] > num_epochs:
            print()
            print_last_result(train_losses, train_accuracies, test_losses, test_accuracies)
            ln1.set_data(epoch[:101], train_accuracies[:101])
            ln2.set_data(epoch[:101], test_accuracies[:101])
            stop_epoch[0] = True
            return ln1, ln2
        print("Epoch {}/{}:".format(counts[0], num_epochs))
        # train
        model.train()
        print("Train Stage:")
        epoch_duration, total_loss, total_accuracies, confusion, tpr, outputs = epoch_train_test(train_loader, train=True)
        train_losses.append(total_loss)
        epoch.append(counts[0])
        if total_accuracies > 0:
            train_accuracies.append(total_accuracies / train_total_labels)
        if counts[0] > thres_epochs and len(train_accuracies) > 0 and thres_accuracy <= train_accuracies[-1]:
            counts[1] += 1
            if counts[1] == over_thres or counts[2] == over_thres:
                print()
                print_last_result(train_losses, train_accuracies, test_losses, test_accuracies)
                ln1.set_data(epoch[:101], train_accuracies[:101])
                ln2.set_data(epoch[:101], test_accuracies[:101])
                stop_epoch[0] = True
                return ln1, ln2
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
                if counts[1] == over_thres or counts[2] == over_thres:
                    print()
                    print_last_result(train_losses, train_accuracies, test_losses, test_accuracies)
                    ln1.set_data(epoch[:101], train_accuracies[:101])
                    ln2.set_data(epoch[:101], test_accuracies[:101])
                    stop_epoch[0] = True
                    return ln1, ln2
        ln1.set_data(epoch[:101], train_accuracies[:101])
        ln2.set_data(epoch[:101], test_accuracies[:101])
        print()
        counts[0] += 1
        return ln1, ln2


    # 绘制训练-测试准确率动态图
    plt.rcParams["font.sans-serif"] = 'KaiTi'
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots()
    ax.set_title('cifar10数据集分类', fontsize=10)
    ax.set_xlabel('Epoch', fontsize=8)
    ax.set_ylabel('Acc', fontsize=8)
    ax.set_xlim(-5, 105)
    ax.set_ylim(0., 1.)
    ln1, = plt.plot([], [], '.-', color='orange', linewidth=1, ms=2, mec='orange', mfc='w')
    ln2, = plt.plot([], [], 'm.-', linewidth=1, ms=2, mec='m', mfc='w')
    ax.legend(['训练准确率', '测试准确率'], prop={'size': 8.5})


    # 创建动画
    ani = FuncAnimation(fig, train_test, frames=np.arange(110), blit=True)
    # 将动图保存为gif
    # ani.save("pratice1.1.gif", writer='yy')
    # 图像展示
    plt.show()

