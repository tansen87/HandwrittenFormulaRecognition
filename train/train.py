import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter("./logs/tf-logs")


class Net(nn.Module):
    def __init__(self, num_classes=14):
        super(Net, self).__init__()
        self.BackBone = models.resnet18(pretrained=False)
        self.BackBone.fc = nn.Linear(self.BackBone.fc.in_features, 82)

    def forward(self, x):
        x = self.BackBone(x)
        return x


def train(model, device, train_loader, test_loader, epochs, lr=0.15, save_dir='./checkpoints'):
    # 如果采用默认初始化权重，很难train的动
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

    # 将初始化的参数加载到网络上
    model.apply(init_weights)
    # 将网络移动到gpu或者cpu上
    model.to(device)

    # 设置优化器，采用lr的学习率
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.001)
    loss = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # 网络进入训练模式，会计算梯度
        model.train()
        total_loss = 0
        for i, (data, label) in enumerate(train_loader):
            # 获取数据的data，和对应的label
            data, label = data.to(device), label.to(device)
            # 清空梯度，否则梯度会累加，造成计算错误
            optimizer.zero_grad()
            # 获取网络的输出
            output = model(data)
            # 计算loss
            l = loss(output, label)
            total_loss += l
            # 反向传播
            l.backward()
            # 更新网络参数
            optimizer.step()

            # 每计算十轮，打印一次
            if i % 10 == 0:
                # 打印此时的epoch轮数，loss值，和学习率
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t lr:'.format(
                    epoch, i * len(data), len(train_loader.dataset),
                    100. * i / len(train_loader), l),
                    optimizer.state_dict()['param_groups'][0]['lr'])

        # 将训练集，测试集上的loss
        # 测试集上的准确率记录下来
        test_loss, acc = test(model, device, test_loader)
        total_loss = total_loss / len(train_loader)
        writer.add_scalars(
            'check/Loss',
            {
                'Train': total_loss,
                'Test': test_loss},
            epoch)
        writer.add_scalar('check/accuracy', acc, epoch)

        # 保存当前epoch所训练好的模型
        save_dirs = os.path.join(save_dir, f"{str(epoch)}.pth")
        torch.save(model.state_dict(), save_dirs)


def test(model, device, test_loader):
    # model进入测试模式，可以不计算梯度，加快测试速度
    model.eval()
    test_loss = 0
    correct = 0
    # 创建评价函数
    criteria = nn.CrossEntropyLoss()
    with torch.no_grad():  # 无需计算梯度
        for i, (data, label) in enumerate(test_loader):
            # 获取数据的data，和对应的label
            data, label = data.to(device), label.to(device)
            output = model(data)
            # 汇总批次损失
            loss = criteria(output, label)
            test_loss += loss
            # 找到概率值最大的下标
            pred = output.argmax(dim=1, keepdim=True)
            # item返回一个python标准数据类型 将tensor转换
            correct += pred.eq(label.view_as(pred)).sum()

    # 打印loss，准确度
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss / len(test_loader), correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return (test_loss / len(test_loader), 100. *
            correct / len(test_loader.dataset))


def main():
    # 检查是否有可用的gpu，如果没有则使用cpu
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")

    device = torch.device("cuda:0")

    # 数据预处理模块
    data_transform = transforms.Compose([
        # 缩放到224*224
        transforms.Resize((224, 224)),
        # 将图片转换为tensor
        transforms.ToTensor(),
        # 正则化：降低模型复杂度
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    ])

    # 对"train_image_path"中的数据进行变换，并加载
    train_image_path = ""
    orig_set = datasets.ImageFolder(train_image_path, transform=data_transform)
    print(orig_set.class_to_idx)
    # 获取数据集长度
    n = len(orig_set)
    # 打乱数据集
    n_list = np.arange(n)
    np.random.shuffle(n_list)
    # 训练集与测试集比例
    k = 0.95
    train_set = torch.utils.data.Subset(orig_set, n_list[:int(n * k)])
    test_set = torch.utils.data.Subset(orig_set, n_list[int(n * k):])

    # 加载数据集
    BATCH_SIZE = 128  # 每批处理的数据
    num_works = 7  # 加载数据集用的cpu核数
    pin_memory = True  # 使用内存更快
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_works,
        pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=BATCH_SIZE,
        num_workers=num_works,
        pin_memory=pin_memory)

    # 初始化网络
    model = Net()
    # 训练轮数
    epochs = 10
    train(model, device, train_loader, test_loader, epochs)


if __name__ == "__main__":
    main()
