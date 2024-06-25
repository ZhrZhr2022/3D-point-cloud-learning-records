import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)  # 定义三个一维卷积层，分别是通道数分别为 channel、64、128 和 1024
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)  # 定义三个全连接层，分别是输入维度为 1024、512、256，输出维度为 512、256 和 9
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)  # 五个批标准化层，分别应用于五个卷积层和全连接层

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # 依次对输入数据进行一维卷积操作，并在每个卷积后应用批标准化和 ReLU 激活函数
        x = torch.max(x, 2, keepdim=True)[0]  # 沿着数据的维度 2（即每个通道），取最大值
        x = x.view(-1, 1024)  # 将数据展平为二维形状

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # 通过两个全连接层，依次应用批标准化和 ReLU 激活函数

        """
            [1 0 0
             0 1 0
             0 0 1]
        """
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden  # 生成一个单位矩阵，并将其与前面的全连接层的输出相加，得到一个 3x3 的变换矩阵
        x = x.view(-1, 3, 3)  # 将结果变换为 3x3 的矩阵形式
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()  # batch size/每个点的维度/点的数量
        trans = self.stn(x)  # 第一个T-net
        x = x.transpose(2, 1)  # 进行转置  便于进行矩阵乘法
        if D > 3:   # 维度大于 3 表示包含了除了坐标之外的其他特征   额外特征提取出来，放入 feature 中
            feature = x[:, :, 3:]
            x = x[:, :, :3]  # x 只保留坐标信息
        x = torch.bmm(x, trans)  # 和变换矩阵相乘，实现空间变换
        if D > 3:
            x = torch.cat([x, feature], dim=2)  # 额外特征再添加回去
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))  # 第一个MLP 升维 3->64

        if self.feature_transform:  # 启用了特征变换，进行第二次T-net变换
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]  # max pooling 最大值池化，取每个通道的最大值
        x = x.view(-1, 1024)  # 将结果展平为二维形状
        if self.global_feat:
            return x, trans, trans_feat
        else:  # 将全局特征 x 和之前的点特征 point feat 拼接起来返回。
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):  # 用于计算特征变换矩阵的正则化损失，约束特征变换矩阵接近于单位矩阵，保证特征变换的稳定性和有效性
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))  # 特征变换矩阵与其转置矩阵的乘积 然后与单位矩阵进行比较  计算它们之间的欧几里得范数的平均值作为正则化损失
    return loss
