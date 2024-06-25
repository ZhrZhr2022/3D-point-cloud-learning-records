import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:  # 使用法向量
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)  # 丢弃层，丢弃概率为 0.4
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)  # 两个批标准化层，分别应用于第一个全连接层和第二个全连接层
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)  # 经过特征提取器 self.feat 提取特征
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)  # 通过三个全连接层和激活函数 ReLU 进行特征变换
        x = F.log_softmax(x, dim=1)   # 得到预测结果
        return x, trans_feat  # 返回预测结果和特征变换


class get_loss(torch.nn.Module):  # 负对数似然损失 和 特征变换正则化损失 组合
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale  # 控制特征变换正则化损失在总损失中的权重

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)  # 计算预测值 pred 和实际标签 target 之间的负对数似然损失
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)  # 计算特征变换 正则化损失

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
