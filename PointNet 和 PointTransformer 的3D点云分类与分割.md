
# PointNet 和 PointTransformer 的3D点云分类与分割
注意：本文一些对于PointNet的理解来自Bilibili：【[5分钟点云学习] #02 PointNet 开山之作】 https://www.bilibili.com/video/BV1bh4y1o7Ji/?share_source=copy_web&vd_source=c4718ddb7961a0a304c6958bae5a7589
这位UP主讲的很好！
## 一、何为点云？

​		点云是某个坐标系下的点的数据集。点包含了丰富的信息，包括**三维坐标 X，Y，Z**、颜色、分类值、强度值、时间等等。

​		使用的点云数据集：ModelNet10、ModelNet40、ShapeNet

​		截取ModelNet40中某个点云的其中几个点，每一行代表一个点，前三个是三维的XYZ坐标，后三个则是对应的法向量

```
-0.098790,-0.182300,0.163800,0.829000,-0.557200,-0.048180
0.994600,0.074420,0.010250,0.331800,-0.939500,0.085320
0.189900,-0.292200,-0.926300,0.239000,-0.178100,-0.954500
-0.989200,0.074610,-0.012350,-0.816500,-0.250800,-0.520100
0.208700,0.221100,0.565600,0.837600,-0.019280,0.545900
```

​		而对于ShapeNet数据集，这个专门用来分割，同样截取几个点，前六个和上面是一样的，而最后一个表示该点属于哪一部分。

```
0.073880 0.169750 -0.193260 0.006987 0.999600 0.027960 1.000000
0.152860 0.150500 0.243550 -0.034790 -0.999200 0.017640 1.000000
0.209480 0.150500 0.290810 0.000464 -1.000000 0.000590 1.000000
0.186060 0.150500 0.276140 0.000000 -1.000000 0.000000 1.000000
0.087520 0.169750 0.258150 0.027970 0.999500 -0.014000 1.000000
```

## 二、无法像处理图像一样对点云进行分类与分割

​		对于最常处理的二维图像，图像数据具有完美的结构特性，知道任意一个像素它的邻域是什么。如果改变了像素的顺序，图像内容也发生变化。二维图像这样的性质称为有序性，二维图像这种数据也叫结构化数据。那么我们就可以使用很多方法来提取图像的特征，比如通过最常用的**卷积**就可以提取颜色、几何和语义特征。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/d7616abca0724efabe4226e66695dd49.png#pic_center)

​		但是点云数据并不具备这样优秀的性质，对于一个点云，我们不存在唯一的顺序来处理点云 ，即使更改了点云中每一个元素的顺序，点云仍然会保持原有的几何和语类别信息，那如果我们还是选择像处理图像那样（比如上文所说的卷积操作）去处理同一个点云，从不同的方向入手处理最后得到的特征是不一样的，那这个特征就无法实现对点云的分类以及分割。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/fe686cce0b5e454eaa00a1f32a8fd2f4.png#pic_center)

## 三、PointNet的做法

​		核心原理是对于一个点云数据，先将每个点按照随即一种顺序进行排列，排列完成之后经过一个MLP进行升维，比如可以将三维特征升维编码到八维特征，那再对每一维的特征取最大值（Max Pooling最大值池化），那么八维就可以得到八个最大值，将这八个最大值来作为该点云的特征。由于进行了升维和最大值池化的操作，那么我们无论采取哪一种顺序排列点云，最终得到的点云特征一定是一样的，那这样的话我们就成功的提取到了点云特征，而且保证了不同的点云处理得到的特征也不相同，那接下来就可以使用很多方式对特征学习从而完成我们想要的分类或者分割任务。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/765ca8317d86413885e6ba188db1fbae.png#pic_center)

​		当然，这里说使用了最大值池化，实际上无论是取最大值还是平均值，还是最小值等等操作，都可以唯一代表这一维，PointNet的作者使用了最大值池化的原因则是因为，相比于其他做法，最大值的池化的效果是最好的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/cb593e7a2687445e85bc16f820f20217.png#pic_center)

## 四、PointNet的网络结构

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/ac7e776393874d7a980c7490395c0481.png#pic_center)

​		该网络的上面是PointNet的分类任务，下面是分割任务。

### 1、分类任务	

​		先看分类，可以看到，n为一个点云数据中点的个数，3则是三维数据，这样的一个nx3的点云数据先经过一个T-Net微型网络进行处理，然后通过一个MLP升维编码到64维，再次通过一个T-Net微型网络处理，再通过MLP依次升维到128、1024维，然后对得到的一个nx1024的数据进行最大值池化，得到一个1x1024的全局特征，对这个特征通过MLP进行降维（依次降到512、256、k），最后的k维则是需要分类的类别数量。降到k维之后就可以知道哪一个类别对应的值最高，即属于哪一类。

#### Ⅰ、第一个T-Net

```python
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
```

​		从代码我们可以看到，这个所谓的微型网络，实际上就是使用了PointNet进行了一次处理，从3维依次进行升维到64、128、1024，然后进行最大值池化，再依次降维（从1024维降到512，再降到256，再降到9），之所以降到9（这里是特征降到9维，为1x9），是为了变换成一个3x3的变换矩阵，原本nx3的数据乘上这个变换矩阵仍然是nx3。那经过T-Net处理之后实际上是初步的提取了点云的特征，并把这些特征合并到原始数据中，这样做实际上可以提升提升网络对点云刚性变换的泛化能力，使得最后处理的效果更好。

#### Ⅱ、第二个T-Net

```python
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
```

​		同样的操作，只不过最终降维时不是降到9维了，而是k*k维，当然根据网络结构这里的k是64，也是得到一个特征与原数据相乘，原理都是一样的。

### 2、分割任务

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/cf13e82099a3497ea3ba6a974549a672.png#pic_center)

​			与分类不同的点是，将最后得到的全局特征再拼接第一个感知机得到的特征（类别信息），这样的话既保留了整体点云的特征，又保留了点与点之间的差异性，那就可以进行分割任务了。

## 五、PointNet的代码实现

```python
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
```

```python
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
```

​		STN3d和STNkd是上面介绍的两个T-Net，代码在上面。

## 六、运行结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/f4d2a5928df0454689cc6d05819558f1.png#pic_center)


​		经过100轮的训练，最终得到的结果：

```xml
Train Instance Accuracy: 0.980188
Test Instance Accuracy: 0.927937, Class Accuracy: 0.901373
Best Instance Accuracy: 0.927937, Class Accuracy: 0.901373
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/ff84ffb05df749fbaedb915b141827ea.png#pic_center)

​		测试集的结果：

```xml
Test Instance Accuracy: 0.926214, Class Accuracy: 0.899735
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/bbbc7a51dfbb4d779a49aa65af34e409.png#pic_center)

​		少测试几个，效果仍然不错。


![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/3f7775b3136d4d5f89562e397208a128.png#pic_center)


​		分割图像的可视化。





# 探寻别的方法——Point Transformer

​		我们前面说过，点云数据是无序的，不规则的，实际上Transformer能够有效应对点云数据的不规则性和无序性，并且通过自注意力机制，捕捉全局信息，从而更好地理解点云数据的整体结构和特征。因此，**将Transformer应用于处理三维点云数据的任务已经成为了一种趋势。**

​		或者说，Transformer最初就是为了处理序列数据，而点云不恰好可以当作序列数据处理吗，而且还有更好的地方在于，我们可以直接将每一个点的坐标作为一个序列丢给Transformer，甚至不需要手动分割序列数据了，那我们就可以使用Transformer去处理点云，参考的论文是Point Transformer。



## 一、Point Transformer的注意力机制



![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/ed11804b1ad84f9482af992eea5b66c6.png#pic_center)

```
φ, ψ, and α are pointwise feature transformations, such as linear projections or MLPs.δ is a position encoding function and ρ is a normalization function such as softmax.
```

原论文是这样说的，φ、 ψ和α是逐点特征变换，如线性投影或MLP。δ是位置编码函数，ρ是归一化函数，如softmax函数。

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        """
        在点云处理中，保持全局特征和局部特征的结合是非常重要的。
            全局特征（q）: q 是对整个特征空间进行线性变换后的结果，它保留了输入特征的全局信息。
            局部特征（k 和 v）: k 和 v 是基于局部邻域的特征，这些特征可以捕捉点与其邻近点之间的关系和局部结构。
        通过这种方式，模型可以同时考虑全局和局部信息，在计算注意力权重时，不仅关注点本身的特征，还能考虑其邻近点的特征。
        """
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn
```

​		结合着代码我们不难发现，w_qs、w_ks、w_vs实际上对应的就是φ、 ψ和α，也正如论文中所说，他们是输入点的数据经过线性层进行线性变换得到的，实际上也应该对应了Transformer中的Q、K、V三个矩阵。当然这里也进行了简单的处理，比如说Q直接使用w_qs,K和V则是通过KNN取了局部的一些点，这样的话Q为全局特征，K和V为局部特征，通过这种方式，模型可以同时考虑全局和局部信息，在计算注意力权重时，不仅关注点本身的特征，还能考虑其邻近点的特征。

## 二、Point Transformer的网络结构
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/17776147ea49402da7ee2bbfcac83f16.png#pic_center)


​		该网络结构中上面的是分割任务，下面的分类任务。两种网络实际上都是不同的网络层（比如Point Transformer层、Transition Down层、Transition UP层、MLP层、Global AvgPooling层）组合而成的。那我们接下来着重介绍这几个网络层。

### 1、Point Transformer层
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/a869ddeb6c1148ca8388ca07fa0855fb.png#pic_center)


​		可以看到，Point Transformer层将输入数据通过Linear线性变换，然后经过point transformer操作，再同过Linear层得到输出。


![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/793fedb0c57c4b0d88cef53f3bd04493.png#pic_center)
​		

```
The mapping function γ is an MLP with two linear layers and one ReLU nonlinearity.
```

这就是point transformer的具体操作，其中映射函数γ是一个具有两个线性层和一个非线性ReLU的MLP。

前面的TransformerBlock代码中：

```
self.fc_gamma = nn.Sequential(  # γ映射函数
    nn.Linear(d_model, d_model),
    nn.ReLU(),
    nn.Linear(d_model, d_model)
)
```

完整的过程是：输入的φ、 ψ和α分别通过linear操作，位置编码δ经过fc_delta这样一个MLP（和γ映射函数结构一样），然后带入到注意力机制的公式中计算出注意力attn，在经过简单的处理得到y和p。

### 2、Transition Down层
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/f748057c6e5b45919b0a8ffc8df7f312.png#pic_center)

​		这个模块实质上是一个下采样操作，具体来说通过FPS选取若干个点，以选取的每个点为中心使用KNN算法，将该中心点的K个邻居的信息用max pooling聚合起来，从而实现下采样的操作。总的来说该模块用于减小点云的分辨率，聚合特征。

```python
class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)
        
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, knn=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)                     # fps knn
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, knn=self.knn)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):                 # mlp
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0].transpose(1, 2)          # max pooling
        return new_xyz, new_points
```

### 3、Transition UP层
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/9813252601f948348bdb65b1eaecceeb.png#pic_center)

这个模块实现上采样的操作，通过对两个输入点集的特征进行线性变换、插值以及求和来实现点云特征的上采样和融合，可以增强点云特征表示的精度和丰富性。

```python
class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2
        
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1：输入点的位置数据，形状为 [B, C, N]，B 是批次大小，C 是坐标维度，N 是点的数量。
            xyz2：采样的输入点位置数据，形状为 [B, C, S]，S 是采样点的数量。
            points1：输入点的特征数据，形状为 [B, D, N]，D 是特征维度。
            points2：采样点的特征数据，形状为 [B, D, S]。
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        # 基于距离的加权插值
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)  # dist_recip 是距离的倒数，表示距离越小，加权系数越大。
            norm = torch.sum(dist_recip, dim=2, keepdim=True)  # 归一化
            weight = dist_recip / norm            # 计算每个点到其三个最近采样点的距离并求倒数--归一化后的加权系数
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)  # 得到的最近的三个点插到原来point2中，再乘上计算的归一化后的加权系数

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)   # 插到point1中
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):   # 插值特征可能比较粗糙 ----进一步提取、增强和优化特征
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

```

## 三、运行结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/1e11ec2b3b964fcc82371cc81d725723.png#pic_center)

​		测试分类效果，训练200轮，用了二十多个小时，最后得到的效果：
```
Train Instance Accuracy: 0.888755
Test Instance Accuracy: 0.897984, Class Accuracy: 0.858620
Best Instance Accuracy: 0.904435, Class Accuracy: 0.869475
```

