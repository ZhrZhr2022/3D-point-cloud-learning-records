import argparse  # 导入命令行参数解析模块
import os  # 导入操作系统模块
from data_utils.ShapeNetDataLoader import PartNormalDataset  # 从自定义模块中导入数据加载器
import torch  # 导入PyTorch库
import logging  # 导入日志记录模块
import sys  # 导入系统模块
import importlib  # 导入模块动态加载模块
from tqdm import tqdm  # 导入进度条模块
import numpy as np  # 导入NumPy库

# 设置基本目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本文件所在目录的绝对路径
ROOT_DIR = BASE_DIR  # 根目录等于基本目录
sys.path.append(os.path.join(ROOT_DIR, 'models'))  # 将模型目录添加到系统路径中

# 分割类别字典
seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

# 分割标签到类别的映射字典
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """对张量进行one-hot编码"""
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''解析命令行参数'''
    parser = argparse.ArgumentParser('PointNet')  # 创建参数解析器
    parser.add_argument('--batch_size', type=int, default=72, help='测试中的批量大小')  # 添加批量大小参数
    parser.add_argument('--gpu', type=str, default='0', help='指定GPU设备')  # 添加GPU设备参数
    parser.add_argument('--num_point', type=int, default=2048, help='点的数量')  # 添加点的数量参数
    parser.add_argument('--log_dir', type=str, required=True, help='实验根目录')  # 添加实验根目录参数
    parser.add_argument('--normal', action='store_true', default=False, help='使用法线')  # 添加是否使用法线参数
    parser.add_argument('--num_votes', type=int, default=3, help='用投票聚合分割分数')  # 添加投票次数参数
    return parser.parse_args()  # 返回解析结果


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''超参数'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 设置CUDA可见设备
    experiment_dir = 'log/part_seg/' + args.log_dir  # 设置实验目录

    '''日志记录'''
    args = parse_args()  # 解析命令行参数
    logger = logging.getLogger("Model")  # 获取日志记录器
    logger.setLevel(logging.INFO)  # 设置日志记录级别为INFO
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 设置日志格式
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)  # 创建日志文件处理器
    file_handler.setLevel(logging.INFO)  # 设置文件处理器级别为INFO
    file_handler.setFormatter(formatter)  # 设置文件处理器格式
    logger.addHandler(file_handler)  # 添加文件处理器到日志记录器
    log_string('参数设置 ...')  # 记录参数设置信息
    log_string(args)  # 记录参数

    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'  # 数据根目录

    # 加载测试数据集
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("测试数据的数量为：%d" % len(TEST_DATASET))  # 记录测试数据集的数量
    num_classes = 16  # 类别数量
    num_part = 50  # 部件数量

    '''模型加载'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]  # 获取模型名称
    MODEL = importlib.import_module(model_name)  # 动态导入模型模块
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()  # 获取模型并移至GPU
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')  # 加载模型权重
    classifier.load_state_dict(checkpoint['model_state_dict'])  # 加载模型状态字典

    with torch.no_grad():
        test_metrics = {}  # 测试指标字典
        total_correct = 0  # 总正确数
        total_seen = 0  # 总样本数
        total_seen_class = [0 for _ in range(num_part)]  # 每个部件总数列表
        total_correct_class = [0 for _ in range(num_part)]  # 每个部件正确数列表
        shape_ious = {cat: [] for cat in seg_classes.keys()}  # 分类IoU字典
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()  # 设置为评估模式
        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                      smoothing=0.9):
            batchsize, num_point, _ = points.size()  # 获取批量大小、点的数量、特征维度
            cur_batch_size, NUM_POINT, _ = points.size()  # 当前批量大小、点的数量、特征维度
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()  # 转换为浮点型并移至GPU
            points = points.transpose(2, 1)  # 转置点的维度
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()  # 初始化投票池

            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))  # 获取分割预测结果
                vote_pool += seg_pred  # 将预测结果添加到投票池中

            seg_pred = vote_pool / args.num_votes  # 计算平均预测结果
            cur_pred_val = seg_pred.cpu().data.numpy()  # 获取当前预测结果
            cur_pred_val_logits = cur_pred_val  # 获取当前预测结果的Logits
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)  # 初始化当前预测值数组
            target = target.cpu().data.numpy()  # 获取目标标签

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]  # 获取类别
                logits = cur_pred_val_logits[i, :, :]  # 获取Logits
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]  # 获取预测值

            correct = np.sum(cur_pred_val == target)  # 计算正确预测数量
            total_correct += correct  # 更新总正确数
            total_seen += (cur_batch_size * NUM_POINT)  # 更新总样本数

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)  # 更新每个部件的总数
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))  # 更新每个部件的正确数

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]  # 获取当前预测值
                segl = target[i, :]  # 获取当前目标标签
                cat = seg_label_to_cat[segl[0]]  # 获取类别
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]  # 初始化部件IoU列表
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # 如果部件不存在，且预测值也不存在
                        part_ious[l - seg_classes[cat][0]] = 1.0  # 部件IoU为1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))  # 计算部件IoU
                shape_ious[cat].append(np.mean(part_ious))  # 计算部件平均IoU

                print("输出预测错误的 %d 个点：", total_seen - total_correct)
                # 输出预测错误的点的信息
                for i in range(cur_batch_size):
                    for j in range(NUM_POINT):
                        if cur_pred_val[i, j] != target[i, j]:
                            print("点 %d 的预测前后部件类别：%s %s" % (j, target[i, j], cur_pred_val[i, j]))

        all_shape_ious = []  # 所有部件IoU列表
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)  # 添加每个部件IoU
            shape_ious[cat] = np.mean(shape_ious[cat])  # 计算平均部件IoU
        mean_shape_ious = np.mean(list(shape_ious.values()))  # 计算平均部件IoU
        test_metrics['accuracy'] = total_correct / float(total_seen)  # 计算准确率
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.cfloat))  # 计算类别平均准确率
        for cat in sorted(shape_ious.keys()):
            log_string('评估 %s 的mIoU为 %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))  # 记录每个类别的平均IoU
        test_metrics['class_avg_iou'] = mean_shape_ious  # 类别平均IoU
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)  # 实例平均IoU

    log_string('准确率为：%.5f' % test_metrics['accuracy'])  # 记录准确率
    log_string('类别平均准确率为：%.5f' % test_metrics['class_avg_accuracy'])  # 记录类别平均准确率
    log_string('类别平均mIoU为：%.5f' % test_metrics['class_avg_iou'])  # 记录类别平均IoU
    log_string('实例平均mIoU为：%.5f' % test_metrics['inctance_avg_iou'])  # 记录实例平均IoU


if __name__ == '__main__':
    args = parse_args()  # 解析命令行参数
    main(args)  # 主函数入口
