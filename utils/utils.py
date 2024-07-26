import numpy as np
import tensorboardX
from sklearn.metrics import log_loss, accuracy_score, precision_score, average_precision_score, roc_auc_score, \
    recall_score, confusion_matrix
import torch
import torch.nn.functional as F


class Test_time_agumentation(object):

    def __init__(self, is_rotation=True):
        self.is_rotation = is_rotation

    def __rotation(self, img):
        """
        clockwise rotation 90 180 270
        """
        img90 = img.rot90(-1, [2, 3])  # 1 逆时针； -1 顺时针
        img180 = img.rot90(-1, [2, 3]).rot90(-1, [2, 3])
        img270 = img.rot90(1, [2, 3])
        return [img90, img180, img270]

    def __inverse_rotation(self, img90, img180, img270):
        """
        anticlockwise rotation 90 180 270
        """
        img90 = img90.rot90(1, [2, 3])  # 1 逆时针； -1 顺时针
        img180 = img180.rot90(1, [2, 3]).rot90(1, [2, 3])
        img270 = img270.rot90(-1, [2, 3])
        return img90, img180, img270

    def __flip(self, img):
        """
        Flip vertically and horizontally
        """
        return [img.flip(2), img.flip(3)]

    def __inverse_flip(self, img_v, img_h):
        """
        Flip vertically and horizontally
        """
        return img_v.flip(2), img_h.flip(3)

    def tensor_rotation(self, img):
        """
        img size: [H, W]
        rotation degree: [90 180 270]
        :return a rotated list
        """
        # assert img.shape == (1024, 1024)
        return self.__rotation(img)

    def tensor_inverse_rotation(self, img_list):
        """
        img size: [H, W]
        rotation degree: [90 180 270]
        :return a rotated list
        """
        # assert img.shape == (1024, 1024)
        return self.__inverse_rotation(img_list[0], img_list[1], img_list[2])

    def tensor_flip(self, img):
        """
        img size: [H, W]
        :return a flipped list
        """
        # assert img.shape == (1024, 1024)
        return self.__flip(img)

    def tensor_inverse_flip(self, img_list):
        """
        img size: [H, W]
        :return a flipped list
        """
        # assert img.shape == (1024, 1024)
        return self.__inverse_flip(img_list[0], img_list[1])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    def __init__(self, model_name, header):
        self.header = header
        self.writer = tensorboardX.SummaryWriter(model_name)

    def __del(self):
        self.writer.close()

    def log(self, phase, values):
        epoch = values['epoch']

        for col in self.header[1:]:
            self.writer.add_scalar(phase + "/" + col, float(values[col]), int(epoch))


def calculate_metrics(outputs, targets, metric_name='acc'):
    if len(targets.data.numpy().shape) > 1:
        _, targets = torch.max(targets.detach(), dim=1)
    if len(outputs.data.numpy().shape) > 1 and outputs.data.numpy().shape[1] == 1:  # 尾部是sigmoid
        outputs = torch.cat([1 - outputs, outputs], dim=1)

    # print(outputs.shape, targets.shape, pred_labels.size())
    if metric_name == 'acc':
        pred_labels = torch.max(outputs, 1)[1]
        # print(targets, pred_labels)
        return accuracy_score(targets.data.numpy(), pred_labels.detach().numpy())
    elif metric_name == 'auc':
        pred_labels = outputs[:, 1]  # 为假的概率
        return roc_auc_score(targets.data.numpy(), pred_labels.detach().numpy())


def calculate_fnr(y_true, y_pred):
    """
    计算False Negative Rate

    参数:
    y_true -- 真实的类别标签
    y_pred -- 预测的类别标签

    返回:
    fnr -- False Negative Rate
    """
    # 生成混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 计算FNR
    fnr = fn / float(fn + tp)

    return fnr


def compute_confusion_matric(outputs, targets):
    """
         ｜   预测   ｜
    -----------------
    真｜正｜ TP ｜ FN ｜
       --------------
    值｜负｜ FP ｜ TN ｜
    -----------------
    :param outputs:
    :param targets:
    :return: tp, fp, tn ,fn
    """
    part = outputs ^ targets
    pcount = np.bincount(part)
    tp_list = list(outputs & targets)
    fp_list = list(outputs & ~targets)
    tp = tp_list.count(1)
    fp = fp_list.count(1)
    tn = pcount[0] - tp
    fn = pcount[1] - fp

    return tp, fp, tn, fn


if __name__ == '__main__':
    pass
