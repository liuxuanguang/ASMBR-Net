import torch
import os
from tqdm import tqdm
from torch import nn
import numpy as np
import torchvision
from torch.nn import functional as F
import time
import argparse
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, jaccard_score


def evaluate(device, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    true_labels = []
    pred_labels = []
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            _, inputs, target1, _, = data  # 假设 target2 不用于此计算
            inputs = inputs.to(device)
            target1 = target1.to(device)
            outputs = model(inputs)
            # outputs1 = outputs[0]  # 假设 outputs1 是分类相关的输出
            outputs1 = F.sigmoid(outputs[0])
            # 假设我们有一个阈值来将连续输出转换为类别（例如，0.5）
            threshold = 0.55
            pred_labels_iter = (outputs1 > threshold).float().squeeze(1).long()
            true_labels_iter = target1.squeeze(0).long()

            # 收集标签和预测
            true_labels.append(true_labels_iter.cpu().numpy())
            pred_labels.append(pred_labels_iter.cpu().numpy())

            # 计算损失（保持不变）
            mse = torch.nn.MSELoss()
            mask_loss = mse(outputs1, target1.squeeze(0))
            loss = mask_loss  # 假设只考虑 mask_loss
            losses.append(loss.item())

            # 合并所有迭代的标签和预测
    true_labels = np.concatenate(true_labels).reshape(-1)
    pred_labels = np.concatenate(pred_labels).reshape(-1)

    # 计算指标
    f1 = f1_score(true_labels, pred_labels, average='binary')  # 假设是二分类
    # oa = accuracy_score(true_labels, pred_labels)  # 总体精度
    # precision = precision_score(true_labels, pred_labels, average='binary')
    # recall = recall_score(true_labels, pred_labels, average='binary')
    iou = jaccard_score(true_labels, pred_labels, average='binary')  # IoU 可以视为 Jaccard Index

    # 记录损失和指标
    with open('/metrics.txt', 'a') as f:  # 使用'a'模式以追加内容

        f.write(f"Epoch {epoch}:\n")

        f.write(f"Loss: {np.mean(losses):.4f}\n")

        f.write(f"F1 Score: {f1:.4f}\n")

        # f.write(f"OA: {oa:.4f}\n")
        #
        # f.write(f"Precision: {precision:.4f}\n")
        #
        # f.write(f"Recall: {recall:.4f}\n")

        f.write(f"IoU: {iou:.4f}\n\n")

    # return np.mean(losses), time.perf_counter() - start, f1, oa, precision, recall, iou
    return np.mean(losses), time.perf_counter() - start, f1, iou

# def evaluate(device, epoch, model, data_loader, writer):
#     model.eval()
#     losses = []
#     start = time.perf_counter()
#     with torch.no_grad():
#
#         for iter, data in enumerate(tqdm(data_loader)):
#
#             _, inputs, target1, target2, = data
#             inputs = inputs.to(device)
#             target1 = target1.to(device)
#             target2 = target2.to(device)
#             outputs = model(inputs)
#             outputs1 = outputs[0]
#             outputs2 =outputs[1]
#
#             mse = torch.nn.MSELoss().cuda()
#
#             mask_loss = mse(outputs1, target1.squeeze(0))
#             boun_loss = mse(outputs2, target2.squeeze(1))
#             loss = mask_loss + boun_loss
#             losses.append(loss.item())
#
#         writer.add_scalar("Dev_Loss", np.mean(losses), epoch)
#
#     return np.mean(losses), time.perf_counter() - start


def visualize(device, epoch, model, data_loader, writer, val_batch_size, train=False):
    def save_image(image, tag, val_batch_size):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(
            image, nrow=int(np.sqrt(val_batch_size)), pad_value=0, padding=25
        )
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            _, inputs, targets, _, = data

            inputs = inputs.to(device)

            targets = targets.to(device)
            outputs = model(inputs)

            output_mask = outputs[0].detach().cpu().numpy()
            output_final = np.argmax(output_mask, axis=1).astype(float)

            output_final = torch.from_numpy(output_final).unsqueeze(1)

            if train == "True":
                save_image(targets.float(), "Target_train", val_batch_size)
                save_image(output_final, "Prediction_train", val_batch_size)
            else:
                save_image(targets.float(), "Target", val_batch_size)
                save_image(output_final, "Prediction", val_batch_size)

            break


def create_train_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument("--train_path", default='', type=str, help="path to img jpg files")
    parser.add_argument("--val_path", default='', type=str, help="path to img jpg files")
    parser.add_argument(
        "--model_type",
        default='unet_building_extraction',
        type=str
    )
    parser.add_argument("--object_type", type=str, help="Dataset.")
    parser.add_argument(
        "--distance_type",
        type=str,
        default="dist_signed",
        help="select distance transform type - dist_mask,dist_contour,dist_signed",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="train batch size")
    parser.add_argument(
        "--val_batch_size", type=int, default=8, help="validation batch size"
    )
    parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")
    parser.add_argument(
        "--use_pretrained", type=bool, default=False, help="Load pretrained checkpoint."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="If use_pretrained is true, provide checkpoint.",
    )
    parser.add_argument("--save_path", default='', type=str, help="Model save path.")

    return parser


def create_validation_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument(
        "--model_type",
        default='unet_building_extraction',
        type=str
    )
    parser.add_argument("--val_path", default='', type=str, help="path to img jpg files")
    parser.add_argument("--model_file", default='', type=str, help="model_file")
    parser.add_argument("--save_path", default='', type=str, help="results save path.")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")

    return parser


import os
import random
import numpy as np
from scipy import stats


def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]


def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)


def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


def seprate_batch(dataset, batch_size):
    """Yields lists by batch"""
    num_batch = len(dataset) // batch_size + 1
    batch_len = batch_size
    # print (len(data))
    # print (num_batch)
    batches = []
    for i in range(num_batch):
        batches.append([dataset[j] for j in range(batch_len)])
        # print('current data index: %d' %(i*batch_size+batch_len))
        if (i + 2 == num_batch): batch_len = len(dataset) - (num_batch - 1) * batch_size
    return (batches)


def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255


def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def ImageValStretch2D(img):
    img = img * 255
    # maxval = img.max(axis=0).max(axis=0)
    # minval = img.min(axis=0).min(axis=0)
    # img = (img-minval)*255/(maxval-minval)
    return img.astype(int)


def ConfMap(output, pred):
    # print(output.shape)
    n, h, w = output.shape
    conf = np.zeros(pred.shape, float)
    for h_idx in range(h):
        for w_idx in range(w):
            n_idx = int(pred[h_idx, w_idx])
            sum = 0
            for i in range(n):
                val = output[i, h_idx, w_idx]
                if val > 0: sum += val
            conf[h_idx, w_idx] = output[n_idx, h_idx, w_idx] / sum
            if conf[h_idx, w_idx] < 0: conf[h_idx, w_idx] = 0
    # print(conf)
    return conf


def accuracy(pred, label):
    valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def align_dims(np_input, expected_dims=2):
    dim_input = len(np_input.shape)
    np_output = np_input
    if dim_input > expected_dims:
        np_output = np_input.squeeze(0)
    elif dim_input < expected_dims:
        np_output = np.expand_dims(np_input, 0)
    assert len(np_output.shape) == expected_dims
    return np_output


def binary_accuracy(pred, label):
    pred = align_dims(pred, 2)
    label = align_dims(label, 2)
    pred = (pred >= 0.5)
    label = (label >= 0.5)

    TP = float((pred * label).sum())
    FP = float((pred * (1 - label)).sum())
    FN = float(((1 - pred) * (label)).sum())
    TN = float(((1 - pred) * (1 - label)).sum())
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    IoU = TP / (TP + FP + FN + 1e-10)
    acc = (TP + TN) / (TP + FP + FN + TN)
    F1 = 0
    if acc > 0.99 and TP == 0:
        precision = 1
        recall = 1
        IoU = 1
    if precision > 0 and recall > 0:
        F1 = stats.hmean([precision, recall])
    return acc, precision, recall, F1, IoU


def binary_accuracy_softmax(pred, label):
    valid = (label < 2)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # imPred += 1
    # imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass + 1))
    # print(area_intersection)

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass + 1))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass + 1))
    area_union = area_pred + area_lab - area_intersection
    # print(area_pred)
    # print(area_lab)

    return (area_intersection, area_union)


def CaclTP(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # imPred += 1
    # imLab += 1
    # # Remove classes from unlabeled pixels in gt image.
    # # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    TP = imPred * (imPred == imLab)
    (TP_hist, _) = np.histogram(
        TP, bins=numClass, range=(1, numClass + 1))
    # print(TP.shape)
    # print(TP_hist)

    # Compute area union:
    (pred_hist, _) = np.histogram(imPred, bins=numClass, range=(1, numClass + 1))
    (lab_hist, _) = np.histogram(imLab, bins=numClass, range=(1, numClass + 1))
    # print(pred_hist)
    # print(lab_hist)
    # precision = TP_hist / (lab_hist + 1e-10) + 1e-10
    # recall = TP_hist / (pred_hist + 1e-10) + 1e-10
    # # print(precision)
    # # print(recall)
    # F1 = [stats.hmean([pre, rec]) for pre, rec in zip(precision, recall)]
    # print(F1)

    # print(area_pred)
    # print(area_lab)

    return (TP_hist, pred_hist, lab_hist)