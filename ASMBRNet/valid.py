import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from dataset import DatasetImageMaskContourDist
import glob
# from model_improve_copy import BITU_building_extraction1
from models import ASMBRNet
# from model import leadBackbone, WithDE
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from utils import create_validation_arg_parser
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy
# def cuda(x):
# return x.cuda(async=True) if torch.cuda.is_available() else x
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def build_model(model_type):
    if model_type == "unet_building_extraction":
        # model = BITU_building_extraction1(num_classes=1)
        model = ASMBRNet(num_classes=1)
    return model


if __name__ == "__main__":

    args = create_validation_arg_parser().parse_args()
    # val_path = os.path.join(args.val_path, "*.tif")
    model_file = args.model_file
    save_path = args.save_path
    model_type = args.model_type

    cuda_no = args.cuda_no
    CUDA_SELECT = "cuda:{}".format(cuda_no)
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")
    val_file_names1 = glob.glob(os.path.join(args.val_path, "*.png"))

    valLoader = DataLoader(DatasetImageMaskContourDist(val_file_names1))

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model = build_model(model_type)
    model = model.to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    # 初始化存储预测和真实标签的数组
    all_true_labels = []
    all_pred_labels = []
    for i, (img_file_name, inputs, targets1, targets2) in enumerate(
        tqdm(valLoader)
    ):

        inputs = inputs.to(device)
        outputs1, outputs2 = model(inputs)
        outputs1 = F.sigmoid(outputs1[0]).detach().cpu().numpy().squeeze()
        outputs2 = F.sigmoid(outputs2).detach().cpu().numpy().squeeze()
        # classnum=1
        outputs1[outputs1 >= 0.55] = 255  # 0.55效果可以
        outputs1[outputs1 < 0.55] = 0
        outputs2[outputs2 >= 0.5] = 255
        outputs2[outputs2 < 0.5] = 0
        targets1 = targets1.detach().cpu().numpy().squeeze()
        targets1 = (targets1 > 0).astype(np.uint8)
        all_true_labels.append(targets1)
        all_pred_labels.append(outputs1)
        ## classnum=2
        # outputs1 = outputs1.detach().cpu().numpy().squeeze()
        # outputs2 = outputs2.detach().cpu().numpy().squeeze()
        # res = np.zeros((512, 512), dtype='uint8')
        # indices = np.argmax(outputs1, axis=0)
        # res[indices == 1] = 255
        # res[indices == 0] = 0
        # outputs1.astype(np.uint8)
        # outputs2.astype(np.uint8)
        # output_path1 = os.path.join(
        #     r'/media/lenovo/文档/building_extract/ProposedModel/checkpoints/DY_duibi_transferring//85eProposed/mask1', "m_" + os.path.basename(img_file_name[0][:-4])+'.png'
        # )
        output_path2 = os.path.join('')
        cv2.imwrite(output_path2, outputs1)
        outputs1[outputs1 == 255] = 1
        # cv2.imwrite(output_path2, res)

all_true_labels = np.concatenate(all_true_labels).ravel()
all_pred_labels = np.concatenate(all_pred_labels).ravel()
# 计算指标
oa = accuracy_score(all_true_labels, all_pred_labels)
precision = precision_score(all_true_labels, all_pred_labels)
recall = recall_score(all_true_labels, all_pred_labels)
f1 = f1_score(all_true_labels, all_pred_labels)

# 计算IoU需要按像素计算交集和并集
intersection = np.logical_and(all_true_labels, all_pred_labels)
union = np.logical_or(all_true_labels, all_pred_labels)
iou = np.sum(intersection) / (np.sum(union) + 1e-15)  # 加一个小常数以避免除以零

# 输出结果
print(f"F1 Score: {f1:.4f}")
print(f"IoU: {iou:.4f}")