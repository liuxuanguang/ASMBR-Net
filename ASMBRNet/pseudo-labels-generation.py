import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from dataset import DatasetImageMaskContourDist
import glob
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from utils import create_validation_arg_parser
import torch.nn.functional as F
from models import ASMBRNet
import matplotlib.pyplot as plt
import scipy
# def cuda(x):
# return x.cuda(async=True) if torch.cuda.is_available() else x


def build_model(model_type):
    if model_type == "unet_building_extraction":
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
    val_file_names1 = glob.glob(os.path.join(args.val_path, "*.tif"))

    valLoader = DataLoader(DatasetImageMaskContourDist(val_file_names1))

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model = build_model(model_type)
    model = model.to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    for i, (img_file_name, inputs, targets1, targets2) in enumerate(
        tqdm(valLoader)
    ):

        inputs = inputs.to(device)
        # 消融1
        # outputs1 = model(inputs)
        # 消融2
        pred_out1, pred_out2 = model(inputs)
        # pred_out1 = F.interpolate(pred_out1, (h, w), mode="bilinear", align_corners=True)
        # print(pred_out1)
        # pred_out1 = F.sigmoid(pred_out1)
        # print(pred_out1)
        # logits_u_aug, label_u_aug = torch.max(pred_out1, dim=1)
        # print(logits_u_aug)
        # outputs1 = F.sigmoid(outputs1).detach().cpu().numpy().squeeze()
        # outputs2 = F.sigmoid(outputs2).detach().cpu().numpy().squeeze()
        # classnum=1
        # outputs1[outputs1 >= 0.6] = 1   # 0.55效果可以
        # outputs1[outputs1 < 0.6] = 0
        # outputs2[outputs2 >= 0.5] = 1
        # outputs2[outputs2 < 0.5] = 0
        prob = F.sigmoid(pred_out1)
        outputs1 = prob.squeeze(dim=0)

        outputs1[outputs1 < 0.55] = 255
        outputs1[outputs1 < 255] = 0   # 0.55效果可以
        vis = outputs1.detach().cpu().numpy().squeeze()
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
        low_thresh = np.percentile(entropy[outputs1 != 255].detach().cpu().numpy().flatten(), 0.1)
        low_entropy_mask = (entropy.le(low_thresh).float() * (outputs1 != 255).bool())
        low_entropy_mask_vis = low_entropy_mask.detach().cpu().numpy().squeeze()
        low_entropy_mask_vis[low_entropy_mask_vis==0.1]=255
        pesduo_label_path = os.path.join(
            r'/pesduo_label', "fake_" + os.path.basename(img_file_name[0][:-4])+'.png'
        )
        # cv2.imwrite(output_path1, outputs1)
        cv2.imwrite(pesduo_label_path, low_entropy_mask_vis)
