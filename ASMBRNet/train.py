import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
from torch.optim import Adam
from tqdm import tqdm
import logging
from torch import nn
import numpy as np
import torchvision
import random
from tensorboardX import SummaryWriter
from utils import visualize, evaluate, create_train_arg_parser
from losses import LossUNet, LossDCAN, LossDMTN, LossPsiNet
from model import leadBackbone, WithoutDoubleEncoder, WithDE
from models import UNet_DMTN, UNet_DCAN
from dataset import DatasetImageMaskContourDist
from torch.utils.data import DataLoader
from losses import *
from Boundary_discriminator import ProposedDiscriminator as D_Net
# from Boundary_discriminator import FCDiscriminator as D_Net
from d_model_utils import intersectionAndUnion, AverageMeter
from model_improve_copy import BITU_building_extraction1
from model_improve_copy import ASMBRNet

# train proposed
#####################################################
def define_loss(loss_type, weights=[1, 1, 1]):
    if loss_type == "unet_building_extraction":
        criterion = LossDMTN(weights)
    return criterion


def build_model(model_type):
    if model_type == "unet_building_extraction":
        model = ASMBRNet(num_classes=1)
    return model


def train_model(model, targets, model_type, criterion, optimizer):

    if model_type == "unet_building_extraction":
        optimizer.zero_grad()
        with (torch.set_grad_enabled(True)):

            MSE_loss = torch.nn.MSELoss().cuda()
            outputs = model(inputs)
            loss_mask = MSE_loss(outputs[0].float(), targets[0].float())
            loss_boun = MSE_loss(outputs[1].float(), targets[1].float())
            # criterion(outputs[0], outputs[1], targets[0], targets[1])
            # loss = criterion(outputs, targets)
            loss = loss_mask + loss_boun
            loss.backward()
            optimizer.step()
    return loss, outputs


if __name__ == "__main__":

    args = create_train_arg_parser().parse_args()

    CUDA_SELECT = "cuda:{}".format(args.cuda_no)
    log_path = args.save_path + "/summary"
    writer = SummaryWriter(log_dir=log_path)

    logging.basicConfig(
        filename="".format(args.object_type),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.INFO,
    )
    logging.info("")

    train_file_names = glob.glob(os.path.join(args.train_path, "*.tif"))
    random.shuffle(train_file_names)
    val_file_names = glob.glob(os.path.join(args.val_path, "*.tif"))
    Dpred_loss_meter = AverageMeter()
    Dgt_loss_meter = AverageMeter()
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_type)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model_D = D_Net(1).cuda()
    model_D.train()
    model = model.to(device)
    # for CEparam in model.downsample_layers.parameters():
    #     CEparam.requires_grad = False
    # for TEparam in model.stages.parameters():
    #     TEparam.requires_grad = False
    # for TEparam1 in model.assisting_down.parameters():
    #     TEparam1.requires_grad = False

    # To handle epoch start number and pretrained weight
    epoch_start = "0"
    if args.use_pretrained:
        print("Loading Model {}".format(os.path.basename(args.pretrained_model_path)))
        model.load_state_dict(torch.load(args.pretrained_model_path))
        epoch_start = os.path.basename(args.pretrained_model_path).split(".")[0]
        print(epoch_start)

    trainLoader = DataLoader(
        DatasetImageMaskContourDist(train_file_names),
        batch_size=args.batch_size,
    )
    devLoader = DataLoader(
        DatasetImageMaskContourDist(val_file_names)
    )
    displayLoader = DataLoader(
        DatasetImageMaskContourDist(val_file_names),
        batch_size=args.val_batch_size,
    )
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = define_loss(args.model_type)
    optimizer_D = Adam(model_D.parameters(), lr=1e-3, betas=(0.9, 0.99))
    optimizer_D.zero_grad()
    pred_label = 0
    GT_label = 1
    bce_loss = torch.nn.BCEWithLogitsLoss()
    for epoch in tqdm(
        range(int(epoch_start) + 1, int(epoch_start) + 1 + args.num_epochs)
    ):

        global_step = epoch * len(trainLoader)
        running_loss = 0.0
        for i, (img_file_name, inputs, targets1, targets2) in enumerate(
        # for i, (img_file_name, inputs, targets1, targets2, targets3) in enumerate(
            tqdm(trainLoader)
        ):

            model.train()
            inputs = inputs.to(device).float()
            targets1 = targets1.to(device).float()
            targets2 = targets2.to(device).float()
            targets = [targets1, targets2]
            loss, outputs = train_model(model, targets, args.model_type, criterion, optimizer)

            ### train D
            # unfreeze D
            for param in model_D.parameters():
                param.requires_grad = True

            # train D with prediction map
            D_out = model_D(F.sigmoid(outputs[1]).detach())
            loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(pred_label).to(device))
            loss_D.backward()
            Dpred_loss_meter.update(loss_D.cpu().detach().numpy())

            # train D with GT map
            D_out = model_D(targets2)
            loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(GT_label).to(device))
            loss_D.backward()
            Dgt_loss_meter.update(loss_D.cpu().detach().numpy())
            optimizer_D.step()
            writer.add_scalar("loss", loss.item(), epoch)

            running_loss += loss.item() * inputs.size(0)
            # print('segloss: %.2f, Dpred_loss: %.2f, Dgt_loss: %.2f' % (loss, Dpred_loss_meter.val, Dgt_loss_meter.val))
        epoch_loss = running_loss / len(train_file_names)

        if epoch % 1 == 0:
            dev_loss, dev_time, f1, iou = evaluate(device, epoch, model, devLoader, writer)
            writer.add_scalar("loss_valid", dev_loss, epoch)
            visualize(device, epoch, model, displayLoader, writer, args.val_batch_size)
            # print("Global Loss:{} Val Loss:{} f1:{}, oa:{}, precision:{}, recall:{}, iou:{}".format(epoch_loss, dev_loss, f1, oa, precision, recall, iou))
            print("Global Loss:{} Val Loss:{} f1:{}, iou:{}".format(epoch_loss, dev_loss, f1, iou))
        else:
            # print("Global Loss:{} ".format(epoch_loss))
            print("Global Loss:{} ".format(loss))

        # logging.info("epoch:{} train_loss:{} ".format(epoch, epoch_loss))
        logging.info("epoch:{} train_loss:{} ".format(epoch, loss))
        if epoch % 5 == 0:
            torch.save(
                model.state_dict(), os.path.join(args.save_path, str(epoch) + '_' + str(dev_loss) + ".pt")
            )


