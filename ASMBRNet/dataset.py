import torch
import numpy as np
import cv2
from PIL import Image, ImageFile

# ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import io


class DatasetImageMaskContourDist(Dataset):

    # dataset_type(cup,disc,polyp),
    # distance_type(dist_mask,dist_contour,dist_signed)

    def __init__(self, file_names):

        self.file_names = file_names


    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, idx):

        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name)
        contour = load_contour(img_file_name)
        # dist = load_distance(img_file_name, self.distance_type)
        # return img_file_name, image, mask, contour, dist
        return img_file_name, image, mask, contour


def load_image(path):

    img = Image.open(path)
    data_transforms = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img = data_transforms(img)

    return img


def load_mask(path):

    mask = cv2.imread(path.replace("image", "label").replace("png", "png"), 0)
    mask[mask == 255] = 1

    return torch.from_numpy(np.expand_dims(mask, 0)).long()


def load_contour(path):

    contour = cv2.imread(path.replace("image", "contour").replace("png", "png"), 0)
    contour[contour == 255] = 1

    return torch.from_numpy(np.expand_dims(contour, 0)).long()


def load_distance(path, distance_type):

    if distance_type == "dist_mask":
        path = path.replace("image", "dist_mask").replace("jpg", "mat")
        dist = io.loadmat(path)["mask_dist"]

    if distance_type == "dist_contour":
        path = path.replace("image", "dist_contour").replace("jpg", "mat")
        dist = io.loadmat(path)["contour_dist"]

    if distance_type == "dist_signed":
        path = path.replace("image", "dist_signed").replace("jpg", "mat")
        dist = io.loadmat(path)["dist_norm"]

    return torch.from_numpy(np.expand_dims(dist, 0)).float()
