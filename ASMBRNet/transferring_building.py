import cv2
import numpy as np
import random

def get_building_corners(label_image):
    # 将标签图像二值化
    _, binary_label = cv2.threshold(label_image, 1, 255, cv2.THRESH_BINARY)
    binary_label = cv2.cvtColor(binary_label, cv2.COLOR_BGR2GRAY)
    # 寻找轮廓
    contours, _ = cv2.findContours(binary_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    building_corners = []
    for contour in contours:
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        # 计算四个角点的坐标
        top_left = (x, y)
        top_right = (x + w, y)
        bottom_left = (x, y + h)
        bottom_right = (x + w, y + h)
        center = ((x + x + w) / 2, (y + y + h) / 2)  # 中心坐标
        building_corners.append([top_left, top_right, bottom_left, bottom_right, center])
    return building_corners


def transfer_buildings(imageA, imageB, labelA, preB):
    # 获取图像A和图像B的大小
    label_A = cv2.imread(labelA)
    pre_B = cv2.imread(preB)
    imgA = cv2.imread(imageA)
    imgB = cv2.imread(imageB)
    cornersA = get_building_corners(label_A)
    LB = np.array(Image.open(preB))
    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]
        # 检查BL中建筑物的占比是否小于10%
    building_ratio = (LB == 255).sum() / (hA * wA)
    if building_ratio < 0.3:
        print("BL中建筑物占比小于10%，开始转移建筑物...")
        # 将AL获取的建筑物范围映射到影像A上，并将A范围上的影像转移到影像B中
        for topLeftA, topRightA, bottomLeftA, bottomRightA, centerA in cornersA:
            topLeftB = (int(topLeftA[0] * wB / wA), int(topLeftA[1] * hB / hA))
            topRightB = (int(topRightA[0] * wB / wA), int(topRightA[1] * hB / hA))
            bottomLeftB = (int(bottomLeftA[0] * wB / wA), int(bottomLeftA[1] * hB / hA))
            bottomRightB = (int(bottomRightA[0] * wB / wA), int(bottomRightA[1] * hB / hA))
            # 从图像A中裁剪出指定区域
            cropped_image = imgA[topLeftA[1]:bottomRightA[1], topLeftA[0]:bottomRightA[0]]
            # 将裁剪出的区域复制到图像B的相应位置
            imgB[topLeftB[1]:bottomRightB[1], topLeftB[0]:bottomRightB[0]] = cropped_image
            # 显示复制后的影像B，仅供查看，实际应用中可选择不显示
        cv2.imshow('Image B after Copy', imgB)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('/Trans_results/' + imageA[-15:], imgB)
    else:
        print("BL中建筑物占比大于或等于30%，不进行转移。")




#################################################################################
import random
from PIL import Image
import numpy as np
import os

def copy_buildings(imgA_path, imgB_path, LA_path):
    """
    将imgA中所有建筑物区域复制到imgB中
    :param imgA_path: str，imgA的路径
    :param imgB_path: str，imgB的路径
    :param LA_path: str，LA的路径
    :return: None
    """
    # 读取两张RGB影像和标签图
    imgA = Image.open(imgA_path)
    imgB = Image.open(imgB_path)
    LA = np.array(Image.open(LA_path))  # 将标签图转换为NumPy数组

    # 获取图像的大小
    imgA_width, imgA_height = imgA.size
    imgB_width, imgB_height = imgB.size

    # 遍历LA中的每个像素，找到属于建筑物的区域，并将这些区域从imgA复制到imgB
    for y in range(imgA_height):
        for x in range(imgA_width):
            if LA[y][x] == 1:  # 假设值为1表示建筑物区域，根据实际情况修改
                # 计算在imgB中的对应位置
                start_x = x % imgB_width  # 使用模运算确保x值在imgB的范围内
                start_y = y % imgB_height  # 使用模运算确保y值在imgB_height范围内
                # 将imgA中对应像素的值复制到imgB的相应位置
                imgB.putpixel((start_x, start_y), imgA.getpixel((x, y)))

                # 保存结果图像到指定路径（可选）
    result_path = "result.png"  # 可根据需要修改结果图像的保存路径和文件名
    imgB.save(result_path)


def batch_copy_buildings(imgA_paths, imgB_paths, LA_path, LB_path):
    """
    批量处理多个建筑物复制任务
    :param imgA_paths: list[str]，imgA的路径列表
    :param imgB_paths: list[str]，imgB的路径列表，长度应与imgA_paths相同
    :param LA_path: str，LA的路径
    :return: None
    """
    LA = np.array(Image.open(LA_path))  # 将标签图转换为NumPy数组
    LB = np.array(Image.open(LB_path))
    imgA = Image.open(imgA_paths)
    imgB = Image.open(imgB_paths)
    # 获取图像的大小
    imgA_width, imgA_height = imgA.size
    imgB_width, imgB_height = imgB.size
    # 计算建筑物标签占比
    building_ratio = (LB == 255).sum() / (imgA_width * imgA_height)
    if building_ratio < 0.3:  # 如果占比低于30%
    # 遍历LA中的每个像素，找到属于建筑物的区域，并将这些区域从imgA复制到imgB
            for y in range(imgA_height):
                for x in range(imgA_width):
                    if LA[y][x] == 255:  # 假设值为1表示建筑物区域，根据实际情况修改
                        # 计算在imgB中的对应位置
                        start_x = x % imgB_width
                        start_y = y % imgB_height
                        # 将imgA中对应像素的值复制到imgB的相应位置（随机选择部分建筑物进行复制）
                        if random.random() < 0.5:  # 随机选择50%的建筑物进行复制
                            imgB.putpixel((start_x, start_y), imgA.getpixel((x, y)))
    result_path = ('D:/SCI_Write/building_extract/FanBu_EXP/Trans_results/' + imgA_paths[-15:])
    imgB.save(result_path)

def get_image_paths(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.tif') or file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths

def combine_images(imgA, imgB, LA, LB):
    LA = (LA == 1).astype(np.uint8)
    # 使用形态学操作去除小的噪声
    LA = cv2.dilate(LA, np.ones((3, 3)))
    LA = cv2.erode(LA, np.ones((3, 3)))
    # 找到连通区域
    labels, _ = cv2.connectedComponents(LA)
    # 提取建筑物区域
    buildings = []
    for label in range(1, labels):
        if random.random() < 0.3:
        # 获取连通区域的坐标
            y_coords, x_coords = np.where(_ == label)
            # 提取连通区域作为建筑物区域
            imgB[x_coords, y_coords] = imgA[y_coords, x_coords]
            LB[x_coords, y_coords] = LA[y_coords, x_coords]
        else:
            print('未达到随机阈值，不进行转移！')
    return imgB, LB

# 使用函数获取图像路径
imageA_dir = "/imgA"
imageB_dir = "/imgB"
labelA_dir = imageA_dir.replace('imgA', 'LA')
labelB_dir = imageB_dir.replace('imgB', 'LB')
imageA_paths = get_image_paths(imageA_dir)
imageB_paths = get_image_paths(imageB_dir)
labelA_paths = get_image_paths(labelA_dir)
labelB_paths = get_image_paths(labelB_dir)
for epoch in range(20):
    # a = random.randint(0, len(imageA_paths))
    # b = random.randint(0, len(imageB_paths))
    # imgA = cv2.imread(imageA_paths[a-1]) / 255.0  # 归一化到[0, 1]范围
    # imgB = cv2.imread(imageB_paths[b-1]) / 255.0  # 归一化到[0, 1]范围
    # LA = cv2.imread(labelA_paths[a1], cv2.IMREAD_GRAYSCALE) / 255  # 归一化到[0, 1]范围，假设是灰度图
    # LB = cv2.imread(labelB_paths[b1], cv2.IMREAD_GRAYSCALE) / 255  # 同上，假设是灰度图
    imgA = cv2.imread(imageA_paths[0]) / 255.0  # 归一化到[0, 1]范围
    imgB = cv2.imread(imageB_paths[0]) / 255.0  # 归一化到[0, 1]范围
    LA = cv2.imread(labelA_paths[0], cv2.IMREAD_GRAYSCALE) / 255  # 归一化到[0, 1]范围，假设是灰度图
    LB = cv2.imread(labelB_paths[0], cv2.IMREAD_GRAYSCALE) / 255  # 同上，假设是灰度图
    imgB_width, imgB_height = LB.shape
    building_ratio = (LB == 1).sum() / (imgB_width * imgB_height)
    if building_ratio < 0.1:  # 如果占比低于30%
        imgB, LB = combine_images(imgA, imgB, LA, LB)
        imgB = (imgB * 255).astype(np.uint8)  # 将结果从[0, 1]范围转换回[0, 255]范围并转为uint8类型
        LB = (LB * 255).astype(np.uint8)
        imgB_path = ('')
        labB_path = ('')
        cv2.imwrite(imgB_path, imgB)  # 保存结果图像
        cv2.imwrite(labB_path, LB)
        print('转移成功！')
    else:
        print('BL中建筑物占比大于或等于10%，不进行转移!')

