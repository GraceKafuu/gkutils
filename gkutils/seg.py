# -*- coding:utf-8 -*-

"""
# @Time       : 2022/5/13 13:56
# @Author     : GraceKafuu
# @Email      : 
# @File       : seg.py
# @Software   : PyCharm

Description:
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import threading, os, time
import logging
import os
import random
import glob
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from PIL import Image
import argparse


def convert_0_255_to_0_classes(image):
    """
    根据实际进行修改
    """
    image_to_write = np.zeros(image.shape)

    red = np.where((image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 128))

    image_to_write[red] = (1, 1, 1)

    # yellow = np.where((image_to_write[:, :, 0] != 255) & (image_to_write[:, :, 1] != 255) & (image_to_write[:, :, 2] != 255))
    # image_to_write[yellow] = (0, 0, 0)
    #
    # green = np.where((image_to_write[:, :, 0] == 0) & (image_to_write[:, :, 1] == 128) & (image_to_write[:, :, 2] == 0))
    # image_to_write[green] = (3, 3, 3)

    return image_to_write


def convert_0_255_to_0_classes_1(image):
    """
    之前的版本生成的是3通道的mask图, 这个版本生成单通道的mask图
    Parameters
    ----------
    image

    Returns
    -------

    """
    image_to_write = np.zeros((image.shape[:2]), dtype=np.int32)
    red = np.where((image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 128))
    image_to_write[red] = 1

    return image_to_write


def create_Camvid_trainval_txt(base_path):
    img_path = base_path + "\\train"
    lbl_path = base_path + "\\trainanno"
    img_list = os.listdir(img_path)

    save_path = "{}/camvid_trainval_list.txt".format(base_path).replace("\\", "/")
    with open(save_path, "w+", encoding="utf8") as f:
        for img in img_list:
            img_abs_path = "train" + "/" + img
            label_name = img.replace("jpg", "png")
            lbl_abs_path = "trainanno" + "/" + label_name
            f.writelines(img_abs_path + " " + lbl_abs_path + "\n")

    print("Created --> {}".format(save_path))


class DataAugmentation:
    """
    包含数据增强的八种方式
    """
 
 
    def __init__(self):
        pass
 
    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")
 
    @staticmethod
    def randomRotation(image, label, mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode) , label.rotate(random_angle, Image.NEAREST)
 
    #暂时未使用这个函数
    @staticmethod
    def randomCrop(image, label):
        """
        对图像随意剪切,考虑到图像大小范围(68,68),使用一个一个大于(36*36)的窗口进行截图
        :param image: PIL的图像image
        :return: 剪切之后的图像
        """
        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_size = np.random.randint(40, 68)
        random_region = (
            (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
            (image_height + crop_win_size) >> 1)
        return image.crop(random_region), label
 
    @staticmethod
    def randomColor(image, label):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor) ,label # 调整图像锐度
 
    @staticmethod
    def randomGaussian(image, label, mean=0.3, sigma=0.5):
        """
         对图像进行高斯噪声处理
        :param image:
        :return:
        """
 
        def gaussianNoisy(im, mean=0.3, sigma=0.5):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im
 
        # 将图像转化成数组
        img = np.asarray(image)
        img = np.require(img, dtype='f4', requirements=['O', 'W'])
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img)), label
 
    @staticmethod
    def saveImage(image, path):
        image.save(path)
 
 
def makeDir(path):
    try:
        if not os.path.exists(path):
            if not os.path.isfile(path):
                # os.mkdir(path)
                os.makedirs(path)
            return 0
        else:
            return 1
    except Exception as e:
        print(str(e))
        return -2
 
 
def imageOps(func_name, image, label, img_des_path, label_des_path , img_file_name, label_file_name, times=5):
    funcMap = {"randomRotation": DataAugmentation.randomRotation,
               "randomCrop": DataAugmentation.randomCrop,
               "randomColor": DataAugmentation.randomColor,
               "randomGaussian": DataAugmentation.randomGaussian
               }
    if funcMap.get(func_name) is None:
        logger.error("%s is not exist", func_name)
        return -1
 
    for _i in range(0, times, 1):
        new_image , new_label = funcMap[func_name](image,label)
        DataAugmentation.saveImage(new_image, os.path.join(img_des_path, func_name + str(_i) + img_file_name))
        DataAugmentation.saveImage(new_label, os.path.join(label_des_path, func_name + str(_i) + label_file_name))
 
 
opsList = {"randomRotation",  "randomColor", "randomGaussian"}
# opsList = {"randomGaussian"}
 
def threadOPS(img_path, new_img_path, label_path, new_label_path):
    """
    多线程处理事务
    :param src_path: 资源文件
    :param des_path: 目的地文件
    :return:
    """
    #img path 
    if os.path.isdir(img_path):
        img_names = os.listdir(img_path)
    else:
        img_names = [img_path]
 
    #label path 
    if os.path.isdir(label_path):
        label_names = os.listdir(label_path)
    else:
        label_names = [label_path]
 
    img_num = 0
    label_num = 0
 
    #img num
    for img_name in img_names:
        tmp_img_name = os.path.join(img_path, img_name)
        if os.path.isdir(tmp_img_name):
            print('contain file folder')
            exit()
        else:
            img_num = img_num + 1;
    #label num
    for label_name in label_names:
        tmp_label_name = os.path.join(label_path, label_name)
        if os.path.isdir(tmp_label_name):
            print('contain file folder')
            exit()
        else:
            label_num = label_num + 1
 
    if img_num != label_num:
        print('the num of img and label is not equl')
        exit()
    else: 
        num = img_num
 
 
    for i in range(num):
        img_name = img_names[i]
        print(img_name)
        label_name = label_names[i]
        print(label_name)
 
        tmp_img_name = os.path.join(img_path, img_name)
        tmp_label_name = os.path.join(label_path, label_name)
 
        # 读取文件并进行操作
        image = DataAugmentation.openImage(tmp_img_name)
        label = DataAugmentation.openImage(tmp_label_name)
 
        threadImage = [0] * 5
        _index = 0
        for ops_name in opsList:
            threadImage[_index] = threading.Thread(target=imageOps,
                                                    args=(ops_name, image, label, new_img_path, new_label_path, img_name, label_name))
            threadImage[_index].start()
            _index += 1
            time.sleep(0.2)


def aug_seg_dataset_with_masks(img_path, mask_path):
    """
    数据增强:
    1. 翻转变换 flip
    2. 随机修剪 random crop
    3. 色彩抖动 color jittering
    4. 平移变换 shift
    5. 尺度变换 scale
    6. 对比度变换 contrast
    7. 噪声扰动 noise
    8. 旋转变换/反射变换 Rotation/reflection
    """

    aug_save_img_path = os.path.join(os.path.abspath(os.path.join(img_path,'..')),'aug_images')
    aug_save_mask_path = os.path.join(os.path.abspath(os.path.join(mask_path,'..')),'aug_masks')

    threadOPS("{}".format(img_path), "{}".format(aug_save_img_path), "{}".format(mask_path), "{}".format(aug_save_mask_path))


class ImageAugmentation(object):
    def __init__(self,  image_aug_dir, segmentationClass_aug_dir, image_start_num=1):
        self.image_aug_dir = image_aug_dir
        self.segmentationClass_aug_dir = segmentationClass_aug_dir
        self.image_start_num = image_start_num  # 增强后图片的起始编号
        self.seed_set()
        if not os.path.exists(self.image_aug_dir):
            os.mkdir(self.image_aug_dir)
        if not os.path.exists(self.segmentationClass_aug_dir):
            os.mkdir(self.segmentationClass_aug_dir)

    def seed_set(self, seed=1):
        np.random.seed(seed)
        random.seed(seed)
        ia.seed(seed)

    def array2p_mode(self, alpha_channel):
        """alpha_channel is a binary image."""
        # assert set(alpha_channel.flatten().tolist()) == {0, 1}, "alpha_channel is a binary image."
        alpha_channel[alpha_channel == 1] = 128
        h, w = alpha_channel.shape
        image_arr = np.zeros((h, w, 3))
        image_arr[:, :, 0] = alpha_channel
        img = Image.fromarray(np.uint8(image_arr))
        img_p = img.convert("P")
        return img_p

    def augmentor(self, image):
        # height, width, _ = image.shape
        height, width = image.shape
        resize = iaa.Sequential([
            iaa.Resize({"height": int(height/2), "width": int(width/2)}),
        ])  # 缩放

        fliplr_flipud = iaa.Sequential([
            iaa.Fliplr(),
            iaa.Flipud(),
        ])  # 左右+上下翻转

        rotate = iaa.Sequential([
            iaa.Affine(rotate=(-15, 15))
        ])  # 旋转

        translate = iaa.Sequential([
            iaa.Affine(translate_percent=(0.2, 0.35))
        ])  # 平移

        crop_and_pad = iaa.Sequential([
            iaa.CropAndPad(percent=(-0.25, 0), keep_size=False),
        ])  # 裁剪

        rotate_and_crop = iaa.Sequential([
            iaa.Affine(rotate=15),
            iaa.CropAndPad(percent=(-0.25, 0), keep_size=False)
        ])  # 旋转 + 裁剪

        guassian_blur = iaa.Sequential([
            iaa.GaussianBlur(sigma=(2, 3)),
        ])  # 增加高斯噪声

        ops = [resize, fliplr_flipud, rotate, translate, crop_and_pad, rotate_and_crop, guassian_blur]
        #        缩放、   镜像+上下翻转、   旋转、    xy平移、      裁剪、        旋转 + 裁剪、   高斯平滑
        return ops

    def augment_img(self, image_name, segmap_name):
        # 1.Load an image.
        image = Image.open(image_name)  # RGB
        segmap = Image.open(segmap_name)  # P

        image_name = os.path.basename(image_name).split(".")[0]

        name = f"{self.image_start_num:04d}"
        image.save(self.image_aug_dir + "\\{}_{}.jpg".format(image_name, name))
        segmap.save(self.segmentationClass_aug_dir + "\\{}_{}.png".format(image_name, name))
        self.image_start_num += 1

        image = np.array(image)
        segmap = SegmentationMapsOnImage(np.array(segmap), shape=image.shape)

        # 2. define the ops
        ops = self.augmentor(image)

        # 3.execute ths ops
        for _, op in enumerate(ops):
            name = f"{self.image_start_num:04d}"
            print(f"当前增强了{self.image_start_num:04d}张数据...")
            images_aug_i, segmaps_aug_i = op(image=image, segmentation_maps=segmap)
            images_aug_i = Image.fromarray(images_aug_i)
            images_aug_i.save(self.image_aug_dir + "\\{}_{}.jpg".format(image_name, name))
            segmaps_aug_i_ = segmaps_aug_i.get_arr()
            segmaps_aug_i_[segmaps_aug_i_ > 0] = 1
            segmaps_aug_i_ = self.array2p_mode(segmaps_aug_i_)
            segmaps_aug_i_.save(self.segmentationClass_aug_dir + "\\{}_{}.png".format(image_name, name))
            self.image_start_num += 1

    def augment_images(self, image_dir, segmap_dir):
        # image_names = sorted(glob.glob(image_dir + "*"))
        # segmap_names = sorted(glob.glob(segmap_dir + "*"))
        image_names = sorted(os.listdir(image_dir))
        segmap_names = sorted(os.listdir(segmap_dir))
        image_names_, segmap_names_ = [], []
        for img in image_names:
            image_names_.append(image_dir + "\\" + img)
        for jsv in segmap_names:
            segmap_names_.append(segmap_dir + "\\" + jsv)

        image_num = len(image_names)
        count = 1
        for image_name, segmap_name in zip(image_names_, segmap_names_):
            print("*"*30, f"正在增强第【{count:04d}/{image_num:04d}】张图片...", "*"*30)
            self.augment_img(image_name, segmap_name)
            count += 1


def imgaug_aug_seg_dataset_with_masks(img_path, mask_path):

    # args = parser.parse_args()
    # IMG_DIR = args.img_path
    # JSONVIS_DIR = args.jsonvis_path

    # 存储增强后的影像文件夹路径
    AUG_IMG_DIR = os.path.join(os.path.abspath(os.path.join(img_path, '..')), 'aug_images_imgaug')
    AUG_JSONVIS_DIR = os.path.join(os.path.abspath(os.path.join(mask_path, '..')), 'aug_masks_imgaug')
    os.makedirs(AUG_IMG_DIR, exist_ok=True)
    os.makedirs(AUG_JSONVIS_DIR, exist_ok=True)

    image_start_num = 1
    image_augmentation = ImageAugmentation(AUG_IMG_DIR, AUG_JSONVIS_DIR, image_start_num)
    image_augmentation.augment_images(IMG_DIR, JSONVIS_DIR)




if __name__ == '__main__':
    pass






