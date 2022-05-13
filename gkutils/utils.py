# -*- coding:utf-8 -*-

"""
# @Time       : 2022/5/13 13:56
# @Author     : GraceKafuu
# @Email      : 
# @File       : utils.py
# @Software   : PyCharm

Description:
"""

import os
import cv2
import numpy as np
import shutil
import json
from xml.dom import minidom
import xml.etree.ElementTree as ET
import pypyodbc
import scipy.misc
import scipy
import socket
import time
import logging
import logging.config
import re
from logging.handlers import TimedRotatingFileHandler
from logging.handlers import RotatingFileHandler



def LogInit(prex):
    """
    日志按日输出 单进程适用
    """
    log_fmt = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    formatter = logging.Formatter(log_fmt)
    # 创建TimedRotatingFileHandler对象
    # dirname, filename = os.path.split(os.path.abspath(__file__))
    dirname = os.getcwd()
    # logpath = os.path.dirname(os.getcwd()) + '/Logs/'
    logpath = dirname + '/Logs/'
    if not os.path.exists(logpath):
        os.mkdir(logpath)
    log_file_handler = TimedRotatingFileHandler(filename=logpath + prex+'-log.', when="D",
                                                interval=1)
    log_file_handler.suffix = prex + "-%Y-%m-%d_%H-%M-%S.log"
    log_file_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(prex)

    log.addHandler(log_file_handler)

    return log


def apply_CLAHE(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(3, (4, 4))
    dst = clahe.apply(gray)

    return dst


def bmp2jpg(bmp_abs_path):
    base_name = os.path.basename(bmp_abs_path)
    save_name = base_name.replace(".bmp", ".jpg")
    cv2_img = cv2.imread(bmp_abs_path)
    
    return cv2_img, save_name


def dcm2jpg(dcm_path):
    ds = pydicom.read_file(dcm_path)  # 读取.dcm文件
    img = ds.pixel_array  # 提取图像信息
    # scipy.misc.imsave(out_path, img)
    return img


def calc_brightness(img):
    # 把图片转换为单通道的灰度图
    img = cv2.resize(img, (16, 16))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取形状以及长宽
    img_shape = gray_img.shape
    height, width = img_shape[0], img_shape[1]
    size = gray_img.size
    # 灰度图的直方图
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    # 计算灰度图像素点偏离均值(128)程序
    a = 0
    ma = 0
    reduce_matrix = np.full((height, width), 128)
    shift_value = gray_img - reduce_matrix
    shift_sum = sum(map(sum, shift_value))
    da = shift_sum / size
    # 计算偏离128的平均偏差
    for i in range(256):
        ma += (abs(i - 128 - da) * hist[i])
    m = abs(ma / size)

    # 亮度系数
    if m == 0:
        print("ZeroDivisionError!")
        return 100, -100
    else:
        k = abs(da) / m
        return k[0], da


def change_brightness(img, value=70):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def change_brightness_opencv_official(image, alpha=1.0, beta=0):
    """
    https://docs.opencv.org/4.5.3/d3/dc1/tutorial_basic_linear_transform.html
    Parameters
    ----------
    img
    alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
    beta = int(input('* Enter the beta value [0-100]: '))
    Returns
    -------

    """
    new_image = np.zeros(image.shape, image.dtype)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)

    return new_image


def gamma_correction(img, gamma):
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(img, lookUpTable)

    return res


def add_gaussian_noise(img):
    gauss = np.random.normal(img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
    img_gauss = cv2.add(img, gauss)

    return img_gauss


def noisy(image, noise_typ):
    """
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.

    """
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.2
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy

    elif noise_typ == "gauss_v2":
        mean = 0
        var = 0.2
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, image.size)
        gauss = gauss.reshape(image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
        img_gauss = cv2.add(image, gauss)
        return img_gauss

    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_typ =="speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy

    elif noise_typ =="speckle_v2":
        mean = 0
        var = 0.2
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, image.size)
        gauss = gauss.reshape(image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
        noise = image + image * gauss

        return noise


def cal_mean_std(imageDir):

    img_h, img_w = 64, 64  # 根据自己数据集适当调整，影响不大
    means, stdevs = [], []
    img_list = []

    if os.path.exists(imageDir + "\\Thumbs.db"):
        os.remove(imageDir + "\\Thumbs.db")
    imgs_path_list = os.listdir(imageDir)

    len_ = len(imgs_path_list)
    i = 0
    for item in imgs_path_list:
        img = cv2.imread(os.path.join(imageDir, item))
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
        print(i, '/', len_)

    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
    means = means.reverse()
    stdevs = stdevs.reverse()

    return means, stdevs


def write_one(doc, root, label, value):
    root.appendChild(doc.createElement(label)).appendChild(doc.createTextNode(value))


def create_xml(xml_name, date, lineName, direction, startStation, endStation, startTime, endTime, startKm, endKm, startPoleNo, endPoleNo, panoramisPixel, partPixel):
    doc = minidom.Document()
    root = doc.createElement("detect")
    doc.appendChild(root)
    baseinfolist = doc.createElement("baseInfo")
    root.appendChild(baseinfolist)
    write_one(doc, baseinfolist, "date", date)
    write_one(doc, baseinfolist, "lineName", lineName)
    write_one(doc, baseinfolist, "direction", direction)
    write_one(doc, baseinfolist, "startStation", startStation)
    write_one(doc, baseinfolist, "endStation", endStation)

    appendinfolist = doc.createElement("appendInfo")
    root.appendChild(appendinfolist)
    write_one(doc, appendinfolist, "startTime", startTime)
    write_one(doc, appendinfolist, "endTime", endTime)
    write_one(doc, appendinfolist, "startKm", startKm)
    write_one(doc, appendinfolist, "endKm", endKm)
    write_one(doc, appendinfolist, "startPoleNo", startPoleNo)
    write_one(doc, appendinfolist, "endPoleNo", endPoleNo)
    write_one(doc, appendinfolist, "panoramisPixel", panoramisPixel)
    write_one(doc, appendinfolist, "partPixel", partPixel)

    with open(os.path.join('{}').format(xml_name), 'w', encoding='UTF-8') as fh:
        doc.writexml(fh, indent='', addindent='\t', newl='\n', encoding='UTF-8')


def create_mdb_if_not_exists(ACCESS_DATABASE_FILE):
    ODBC_CONN_STR = 'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=%s;' % ACCESS_DATABASE_FILE
    if not os.path.exists(ACCESS_DATABASE_FILE):
        mdb_file = pypyodbc.win_create_mdb(ACCESS_DATABASE_FILE)

        # ODBC_CONN_STR = 'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=%s;' % ACCESS_DATABASE_FILE
        conn = pypyodbc.connect(ODBC_CONN_STR)
        cur = conn.cursor()

        SQL = """CREATE TABLE PICINDEX (id COUNTER PRIMARY KEY, SETLOC VARCHAR(255) NOT NULL, KM NUMBER NOT NULL, ST VARCHAR(255), PANORAMIS_START_FRAME NUMBER NOT NULL,
                                                PANORAMIS_START_PATH VARCHAR(255) NOT NULL, PANORAMIS_END_FRAME NUMBER NOT NULL, PANORAMIS_END_PATH VARCHAR(255) NOT NULL,
                                                PART_START_FRAME NUMBER NOT NULL, PART_START_PATH VARCHAR(255) NOT NULL, PART_END_FRAME NUMBER NOT NULL, PART_END_PATH VARCHAR(255) NOT NULL);"""
        cur.execute(SQL)
        conn.commit()
        cur.close()
        conn.close()


def write_data_to_mdb(ACCESS_DATABASE_FILE, insert_data):
    ODBC_CONN_STR = 'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=%s;' % ACCESS_DATABASE_FILE

    conn = pypyodbc.connect(ODBC_CONN_STR)
    cur = conn.cursor()

    SQL_ = """insert into PICINDEX (id, SETLOC, KM, ST, PANORAMIS_START_FRAME, PANORAMIS_START_PATH, PANORAMIS_END_FRAME, PANORAMIS_END_PATH, PART_START_FRAME, 
                        PART_START_PATH, PART_END_FRAME, PART_END_PATH) values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

    cur.execute(SQL_, insert_data)
    conn.commit()
    cur.close()
    conn.close()


def change_xml_content(filename, content_orig, content_chg):
    xmlTree = ET.parse(filename)
    rootElement = xmlTree.getroot()
    for element in rootElement.findall("object"):
        if element.find('name').text == content_orig:
            element.find('name').text = content_chg
    xmlTree.write(filename, encoding='UTF-8', xml_declaration=True)


def change_yolo_label_content(txt_file):
    with open(txt_file, "r", encoding="utf-8") as f:
        data = f.readlines()

    new_label = None
    for line in data:
        label = line.split(" ")[0]
        if label == "15":
            new_label = "0" + line[2:]

    save_path = txt_file.replace("labels", "new_labels")
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    with open(save_path, "w", encoding="utf-8") as f_new:
        f_new.writelines(new_label)


def detect_shape(c):
    """
    approxPolyDP()函数是opencv中对指定的点集进行多边形逼近的函数
    :param c:
    :return: 返回形状和折点的坐标
    """
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    if len(approx) == 3:
        shape = "triangle"
        return shape, approx

    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        return shape, approx

    elif len(approx) == 5:
        shape = "pentagon"
        return shape, approx

    elif len(approx) == 6:
        shape = "hexagon"
        return shape, approx

    elif len(approx) == 8:
        shape = "octagon"
        return shape, approx

    elif len(approx) == 10:
        shape = "star"
        return shape, approx

    else:
        shape = "circle"
        return shape, approx


def cv2ImgAddText(img, text, left, top, font_path="simsun.ttc", textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype(font_path, textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontStyle)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def remove_truncated_image(imgPath):
    img_list = os.listdir(imgPath)
    for img in img_list:
        imgAbsPath = imgPath + "\\{}".format(img)
        try:
            cv2_img = cv2.imread(imgAbsPath)
        except Exception as Error:
            os.remove(imgAbsPath)
    print("Removed truncated images!")


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return (b, g, r)


def cal_SVD_var(img):
    img_r, img_g, img_b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    u_r, sigma_r, v_r = np.linalg.svd(img_r)
    u_g, sigma_g, v_g = np.linalg.svd(img_r)
    u_b, sigma_b, v_b = np.linalg.svd(img_r)
    # r
    len_sigma_r = len(sigma_r)
    len_sigma_r_50 = int(round(.5 * len_sigma_r))
    len_sigma_r_20 = int(round(.2 * len_sigma_r))
    var_r_50 = np.var(sigma_r[:len_sigma_r_50])
    var_r_last_20 = np.var(sigma_r[-len_sigma_r_20:])
    # g
    len_sigma_g = len(sigma_g)
    len_sigma_g_50 = int(round(.5 * len_sigma_g))
    len_sigma_g_20 = int(round(.2 * len_sigma_g))
    var_g_50 = np.var(sigma_r[:len_sigma_g_50])
    var_g_last_20 = np.var(sigma_r[-len_sigma_g_20:])
    # b
    len_sigma_b = len(sigma_b)
    len_sigma_b_50 = int(round(.5 * len_sigma_b))
    len_sigma_b_20 = int(round(.2 * len_sigma_b))
    var_b_50 = np.var(sigma_r[:len_sigma_b_50])
    var_b_last_20 = np.var(sigma_r[-len_sigma_b_20:])

    var_50 = np.mean([var_r_50, var_g_50, var_b_50])
    var_last_20 = np.mean([var_r_last_20, var_g_last_20, var_b_last_20])

    return var_50, var_last_20



def findSubStrIndex(substr, str, time):
    """
    # 找字符串substr在str中第time次出现的位置
    """
    times = str.count(substr)
    if (times == 0) or (times < time):
        pass
    else:
        i = 0
        index = -1
        while i < time:
            index = str.find(substr, index+1)
            i += 1
        return index


def change_console_str_color():
    """
    @Time: 2021/1/22 21:16
    @Author: gracekafuu
    https://blog.csdn.net/qq_34857250/article/details/79673698

    """

    print('This is a \033[1;35m test \033[0m!')
    print('This is a \033[1;32;43m test \033[0m!')
    print('\033[1;33;44mThis is a test !\033[0m')


def find_specific_color(img):
    """
    https://stackoverflow.com/questions/42592234/python-opencv-morphologyex-remove-specific-color
    Parameters
    ----------
    img

    Returns
    -------

    """
    # lower = np.array([10, 10, 120])  # -- Lower range --
    # upper = np.array([60, 60, 245])  # -- Upper range --
    lower = np.array([0, 0, 100])  # -- Lower range --
    upper = np.array([80, 80, 255])  # -- Upper range --
    mask = cv2.inRange(img, lower, upper)
    # mask = 255 - mask
    res = cv2.bitwise_and(img, img, mask=mask)  # -- Contains pixels having the gray color--

    return res


def remove_specific_color(image):
    H, W, _ = image.shape
    newimg = image.copy()

    for i in range(H):
        for j in range(W):
            if image[:, :, 0][i, j] < 200 and image[:, :, 1][i, j] < 200 and image[:, :, 2][i, j] > 200:
            # if abs(image[:, :, 0][i, j] - image[:, :, 1][i, j]) < 20 and abs(image[:, :, 2][i, j] - image[:, :, 0][i, j]) > 50:
                newimg[:, :, 0][i, j] = 0
                newimg[:, :, 1][i, j] = 0
                newimg[:, :, 2][i, j] = 0

    return newimg


def remove_specific_color_v2(cv2img):
    hsv = cv2.cvtColor(cv2img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    thresh1 = cv2.threshold(s, 92, 255, cv2.THRESH_BINARY)[1]
    thresh2 = cv2.threshold(v, 10, 255, cv2.THRESH_BINARY)[1]
    thresh2 = 255 - thresh2
    mask = cv2.add(thresh1, thresh2)

    H, W, _ = cv2img.shape
    newimg = cv2img.copy()

    for i in range(H):
        for j in range(W):
            if mask[i, j] != 0:
                newimg[i, j] = cv2img[i - 12, j - 12]

    return newimg


def remove_Thumbs(img_path):
    thumbs = img_path + "/Thumbs.db"
    if os.path.exists(thumbs):
        os.remove(thumbs)
        print("Removed --> {}".format(thumbs))


def remove_list_repeat_elements(list1):
    list2 = []
    [list2.append(i) for i in list1 if i not in list2]

    return list2


def rename_files(imgPath, start_idx):
    imgList = sorted(os.listdir(imgPath))
    for i in range(len(imgList)):
        imgAbsPath = imgPath + "\\" + imgList[i]
        ends = os.path.splitext(imgList[i])[1]
        newName = "{:08d}{}".format(i + start_idx, ends)
        os.rename(imgAbsPath, imgPath + "\\" + newName)

    print("Renamed!")


def resize_images(imgPath, savePath, size_=(256, 32)):
    imgList = sorted(os.listdir(imgPath))
    for i in range(len(imgList)):
        imgAbsPath = imgPath + "\\" + imgList[i]
        img_name = os.path.basename(imgAbsPath).split(".")[0]
        ends = os.path.splitext(imgList[i])[1]
        cv2_img = cv2.imread(imgAbsPath)
        resized_img = cv2.resize(cv2_img, size_)
        cv2.imwrite(savePath + "\\{}{}".format(img_name, ends), resized_img)

    print("Resized!")


def retrieve_image_from_video(video_path, frame_save_path):
    video_name = os.path.basename(video_path).split(".")[0]

    if not os.path.exists(frame_save_path):
        os.makedirs(frame_save_path)

    cap = cv2.VideoCapture(video_path)
    count = 0
    while(True):
        success, frame = cap.read()
        if success:
            cv2.imwrite(frame_save_path + "/{}_{}.jpg".format(video_name, count), frame)
        elif not success:
            break
        count += 1
        print("Processing...")


def timestamp_to_strftime(curr_timestamp):
    strftime_ = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(curr_timestamp))
    return strftime_


def strftime_to_timestamp(curr_strftime):
    pass


def udp_send_txt_content(txtfile, client="127.0.0.1", port=60015):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    with open(txtfile) as f:
        msgs = f.readlines()

    while True:
        for msg in msgs:
            msg = msg.strip().replace("\\", "/")
            if not msg: break
            sock.sendto(bytes(msg, "utf-8"), (client, port))
            print("UDP sent: {}".format(msg))
            time.sleep(.0001)
        sock.close()


def perspective_transform(image, rect):
    """
    透视变换
    """
    tl, tr, br, bl = rect
    # tl, tr, br, bl = np.array([tl[0] - 20, tl[1] - 20]), np.array([tr[0] + 20, tr[1] - 20]), np.array([br[0] + 20, br[1] + 20]), np.array([bl[0] - 20, bl[1] + 20])
    # rect_new = np.array([tl[0] - 20, tl[1] - 20]), np.array([tr[0] + 20, tr[1] - 20]), np.array([br[0] + 20, br[1] + 20]), np.array([bl[0] - 20, bl[1] + 20])
    # 计算宽度
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # 计算高度
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # 定义变换后新图像的尺寸
    dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1],
                   [0, maxHeight-1]], dtype='float32')
    # 变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    # 透视变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def majority_element(arr):
    if arr == []:
        return None
    else:
        dict_ = {}
        for key in arr:
            dict_[key] = dict_.get(key, 0) + 1
        maybe_maj_element = max(dict_, key=lambda k: dict_[k])
        maybe_maj_key = [k for k, v in dict_.items() if v == dict_[maybe_maj_element]]

        if len(maybe_maj_key) == 1:
            maj_element = maybe_maj_element
            return maj_element
        else:
            return None


def second_majority_element(arr, remove_first_mj):
    for i in range(len(arr)):
        if remove_first_mj in arr:
            arr.remove(remove_first_mj)
    if arr != []:
        second_mj = majorityElement_v2(arr)
        return second_mj
    else:
        return None


def find_chinese(chars):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', chars)
    return chinese


def RANSAC_fit_2Dline(X_data, Y_data, iters=100000, sigma=0.25, pretotal=0, P=0.99):
    """

    Parameters
    ----------
    X
    Y
    # 使用RANSAC算法估算模型
    # 迭代最大次数，每次得到更好的估计会优化iters的数值
    iters = 100000
    # 数据和模型之间可接受的差值
    sigma = 0.25
    # 最好模型的参数估计和内点数目
    best_a = 0
    best_b = 0
    pretotal = 0
    # 希望的得到正确模型的概率
    P = 0.99
    Returns
    -------

    """

    SIZE = X_data.shape[0]

    best_a = 0
    best_b = 0

    for i in range(iters):
        # 随机在数据中红选出两个点去求解模型
        # sample_index = random.sample(range(SIZE), 2)
        sample_index = random.choices(range(SIZE), k=2)
        x_1 = X_data[sample_index[0]]
        x_2 = X_data[sample_index[1]]
        y_1 = Y_data[sample_index[0]]
        y_2 = Y_data[sample_index[1]]

        # y = ax + b 求解出a，b
        try:
            a = (y_2 - y_1) / ((x_2 - x_1) + 1e-2)
            b = y_1 - a * x_1
        except Exception as Error:
            print("RANSAC_fit_2Dline: a = (y_2 - y_1) / (x_2 - x_1) --> {}".format(Error))

        # 算出内点数目
        total_inlier = 0
        for index in range(SIZE):
            y_estimate = a * X_data[index] + b
            if abs(y_estimate - Y_data[index]) < sigma:
                total_inlier = total_inlier + 1

        # 判断当前的模型是否比之前估算的模型好
        if total_inlier > pretotal:
            # iters = math.log(1 - P) / math.log(1 - pow(total_inlier / (SIZE), 2))
            pretotal = total_inlier
            best_a = a
            best_b = b

        # 判断是否当前模型已经符合超过一半的点
        if total_inlier > SIZE // 2:
            break

    return best_a, best_b


def median_filter_1d(res_list, k=15):
    """
    中值滤波
    """
    edge = int(k / 2)
    new_res = res_list.copy()
    for i in range(len(res_list)):
        if i <= edge or i >= len(res_list) - edge - 1:
            pass
        else:
            medianv = np.median(res_list[i - edge:i + edge + 1])
            if new_res[i] != medianv:
                new_res[i] = medianv
            else:
                pass

    return new_res


def makeBorderH(image, H, W, W_need=128, H_need=32):
    """
    Horizental
    :param image:
    :param H:
    :param W:
    :param W_need:
    :param H_need:
    :return:
    """
    top_size, bottom_size, left_size, right_size = 0, 0, 0, 0
    if W < W_need:
        lr_pixel_need = W_need - W
        left_size = lr_pixel_need // 2
        right_size = lr_pixel_need - left_size
    if H < H_need:
        tb_pixel_need = H_need - H
        top_size = tb_pixel_need // 2
        bottom_size = tb_pixel_need - top_size

    if top_size != 0 or bottom_size != 0 or left_size != 0 or right_size != 0:
        replicate = cv2.copyMakeBorder(image, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
        replicateResized = cv2.resize(replicate, (W_need, H_need))
        return replicateResized
    else:
        resized = cv2.resize(image, (W_need, H_need))
        return resized


def makeBorderV(image, H, W, W_need=32, H_need=128):
    """
    Vertical
    :param image:
    :param H:
    :param W:
    :param W_need:
    :param H_need:
    :return:
    """
    top_size, bottom_size, left_size, right_size = 0, 0, 0, 0
    if W < W_need:
        lr_pixel_need = W_need - W
        left_size = lr_pixel_need // 2
        right_size = lr_pixel_need - left_size
    if H < H_need:
        tb_pixel_need = H_need - H
        top_size = tb_pixel_need // 2
        bottom_size = tb_pixel_need - top_size

    if top_size != 0 or bottom_size != 0 or left_size != 0 or right_size != 0:
        replicate = cv2.copyMakeBorder(image, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
        replicateResized = cv2.resize(replicate, (W_need, H_need))
        return replicateResized
    else:
        resized = cv2.resize(image, (W_need, H_need))
        return resized


def get_peak_points(heatmaps):
    """

    :param heatmaps: numpy array (N,4,256,256)
    :return:numpy array (N,4,2) #
    """
    N,C,H,W = heatmaps.shape   # N= batch size C=4 hotmaps
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy,xx = np.where(heatmaps[i, j] == heatmaps[i, j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x, y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # 限制最小的值
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    # 一个圆对应内切正方形的高斯分布

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius +
                               bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        # 将高斯分布覆盖到heatmap上，取最大，而不是叠加
    return heatmap
























if __name__ == '__main__':
    pass

