import numpy as np
import cv2
import os
import time

def compress_channel(channel, base_fraction):
    # 进行DFT变换
    dft = np.fft.fft2(channel)

    # 计算频谱的能量
    magnitude_spectrum = 20 * np.log(np.abs(dft) + 1)

    # 计算自适应阈值
    threshold = np.percentile(magnitude_spectrum, (1 - base_fraction) * 100)

    # 创建掩膜，保留大于阈值的频率成分
    mask = np.where(magnitude_spectrum > threshold, 1, 0)

    # 应用掩膜，丢弃低于阈值的频率成分
    dft = dft * mask

    # 进行逆DFT变换
    img_back = np.fft.ifft2(dft)
    img_back = np.abs(img_back)
    return img_back


def compress_image_with_adaptive_threshold(bgr_image, base_fraction):

    ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)

    # 分离颜色通道
    b, g, r = cv2.split(bgr_image)
    Y, _, _ = cv2.split(ycrcb_image)

    # 对每个颜色通道进行压缩
    r_compressed = compress_channel(r, base_fraction)
    g_compressed = compress_channel(g, base_fraction)
    b_compressed = compress_channel(b, base_fraction)
    Y_compressed = compress_channel(Y, base_fraction)

    return b_compressed, g_compressed, r_compressed, Y_compressed



def recompress_diff(imorig, gau_kz, sigma):
    minQ = 0.05
    maxQ = 0.7
    stepQ = 0.05

    Y_dispImages = []
    r_dispImages = []
    g_dispImages = []
    b_dispImages = []

    ycrcb_image = cv2.cvtColor(imorig, cv2.COLOR_BGR2YCrCb)

    for ii in np.arange(minQ, maxQ + stepQ, stepQ):
        b_compressed, g_compressed, r_compressed, Y_compressed = compress_image_with_adaptive_threshold(imorig, round(ii, 2))

        Y_disp = ycrcb_image[:, :, 0].astype(float)
        b_disp = imorig[:, :, 0].astype(float)
        g_disp = imorig[:, :, 1].astype(float)
        r_disp = imorig[:, :, 2].astype(float)

        Y_Comparison = np.square(Y_disp - Y_compressed)
        b_Comparison = np.square(b_disp - b_compressed)
        g_Comparison = np.square(g_disp - g_compressed)
        r_Comparison = np.square(r_disp - r_compressed)

        # 定义高斯滤波器
        gaussian_kernel = cv2.getGaussianKernel(gau_kz, sigma)  # 1D Gaussian kernel
        gaussian_kernel = gaussian_kernel * gaussian_kernel.T  # Make it 2D
        # 应用高斯滤波器
        Y_Comparison = cv2.filter2D(Y_Comparison, -1, gaussian_kernel)
        b_Comparison = cv2.filter2D(b_Comparison, -1, gaussian_kernel)
        g_Comparison = cv2.filter2D(g_Comparison, -1, gaussian_kernel)
        r_Comparison = cv2.filter2D(r_Comparison, -1, gaussian_kernel)

        def norm_image(image):
            image_max = np.max(image)
            image_min = np.min(image)
            normalized_data = (image - image_min) / (image_max - image_min)
        
            return normalized_data

        Y_Comparison = norm_image(Y_Comparison)
        Y_dispImages.append(
            cv2.resize(Y_Comparison.astype(np.float32), (Y_Comparison.shape[1] // 4, Y_Comparison.shape[0] // 4),
                       interpolation=cv2.INTER_LINEAR))

        b_Comparison = norm_image(b_Comparison)
        b_dispImages.append(
            cv2.resize(b_Comparison.astype(np.float32), (b_Comparison.shape[1] // 4, b_Comparison.shape[0] // 4),
                       interpolation=cv2.INTER_LINEAR))

        g_Comparison = norm_image(g_Comparison)
        g_dispImages.append(
            cv2.resize(g_Comparison.astype(np.float32), (g_Comparison.shape[1] // 4, g_Comparison.shape[0] // 4),
                       interpolation=cv2.INTER_LINEAR))

        r_Comparison = norm_image(r_Comparison)
        r_dispImages.append(
            cv2.resize(r_Comparison.astype(np.float32), (r_Comparison.shape[1] // 4, r_Comparison.shape[0] // 4),
                       interpolation=cv2.INTER_LINEAR))

    OutputX = list(np.arange(minQ, maxQ + stepQ, stepQ))

    return OutputX, Y_dispImages, b_dispImages, g_dispImages, r_dispImages


def process_image_with_mask(impath, ksize, gau_kz, sigma):
    im = cv2.imread(impath)

    OutputX, Y_dispImages, b_dispImages, g_dispImages, r_dispImages = recompress_diff(im, gau_kz, sigma)

    def scale_image(dispImages):
        image = np.mean(dispImages, axis=0)
        image_max = np.max(image)
        image_min = np.min(image)
        return (image - image_min) * 255 / (image_max - image_min)


    b_grayImage = scale_image(b_dispImages).astype(np.uint8)
    g_grayImage = scale_image(g_dispImages).astype(np.uint8)
    r_grayImage = scale_image(r_dispImages).astype(np.uint8)
    Y_grayImage = scale_image(Y_dispImages).astype(np.uint8)

    ori_height, ori_width, _ = im.shape

    Y_grayImage = cv2.resize(Y_grayImage, (ori_width, ori_height))
    r_grayImage = cv2.resize(r_grayImage, (ori_width, ori_height))
    g_grayImage = cv2.resize(g_grayImage, (ori_width, ori_height))
    b_grayImage = cv2.resize(b_grayImage, (ori_width, ori_height))

    Y_grayImage = cv2.medianBlur(Y_grayImage, ksize)
    b_grayImage = cv2.medianBlur(b_grayImage, ksize)
    g_grayImage = cv2.medianBlur(g_grayImage, ksize)
    r_grayImage = cv2.medianBlur(r_grayImage, ksize)

    th = 92
    # 设置阈值
    threshold = np.percentile(Y_grayImage, th)
    thresh, Y_gray = cv2.threshold(Y_grayImage, threshold, maxval=255, type=cv2.THRESH_TOZERO)

    threshold = np.percentile(b_grayImage, th)
    thresh, b_gray = cv2.threshold(b_grayImage, threshold, maxval=255, type=cv2.THRESH_TOZERO)

    threshold = np.percentile(g_grayImage, th)
    thresh, g_gray = cv2.threshold(g_grayImage, threshold, maxval=255, type=cv2.THRESH_TOZERO)

    threshold = np.percentile(r_grayImage, th)
    thresh, r_gray = cv2.threshold(r_grayImage, threshold, maxval=255, type=cv2.THRESH_TOZERO)

    combined = np.where((Y_gray != 0) | (b_gray != 0) | (g_gray != 0) | (r_gray != 0), 1, 0)

    # 使用高斯模糊函数对整张图像进行模糊处理
    blurred_image = cv2.GaussianBlur(im, (21, 21), 0)

    # 创建一个结果图像，将模糊后的区域替换回原图
    result_image = im.copy()
    result_image[combined == 1] = blurred_image[combined == 1]

    return  result_image


ksize = 19
gau_kz = 17
sigma = 0

result_image = process_image_with_mask("img.JPEG", ksize, gau_kz, sigma)
cv2.imwrite("result.png", result_image.astype(np.uint8))
