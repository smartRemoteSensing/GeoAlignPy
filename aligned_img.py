# -*- coding: utf-8 -*-
"""
Created on Tue May 13 10:24:00 2025

@author: lixiunan
"""
import rasterio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as RioResampling
from rasterio.transform import Affine
import os

def read_as_array(path):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile
        transform = src.transform
        crs = src.crs
    return data, profile, transform, crs

def save_array_to_tif(output_path, array, profile):
    profile.update(dtype=rasterio.float32, count=array.shape[0])
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(array.astype(np.float32))

def reproject_match(target_data, target_transform, target_crs, src_data, src_transform, src_crs, out_shape):
    dst = np.zeros(out_shape, dtype=np.float32)
    for i in range(src_data.shape[0]):
        reproject(
            source=src_data[i],
            destination=dst[i],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )
    return dst

def preprocess_gray(img):
    if img.shape[0] > 1:
        gray = np.mean(img, axis=0)
    else:
        gray = img[0]
    gray /= np.max(gray)
    return gray.astype(np.float32)

def ecc_register(ref_img, mov_img, mode=cv2.MOTION_AFFINE):
    warp_matrix = np.eye(2, 3, dtype=np.float32) if mode != cv2.MOTION_HOMOGRAPHY else np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)

    try:
        cc, warp_matrix = cv2.findTransformECC(ref_img, mov_img, warp_matrix, mode, criteria)
        print("ECC correlation coefficient:", cc)
    except cv2.error as e:
        print("ECC failed:", e)
        return None, None

    h, w = ref_img.shape
    if mode == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(mov_img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        aligned = cv2.warpAffine(mov_img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return aligned, warp_matrix

def plot_diff(ref, aligned):
    diff = np.abs(ref - aligned)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(ref, cmap='gray'); plt.title("Reference")
    plt.subplot(1, 3, 2); plt.imshow(aligned, cmap='gray'); plt.title("Aligned")
    plt.subplot(1, 3, 3); plt.imshow(diff, cmap='hot'); plt.title("Difference")
    plt.tight_layout(); plt.show()

def main(ref_path, mov_path, out_aligned_path):
    # 读取图像
    ref_data, ref_profile, ref_transform, ref_crs = read_as_array(ref_path)
    mov_data, mov_profile, mov_transform, mov_crs = read_as_array(mov_path)

    # 重采样到参考图像网格
    out_shape = ref_data.shape
    mov_data_reproj = reproject_match(ref_data, ref_transform, ref_crs,
                                      mov_data, mov_transform, mov_crs,
                                      out_shape)

    # 灰度图用于配准
    ref_gray = preprocess_gray(ref_data)
    mov_gray = preprocess_gray(mov_data_reproj)

    # ECC 精配准
    aligned_gray, warp_matrix = ecc_register(ref_gray, mov_gray)

    if aligned_gray is None:
        print("Registration failed.")
        return

    # 对原多光谱图像应用同样变换
    aligned_data = np.zeros_like(ref_data)
    for i in range(mov_data_reproj.shape[0]):
        aligned_data[i] = cv2.warpAffine(mov_data_reproj[i], warp_matrix, (ref_gray.shape[1], ref_gray.shape[0]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # 保存配准后图像
    save_array_to_tif(out_aligned_path, aligned_data, ref_profile)

    # 可视化
    plot_diff(ref_gray, aligned_gray)
    print("完成图像配准并保存到：", out_aligned_path)

if __name__ == "__main__":
    # 替换为你的文件路径
    ref_img_path = r"H:\1CD_dataset\001\GF2_PMS2_E114.8_N22.8_20231125_L1A13495975001\GF2_PMS2_E114.8_N22.8_20231125_L1A13495975001-MSS2.tiff"
    mov_img_path = r"H:\0ZJU_CD\jz\GF2_PMS2_E114.8_N22.8_20220403_L1A0006387395\GF2_PMS2_E114.8_N22.8_20220403_L1A0006387395-MSS2.tiff"
    output_path = "aligned_result.tif"
    main(ref_img_path, mov_img_path, output_path)
