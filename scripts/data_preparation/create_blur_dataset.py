"""
Script tạo dataset blur từ ảnh sharp.
Áp dụng các loại blur khác nhau phù hợp với từng cấp độ.

Cách dùng:
    python scripts/data_preparation/create_blur_dataset.py \
        --input datasets/face/train/sharp \
        --output datasets/face/train/blur \
        --task face
"""

import argparse
import cv2
import glob
import numpy as np
import os
import random
from os import path as osp


def apply_gaussian_blur(img, kernel_range=(7, 31)):
    """Gaussian blur với kernel size ngẫu nhiên."""
    k = random.choice(range(kernel_range[0], kernel_range[1] + 1, 2))
    sigma = random.uniform(1.0, 5.0)
    return cv2.GaussianBlur(img, (k, k), sigma)


def apply_motion_blur(img, kernel_range=(10, 30)):
    """Motion blur theo hướng ngẫu nhiên."""
    k = random.randint(kernel_range[0], kernel_range[1])
    angle = random.uniform(0, 360)

    # Tạo motion blur kernel
    kernel = np.zeros((k, k), dtype=np.float32)
    center = k // 2

    # Tính tọa độ đường thẳng
    cos_val = np.cos(np.deg2rad(angle))
    sin_val = np.sin(np.deg2rad(angle))

    for i in range(k):
        offset = i - center
        x = int(center + offset * cos_val)
        y = int(center + offset * sin_val)
        if 0 <= x < k and 0 <= y < k:
            kernel[y, x] = 1.0

    kernel = kernel / kernel.sum() if kernel.sum() > 0 else kernel
    return cv2.filter2D(img, -1, kernel)


def apply_defocus_blur(img, radius_range=(3, 12)):
    """Defocus (out-of-focus) blur."""
    radius = random.randint(radius_range[0], radius_range[1])
    kernel_size = 2 * radius + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    cv2.circle(kernel, (radius, radius), radius, 1, -1)
    kernel = kernel / kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def create_blur(img, task='face'):
    """Tạo blur phù hợp với từng task."""
    if task == 'face':
        # Mặt người: chủ yếu gaussian và motion blur nhẹ
        blur_type = random.choice(['gaussian', 'motion', 'defocus'])
        if blur_type == 'gaussian':
            return apply_gaussian_blur(img, kernel_range=(5, 21))
        elif blur_type == 'motion':
            return apply_motion_blur(img, kernel_range=(5, 20))
        else:
            return apply_defocus_blur(img, radius_range=(2, 8))

    elif task == 'scene':
        # Cảnh: motion blur mạnh hơn (camera shake)
        blur_type = random.choice(['gaussian', 'motion', 'motion', 'defocus'])
        if blur_type == 'gaussian':
            return apply_gaussian_blur(img, kernel_range=(7, 31))
        elif blur_type == 'motion':
            return apply_motion_blur(img, kernel_range=(10, 35))
        else:
            return apply_defocus_blur(img, radius_range=(3, 12))

    elif task == 'idcard':
        # ID Card: gaussian và motion blur nhẹ-vừa
        blur_type = random.choice(['gaussian', 'motion', 'defocus'])
        if blur_type == 'gaussian':
            return apply_gaussian_blur(img, kernel_range=(5, 15))
        elif blur_type == 'motion':
            return apply_motion_blur(img, kernel_range=(5, 15))
        else:
            return apply_defocus_blur(img, radius_range=(2, 6))

    return img


def main():
    parser = argparse.ArgumentParser(description='Tạo dataset blur từ ảnh sharp')
    parser.add_argument('--input', type=str, required=True,
                        help='Thư mục chứa ảnh sharp')
    parser.add_argument('--output', type=str, required=True,
                        help='Thư mục lưu ảnh blur')
    parser.add_argument('--task', type=str, required=True,
                        choices=['face', 'scene', 'idcard'],
                        help='Loại ảnh: face, scene, idcard')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # Lấy danh sách ảnh
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    img_paths = []
    for ext in extensions:
        img_paths.extend(glob.glob(osp.join(args.input, ext)))
        img_paths.extend(glob.glob(osp.join(args.input, ext.upper())))
    img_paths = sorted(list(set(img_paths)))

    print(f'[*] Task: {args.task}')
    print(f'[*] Tìm thấy {len(img_paths)} ảnh sharp')
    print(f'[*] Output: {args.output}')

    for idx, img_path in enumerate(img_paths, 1):
        img = cv2.imread(img_path)
        if img is None:
            print(f'  [!] Bỏ qua: {img_path}')
            continue

        blurred = create_blur(img, args.task)

        img_name = osp.basename(img_path)
        save_path = osp.join(args.output, img_name)
        cv2.imwrite(save_path, blurred)

        if idx % 100 == 0 or idx == len(img_paths):
            print(f'  [{idx}/{len(img_paths)}] Đã xử lý')

    print(f'[✓] Hoàn tất! Đã tạo {len(img_paths)} ảnh blur')


if __name__ == '__main__':
    main()
