"""
Script tải và chuẩn bị dataset cho đồ án Deblur/Unblur.
Tự động tải, giải nén, và sắp xếp vào đúng cấu trúc thư mục.

Cách dùng:
    # Tải tất cả datasets:
    python scripts/data_preparation/download_deblur_datasets.py --all

    # Tải từng loại:
    python scripts/data_preparation/download_deblur_datasets.py --scene
    python scripts/data_preparation/download_deblur_datasets.py --face
    python scripts/data_preparation/download_deblur_datasets.py --idcard

Datasets sử dụng:
    - Scene: GoPro Large (CVPR 2017) - 2,103 train + 1,111 test pairs
    - Face: CelebA-HQ (tải sharp, tự tạo blur)
    - ID Card: Tự tạo từ ảnh mẫu (không có dataset công khai)
"""

import argparse
import glob
import os
import shutil
import sys
import zipfile
from os import path as osp

# ============================================================================ #
#                              DATASET SOURCES                                  #
# ============================================================================ #

DATASETS = {
    'gopro': {
        'name': 'GoPro Large (Scene Deblurring)',
        'url_huggingface': 'https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large.zip',
        'url_gdrive': 'https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing',
        'gdrive_id': '1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK',
        'filename': 'GOPRO_Large.zip',
        'size': '~5.6GB',
        'license': 'CC BY 4.0',
        'citation': 'Nah et al., Deep Multi-Scale CNN for Dynamic Scene Deblurring, CVPR 2017',
    },
}


def download_with_urllib(url, save_path):
    """Tải file bằng urllib với progress bar."""
    import urllib.request

    print(f'  Đang tải từ: {url}')
    print(f'  Lưu tại: {save_path}')

    def reporthook(count, block_size, total_size):
        downloaded = count * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f'\r  [{percent:5.1f}%] {mb_downloaded:.1f}/{mb_total:.1f} MB')
        else:
            mb_downloaded = downloaded / (1024 * 1024)
            sys.stdout.write(f'\r  {mb_downloaded:.1f} MB downloaded')
        sys.stdout.flush()

    urllib.request.urlretrieve(url, save_path, reporthook)
    print()


def download_with_gdown(gdrive_id, save_path):
    """Tải file từ Google Drive bằng gdown."""
    try:
        import gdown
    except ImportError:
        print('  [!] Cần cài gdown: pip install gdown')
        print('  Đang cài gdown...')
        os.system(f'{sys.executable} -m pip install gdown')
        import gdown

    url = f'https://drive.google.com/uc?id={gdrive_id}'
    print(f'  Đang tải từ Google Drive...')
    gdown.download(url, save_path, quiet=False)


def extract_zip(zip_path, extract_to):
    """Giải nén file zip."""
    print(f'  Đang giải nén: {zip_path}')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f'  [✓] Giải nén xong!')


def collect_images(folder, extensions=('*.png', '*.jpg', '*.jpeg', '*.bmp')):
    """Thu thập tất cả ảnh trong folder (recursive)."""
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(osp.join(folder, '**', ext), recursive=True))
        paths.extend(glob.glob(osp.join(folder, '**', ext.upper()), recursive=True))
    return sorted(list(set(paths)))


# ============================================================================ #
#                           SCENE: GoPro Dataset                                #
# ============================================================================ #

def setup_scene_dataset(data_root='datasets', download_dir='downloads', use_gdrive=False):
    """
    Tải và chuẩn bị GoPro dataset cho Scene Deblurring.

    GoPro structure:
        GOPRO_Large/
            train/
                GOPR0372_07_00/
                    blur/          ← ảnh blur
                    blur_gamma/
                    sharp/         ← ảnh sharp
                ...
            test/
                GOPR0384_11_00/
                    blur/
                    sharp/
                ...
    """
    print('\n' + '='*60)
    print('📸 SCENE DEBLURRING - GoPro Large Dataset')
    print('='*60)

    info = DATASETS['gopro']
    print(f'  Dataset: {info["name"]}')
    print(f'  Kích thước: {info["size"]}')
    print(f'  License: {info["license"]}')

    os.makedirs(download_dir, exist_ok=True)
    zip_path = osp.join(download_dir, info['filename'])

    # Tải dataset
    if not osp.exists(zip_path):
        if use_gdrive:
            download_with_gdown(info['gdrive_id'], zip_path)
        else:
            download_with_urllib(info['url_huggingface'], zip_path)
    else:
        print(f'  [✓] Đã có file: {zip_path}')

    # Giải nén
    gopro_dir = osp.join(download_dir, 'GOPRO_Large')
    if not osp.exists(gopro_dir):
        extract_zip(zip_path, download_dir)

    # Sắp xếp vào cấu trúc datasets/scene/
    scene_dir = osp.join(data_root, 'scene')
    for split in ['train', 'val', 'test']:
        os.makedirs(osp.join(scene_dir, split, 'blur'), exist_ok=True)
        os.makedirs(osp.join(scene_dir, split, 'sharp'), exist_ok=True)

    # GoPro chỉ có train và test, ta dùng test làm val+test
    print('\n  Đang sắp xếp ảnh...')

    # Copy train images
    train_dir = osp.join(gopro_dir, 'train')
    if osp.exists(train_dir):
        count = 0
        for scene_folder in sorted(os.listdir(train_dir)):
            scene_path = osp.join(train_dir, scene_folder)
            if not osp.isdir(scene_path):
                continue

            blur_images = collect_images(osp.join(scene_path, 'blur'))
            sharp_images = collect_images(osp.join(scene_path, 'sharp'))

            for blur_img, sharp_img in zip(blur_images, sharp_images):
                blur_name = f'{scene_folder}_{osp.basename(blur_img)}'
                sharp_name = f'{scene_folder}_{osp.basename(sharp_img)}'
                shutil.copy2(blur_img, osp.join(scene_dir, 'train', 'blur', blur_name))
                shutil.copy2(sharp_img, osp.join(scene_dir, 'train', 'sharp', sharp_name))
                count += 1

        print(f'  [✓] Train: {count} pairs')

    # Copy test images (chia 50/50 cho val và test)
    test_dir = osp.join(gopro_dir, 'test')
    if osp.exists(test_dir):
        all_test_scenes = sorted([d for d in os.listdir(test_dir) if osp.isdir(osp.join(test_dir, d))])
        mid = len(all_test_scenes) // 2
        val_scenes = all_test_scenes[:mid]
        test_scenes = all_test_scenes[mid:]

        for split_name, scenes in [('val', val_scenes), ('test', test_scenes)]:
            count = 0
            for scene_folder in scenes:
                scene_path = osp.join(test_dir, scene_folder)
                blur_images = collect_images(osp.join(scene_path, 'blur'))
                sharp_images = collect_images(osp.join(scene_path, 'sharp'))

                for blur_img, sharp_img in zip(blur_images, sharp_images):
                    blur_name = f'{scene_folder}_{osp.basename(blur_img)}'
                    sharp_name = f'{scene_folder}_{osp.basename(sharp_img)}'
                    shutil.copy2(blur_img, osp.join(scene_dir, split_name, 'blur', blur_name))
                    shutil.copy2(sharp_img, osp.join(scene_dir, split_name, 'sharp', sharp_name))
                    count += 1
            print(f'  [✓] {split_name.capitalize()}: {count} pairs')

    print(f'\n  [✓] Scene dataset sẵn sàng tại: {scene_dir}')


# ============================================================================ #
#                           FACE: CelebA-HQ + Synthetic Blur                    #
# ============================================================================ #

def setup_face_dataset(data_root='datasets', download_dir='downloads', num_images=3000):
    """
    Chuẩn bị Face Deblurring dataset.

    Chiến lược: Tải ảnh face chất lượng cao, rồi tạo blur tổng hợp.
    Dùng CelebA-HQ từ Kaggle hoặc tự chuẩn bị.
    """
    print('\n' + '='*60)
    print('🧑 FACE DEBLURRING Dataset')
    print('='*60)

    face_dir = osp.join(data_root, 'face')

    # Kiểm tra xem đã có ảnh sharp chưa
    existing_sharp = collect_images(osp.join(face_dir, 'train', 'sharp'))

    if len(existing_sharp) > 0:
        print(f'  [✓] Đã có {len(existing_sharp)} ảnh sharp trong train/')
    else:
        print(f'''
  ⚠️  Cần chuẩn bị ảnh mặt người chất lượng cao (sharp).

  CÁCH 1 - CelebA-HQ (khuyến nghị):
    1. Tải từ Kaggle: https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256
    2. Giải nén và copy ảnh vào: {osp.join(face_dir, 'train', 'sharp')}

  CÁCH 2 - FFHQ:
    1. Tải từ: https://github.com/NVlabs/ffhq-dataset
    2. Copy ảnh vào: {osp.join(face_dir, 'train', 'sharp')}

  CÁCH 3 - Tự thu thập:
    1. Thu thập ảnh mặt người rõ nét (không blur)
    2. Resize về 256x256 hoặc 512x512
    3. Copy vào: {osp.join(face_dir, 'train', 'sharp')}

  Sau khi có ảnh sharp, chạy lệnh sau để tạo blur:
    python scripts/data_preparation/create_blur_dataset.py \\
        --input {osp.join(face_dir, 'train', 'sharp')} \\
        --output {osp.join(face_dir, 'train', 'blur')} \\
        --task face
''')

    return face_dir


# ============================================================================ #
#                           ID CARD: Synthetic Dataset                          #
# ============================================================================ #

def setup_idcard_dataset(data_root='datasets'):
    """
    Chuẩn bị ID Card Deblurring dataset.

    ID Card là dạng dữ liệu đặc thù, không có dataset công khai.
    Cần tự tạo hoặc thu thập.
    """
    print('\n' + '='*60)
    print('🪪 ID CARD DEBLURRING Dataset')
    print('='*60)

    idcard_dir = osp.join(data_root, 'idcard')

    existing_sharp = collect_images(osp.join(idcard_dir, 'train', 'sharp'))

    if len(existing_sharp) > 0:
        print(f'  [✓] Đã có {len(existing_sharp)} ảnh sharp trong train/')
    else:
        print(f'''
  ⚠️  ID Card dataset cần tự chuẩn bị (không có dataset công khai).

  CÁCH CHUẨN BỊ:
    1. Thu thập/scan ID card rõ nét (CMND, CCCD, bằng lái, passport...)
       - Có thể dùng ảnh mẫu, hoặc tạo ID card giả bằng template
       - Khuyến nghị: 500-2000 ảnh cho train, 100-200 cho val

    2. Resize tất cả ảnh về cùng kích thước (ví dụ 256x256 hoặc 512x256)

    3. Copy vào: {osp.join(idcard_dir, 'train', 'sharp')}

    4. Tạo blur tự động:
       python scripts/data_preparation/create_blur_dataset.py \\
           --input {osp.join(idcard_dir, 'train', 'sharp')} \\
           --output {osp.join(idcard_dir, 'train', 'blur')} \\
           --task idcard

  GỢI Ý NGUỒN DỮ LIỆU:
    - MIDV-500: Dataset ID documents (https://arxiv.org/abs/1807.05786)
    - MIDV-2019: Extended ID documents dataset
    - BID Dataset: Blurry ID card images
    - Tự tạo template ID card và render nhiều biến thể
''')

    return idcard_dir


# ============================================================================ #
#                              MAIN                                             #
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(description='Tải và chuẩn bị datasets cho Deblur')
    parser.add_argument('--all', action='store_true', help='Chuẩn bị tất cả datasets')
    parser.add_argument('--scene', action='store_true', help='Tải GoPro scene dataset')
    parser.add_argument('--face', action='store_true', help='Chuẩn bị face dataset')
    parser.add_argument('--idcard', action='store_true', help='Chuẩn bị ID card dataset')
    parser.add_argument('--data_root', type=str, default='datasets', help='Thư mục gốc datasets')
    parser.add_argument('--download_dir', type=str, default='downloads', help='Thư mục tải về')
    parser.add_argument('--gdrive', action='store_true', help='Dùng Google Drive thay vì HuggingFace')
    args = parser.parse_args()

    if not any([args.all, args.scene, args.face, args.idcard]):
        parser.print_help()
        print('\n  Ví dụ: python scripts/data_preparation/download_deblur_datasets.py --all')
        return

    print('🚀 Chuẩn bị datasets cho Deblur/Unblur ảnh')
    print(f'   Data root: {osp.abspath(args.data_root)}')

    if args.all or args.scene:
        setup_scene_dataset(args.data_root, args.download_dir, args.gdrive)

    if args.all or args.face:
        setup_face_dataset(args.data_root, args.download_dir)

    if args.all or args.idcard:
        setup_idcard_dataset(args.data_root)

    print('\n' + '='*60)
    print('📋 TỔNG KẾT')
    print('='*60)
    for task in ['face', 'scene', 'idcard']:
        task_dir = osp.join(args.data_root, task)
        for split in ['train', 'val', 'test']:
            blur_count = len(collect_images(osp.join(task_dir, split, 'blur')))
            sharp_count = len(collect_images(osp.join(task_dir, split, 'sharp')))
            if blur_count > 0 or sharp_count > 0:
                status = '✓' if blur_count > 0 and sharp_count > 0 else '⚠️'
                print(f'  [{status}] {task}/{split}: {sharp_count} sharp, {blur_count} blur')
            else:
                print(f'  [  ] {task}/{split}: trống')

    print('\n  Bước tiếp theo:')
    print('  1. Đảm bảo mỗi thư mục có đủ cặp blur-sharp')
    print('  2. Nếu chỉ có sharp, chạy create_blur_dataset.py để tạo blur')
    print('  3. Bắt đầu training:')
    print('     python basicsr/train.py -opt options/train/train_deblur_face.yml')


if __name__ == '__main__':
    main()
