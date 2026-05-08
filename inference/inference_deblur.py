"""
Inference script cho Deblur/Unblur ảnh.
Hỗ trợ 3 cấp độ: face, scene, idcard.

Cách dùng:
    python inference/inference_deblur.py \
        --input path/to/blurry/images \
        --output results/deblur \
        --model_path experiments/DeblurFace_SwinIR/models/net_g_latest.pth \
        --task face
"""

import argparse
import cv2
import glob
import numpy as np
import os
import sys
import torch
from os import path as osp

# Tự động thêm đường dẫn thư mục cha để nhận diện module basicsr (không cần pip install)
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))

from basicsr.archs.swinir_arch import SwinIR


# ======================== Model configs cho 3 cấp độ ======================== #
MODEL_CONFIGS = {
    'face': dict(
        img_size=128,
        depths=[6, 6, 6, 6],
        embed_dim=96,
        num_heads=[6, 6, 6, 6],
    ),
    'scene': dict(
        img_size=128,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
    ),
    'idcard': dict(
        img_size=128,
        depths=[6, 6, 6, 6],
        embed_dim=96,
        num_heads=[6, 6, 6, 6],
    ),
}


def get_image_paths(input_path):
    """Lấy danh sách ảnh từ file hoặc folder."""
    if osp.isfile(input_path):
        return [input_path]

    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(osp.join(input_path, ext)))
        paths.extend(glob.glob(osp.join(input_path, ext.upper())))
    return sorted(list(set(paths)))


def load_model(model_path, task, device='cuda'):
    """Load SwinIR model cho task cụ thể."""
    cfg = MODEL_CONFIGS[task]

    model = SwinIR(
        upscale=1,
        in_chans=3,
        img_range=1.,
        window_size=8,
        mlp_ratio=4,
        upsampler='',
        resi_connection='1conv',
        **cfg,
    )

    # Load pretrained weights
    loadnet = torch.load(model_path, map_location=torch.device('cpu'))

    # Xử lý các format khác nhau của checkpoint
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    elif 'params' in loadnet:
        keyname = 'params'
    else:
        keyname = None

    if keyname is not None:
        model.load_state_dict(loadnet[keyname], strict=True)
    else:
        model.load_state_dict(loadnet, strict=True)

    model.eval()
    model = model.to(device)
    return model


def pad_to_window(img, window_size=8):
    """Pad ảnh để kích thước chia hết cho window_size."""
    _, _, h, w = img.size()
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    return img, h, w


def inference_single(model, img_path, device='cuda', window_size=8):
    """Xử lý deblur 1 ảnh."""
    # Đọc ảnh (BGR -> RGB, [0,255] -> [0,1])
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f'  [!] Không đọc được ảnh: {img_path}')
        return None

    # CHỐNG TRÀN RAM: Tự động thu nhỏ ảnh nếu quá lớn (Tối đa 800px)
    max_dim = 800
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        print(f'  (Đã tự động thu nhỏ xuống {img.shape[1]}x{img.shape[0]} để chống tràn RAM) ', end='')

    img = img.astype(np.float32) / 255.0
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # HWC -> CHW -> NCHW
    img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0).to(device)

    # Pad ảnh
    img_padded, h_orig, w_orig = pad_to_window(img_tensor, window_size)

    # Inference
    with torch.no_grad():
        output = model(img_padded)

    # Crop lại kích thước gốc
    output = output[:, :, :h_orig, :w_orig]

    # Chuyển tensor -> numpy
    output = output.squeeze(0).cpu().clamp(0, 1).numpy()
    output = np.transpose(output, (1, 2, 0))  # CHW -> HWC
    # RGB -> BGR cho cv2
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    output = (output * 255.0).round().astype(np.uint8)

    return output


def main():
    parser = argparse.ArgumentParser(description='Deblur/Unblur ảnh với SwinIR')
    parser.add_argument('--input', type=str, required=True,
                        help='Đường dẫn tới ảnh hoặc thư mục ảnh blur')
    parser.add_argument('--output', type=str, default='results/deblur',
                        help='Thư mục lưu kết quả')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Đường dẫn tới file model (.pth)')
    parser.add_argument('--task', type=str, required=True,
                        choices=['face', 'scene', 'idcard'],
                        help='Loại ảnh cần deblur: face, scene, hoặc idcard')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device để chạy inference')
    args = parser.parse_args()

    # Kiểm tra CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[!] CUDA không khả dụng, chuyển sang CPU...')
        args.device = 'cpu'

    # Tạo thư mục output
    os.makedirs(args.output, exist_ok=True)

    # Load model
    print(f'[*] Loading model: {args.task} từ {args.model_path}')
    model = load_model(args.model_path, args.task, args.device)
    print(f'[✓] Model loaded thành công!')

    # Lấy danh sách ảnh
    img_paths = get_image_paths(args.input)
    print(f'[*] Tìm thấy {len(img_paths)} ảnh')

    # Xử lý từng ảnh
    for idx, img_path in enumerate(img_paths, 1):
        img_name = osp.splitext(osp.basename(img_path))[0]
        print(f'  [{idx}/{len(img_paths)}] Đang xử lý: {img_name}...', end=' ')

        result = inference_single(model, img_path, args.device)
        if result is not None:
            save_path = osp.join(args.output, f'{img_name}_deblur.png')
            cv2.imwrite(save_path, result)
            print(f'✓ Đã lưu: {save_path}')
        else:
            print('✗ Lỗi!')

    print(f'\n[✓] Hoàn tất! Kết quả lưu tại: {args.output}')


if __name__ == '__main__':
    main()
