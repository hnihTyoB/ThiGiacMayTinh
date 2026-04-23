# Đồ án Deblur/Unblur Hình Ảnh - Thị Giác Máy Tính
> Hệ thống khôi phục ảnh mờ (Deblur) cho 3 cấp độ: Mặt người, Cảnh vật và Căn cước công dân (ID Card).

## 1. Giới thiệu
Dự án được xây dựng dựa trên framework **BasicSR** và sử dụng kiến trúc mạng **SwinIR** để giải quyết bài toán Image Deblurring. Thay vì sử dụng một model chung cho mọi trường hợp, hệ thống sử dụng 3 mô hình chuyên biệt cho 3 domain khác nhau nhằm đạt hiệu quả tối ưu nhất.

Các cấp độ khôi phục:
1. **Face (Mặt người)**: Tập trung vào chi tiết da, mắt, tóc.
2. **Scene (Cảnh vật)**: Tập trung vào các chi tiết phong cảnh phức tạp (dùng cấu trúc mạng SwinIR lớn hơn).
3. **ID Card (Giấy tờ tùy thân)**: Tối ưu cho việc làm rõ chữ (text clarity) mà không làm biến dạng cấu trúc văn bản.

## 2. Cấu trúc thư mục (Các file quan trọng)
```
├── basicsr/             # Framework core (pipeline train, loss, metrics)
├── datasets/            # Chứa ảnh sharp và blur
├── inference/
│   └── inference_deblur.py # Script để test model trên ảnh mới
├── options/             # Các file YAML cấu hình training/testing
│   ├── train/
│   │   ├── train_deblur_face.yml
│   │   ├── train_deblur_scene.yml
│   │   └── train_deblur_idcard.yml
│   └── test/
├── scripts/data_preparation/
│   ├── create_blur_dataset.py       # Script tạo ảnh blur tổng hợp
│   └── download_deblur_datasets.py  # Script tải dataset tự động
└── train_deblur_colab.ipynb         # Notebook train trên Google Colab
```

## 3. Hướng dẫn sử dụng trên Google Colab
Dự án đã được tối ưu để huấn luyện miễn phí trên GPU của Google Colab.

1. Đẩy mã nguồn này lên GitHub của bạn.
2. Mở Google Colab, tạo sổ tay mới.
3. Chạy `git clone https://github.com/<user_name>/ThiGiacMayTinh.git`
4. Tham khảo các bước chi tiết trong file `train_deblur_colab.ipynb` để tải dataset và bắt đầu training.

## 4. Chuẩn bị Dataset
Nếu bạn muốn chạy ở máy cá nhân, sử dụng script tải tự động:
```bash
python scripts/data_preparation/download_deblur_datasets.py --all
```

## 5. Training
```bash
python basicsr/train.py -opt options/train/train_deblur_face.yml
```

## 6. Inference (Chạy thử)
```bash
python inference/inference_deblur.py \
    --input test_images \
    --output results \
    --model_path experiments/DeblurFace_SwinIR/models/net_g_latest.pth \
    --task face
```
