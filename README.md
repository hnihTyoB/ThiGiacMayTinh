# Đồ án Deblur/Unblur Hình Ảnh - Thị Giác Máy Tính

> Hệ thống khôi phục ảnh mờ (Deblur) cho 2 cấp độ: Mặt người (Face) và Căn cước công dân (ID Card).

---

## Mục lục

- [1. Giới thiệu](#1-giới-thiệu)
- [2. Cấu trúc thư mục (Các file quan trọng)](#2-cấu-trúc-thư-mục-các-file-quan-trọng)
- [3. Tải Trọng số Mô hình (Pretrained Checkpoints)](#3-tải-trọng-số-mô-hình-pretrained-checkpoints)
- [4. Hướng dẫn sử dụng trên Google Colab](#4-hướng-dẫn-sử-dụng-trên-google-colab)
- [5. Chuẩn bị Dataset](#5-chuẩn-bị-dataset)
- [6. Training (Huấn luyện)](#6-training-huấn-luyện)
- [7. Inference (Chạy thử nghiệm)](#7-inference-chạy-thử-nghiệm)

---

## 1. Giới thiệu

Dự án sử dụng kiến trúc mạng **SwinIR** (Swin Transformer for Image Restoration) để giải quyết bài toán Image Deblurring. Thay vì sử dụng một model chung cho mọi trường hợp, hệ thống sử dụng 2 mô hình chuyên biệt cho 2 nhóm đối tượng nhằm đạt hiệu quả tối ưu nhất:

1. **Face (Mặt người)**: Tập trung vào khôi phục chi tiết da, mắt, tóc và các đặc trưng nhận diện trên khuôn mặt.
2. **ID Card (Giấy tờ tùy thân / CCCD)**: Tối ưu đặc biệt cho việc làm rõ chữ (text clarity), giữ nét cạnh văn bản phục vụ cho các hệ thống OCR tiếp theo mà không làm biến dạng cấu trúc ký tự.

## 2. Cấu trúc thư mục (Các file quan trọng)

```
├── basicsr/             # Framework core (pipeline train, loss, metrics)
├── datasets/            # Chứa ảnh sharp và blur dùng để huấn luyện
├── inference/
│   └── inference_deblur.py # Script để test model trên ảnh mới
├── options/             # Các file YAML cấu hình training/testing
│   ├── train/
│   │   ├── train_deblur_face.yml
│   │   └── train_deblur_idcard.yml
│   └── test/
│       ├── test_deblur_face.yml
│       └── test_deblur_idcard.yml
├── scripts/data_preparation/
│   ├── create_blur_dataset.py       # Script tạo ảnh blur tổng hợp
│   └── download_deblur_datasets.py  # Script tải dataset tự động
└── train_deblur_colab.ipynb         # Notebook train trên Google Colab
```

## 3. Tải Trọng số Mô hình (Pretrained Checkpoints)

Các mô hình huấn luyện sẵn đã được lưu trữ công khai và chia sẻ tại liên kết sau:

- **Link Google Drive**: [DeblurModels Checkpoints](https://drive.google.com/drive/folders/1j_NIC4P0BB2DhEVLrT4FyytxeWWvsLwl?usp=sharing)
  - `DeblurFace_SwinIR_V2/`: Chứa các file checkpoint của mô hình mặt người.
  - `DeblurIDCard_SwinIR_V2/`: Chứa các file checkpoint của mô hình Căn cước công dân.

### A. Chạy thử nghiệm trực tiếp trên máy cục bộ (Local)

1. Truy cập vào Link Drive ở trên và tải thư mục hoặc các file mô hình `.pth` mong muốn (ví dụ `net_g_latest.pth` hoặc `net_g_500000.pth`).
2. Di chuyển các file mô hình đã tải về vào đúng thư mục tương ứng trong dự án của bạn theo cấu trúc sau:
   - Đối với Face: `experiments/DeblurFace_SwinIR_V2/models/net_g_latest.pth`
   - Đối với ID Card: `experiments/DeblurIDCard_SwinIR_V2/models/net_g_latest.pth`
3. Tiến hành chạy lệnh kiểm tra tại mục [7. Inference](#7-inference-chạy-thử-nghiệm).

### B. Sử dụng trên Google Colab để huấn luyện tiếp hoặc kiểm thử

1. Truy cập Link Drive được chia sẻ ở trên.
2. Nhấp chuột phải vào thư mục **`DeblurModels`** -> Chọn **"Thêm lối tắt vào Drive" (Add shortcut to Drive)** và chọn lưu tại **"Drive của tôi" (My Drive)** của bạn.
3. Khi bạn chạy file `train_deblur_colab.ipynb` trên Colab, tiến trình sẽ tự động tạo symlink liên kết thư mục `experiments` với `MyDrive/DeblurModels`. Việc này giúp Colab tự nhận diện các checkpoint có sẵn để tiếp tục huấn luyện (`resume_state`) hoặc sử dụng trực tiếp để chạy test.

## 4. Hướng dẫn sử dụng trên Google Colab

Dự án đã được tối ưu để huấn luyện miễn phí trên GPU của Google Colab.

1. Đẩy mã nguồn này lên GitHub cá nhân của bạn.
2. Thực hiện các bước tạo lối tắt thư mục `DeblurModels` trên Google Drive như hướng dẫn ở mục 3.B.
3. Mở Google Colab và tải file `train_deblur_colab.ipynb` lên.
4. Chạy tuần tự các cell trong notebook để Clone code, Mount Drive, cài đặt thư viện và bắt đầu chạy huấn luyện/kiểm tra.

## 5. Chuẩn bị Dataset

Chúng tôi cung cấp sẵn bộ dữ liệu huấn luyện đã được đóng gói dưới định dạng `.zip` cho cả 2 cấp độ:
*   **Dataset Face (Mặt người)**: [celeba_hq_256.zip (Google Drive)](https://drive.google.com/file/d/1pRr5nMiyRjNwHLqboGXJyzMfQUKthhAz/view?usp=sharing)
*   **Dataset ID Card (Căn cước công dân)**: [idcardv2.zip (Google Drive)](https://drive.google.com/file/d/1qwXyltXDjVRBxNWURr4Oj903y7yRjffm/view?usp=sharing)

### A. Chạy trên máy tính cá nhân (Local)
1. Tải các file `.zip` dữ liệu ở các đường dẫn trên về máy tính.
2. Giải nén ảnh sắc nét (sharp) của từng tập dữ liệu vào đúng vị trí tương ứng trong dự án của bạn:
   - **Đối với Face**: Giải nén ảnh sắc nét vào thư mục `datasets/face/train/sharp/`
   - **Đối với ID Card**: Giải nén ảnh sắc nét vào thư mục `datasets/idcard/train/sharp/`
3. Chạy script để tự động tạo ảnh mờ (blur) tương ứng:
   ```bash
   # Tạo ảnh blur cho Face
   python scripts/data_preparation/create_blur_dataset.py --input datasets/face/train/sharp/ --output datasets/face/train/blur --task face

   # Tạo ảnh blur cho ID Card
   python scripts/data_preparation/create_blur_dataset.py --input datasets/idcard/train/sharp/ --output datasets/idcard/train/blur --task idcard
   ```

### B. Chạy trên Google Colab (ipynb)
Để việc chạy huấn luyện trên Colab diễn ra hoàn toàn tự động và nhanh chóng, bạn làm như sau:
1. Tải trực tiếp file `celeba_hq_256.zip` (cho Face) và/hoặc `idcardv2.zip` (cho ID Card) từ các link Drive trên lên **thư mục gốc** trong Google Drive của bạn (`My Drive` / `Drive của tôi`).
2. Khi mở và chạy file `train_deblur_colab.ipynb` trên Colab, các ô lệnh trong Notebook sẽ tự động sao chép (`cp`) file zip này từ Drive của bạn vào môi trường ảo Colab, giải nén và tự động chạy mã lệnh sinh ảnh mờ (blur) mà bạn không cần phải thực hiện thủ công.

## 6. Training (Huấn luyện)

Để bắt đầu quá trình huấn luyện lại từ đầu hoặc tiếp tục từ checkpoint có sẵn:

```bash
# Huấn luyện mô hình Face
python basicsr/train.py -opt options/train/train_deblur_face.yml

# Huấn luyện mô hình ID Card
python basicsr/train.py -opt options/train/train_deblur_idcard.yml
```

## 7. Inference (Chạy thử nghiệm)

Sử dụng script để khử mờ các ảnh mới bên ngoài tập huấn luyện:

```bash
# Chạy thử nghiệm mô hình Face
python inference/inference_deblur.py \
    --input test_images \
    --output results \
    --model_path experiments/DeblurFace_SwinIR_V2/models/net_g_latest.pth \
    --task face

# Chạy thử nghiệm mô hình ID Card
python inference/inference_deblur.py \
    --input test_images \
    --output results \
    --model_path experiments/DeblurIDCard_SwinIR_V2/models/net_g_latest.pth \
    --task idcard
```
