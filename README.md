# CITD_Advance_ML_Project

Notebook đang dùng: `build_model - Nam Anh.ipynb`.

Chỉ tiếp tục tune **STID + StandardScaler**; không tune STGCN. Chọn checkpoint và
cấu hình bằng **validation MAPE**. Giữ nguyên công thức metric, cách chia dữ liệu
theo thời gian và loss MAE có mask; không dùng test để chọn cấu hình.

Chạy trong môi trường Python có PyTorch CUDA (kernel `intel-dl`):

```powershell
python -u tune_models.py --smoke
python -u tune_models.py --run --stage screen --max-trials 6 --batch-size 64 --eval-batch-size 64
python -u tune_models.py --run --stage full --max-trials 2 --batch-size 64 --eval-batch-size 64
```

Giữ nguyên các tham số search khi chạy tiếp để resume. Kết quả, cấu hình tốt nhất
và model `.pkl` nằm trong `outputs/tuning_stid/`. Checkpoint cũ được giữ riêng.
Chỉ thêm `--evaluate-final-test` sau khi hoàn tất chọn cấu hình trên validation.

Kết quả lượt STID đã hoàn tất (6 cấu hình sàng lọc, 2 lượt train dài):

| MAPE trung bình | Bản cũ | Bản tune |
| --- | ---: | ---: |
| Validation | 11.5452% | 11.0745% |
| Test sạch | 11.9980% | 11.6213% |

Model mới: `outputs/stid_standard_tuned.pkl`. Cấu hình đã áp dụng vào notebook:
24 bước quá khứ, hidden 128, embedding 32, depth 3, dropout 0.05, LR 0.001;
checkpoint tốt nhất ở epoch 65. File model và kết quả cũ được giữ nguyên.
Chi tiết: `outputs/stid_tuning_result.json`,
`outputs/stid_standard_tuned_train.csv`.

Để tiếp tục đúng lượt đã lưu sau khi notebook được cập nhật cấu hình, dùng nguồn
snapshot của lượt đó. Lệnh sau bỏ qua 6 cấu hình đã xong và chạy tiếp phần sàng lọc:

```powershell
python -u tune_models.py --run --stage screen --max-trials 6 --batch-size 64 --eval-batch-size 64 --notebook outputs/tuning_stid/staged-6587c07b23fa/source-notebook.ipynb
```
