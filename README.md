# CITD_Advance_ML_Project

## Hướng dẫn sử dụng giao diện

Ứng dụng chạy trên CPU, dùng model STID và scaler đã lưu trong repo.

1. **Lần đầu**, mở thư mục chứa `app.py` trong VS Code. Cài Python 3.11 rồi chạy hai lệnh sau trong terminal PowerShell của dự án.

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements-gradio.txt
   ```

2. **Mở giao diện**, nhấn `Ctrl+Shift+P`, chọn **Python: Select Interpreter** rồi chọn `.venv\Scripts\python.exe`. Mở `app.py`, bấm **Run Python File in Terminal** và mở link terminal hiển thị, thường là `http://127.0.0.1:7860`.

3. **Lần sau**, chạy lại `app.py`. Nếu gặp `No module named 'gradio'`, chọn lại `.venv` và chạy lại lệnh `pip install` ở bước 1.

- **Dự báo** mở sẵn tập test PeMS04. Chọn cảm biến và bấm **Dự báo**. STID tự dùng 24 mốc cuối dữ liệu để dự báo sau mốc cuối 15 và 30 phút, tính bằng xe/5 phút. Đổi cảm biến để xem kết quả khác của cùng lượt chạy. File CSV kết quả chứa cả 307 cảm biến và hai mốc dự báo.
- **Tải CSV riêng** có file mẫu. Dùng CSV UTF-8 gồm cột `timestamp` và đủ các ID `0` đến `306`, ít nhất 24 dòng liên tiếp, mỗi dòng cách nhau 5 phút. Dùng giờ địa phương của PeMS04, ví dụ `2018-02-17 08:00`. Các cột cảm biến có thể đổi thứ tự. STID dùng 24 mốc cuối trong file; giữ nguyên timestamp gốc. Ô trống là số đo thiếu. Số 0 là lưu lượng hợp lệ.
- **Giả lập cảm biến hỏng** chỉ có khi chọn **PeMS04 có sẵn**. Chức năng này so sánh hai dự báo trên cùng cửa sổ. Chọn mất ngẫu nhiên 5/20/30% trong 15/30/45 phút hoặc mất theo vùng 10% trong 15/30 phút cuối. Cảm biến đang chọn khi bấm mô phỏng là tâm vùng. Tâm cần có đủ 31 cảm biến nối với nhau, ví dụ ID 0.
- **Đánh giá mô hình** hiển thị báo cáo đã lưu của 12 kịch bản trên toàn tập test. Báo cáo giữ ba mốc 15, 30, 60 phút và trung bình ba mốc. Có thể lọc MAE, RMSE, MAPE theo kịch bản và mốc dự báo rồi tải CSV.

## Huấn luyện và tuning

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
