Phục hồi build_model - Nam Anh.ipynb

- File chính đã được phục hồi thành pipeline STID/STGCN gọn, mặc định STID, có lưu/load .pkl.
- original-with-conflicts.ipynb: nguyên bản trước khi sửa, giữ đúng từng byte.
- upstream-reference.ipynb và stashed-reference.ipynb: hai phía conflict để đối chiếu; chưa được sửa logic chạy.
- recovered-stid-stgcn.ipynb: bản đã ghép và kiểm tra, trùng file chính lúc phục hồi.
- recovery-manifest.json: nguồn các cell và hash bản gốc.
- recovered-source.py: source ghép để kiểm tra cú pháp, không thay thế notebook.

Kiểm tra: JSON hợp lệ; compile 16 code cells (bỏ qua lệnh %pip); không thiếu biến global/khóa CFG trong phân tích tĩnh. Chưa train hoặc kiểm tra save/load thực tế vì runtime kiểm tra thiếu torch/scipy/matplotlib/IPython.
