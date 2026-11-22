"""Dữ liệu PeMS04, cửa sổ dự báo và báo cáo dùng cho giao diện."""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


HISTORY_STEPS = 24
STEP = pd.Timedelta(minutes=5)
HORIZON_MINUTES = (15, 30, 60)


def _flow_frame(values, timestamps, sensor_ids):
    with np.errstate(over="ignore"):
        flow = np.asarray(values, dtype=np.float32)
    flow[~np.isfinite(flow) | (flow < 0)] = np.nan
    return pd.DataFrame(flow, index=timestamps.rename("timestamp"), columns=sensor_ids)


def load_demo(root):
    """Đọc tập test theo đúng cách chia 70/10/20 của notebook."""
    path = Path(root) / "data" / "pems04" / "pems04.npz"
    with np.load(path, allow_pickle=False) as archive:
        data = archive["data"]
    if data.ndim != 3 or data.shape[1] != 307 or data.shape[2] < 1:
        raise ValueError("Dữ liệu mẫu PeMS04 phải có 307 cảm biến và kênh flow.")
    total = len(data)
    first = int(total * 0.7) + int(total * 0.1)
    timestamps = pd.date_range(pd.Timestamp("2018-01-01") + first * STEP,
                               periods=total - first, freq=STEP)
    return _flow_frame(data[first:, :, 0].copy(), timestamps, [str(i) for i in range(307)])


def _short_list(labels):
    return ", ".join(labels[:6]) + ("…" if len(labels) > 6 else "")


def _csv_timestamps(raw):
    try:
        parsed = pd.to_datetime(raw, format="ISO8601", errors="coerce")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Dùng thời gian địa phương theo ISO, ví dụ 2018-02-17 07:00.") from exc
    if parsed.isna().any():
        row = int(np.flatnonzero(parsed.isna())[0]) + 1
        raise ValueError(f"Mốc thời gian không hợp lệ ở dòng dữ liệu {row}.")
    if not pd.api.types.is_datetime64_dtype(parsed.dtype):
        raise ValueError("Cột timestamp cần dùng giờ địa phương PeMS04, không kèm múi giờ.")
    timestamps = pd.DatetimeIndex(parsed, name="timestamp")
    if timestamps.has_duplicates:
        duplicate = timestamps[timestamps.duplicated()][0]
        raise ValueError(f"Mốc thời gian bị trùng: {duplicate}.")
    off_grid = timestamps != timestamps.floor("5min")
    if off_grid.any():
        raise ValueError(f"Mốc {timestamps[off_grid][0]} lệch lưới 5 phút.")
    gaps = timestamps[1:] - timestamps[:-1] != STEP
    if gaps.any():
        i = int(np.flatnonzero(gaps)[0])
        raise ValueError(f"Hai mốc {timestamps[i]} và {timestamps[i + 1]} phải tăng đúng 5 phút.")
    return timestamps


def load_csv(path, sensor_ids):
    """Đọc CSV UTF-8, kiểm tra ID và trả flow theo thứ tự cảm biến của scaler."""
    expected = list(sensor_ids)
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = [name.strip() for name in next(reader, [])]
            duplicates = [name for name, count in Counter(header).items() if count > 1]
            if duplicates:
                raise ValueError(f"Tên cột bị trùng: {_short_list(duplicates)}.")
            if "timestamp" not in header:
                raise ValueError("CSV cần cột timestamp và dùng dấu phẩy để phân cách cột.")
            missing = [sensor for sensor in expected if sensor not in header]
            extra = [name for name in header if name != "timestamp" and name not in expected]
            errors = []
            if missing:
                errors.append(f"Thiếu ID cảm biến: {_short_list(missing)}")
            if extra:
                errors.append(f"Cột ngoài mẫu: {_short_list(extra)}")
            if errors:
                raise ValueError(". ".join(errors) + ".")
            # pandas tự đổi tên cột trùng và tự điền NaN vào dòng thiếu trường.
            # Kiểm tra CSV gốc để báo lỗi đúng dòng trước khi đọc bảng số.
            for fields in reader:
                if fields and len(fields) != len(header):
                    raise ValueError(f"Dòng {reader.line_num} có {len(fields)} trường, cần {len(header)}.")
            stream.seek(0)
            rows = pd.read_csv(stream, header=0, names=header, dtype={"timestamp": str})
    except UnicodeDecodeError as exc:
        raise ValueError("CSV cần được lưu với mã hóa UTF-8.") from exc
    except (csv.Error, pd.errors.ParserError) as exc:
        raise ValueError("CSV lỗi định dạng. Kiểm tra dấu phẩy và số cột ở mỗi dòng.") from exc
    if len(rows) < HISTORY_STEPS:
        raise ValueError(f"CSV cần ít nhất {HISTORY_STEPS} mốc liên tiếp, hiện có {len(rows)}.")
    timestamps = _csv_timestamps(rows.pop("timestamp"))
    for sensor in expected:
        column = rows[sensor]
        if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
            continue
        raw = column.astype("string").str.strip()
        absent = raw.isna() | raw.str.lower().isin(["", "nan", "na", "null", "none"])
        numeric = pd.to_numeric(raw, errors="coerce")
        invalid = numeric.isna() & ~absent
        if invalid.any():
            row = int(np.flatnonzero(invalid)[0])
            raise ValueError(f"Flow của cảm biến {sensor} tại {timestamps[row]} phải là số hoặc ô trống.")
        rows[sensor] = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    return _flow_frame(rows[expected].to_numpy(copy=True), timestamps, expected)


def window_at(frame, origin):
    """Lấy đúng 24 mốc lịch sử kết thúc tại origin, kể cả origin."""
    try:
        position = frame.index.get_loc(pd.Timestamp(origin))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Thời điểm dự báo phải nằm trong dữ liệu đã chọn.") from exc
    if position < HISTORY_STEPS - 1:
        raise ValueError(f"Cần đủ {HISTORY_STEPS} mốc tính đến thời điểm dự báo.")
    return frame.iloc[position + 1 - HISTORY_STEPS:position + 1]


def observations_at(frame, origin, minutes=HORIZON_MINUTES):
    """Bảng số đo thực tế theo cảm biến và mốc dự báo, NaN nếu chưa có số đo."""
    horizons = list(minutes)
    future = pd.Timestamp(origin) + pd.to_timedelta(horizons, unit="min")
    result = frame.reindex(future).T.rename_axis("sensor_id")
    result.columns = horizons
    return result


def load_report(root):
    """Đọc kết quả STID trên toàn tập test, giữ nguyên cột và độ chính xác đã lưu."""
    path = Path(root) / "outputs" / "stid_standard_tuned_robustness.csv"
    return pd.read_csv(path)
