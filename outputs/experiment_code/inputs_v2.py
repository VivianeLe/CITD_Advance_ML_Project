# EXPERIMENT_INPUTS_V2
# Chỉ bổ sung dữ liệu quá khứ; nhãn flow, các horizon, loss và metric giữ nguyên.
import hashlib as _v2_hashlib
import json as _v2_json
import gc as _v2_gc

_V2_BASE_WINDOWS = getattr(Windows, "_v2_original", Windows)
_V2_EXOG_FRAME = None
_V2_EXOG_FILL = _V2_EXOG_MEAN = _V2_EXOG_STD = None
if DATASET == "PEMS04":
    _v2_npz = next(Path(CFG["DATA_DIR"]).rglob(DATASETS[DATASET]["main_file"]))
    with np.load(_v2_npz) as _v2_archive:
        _v2_exog = _v2_archive["data"][:, :, 1:3].astype("float32")
    if _v2_exog.shape[:2] != raw_df.shape or _v2_exog.shape[2] != 2:
        raise ValueError("Occupancy/speed không khớp dữ liệu flow")
    _v2_exog[~np.isfinite(_v2_exog) | (_v2_exog < 0)] = np.nan
    _V2_EXOG_FRAME = pd.DataFrame(_v2_exog.reshape(len(raw_df), -1), index=raw_df.index)
    # Mọi thống kê chỉ fit trên các timestamp thuộc train.
    _v2_exog_train = _V2_EXOG_FRAME.reindex(train_data.index).to_numpy()
    _V2_EXOG_FILL = np.nanmedian(_v2_exog_train, axis=0).astype("float32")
    _V2_EXOG_FILL = np.nan_to_num(_V2_EXOG_FILL, nan=0.0)
    _v2_exog_train = impute(_v2_exog_train, _V2_EXOG_FILL)
    _V2_EXOG_MEAN = _v2_exog_train.mean(axis=0)
    _V2_EXOG_STD = np.maximum(_v2_exog_train.std(axis=0), 1e-6)
    del _v2_exog, _v2_exog_train


class Windows(_V2_BASE_WINDOWS):
    """Cửa sổ gốc, tùy chọn thêm occupancy/speed cùng cửa sổ quá khứ."""
    def __init__(self, split, scaler, fill, n_past, stride=1):
        super().__init__(split, scaler, fill, n_past, stride)
        self.exog_windows = None
        if CFG.get("USE_EXOG", False):
            if _V2_EXOG_FRAME is None:
                raise ValueError("USE_EXOG hiện chỉ hỗ trợ PeMS04 có đủ 3 kênh")
            if not split.index.isin(_V2_EXOG_FRAME.index).all():
                raise ValueError("Timestamp input bổ sung không khớp tập dữ liệu")
            raw = _V2_EXOG_FRAME.reindex(split.index).to_numpy()
            # Forward-fill riêng trong split, không lấy dữ liệu tương lai.
            z = (impute(raw, _V2_EXOG_FILL) - _V2_EXOG_MEAN) / _V2_EXOG_STD
            z = torch.as_tensor(z.reshape(len(split), N_NODES, 2), dtype=torch.float32, device=DEVICE)
            self.exog_windows = z.unfold(0, n_past, 1)

    def get(self, pos):
        xb, yr, ym = super().get(pos)
        if self.exog_windows is not None:
            extra = self.exog_windows[self.starts[pos]].permute(0, 3, 1, 2)
            xb = torch.cat([xb, extra], dim=-1)
        return xb, yr, ym


Windows._v2_original = _V2_BASE_WINDOWS


def configure_inputs_v2(use_exog):
    """Đặt số kênh trước khi tạo model; không đổi biến đích hoặc horizon."""
    global N_IN
    CFG["USE_EXOG"] = bool(use_exog)
    N_IN = 2 + (3 if CFG["USE_TOD"] else 0) + (1 if CFG["USE_ADJ_FEAT"] else 0) + (2 if use_exog else 0)


configure_inputs_v2(False)
print("Input thử nghiệm sẵn sàng: flow/mask/thời gian; tùy chọn occupancy/speed quá khứ.")
