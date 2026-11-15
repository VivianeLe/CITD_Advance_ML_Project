# %pip install torch

# ===== 0. Môi trường & cấu hình (chạy local trên GPU) =====
import os, time, math, zipfile, random, pickle, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from IPython.display import display

# --- Đường dẫn dự án ---
PROJECT_DIR = Path("/run/media/hoang/HDD-512GB")
if not (PROJECT_DIR / "Do_an_Advanced_ML.ipynb").exists():
    PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data" / "pems04"
OUTPUT_DIR = PROJECT_DIR / "outputs"; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UNIT = "xe/5 phút"

# --- Cấu hình thí nghiệm ---
CFG = dict(
    SEED=42,
    # Dữ liệu PeMS04: npz (T, N, 3) = [flow, occupancy, speed] -> lấy kênh 0 = flow
    DATA_DIR=str(DATA_DIR), HF_REPO="TorchSpatiotemporal/ProcessedDatasets", HF_FILE="traffic/pems04.zip",
    MAIN_FILE="pems04.npz", DIST_CSV="distance.csv", FEATURE_IDX=0, START="2018-01-01 00:00",
    ADJ_THR=0.0,                # distance.csv chỉ có ~340 cạnh nối trực tiếp -> giữ mọi cạnh
    # Cửa sổ: 12 bước vào (60 phút), 3 mốc ra (15/30/60 phút)
    N_PAST=12, HORIZONS=(3, 6, 12), STEP_MIN=5,
    SPLIT=(0.7, 0.1, 0.2),
    MISSING_FILL=-1.0,          # giá trị flow (đơn vị gốc) thay cho NaN / cảm biến hỏng; flow = 0 là giá trị thật
    # Model
    # MODEL="STGCN", SCALER="standard",
    MODEL="STID", SCALER="standard",
    EPOCHS=100, PATIENCE=10, BATCH=64, LR=1e-3, HIDDEN=64, EVAL_BATCH=64, TRAIN_STRIDE=1,
    NODE_EMB=16, MODEL_DEPTH=3, DROPOUT=0.05,  # STID: node/time embeddings + residual MLP
    LR_FACTOR=0.5, LR_PATIENCE=2, MIN_LR=0.0,
    # Masking augmentation khi train: mỗi batch chọn ngẫu nhiên 1 tỉ lệ cảm biến bị xoá.
    # 0.0 = batch sạch (~1/5 số batch) để model vẫn gặp dữ liệu đầy đủ. Đặt () để train hoàn toàn trên dữ liệu sạch.
    MISS_RATES=(0.0, 0.05, 0.10, 0.20, 0.30),
    USE_ADJ_FEAT=True,          # kênh đặc trưng láng giềng còn hoạt động
    RESIDUAL=True,              # dự báo phần chênh so với giá trị cuối cửa sổ
    USE_TOD=True,               # sin/cos giờ trong ngày + cờ cuối tuần
    # Test
    N_REP=3,                    # số lần lặp cho kịch bản mất ngẫu nhiên
    N_REGION_CENTERS=5,         # số tâm khu vực cho kịch bản hỏng theo khu vực
    TEST_SEED=2026,             # seed riêng cho kịch bản test (khác hoàn toàn với ngẫu nhiên lúc train)
    AMP=True, FAST=False,       # FAST=True chỉ để chạy thử pipeline
)
if CFG["FAST"]:
    CFG.update(EPOCHS=2, TRAIN_STRIDE=12, N_REP=1)

# --- Thiết bị ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_AMP = bool(CFG["AMP"] and DEVICE.type == "cuda")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(DEVICE)
HORIZONS = list(CFG["HORIZONS"]); H = len(HORIZONS)
HOR_MIN = [h * CFG["STEP_MIN"] for h in HORIZONS]   # 15, 30, 60 phút

def set_seed(s):
    """Đặt seed cho random / numpy / torch để kết quả tái lập."""
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed(CFG["SEED"])

print(f"device: {DEVICE} | torch {torch.__version__} | AMP {USE_AMP}")
if DEVICE.type == "cuda":
    props = torch.cuda.get_device_properties(DEVICE)
    print(f"GPU: {props.name} | VRAM {props.total_memory / 1e9:.1f} GB")
else:
    print("CẢNH BÁO: CUDA không khả dụng, train trên CPU sẽ rất chậm.")

def _sensor_key(v):
    """Chuẩn hoá ID cảm biến về string, bỏ đuôi .0 do pandas tạo ra."""
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s

def _distance_from_csv(csv_path, sensor_ids):
    """Đọc distance.csv (from,to,cost) và dựng ma trận khoảng cách (N,N); cặp không nối = inf."""
    d = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in d.columns}
    index = {_sensor_key(s): i for i, s in enumerate(sensor_ids)}
    D = np.full((len(index), len(index)), np.inf, dtype=np.float32)
    for a, b, c in zip(d[cols["from"]], d[cols["to"]], d[cols["cost"]]):
        ia, ib = index.get(_sensor_key(a)), index.get(_sensor_key(b))
        if ia is not None and ib is not None:
            D[ia, ib] = np.float32(c)
    D = np.minimum(D, D.T)          # đồ thị đối xứng
    np.fill_diagonal(D, 0.0)
    print(f"[*] Ma trận khoảng cách: {Path(csv_path).name} | {np.isfinite(D).sum():,} / {D.size:,} cặp có khoảng cách")
    return D

def load_data(cfg):
    """Đọc PeMS04 local (tải từ Hugging Face nếu chưa có). Trả (df flow (T,N), ma trận khoảng cách (N,N))."""
    data_dir = Path(cfg["DATA_DIR"]); data_dir.mkdir(parents=True, exist_ok=True)
    found = sorted(data_dir.rglob(cfg["MAIN_FILE"]))
    if not found:
        from huggingface_hub import hf_hub_download     # chỉ cần khi chưa có dữ liệu local
        print(f"[*] Đang tải {cfg['HF_REPO']}/{cfg['HF_FILE']} ...")
        zp = hf_hub_download(repo_id=cfg["HF_REPO"], filename=cfg["HF_FILE"], repo_type="dataset")
        with zipfile.ZipFile(zp) as z:
            z.extractall(data_dir)
        found = sorted(data_dir.rglob(cfg["MAIN_FILE"]))
    print(f"[*] Dùng dữ liệu: {found[0]}")
    with np.load(found[0]) as z:
        arr = z["data"] if "data" in z.files else z[z.files[0]]
    print(f"[*] shape {arr.shape} (T, N, F) -> lấy kênh {cfg['FEATURE_IDX']} = flow")
    if arr.ndim == 3:
        arr = arr[..., cfg["FEATURE_IDX"]]
    # npz không có mốc thời gian -> dựng index 5 phút từ ngày bắt đầu.
    idx = pd.date_range(cfg["START"], periods=arr.shape[0], freq=f"{cfg['STEP_MIN']}min")
    df = pd.DataFrame(arr.astype("float32"), index=idx, columns=[str(i) for i in range(arr.shape[1])])
    csvs = sorted(data_dir.rglob(cfg["DIST_CSV"]))
    if not csvs:
        raise FileNotFoundError(f"Không thấy {cfg['DIST_CSV']} trong {data_dir}")
    return df, _distance_from_csv(csvs[0], df.columns)

raw_df, DIST = load_data(CFG)
print("shape:", raw_df.shape)
display(raw_df.iloc[:3, :6])

def clean(df, cfg):
    """Làm sạch: bỏ trùng, điền lưới thời gian, inf/âm -> NaN. Flow = 0 được giữ (là giá trị thật)."""
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq=f"{cfg['STEP_MIN']}min")
    df = df.reindex(full).replace([np.inf, -np.inf], np.nan)
    df[df < 0] = np.nan           # flow âm là lỗi đo -> thiếu
    return df.astype("float32")

df = clean(raw_df, CFG)
VALUES = df.to_numpy()                          # (T, N) có NaN
T_ALL, N_NODES = VALUES.shape
print(f"sau làm sạch: T={T_ALL}, N={N_NODES}, NaN={np.isnan(VALUES).mean()*100:.3f}% | "
      f"flow = 0 (giữ lại)={(VALUES == 0).mean()*100:.3f}%")

import matplotlib.pyplot as plt

# --- Thống kê mô tả trên toàn bộ giá trị quan sát (bỏ NaN) ---
obs = VALUES[~np.isnan(VALUES)]
desc = pd.Series({
    "count": obs.size, "mean": obs.mean(), "median": np.median(obs), "std": obs.std(),
    "min": obs.min(), "p5": np.percentile(obs, 5), "p95": np.percentile(obs, 95), "max": obs.max(),
    "zero_pct": (obs == 0).mean() * 100, "missing_pct": np.isnan(VALUES).mean() * 100,
}, name=f"flow ({UNIT})").to_frame()
display(desc.round(3))

# --- Thống kê theo từng cảm biến, rồi mô tả phân bố của các thống kê đó qua 307 cảm biến ---
per_sensor = pd.DataFrame({
    "mean": np.nanmean(VALUES, 0), "median": np.nanmedian(VALUES, 0),
    "min": np.nanmin(VALUES, 0), "max": np.nanmax(VALUES, 0),
    "missing_pct": np.isnan(VALUES).mean(0) * 100,
}, index=df.columns)
display(per_sensor.describe().T.round(3).rename_axis("thống kê theo cảm biến"))

# --- Lưu lượng trung bình theo giờ trong ngày: trung bình mọi cảm biến tại từng mốc, rồi gộp theo giờ ---
prof = pd.DataFrame({"flow": np.nanmean(VALUES, 1)}, index=df.index)
prof["hour"] = prof.index.hour
prof["day_type"] = np.where(prof.index.dayofweek >= 5, "cuối tuần", "ngày thường")
HOURLY = pd.concat([
    prof.groupby("hour").flow.mean().rename("tất cả"),
    prof.pivot_table(index="hour", columns="day_type", values="flow", aggfunc="mean"),
], axis=1)
HOURLY.index.name = "giờ"
display(HOURLY.round(1).T)
pk = HOURLY["tất cả"]
print(f"Giờ cao điểm: {pk.idxmax()}h ({pk.max():.1f} {UNIT}) | giờ thấp nhất: {pk.idxmin()}h ({pk.min():.1f} {UNIT})")

# --- Biểu đồ ---
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
ax[0].hist(obs, bins=80, color="steelblue")
ax[0].axvline(obs.mean(), c="r", ls="--", label=f"mean {obs.mean():.1f}")
ax[0].axvline(np.median(obs), c="g", ls="--", label=f"median {np.median(obs):.1f}")
ax[0].set_title("Phân phối flow (toàn bộ giá trị)"); ax[0].set_xlabel(f"flow ({UNIT})"); ax[0].set_ylabel("số ô"); ax[0].legend()
ax[1].hist(per_sensor["mean"], bins=40, color="darkorange")
ax[1].set_title("Mean flow theo từng cảm biến"); ax[1].set_xlabel(f"mean flow ({UNIT})"); ax[1].set_ylabel("số cảm biến")
for col, style in (("tất cả", "-o"), ("ngày thường", "--"), ("cuối tuần", "--")):
    if col in HOURLY: ax[2].plot(HOURLY.index, HOURLY[col], style, label=col)
ax[2].set_title("Lưu lượng trung bình theo giờ trong ngày"); ax[2].set_xlabel("giờ"); ax[2].set_ylabel(f"flow ({UNIT})")
ax[2].set_xticks(range(0, 24, 2)); ax[2].grid(alpha=.3); ax[2].legend()
plt.tight_layout(); plt.show()

class SplitData:
    """Một tập con theo thời gian, giữ nguyên NaN để imputer xử lý sau."""
    def __init__(self, values, index, name):
        self.values, self.index, self.name = values, index, name   # values: (T,N) float32, có NaN
    def __len__(self): return len(self.values)

def make_split(values, index, ratios):
    """Cắt liền mạch theo thời gian thành train / val / test.

    Cắt trước rồi mới tạo cửa sổ trong từng tập để không có cửa sổ
    bắc qua ranh giới (tránh rò rỉ giữa các tập).
    """
    # a, b là ranh giới: train [0,a), val [a,b), test [b,T).
    T = len(values); a = int(T * ratios[0]); b = a + int(T * ratios[1])
    parts = [("train", 0, a), ("val", a, b), ("test", b, T)]
    return {n: SplitData(values[s:e].copy(), index[s:e], n) for n, s, e in parts}

# Tạo 3 tập và kiểm tra thứ tự thời gian.
SPLITS = make_split(VALUES, df.index, CFG["SPLIT"])
train_data, val_data, test_data = SPLITS["train"], SPLITS["val"], SPLITS["test"]
for k, s in SPLITS.items():
    print(f"{k:5s} {len(s):6d} bước | {s.index[0]} -> {s.index[-1]} | NaN {np.isnan(s.values).mean()*100:.2f}%")
assert train_data.index[-1] < val_data.index[0] < val_data.index[-1] < test_data.index[0]

from scipy.sparse.csgraph import shortest_path

def norm_adj(A):
    """Chuẩn hoá kiểu DCRNN: A_hat = D^{-1/2} (A + I) D^{-1/2}."""
    At = A + np.eye(len(A)); d = At.sum(1); Dm = 1.0 / np.sqrt(d)
    return (Dm[:, None] * At * Dm[None, :]).astype(np.float32)

def build_adjacency(dist, thr=0.0):
    """Ma trận kề Gaussian từ khoảng cách đường đi.

    Trả về: A (N,N) thô, A_hat đã chuẩn hoá cho STGCN, PROX = khoảng cách đường đi ngắn nhất giữa mọi cặp.
    """
    d = dist.astype("float64")
    fin = np.isfinite(d) & (d > 0)
    sigma = d[fin].std()
    with np.errstate(all="ignore"):
        A = np.exp(-(d / sigma) ** 2)
    A[~np.isfinite(d)] = 0; A[A < thr] = 0; np.fill_diagonal(A, 0)
    A = np.maximum(A, A.T)
    # Khoảng cách ngắn nhất qua đồ thị (Dijkstra): đồ thị PeMS04 rất thưa nên
    # chỉ dùng cạnh trực tiếp không đủ để xác định "khu vực" gồm hàng chục cảm biến.
    prox = shortest_path(np.where(fin, d, 0.0), method="D", directed=False)
    prox[~np.isfinite(prox)] = 1e9          # cặp không liên thông
    print(f"adjacency: cạnh {(A > 0).sum()} | độ thưa {(A > 0).mean():.4f} | "
          f"cặp liên thông {(prox < 1e9).mean()*100:.1f}%")
    return A, norm_adj(A), prox

ADJ_RAW, A_HAT, PROX = build_adjacency(DIST, thr=CFG["ADJ_THR"])
A_HAT_T = torch.tensor(A_HAT, device=DEVICE)

# ---------- Imputer ----------
def fit_imputer(train_values):
    """Giá trị thay cho NaN của từng cảm biến: hằng số CFG['MISSING_FILL'] (= -1, ngoài miền flow >= 0)."""
    return np.full(train_values.shape[1], CFG["MISSING_FILL"], dtype="float32")

def impute(values, fill):
    """Thay NaN bằng `fill` (-1) để model phân biệt ô thiếu với flow thật (kể cả flow = 0)."""
    arr = np.array(values, dtype="float32", copy=True)
    nan = np.isnan(arr)
    arr[nan] = np.broadcast_to(fill, arr.shape)[nan]
    return arr.astype("float32")

# ---------- StandardScaler (theo từng cảm biến, fit trên train) ----------
class StandardScaler:
    """Chuẩn hoá (x - mean) / std theo từng cảm biến; làm việc trên torch tensor."""
    name = "standard"
    def _p(self, v, x): return torch.as_tensor(v, device=x.device, dtype=x.dtype)
    def fit(self, x): self.mu = np.nanmean(x, 0); self.sd = np.nanstd(x, 0) + 1e-6; return self
    def fwd(self, x): return (x - self._p(self.mu, x)) / self._p(self.sd, x)
    def inv(self, z): return z * self._p(self.sd, z) + self._p(self.mu, z)
    def transform(self, x): return self.fwd(torch.as_tensor(x, dtype=torch.float32))
    def transform_bnh(self, y): return self.fwd(y.transpose(1, 2)).transpose(1, 2)      # (B,N,H)
    def inverse_bnh(self, z): return self.inv(z.transpose(1, 2)).transpose(1, 2)        # (B,N,H)

SCALERS = {"standard": StandardScaler}

# ---------- Đặc trưng thời gian ----------
def time_features(index):
    """sin/cos giờ trong ngày + cờ cuối tuần, shape (T,3)."""
    ang = 2 * np.pi * (index.hour.values * 60 + index.minute.values) / 1440.0
    wk = (index.dayofweek.values >= 5).astype("float32")
    return np.stack([np.sin(ang), np.cos(ang), wk], 1).astype("float32")

# ---------- Cửa sổ trượt trong từng tập ----------
class Windows:
    """Cửa sổ trượt của một tập. get(pos) trả x (B,n_past,N,C), y_raw (B,N,H), y_mask (B,N,H)."""
    def __init__(self, split, scaler, fill, n_past, stride=1):
        raw = split.values
        observed = ~np.isnan(raw)
        z = scaler.transform(impute(raw, fill))
        # Giá trị đã scale tương ứng flow = -1 của từng cảm biến, dùng khi giả lập cảm biến hỏng.
        self.miss_z = scaler.transform(np.asarray(fill, dtype="float32")[None, :])[0].to(DEVICE)   # (N,)
        # Kênh 0: giá trị đã scale (NaN -> -1 rồi scale); kênh 1: mask 1 = có giá trị, 0 = NaN.
        x = torch.stack([z, torch.as_tensor(observed, dtype=torch.float32)], -1)               # (T,N,2)
        if CFG["USE_TOD"]:
            tf = torch.as_tensor(time_features(split.index))[:, None, :].expand(-1, x.shape[1], -1)
            x = torch.cat([x, tf], -1)                                                         # (T,N,5)
        x = x.to(DEVICE).contiguous()
        self.n_past = n_past
        self.xw = x.unfold(0, n_past, 1)                                   # (T-n+1, N, C, n_past) view
        # Nhãn thô (đơn vị gốc); NaN -> 0 nhưng mask = 0 nên không vào loss/metric.
        self.y_raw = torch.as_tensor(np.nan_to_num(raw, nan=0.0), dtype=torch.float32, device=DEVICE)
        self.y_mask = torch.as_tensor(observed, device=DEVICE)
        self.h = torch.tensor(HORIZONS, device=DEVICE)
        last = len(raw) - n_past - max(HORIZONS)
        assert last >= 0, "tập quá ngắn"
        self.starts = torch.arange(0, last + 1, stride, device=DEVICE)
    def __len__(self): return len(self.starts)
    def get(self, pos):
        s = self.starts[pos]
        xb = self.xw[s].permute(0, 3, 1, 2)                                  # (B,n,N,C)
        rows = s[:, None] + self.n_past - 1 + self.h[None, :]                # (B,H)
        yr = self.y_raw[rows].permute(0, 2, 1)                               # (B,N,H)
        ym = self.y_mask[rows].permute(0, 2, 1)
        return xb, yr, ym

def batches(n, bs, shuffle):
    """Sinh chỉ số batch (shuffle bằng random permutation trên GPU)."""
    idx = torch.randperm(n, device=DEVICE) if shuffle else torch.arange(n, device=DEVICE)
    for i in range(0, n, bs): yield idx[i:i + bs]

# Số kênh input: giá trị + mask (+3 thời gian) (+1 đặc trưng láng giềng).
N_IN = 2 + (3 if CFG["USE_TOD"] else 0) + (1 if CFG["USE_ADJ_FEAT"] else 0)

# Ma trận kề thô (đường chéo = 0) cho đặc trưng láng giềng.
_A_nb = ADJ_RAW.astype("float32").copy(); np.fill_diagonal(_A_nb, 0.0)
A_NB_T = torch.tensor(_A_nb, device=DEVICE)

def add_adj_feature(xb, A_nb):
    """Thêm 1 kênh: trung bình có trọng số A của các láng giềng ĐANG HOẠT ĐỘNG (mask = 1).

    Gọi SAU inject_fault để cảm biến hỏng không đóng góp. Không có láng giềng hoạt động -> dùng chính giá trị nút.
    """
    if not CFG["USE_ADJ_FEAT"]: return xb
    v, m = xb[..., 0].float(), xb[..., 1].float()
    num = torch.einsum("nm,btm->btn", A_nb, v * m)
    den = torch.einsum("nm,btm->btn", A_nb, m)
    nb = torch.where(den > 1e-6, num / den.clamp_min(1e-6), v)
    return torch.cat([xb, nb.unsqueeze(-1).to(xb.dtype)], -1)

def _zero_(m):
    """Khởi tạo trọng số/bias bằng 0."""
    nn.init.zeros_(m.weight)
    if m.bias is not None: nn.init.zeros_(m.bias)
    return m

class BaseNet(nn.Module):
    """core() sinh phần chênh; forward() cộng giá trị cuối cửa sổ nếu RESIDUAL."""
    def core(self, x): raise NotImplementedError
    def forward(self, x):
        out = self.core(x)
        if CFG["RESIDUAL"]: out = out + x[:, -1, :, 0:1]
        return out

class TemporalGLU(nn.Module):
    """Conv theo trục thời gian + gated linear unit."""
    def __init__(self, cin, cout, k=3):
        super().__init__(); self.conv = nn.Conv2d(cin, 2 * cout, (k, 1), padding=(k // 2, 0))
    def forward(self, x):                                  # x: (B,C,T,N)
        p, q = self.conv(x).chunk(2, 1)
        return p * torch.sigmoid(q)

class ChanNorm(nn.Module):
    """LayerNorm theo kênh cho từng (thời điểm, cảm biến)."""
    def __init__(self, c): super().__init__(); self.ln = nn.LayerNorm(c)
    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class STBlock(nn.Module):
    """Temporal GLU -> self + láng giềng -> temporal GLU -> skip -> norm."""
    def __init__(self, cin, cmid, cout, A):
        super().__init__()
        self.A = A
        self.t1 = TemporalGLU(cin, cmid)
        self.slf = nn.Conv2d(cmid, cmid, 1); self.nbr = _zero_(nn.Conv2d(cmid, cmid, 1))
        self.t2 = TemporalGLU(cmid, cout); self.norm = ChanNorm(cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
    def forward(self, x):
        h = self.t1(x)
        agg = torch.einsum("nm,bctm->bctn", self.A, h)
        h = F.relu(self.slf(h) + self.nbr(agg))
        return self.norm(self.t2(h) + self.skip(x))

class STGCN(BaseNet):
    """Hai khối STBlock; head dự báo H horizon (15/30/60 phút) cho từng cảm biến."""
    def __init__(self, n_nodes, n_in, n_out, hidden, n_past, A):
        super().__init__()
        self.register_buffer("A", A)
        self.b1 = STBlock(n_in, hidden, hidden, self.A); self.b2 = STBlock(hidden, hidden, hidden, self.A)
        self.head = _zero_(nn.Linear(hidden * n_past, n_out))
    def core(self, x):
        h = self.b2(self.b1(x.permute(0, 3, 1, 2)))        # (B,C,T,N)
        B, C, T, N = h.shape
        return self.head(h.permute(0, 3, 1, 2).reshape(B, N, C * T))

class _STIDContext(nn.Module):
    """Mã hóa cửa sổ, danh tính cảm biến và thời gian ở cuối cửa sổ."""
    def __init__(self, n_nodes, n_in, hidden, n_past, node_emb=None, use_time=None):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.emb_dim = int(CFG.get("NODE_EMB", 16) if node_emb is None else node_emb)
        self.use_time = bool(CFG.get("USE_TOD", True) if use_time is None else use_time)
        if self.emb_dim < 1:
            raise ValueError("NODE_EMB phải là số nguyên dương")
        if self.use_time and n_in < 5:
            raise ValueError("Embedding thời gian cần các kênh sin/cos/cuối tuần ở vị trí 2/3/4")
        self.past = nn.Linear(n_past * n_in, hidden)
        self.node = nn.Embedding(n_nodes, self.emb_dim)
        self.tod = nn.Embedding(288, self.emb_dim)
        self.weekend = nn.Embedding(2, self.emb_dim)
        for embedding in (self.node, self.tod, self.weekend):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        self.width = hidden + 3 * self.emb_dim

    def forward(self, x):
        batch, steps, nodes, channels = x.shape
        if nodes != self.n_nodes:
            raise ValueError("Số cảm biến của input không khớp mô hình")
        past = F.gelu(self.past(x.permute(0, 2, 1, 3).reshape(batch, nodes, steps * channels)))
        node = self.node.weight.unsqueeze(0).expand(batch, -1, -1)
        if self.use_time:
            # Tái dựng slot 5 phút từ sin/cos đã có; không đọc thời điểm tương lai.
            sin_t = x[:, -1, 0, 2].float()
            cos_t = x[:, -1, 0, 3].float()
            angle = torch.remainder(torch.atan2(sin_t, cos_t), 2 * math.pi)
            slots = torch.round(angle * (288 / (2 * math.pi))).long().remainder(288)
            weekend = x[:, -1, 0, 4].float().round().long().clamp(0, 1)
            time = self.tod(slots).unsqueeze(1).expand(-1, nodes, -1)
            week = self.weekend(weekend).unsqueeze(1).expand(-1, nodes, -1)
        else:
            time = past.new_zeros(batch, nodes, self.emb_dim)
            week = past.new_zeros(batch, nodes, self.emb_dim)
        return torch.cat([past, node.to(past.dtype), time.to(past.dtype), week.to(past.dtype)], dim=-1)


class _STIDResidualMLP(nn.Module):
    """Khối MLP có chuẩn hóa, dropout và đường residual."""
    def __init__(self, width, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.first = nn.Linear(width, 2 * width)
        self.second = nn.Linear(2 * width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.gelu(self.first(self.norm(x))))
        return x + self.drop(self.second(h))


class STIDNet(BaseNet):
    """MLP kết hợp embedding cảm biến và thời gian;head dự báo H horizon (15/30/60 phút) cho từng cảm biến."""
    def __init__(self, n_nodes, n_in, n_out, hidden, n_past, A, model_options=None):
        super().__init__()
        options = dict(model_options or {})
        self.register_buffer("A", A.detach().clone())
        node_emb = int(options.get("node_emb", CFG.get("NODE_EMB", 16)))
        use_time = bool(options.get("use_time", CFG.get("USE_TOD", True)))
        self.residual = bool(options.get("residual", CFG.get("RESIDUAL", True)))
        depth = int(options.get("depth", CFG.get("MODEL_DEPTH", 3)))
        dropout = float(options.get("dropout", CFG.get("DROPOUT", 0.05)))
        if depth < 1 or not 0 <= dropout < 1:
            raise ValueError("MODEL_DEPTH >= 1 và 0 <= DROPOUT < 1")
        self.context = _STIDContext(n_nodes, n_in, hidden, n_past, node_emb=node_emb, use_time=use_time)
        self.blocks = nn.Sequential(*[_STIDResidualMLP(self.context.width, dropout) for _ in range(depth)])
        self.head = _zero_(nn.Linear(self.context.width, n_out))
        self.use_adj = True
        self.model_options = dict(hidden=hidden, n_past=n_past, node_emb=self.context.emb_dim,
                                  tod_slots=288, weekend_slots=2, use_time=self.context.use_time,
                                  depth=depth, dropout=dropout, use_adj=True, residual=self.residual)

    def forward(self, x):
        out = self.core(x)
        if self.residual:
            out = out + x[:, -1, :, 0:1]
        return out

    def core(self, x):
        return self.head(self.blocks(self.context(x)))


MODELS = {"STGCN": STGCN, "STID": STIDNet}

def build_model(name, n_nodes, n_past, hidden, use_adj=True):
    """Tạo model; use_adj=False thay A bằng I (không dùng thông tin không gian)."""
    A = A_HAT_T if use_adj else torch.eye(n_nodes, device=DEVICE)
    m = MODELS[name](n_nodes, N_IN, H, hidden, n_past, A)
    m.A_nb = A_NB_T if use_adj else torch.zeros(n_nodes, n_nodes, device=DEVICE)
    return m.to(DEVICE)

def inject_fault(xb, L, k=None, sensors=None, gen=None, fill_z=None):
    """Giả lập cảm biến hỏng trong L bước cuối của cửa sổ.

    xb: (B,T,N,C). k = số cảm biến hỏng mỗi mẫu (chọn ngẫu nhiên);
    sensors: bool (N,) cố định, dùng cho kịch bản hỏng theo khu vực.
    fill_z: (N,) giá trị đã scale của flow = -1 (Windows.miss_z). Giá trị bị mất được thay bằng fill_z
    và mask = 0, giống hệt cách xử lý NaN thật trong dữ liệu.
    """
    B, T, N, _ = xb.shape
    assert T > L, "n_past phải > số bước mất"
    # hit[b,n] = True nghĩa cảm biến n hỏng ở mẫu b.
    if sensors is not None:
        hit = sensors.view(1, N).expand(B, N)
    else:
        if not k: return xb
        # Mỗi mẫu chọn ngẫu nhiên k cảm biến; generator giúp tái lập kịch bản.
        idx = torch.rand(B, N, device=xb.device, generator=gen).topk(k, dim=1).indices
        hit = torch.zeros(B, N, dtype=torch.bool, device=xb.device).scatter_(1, idx, True)
    out = xb.clone()
    tail_v, tail_m = xb[:, T - L:, :, 0], xb[:, T - L:, :, 1]
    assert fill_z is not None, "cần fill_z (Windows.miss_z) để thay giá trị bị mất bằng -1"
    miss_v = fill_z.to(xb.dtype).view(1, 1, N).expand(B, L, N)
    h3 = hit.unsqueeze(1)
    # Chỉ sửa kênh giá trị và mask; các kênh thời gian giữ nguyên.
    out[:, T - L:, :, 0] = torch.where(h3, miss_v, tail_v)
    out[:, T - L:, :, 1] = torch.where(h3, torch.zeros_like(tail_m), tail_m)
    return out

class MetricAcc:
    """Cộng dồn tử số/mẫu số để tính MAE/RMSE/MAPE theo từng horizon; bỏ qua nhãn thiếu."""
    def __init__(self):
        # Cộng dồn: SAE = Σ|e|, SSE = Σe², SAPE = Σ|e|/true; n và nape là số phần tử hợp lệ.
        self.sae = torch.zeros(H, dtype=torch.float64); self.sse = torch.zeros(H, dtype=torch.float64)
        self.sape = torch.zeros(H, dtype=torch.float64); self.n = torch.zeros(H, dtype=torch.float64); self.nape = torch.zeros(H, dtype=torch.float64)
    def add(self, pred, true, mask):
        m = mask.double(); e = (pred - true).double()
        # mask = 1 chỉ tại nhãn thật; nhãn NA không được tính vào metric.
        self.sae += (e.abs() * m).sum((0, 1)).cpu(); self.sse += ((e ** 2) * m).sum((0, 1)).cpu(); self.n += m.sum((0, 1)).cpu()
        # MAPE bỏ qua nhãn <= 1 (flow gần 0 ban đêm) để tránh chia cho số quá nhỏ.
        mm = (mask & (true > 1.0)).double()
        self.sape += ((e.abs() / true.clamp_min(1.0).double()) * mm).sum((0, 1)).cpu(); self.nape += mm.sum((0, 1)).cpu()
    def result(self):
        # MAE/RMSE/MAPE cho từng horizon, rồi lấy trung bình 3 horizon cho *_avg.
        mae = (self.sae / self.n).numpy(); rmse = (self.sse / self.n).sqrt().numpy(); mape = (100 * self.sape / self.nape).numpy()
        out = {}
        for i, mn in enumerate(HOR_MIN):
            out[f"MAE_{mn}"], out[f"RMSE_{mn}"], out[f"MAPE_{mn}"] = float(mae[i]), float(rmse[i]), float(mape[i])
        out["MAE_avg"], out["RMSE_avg"], out["MAPE_avg"] = float(mae.mean()), float(rmse.mean()), float(mape.mean())
        return out

@torch.no_grad()          # đánh giá không cần gradient
def evaluate(model, win, scaler, fault=None, bs=None):
    """Chạy model trên toàn bộ Windows, có thể inject fault trước khi dự báo."""
    bs = int(bs if bs is not None else getattr(model, "eval_bs", CFG.get("EVAL_BATCH", 64)))
    if bs <= 0: raise ValueError("Evaluation batch size must be positive")
    model.eval(); acc = MetricAcc()
    for pos in batches(len(win), bs, False):
        xb, yr, ym = win.get(pos)
        if fault is not None: xb = inject_fault(xb, fill_z=win.miss_z, **fault)
        xb = add_adj_feature(xb, model.A_nb)              # kênh ma trận kề tính sau khi hỏng
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            pred = model(xb)
        # Đưa dự báo về đơn vị gốc (UNIT) trước khi tính metric.
        pred = scaler.inverse_bnh(pred.float()).clamp_min(0)
        if not torch.isfinite(pred).all():
            raise FloatingPointError("Non-finite predictions during evaluation")
        acc.add(pred, yr, ym)
    return acc.result()

RESULTS = []      # Mỗi lần train_model ghi một cấu hình; test có thể tắt khi tuning.
ROBUST = []       # Mỗi lần test_model(log=True) ghi một kịch bản mất dữ liệu.

def results_df(): return pd.DataFrame(RESULTS)
def robust_df(): return pd.DataFrame(ROBUST)

def train_model(train_data, val_data, test_data, scaler_name, model_name, n_past,
                mask_aug=0.0, use_adj=True, epochs=None, seed=None, tag="", verbose=True,
                miss_rates=None, hidden=None, lr=None, batch_size=None, eval_bs=None,
                monitor="MAE_avg", evaluate_test=True):
    """Train trên train, chọn checkpoint theo validation; test chỉ chạy khi evaluate_test=True.

    MAE/RMSE có đơn vị gốc; MAPE có đơn vị %. Loss giữ nguyên MAE chuẩn hóa có mask.
    best_train_loss là loss của epoch có checkpoint validation tốt nhất.
    """
    if monitor not in ("MAE_avg", "MAPE_avg"):
        raise ValueError("monitor must be MAE_avg or MAPE_avg")
    seed = CFG["SEED"] if seed is None else int(seed)
    epochs = CFG["EPOCHS"] if epochs is None else int(epochs)
    hidden = CFG["HIDDEN"] if hidden is None else int(hidden)
    lr = CFG["LR"] if lr is None else float(lr)
    batch_size = CFG["BATCH"] if batch_size is None else int(batch_size)
    eval_bs = CFG.get("EVAL_BATCH", 64) if eval_bs is None else int(eval_bs)
    if min(epochs, hidden, batch_size, eval_bs) <= 0 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("epochs, hidden, lr and batch sizes must be positive and finite")
    set_seed(seed); t0 = time.time()
    # Fit imputer/scaler chỉ trên train; giữ nguyên công thức loss trước đây.
    scaler = SCALERS[scaler_name]().fit(train_data.values)
    fill = fit_imputer(train_data.values)
    ls = float(scaler.transform(impute(train_data.values, fill)).std())
    if not np.isfinite(ls) or ls <= 0:
        raise ValueError("Invalid scaled training standard deviation")
    tr = Windows(train_data, scaler, fill, n_past, CFG["TRAIN_STRIDE"])
    va = Windows(val_data, scaler, fill, n_past, 1)
    # Tuning không tạo cửa sổ test và không đánh giá test.
    te = Windows(test_data, scaler, fill, n_past, 1) if evaluate_test else None
    model = build_model(model_name, N_NODES, n_past, hidden, use_adj)
    model.eval_bs = eval_bs
    model.train_config = dict(model=model_name, scaler=scaler_name, n_past=n_past,
                              hidden=hidden, lr=lr, batch_size=batch_size, eval_bs=eval_bs,
                              epochs=epochs, seed=seed, train_stride=CFG["TRAIN_STRIDE"],
                              mask_aug=mask_aug, miss_rates=tuple(miss_rates) if miss_rates else (),
                              use_adj=use_adj, adj_feat=CFG["USE_ADJ_FEAT"],
                              monitor=monitor, evaluate_test=bool(evaluate_test))
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = int(sum(p.numel() for p in params))
    hist = {"train_loss": [], "val_MAE": [], "val_MAPE": [], "val_RMSE": [], "lr": []}
    best_score, best_state, best_epoch = float("inf"), None, 0
    best_val_metrics, best_train_loss = None, None
    if params:
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=CFG.get("LR_FACTOR", 0.5),
                                                        patience=CFG.get("LR_PATIENCE", 2), min_lr=CFG.get("MIN_LR", 0.0))
        amp_scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
        bad = 0
        for ep in range(1, epochs + 1):
            model.train(); tot, cnt = 0.0, 0
            epoch_lr = float(opt.param_groups[0]["lr"])
            for pos in batches(len(tr), batch_size, True):
                xb, yr, ym = tr.get(pos)
                if miss_rates:
                    rate = float(np.random.choice(miss_rates))
                    k = int(round(rate * N_NODES))
                    if k > 0: xb = inject_fault(xb, L=int(np.random.choice([3, 6])), k=k, fill_z=tr.miss_z)
                elif mask_aug > 0:
                    k = int(round(np.random.rand() * mask_aug * N_NODES))
                    if k > 0: xb = inject_fault(xb, L=int(np.random.choice([3, 6])), k=k, fill_z=tr.miss_z)
                xb = add_adj_feature(xb, model.A_nb)
                yz = scaler.transform_bnh(yr)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                    pred = model(xb)
                loss = (F.l1_loss(pred.float() / ls, yz / ls, reduction="none") * ym).sum() / ym.sum().clamp_min(1)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch {ep}")
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, 5.0)
                amp_scaler.step(opt); amp_scaler.update()
                valid_count = int(ym.sum().item())
                tot += float(loss.detach().item()) * valid_count; cnt += valid_count
            if cnt == 0:
                raise ValueError("Training windows have no observed targets")
            epoch_loss = tot / cnt
            val_metrics = evaluate(model, va, scaler)
            if not all(np.isfinite(v) for v in val_metrics.values()):
                raise FloatingPointError(f"Non-finite validation metrics at epoch {ep}")
            score = val_metrics[monitor]
            hist["train_loss"].append(epoch_loss)
            for metric in ("MAE", "MAPE", "RMSE"):
                hist[f"val_{metric}"].append(val_metrics[f"{metric}_avg"])
            hist["lr"].append(epoch_lr)
            sched.step(score)
            if verbose:
                print(f"  [{model_name}/{scaler_name}{tag}] ep {ep:03d} loss {epoch_loss:.4f} "
                      f"val_MAE {val_metrics['MAE_avg']:.3f} val_MAPE {val_metrics['MAPE_avg']:.3f}% "
                      f"val_RMSE {val_metrics['RMSE_avg']:.3f} lr {epoch_lr:.2g}", flush=True)
            if score < best_score:
                best_score = score
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch, best_train_loss = ep, epoch_loss
                best_val_metrics = val_metrics.copy(); bad = 0
            else:
                bad += 1
                if bad >= CFG["PATIENCE"]: break
        if best_state is None:
            raise RuntimeError("Training produced no valid validation checkpoint")
        model.load_state_dict(best_state)
    else:
        best_val_metrics = evaluate(model, va, scaler)
        if not all(np.isfinite(v) for v in best_val_metrics.values()):
            raise FloatingPointError("Non-finite baseline validation metrics")
        best_score = best_val_metrics[monitor]
    test_res = evaluate(model, te, scaler) if evaluate_test else {}
    train_res = {"best_val_MAE": best_val_metrics["MAE_avg"],
                 "best_val_MAPE": best_val_metrics["MAPE_avg"],
                 "best_val_RMSE": best_val_metrics["RMSE_avg"],
                 "best_train_loss": best_train_loss, "best_val_metrics": best_val_metrics,
                 "best_epoch": best_epoch, "epochs_run": len(hist["train_loss"]),
                 "n_params": n_params, "train_time_s": time.time() - t0,
                 "monitor": monitor, "best_monitor": best_score, "history": hist}
    model.scaler, model.fill, model.n_past, model.name_ = scaler, fill, n_past, model_name
    RESULTS.append(dict(model.train_config, miss_rates=str(tuple(miss_rates)) if miss_rates else "",
                        tag=tag, n_params=n_params, best_epoch=best_epoch,
                        train_time_s=round(train_res["train_time_s"], 1),
                        val_MAE=best_val_metrics["MAE_avg"], val_MAPE=best_val_metrics["MAPE_avg"],
                        val_RMSE=best_val_metrics["RMSE_avg"], train_loss=best_train_loss,
                        **{f"test_{k}": v for k, v in test_res.items()}))
    return model, train_res, test_res

def get_region(center, k):
    """Lấy k cảm biến gần tâm `center` nhất theo PROX (đường đi hoặc tương quan)."""
    return np.argsort(PROX[center])[:k]

def _avg(dicts):
    """Trung bình cộng từng metric trong danh sách kết quả (khi chạy lặp nhiều lần)."""
    return {k: float(np.mean([d[k] for d in dicts])) for k in dicts[0]}

def test_model(test_data, scaler, model, miss_pct=0, miss_minutes=0, region=None, n_rep=None,
               seed=0, label=None, scenario=None, log=True):
    """Đánh giá một model dưới kịch bản mất cảm biến.

    miss_pct     : % cảm biến bị mất (chọn ngẫu nhiên mỗi mẫu)
    miss_minutes : số phút mất ở cuối cửa sổ (15/30 phút)
    region       : danh sách chỉ số cảm biến cố định (hỏng theo khu vực)
    n_rep        : số lần lặp ngẫu nhiên, lấy trung bình
    seed         : seed gốc cho các lần lặp ngẫu nhiên
    label        : tên model ghi vào ROBUST (mặc định lấy model.name_)
    scenario     : nhãn kịch bản ghi vào ROBUST

    Nếu log=True, mỗi lần gọi ghi thêm một dòng vào ROBUST.
    """
    n_rep = n_rep or CFG["N_REP"]; n_past = model.n_past
    win = Windows(test_data, scaler, model.fill, n_past, 1)
    L = int(miss_minutes // CFG["STEP_MIN"])          # số bước 5 phút bị mất
    if region is None and (miss_pct == 0 or L == 0):
        # Test sạch: không can thiệp vào input.
        res = evaluate(model, win, scaler); k = 0
    elif region is not None:
        # Hỏng cố định theo khu vực: cùng một tập cảm biến cho mọi mẫu.
        sens = torch.zeros(N_NODES, dtype=torch.bool, device=DEVICE); sens[torch.as_tensor(region, device=DEVICE)] = True
        res = evaluate(model, win, scaler, fault=dict(L=L, sensors=sens)); k = len(region)
    else:
        # Hỏng ngẫu nhiên: lặp n_rep lần với seed khác nhau rồi lấy trung bình.
        k = int(round(miss_pct / 100 * N_NODES)); outs = []
        for r in range(n_rep):
            g = torch.Generator(device=DEVICE); g.manual_seed(seed * 1000 + r)
            outs.append(evaluate(model, win, scaler, fault=dict(L=L, k=k, gen=g)))
        res = _avg(outs)
    # Ghi một dòng vào ROBUST để cuối notebook pivot thành bảng.
    if log:
        ROBUST.append(dict(model=label or getattr(model, "name_", "model"), scenario=scenario or ("region" if region is not None else "random"),
                           miss_pct=miss_pct, miss_minutes=miss_minutes, n_sensors_lost=k, **res))
    return res

print(f"N_IN = {N_IN} kênh | MISS_RATES = {CFG['MISS_RATES']} | USE_ADJ_FEAT = {CFG['USE_ADJ_FEAT']}")
trained_model, TRAIN_RES, TEST_CLEAN = train_model(
    train_data, val_data, test_data, CFG["SCALER"], CFG["MODEL"], CFG["N_PAST"],
    miss_rates=CFG["MISS_RATES"], monitor="MAE_avg", verbose=True)

print(f"\n{trained_model.name_}/{trained_model.scaler.name} | best epoch {TRAIN_RES['best_epoch']} / {TRAIN_RES['epochs_run']} | "
      f"val MAE {TRAIN_RES['best_val_MAE']:.3f} | val MAPE {TRAIN_RES['best_val_MAPE']:.3f}% | "
      f"test MAE {TEST_CLEAN['MAE_avg']:.3f} | test MAPE {TEST_CLEAN['MAPE_avg']:.3f}% | "
      f"{TRAIN_RES['train_time_s']:.0f}s | params {TRAIN_RES['n_params']:,}")
display(results_df().round(3))

MODEL_STEM = f"{trained_model.name_.lower()}_{trained_model.scaler.name}"
MODEL_PKL = OUTPUT_DIR / f"{MODEL_STEM}.pkl"

def save_model_pkl(model, path, train_res=None, test_res=None):
    """Lưu model + scaler + imputer + cấu hình vào một file pickle."""
    bundle = dict(
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        model_name=model.name_, train_config=dict(model.train_config), cfg=dict(CFG),
        model_options=dict(getattr(model, "model_options", {})),
        n_nodes=N_NODES, n_in=N_IN, n_past=model.n_past, horizons=list(HORIZONS),
        scaler=dict(name=model.scaler.name, mu=np.asarray(model.scaler.mu), sd=np.asarray(model.scaler.sd)),
        fill=np.asarray(model.fill), a_nb=model.A_nb.detach().cpu().numpy(),
        train_result={k: v for k, v in (train_res or {}).items()}, test_clean=dict(test_res or {}),
        saved_at=pd.Timestamp.now().isoformat(), torch_version=torch.__version__,
    )
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[*] Đã lưu model: {path} ({Path(path).stat().st_size / 1e6:.2f} MB)")

def load_model_pkl(path, device=DEVICE):
    """Dựng lại model từ file pickle, gắn sẵn scaler / fill / n_past để dùng với test_model."""
    with open(path, "rb") as f:
        b = pickle.load(f)
    tc = b["train_config"]
    A = b["state_dict"]["A"].to(device)
    model_kwargs = {"model_options": b.get("model_options", {})} if b["model_name"] == "STID" else {}
    m = MODELS[b["model_name"]](b["n_nodes"], b["n_in"], len(b["horizons"]),
                                tc["hidden"], b["n_past"], A, **model_kwargs)
    m.load_state_dict(b["state_dict"]); m = m.to(device).eval()
    m.A_nb = torch.tensor(b["a_nb"], device=device)
    sc = SCALERS[b["scaler"]["name"]](); sc.mu, sc.sd = b["scaler"]["mu"], b["scaler"]["sd"]
    m.scaler, m.fill, m.n_past, m.name_ = sc, b["fill"], b["n_past"], b["model_name"]
    m.train_config, m.eval_bs = tc, tc.get("eval_bs", CFG["EVAL_BATCH"])
    return m

save_model_pkl(trained_model, MODEL_PKL, TRAIN_RES, TEST_CLEAN)

# Kiểm tra: model load lại phải cho cùng kết quả test sạch (sai khác nhỏ do AMP FP16).
model_eval = load_model_pkl(MODEL_PKL)
_chk = test_model(test_data, model_eval.scaler, model_eval, miss_pct=0, miss_minutes=0, log=False)
_diff = abs(_chk["MAE_avg"] - TEST_CLEAN["MAE_avg"])
print(f"test MAE gốc {TEST_CLEAN['MAE_avg']:.4f} | sau khi load {_chk['MAE_avg']:.4f} | chênh {_diff:.2e}")
assert _diff < 1e-2, "Model load từ pkl không khớp với model vừa train"

ROBUST.clear()                      # chỉ giữ kết quả của lần chạy này
LABEL = f"{model_eval.name_} + {model_eval.scaler.name}"
RANDOM_PCTS = (5, 20, 30)           # % cảm biến mất ngẫu nhiên
REGION_PCT = 10                     # % cảm biến gần nhau cùng hỏng
RANDOM_MINUTES = (15, 30, 45)       # thời lượng mất dữ liệu cho kịch bản ngẫu nhiên (cần < N_PAST * 5 = 60 phút)
REGION_MINUTES = (15, 30)           # thời lượng mất dữ liệu cho kịch bản khu vực
assert max(RANDOM_MINUTES + REGION_MINUTES) < CFG["N_PAST"] * CFG["STEP_MIN"], "thời gian mất phải ngắn hơn cửa sổ input"
t0 = time.time()

# (1) Test sạch: 0% cảm biến mất.
test_model(test_data, model_eval.scaler, model_eval, miss_pct=0, miss_minutes=0, label=LABEL, scenario="clean")

# (2) Mất ngẫu nhiên 5/20/30% cảm biến, mỗi mức thử 15/30/45 phút.
for p in RANDOM_PCTS:
    for mnt in RANDOM_MINUTES:
        test_model(test_data, model_eval.scaler, model_eval, miss_pct=p, miss_minutes=mnt,
                   seed=CFG["TEST_SEED"], label=LABEL, scenario="random")

# (3) Mất theo khu vực: tâm ngẫu nhiên (cố định theo seed) + k cảm biến gần nhất theo đường đi ngắn nhất.
rng = np.random.default_rng(CFG["TEST_SEED"])
CENTERS = rng.choice(N_NODES, size=CFG["N_REGION_CENTERS"], replace=False)
k_region = max(1, int(round(REGION_PCT / 100 * N_NODES)))
REGIONS = {int(c): get_region(int(c), k_region) for c in CENTERS}
for c, reg in REGIONS.items():
    # Kiểm tra khu vực thật sự gần nhau: khoảng cách xa nhất từ tâm vs. trung vị toàn mạng.
    print(f"tâm {c:3d}: {len(reg)} cảm biến | xa nhất từ tâm {PROX[c, reg].max():.0f} | "
          f"trung vị từ tâm tới mọi cảm biến liên thông {np.median(PROX[c][PROX[c] < 1e9]):.0f}")
for mnt in REGION_MINUTES:
    outs = [test_model(test_data, model_eval.scaler, model_eval, miss_pct=REGION_PCT, miss_minutes=mnt,
                       region=reg, log=False) for reg in REGIONS.values()]
    ROBUST.append(dict(model=LABEL, scenario="region", miss_pct=REGION_PCT, miss_minutes=mnt,
                       n_sensors_lost=k_region, **_avg(outs)))

print(f"Xong {len(ROBUST)} kịch bản trong {time.time() - t0:.0f}s")

REPORT = robust_df()
clean_mae = REPORT.loc[REPORT.scenario == "clean", "MAE_avg"].iloc[0]
REPORT["MAE_increase_pct"] = 100 * (REPORT["MAE_avg"] / clean_mae - 1)
REPORT["scenario"] = pd.Categorical(REPORT["scenario"], ["clean", "random", "region"], ordered=True)
metric_cols = [f"{m}_{h}" for m in ("MAE", "RMSE", "MAPE") for h in HOR_MIN] + ["MAE_avg", "RMSE_avg", "MAPE_avg"]
REPORT = (REPORT[["model", "scenario", "miss_pct", "miss_minutes", "n_sensors_lost"] + metric_cols + ["MAE_increase_pct"]]
          .sort_values(["scenario", "miss_pct", "miss_minutes"]).round(3).reset_index(drop=True))
display(REPORT)

# Bảng tóm tắt: MAE / MAPE trung bình theo kịch bản x thời lượng mất.
SUMMARY = REPORT.pivot_table(index=["scenario", "miss_pct"], columns="miss_minutes",
                             values=["MAE_avg", "MAPE_avg"], observed=True).round(3)
display(SUMMARY)

RESULT_STEM = f"{model_eval.name_.lower()}_{model_eval.scaler.name}"
ROBUST_CSV = OUTPUT_DIR / f"{RESULT_STEM}_robustness.csv"
TRAIN_CSV = OUTPUT_DIR / f"{RESULT_STEM}_train.csv"
REPORT.to_csv(ROBUST_CSV, index=False)
results_df().to_csv(TRAIN_CSV, index=False)
print("[*] Đã lưu:", ROBUST_CSV, "|", TRAIN_CSV)


