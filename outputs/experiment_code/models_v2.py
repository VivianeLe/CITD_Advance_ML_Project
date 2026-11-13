# Cell mở rộng mô hình: chạy sau cell định nghĩa BaseNet, MODELS và build_model.
# Giữ nguyên input, target, scaler, residual, loss và cách tính metric hiện có.

class _V2Context(nn.Module):
    """Mã hóa cửa sổ, danh tính cảm biến và thời gian ở cuối cửa sổ."""
    def __init__(self, n_nodes, n_in, hidden, n_past):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.emb_dim = int(CFG.get("NODE_EMB", 16))
        self.use_time = bool(CFG.get("USE_TOD", True))
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


class _V2ResidualMLP(nn.Module):
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
    """MLP thời gian có embedding cảm biến, giờ trong ngày và cuối tuần."""
    def __init__(self, n_nodes, n_in, n_out, hidden, n_past, A):
        super().__init__()
        depth = int(CFG.get("MODEL_DEPTH", 3))
        dropout = float(CFG.get("DROPOUT", 0.05))
        if depth < 1 or not 0 <= dropout < 1:
            raise ValueError("MODEL_DEPTH >= 1 và 0 <= DROPOUT < 1")
        self.context = _V2Context(n_nodes, n_in, hidden, n_past)
        self.blocks = nn.Sequential(*[_V2ResidualMLP(self.context.width, dropout) for _ in range(depth)])
        self.head = _zero_(nn.Linear(self.context.width, n_out))
        self.use_adj = True
        self.model_options = dict(hidden=hidden, n_past=n_past, node_emb=self.context.emb_dim,
                                  tod_slots=288, weekend_slots=2, use_time=self.context.use_time,
                                  depth=depth, dropout=dropout, use_adj=True)

    def core(self, x):
        return self.head(self.blocks(self.context(x)))


class _V2GraphBlock(nn.Module):
    """Ba đường riêng: cảm biến hiện tại, graph vật lý và graph học được."""
    def __init__(self, hidden, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.self_path = nn.Linear(hidden, hidden)
        # Nhánh láng giềng bắt đầu ở 0, rồi học mức đóng góp từ dữ liệu train.
        self.physical_path = _zero_(nn.Linear(hidden, hidden, bias=False))
        self.adaptive_path = _zero_(nn.Linear(hidden, hidden, bias=False))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, physical, adaptive):
        normalized = self.norm(h)
        own = self.self_path(normalized)
        physical_h = torch.einsum("nm,bmc->bnc", physical.to(normalized.dtype), normalized)
        out = own + self.physical_path(physical_h)
        if adaptive is not None:
            adaptive_h = torch.einsum("nm,bmc->bnc", adaptive.to(normalized.dtype), normalized)
            out = out + self.adaptive_path(adaptive_h)
        return h + self.drop(F.gelu(out))


class AdaptiveGCNNet(BaseNet):
    """GNN có embedding ngữ cảnh, graph vật lý và quan hệ thích nghi."""
    def __init__(self, n_nodes, n_in, n_out, hidden, n_past, A):
        super().__init__()
        depth = int(CFG.get("MODEL_DEPTH", 3))
        dropout = float(CFG.get("DROPOUT", 0.05))
        self.topk = int(CFG.get("ADAPT_TOPK", 0))
        if depth < 1 or not 0 <= dropout < 1 or self.topk < 0:
            raise ValueError("Cần MODEL_DEPTH >= 1, 0 <= DROPOUT < 1 và ADAPT_TOPK >= 0")
        self.context = _V2Context(n_nodes, n_in, hidden, n_past)
        self.project = nn.Linear(self.context.width, hidden)
        self.register_buffer("A", A.detach().clone())
        self.adaptive_source = nn.Parameter(torch.randn(n_nodes, self.context.emb_dim) * 0.1)
        self.adaptive_target = nn.Parameter(torch.randn(n_nodes, self.context.emb_dim) * 0.1)
        self.blocks = nn.ModuleList([_V2GraphBlock(hidden, dropout) for _ in range(depth)])
        self.head = _zero_(nn.Linear(hidden, n_out))
        self.use_adj = True
        self.model_options = dict(hidden=hidden, n_past=n_past, node_emb=self.context.emb_dim,
                                  tod_slots=288, weekend_slots=2, use_time=self.context.use_time,
                                  depth=depth, dropout=dropout, adaptive_topk=self.topk, use_adj=True)

    def adaptive_support(self):
        # Tính softmax ở FP32 kể cả khi forward chạy AMP, tránh mất ổn định số.
        nodes = self.adaptive_source.shape[0]
        with torch.autocast(device_type=self.adaptive_source.device.type, enabled=False):
            if nodes == 1:
                return self.adaptive_source.new_zeros((1, 1), dtype=torch.float32)
            scores = F.relu(self.adaptive_source.float() @ self.adaptive_target.float().T
                            / math.sqrt(self.context.emb_dim))
            diagonal = torch.eye(nodes, device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(diagonal, float("-inf"))
            # Mặc định dùng graph dense; chỉ giới hạn top-k khi cấu hình yêu cầu.
            if 0 < self.topk < nodes - 1:
                indices = scores.topk(self.topk, dim=-1).indices
                selected = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, indices, True)
                scores = scores.masked_fill(~selected, float("-inf"))
            return torch.softmax(scores, dim=-1)

    def core(self, x):
        h = F.gelu(self.project(self.context(x)))
        adaptive = self.adaptive_support() if self.use_adj else None
        for block in self.blocks:
            h = block(h, self.A, adaptive)
        return self.head(h)


MODELS.update({"STID": STIDNet, "GNN_ADAPT": AdaptiveGCNNet})

# Bọc factory một lần; chạy lại cell vẫn gọi đúng factory gốc, tránh đệ quy.
_build_model_v2_original = getattr(build_model, "_v2_original", build_model)


def build_model(name, n_nodes, n_past, hidden, use_adj=True):
    model = _build_model_v2_original(name, n_nodes, n_past, hidden, use_adj=use_adj)
    if isinstance(model, (STIDNet, AdaptiveGCNNet)):
        model.use_adj = bool(use_adj)
        model.model_options["use_adj"] = bool(use_adj)
    return model


build_model._v2_original = _build_model_v2_original
