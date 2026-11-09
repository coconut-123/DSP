"""
train_classifier.py
-------------------
Huấn luyện mạng nơ-ron phân loại biển báo giao thông (TrafficSignClassifier - MLP + LoRA Residual Head)
đặt trên đặc trưng kết hợp của:
  1) ViT-B/16 (ImageNet pretrained, frozen, 768-d)
  2) Đặc trưng hình học - phổ màu bất biến ánh sáng (compute_image_signature, 2544-d)

Mô hình học tham số thực sự bằng CrossEntropyLoss + AdamW + CosineAnnealingLR,
lưu trọng số mạng đã huấn luyện vào `models/traffic_classifier_head.pt`.
Khi suy luận trên Web (`server.py`), hệ thống CHỈ chạy forward qua mạng `TrafficSignClassifier`
để lấy phân phối xác suất `softmax(logits)`, hoàn toàn KHÔNG dùng k-NN so khớp kho ảnh mẫu.
"""

import sys
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torchvision.models as tv_models

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = MODELS_DIR / "traffic_classifier_head.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_and_crop_traffic_sign(img: Image.Image) -> Image.Image:
    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    h, w, _ = arr.shape
    if h < 24 or w < 24:
        return rgb
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    red_score = r - np.maximum(g, b)
    red_mask = (red_score > 38) & (r > 105)
    yellow_score = np.minimum(r, g) - b
    yellow_mask = (yellow_score > 45) & (r > 140) & (g > 110) & (r >= g * 0.90)
    blue_score = b - np.maximum(r, g)
    blue_mask = (blue_score > 35) & (b > 100)

    best_box = None
    for mask in [red_mask, yellow_mask, blue_mask]:
        ys, xs = np.where(mask)
        if len(ys) < max(20, int(h * w * 0.004)):
            continue
        y0, y1 = int(np.percentile(ys, 2)), int(np.percentile(ys, 98))
        x0, x1 = int(np.percentile(xs, 2)), int(np.percentile(xs, 98))
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        if bw < 14 or bh < 14:
            continue
        if bh > bw * 1.35:
            win_h = int(bw * 1.08)
            best_y0, max_cnt = y0, -1
            for cand_y in range(y0, max(y0 + 1, y1 - win_h + 2), max(1, win_h // 10)):
                cnt = int(mask[cand_y : cand_y + win_h, x0 : x1 + 1].sum())
                if cnt > max_cnt:
                    max_cnt, best_y0 = cnt, cand_y
            y0 = best_y0
            y1 = min(h - 1, y0 + win_h)
            bh = y1 - y0 + 1
        aspect = bw / max(bh, 1.0)
        if 0.55 <= aspect <= 1.85:
            best_box = (x0, y0, min(w, x1 + 1), min(h, y1 + 1))
            break
    return rgb.crop(best_box) if best_box is not None else rgb


def compute_image_signature(img: Image.Image, apply_detector: bool = True) -> np.ndarray:
    cropped = detect_and_crop_traffic_sign(img) if apply_detector else img.convert("RGB")
    im_small = cropped.resize((48, 48), Image.Resampling.BILINEAR)
    arr = np.asarray(im_small, dtype=np.float32)
    lo, hi = float(np.percentile(arr, 3)), float(np.percentile(arr, 97))
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) if (hi - lo > 15.0) else (arr / 255.0)

    yy, xx = np.mgrid[0:48, 0:48]
    dist_sq = (yy - 23.5) ** 2 + (xx - 23.5) ** 2
    corner_mask = (dist_sq <= 23.5 ** 2).astype(np.float32)[:, :, None]
    arr_masked = (arr - 0.5) * corner_mask

    r_c, g_c, b_c = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = 0.299 * r_c + 0.587 * g_c + 0.114 * b_c
    is_red = (r_c - np.maximum(g_c, b_c)) > 0.16
    inner_mask = dist_sq <= 16.5 ** 2
    inner_gray_vals = gray[inner_mask & (~is_red)]
    thr = float(np.percentile(inner_gray_vals, 30)) if len(inner_gray_vals) > 10 else 0.45
    thr = min(0.55, max(0.25, thr))

    black_symbol = ((gray <= thr) & (~is_red) & inner_mask).astype(np.float32)
    sym_24 = black_symbol.reshape(24, 2, 24, 2).mean(axis=(1, 3))
    sym_24_vec = (sym_24.reshape(-1) - sym_24.mean()) * 3.2

    bys, bxs = np.where(black_symbol > 0.5)
    if len(bys) >= 8:
        by0, by1 = int(bys.min()), int(bys.max()) + 1
        bx0, bx1 = int(bxs.min()), int(bxs.max()) + 1
        bh_s, bw_s = by1 - by0, bx1 - bx0
        side = max(bh_s, bw_s, 1)
        sq_canvas = np.zeros((side, side), dtype=np.float32)
        off_y, off_x = (side - bh_s) // 2, (side - bw_s) // 2
        sq_canvas[off_y : off_y + bh_s, off_x : off_x + bw_s] = black_symbol[by0:by1, bx0:bx1]
        sym_crop = Image.fromarray((sq_canvas * 255.0).astype(np.uint8)).resize((24, 24), Image.Resampling.BILINEAR)
        sym_norm_box = np.asarray(sym_crop, dtype=np.float32).reshape(-1) / 255.0
        sym_norm_box = (sym_norm_box - sym_norm_box.mean()) * 3.2
        sym_aspect = np.full(48, (bw_s - bh_s) / float(side) * 4.2, dtype=np.float32)
    else:
        sym_norm_box = np.zeros(576, dtype=np.float32)
        sym_aspect = np.zeros(48, dtype=np.float32)

    # Bổ sung đặc trưng vạch chéo phụ (anti-diagonal redness) để mạng tự học phân biệt P.130 vs P.131a
    redness_map = r_c - np.maximum(g_c, b_c)
    inner_ring = (dist_sq >= 5.0 ** 2) & (dist_sq <= 17.0 ** 2)
    anti_diag = (np.abs(xx + yy - 47.0) / 1.414) <= 2.8
    main_diag = (np.abs(xx - yy) / 1.414) <= 2.8
    anti_val = float(np.mean(redness_map[inner_ring & anti_diag]))
    main_val = float(np.mean(redness_map[inner_ring & main_diag]))
    diag_feats = np.full(32, anti_val * 3.5, dtype=np.float32)
    main_feats = np.full(32, main_val * 3.5, dtype=np.float32)

    sym_vec = np.concatenate([sym_24_vec, sym_norm_box, sym_aspect, diag_feats, main_feats])
    rgb_vec = arr_masked.reshape(-1) * 0.90
    combined = np.concatenate([rgb_vec, sym_vec])
    norm = np.linalg.norm(combined)
    return combined / max(norm, 1e-6)


class TrafficSignClassifier(nn.Module):
    """
    Mạng phân loại biển báo giao thông có học tham số:
    LayerNorm -> Linear(in_dim, hidden_dim) + Nhánh Low-Rank LoRA (r=32) -> GELU -> Linear(hidden_dim, num_classes).
    """

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 768, lora_rank: int = 32):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.lora_rank = lora_rank

        self.norm_in = nn.LayerNorm(in_dim)
        self.stem = nn.Linear(in_dim, hidden_dim)
        self.lora_A = nn.Linear(in_dim, lora_rank, bias=False)
        self.lora_B = nn.Linear(lora_rank, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = 32.0 / lora_rank

        self.dropout = nn.Dropout(0.18)
        self.norm_mid = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_in(x)
        base_out = F.gelu(self.stem(h))
        lora_out = self.lora_B(F.gelu(self.lora_A(self.dropout(h)))) * self.scaling
        return self.norm_mid(base_out + lora_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.head(self.dropout(feat))


def augment_pil_image(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    im = img.convert("RGB")
    w, h = im.size
    if w >= 28 and h >= 28 and rng.random() < 0.7:
        pad = int(max(w, h) * rng.uniform(0.0, 0.07))
        if pad > 0:
            canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (128, 128, 128))
            canvas.paste(im, (pad, pad))
            im = canvas
        cw, ch = im.size
        dx = int(cw * rng.uniform(0.0, 0.05))
        dy = int(ch * rng.uniform(0.0, 0.05))
        im = im.crop((dx, dy, cw - dx, ch - dy))
    if rng.random() < 0.7:
        im = ImageEnhance.Brightness(im).enhance(float(rng.uniform(0.78, 1.22)))
    if rng.random() < 0.7:
        im = ImageEnhance.Contrast(im).enhance(float(rng.uniform(0.78, 1.28)))
    if rng.random() < 0.5:
        im = ImageEnhance.Color(im).enhance(float(rng.uniform(0.82, 1.20)))
    if rng.random() < 0.25:
        im = im.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 0.9))))
    return im


def main():
    print(f"[*] Khởi tạo huấn luyện TrafficSignClassifier trên thiết bị: {DEVICE}", flush=True)
    with open(DATA_DIR / "split_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    img_to_splits = {r["image_id"]: r["grouped_split"] for r in manifest["records"]}

    with open(DATA_DIR / "scene_annotations.json", "r", encoding="utf-8") as f:
        scenes = json.load(f)["scenes"]

    real_items: List[Tuple[Path, str, str]] = []
    for sc in scenes:
        sp = img_to_splits.get(sc["image_id"], "train")
        for b in sc["boxes"]:
            if "crop_path" in b:
                p = BASE_DIR / b["crop_path"]
                if p.exists():
                    real_items.append((p, b["canonical_code"], sp))

    pdf_items: List[Tuple[Path, str]] = []
    with open(DATA_DIR / "pdf_signs_catalog.json", "r", encoding="utf-8") as f:
        pdf_cat = json.load(f)["dataset"]
    for ps in pdf_cat:
        p = BASE_DIR / ps["image"]
        if p.exists():
            pdf_items.append((p, ps["sign_code"]))

    # Thêm bất kỳ ảnh chuẩn nào khác trong data/pdf_signs
    existing_pdf_paths = set(p.resolve() for p, _ in pdf_items)
    pdf_extra: List[Tuple[Path, str]] = []
    for p in sorted((DATA_DIR / "pdf_signs").rglob("*.png")):
        if p.resolve() not in existing_pdf_paths:
            pdf_extra.append((p, p.parent.name))

    all_codes = sorted(set([c for _, c, _ in real_items] + [c for _, c in pdf_items] + [c for _, c in pdf_extra]))
    code_to_idx: Dict[str, int] = {c: i for i, c in enumerate(all_codes)}
    idx_to_code: Dict[int, str] = {i: c for i, c in enumerate(all_codes)}
    num_classes = len(all_codes)
    print(f"[*] Tổng số lớp: {num_classes} | Real crops: {len(real_items)} | Standard signs: {len(pdf_items) + len(pdf_extra)}", flush=True)

    # Tận dụng đặc trưng ViT-B/16 đã trích trong backbone_features_clean.pt
    clean_cache = torch.load(DATA_DIR / "backbone_features_clean.pt", map_location="cpu", weights_only=False)
    grp_mean = clean_cache["grp_mean"]
    grp_std = clean_cache["grp_std"]
    vit_all = F.normalize(clean_cache["vit_grp"].float(), dim=1)
    n_real = int(clean_cache["n_real"])

    mean_t = torch.tensor(grp_mean, dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
    std_t = torch.tensor(grp_std, dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
    vit = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.DEFAULT).to(DEVICE).eval()
    vit.heads = nn.Identity()

    print("[*] Đang tính đặc trưng kết hợp (ViT-B/16 + Structural Signature)...", flush=True)
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    for i, (p, code, sp) in enumerate(real_items):
        with Image.open(p) as im:
            sig = torch.from_numpy(compute_image_signature(im, apply_detector=False)).float()
        feat = torch.cat([vit_all[i], sig * 1.35], dim=0)
        if sp == "val":
            X_val_list.append(feat)
            y_val_list.append(code_to_idx[code])
        else:
            X_train_list.append(feat)
            y_train_list.append(code_to_idx[code])

    rng = np.random.default_rng(42)
    aug_imgs, aug_sigs, aug_labels = [], [], []
    for j, (p, code) in enumerate(pdf_items):
        idx_cache = n_real + j
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            sig0 = torch.from_numpy(compute_image_signature(rgb, apply_detector=True)).float()
            if idx_cache < len(vit_all):
                X_train_list.append(torch.cat([vit_all[idx_cache], sig0 * 1.35], dim=0))
                y_train_list.append(code_to_idx[code])
            for _ in range(5):
                im_aug = augment_pil_image(rgb, rng)
                aug_imgs.append(im_aug)
                aug_sigs.append(torch.from_numpy(compute_image_signature(im_aug, apply_detector=True)).float())
                aug_labels.append(code_to_idx[code])

    for p, code in pdf_extra:
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            for k in range(6):
                im_use = rgb if k == 0 else augment_pil_image(rgb, rng)
                aug_imgs.append(im_use)
                aug_sigs.append(torch.from_numpy(compute_image_signature(im_use, apply_detector=True)).float())
                aug_labels.append(code_to_idx[code])

    if aug_imgs:
        raw_aug = torch.zeros((len(aug_imgs), 3, 224, 224), dtype=torch.float32)
        for k, im_a in enumerate(aug_imgs):
            arr_a = np.asarray(im_a.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32).transpose(2, 0, 1) / 255.0
            raw_aug[k] = torch.from_numpy(arr_a)
        vit_aug_list = []
        with torch.no_grad():
            for st in range(0, len(aug_imgs), 64):
                b = raw_aug[st : st + 64].to(DEVICE)
                vit_aug_list.append(F.normalize(vit((b - mean_t) / std_t).float().cpu(), dim=1))
        vit_aug = torch.cat(vit_aug_list, dim=0)
        for k in range(len(aug_imgs)):
            X_train_list.append(torch.cat([vit_aug[k], aug_sigs[k] * 1.35], dim=0))
            y_train_list.append(aug_labels[k])

    X_train = torch.stack(X_train_list, dim=0)
    y_train = torch.tensor(y_train_list, dtype=torch.long)
    X_val = torch.stack(X_val_list, dim=0)
    y_val = torch.tensor(y_val_list, dtype=torch.long)

    in_dim = X_train.shape[1]
    print(f"[*] Kích thước vector đầu vào: {in_dim}-d | Train samples: {len(X_train)} | Val samples: {len(X_val)}", flush=True)

    cls_counts = np.bincount(y_train.numpy(), minlength=num_classes).astype(np.float64)
    inv_w = np.where(cls_counts > 0, 1.0 / np.power(np.maximum(cls_counts, 1), 0.7), 1.0)
    inv_w = inv_w / np.mean(inv_w)
    sample_weights = [inv_w[int(y)] for y in y_train.numpy()]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, sampler=sampler)

    model = TrafficSignClassifier(in_dim=in_dim, num_classes=num_classes, hidden_dim=768, lora_rank=32).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=0.04)
    epochs = 45
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            bx = bx + torch.randn_like(bx) * 0.02
            logits = model(bx)
            loss = F.cross_entropy(logits, by, label_smoothing=0.05)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * bx.size(0)
        scheduler.step()

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val.to(DEVICE))
                val_preds = val_logits.argmax(dim=-1).cpu()
                val_acc = float((val_preds == y_val).float().mean().item())
                train_logits = model(X_train.to(DEVICE))
                train_acc = float((train_logits.argmax(dim=-1).cpu() == y_train).float().mean().item())
            print(f"  Epoch {ep:02d}/{epochs} | Loss: {total_loss/len(X_train):.4f} | Train Acc: {train_acc*100:.2f}% | Real Val Acc: {val_acc*100:.2f}%", flush=True)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_payload = {
        "architecture": "TrafficSignClassifier_ViT_LoRA_MLP",
        "in_dim": in_dim,
        "hidden_dim": 768,
        "lora_rank": 32,
        "num_classes": num_classes,
        "all_codes": all_codes,
        "code_to_idx": code_to_idx,
        "idx_to_code": idx_to_code,
        "grp_mean": grp_mean,
        "grp_std": grp_std,
        "best_real_val_acc": round(best_val_acc, 4),
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    }
    torch.save(checkpoint_payload, CKPT_PATH)
    print(f"[✓] Đã lưu trọng số mạng phân loại tại: {CKPT_PATH} (Real Val Acc = {best_val_acc*100:.2f}%)", flush=True)


if __name__ == "__main__":
    main()
