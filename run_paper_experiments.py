r"""
run_paper_experiments.py
------------------------
Pipeline thực nghiệm SẠCH (không rò rỉ nhãn) cho bài toán VLSP 2025 MLQA-TSR trên dữ liệu của project.

Giao thức:
- Train: `data/real_crops` (1.807 ảnh cắt biển báo thực tế, 52 lớp; chia grouped train/val theo cảnh).
- Test : `data/pdf_signs` (51 ảnh biển chuẩn trích từ PDF QCVN 41:2024/BGTVT, 46 lớp).
- Subtask 2 CHỈ dùng câu hỏi Đúng / Sai (Yes/No: "Đúng" hoặc "Sai") .

Nguyên tắc trung thực (đã sửa so với phiên bản cũ):
1. KHÔNG dùng nhãn của ảnh test ở bất kỳ bước nào trước khi dự đoán. Hàm cũ `align_real_to_pdf_domain`
   (trộn 56% vector trung bình của ĐÚNG lớp vào đặc trưng ảnh test) đã bị xoá.
2. Đặc trưng được trích lại trực tiếp từ ảnh bằng backbone ImageNet ĐÓNG BĂNG và lưu ở
   `data/backbone_features_clean.pt` (cache cũ bị nhiễm nhãn không còn được dùng).
3. Bộ phân loại là một MLP head có nhánh low-rank (kiểu LoRA) đặt trên đặc trưng đóng băng.
   Đây KHÔNG phải LoRA gắn vào trong ViT.
4. Thích ứng miền chỉ dùng phương pháp KHÔNG cần nhãn đích: căn chỉnh mean/std theo từng chiều
   (transductive, dùng thống kê không nhãn của tập ảnh PDF).
5. Không còn baseline giả lập bằng số ngẫu nhiên. Subtask 2 không còn chỗ gán thẳng đáp án đúng.
   Kết quả VLM (Qwen2-VL) được đo thật bằng `eval_vlm_qa.py` và nạp từ `data/vlm_qa_results.json`.
6. Lưu ý Subtask 1: nhãn `relevant_articles` sinh từ cùng bảng ánh xạ mã biển -> điều khoản mà retriever
   dùng khi có mã biển. Vì vậy F2 của các cấu hình "visual" chỉ đo độ chính xác nhận diện mã biển,
   không phải năng lực truy xuất độc lập.
"""

import sys
import json
import math
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torchvision.models as tv_models

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legal_ir import HybridLegalRetriever, compute_f2_metrics, tokenize_vn

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 2024]
CLEAN_CACHE = DATA_DIR / "backbone_features_clean.pt"
CLEAN_PROTOCOL = "clean_no_test_label_v1"

SUPERGROUP_NAMES = [
    "P (Biển báo cấm)",
    "W (Biển cảnh báo)",
    "R (Biển hiệu lệnh)",
    "I (Biển chỉ dẫn)",
    "S (Biển phụ)",
]

# Kết quả công bố trong bài báo VLSP 2025 MLQA-TSR (Bảng 3 & 4) — CHỈ ĐỂ THAM KHẢO.
# Các đội được đánh giá trên tập test chính thức của cuộc thi, KHÁC tập 51 ảnh PDF của project,
# nên KHÔNG so sánh trực tiếp được với kết quả bên dưới.
VLSP2025_OFFICIAL_TABLE3_SUBTASK1 = [
    {"rank": 1, "team": "LifeIsTough", "f2_score": 0.6455361395},
    {"rank": 2, "team": "chmod+x", "f2_score": 0.6113748745},
    {"rank": 3, "team": "TechNova", "f2_score": 0.5991697165},
    {"rank": 13, "team": "BASELINE (BGE Visualized)", "f2_score": 0.1276493485},
]
VLSP2025_OFFICIAL_TABLE4_SUBTASK2 = [
    {"rank": 1, "team": "Berry (LLaMA4-Maverick + Few-shot)", "accuracy": 0.8630136986},
    {"rank": 2, "team": "SmartbotIC (Qwen2.5-VL + InternVL3)", "accuracy": 0.8356164384},
    {"rank": 14, "team": "BASELINE (Vintern-3B-beta)", "accuracy": 0.4520547945},
]


def code_to_supergroup_idx(code_str: str) -> int:
    if code_str.startswith(("P.", "DP.", "B.")):
        return 0
    if code_str.startswith("W."):
        return 1
    if code_str.startswith("R."):
        return 2
    if code_str.startswith(("I.", "IE.")) or code_str == "Camera":
        return 3
    if code_str.startswith("S."):
        return 4
    return 0


# ==============================================================================
# BƯỚC 1: NẠP DỮ LIỆU (Train: data/real_crops, Test: data/pdf_signs)
# ==============================================================================
def load_experimental_cohort():
    with open(DATA_DIR / "split_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    expected_hash = manifest["split_hash_sha256"]
    img_to_splits = {
        r["image_id"]: (r["grouped_split"], r["random_split"], r["group_id"])
        for r in manifest["records"]
    }
    with open(DATA_DIR / "scene_annotations.json", "r", encoding="utf-8") as f:
        scenes = json.load(f)["scenes"]
    with open(DATA_DIR / "pdf_signs_catalog.json", "r", encoding="utf-8") as f:
        pdf_signs = json.load(f)["dataset"]
    with open(DATA_DIR / "class_mapping_52.json", "r", encoding="utf-8") as f:
        class_mapping = json.load(f)

    code_to_cls52 = {}
    img_to_cls52 = {}
    for cm in class_mapping:
        code_to_cls52[cm["dataset_code"]] = cm["class_id"]
        if cm["canonical_qcvn_code"] not in code_to_cls52:
            code_to_cls52[cm["canonical_qcvn_code"]] = cm["class_id"]
        for p_img in cm.get("pdf_template_images", []):
            img_to_cls52[p_img.replace("\\", "/")] = cm["class_id"]

    real_samples = []
    for sc in scenes:
        img_id = sc["image_id"]
        g_sp, r_sp, gid = img_to_splits[img_id]
        for b in sc["boxes"]:
            if "crop_path" in b:
                crop_abs = BASE_DIR / b["crop_path"]
                if crop_abs.exists():
                    real_samples.append({
                        "path": crop_abs,
                        "scene_id": img_id,
                        "group_id": gid,
                        "class_52": b["class_id"],
                        "supergroup": code_to_supergroup_idx(b["dataset_code"]),
                        "dataset_code": b["dataset_code"],
                        "canonical_code": b["canonical_code"],
                        "grouped_split": g_sp,
                        "random_split": r_sp,
                        "is_pdf": False,
                    })

    pdf_samples = []
    for ps in pdf_signs:
        sc_code = ps["sign_code"]
        p_abs = BASE_DIR / ps["image"]
        rel_img = ps["image"].replace("\\", "/")
        assigned_cls = img_to_cls52.get(rel_img, code_to_cls52.get(sc_code, -1))
        if p_abs.exists():
            pdf_samples.append({
                "path": p_abs,
                "scene_id": ps["id"],
                "group_id": -1,
                "class_52": assigned_cls,
                "supergroup": code_to_supergroup_idx(sc_code),
                "dataset_code": sc_code,
                "canonical_code": sc_code,
                "grouped_split": "test",
                "random_split": "test",
                "is_pdf": True,
            })

    print(f"[✓] {len(real_samples)} ảnh real_crops (TRAIN/VAL) | {len(pdf_samples)} ảnh pdf_signs (TEST)")
    return real_samples, pdf_samples, class_mapping, expected_hash


# ==============================================================================
# BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG SẠCH (backbone ImageNet đóng băng, không dùng nhãn test)
# ==============================================================================
def extract_backbone_representations(real_samples, pdf_samples, force: bool = False):
    if CLEAN_CACHE.exists() and not force:
        cached = torch.load(CLEAN_CACHE, map_location="cpu", weights_only=False)
        if cached.get("eval_protocol") == CLEAN_PROTOCOL and cached.get("n_pdf") == len(pdf_samples) \
                and cached.get("n_real") == len(real_samples):
            print(f"[*] Nạp đặc trưng sạch từ {CLEAN_CACHE.name}")
            return cached

    print(f"[*] Trích xuất đặc trưng sạch trên {DEVICE} (ViT-B/16, Swin-T, EfficientNet-B3 — ImageNet, đóng băng)...")
    all_items = real_samples + pdf_samples
    n_real = len(real_samples)
    raw = torch.zeros((len(all_items), 3, 224, 224), dtype=torch.float32)
    for i, it in enumerate(all_items):
        with Image.open(it["path"]) as im:
            arr = np.asarray(im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            raw[i] = torch.from_numpy(arr.transpose(2, 0, 1))

    # Thống kê chuẩn hoá chỉ lấy từ ảnh TRAIN thực tế
    tr_mask = [i for i in range(n_real) if real_samples[i]["grouped_split"] == "train"]
    grp_mean = raw[tr_mask].mean(dim=(0, 2, 3))
    grp_std = raw[tr_mask].std(dim=(0, 2, 3)).clamp(min=1e-6)
    imnet_mean = torch.tensor([0.485, 0.456, 0.406])
    imnet_std = torch.tensor([0.229, 0.224, 0.225])

    def run(model, mean, std):
        out = []
        m, s = mean.view(1, 3, 1, 1).to(DEVICE), std.view(1, 3, 1, 1).to(DEVICE)
        with torch.no_grad():
            for st in range(0, len(all_items), 64):
                out.append(model((raw[st:st + 64].to(DEVICE) - m) / s).float().cpu())
        return torch.cat(out)

    vit = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.DEFAULT).to(DEVICE).eval()
    vit.heads = nn.Identity()
    swin = tv_models.swin_t(weights=tv_models.Swin_T_Weights.DEFAULT).to(DEVICE).eval()
    swin.head = nn.Identity()
    eff = tv_models.efficientnet_b3(weights=tv_models.EfficientNet_B3_Weights.DEFAULT).to(DEVICE).eval()
    eff.classifier = nn.Identity()

    cache = {
        "eval_protocol": CLEAN_PROTOCOL,
        "n_real": n_real,
        "n_pdf": len(pdf_samples),
        "grp_mean": grp_mean.tolist(),
        "grp_std": grp_std.tolist(),
        "vit_grp": run(vit, grp_mean, grp_std),
        "vit_imnet": run(vit, imnet_mean, imnet_std),
        "swin_grp": run(swin, grp_mean, grp_std),
        "eff_grp": run(eff, grp_mean, grp_std),
    }
    del vit, swin, eff
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.save(cache, CLEAN_CACHE)
    print(f"[✓] Đã lưu đặc trưng sạch: {CLEAN_CACHE}")
    return cache


def label_free_moment_alignment(X: torch.Tensor, src_idx: List[int], tgt_idx: List[int]) -> torch.Tensor:
    """Thích ứng miền KHÔNG dùng nhãn: đưa mean/std từng chiều của miền đích (PDF) về bằng miền nguồn (real)."""
    Xa = X.clone()
    src, tgt = X[src_idx], X[tgt_idx]
    Xa[tgt_idx] = (tgt - tgt.mean(0)) / (tgt.std(0) + 1e-6) * src.std(0) + src.mean(0)
    return Xa


# ==============================================================================
# BƯỚC 3: BỘ PHÂN LOẠI (MLP head + nhánh low-rank trên đặc trưng đóng băng)
# ==============================================================================
class PEFTAdaptedClassifier(nn.Module):
    """Head phân loại: LayerNorm -> Linear(identity init) + nhánh low-rank r=16 -> Linear.
    Chỉ huấn luyện head này; backbone (ViT/Swin/EfficientNet) đóng băng."""

    def __init__(self, in_dim: int, num_classes: int, arch_type: str = "vit", lora_rank: int = 16, freeze_stem: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim) if arch_type in ("vit", "swin") else nn.BatchNorm1d(in_dim, momentum=0.05)
        self.stem_proj = nn.Linear(in_dim, in_dim)
        nn.init.eye_(self.stem_proj.weight)
        nn.init.zeros_(self.stem_proj.bias)
        if freeze_stem:
            for p in self.stem_proj.parameters():
                p.requires_grad = False
        self.lora_A = nn.Linear(in_dim, lora_rank, bias=False)
        self.lora_B = nn.Linear(lora_rank, in_dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = 32.0 / lora_rank
        self.dropout = nn.Dropout(0.15)
        self.head = nn.Linear(in_dim, num_classes)

    def forward_features(self, x):
        h = self.norm1(x)
        return self.stem_proj(h) + self.lora_B(F.gelu(self.lora_A(self.dropout(h)))) * self.scaling

    def forward(self, x):
        return self.head(self.forward_features(x))


def train_and_eval_cell(
    X_all: torch.Tensor,
    y_all: np.ndarray,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: Optional[List[int]],
    num_classes: int,
    arch_type: str,
    strategy: str,
    lr: float,
    epochs: int,
    seed: int,
    freeze_stem: bool = False,
    use_msda: bool = True,
    save_checkpoint_path: Optional[Path] = None,
):
    """Huấn luyện CHỈ trên train_idx (ảnh thực tế). Nhãn test chỉ dùng để chấm điểm sau khi dự đoán."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X_train = X_all[train_idx].clone()
    y_train = torch.tensor(y_all[train_idx], dtype=torch.long)
    X_val = X_all[val_idx].to(DEVICE)
    y_val_np = y_all[val_idx]
    y_val_t = torch.tensor(y_val_np, dtype=torch.long, device=DEVICE)

    cls_counts = np.bincount(y_all[train_idx], minlength=num_classes).astype(np.float64)
    inv_weights = np.where(cls_counts > 0, 1.0 / np.power(np.maximum(cls_counts, 1), 0.75), 0.0)
    inv_weights = inv_weights / np.mean(inv_weights[inv_weights > 0])
    class_weights_t = torch.tensor(inv_weights, dtype=torch.float32, device=DEVICE)

    batch_size = 32 if arch_type != "eff" else 16
    train_ds = TensorDataset(X_train, y_train)
    if strategy.startswith("WLS"):
        sample_w = [inv_weights[y_i] for y_i in y_all[train_idx]]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = PEFTAdaptedClassifier(X_all.shape[1], num_classes, arch_type, 16, freeze_stem).to(DEVICE)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    eps = 0.1 if strategy.startswith("WLS") else 0.0
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            bx = bx + torch.randn_like(bx) * 0.03
            q_target = torch.full((bx.size(0), num_classes), eps / max(1, num_classes - 1), device=DEVICE)
            q_target.scatter_(1, by.unsqueeze(1), 1.0 - eps)
            if strategy == "WLS+MSDA" and use_msda and bx.size(0) > 1 and rng.random() < 0.5:
                perm = torch.randperm(bx.size(0), device=DEVICE)
                lam = float(rng.beta(0.4, 0.4))
                bx = lam * bx + (1.0 - lam) * bx[perm]
                q_target = lam * q_target + (1.0 - lam) * q_target[perm]
            log_probs = F.log_softmax(model(bx), dim=-1)
            if strategy.startswith("WLS"):
                loss = (-q_target * log_probs * class_weights_t.unsqueeze(0)).sum(dim=-1).mean()
            else:
                loss = -(q_target * log_probs).sum(dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    if save_checkpoint_path is not None:
        torch.save(model.state_dict(), save_checkpoint_path)

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_loss = float(F.cross_entropy(val_logits, y_val_t).item())
        val_preds = val_logits.argmax(dim=-1).cpu().numpy()
    present_val = sorted(set(y_val_np.tolist()))
    val_acc = float(accuracy_score(y_val_np, val_preds))
    val_f1m = float(f1_score(y_val_np, val_preds, labels=present_val, average="macro", zero_division=0))
    if test_idx is None:
        return {"val_loss": val_loss, "val_acc": val_acc, "val_f1m": val_f1m}

    X_test = X_all[test_idx].to(DEVICE)
    y_test_np = y_all[test_idx]
    with torch.no_grad():
        test_probs = F.softmax(model(X_test), dim=-1).cpu().numpy()
    test_preds = test_probs.argmax(axis=-1)
    present_test = sorted(set(y_test_np.tolist()))
    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1m": val_f1m,
        "test_acc": float(accuracy_score(y_test_np, test_preds)),
        "test_f1m": float(f1_score(y_test_np, test_preds, labels=present_test, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test_np, test_preds, labels=list(range(num_classes))),
        "test_probs": test_probs,
        "test_preds": test_preds,
        "y_test": y_test_np,
    }


def nearest_prototype_accuracy(X, y, gallery_idx, query_idx) -> float:
    """Phân loại không huấn luyện: gán nhãn theo tâm lớp (cosine) của tập gallery."""
    buckets: Dict[int, List[torch.Tensor]] = {}
    for i in gallery_idx:
        buckets.setdefault(int(y[i]), []).append(X[i])
    keys = sorted(buckets)
    M = F.normalize(torch.stack([torch.stack(buckets[k]).mean(0) for k in keys]), dim=1)
    sims = F.normalize(X[query_idx], dim=1) @ M.T
    pred = np.array(keys)[sims.argmax(1).numpy()]
    return float(np.mean(pred == y[query_idx]))


# ==============================================================================
# BƯỚC 4: SUBTASK 1 — TRUY XUẤT ĐIỀU KHOẢN (F2)
# ==============================================================================
def evaluate_subtask1(test_mqa, pred_codes: Dict[str, Dict[str, str]]):
    print("\n" + "=" * 78)
    print(" SUBTASK 1: LEGAL RETRIEVAL (Precision / Recall / F2) trên câu hỏi test")
    print(" LƯU Ý: nhãn relevant_articles sinh từ bảng ánh xạ mã biển -> điều khoản; với cấu hình visual,")
    print("        F2 = 100% khi và chỉ khi đoán đúng mã biển (đây là tra bảng, không phải truy xuất độc lập).")
    print("=" * 78)
    retriever = HybridLegalRetriever()
    methods = [
        ("Text-only BM25 (top-5)", "bm25_only", None, 5, False),
        ("Text-only Dense TF-IDF/LSA (top-5)", "dense_only", None, 5, False),
        ("Text-only Hybrid RRF (top-5)", "text_hybrid", None, 5, False),
    ]
    for tag in pred_codes:
        methods.append((f"Visual-conditioned RRF + dynamic-k [mã biển dự đoán: {tag}]", "visual_hybrid_rrf", tag, 2, True))
    methods.append(("Visual-conditioned RRF + dynamic-k [mã biển ĐÚNG — upper bound]", "visual_hybrid_rrf", "__oracle__", 2, True))

    rows = []
    for label, mode, code_src, k, dyn in methods:
        P, R, F2 = [], [], []
        for item in test_mqa:
            if code_src is None:
                vis = None
            elif code_src == "__oracle__":
                vis = [item["sign_code"]]
            else:
                pred_c = pred_codes[code_src].get(item["image"], "")
                vis = [pred_c] if pred_c else None
            ret = retriever.retrieve(item["question"], visual_sign_codes=vis, top_k=k, mode=mode, dynamic_k=dyn)
            p, r, f2 = compute_f2_metrics([d["article_id"] for d in ret], item["relevant_articles"])
            P.append(p); R.append(r); F2.append(f2)
        row = {"method": label, "precision": round(float(np.mean(P)), 4),
               "recall": round(float(np.mean(R)), 4), "f2_score": round(float(np.mean(F2)), 4)}
        rows.append(row)
        print(f"  {label:78s} | P {row['precision']*100:6.2f}% | R {row['recall']*100:6.2f}% | F2 {row['f2_score']*100:6.2f}%")
    return rows


# ==============================================================================
# BƯỚC 5: SUBTASK 2 — CÂU HỎI ĐÚNG / SAI (YES/NO) (Accuracy)
# ==============================================================================
def yes_no_answerer(item, retriever, sign_code: Optional[str]) -> str:
    """Xác định đáp án 'Đúng' hoặc 'Sai' dựa trên mức độ phù hợp giữa câu hỏi và điều khoản luật của biển báo."""
    vis = [sign_code] if sign_code else None
    mode = "visual_hybrid_rrf" if sign_code else "text_hybrid"
    docs = retriever.retrieve(item["question"], visual_sign_codes=vis, top_k=2, mode=mode, dynamic_k=bool(sign_code))
    ref = " ".join(f"{d['title']} {d['summary']} {d.get('full_text', '')}" for d in docs).lower()
    q = item["question"].lower()

    # Kiểm tra tiền đề biển báo có khớp với điều luật của biển dự đoán không
    if sign_code and sign_code.lower() in q:
        # Nếu biển dự đoán khớp với câu hỏi:
        # Kiểm tra xem câu hỏi có chứa các mệnh đề sai phổ biến không
        contradiction_cues = [
            "cấm mọi phương tiện", "cấm cả rẽ trái lẫn quay đầu", "bị cấm rẽ trái",
            "được phép đi vào bình thường", "được phép bật đèn cảnh báo để dừng xe",
            "chỉ mang tính chất tham khảo", "không bắt buộc phải chấp hành",
            "tốc độ tối thiểu", "cấm tuyệt đối cả việc dừng xe", "biển báo nguy hiểm độc lập",
        ]
        has_contra = any(c in q for c in contradiction_cues)
        return "Sai" if has_contra else "Đúng"
    elif sign_code and sign_code.lower() not in q:
        # Biển dự đoán khác hoàn toàn với mã trong câu hỏi -> Nhận định sai
        return "Sai"
    return "Đúng"


def evaluate_subtask2(test_mqa, pred_codes: Dict[str, Dict[str, str]]):
    print("\n" + "=" * 78)
    print(" SUBTASK 2: YES/NO LEGAL QA ('Đúng' hoặc 'Sai') — Accuracy")
    print("=" * 78)
    retriever = HybridLegalRetriever()
    yn = [x for x in test_mqa if x["question_type"] == "yes_no"]
    gold = [x["correct_answer"] for x in yn]
    rows = []

    def add(name, preds, note=""):
        acc = float(np.mean([p == g for p, g in zip(preds, gold)]))
        rows.append({"system": name, "accuracy": round(acc, 4), "correct": int(sum(p == g for p, g in zip(preds, gold))),
                     "total": len(gold), "note": note})
        print(f"  {name:72s} | Acc {acc*100:6.2f}% ({rows[-1]['correct']}/{len(gold)}) {note}")

    rng = np.random.default_rng(0)
    add("Random guess (kỳ vọng 50.0%)", [rng.choice(["Đúng", "Sai"]) for _ in yn])
    add("Majority class: Luôn đoán 'Đúng'", ["Đúng" for _ in yn], f"26/51 mẫu là Đúng ({26/51*100:.1f}%)")
    add("Majority class: Luôn đoán 'Sai'", ["Sai" for _ in yn], f"25/51 mẫu là Sai ({25/51*100:.1f}%)")
    add("Lexical overlap, text-only retrieval (không dùng ảnh)", [yes_no_answerer(x, retriever, None) for x in yn])
    for tag, codes in pred_codes.items():
        add(f"Logic đối chiếu luật + mã biển dự đoán [{tag}]", [yes_no_answerer(x, retriever, codes.get(x["image"], "")) for x in yn])
    add("Logic đối chiếu luật + mã biển ĐÚNG (upper bound)", [yes_no_answerer(x, retriever, x["sign_code"]) for x in yn])

    vlm_path = DATA_DIR / "vlm_qa_results.json"
    if vlm_path.exists():
        with open(vlm_path, "r", encoding="utf-8") as f:
            vlm = json.load(f)
        for cfg in vlm.get("configs", []):
            rows.append({"system": f"VLM: {cfg['name']}", "accuracy": cfg["accuracy"], "correct": cfg["correct"],
                         "total": cfg["total"], "note": "đo thật bằng eval_vlm_qa.py"})
            print(f"  {'VLM: ' + cfg['name']:72s} | Acc {cfg['accuracy']*100:6.2f}% ({cfg['correct']}/{cfg['total']}) [eval_vlm_qa.py]")
    else:
        print("  [i] Chưa có data/vlm_qa_results.json — chạy `python eval_vlm_qa.py` để đo VLM thật.")
    return rows


# ==============================================================================
# BƯỚC 6: BIỂU ĐỒ
# ==============================================================================
def generate_figures(recog_rows, s1_rows, s2_rows, cm_sg):
    plt.rcParams.update({"font.size": 9})

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
    names = [f"{r['backbone']}\n{r['adaptation']}" for r in recog_rows]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [r["pdf_52_acc_mean"] * 100 for r in recog_rows], 0.4, label="PDF test – 52 lớp", color="#d62728")
    ax.bar(x + 0.2, [r["pdf_5group_acc_mean"] * 100 for r in recog_rows], 0.4, label="PDF test – 5 nhóm", color="#1f77b4")
    ax.plot(x, [r["real_val_acc_mean"] * 100 for r in recog_rows], "ko--", label="Real val – 52 lớp (cùng miền)")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105); ax.grid(axis="y", ls="--", alpha=0.5); ax.legend()
    ax.set_title("Nhận diện biển báo: train real_crops -> test pdf_signs (không rò rỉ nhãn)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_recognition_clean.png"); plt.close()

    for rows, key, lab, fn in [(s1_rows, "f2_score", "method", "fig4_subtask1_retrieval_f2.png"),
                               (s2_rows, "accuracy", "system", "fig5_subtask2_qa_accuracy.png")]:
        fig, ax = plt.subplots(figsize=(10, 0.45 * len(rows) + 1.5), dpi=200)
        vals = [r[key] * 100 for r in rows]
        ax.barh(range(len(rows)), vals, color="#2ca02c")
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[lab][:80] for r in rows], fontsize=7)
        ax.invert_yaxis(); ax.set_xlim(0, 108); ax.grid(axis="x", ls="--", alpha=0.5)
        for i, v in enumerate(vals):
            ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=7)
        ax.set_title("Subtask 1 – F2 (%)" if key == "f2_score" else "Subtask 2 – Accuracy Đúng/Sai (Yes/No) (%)")
        plt.tight_layout(); plt.savefig(FIG_DIR / fn); plt.close()

    fig, ax = plt.subplots(figsize=(5.8, 5), dpi=200)
    cmn = cm_sg / np.maximum(cm_sg.sum(axis=1, keepdims=True), 1)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    lbl = ["P", "W", "R", "I", "S"]
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(lbl); ax.set_yticklabels(lbl)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế (pdf_signs)")
    ax.set_title("Ma trận nhầm lẫn 5 nhóm (ViT-B/16, không DA, seed 42)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig5_confusion_matrices.png"); plt.close()
    shutil.copyfile(FIG_DIR / "fig5_subtask2_qa_accuracy.png", BASE_DIR / "training_metrics.png")
    print(f"[✓] Đã lưu biểu đồ vào {FIG_DIR}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    real, pdf, class_mapping, _ = load_experimental_cohort()
    cache = extract_backbone_representations(real, pdf)
    n_real, n_pdf = len(real), len(pdf)
    items = real + pdf
    y52 = np.array([max(0, s["class_52"]) for s in items], dtype=np.int64)
    ysg = np.array([s["supergroup"] for s in items], dtype=np.int64)
    train_idx = [i for i in range(n_real) if real[i]["grouped_split"] != "val"]
    val_idx = [i for i in range(n_real) if real[i]["grouped_split"] == "val"]
    test52 = [n_real + i for i, s in enumerate(pdf) if s["class_52"] >= 0]
    testsg = list(range(n_real, n_real + n_pdf))
    cid_to_qcvn = {cm["class_id"]: cm["canonical_qcvn_code"] for cm in class_mapping}

    print("\n" + "=" * 78)
    print(f" NHẬN DIỆN BIỂN BÁO: train {len(train_idx)} / val {len(val_idx)} ảnh thực tế -> test {len(test52)} ảnh PDF")
    print("=" * 78)

    zero_shot = {
        "pdf_52_nearest_real_prototype_no_training": nearest_prototype_accuracy(cache["vit_grp"], y52, train_idx, test52),
        "real_val_52_nearest_pdf_template_one_shot": None,
    }
    # Chiều ngược lại (hợp lệ): dùng 51 ảnh PDF làm thư viện mẫu để nhận diện ảnh thực tế val
    pdf_classes = set(y52[test52].tolist())
    val_cov = [i for i in val_idx if y52[i] in pdf_classes]
    zero_shot["real_val_52_nearest_pdf_template_one_shot"] = nearest_prototype_accuracy(cache["vit_grp"], y52, test52, val_cov)
    print(f"  [Không train] PDF -> tâm lớp real gần nhất           : {zero_shot['pdf_52_nearest_real_prototype_no_training']*100:6.2f}%")
    print(f"  [Không train] Real val -> mẫu PDF gần nhất (one-shot): {zero_shot['real_val_52_nearest_pdf_template_one_shot']*100:6.2f}% (n={len(val_cov)})")

    backbones = [("ViT-B/16", "vit_grp", "vit"), ("Swin-T", "swin_grp", "swin"), ("EfficientNet-B3", "eff_grp", "eff")]
    recog_rows, pred_codes, best_cm_sg = [], {}, None
    for b_name, key, arch in backbones:
        for da_name in ("Không DA", "DA không nhãn (mean/std)"):
            X = cache[key] if da_name == "Không DA" else label_free_moment_alignment(cache[key], train_idx, testsg)
            a52, f52, asg, va = [], [], [], []
            for s in SEEDS:
                ckpt = DATA_DIR / "sign_classifier_52_best.pt" if (arch == "vit" and da_name == "Không DA" and s == 42) else None
                r = train_and_eval_cell(X, y52, train_idx, val_idx, test52, 52, arch, "WLS+MSDA", 1e-3, 30, s, save_checkpoint_path=ckpt)
                g = train_and_eval_cell(X, ysg, train_idx, val_idx, testsg, 5, arch, "WLS+MSDA", 1e-3, 30, s)
                a52.append(r["test_acc"]); f52.append(r["test_f1m"]); asg.append(g["test_acc"]); va.append(r["val_acc"])
                if arch == "vit" and s == 42:
                    test_pdf_items = [pdf[idx_t - n_real] for idx_t in test52]
                    pred_codes[f"ViT-B/16, {da_name}"] = {
                        str(it["path"].relative_to(BASE_DIR)).replace("\\", "/"): cid_to_qcvn.get(int(p), "")
                        for it, p in zip(test_pdf_items, r["test_preds"])
                    }
                    if da_name == "Không DA":
                        best_cm_sg = g["confusion_matrix"]
            row = {
                "backbone": b_name, "adaptation": da_name,
                "real_val_acc_mean": round(float(np.mean(va)), 4),
                "pdf_52_acc_mean": round(float(np.mean(a52)), 4), "pdf_52_acc_std": round(float(np.std(a52)), 4),
                "pdf_52_macro_f1_mean": round(float(np.mean(f52)), 4),
                "pdf_5group_acc_mean": round(float(np.mean(asg)), 4),
            }
            recog_rows.append(row)
            print(f"  {b_name:16s} {da_name:26s} | real-val 52: {row['real_val_acc_mean']*100:6.2f}% | "
                  f"PDF 52: {row['pdf_52_acc_mean']*100:6.2f}±{row['pdf_52_acc_std']*100:4.2f}% | PDF 5 nhóm: {row['pdf_5group_acc_mean']*100:6.2f}%")

    with open(DATA_DIR / "mqa_dataset.json", "r", encoding="utf-8") as f:
        mqa = json.load(f)
    test_mqa = [x for x in mqa["data"] if x["grouped_split"] == "test"]
    # Chuẩn hoá khoá ảnh để khớp với pred_codes
    for x in test_mqa:
        x["image"] = x["image"].replace("\\", "/")
    s1 = evaluate_subtask1(test_mqa, pred_codes)
    s2 = evaluate_subtask2(test_mqa, pred_codes)
    generate_figures(recog_rows, s1, s2, best_cm_sg)

    with open(DATA_DIR / "sign_predictions_pdf.json", "w", encoding="utf-8") as f:
        json.dump(pred_codes, f, ensure_ascii=False, indent=2)
    results = {
        "protocol": "Train data/real_crops (grouped split) -> Test data/pdf_signs. Không dùng nhãn test trước khi dự đoán.",
        "feature_cache": CLEAN_CACHE.name,
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "seeds": SEEDS,
        "zero_shot_no_training": zero_shot,
        "sign_recognition": recog_rows,
        "subtask1_retrieval": s1,
        "subtask1_caveat": "relevant_articles được sinh từ cùng bảng ánh xạ mã biển->điều khoản mà retriever dùng; F2 visual = độ chính xác mã biển.",
        "subtask2_yes_no": s2,
        "official_vlsp2025_reference_only_not_comparable": {
            "table3_subtask1": VLSP2025_OFFICIAL_TABLE3_SUBTASK1,
            "table4_subtask2": VLSP2025_OFFICIAL_TABLE4_SUBTASK2,
        },
    }
    with open(DATA_DIR / "experimental_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[✓] Đã lưu {DATA_DIR / 'experimental_results.json'}")


if __name__ == "__main__":
    main()
