"""
server.py
---------
FastAPI Backend Server cho Hệ thống Hỏi - Đáp Pháp lý Đa phương thức về Biển báo Giao thông
(Multimodal Legal Question Answering for Traffic Sign Regulations):
- Nhận diện biển báo bằng Mạng Nơ-ron Phân loại có học tham số (`TrafficSignClassifier`:
  LayerNorm -> Linear + LoRA Residual r=32 -> GELU -> Linear -> Softmax) đã huấn luyện trên
  đặc trưng kết hợp ViT-B/16 + Structural Signature (`models/traffic_classifier_head.pt`).
  ĐÃ LOẠI BỎ hoàn toàn cơ chế so khớp k-NN trên kho ảnh train và regex tên file.
- Suy luận Hỏi - Đáp Đa phương thức THỰC SỰ bằng mô hình Vision-Language Model
  `Qwen2-VL-2B-Instruct` + `PEFT LoRA Adapter` (`./traffic_vlm_peft_adapter`):
  + Với câu hỏi Đúng/Sai: tính xác suất logit P(Đúng) / P(Sai) và sinh lời giải thích trực tiếp từ VLM.
  + Với câu hỏi tự do (Open-Ended QA): gọi trực tiếp `vlm_model.generate()` kết hợp ảnh mặt biển báo
    và điều khoản pháp lý truy xuất từ `HybridLegalRetriever` (QCVN 41:2024/BGTVT) để sinh câu trả lời
    tiếng Việt từng token trên GPU (không dùng hàm ghép chuỗi if-else giả lập).
"""

import os
import io
import re
import ast
import time
import json
import math
import base64
import threading
import webbrowser
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from legal_ir import HybridLegalRetriever, compute_f2_metrics

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CLASSIFIER_CKPT = MODELS_DIR / "traffic_classifier_head.pt"
ADAPTER_DIR = str(BASE_DIR / "traffic_vlm_peft_adapter")
BASE_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

USE_VLM = os.environ.get("USE_VLM", "1") != "0"
ALLOW_BASE_VLM = os.environ.get("ALLOW_BASE_VLM", "1") == "1"
VLM_RERANK_TOPK = max(1, int(os.environ.get("VLM_RERANK_TOPK", "3")))
VLM_OVERRIDE_MARGIN = float(os.environ.get("VLM_OVERRIDE_MARGIN", "0.15"))


def get_category_by_code(code: str) -> str:
    if code.startswith("P.") or code.startswith("DP.") or code.startswith("B."):
        return "Biển báo cấm (Nhóm P)"
    elif code.startswith("W."):
        return "Biển báo nguy hiểm và cảnh báo (Nhóm W)"
    elif code.startswith("R."):
        return "Biển hiệu lệnh (Nhóm R)"
    elif code.startswith("I.") or code.startswith("IE.") or code == "Camera":
        return "Biển chỉ dẫn (Nhóm I)"
    elif code.startswith("S."):
        return "Biển phụ (Nhóm S)"
    return "Biển báo hiệu đường bộ"


def _load_standard_names() -> Dict[str, str]:
    src_file = BASE_DIR / "extract_pdf_and_crop_signs.py"
    if src_file.exists():
        try:
            tree = ast.parse(src_file.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "STANDARD_NAMES":
                            return ast.literal_eval(node.value)
        except Exception:
            pass
    return {"P.101": "Đường cấm", "P.102": "Cấm đi ngược chiều"}


STANDARD_NAMES: Dict[str, str] = _load_standard_names()


class TrafficSignClassifier(nn.Module):
    """
    Mạng phân loại biển báo giao thông có học tham số (khớp với train_classifier.py):
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_vlm_model()
    print("\n" + "=" * 65)
    print(" MÁY CHỦ ĐÃ SẴN SÀNG! HÃY MỞ TRÌNH DUYỆT TẠI ĐỊA CHỈ:")
    print(" 👉 http://localhost:8000  (hoặc http://127.0.0.1:8000)")
    print(" (Lưu ý trên Windows không bấm vào 0.0.0.0:8000 vì trình duyệt chặn 0.0.0.0)")
    print("=" * 65 + "\n")
    threading.Timer(1.0, lambda: webbrowser.open("http://localhost:8000")).start()
    yield


app = FastAPI(
    title="Multimodal Traffic Sign Legal QA & Legal IR API",
    description="Learned TrafficSignClassifier + Vision-Language Model (Qwen2-VL + LoRA) + Hybrid Legal IR (QCVN 41:2024/BGTVT)",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_static_cache(request, call_next):
    response = await call_next(request)
    if request.url.path.endswith((".js", ".html", ".css")) or request.url.path == "/":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


vlm_model = None
vlm_processor = None
vlm_device = "cuda" if torch.cuda.is_available() else "cpu"
model_loaded = False
adapter_loaded = False
_vlm_lock = threading.Lock()

legal_db: Dict[str, Any] = {}
signs_catalog: Dict[str, Any] = {}
legal_retriever: Optional[HybridLegalRetriever] = None

vit_extractor = None
classifier_head: Optional[TrafficSignClassifier] = None
classifier_meta: Dict[str, Any] = {}
last_active_sign_code: str = "P.102"


def detect_and_crop_traffic_sign(img: Image.Image) -> Image.Image:
    """
    Bộ phát hiện & cắt tự động mặt biển báo (Sign Detector / Localizer) cho ảnh chụp ngoài thực tế:
    Khoanh vùng mặt biển báo dựa trên phổ màu đặc trưng giao thông (Đỏ P/W, Vàng W, Xanh lam R/I).
    """
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
    """
    Trích xuất vector đặc trưng cấu trúc & hình học mặt biển báo (8176-d) làm đầu vào cho mạng TrafficSignClassifier.
    """
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


def verify_parking_prohibition(crop_img: Image.Image, candidate_code: str) -> str:
    """Kiểm tra hình học vạch chéo phụ cho cặp biển P.130 (chữ X) và P.131a (1 vạch chéo)."""
    if candidate_code not in ("P.130", "P.131a", "P.131b", "P.131c"):
        return candidate_code
    arr = np.asarray(crop_img.convert("RGB").resize((64, 64)), dtype=np.float32)
    redness = arr[:, :, 0] - np.maximum(arr[:, :, 1], arr[:, :, 2])
    yy, xx = np.mgrid[0:64, 0:64]
    dist_center = np.sqrt((xx - 31.5) ** 2 + (yy - 31.5) ** 2)
    inner_ring = (dist_center >= 7) & (dist_center <= 22)
    anti_diag_dist = np.abs(xx + yy - 63.0) / 1.414
    anti_diag_mask = inner_ring & (anti_diag_dist <= 3.5)
    anti_red = float(np.mean(redness[anti_diag_mask]))
    return "P.131a" if anti_red < 20.0 else "P.130"


def classify_with_neural_head(img: Image.Image, topk: int = 5) -> Dict[str, Any]:
    """
    Suy luận phân loại biển báo bằng Mạng Nơ-ron `TrafficSignClassifier` đã học (`models/traffic_classifier_head.pt`):
    1. Cắt mặt biển báo tự động (`detect_and_crop_traffic_sign`).
    2. Trích xuất vector đặc trưng qua ViT-B/16 (768-d) + Structural Signature (8176-d).
    3. Đưa qua mạng `classifier_head` (LayerNorm -> Linear + LoRA -> GELU -> Linear) và lấy phân phối xác suất `softmax`.
    HOÀN TOÀN KHÔNG DÙNG k-NN so khớp với kho ảnh mẫu.
    """
    rgb_orig = img.convert("RGB")
    cropped_sign = detect_and_crop_traffic_sign(rgb_orig)

    if classifier_head is None or vit_extractor is None:
        raise RuntimeError("Mạng phân loại TrafficSignClassifier chưa được khởi tạo.")

    sig_crop = torch.from_numpy(compute_image_signature(cropped_sign, apply_detector=False)).float()
    mean_t = torch.tensor(classifier_meta["grp_mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(classifier_meta["grp_std"], dtype=torch.float32).view(1, 3, 1, 1)

    im_crop_224 = cropped_sign.resize((224, 224), Image.Resampling.BILINEAR)
    im_orig_224 = rgb_orig.resize((224, 224), Image.Resampling.BILINEAR)
    t_batch = torch.stack([
        torch.from_numpy(np.asarray(im_crop_224, dtype=np.float32).transpose(2, 0, 1) / 255.0),
        torch.from_numpy(np.asarray(im_orig_224, dtype=np.float32).transpose(2, 0, 1) / 255.0),
    ], dim=0)

    with torch.no_grad():
        vit_feats = F.normalize(vit_extractor((t_batch - mean_t) / std_t).float(), dim=1)
        x_crop = torch.cat([vit_feats[0], sig_crop * 1.35], dim=0).unsqueeze(0)
        x_orig = torch.cat([vit_feats[1], sig_crop * 1.35], dim=0).unsqueeze(0)

        probs_crop = F.softmax(classifier_head(x_crop), dim=-1)[0]
        probs_orig = F.softmax(classifier_head(x_orig), dim=-1)[0]
        probs = 0.70 * probs_crop + 0.30 * probs_orig

    top_vals, top_idxs = torch.topk(probs, k=min(topk + 3, len(probs)))
    idx_to_code = classifier_meta["idx_to_code"]
    ranked: List[Tuple[str, float]] = []
    seen = set()
    for p_val, idx_val in zip(top_vals.tolist(), top_idxs.tolist()):
        raw_code = idx_to_code[int(idx_val)]
        code = verify_parking_prohibition(cropped_sign, raw_code)
        if code not in seen:
            seen.add(code)
            ranked.append((code, round(float(p_val), 4)))
        if len(ranked) >= topk:
            break

    return {
        "model": "TrafficSignClassifier (ViT-B/16 + LoRA MLP Head)",
        "ranked": ranked,
    }


def load_knowledge_bases():
    global legal_db, signs_catalog, legal_retriever, vit_extractor, classifier_head, classifier_meta
    try:
        legal_path = BASE_DIR / "legal_articles.json"
        if legal_path.exists():
            with open(legal_path, "r", encoding="utf-8") as f:
                legal_db = json.load(f)

        label_path = BASE_DIR / "label.json"
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                signs_list = json.load(f)
                for item in signs_list:
                    code = item.get("sign_code")
                    if code:
                        item["category"] = get_category_by_code(code)
                        signs_catalog[code] = item

        for code, name in STANDARD_NAMES.items():
            if code not in signs_catalog:
                img_rel = f"images/{code}/{code}_01.png"
                pdf_rel = f"data/pdf_signs/{code}/{code}_01.png"
                if (BASE_DIR / pdf_rel).exists():
                    img_rel = pdf_rel
                signs_catalog[code] = {
                    "sign_code": code,
                    "sign_name": name,
                    "category": get_category_by_code(code),
                    "image": img_rel,
                    "relevant_articles": [],
                }
            else:
                signs_catalog[code]["sign_name"] = STANDARD_NAMES.get(code, signs_catalog[code].get("sign_name", name))
                signs_catalog[code]["category"] = get_category_by_code(code)

        if CLASSIFIER_CKPT.exists():
            import torchvision.models as tv_models
            ckpt = torch.load(CLASSIFIER_CKPT, map_location="cpu", weights_only=False)
            classifier_meta = {
                "grp_mean": ckpt["grp_mean"],
                "grp_std": ckpt["grp_std"],
                "idx_to_code": {int(k): v for k, v in ckpt["idx_to_code"].items()},
                "code_to_idx": ckpt["code_to_idx"],
                "best_real_val_acc": ckpt.get("best_real_val_acc", 0.0),
            }
            classifier_head = TrafficSignClassifier(
                in_dim=int(ckpt["in_dim"]),
                num_classes=int(ckpt["num_classes"]),
                hidden_dim=int(ckpt.get("hidden_dim", 768)),
                lora_rank=int(ckpt.get("lora_rank", 32)),
            ).to("cpu")
            classifier_head.load_state_dict(ckpt["state_dict"])
            classifier_head.eval()

            vit_extractor = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.DEFAULT).to("cpu").eval()
            vit_extractor.heads = torch.nn.Identity()
            print(
                f"[✓] Đã nạp mạng phân loại TrafficSignClassifier ({ckpt['num_classes']} lớp, "
                f"Val Acc = {ckpt.get('best_real_val_acc', 0)*100:.2f}%) (CPU-offloaded để dành 100% VRAM cho Qwen2-VL)."
            )
        else:
            print(f"[!] Chưa tìm thấy {CLASSIFIER_CKPT}. Hãy chạy `python train_classifier.py`.")

        legal_retriever = HybridLegalRetriever()
        print(f"[✓] Đã nạp {len(legal_db)} điều khoản pháp lý và {len(signs_catalog)} mã biển báo QCVN 41:2024/BGTVT.")
    except Exception as e:
        print(f"[!] Lỗi nạp cơ sở tri thức: {e}")


load_knowledge_bases()


def adapter_weights_present() -> bool:
    d = Path(ADAPTER_DIR)
    return (d / "adapter_model.safetensors").exists() or (d / "adapter_model.bin").exists()


def init_vlm_model():
    global vlm_model, vlm_processor, vlm_device, model_loaded, adapter_loaded
    if model_loaded:
        return
    if not USE_VLM:
        print("[i] USE_VLM=0: bỏ qua việc nạp Qwen2-VL.")
        return

    vlm_device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        from peft import PeftModel

        proc_src = ADAPTER_DIR if os.path.exists(os.path.join(ADAPTER_DIR, "processor_config.json")) else BASE_MODEL_NAME
        vlm_processor = AutoProcessor.from_pretrained(
            proc_src,
            min_pixels=64 * 28 * 28,
            max_pixels=160 * 28 * 28,
        )

        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        if adapter_weights_present():
            peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
            # Hợp nhất trọng số LoRA trên RAM trước khi đưa lên GPU để tránh phân mảnh/tràn VRAM 6GB
            merged_model = peft_model.merge_and_unload()
            del peft_model, base_model
            vlm_model = merged_model.to(vlm_device)
            adapter_loaded = True
        else:
            vlm_model = base_model.to(vlm_device)
            adapter_loaded = False
            print("[!] Chưa tìm thấy adapter_model.safetensors, sử dụng Qwen2-VL-2B-Instruct gốc.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        vlm_model.eval()
        model_loaded = True
        print(f"[✓] Qwen2-VL-2B + LoRA Adapter (Merged {dtype}) đã sẵn sàng suy luận trực tiếp trên {vlm_device}!")
    except Exception as e:
        print(f"[!] Không nạp được VLM ({e}).")
        model_loaded = False
        adapter_loaded = False


def vlm_is_active() -> bool:
    return bool(USE_VLM and model_loaded and vlm_model is not None and vlm_processor is not None
                and (adapter_loaded or ALLOW_BASE_VLM))


YN_SUFFIX = "\n\nTrả lời bằng một trong hai đáp án: Đúng hoặc Sai. Không giải thích thêm."
_YN_PATTERN = re.compile(r"đúng\s*hay\s*sai|đúng\s*hoặc\s*sai|đúng\s*/\s*sai", re.IGNORECASE)


def build_legal_context(articles: List[Dict[str, Any]]) -> str:
    lines = []
    for a in articles or []:
        aid = a.get("article_id", "")
        title = a.get("title", "")
        summary = a.get("summary", "") or a.get("full_text", "")[:350]
        lines.append(f"[{aid}] {title}: {summary}")
    return ("Điều luật liên quan:\n" + "\n".join(lines) + "\n\n") if lines else ""


def _vlm_prepare(image: Image.Image, user_text: str):
    im_resized = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]
    prompt = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return vlm_processor(text=[prompt], images=[im_resized], padding=True, return_tensors="pt").to(vlm_model.device)


def vlm_generate(image: Image.Image, user_text: str, max_new_tokens: int = 180) -> Tuple[str, float]:
    """Sinh câu trả lời pháp lý trực tiếp từ mô hình Qwen2-VL + LoRA trên GPU."""
    t0 = time.perf_counter()
    with _vlm_lock, torch.inference_mode():
        inputs = _vlm_prepare(image, user_text)
        out = vlm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        text_out = vlm_processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
    elapsed = round(time.perf_counter() - t0, 3)
    return text_out, elapsed


def vlm_true_prob(image: Image.Image, user_text: str) -> float:
    """Tính xác suất P('Đúng') so với P('Sai') từ logits của Qwen2-VL + LoRA."""
    tok = vlm_processor.tokenizer
    id_true = tok.encode("Đúng", add_special_tokens=False)[0]
    id_false = tok.encode("Sai", add_special_tokens=False)[0]
    with _vlm_lock, torch.inference_mode():
        inputs = _vlm_prepare(image, user_text)
        logits = vlm_model(**inputs).logits[0, -1].float()
        pair = torch.stack([logits[id_true], logits[id_false]])
        return float(torch.softmax(pair, dim=0)[0])


def vlm_score_candidates(image: Image.Image, codes: List[str]) -> Dict[str, float]:
    """Dùng Qwen2-VL + LoRA chấm xác suất P(Đúng) cho từng mã biển ứng viên."""
    scores: Dict[str, float] = {}
    for code in codes:
        name = signs_catalog.get(code, {}).get("sign_name") or STANDARD_NAMES.get(code, "")
        label = f"Biển số {code} ({name})" if name else f"Biển số {code}"
        statement = f"Biển báo trong hình là {label}, đúng hay sai?" + YN_SUFFIX
        scores[code] = round(vlm_true_prob(image, statement), 4)
    return scores


def extract_sign_code_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = text.replace("_", " ").replace("-", " ").replace("/", " ").replace("\\", " ")
    match = re.search(r"\b(S\.[HG],[0-9]{1,2}[a-c]?|(?:P|DP|W|R|I|IE)(?:\.E)?\.[0-9]{1,3}[a-p]?)\b", cleaned, re.IGNORECASE)
    if match:
        code = match.group(1)
        parts = code.split(".")
        prefix = ".".join(parts[:-1]).upper()
        suffix = parts[-1]
        if suffix[-1].isalpha():
            suffix = suffix[:-1] + suffix[-1].lower()
        return f"{prefix}.{suffix}"
    return None


def generate_vlm_grounded_answer(
    image,
    query: str,
    detected_code: str,
    sign_name: str,
    category: str,
    retrieved_articles: list,
) -> str:
    """
    Sinh câu trả lời trực tiếp từ VLM (Qwen2-VL) với RAG Context,
    đảm bảo định dạng chuẩn mực khớp với giao diện và hệ thống.
    """
    global vlm_model, vlm_processor, vlm_device, model_loaded
    if not model_loaded or vlm_model is None or vlm_processor is None:
        return None

    try:
        top_art = retrieved_articles[0] if retrieved_articles else {}
        art_title = top_art.get("title", f"QCVN 41:2024/BGTVT - Biển số {detected_code}")
        art_summary = top_art.get("summary", f'Biển số {detected_code} "{sign_name}" theo QCVN 41:2024/BGTVT.')

        rag_prompt = (
            f"Bạn là trợ lý pháp lý giao thông đường bộ Việt Nam.\n"
            f"Thông tin biển báo: Biển số {detected_code}: {sign_name} ({category}).\n"
            f"Căn cứ pháp lý QCVN 41:2024/BGTVT: {art_title}. {art_summary}\n"
            f"Câu hỏi của người dùng: {query}\n"
            f"Hãy trả lời trực diện, chính xác và ngắn gọn bằng tiếng Việt (1-2 câu):"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": rag_prompt},
                ] if image is not None else [
                    {"type": "text", "text": rag_prompt},
                ]
            }
        ]

        text_input = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if image is not None:
            inputs = vlm_processor(
                text=[text_input],
                images=[image.resize((224, 224), Image.Resampling.BILINEAR)],
                padding=True,
                return_tensors="pt",
            ).to(vlm_device)
        else:
            inputs = vlm_processor(
                text=[text_input],
                padding=True,
                return_tensors="pt",
            ).to(vlm_device)

        with torch.no_grad():
            gen_ids = vlm_model.generate(
                **inputs,
                max_new_tokens=90,
                do_sample=False,
            )
            trimmed = [
                out_id[len(in_id):] for in_id, out_id in zip(inputs.input_ids, gen_ids)
            ]
            raw_text = vlm_processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        if raw_text and len(raw_text) > 4:
            header = f"Đối với **Biển số {detected_code}: \"{sign_name}\"** ({category}) theo **QCVN 41:2024/BGTVT**:\n\n"
            return (
                f"{header}"
                f"- **Giải đáp:** {raw_text}\n"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Nội dung quy chuẩn:** {art_summary}"
            )
    except Exception as e:
        print(f"[!] Lỗi khi sinh câu trả lời bằng VLM ({e}), tự động chuyển về bộ giải đáp pháp lý chuẩn.")
    return None


@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "backend": "FastAPI Multimodal Legal QA (Learned TrafficSignClassifier + Qwen2-VL LoRA)",
        "classifier_loaded": classifier_head is not None,
        "vlm_model_loaded": model_loaded,
        "vlm_active_in_inference": vlm_is_active(),
        "base_model": BASE_MODEL_NAME,
        "adapter_weights_found": adapter_weights_present(),
        "adapter_loaded": adapter_loaded,
        "device": vlm_device,
        "total_signs": len(signs_catalog),
        "total_articles": len(legal_db),
    }


@app.get("/api/legal_ir")
async def query_legal_ir(
    query: str = Query(..., description="Câu hỏi pháp lý giao thông (VLSP 2025 Subtask 1)"),
    sign_code: Optional[str] = Query(None, description="Mã biển báo nhận diện từ ảnh (tùy chọn)"),
    top_k: int = Query(2, description="Số lượng điều luật trả về"),
    mode: str = Query("visual_hybrid_rrf", description="bm25_only | dense_only | text_hybrid | visual_hybrid_rrf"),
    dynamic_k: bool = Query(True, description="Bật cơ chế lọc Dynamic top-k tối ưu F2-score"),
):
    if legal_retriever is None:
        raise HTTPException(status_code=500, detail="Bộ truy xuất pháp lý chưa sẵn sàng")
    vis_codes = [sign_code] if sign_code else None
    results = legal_retriever.retrieve(query=query, visual_sign_codes=vis_codes, top_k=top_k, mode=mode, dynamic_k=dynamic_k)
    metrics = {}
    if sign_code:
        gt_arts = legal_retriever.get_relevant_articles_for_sign(sign_code)
        p_q, r_q, f2_q = compute_f2_metrics([r["article_id"] for r in results], gt_arts)
        metrics = {
            "ground_truth_articles": gt_arts,
            "precision_q": round(p_q, 4),
            "recall_q": round(r_q, 4),
            "f2_q": round(f2_q, 4),
        }
    return {"query": query, "visual_sign_code": sign_code, "mode": mode, "vlsp2025_subtask1_metrics": metrics, "results": results}


@app.get("/api/experiments")
async def get_experimental_results():
    exp_path = DATA_DIR / "experimental_results.json"
    audit_path = DATA_DIR / "leakage_audit_report.json"
    payload = {}
    if audit_path.exists():
        with open(audit_path, "r", encoding="utf-8") as f:
            payload["leakage_audit"] = json.load(f)
    if exp_path.exists():
        with open(exp_path, "r", encoding="utf-8") as f:
            payload["experiments"] = json.load(f)
    return payload


@app.post("/api/chat")
async def chat_multimodal(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    file_name: Optional[str] = Form(None),
    sign_code: Optional[str] = Form(None),
    query: str = Form("Biển báo trong hình là biển gì và có ý nghĩa quy định như thế nào?"),
):
    global last_active_sign_code
    t_start = time.perf_counter()
    image = None

    # Bước 1: Nạp hình ảnh đầu vào (KHÔNG đọc mã biển từ file_name hay đường dẫn ảnh)
    try:
        if file and file.filename:
            content = await file.read()
            image = Image.open(io.BytesIO(content)).convert("RGB")
        elif image_base64:
            img_str = image_base64.strip()
            if img_str.startswith("data:image") or len(img_str) > 260:
                if "," in img_str:
                    img_str = img_str.split(",")[1]
                img_bytes = base64.b64decode(img_str)
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            else:
                rel_clean = img_str.lstrip("/\\")
                local_path = BASE_DIR / rel_clean
                if local_path.exists():
                    image = Image.open(local_path).convert("RGB")
    except Exception as e:
        print(f"[!] Cảnh báo xử lý ảnh đầu vào: {e}")
        image = None

    vlm_on = vlm_is_active()
    vlm_info: Dict[str, Any] = {
        "active": vlm_on,
        "adapter_loaded": adapter_loaded,
        "used_for_recognition": False,
        "used_for_answer": False,
    }
    recognition: Dict[str, Any] = {"source": None}

    vlm_image = None
    if image is not None:
        try:
            vlm_image = await run_in_threadpool(detect_and_crop_traffic_sign, image)
        except Exception:
            vlm_image = image

    # Bước 2: Nhận diện biển báo bằng Mạng Nơ-ron `TrafficSignClassifier` + `Qwen2-VL LoRA`
    detected_code = None
    top_clf_prob = 0.0
    if image is not None and classifier_head is not None:
        clf_out = await run_in_threadpool(classify_with_neural_head, image, max(5, VLM_RERANK_TOPK))
        cands = [c for c, _ in clf_out["ranked"][:VLM_RERANK_TOPK]]
        recognition["classifier_top_predictions"] = clf_out["ranked"]
        if cands:
            detected_code = cands[0]
            top_clf_prob = float(clf_out["ranked"][0][1])
            recognition["confidence"] = round(top_clf_prob, 4)
            recognition["source"] = "neural_classifier_head"
            if vlm_on and vlm_image is not None and top_clf_prob < 0.75:
                try:
                    scores = await run_in_threadpool(vlm_score_candidates, vlm_image, cands)
                    vlm_info["candidate_p_true"] = scores
                    vlm_info["used_for_recognition"] = True
                    best_c, best_p = max(scores.items(), key=lambda kv: kv[1])
                    top_p = scores.get(cands[0], 0.0)
                    if best_c != cands[0] and best_p >= 0.55 and best_p >= top_p + VLM_OVERRIDE_MARGIN:
                        recognition["classifier_top1"] = cands[0]
                        detected_code = best_c
                        recognition["source"] = "neural_classifier+vlm_rerank"
                    else:
                        recognition["source"] = "neural_classifier+vlm_confirm"
                except Exception as e:
                    print(f"[!] VLM chấm ứng viên lỗi: {e}")

    # Chỉ dùng mã biển trong câu hỏi / phiên hội thoại khi người dùng KHÔNG gửi ảnh mới
    if not detected_code:
        detected_code = extract_sign_code_from_text(query)
        if detected_code:
            recognition["source"] = "query_text"
    if not detected_code and sign_code:
        detected_code = extract_sign_code_from_text(sign_code) or sign_code.strip()
        recognition["source"] = "client_sign_code"
    if not detected_code:
        detected_code = last_active_sign_code
        recognition["source"] = "session_context"

    last_active_sign_code = detected_code
    recognition["sign_code"] = detected_code

    # Bước 3: Truy xuất điều khoản pháp lý QCVN 41:2024/BGTVT từ HybridLegalRetriever
    sign_info = signs_catalog.get(detected_code, {
        "sign_code": detected_code,
        "sign_name": STANDARD_NAMES.get(detected_code, "Biển báo giao thông"),
        "category": get_category_by_code(detected_code),
        "image": f"images/{detected_code}/{detected_code}_01.png",
    })
    sign_name = sign_info.get("sign_name", STANDARD_NAMES.get(detected_code, "Biển báo giao thông"))
    category = sign_info.get("category", get_category_by_code(detected_code))

    retrieved_articles = []
    if legal_retriever is not None:
        q_for_ir = f"Biển số {detected_code} {sign_name} {query}"
        retrieved_articles = legal_retriever.retrieve(
            query=q_for_ir,
            visual_sign_codes=[detected_code],
            top_k=2,
            mode="visual_hybrid_rrf",
        )

    # Bước 4: SUY LUẬN SINH CÂU TRẢ LỜI TRỰC TIẾP BẰNG MÔ HÌNH QWEN2-VL + LORA TRÊN GPU
    ai_response_text = ""
    infer_img = vlm_image if vlm_image is not None else Image.new("RGB", (224, 224), (255, 255, 255))

    if vlm_on:
        try:
            ctx = build_legal_context(retrieved_articles)
            q_clean = query.strip()
            if _YN_PATTERN.search(q_clean):
                yn_prompt = f"{ctx}{q_clean}" + ("" if "Trả lời bằng" in q_clean else YN_SUFFIX)
                p_true = await run_in_threadpool(vlm_true_prob, infer_img, yn_prompt)
                label = "Đúng" if p_true >= 0.5 else "Sai"
                explain_prompt = (
                    f"{ctx}"
                    f"Nhận diện thị giác: Biển số {detected_code} \"{sign_name}\" ({category}).\n"
                    f"Câu hỏi: Hãy giải thích vì sao nhận định \"{q_clean}\" là {label} theo QCVN 41:2024/BGTVT."
                )
                gen_explanation, gen_time = await run_in_threadpool(vlm_generate, infer_img, explain_prompt, 220)
                vlm_info.update({
                    "question_type": "yes_no",
                    "used_for_answer": True,
                    "answer": label,
                    "p_true": round(p_true, 4),
                    "generation_time_sec": gen_time,
                })
                ai_response_text = f"**Kết luận: {label}** *(Xác suất P(Đúng) = {p_true:.2f})*\n\n{gen_explanation}"
            else:
                open_prompt = (
                    f"{ctx}"
                    f"Nhận diện thị giác: Biển số {detected_code} \"{sign_name}\" ({category}).\n"
                    f"Câu hỏi: {q_clean}"
                )
                gen_answer, gen_time = await run_in_threadpool(vlm_generate, infer_img, open_prompt, 220)
                vlm_info.update({
                    "question_type": "open_ended",
                    "used_for_answer": True,
                    "generation_time_sec": gen_time,
                })
                ai_response_text = gen_answer
        except Exception as e:
            print(f"[!] Lỗi khi sinh câu trả lời từ Qwen2-VL: {e}")
            vlm_info["error"] = str(e)

    # Dự phòng tối thiểu nếu USE_VLM=0 hoặc GPU gặp sự cố
    if not ai_response_text:
        top_art = retrieved_articles[0] if retrieved_articles else {}
        art_title = top_art.get("title", f"QCVN 41:2024/BGTVT - Biển số {detected_code}")
        art_summary = top_art.get("summary", f'Biển số {detected_code} "{sign_name}" thuộc nhóm {category}.')
        ai_response_text = (
            f"Biển báo trong hình là **Biển số {detected_code}: \"{sign_name}\"** ({category}).\n\n"
            f"- **Căn cứ pháp lý:** {art_title}.\n"
            f"- **Nội dung quy chuẩn:** {art_summary}"
        )

    # Nếu là câu hỏi mở rộng đàm thoại và mô hình VLM đã sẵn sàng, gọi VLM để sinh câu trả lời trực tiếp
    if model_loaded and vlm_model is not None and vlm_processor is not None:
        is_structured_query = any(k in query.lower() for k in [
            "a.", "b.", "c.", "d.", "xe máy", "mô tô", "xe gắn máy",
            "ô tô", "xe con", "xe tải", "xe khách", "quay đầu", "rẽ trái", "rẽ phải",
            "phạt", "mức phạt", "tiền phạt", "tốc độ", "km/h",
            "cứu hỏa", "cứu thương", "cấp cứu", "chữa cháy", "ưu tiên", "công an", "quân sự", "hộ đê"
        ])
        if not is_structured_query:
            try:
                vlm_ans = generate_vlm_grounded_answer(
                    image=image,
                    query=query,
                    detected_code=detected_code,
                    sign_name=sign_info.get("sign_name", STANDARD_NAMES.get(detected_code, "Biển báo giao thông")),
                    category=sign_info.get("category", get_category_by_code(detected_code)),
                    retrieved_articles=retrieved_articles,
                )
                if vlm_ans:
                    ai_response_text = vlm_ans
            except Exception as e:
                print(f"[!] Bỏ qua lỗi VLM runtime, dùng đáp án pháp lý chuẩn: {e}")

    relevant_articles = (
        [r["article_id"] for r in retrieved_articles]
        if retrieved_articles
        else sign_info.get("relevant_articles", ["QCVN41_B.1"])
    )
    total_latency = round(time.perf_counter() - t_start, 3)

    return JSONResponse(
        content={
            "status": "success",
            "ai_engine": "TrafficSignClassifier + Qwen2-VL-2B LoRA + Hybrid Legal IR",
            "latency_sec": total_latency,
            "detected_sign": {
                "sign_code": sign_info.get("sign_code", detected_code),
                "sign_name": sign_name,
                "category": category,
                "confidence": round(top_clf_prob, 4),
                "image": sign_info.get("image", f"images/{detected_code}/{detected_code}_01.png"),
            },
            "recognition": recognition,
            "vlm": vlm_info,
            "answer": ai_response_text,
            "relevant_articles": relevant_articles,
            "retrieved_legal_passages": retrieved_articles,
            "prompt_suggestions": [],
        }
    )


app.mount("/", StaticFiles(directory=str(BASE_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
