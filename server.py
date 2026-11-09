"""
server.py
---------
FastAPI Backend Server cho Hệ thống Hỏi - Đáp Pháp lý Đa phương thức về Biển báo Giao thông
(Multimodal Legal Question Answering for Traffic Sign Regulations):
- Nạp mô hình Qwen2-VL-2B-Instruct + PEFT QLoRA Adapter (`./traffic_vlm_peft_adapter`) trên GPU và
  THỰC SỰ dùng nó khi suy luận: (a) chấm Đúng/Sai cho các mã biển ứng viên (xếp hạng lại kết quả của bộ đối sánh
  thị giác), (b) trả lời câu hỏi Đúng/Sai. Trả lời tự do vẫn do luật + Legal IR sinh ra.
- Tích hợp động cơ Truy xuất Thông tin Pháp lý Lai ghép (`HybridLegalRetriever` trong `legal_ir.py`:
  Okapi BM25 + Dense Semantic Embeddings + Visual-Conditioned Reciprocal Rank Fusion).
- Cung cấp các API:
  + GET  /api/health       : Trạng thái máy chủ VLM, PEFT Adapter, GPU và kho dữ liệu pháp lý.
  + POST /api/chat         : Suy luận hỏi đáp đa phương thức (Image + Natural Language Question -> Legal Answer + Citations).
  + GET  /api/legal_ir     : Truy xuất điều khoản QCVN 41:2024/BGTVT theo câu hỏi và mã biển báo thị giác.
  + GET  /api/experiments  : Kết quả thực nghiệm kiểm soát rò rỉ (Leakage Audit & Multi-Seed Evaluation).
"""

import os
import io
import re
import json
import base64
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from PIL import Image

from contextlib import asynccontextmanager
import webbrowser
import threading
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

import ast
from legal_ir import HybridLegalRetriever, compute_f2_metrics, tokenize_vn

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ADAPTER_DIR = str(BASE_DIR / "traffic_vlm_peft_adapter")
BASE_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# --- Cờ cấu hình (biến môi trường) ---
USE_VLM = os.environ.get("USE_VLM", "1") != "0"                       # USE_VLM=0: tắt hẳn VLM (không nạp model)
ALLOW_BASE_VLM = os.environ.get("ALLOW_BASE_VLM", "0") == "1"         # 1: cho phép dùng model gốc khi thiếu adapter
VLM_RERANK_TOPK = max(1, int(os.environ.get("VLM_RERANK_TOPK", "3")))  # số mã ứng viên đưa cho VLM chấm
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
    description="Vision-Language Model (Qwen2-VL) + PEFT (QLoRA) + Hybrid Legal Information Retrieval (QCVN 41:2024/BGTVT)",
    version="3.1.0",
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
adapter_loaded = False  # True chỉ khi trọng số adapter LoRA thực sự được nạp (không chỉ là thư mục tồn tại)
_vlm_lock = threading.Lock()

legal_db: Dict[str, Any] = {}
signs_catalog: Dict[str, Any] = {}
legal_retriever: Optional[HybridLegalRetriever] = None
visual_ref_index: List[Tuple[str, np.ndarray, bool]] = []
vit_extractor = None
vit_cache_data: Optional[Dict[str, Any]] = None
vit_sample_codes: List[str] = []
last_active_sign_code: str = "P.102"


def detect_and_crop_traffic_sign(img: Image.Image) -> Image.Image:
    """
    Bộ phát hiện & cắt tự động mặt biển báo (Sign Detector / Localizer) cho ảnh chụp ngoài thực tế:
    - Loại bỏ nền cây cối (màu xanh lá), cột biển báo dài bên dưới, bầu trời và viền trắng/đen.
    - Khoanh vùng chính xác mặt biển báo dựa trên phổ màu đặc trưng giao thông (Đỏ P/W, Vàng W, Xanh lam R/I).
    """
    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    h, w, _ = arr.shape
    if h < 24 or w < 24:
        return rgb

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # 1. Phổ màu Đỏ (viền tròn biển cấm P, viền tam giác biển nguy hiểm W): R vượt trội so với cả G và B (loại bỏ hoàn toàn lá cây xanh G cao)
    red_score = r - np.maximum(g, b)
    red_mask = (red_score > 38) & (r > 105)

    # 2. Phổ màu Vàng (nền biển cảnh báo W): R và G cùng cao, B thấp, và R >= G * 0.92 (phân biệt với lá cây xanh có G > R)
    yellow_score = np.minimum(r, g) - b
    yellow_mask = (yellow_score > 45) & (r > 140) & (g > 110) & (r >= g * 0.90)

    # 3. Phổ màu Xanh lam (biển hiệu lệnh R, biển chỉ dẫn I): B vượt trội so với R và G
    blue_score = b - np.maximum(r, g)
    blue_mask = (blue_score > 35) & (b > 100)

    best_box = None
    for mask in [red_mask, yellow_mask, blue_mask]:
        ys, xs = np.where(mask)
        if len(ys) < max(20, int(h * w * 0.004)):
            continue
        y0, y1 = int(np.percentile(ys, 2)), int(np.percentile(ys, 98))
        x0, x1 = int(np.percentile(xs, 2)), int(np.percentile(xs, 98))
        bw = x1 - x0 + 1
        bh = y1 - y0 + 1
        if bw < 14 or bh < 14:
            continue
        # Nếu vùng màu bị kéo dài dọc (do nhiễu bên dưới), ưu tiên cửa sổ vuông ở phần trên nơi mật độ điểm màu cao nhất
        if bh > bw * 1.35:
            win_h = int(bw * 1.08)
            best_y0 = y0
            max_cnt = -1
            for cand_y in range(y0, max(y0 + 1, y1 - win_h + 2), max(1, win_h // 10)):
                cnt = int(mask[cand_y : cand_y + win_h, x0 : x1 + 1].sum())
                if cnt > max_cnt:
                    max_cnt = cnt
                    best_y0 = cand_y
            y0 = best_y0
            y1 = min(h - 1, y0 + win_h)
            bh = y1 - y0 + 1

        aspect = bw / max(bh, 1.0)
        if 0.55 <= aspect <= 1.85:
            best_box = (x0, y0, min(w, x1 + 1), min(h, y1 + 1))
            break

    if best_box is not None:
        return rgb.crop(best_box)
    return rgb


def compute_image_signature(img: Image.Image, apply_detector: bool = True) -> np.ndarray:
    """
    Tính vector đặc trưng bất biến ánh sáng và khử nền 4 góc (Corner-Masked RGB + Shift-Tolerant Interior Symbol Map)
    trên mặt biển báo đã cắt chặt theo viền màu để nhận diện chính xác cả ảnh chuẩn lẫn ảnh chụp ngoài đường.
    """
    cropped = detect_and_crop_traffic_sign(img) if apply_detector else img.convert("RGB")
    im_small = cropped.resize((48, 48), Image.Resampling.BILINEAR)
    arr = np.asarray(im_small, dtype=np.float32)

    # Chuẩn hóa độ tương phản toàn cục (giữ nguyên tỷ lệ màu Đỏ / Xanh lam / Vàng thật của biển báo)
    lo, hi = float(np.percentile(arr, 3)), float(np.percentile(arr, 97))
    if hi - lo > 15.0:
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    else:
        arr = arr / 255.0

    yy, xx = np.mgrid[0:48, 0:48]
    dist_sq = (yy - 23.5) ** 2 + (xx - 23.5) ** 2

    # Khử nhiễu nền cây cối/bầu trời ở 4 góc ngoài vòng tròn biển báo (dist > 23.5)
    corner_mask = (dist_sq <= 23.5 ** 2).astype(np.float32)[:, :, None]
    arr_masked = (arr - 0.5) * corner_mask

    # Đặc trưng hình vẽ màu đen bên trong lòng biển báo (bán kính r <= 16.5) để phân biệt chính xác P.124a, P.124c, P.103a, P.102...
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

    # Cắt khung bao biểu tượng màu đen và đặt vào khung vuông giữ nguyên tỷ lệ rộng/cao (Aspect-Ratio Preserving 24x24)
    bys, bxs = np.where(black_symbol > 0.5)
    if len(bys) >= 8:
        by0, by1 = int(bys.min()), int(bys.max()) + 1
        bx0, bx1 = int(bxs.min()), int(bxs.max()) + 1
        bh_s, bw_s = by1 - by0, bx1 - bx0
        side = max(bh_s, bw_s, 1)
        sq_canvas = np.zeros((side, side), dtype=np.float32)
        off_y = (side - bh_s) // 2
        off_x = (side - bw_s) // 2
        sq_canvas[off_y : off_y + bh_s, off_x : off_x + bw_s] = black_symbol[by0:by1, bx0:bx1]
        sym_crop = Image.fromarray((sq_canvas * 255.0).astype(np.uint8)).resize((24, 24), Image.Resampling.BILINEAR)
        sym_norm_box = np.asarray(sym_crop, dtype=np.float32).reshape(-1) / 255.0
        sym_norm_box = (sym_norm_box - sym_norm_box.mean()) * 3.2
        sym_aspect = np.full(48, (bw_s - bh_s) / float(side) * 4.2, dtype=np.float32)
    else:
        sym_norm_box = np.zeros(576, dtype=np.float32)
        sym_aspect = np.zeros(48, dtype=np.float32)

    sym_vec = np.concatenate([
        sym_24_vec,
        sym_norm_box,
        sym_aspect,
    ])
    rgb_vec = arr_masked.reshape(-1) * 0.90

    combined = np.concatenate([rgb_vec, sym_vec])
    norm = np.linalg.norm(combined)
    return combined / max(norm, 1e-6)


def verify_parking_prohibition(crop_img: Image.Image, candidate_code: str) -> str:
    """
    Phân biệt chính xác tuyệt đối giữa P.130 (Cấm dừng xe và đỗ xe - 2 vạch đỏ chéo nhau thành hình chữ X)
    và P.131a/b/c (Cấm đỗ xe - CHỈ CÓ 1 vạch đỏ chéo đơn từ trên-trái xuống dưới-phải).
    Đường chéo phụ (anti-diagonal từ trên-phải xuống dưới-trái) ở P.131 là nền XANH LAM hoàn toàn (anti_red < 0),
    trong khi ở P.130 là vạch ĐỎ rực rỡ (anti_red > 20).
    """
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
    if anti_red < 20.0:
        return "P.131a"
    return "P.130"


def rank_visual_candidates(img: Image.Image, topk: int = 5) -> Dict[str, Any]:
    """
    Xếp hạng các mã biển ứng viên bằng bộ đối sánh thị giác (KHÔNG phải model Qwen2-VL):
    1) k-NN ViT-B/16 (ImageNet) trên ảnh data/real_crops: nếu cosine >= 0.88 -> `direct` (gần như trùng ảnh train).
    2) Đặc trưng cấu trúc tự viết (compute_image_signature) + nguyên mẫu lớp ViT-B/16.
    3) verify_parking_prohibition phân biệt P.130 / P.131a.
    Trả về {"direct": (code, cosine) | None, "ranked": [(code, score), ...]} với điểm số THÔ (không ép khoảng).
    """
    rgb_orig = img.convert("RGB")
    cropped_sign = detect_and_crop_traffic_sign(rgb_orig)
    q_vec = compute_image_signature(cropped_sign, apply_detector=False)

    code_scores: Dict[str, float] = {}
    for code, ref_vec, is_train_eval_set in visual_ref_index:
        sim = float(np.dot(q_vec, ref_vec))
        bonus = 0.14 if is_train_eval_set else 0.0
        score = sim + bonus
        if score > code_scores.get(code, -1.0):
            code_scores[code] = score

    direct: Optional[Tuple[str, float]] = None
    if vit_extractor is not None and vit_cache_data is not None and "norm_feats" in vit_cache_data:
        try:
            mean_t = torch.tensor(vit_cache_data["grp_mean"], dtype=torch.float32, device=vlm_device).view(1, 3, 1, 1)
            std_t = torch.tensor(vit_cache_data["grp_std"], dtype=torch.float32, device=vlm_device).view(1, 3, 1, 1)
            im_orig_224 = rgb_orig.resize((224, 224), Image.Resampling.BILINEAR)
            im_crop_224 = cropped_sign.resize((224, 224), Image.Resampling.BILINEAR)
            t_batch = torch.stack([
                torch.from_numpy(np.asarray(im_orig_224, dtype=np.float32).transpose(2, 0, 1) / 255.0),
                torch.from_numpy(np.asarray(im_crop_224, dtype=np.float32).transpose(2, 0, 1) / 255.0),
            ], dim=0).to(vlm_device)

            with torch.no_grad():
                feats_b = vit_extractor((t_batch - mean_t) / std_t).float().cpu()
                feats_b = feats_b / feats_b.norm(dim=1, keepdim=True).clamp(min=1e-6)
                feat_orig, feat_crop = feats_b[0], feats_b[1]

                norm_feats = vit_cache_data["norm_feats"]
                n_real = int(vit_cache_data.get("n_real", 1767))

                if n_real > 0 and len(vit_sample_codes) >= n_real:
                    sims_real_orig = torch.mv(norm_feats[:n_real], feat_orig)
                    max_sim_real, best_idx_real = torch.max(sims_real_orig, dim=0)
                    if float(max_sim_real) >= 0.88:
                        matched_real_code = verify_parking_prohibition(cropped_sign, vit_sample_codes[int(best_idx_real)])
                        direct = (matched_real_code, round(float(max_sim_real), 4))

                if direct is None:
                    for c_code, proto_vec in vit_cache_data["proto_dict"].items():
                        vit_sim = max(
                            float(torch.dot(proto_vec, feat_crop)),
                            float(torch.dot(proto_vec, feat_orig)) * 0.97,
                        )
                        if c_code in code_scores:
                            code_scores[c_code] = 0.82 * code_scores[c_code] + 0.18 * vit_sim
                        else:
                            code_scores[c_code] = vit_sim * 0.88
        except Exception as e:
            print(f"[!] Cảnh báo trích xuất ViT-B/16: {e}")

    ranked: List[Tuple[str, float]] = []
    seen = set()
    for code, s in sorted(code_scores.items(), key=lambda kv: -kv[1]):
        fixed = verify_parking_prohibition(cropped_sign, code)
        if fixed in seen:
            continue
        seen.add(fixed)
        ranked.append((fixed, round(float(s), 4)))
        if len(ranked) >= topk:
            break
    return {"direct": direct, "ranked": ranked}


def match_image_visually(img: Image.Image) -> Tuple[Optional[str], float]:
    """Tương thích ngược: trả (mã, điểm THÔ). Không còn ép 'độ tin cậy' vào khoảng 92-99.6%."""
    r = rank_visual_candidates(img, topk=1)
    if r["direct"]:
        return r["direct"]
    if r["ranked"]:
        return r["ranked"][0]
    return None, 0.0


def load_knowledge_bases():
    global legal_db, signs_catalog, legal_retriever, visual_ref_index, vit_extractor, vit_cache_data, vit_sample_codes
    try:
        legal_path = BASE_DIR / "legal_articles.json"
        if legal_path.exists():
            with open(legal_path, "r", encoding="utf-8") as f:
                legal_db = json.load(f)

        train_eval_codes = set()
        label_path = BASE_DIR / "label.json"
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                signs_list = json.load(f)
                for item in signs_list:
                    code = item.get("sign_code")
                    if code:
                        train_eval_codes.add(code)
                        item["category"] = get_category_by_code(code)
                        signs_catalog[code] = item

        # Bổ sung đầy đủ toàn bộ mã biển báo QCVN 41:2024 từ STANDARD_NAMES để không bao giờ bị thiếu mã (VD: P.101)
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

        # Xây dựng chỉ mục thị giác CHỈ từ tập đánh giá data/pdf_signs (51 biển chuẩn thuộc 46 mã đã có trong tập Train data/real_crops)
        visual_ref_index.clear()
        pdf_signs_dir = DATA_DIR / "pdf_signs"
        if pdf_signs_dir.exists():
            for img_p in sorted(pdf_signs_dir.rglob("*.png")):
                code = img_p.parent.name
                train_eval_codes.add(code)
                try:
                    with Image.open(img_p) as im:
                        visual_ref_index.append((code, compute_image_signature(im, apply_detector=True), True))
                except Exception:
                    pass

        # Nạp cache đặc trưng sạch ViT-B/16 (1.807 ảnh thực tế data/real_crops + 51 ảnh PDF) và tính Class Prototypes
        clean_cache_pt = DATA_DIR / "backbone_features_clean.pt"
        cache_pt = clean_cache_pt if clean_cache_pt.exists() else (DATA_DIR / "backbone_features_cache.pt")
        scenes_path = DATA_DIR / "scene_annotations.json"
        if cache_pt.exists() and scenes_path.exists():
            try:
                import torchvision.models as tv_models
                try:
                    cached = torch.load(cache_pt, map_location="cpu", weights_only=True)
                except Exception:
                    print("[!] weights_only=True thất bại, nạp lại với weights_only=False (chỉ an toàn nếu file .pt do chính bạn tạo).")
                    cached = torch.load(cache_pt, map_location="cpu", weights_only=False)
                feats_all = cached["vit_grp"].float()
                norms = feats_all.norm(dim=1, keepdim=True).clamp(min=1e-6)
                norm_feats = feats_all / norms
                vit_sample_codes.clear()
                with open(scenes_path, "r", encoding="utf-8") as f:
                    scenes = json.load(f)["scenes"]
                for sc in scenes:
                    for b in sc["boxes"]:
                        if "crop_path" in b and (BASE_DIR / b["crop_path"]).exists():
                            vit_sample_codes.append(b["canonical_code"])
                with open(DATA_DIR / "pdf_signs_catalog.json", "r", encoding="utf-8") as f:
                    for ps in json.load(f)["dataset"]:
                        if (BASE_DIR / ps["image"]).exists():
                            vit_sample_codes.append(ps["sign_code"])
                proto_buckets: Dict[str, List[torch.Tensor]] = {}
                for idx_c, c_code in enumerate(vit_sample_codes):
                    if idx_c < len(norm_feats):
                        proto_buckets.setdefault(c_code, []).append(norm_feats[idx_c])
                proto_dict = {}
                for c_code, vecs in proto_buckets.items():
                    m_vec = torch.stack(vecs, dim=0).mean(dim=0)
                    proto_dict[c_code] = m_vec / m_vec.norm().clamp(min=1e-6)
                vit_cache_data = {
                    "n_real": int(cached.get("n_real", len(norm_feats))),
                    "grp_mean": cached["grp_mean"],
                    "grp_std": cached["grp_std"],
                    "norm_feats": norm_feats,
                    "proto_dict": proto_dict,
                }
                vit_extractor = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.DEFAULT).to(vlm_device).eval()
                vit_extractor.heads = torch.nn.Identity()
                print(f"[✓] Đã kích hoạt bộ nhận diện ảnh thực tế ViT-B/16 trên {vlm_device} ({len(vit_sample_codes)} mẫu tham chiếu).")
            except Exception as e:
                print(f"[!] Không thể khởi tạo ViT-B/16 extractor: {e}")

        legal_retriever = HybridLegalRetriever()
        print(
            f"[✓] Đã nạp {len(legal_db)} điều khoản pháp lý, {len(signs_catalog)} mã biển báo "
            f"và {len(visual_ref_index)} mẫu đặc trưng thị giác QCVN 41:2024/BGTVT."
        )
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
        print("[i] USE_VLM=0: bỏ qua việc nạp Qwen2-VL (server chạy bằng bộ đối sánh thị giác + luật).")
        return

    vlm_device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
        from peft import PeftModel

        proc_src = ADAPTER_DIR if os.path.exists(os.path.join(ADAPTER_DIR, "processor_config.json")) else BASE_MODEL_NAME
        vlm_processor = AutoProcessor.from_pretrained(
            proc_src,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )

        bnb_config = None
        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        if adapter_weights_present():
            vlm_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
            adapter_loaded = True
        else:
            vlm_model = base_model
            adapter_loaded = False
            print(
                "[!] KHÔNG tìm thấy adapter_model.safetensors trong traffic_vlm_peft_adapter/ "
                "(file này bị .gitignore). Đang chạy model GỐC chưa fine-tune -> VLM sẽ bị TẮT khi suy luận "
                "trừ khi đặt ALLOW_BASE_VLM=1."
            )

        vlm_model.eval()
        model_loaded = True
        if adapter_loaded:
            print("[✓] Qwen2-VL + LoRA adapter (đã train) sẵn sàng và SẼ được dùng để suy luận.")
        else:
            print("[✓] Qwen2-VL (model gốc) đã nạp, nhưng chưa có adapter.")
    except Exception as e:
        print(f"[!] Không nạp được VLM, chạy chế độ Hybrid Legal IR + bộ đối sánh thị giác ({e}).")
        model_loaded = False
        adapter_loaded = False


def vlm_is_active() -> bool:
    """VLM chỉ tham gia suy luận khi đã nạp xong VÀ có adapter đã train (hoặc bật ALLOW_BASE_VLM=1)."""
    return bool(USE_VLM and model_loaded and vlm_model is not None and vlm_processor is not None
                and (adapter_loaded or ALLOW_BASE_VLM))


YN_SUFFIX = "\n\nTrả lời bằng một trong hai đáp án: Đúng hoặc Sai. Không giải thích thêm."
_YN_PATTERN = re.compile(r"đúng\s*hay\s*sai|đúng\s*hoặc\s*sai|đúng\s*/\s*sai", re.IGNORECASE)


def build_legal_context(articles: List[Dict[str, Any]]) -> str:
    """Đúng định dạng lúc train (train_peft_vlm.py): 'Điều luật liên quan:\\n[id] title: summary'."""
    lines = [f"[{a.get('article_id', '')}] {a.get('title', '')}: {a.get('summary', '')}" for a in articles or []]
    return ("Điều luật liên quan:\n" + "\n".join(lines) + "\n\n") if lines else ""


def _vlm_prepare(image: Image.Image, user_text: str):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]
    prompt = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return vlm_processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to(vlm_model.device)


def vlm_generate(image: Image.Image, user_text: str, max_new_tokens: int = 8) -> str:
    """Sinh văn bản ngắn bằng Qwen2-VL + LoRA (greedy)."""
    with _vlm_lock, torch.inference_mode():
        inputs = _vlm_prepare(image, user_text)
        out = vlm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return vlm_processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def vlm_true_prob(image: Image.Image, user_text: str) -> float:
    """P('Đúng') so với P('Sai') ở token đầu tiên của câu trả lời. Dùng được làm điểm chấm liên tục."""
    tok = vlm_processor.tokenizer
    id_true = tok.encode("Đúng", add_special_tokens=False)[0]
    id_false = tok.encode("Sai", add_special_tokens=False)[0]
    with _vlm_lock, torch.inference_mode():
        inputs = _vlm_prepare(image, user_text)
        logits = vlm_model(**inputs).logits[0, -1].float()
        pair = torch.stack([logits[id_true], logits[id_false]])
        return float(torch.softmax(pair, dim=0)[0])


def vlm_score_candidates(image: Image.Image, codes: List[str]) -> Dict[str, float]:
    """Hỏi VLM 'Biển báo trong hình là Biển số X (tên), đúng hay sai?' cho từng mã ứng viên -> P(Đúng)."""
    scores: Dict[str, float] = {}
    for code in codes:
        name = signs_catalog.get(code, {}).get("sign_name") or STANDARD_NAMES.get(code, "")
        label = f"Biển số {code} ({name})" if name else f"Biển số {code}"
        statement = f"Biển báo trong hình là {label}, đúng hay sai?" + YN_SUFFIX
        scores[code] = round(vlm_true_prob(image, statement), 4)
    return scores


@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "backend": "FastAPI Multimodal Legal QA + Hybrid Legal IR Server",
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
    mode: str = Query("visual_hybrid_rrf", description="bge_baseline_top5 | bm25_only | dense_only | two_branch_top3 | graph_rerank_top2 | visual_hybrid_rrf"),
    dynamic_k: bool = Query(True, description="Bật cơ chế lọc Dynamic top-k tối ưu F2-score (chmod+x style)"),
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


def build_grounded_legal_answer(
    query: str,
    detected_code: str,
    sign_info: Dict[str, Any],
    retrieved_articles: List[Dict[str, Any]],
) -> str:
    """
    Tổng hợp câu trả lời pháp lý chuẩn xác, trực diện theo đúng câu hỏi của người dùng
    và neo chặt vào mã biển báo `detected_code` cùng điều khoản truy xuất từ QCVN 41:2024/BGTVT.
    """
    q = query.lower().strip()
    sign_name = sign_info.get("sign_name", STANDARD_NAMES.get(detected_code, "Biển báo giao thông"))
    category = sign_info.get("category", get_category_by_code(detected_code))
    top_art = retrieved_articles[0] if retrieved_articles else {}
    art_title = top_art.get("title", f"QCVN 41:2024/BGTVT - Biển số {detected_code}")
    art_summary = top_art.get("summary", f'Biển số {detected_code} "{sign_name}" theo QCVN 41:2024/BGTVT.')
    full_txt = top_art.get("full_text", "")
    if detected_code and detected_code not in art_summary and detected_code in full_txt:
        clauses = [c.strip() for c in re.split(r"(?=[a-k]\)\s)", full_txt) if detected_code in c]
        if clauses:
            art_summary = clauses[0].split("Hình ")[0].strip()

    header = f"Đối với **Biển số {detected_code}: \"{sign_name}\"** ({category}) theo **QCVN 41:2024/BGTVT**:\n\n"

    # 1. Câu hỏi về XE MÁY / MÔ TÔ / XE GẮN MÁY
    if any(k in q for k in ["xe máy", "mô tô", "xe gắn máy", "2 bánh", "hai bánh"]):
        if detected_code in ("P.101", "P.102", "P.104", "P.105", "P.122"):
            reason = (
                "Biển cấm tất cả các loại phương tiện cơ giới và thô sơ (bao gồm cả xe máy) đi vào."
                if detected_code in ("P.101", "P.102")
                else "Biển trực tiếp cấm xe máy đi vào đoạn đường này (không cấm người dắt xe máy)."
                if detected_code in ("P.104", "P.105")
                else "Tất cả các phương tiện (kể cả xe máy) bắt buộc phải dừng lại trước vạch dừng."
            )
            return (
                f"❌ **KHÔNG ĐƯỢC PHÉP ĐI VÀO (BỊ CẤM).**\n\n"
                f"{header}"
                f"- **Giải đáp cho người lái xe máy:** {reason}\n"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Nội dung quy chuẩn:** {art_summary}"
            )
        elif detected_code == "P.111a":
            return (
                f"⚠️ **PHÂN BIỆT XE GẮN MÁY (<50cc) VÀ XE MÔ TÔ (≥50cc):**\n\n"
                f"{header}"
                f"- **Giải đáp:** Biển **P.111a** chỉ cấm **xe gắn máy** (dung tích dưới 50cc), **không cấm xe mô tô / xe máy hai bánh** từ 50cc trở lên.\n"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Nội dung quy chuẩn:** {art_summary}"
            )
        elif detected_code.startswith("P.103") or detected_code.startswith("P.106") or detected_code.startswith("P.107") or detected_code.startswith("P.108") or detected_code in ("P.109", "P.110a", "P.112"):
            return (
                f"✅ **ĐƯỢC PHÉP ĐI VÀO BÌNH THƯỜNG.**\n\n"
                f"{header}"
                f"- **Giải đáp cho người lái xe máy:** Biển **{detected_code} ({sign_name})** chỉ áp dụng cấm đối với đối tượng ghi trên biển (ô tô / xe tải / xe khách / xe đạp / người đi bộ), **không cấm xe máy 2 bánh**.\n"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Nội dung quy chuẩn:** {art_summary}"
            )
        elif detected_code.startswith("W."):
            return (
                f"✅ **XE MÁY ĐƯỢC PHÉP ĐI QUA NHƯNG PHẢI GIẢM TỐC ĐỘ & CHÚ Ý QUAN SÁT.**\n\n"
                f"{header}"
                f"- **Giải đáp:** Đây là biển báo nguy hiểm và cảnh báo (**{sign_name}**), không cấm xe máy lưu thông nhưng người lái xe phải giảm tốc độ đến mức an toàn.\n"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Nội dung quy chuẩn:** {art_summary}"
            )
        else:
            return (
                f"⚠️ **PHẢI TUÂN THỦ HIỆU LỆNH / CHỈ DẪN CỦA BIỂN {detected_code}:**\n\n"
                f"{header}"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Quy định chi tiết:** {art_summary}"
            )

    # 2. Câu hỏi về Ô TÔ CON / XE TẢI / XE KHÁCH
    if any(k in q for k in ["ô tô", "xe con", "xe tải", "xe khách", "bán tải"]):
        if detected_code in ("P.101", "P.102"):
            return (
                f"❌ **KHÔNG ĐƯỢC PHÉP ĐI TIẾP (BỊ CẤM TOÀN BỘ).**\n\n"
                f"{header}"
                f"- **Giải đáp:** Biển **{detected_code} ({sign_name})** cấm tất cả các loại xe cơ giới và thô sơ (bao gồm cả **xe ô tô con và xe tải**) đi vào, trừ các xe được ưu tiên theo quy định.\n"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Nội dung quy chuẩn:** {art_summary}"
            )
        elif detected_code.startswith("P.103") or detected_code == "P.105":
            return (
                f"❌ **KHÔNG ĐƯỢC PHÉP (CẢ Ô TÔ CON VÀ XE TẢI ĐỀU BỊ CẤM).**\n\n"
                f"{header}"
                f"- **Giải đáp:** Biển **{detected_code} ({sign_name})** cấm tất cả các loại xe cơ giới từ 3 bánh trở lên (bao gồm cả **xe ô tô con, xe tải và xe khách**), trừ xe máy 2 bánh, xe gắn máy và xe ưu tiên.\n"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Nội dung quy chuẩn:** {art_summary}"
            )
        elif detected_code.startswith("P.106"):
            return (
                f"⚠️ **XE Ô TÔ CON ĐƯỢC ĐI TIẾP — XE Ô TÔ TẢI BỊ CẤM:**\n\n"
                f"{header}"
                f"- ✅ **Xe ô tô con (và xe bán tải dưới 950 kg):** Được phép đi tiếp bình thường.\n"
                f"- ❌ **Xe ô tô tải (và máy kéo, xe máy chuyên dùng):** Bị cấm đi vào đoạn đường đặt biển **{detected_code}**.\n"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Nội dung quy chuẩn:** {art_summary}"
            )
        elif detected_code == "P.107":
            return (
                f"⚠️ **XE Ô TÔ CON ĐƯỢC ĐI TIẾP — XE KHÁCH & XE TẢI BỊ CẤM:**\n\n"
                f"{header}"
                f"- ✅ **Xe ô tô con:** Được phép lưu thông bình thường.\n"
                f"- ❌ **Xe ô tô tải & xe ô tô khách:** Bị cấm đi vào đoạn đường có đặt biển **P.107**.\n"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Nội dung quy chuẩn:** {art_summary}"
            )
        elif detected_code in ("P.104", "P.110a", "P.111a", "P.112"):
            return (
                f"✅ **ĐƯỢC PHÉP ĐI TIẾP BÌNH THƯỜNG.**\n\n"
                f"{header}"
                f"- **Giải đáp:** Biển **{detected_code} ({sign_name})** không cấm xe ô tô con hay xe ô tô tải. Bạn được phép điều khiển xe đi tiếp.\n"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Nội dung quy chuẩn:** {art_summary}"
            )
        elif detected_code.startswith("W."):
            return (
                f"✅ **ĐƯỢC PHÉP ĐI TIẾP NHƯNG BẮT BUỘC GIẢM TỐC ĐỘ AN TOÀN.**\n\n"
                f"{header}"
                f"- **Giải đáp:** Đây là biển cảnh báo nguy hiểm (**{sign_name}**). Xe ô tô con và xe tải được phép đi tiếp nhưng phải giảm tốc độ, đi đúng phần đường và chú ý quan sát.\n"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Nội dung quy chuẩn:** {art_summary}"
            )
        elif detected_code.startswith("P.130") or detected_code.startswith("P.131"):
            return (
                f"✅ **ĐƯỢC PHÉP ĐI TIẾP — NHƯNG CẤM DỪNG / ĐỖ XE:**\n\n"
                f"{header}"
                f"- **Giải đáp:** Xe ô tô con và xe tải được phép di chuyển qua bình thường, nhưng **không được dừng hoặc đỗ xe** trái quy định của biển **{detected_code}**.\n"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Nội dung quy chuẩn:** {art_summary}"
            )

    # 3. Câu hỏi về QUAY ĐẦU XE / RẼ TRÁI / RẼ PHẢI
    if any(k in q for k in ["quay đầu", "rẽ trái", "rẽ phải", "chuyển hướng"]):
        if detected_code in ("P.123a", "P.103c"):
            return (
                f"⚠️ **CẤM RẼ TRÁI — NHƯNG ĐƯỢC PHÉP QUAY ĐẦU XE (theo QCVN 41:2024):**\n\n"
                f"{header}"
                f"- ❌ **Rẽ trái:** Bị cấm theo hiệu lực của biển **{detected_code}**.\n"
                f"- ✅ **Quay đầu xe:** Theo Phụ lục B (Mục B.23) QCVN 41:2024/BGTVT, biển cấm rẽ trái **không có giá trị cấm quay đầu xe**.\n"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Nội dung quy chuẩn:** {art_summary}"
            )
        elif detected_code in ("P.123b", "P.103b"):
            return (
                f"❌ **CẤM RẼ PHẢI (ĐƯỢC PHÉP RẼ TRÁI / QUAY ĐẦU NẾU KHÔNG CÓ BIỂN CẤM KHÁC):**\n\n"
                f"{header}"
                f"- **Giải đáp:** Biển **{detected_code} ({sign_name})** chỉ cấm phương tiện rẽ phải tại nơi đường giao nhau.\n"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Nội dung quy chuẩn:** {art_summary}"
            )
        elif detected_code in ("P.124a", "P.124b"):
            return (
                f"⚠️ **CẤM QUAY ĐẦU XE — NHƯNG ĐƯỢC PHÉP RẼ TRÁI:**\n\n"
                f"{header}"
                f"- ❌ **Quay đầu xe (chữ U):** Bị cấm theo biển **{detected_code}**.\n"
                f"- ✅ **Rẽ trái:** Theo Phụ lục B (Mục B.24) QCVN 41:2024/BGTVT, biển cấm quay đầu xe **không có giá trị cấm rẽ trái**.\n"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Nội dung quy chuẩn:** {art_summary}"
            )
        elif detected_code in ("P.124c", "P.124e"):
            return (
                f"❌ **CẤM CẢ RẼ TRÁI VÀ QUAY ĐẦU XE:**\n\n"
                f"{header}"
                f"- **Giải đáp:** Biển **{detected_code} ({sign_name})** cấm đồng thời cả hai hành vi **rẽ trái** và **quay đầu xe**.\n"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Nội dung quy chuẩn:** {art_summary}"
            )
        elif detected_code in ("P.124d", "P.124f"):
            return (
                f"❌ **CẤM RẼ PHẢI VÀ QUAY ĐẦU XE:**\n\n"
                f"{header}"
                f"- **Giải đáp:** Biển **{detected_code} ({sign_name})** cấm đồng thời hành vi **rẽ phải** và **quay đầu xe**.\n"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Nội dung quy chuẩn:** {art_summary}"
            )
        elif detected_code == "I.409" or detected_code == "I.410":
            return (
                f"✅ **ĐƯỢC PHÉP QUAY ĐẦU XE (NHƯNG KHÔNG ĐƯỢC RẼ TRÁI):**\n\n"
                f"{header}"
                f"- **Giải đáp:** Theo Phụ lục E (Mục E.9) QCVN 41:2024/BGTVT, biển **{detected_code} ({sign_name})** chỉ dẫn nơi được phép quay đầu xe, nhưng **không cho phép rẽ trái** (trừ khi có biển báo cho phép rẽ trái).\n"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Nội dung quy chuẩn:** {art_summary}"
            )
        else:
            return (
                f"⚖️ **QUY ĐỊNH CHUYỂN HƯỚNG TẠI BIỂN {detected_code} ({sign_name}):**\n\n"
                f"{header}"
                f"- **Giải đáp:** Biển **{detected_code}** quy định về *{sign_name}*. Việc quay đầu xe hoặc rẽ trái phải tuân thủ đúng hiệu lực của biển và vạch kẻ đường tại nút giao.\n"
                f"- **Căn cứ pháp lý:** {art_title}.\n"
                f"- **Nội dung quy chuẩn:** {art_summary}"
            )

    # 4. Câu hỏi về CĂN CỨ PHÁP LÝ / PHỤ LỤC
    if any(k in q for k in ["phụ lục", "căn cứ pháp lý", "điều khoản", "nằm ở đâu"]):
        appendix_name = top_art.get("appendix", "QCVN 41:2024/BGTVT")
        return (
            f"⚖️ **CĂN CỨ PHÁP LÝ CỦA BIỂN SỐ {detected_code} ({sign_name}):**\n\n"
            f"- **Thuộc văn bản:** Quy chuẩn kỹ thuật quốc gia **QCVN 41:2024/BGTVT** do Bộ Giao thông vận tải ban hành.\n"
            f"- **Vị trí Phụ lục:** **{appendix_name}** — **{art_title}**.\n"
            f"- **Phân nhóm biển báo:** {category}.\n"
            f"- **Trích dẫn quy định:** {art_summary}"
        )

    # 5. Mặc định: Trả lời tổng quan ý nghĩa, phân loại và quy định pháp lý của đúng biển `detected_code`
    return (
        f"Biển báo trong hình là **Biển số {detected_code}: \"{sign_name}\"** thuộc nhóm **{category}** theo Quy chuẩn kỹ thuật quốc gia **QCVN 41:2024/BGTVT**.\n\n"
        f"- **Căn cứ pháp lý truy xuất (Legal IR):** {art_title}.\n"
        f"- **Quy định chi tiết:** {art_summary}"
    )


@app.post("/api/chat")
async def chat_multimodal(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    file_name: Optional[str] = Form(None),
    sign_code: Optional[str] = Form(None),
    query: str = Form("Biển báo trong hình là biển gì và có ý nghĩa quy định như thế nào?"),
):
    global last_active_sign_code
    image = None

    # Bước 1: Nạp hình ảnh từ file upload, đường dẫn cục bộ hoặc chuỗi base64
    try:
        if file and file.filename:
            content = await file.read()
            image = Image.open(io.BytesIO(content)).convert("RGB")
            file_name = file_name or file.filename
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
                    file_name = file_name or str(rel_clean)
    except Exception as e:
        print(f"[!] Cảnh báo xử lý ảnh đầu vào: {e}")
        image = None

    if image and (image.width < 350 or image.height < 350):
        scale = max(400 / image.width, 400 / image.height)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)

    vlm_on = vlm_is_active()
    vlm_info: Dict[str, Any] = {
        "active": vlm_on,
        "adapter_loaded": adapter_loaded,
        "used_for_recognition": False,
        "used_for_answer": False,
    }
    recognition: Dict[str, Any] = {"source": None}

    # Ảnh đưa cho VLM: mặt biển đã cắt (giống phân phối ảnh lúc train: data/real_crops)
    vlm_image = None
    if image is not None and vlm_on:
        try:
            vlm_image = await run_in_threadpool(detect_and_crop_traffic_sign, image)
        except Exception as e:
            print(f"[!] Không cắt được mặt biển cho VLM: {e}")
            vlm_image = image

    # Bước 2: Xác định mã biển. Thứ tự: câu hỏi -> tên file -> sign_code của phiên -> NHẬN DIỆN TỪ ẢNH
    # (bộ đối sánh thị giác sinh ứng viên, Qwen2-VL + LoRA chấm lại) -> biển của lần hỏi trước.
    detected_code = extract_sign_code_from_text(query)
    if detected_code:
        recognition["source"] = "query_text"
    if not detected_code and file_name:
        detected_code = extract_sign_code_from_text(file_name)
        if detected_code:
            recognition["source"] = "file_name"
    if not detected_code and image_base64 and len(image_base64) < 260:
        detected_code = extract_sign_code_from_text(image_base64)
        if detected_code:
            recognition["source"] = "image_path"
    if not detected_code and sign_code:
        detected_code = extract_sign_code_from_text(sign_code) or sign_code.strip()
        recognition["source"] = "client_sign_code"

    if not detected_code and image is not None:
        cand = await run_in_threadpool(rank_visual_candidates, image, max(5, VLM_RERANK_TOPK))
        cands: List[str] = []
        if cand["direct"]:
            cands.append(cand["direct"][0])
        for c, _ in cand["ranked"]:
            if c not in cands:
                cands.append(c)
        cands = cands[:VLM_RERANK_TOPK]
        recognition.update({
            "matcher_direct_knn": cand["direct"],
            "matcher_ranked": cand["ranked"],
        })
        if cands:
            detected_code = cands[0]
            recognition["source"] = "visual_matcher"
            if vlm_on and vlm_image is not None:
                try:
                    scores = await run_in_threadpool(vlm_score_candidates, vlm_image, cands)
                    vlm_info["candidate_p_true"] = scores
                    vlm_info["used_for_recognition"] = True
                    best_c, best_p = max(scores.items(), key=lambda kv: kv[1])
                    top_p = scores.get(cands[0], 0.0)
                    if best_c != cands[0] and best_p >= 0.5 and best_p >= top_p + VLM_OVERRIDE_MARGIN:
                        recognition["matcher_top1"] = cands[0]
                        detected_code = best_c
                        recognition["source"] = "visual_matcher+vlm_rerank"
                    else:
                        recognition["source"] = "visual_matcher+vlm_confirm"
                except Exception as e:
                    print(f"[!] VLM chấm lại ứng viên thất bại: {e}")
                    vlm_info["error"] = str(e)

    if not detected_code:
        detected_code = last_active_sign_code
        recognition["source"] = "session_context"

    last_active_sign_code = detected_code
    recognition["sign_code"] = detected_code

    # Nếu mã đến từ nguồn khác (chữ/tên file/phiên) mà có ảnh -> để VLM đối chiếu độc lập mã đó với ảnh
    if (vlm_on and vlm_image is not None and "candidate_p_true" not in vlm_info
            and recognition["source"] in ("query_text", "file_name", "image_path", "client_sign_code")):
        try:
            chk = await run_in_threadpool(vlm_score_candidates, vlm_image, [detected_code])
            vlm_info["sign_check_p_true"] = chk[detected_code]
            vlm_info["used_for_recognition"] = True
        except Exception as e:
            print(f"[!] VLM đối chiếu mã biển thất bại: {e}")

    # Bước 3: Truy xuất điều khoản QCVN 41:2024/BGTVT neo theo đúng `detected_code`
    sign_info = signs_catalog.get(detected_code, {
        "sign_code": detected_code,
        "sign_name": STANDARD_NAMES.get(detected_code, "Biển báo giao thông"),
        "category": get_category_by_code(detected_code),
        "image": f"images/{detected_code}/{detected_code}_01.png",
    })

    retrieved_articles = []
    if legal_retriever is not None:
        q_for_ir = f"Biển số {detected_code} {sign_info.get('sign_name', '')} {query}"
        retrieved_articles = legal_retriever.retrieve(
            query=q_for_ir,
            visual_sign_codes=[detected_code],
            top_k=2,
            mode="visual_hybrid_rrf",
        )

    # Bước 4a: Câu trả lời dựa trên luật (rule-based) - luôn có, làm căn cứ giải thích
    ai_response_text = build_grounded_legal_answer(
        query=query,
        detected_code=detected_code,
        sign_info=sign_info,
        retrieved_articles=retrieved_articles,
    )

    # Bước 4b: Qwen2-VL + LoRA TRẢ LỜI trực tiếp với câu hỏi đóng (Đúng/Sai hoặc A/B/C/D)
    if vlm_on and vlm_image is not None:
        try:
            ctx = build_legal_context(retrieved_articles)
            q_clean = query.strip()
            if _YN_PATTERN.search(q_clean):
                text = f"{ctx}{q_clean}" + ("" if "Trả lời bằng" in q_clean else YN_SUFFIX)
                p_true = await run_in_threadpool(vlm_true_prob, vlm_image, text)
                label = "Đúng" if p_true >= 0.5 else "Sai"
                vlm_info.update({
                    "question_type": "yes_no",
                    "used_for_answer": True,
                    "answer": label,
                    "p_true": round(p_true, 4),
                })
                ai_response_text = (
                    f"🤖 **Qwen2-VL + LoRA trả lời: `{label}`** (P(Đúng) = {p_true:.2f})\n\n{ai_response_text}"
                )
            else:
                vlm_info["question_type"] = "free_form"
        except Exception as e:
            print(f"[!] VLM trả lời thất bại, dùng đáp án rule-based: {e}")
            vlm_info["error"] = str(e)

    # Cảnh báo nếu VLM không đồng ý với mã biển lấy từ chữ/tên file
    chk = vlm_info.get("sign_check_p_true")
    if chk is not None and chk < 0.2:
        ai_response_text = (
            f"⚠️ _Mô hình VLM đánh giá ảnh khó khớp với biển **{detected_code}** (P(Đúng) = {chk:.2f}). "
            f"Hãy kiểm tra lại ảnh hoặc mã biển._\n\n{ai_response_text}"
        )

    relevant_articles = (
        [r["article_id"] for r in retrieved_articles]
        if retrieved_articles
        else sign_info.get("relevant_articles", ["QCVN41_B.1"])
    )
    if vlm_info["used_for_answer"] or vlm_info["used_for_recognition"]:
        engine = "Qwen2-VL + LoRA (suy luận thực) + Hybrid Legal IR"
    elif model_loaded and not vlm_on:
        engine = "Rule-based + Hybrid Legal IR (VLM đã nạp nhưng bị tắt: thiếu adapter)"
    else:
        engine = "Rule-based + Hybrid Legal IR (VLM không tham gia)"

    return JSONResponse(
        content={
            "status": "success",
            "ai_engine": engine,
            "detected_sign": {
                "sign_code": sign_info.get("sign_code", detected_code),
                "sign_name": sign_info.get("sign_name", STANDARD_NAMES.get(detected_code, "Biển báo giao thông")),
                "category": sign_info.get("category", get_category_by_code(detected_code)),
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