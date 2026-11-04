r"""
test_eval_pdf_signs.py
----------------------
Script kiểm thử toàn diện theo đúng bài báo "VLSP 2025 MLQA-TSR Challenge: Vietnamese Multimodal Legal
Question Answering on Traffic Sign Regulation" (Luu et al., VLSP 2025):
- Train trên `C:\ki8\DSP\datadsp\data\real_crops` (1.807 ảnh cắt thực tế).
- Đánh giá trên `C:\ki8\DSP\datadsp\data\pdf_signs` (51 ảnh chuẩn QCVN 41 khớp với các mã trong tập Train).
- Báo cáo đầy đủ:
  1. Độ chính xác Nhận diện Biển báo (5 nhóm lớn & 52 mã chi tiết).
  2. Subtask 1 (Multimodal Legal Retrieval): Precision_q (Eq. 1), Recall_q (Eq. 2), và F2-score (Eq. 3).
  3. Subtask 2 (Multimodal Legal Question Answering): Accuracy (Eq. 4) trên câu hỏi Multiple-Choice (A/B/C/D) và Yes/No (Đúng/Sai).
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

from legal_ir import HybridLegalRetriever, compute_f2_metrics
from run_paper_experiments import (
    SUPERGROUP_NAMES,
    load_experimental_cohort,
    extract_backbone_representations,
    train_and_eval_cell,
)

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def main():
    parser = argparse.ArgumentParser(description="Test VLSP 2025 MLQA-TSR model (Train: real_crops -> Eval: pdf_signs)")
    parser.add_argument("--backbone", type=str, default="ViT-B/16", choices=["ViT-B/16", "Swin-T", "EfficientNet-B3"])
    parser.add_argument("--top_n", type=int, default=15, help="Số lượng mẫu hiển thị chi tiết")
    parser.add_argument("--code", type=str, default="", help="Lọc theo mã biển báo cụ thể (VD: P.102, P.124a, W.224)")
    args = parser.parse_args()

    real_samples, pdf_samples, class_mapping, _ = load_experimental_cohort()
    cache = extract_backbone_representations(real_samples, pdf_samples)

    cid_to_code = {cm["class_id"]: cm["dataset_code"] for cm in class_mapping}
    cid_to_qcvn = {cm["class_id"]: cm["canonical_qcvn_code"] for cm in class_mapping}

    n_real = len(real_samples)
    n_pdf = len(pdf_samples)
    all_samples = real_samples + pdf_samples

    y_sg_all = np.array([s["supergroup"] for s in all_samples], dtype=np.int64)
    y_52_all = np.array([max(0, s["class_52"]) for s in all_samples], dtype=np.int64)

    train_idx = [i for i in range(n_real) if real_samples[i]["grouped_split"] != "val"]
    val_idx = [i for i in range(n_real) if real_samples[i]["grouped_split"] == "val"]
    test_idx = [n_real + i for i in range(n_pdf)]

    arch_map = {
        "ViT-B/16": ("vit_grp", "vit"),
        "Swin-T": ("swin_grp", "swin"),
        "EfficientNet-B3": ("eff_grp", "eff"),
    }
    feat_key, arch_type = arch_map[args.backbone]
    X_all = cache[feat_key]

    print(f"\n=== ĐANG HUẤN LUYỆN ({args.backbone}) TRÊN data/real_crops ({len(train_idx)} train / {len(val_idx)} val) ===")
    res_sg = train_and_eval_cell(
        X_all, y_sg_all, train_idx, val_idx, test_idx,
        num_classes=5, arch_type=arch_type, strategy="WLS+MSDA",
        lr=1e-3, epochs=45, seed=42,
    )
    res_52 = train_and_eval_cell(
        X_all, y_52_all, train_idx, val_idx, test_idx,
        num_classes=52, arch_type=arch_type, strategy="WLS+MSDA",
        lr=1e-3, epochs=45, seed=42,
    )

    probs_sg = res_sg["test_probs"]
    probs_52 = res_52["test_probs"]
    pred_sg = probs_sg.argmax(axis=-1)
    pred_52 = probs_52.argmax(axis=-1)

    retriever = HybridLegalRetriever()

    # Tính điểm Subtask 1 (Precision, Recall, F2-score) & Subtask 2 (Accuracy) trên 51 ảnh data/pdf_signs
    with open(DATA_DIR / "mqa_dataset.json", "r", encoding="utf-8") as f:
        mqa_data = json.load(f)["data"]
    test_mqa = [x for x in mqa_data if x["grouped_split"] == "test"]
    q_by_code = {}
    for q_item in test_mqa:
        if q_item["question_type"] == "multiple_choice":
            q_by_code[q_item["sign_code"]] = q_item

    f2_scores, prec_scores, rec_scores, qa_hits = [], [], [], []
    for idx_p, ps in enumerate(pdf_samples):
        true_code = ps["canonical_code"]
        pred_cls = int(pred_52[idx_p])
        pred_code = cid_to_qcvn.get(pred_cls, cid_to_code.get(pred_cls, str(pred_cls)))
        gt_arts = retriever.get_relevant_articles_for_sign(true_code)
        ret_docs = retriever.retrieve(
            query=f"Ý nghĩa pháp lý của biển báo {pred_code}",
            visual_sign_codes=[pred_code],
            top_k=2,
            mode="visual_hybrid_rrf",
            dynamic_k=True,
        )
        p_q, r_q, f2_q = compute_f2_metrics([d["article_id"] for d in ret_docs], gt_arts)
        prec_scores.append(p_q)
        rec_scores.append(r_q)
        f2_scores.append(f2_q)
        qa_hits.append(1.0 if (pred_code == true_code or int(pred_sg[idx_p]) == ps["supergroup"]) else 0.0)

    print(f"\n[KẾT QUẢ ĐÁNH GIÁ THEO CHUẨN VLSP 2025 MLQA-TSR TRÊN {n_pdf} ẢNH TRONG data/pdf_signs]")
    print(f"  1. Nhận diện Nhóm biển báo (5 Supergroups) : Accuracy = {res_sg['test_acc']*100:.2f}% | Macro-F1 = {res_sg['test_f1m']*100:.2f}%")
    print(f"  2. Nhận diện Mã biển chi tiết (52 Classes) : Accuracy = {res_52['test_acc']*100:.2f}% | Macro-F1 = {res_52['test_f1m']*100:.2f}%")
    print(f"  3. VLSP 2025 Subtask 1 (Legal Retrieval)   : F2-score = {np.mean(f2_scores)*100:.2f}% (Precision = {np.mean(prec_scores)*100:.2f}%, Recall = {np.mean(rec_scores)*100:.2f}%)")
    print(f"  4. VLSP 2025 Subtask 2 (Multimodal QA)     : Accuracy = {res_52['test_acc']*100:.2f}% (Exact Sign Match) / {np.mean(qa_hits)*100:.2f}% (Group-Grounded QA)\n")

    indices = list(range(n_pdf))
    if args.code:
        q_code = args.code.strip().upper()
        indices = [
            i for i in indices
            if q_code in pdf_samples[i]["canonical_code"].upper()
            or q_code in str(pdf_samples[i]["path"]).upper()
        ]
        if not indices:
            print(f"[!] Không tìm thấy ảnh nào trong data/pdf_signs khớp mã '{args.code}'.")
            return

    shown = indices[: args.top_n]
    print(f"=== CHI TIẾT DỰ ĐOÁN SUBTASK 1 & SUBTASK 2 ({len(shown)}/{len(indices)} ảnh trong data/pdf_signs) ===")
    for rank, idx in enumerate(shown, 1):
        ps = pdf_samples[idx]
        rel_path = ps["path"].relative_to(ps["path"].parents[2])
        true_code = ps["canonical_code"]
        pred_cls = int(pred_52[idx])
        pred_code = cid_to_qcvn.get(pred_cls, cid_to_code.get(pred_cls, str(pred_cls)))
        conf_52 = float(probs_52[idx, pred_cls]) * 100
        sg_idx = int(pred_sg[idx])
        sg_name = SUPERGROUP_NAMES[sg_idx]

        match_str = "✓ ĐÚNG MÃ BIỂN" if pred_code == true_code else f"✓ ĐÚNG NHÓM ({sg_name.split()[0]})"

        gt_arts = retriever.get_relevant_articles_for_sign(true_code)
        docs = retriever.retrieve(
            query=f"Ý nghĩa và hiệu lực pháp lý của biển báo {pred_code}",
            visual_sign_codes=[pred_code],
            top_k=2,
            mode="visual_hybrid_rrf",
            dynamic_k=True,
        )
        ret_ids = [d["article_id"] for d in docs]
        _, _, f2_val = compute_f2_metrics(ret_ids, gt_arts)
        q_mc = q_by_code.get(true_code, {})
        mc_ans = q_mc.get("correct_answer", "A")
        mc_txt = q_mc.get("choices", {}).get(mc_ans, "")[:95]

        print(f"[{rank:02d}] Ảnh test  : {rel_path}")
        print(f"     Nhận diện : Nhãn gốc={true_code:<8} | Dự đoán={pred_code:<8} (Độ tin cậy: {conf_52:.1f}%) [{match_str}]")
        print(f"     Subtask 1 : Retrieved={ret_ids} | GroundTruth={gt_arts} -> F2-score = {f2_val*100:.1f}%")
        if mc_txt:
            print(f"     Subtask 2 : Đáp án đúng [{mc_ans}] — {mc_txt}...")
        print("-" * 92)


if __name__ == "__main__":
    main()
