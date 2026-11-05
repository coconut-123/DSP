r"""
eval_vlm_qa.py
--------------
Đánh giá THỰC TẾ mô hình Vision-Language Model (Qwen2-VL-2B-Instruct) trên 51 câu hỏi
Đúng / Sai (Yes/No: "Đúng" hoặc "Sai") trích xuất từ 51 ảnh chuẩn QCVN 41 (data/pdf_signs).

Đo lường trung thực:
1. Qwen2-VL-2B-Instruct Zero-Shot (không fine-tune, không context pháp lý)
2. Qwen2-VL-2B-Instruct + Grounded Legal Retrieval (Context từ QCVN 41:2024)
3. Qwen2-VL-2B-Instruct + PEFT LoRA Adapter (huấn luyện trên data/real_crops)

Tất cả kết quả lưu vào data/vlm_qa_results.json để run_paper_experiments.py nạp vào bảng tổng hợp.
"""

import os
import sys
import json
import re
from pathlib import Path
from PIL import Image

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

from legal_ir import HybridLegalRetriever

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MQA_FILE = DATA_DIR / "mqa_dataset.json"
ADAPTER_DIR = BASE_DIR / "traffic_vlm_peft_adapter"
OUT_FILE = DATA_DIR / "vlm_qa_results.json"

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_yes_no(generated_text: str) -> str:
    """Trích xuất đáp án 'Đúng' hoặc 'Sai' từ văn bản sinh ra của mô hình."""
    gen = generated_text.strip().lower()
    has_dung = "đúng" in gen or "dung" in gen
    has_sai = "sai" in gen
    if has_dung and not has_sai:
        return "Đúng"
    elif has_sai and not has_dung:
        return "Sai"
    elif has_dung and has_sai:
        pos_dung = min([gen.find(w) for w in ["đúng", "dung"] if gen.find(w) != -1] or [9999])
        pos_sai = gen.find("sai")
        return "Đúng" if pos_dung < pos_sai else "Sai"
    return "Đúng"


def evaluate_model_pipeline(model, processor, test_items, retriever, with_legal_ctx: bool = False, tag: str = "Zero-Shot"):
    print(f"\n[*] Đang đánh giá {tag} trên {len(test_items)} câu hỏi Yes/No (Đúng / Sai)...")
    correct_count = 0
    predictions = []

    for idx, item in enumerate(test_items):
        img_rel = item["image"].replace("\\", "/")
        img_path = BASE_DIR / img_rel
        if not img_path.exists():
            continue

        raw_img = Image.open(img_path).convert("RGB")
        q_prompt = item["vlsp_prompt"].replace("<image>\n", "").strip()

        if with_legal_ctx:
            # Truy xuất ngữ cảnh điều luật liên quan
            ret_docs = retriever.retrieve(item["question"], top_k=2, mode="text_hybrid")
            ctx_str = "\n".join(f"[{d['article_id']}] {d['title']}: {d.get('summary', '')}" for d in ret_docs)
            full_user_text = f"Căn cứ pháp lý QCVN 41:2024/BGTVT:\n{ctx_str}\n\n{q_prompt}"
        else:
            full_user_text = q_prompt

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": full_user_text},
                ],
            }
        ]

        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[prompt_text],
            images=[raw_img],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        pred_ans = extract_yes_no(response_text)
        gt_ans = item["correct_answer"]
        is_correct = (pred_ans == gt_ans)
        if is_correct:
            correct_count += 1

        predictions.append({
            "id": item["id"],
            "image": img_rel,
            "sign_code": item.get("sign_code", ""),
            "generated_raw": response_text.strip(),
            "predicted_answer": pred_ans,
            "gold_answer": gt_ans,
            "is_correct": is_correct,
        })

    acc = correct_count / max(1, len(predictions))
    print(f"  [✓] {tag}: Accuracy = {acc*100:.2f}% ({correct_count}/{len(predictions)})")
    return {
        "name": tag,
        "accuracy": round(acc, 4),
        "correct": correct_count,
        "total": len(predictions),
        "predictions": predictions,
    }


def main():
    if not MQA_FILE.exists():
        print(f"[!] Không tìm thấy {MQA_FILE}")
        return

    with open(MQA_FILE, "r", encoding="utf-8") as f:
        mqa_data = json.load(f)["data"]

    test_items = [x for x in mqa_data if x["grouped_split"] == "test" and x["question_type"] == "yes_no"]
    print(f"[*] Tìm thấy {len(test_items)} câu hỏi Yes/No test trên data/pdf_signs.")

    retriever = HybridLegalRetriever()

    print(f"[*] Đang nạp Processor: {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=512*28*28)

    bnb_config = None
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"[*] Đang nạp Base Model {MODEL_ID} (4-bit NF4)...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    base_model.eval()

    configs = []

    # 1. Base Model Zero-Shot
    res1 = evaluate_model_pipeline(base_model, processor, test_items, retriever, with_legal_ctx=False, tag="Qwen2-VL-2B (Zero-Shot)")
    configs.append(res1)

    # 2. Base Model + Grounded Legal Retrieval (Text-Hybrid Context)
    res2 = evaluate_model_pipeline(base_model, processor, test_items, retriever, with_legal_ctx=True, tag="Qwen2-VL-2B + Text-Hybrid Legal IR")
    configs.append(res2)

    # 3. Fine-tuned PEFT LoRA Adapter (nếu có checkpoint)
    if ADAPTER_DIR.exists():
        print(f"\n[*] Đang nạp PEFT LoRA Adapter từ {ADAPTER_DIR}...")
        try:
            peft_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
            peft_model.eval()
            res3 = evaluate_model_pipeline(peft_model, processor, test_items, retriever, with_legal_ctx=True, tag="Qwen2-VL-2B + PEFT LoRA + Legal IR")
            configs.append(res3)
        except Exception as e:
            print(f"[!] Không thể nạp PEFT adapter: {e}")

    out_payload = {
        "description": "Đánh giá thực tế Qwen2-VL-2B trên 51 câu hỏi Yes/No (Đúng / Sai) data/pdf_signs (không nhãn rò rỉ)",
        "configs": configs,
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    print(f"\n[✓] Đã lưu kết quả đánh giá VLM thật vào: {OUT_FILE}")


if __name__ == "__main__":
    main()
