r"""
train_peft_vlm.py
-----------------
Huấn luyện tinh chỉnh tham số hiệu quả (Parameter-Efficient Fine-Tuning - PEFT LoRA)
mô hình ngôn ngữ - thị giác mã nguồn mở (Qwen2-VL-2B-Instruct) cho bài toán
Hỏi - Đáp Pháp lý Đa phương thức về Biển báo Giao thông Việt Nam (QCVN 41:2024/BGTVT).

Cải tiến cốt lõi (đã sửa lỗi):
1. Che nhãn (Label Masking `-100`) toàn bộ phần User Prompt + Image Tokens:
   Chỉ tính hàm mất mát Cross-Entropy trên phần câu trả lời của Assistant và token kết thúc `<|im_end|>`,
   giúp mô hình học sinh câu trả lời thực sự thay vì lặp lại câu hỏi.
2. Huấn luyện đa tác vụ (Multi-Task Instruction Tuning) trên tập `data/real_crops`:
   - Tác vụ 1 (Subtask 2 Binary QA): Câu hỏi Đúng/Sai -> trả lời chính xác `Đúng` hoặc `Sai`.
   - Tác vụ 2 (Open-Ended Legal QA): Câu hỏi tự nhiên ("Biển báo này có ý nghĩa gì?",
     "Xe nào được phép đi vào?", "Quy định xử lý khi gặp biển này?") -> sinh câu trả lời giải thích
     pháp lý chi tiết, mạch lạc theo QCVN 41:2024/BGTVT.
"""

import os
import sys
import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from legal_ir import HybridLegalRetriever

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2-VL-2B-Instruct",
        metadata={"help": "Open-source Vision-Language Model backbone"},
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA rank r"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})


@dataclass
class DataArguments:
    mqa_dataset_file: str = field(
        default=str(DATA_DIR / "mqa_dataset.json"),
        metadata={"help": "Path to VLSP 2025 MLQA-TSR dataset"},
    )


OPEN_QUESTIONS = [
    "Biển báo trong hình là biển gì và có ý nghĩa quy định như thế nào?",
    "Biển báo này có ý nghĩa gì và phương tiện nào cần lưu ý khi gặp biển?",
    "Hãy phân tích ý nghĩa pháp lý và phạm vi hiệu lực của biển báo trong ảnh theo QCVN 41:2024/BGTVT.",
    "Khi gặp biển báo giao thông này trên đường, người lái xe phải chấp hành quy định gì?",
]


def clean_open_answer(item: dict, retriever: HybridLegalRetriever) -> str:
    """Xây dựng câu trả lời giải thích pháp lý chuẩn xác, tự nhiên từ thông tin biển báo và điều khoản QCVN 41:2024."""
    code = item.get("sign_code", "")
    name = item.get("sign_name", "Biển báo giao thông")
    category = item.get("category", "Biển báo hiệu đường bộ")
    rel_ids = item.get("relevant_articles", [])

    art_title = f"Quy chuẩn QCVN 41:2024/BGTVT - Biển số {code}"
    art_summary = f'Biển số {code} "{name}" thuộc nhóm {category}.'
    for aid in rel_ids:
        art = retriever.articles_dict.get(aid, {})
        if art:
            art_title = art.get("title", art_title)
            s_text = art.get("summary", "") or art.get("full_text", "")
            if code in art.get("full_text", "") and code not in s_text:
                clauses = [c.strip() for c in re.split(r"(?=[a-k]\)\s)", art["full_text"]) if code in c]
                if clauses:
                    s_text = clauses[0].split("Hình ")[0].strip()
            if s_text:
                art_summary = s_text
            break

    return (
        f"Biển báo trong hình là **Biển số {code}: \"{name}\"** thuộc nhóm **{category}** theo **QCVN 41:2024/BGTVT**.\n\n"
        f"- **Căn cứ pháp lý:** {art_title}.\n"
        f"- **Ý nghĩa & Quy định chấp hành:** {art_summary}"
    )


class VLSP2025MLQADataset(Dataset):
    """
    Dataset huấn luyện Qwen2-VL + LoRA có che nhãn (`-100`) cho phần User Prompt,
    kết hợp cả câu hỏi Đúng/Sai và câu hỏi giải thích pháp lý tự do.
    """

    def __init__(
        self,
        mqa_file: str,
        base_dir: Path,
        processor: AutoProcessor,
        retriever: HybridLegalRetriever,
        source_filter: str = "real_street_crop",
        target_dir_prefix: str = "data/real_crops",
        include_open_qa: bool = True,
    ):
        with open(mqa_file, "r", encoding="utf-8") as f:
            raw_mqa = json.load(f)
        entries = raw_mqa["data"] if isinstance(raw_mqa, dict) and "data" in raw_mqa else raw_mqa

        self.base_dir = base_dir
        self.processor = processor
        self.retriever = retriever
        self.samples = []

        for idx, item in enumerate(entries):
            if item.get("source_type") != source_filter:
                continue
            img_rel = item.get("image", "").replace("\\", "/")
            if target_dir_prefix and not img_rel.startswith(target_dir_prefix):
                continue
            img_path = self.base_dir / img_rel
            if not img_path.exists():
                continue

            rel_ids = item.get("relevant_articles", [])
            legal_ctx_lines = []
            for aid in rel_ids:
                art = self.retriever.articles_dict.get(aid, {})
                if art:
                    legal_ctx_lines.append(f"[{aid}] {art.get('title', '')}: {art.get('summary', '')}")
            legal_context_str = "\n".join(legal_ctx_lines)
            ctx_prefix = f"Điều luật liên quan:\n{legal_context_str}\n\n" if legal_context_str else ""

            # 1. Mẫu câu hỏi Đúng/Sai (Subtask 2)
            vlsp_prompt = item.get("vlsp_prompt", "").replace("<image>\n", "").strip()
            self.samples.append({
                "image_path": str(img_path),
                "user_prompt": f"{ctx_prefix}{vlsp_prompt}",
                "assistant_answer": item.get("correct_answer", "Đúng"),
                "task": "yes_no",
            })

            # 2. Mẫu câu hỏi tự do (Open-Ended Legal QA)
            if include_open_qa:
                q_open = OPEN_QUESTIONS[idx % len(OPEN_QUESTIONS)]
                code = item.get("sign_code", "")
                name = item.get("sign_name", "")
                cat = item.get("category", "")
                open_prompt = (
                    f"{ctx_prefix}"
                    f"Nhận diện thị giác: Biển số {code} \"{name}\" ({cat}).\n"
                    f"Câu hỏi: {q_open}"
                )
                self.samples.append({
                    "image_path": str(img_path),
                    "user_prompt": open_prompt,
                    "assistant_answer": clean_open_answer(item, self.retriever),
                    "task": "open_qa",
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)

        user_msg = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": item["user_prompt"]},
                ],
            }
        ]
        full_conv = user_msg + [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": item["assistant_answer"]},
                ],
            }
        ]

        prompt_text = self.processor.apply_chat_template(
            user_msg,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.processor.apply_chat_template(
            full_conv,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_inputs = self.processor(
            text=[prompt_text],
            images=[image],
            padding=False,
            return_tensors="pt",
        )
        full_inputs = self.processor(
            text=[full_text],
            images=[image],
            padding=False,
            return_tensors="pt",
        )

        prompt_len = prompt_inputs["input_ids"].shape[1]
        item_inputs = {k: v.squeeze(0) for k, v in full_inputs.items()}
        labels = item_inputs["input_ids"].clone()
        # Che toàn bộ token của phần câu hỏi + ảnh bằng -100 để chỉ tính Loss trên câu trả lời của Assistant
        labels[:prompt_len] = -100
        item_inputs["labels"] = labels
        return item_inputs


def collate_fn(batch):
    """Pad batch cho Qwen2-VL."""
    if len(batch) == 1:
        return {k: v.unsqueeze(0) if k in ("input_ids", "attention_mask", "labels") else v for k, v in batch[0].items()}
    raise NotImplementedError("Use per_device_train_batch_size=1 with gradient_accumulation_steps")


def train():
    model_args = ModelArguments()
    data_args = DataArguments()
    retriever = HybridLegalRetriever()

    print(f"[*] Loading Processor: {model_args.model_name_or_path}...", flush=True)
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        min_pixels=64 * 28 * 28,
        max_pixels=128 * 28 * 28,
    )

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    bnb_config = None
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading Base Vision-Language Model on {device_map} ({dtype}, 4-bit NF4)...", flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=dtype,
        device_map=device_map,
    )
    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = VLSP2025MLQADataset(
        mqa_file=data_args.mqa_dataset_file,
        base_dir=BASE_DIR,
        processor=processor,
        retriever=retriever,
        source_filter="real_street_crop",
        target_dir_prefix="data/real_crops",
        include_open_qa=True,
    )

    eval_dataset = VLSP2025MLQADataset(
        mqa_file=data_args.mqa_dataset_file,
        base_dir=BASE_DIR,
        processor=processor,
        retriever=retriever,
        source_filter="pdf_canonical",
        target_dir_prefix="data/pdf_signs",
        include_open_qa=False,
    )

    output_dir = str(BASE_DIR / "traffic_vlm_peft_adapter")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_strategy="steps",
        save_steps=30,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("\n" + "=" * 74, flush=True)
    print(" TRAINING PEFT VLM (WITH PROMPT MASKING -100 & MULTI-TASK QA)", flush=True)
    print("=" * 74, flush=True)
    print(f"Train samples (Yes/No + Open-Ended Legal QA): {len(train_dataset)}", flush=True)
    print(f"Eval samples  (data/pdf_signs Yes/No)       : {len(eval_dataset)}", flush=True)
    trainer.train()

    print("\n[*] Đang đánh giá mô hình trên tập data/pdf_signs...", flush=True)
    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print(f"[✓] Kết quả đánh giá trên data/pdf_signs: {eval_metrics}", flush=True)

    with open(DATA_DIR / "vlm_eval_pdf_signs.json", "w", encoding="utf-8") as f:
        json.dump({
            "paper_reference": "VLSP 2025 MLQA-TSR Challenge: Vietnamese Multimodal Legal Question Answering on Traffic Sign Regulation",
            "train_source": str(DATA_DIR / "real_crops"),
            "eval_source": str(DATA_DIR / "pdf_signs"),
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "eval_metrics": eval_metrics,
        }, f, ensure_ascii=False, indent=2)

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"[✓] Saved PEFT LoRA Adapter to: {output_dir}", flush=True)


if __name__ == "__main__":
    train()
