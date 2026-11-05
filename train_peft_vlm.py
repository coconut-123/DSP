r"""
train_peft_vlm.py
-----------------
Huấn luyện tinh chỉnh tham số hiệu quả (Parameter-Efficient Fine-Tuning - PEFT / QLoRA 4-bit)
mô hình ngôn ngữ - thị giác mã nguồn mở (Qwen2-VL-2B-Instruct) theo đúng bài báo:
"VLSP 2025 MLQA-TSR Challenge: Vietnamese Multimodal Legal Question Answering on Traffic Sign Regulation"
(Luu et al., VLSP 2025 - Mục 2, 3 & 4).

Đặc điểm kỹ thuật theo bài báo VLSP 2025 MLQA-TSR:
1. Tập Huấn luyện (Train Set): Toàn bộ 1.807 mẫu biển báo cắt thực tế từ đường phố Việt Nam
   trong `C:\ki8\DSP\datadsp\data\real_crops`.
2. Tập Đánh giá (Evaluation / Test Set): Các mẫu câu hỏi trên 51 ảnh biển báo chuẩn trích xuất từ PDF QCVN 41
   trong `C:\ki8\DSP\datadsp\data\pdf_signs`.
3. Mẫu câu lệnh (Prompt Templates) chuẩn theo Mục 4.1 của bài báo:
   - Câu hỏi Multiple-choice (4 lựa chọn A, B, C, D):
     `<image>\n<question>\nA. ...\nB. ...\nC. ...\nD. ...\n\nTrả lời bằng một trong bốn đáp án: A, B, C hoặc D. Không giải thích thêm`
   (Dạng câu hỏi Yes/No Đúng/Sai đã được loại bỏ khỏi bộ dữ liệu).
4. Tích hợp ngữ cảnh điều luật truy xuất từ Subtask 1 (Multimodal Legal Retrieval) vào quá trình huấn luyện và suy luận Subtask 2.
"""

import os
import sys
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
        metadata={"help": "Open-source Vision-Language Model backbone (Section 4.2 VLSP 2025)"},
    )
    use_qlora: bool = field(
        default=True,
        metadata={"help": "Enable 4-bit NormalFloat (NF4) quantization"},
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA rank dimension"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha scaling"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})


@dataclass
class DataArguments:
    mqa_dataset_file: str = field(
        default=str(DATA_DIR / "mqa_dataset.json"),
        metadata={"help": "Path to VLSP 2025 MLQA-TSR dataset"},
    )
    train_crops_dir: str = field(
        default=str(DATA_DIR / "real_crops"),
        metadata={"help": "Directory containing real street crops for training (C:\\ki8\\DSP\\datadsp\\data\\real_crops)"},
    )
    eval_pdf_dir: str = field(
        default=str(DATA_DIR / "pdf_signs"),
        metadata={"help": "Directory containing canonical PDF signs for evaluation (C:\\ki8\\DSP\\datadsp\\data\\pdf_signs)"},
    )


class VLSP2025MLQADataset(Dataset):
    """Dataset nạp dữ liệu theo chuẩn Subtask 1 & Subtask 2 của bài báo VLSP 2025 MLQA-TSR (Mục 4.1)."""

    def __init__(
        self,
        mqa_file: str,
        base_dir: Path,
        processor: AutoProcessor,
        retriever: HybridLegalRetriever,
        source_filter: str = "real_street_crop",
        target_dir_prefix: str = "data/real_crops",
    ):
        with open(mqa_file, "r", encoding="utf-8") as f:
            raw_mqa = json.load(f)
        entries = raw_mqa["data"] if isinstance(raw_mqa, dict) and "data" in raw_mqa else raw_mqa

        self.base_dir = base_dir
        self.processor = processor
        self.retriever = retriever
        self.samples = []

        for item in entries:
            if item.get("source_type") != source_filter:
                continue
            img_rel = item.get("image", "").replace("\\", "/")
            if target_dir_prefix and not img_rel.startswith(target_dir_prefix):
                continue
            img_path = self.base_dir / img_rel
            if not img_path.exists():
                continue

            # Lấy văn bản điều luật liên quan R = {d_i} (Subtask 1) ghép cùng câu lệnh chuẩn Mục 4.1 (Subtask 2)
            rel_ids = item.get("relevant_articles", [])
            legal_ctx_lines = []
            for aid in rel_ids:
                art = self.retriever.articles_dict.get(aid, {})
                if art:
                    legal_ctx_lines.append(f"[{aid}] {art.get('title', '')}: {art.get('summary', '')}")
            legal_context_str = "\n".join(legal_ctx_lines)

            vlsp_prompt = item.get("vlsp_prompt", "").replace("<image>\n", "").strip()
            if legal_context_str:
                full_user_prompt = f"Điều luật liên quan:\n{legal_context_str}\n\n{vlsp_prompt}"
            else:
                full_user_prompt = vlsp_prompt

            self.samples.append({
                "image_path": str(img_path),
                "user_prompt": full_user_prompt,
                "correct_answer": item.get("correct_answer", "A"),
                "question_type": item.get("question_type", "multiple_choice"),
                "sign_code": item.get("sign_code", ""),
                "relevant_articles": rel_ids,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": item["user_prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": item["correct_answer"]},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        item_inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        item_inputs["labels"] = item_inputs["input_ids"].clone()
        return item_inputs


def train():
    model_args = ModelArguments()
    data_args = DataArguments()
    retriever = HybridLegalRetriever()

    print(f"[*] Loading Processor: {model_args.model_name_or_path}...")
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        min_pixels=256 * 28 * 28,
        max_pixels=1024 * 28 * 28,
    )

    bnb_config = None
    if model_args.use_qlora and torch.cuda.is_available():
        print("[*] Configuring 4-bit NF4 QLoRA Quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading Base Vision-Language Model on {device_map}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32,
        device_map=device_map,
    )

    if model_args.use_qlora and torch.cuda.is_available():
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
    )

    eval_dataset = VLSP2025MLQADataset(
        mqa_file=data_args.mqa_dataset_file,
        base_dir=BASE_DIR,
        processor=processor,
        retriever=retriever,
        source_filter="pdf_canonical",
        target_dir_prefix="data/pdf_signs",
    )

    output_dir = str(BASE_DIR / "traffic_vlm_peft_adapter")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("\n" + "=" * 74)
    print(" VLSP 2025 MLQA-TSR: TRAINING PEFT VLM ON data/real_crops -> EVAL ON data/pdf_signs ")
    print("=" * 74)
    print(f"Train samples (C:\\ki8\\DSP\\datadsp\\data\\real_crops): {len(train_dataset)}")
    print(f"Eval samples  (C:\\ki8\\DSP\\datadsp\\data\\pdf_signs) : {len(eval_dataset)}")
    trainer.train()

    print("\n[*] Đang đánh giá mô hình trên tệp C:\\ki8\\DSP\\datadsp\\data\\pdf_signs...")
    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print(f"[✓] Kết quả đánh giá trên data/pdf_signs: {eval_metrics}")

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
    print(f"[✓] Saved PEFT LoRA Adapter to: {output_dir}")


if __name__ == "__main__":
    train()
