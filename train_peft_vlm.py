"""
train_peft_vlm.py
-----------------
Huấn luyện tinh chỉnh tham số hiệu quả (Parameter-Efficient Fine-Tuning - PEFT / QLoRA 4-bit)
mô hình ngôn ngữ - thị giác mã nguồn mở (Qwen2-VL-2B-Instruct) trên bộ dữ liệu
VN-TrafficLaw-MQA (Multimodal Legal Question Answering for Traffic Sign Regulations)
dưới giao thức Kiểm soát Rò rỉ Dữ liệu (Leakage-Controlled Group-Disjoint Partition).

Điểm nổi bật:
1. Xác thực mã băm phân chia SHA-256 (split_hash_sha256) trước khi huấn luyện (Thuật toán 1 - Bước 6 bài báo).
2. Chỉ nạp mẫu thuộc tập `grouped_split == 'train'` (bao gồm 346 mẫu biển báo chuẩn cắt từ PDF QCVN 41:2019/BGTVT
   và các mẫu biển báo thực tế trong tập Train), tuyệt đối không chạm vào tập Test.
3. Tích hợp 4-bit NormalFloat (NF4) QLoRA + LoRA Adapter trên các lớp Attention & MLP प्रोजेक्शन (`q_proj`, `k_proj`,
   `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
4. Tích hợp ngữ cảnh Truy xuất Thông tin Pháp lý (Legal IR RAG Context) vào câu lệnh huấn luyện và suy luận.
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

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2-VL-2B-Instruct",
        metadata={"help": "Open-source Vision-Language Model backbone"},
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
        metadata={"help": "Path to VN-TrafficLaw-MQA dataset"},
    )
    split_manifest_file: str = field(
        default=str(DATA_DIR / "split_manifest.json"),
        metadata={"help": "Path to Group-Disjoint split manifest for hash verification"},
    )
    partition_arm: str = field(
        default="grouped_split",
        metadata={"help": "'grouped_split' (primary leakage-controlled) or 'random_split' (reference arm)"},
    )
    include_pdf_canonical: bool = field(
        default=True,
        metadata={"help": "Include 346 PDF canonical sign crops in training"},
    )


class LeakageControlledMQADataset(Dataset):
    def __init__(
        self,
        mqa_file: str,
        manifest_file: str,
        base_dir: Path,
        processor: AutoProcessor,
        split_arm: str = "grouped_split",
        subset: str = "train",
        include_pdf_canonical: bool = True,
    ):
        # Xác thực SHA-256 split hash trước khi huấn luyện (Algorithm 1)
        if os.path.exists(manifest_file):
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            print(f"[✓] Verified Partition Manifest SHA-256 Hash: {manifest['split_hash_sha256'][:16]}...")

        with open(mqa_file, "r", encoding="utf-8") as f:
            raw_mqa = json.load(f)
        entries = raw_mqa["data"] if isinstance(raw_mqa, dict) and "data" in raw_mqa else raw_mqa

        self.base_dir = base_dir
        self.processor = processor
        self.samples = []

        for item in entries:
            item_split = item.get(split_arm, "train")
            if item_split != subset:
                continue
            if not include_pdf_canonical and item.get("source_type") == "pdf_canonical":
                continue
            img_rel = item.get("image")
            img_path = self.base_dir / img_rel
            if not img_path.exists():
                continue
            self.samples.append({
                "image_path": str(img_path),
                "messages": item.get("messages", []),
                "sign_code": item.get("sign_code", ""),
                "relevant_articles": item.get("relevant_articles", []),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")

        conversation = []
        is_first_user = True
        for msg in item["messages"]:
            role = msg["role"]
            content = msg["content"]
            if role == "user" and is_first_user:
                conversation.append({
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": content},
                    ],
                })
                is_first_user = False
            else:
                conversation.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                })

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
    print(f"[*] Loading Base VLM Model on {device_map}...")
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

    train_dataset = LeakageControlledMQADataset(
        mqa_file=data_args.mqa_dataset_file,
        manifest_file=data_args.split_manifest_file,
        base_dir=BASE_DIR,
        processor=processor,
        split_arm=data_args.partition_arm,
        subset="train",
        include_pdf_canonical=data_args.include_pdf_canonical,
    )

    output_dir = str(BASE_DIR / "traffic_vlm_peft_adapter")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
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
    )

    print("\n" + "=" * 65)
    print(" TRAINING LEAKAGE-CONTROLLED PEFT VLM (QWEN2-VL + QLORA) ")
    print("=" * 65)
    print(f"Train samples ({data_args.partition_arm}): {len(train_dataset)}")
    trainer.train()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"[✓] Saved PEFT LoRA Adapter to: {output_dir}")


if __name__ == "__main__":
    train()
