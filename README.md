# Multimodal Legal Question Answering for Traffic Sign Regulations (VN-TrafficLaw-MQA)

**Main Technologies:** Vision-Language Models (VLMs), Parameter-Efficient Fine-Tuning (PEFT / QLoRA), Multimodal Question Answering (MQA), Hybrid Legal Information Retrieval (Legal IR / RAG), Leakage-Controlled Group-Disjoint Evaluation.

Dự án xây dựng Hệ thống Hỏi - Đáp Pháp lý Đa phương thức về Biển báo Giao thông Đường bộ Việt Nam dựa trên **Quy chuẩn kỹ thuật quốc gia QCVN 41:2019/BGTVT** (`51-bgtvt-kem.pdf`), tuân thủ nghiêm ngặt giao thức thực nghiệm kiểm soát rò rỉ dữ liệu (*Leakage-Controlled Evaluation Protocol*) theo bài báo nghiên cứu tham khảo.

---

## 📁 Cấu trúc thư mục dự án (`c:\ki8\DSP\datadsp`)

```text
c:/ki8/DSP/datadsp/
├── extract_pdf_and_crop_signs.py  # [Phase 1] Cắt 346 ảnh biển báo chuẩn từ PDF QCVN 41:2019 (trang 80–157: Phụ lục B,C,D,E,F) & 1.807 biển thực tế
├── leakage_audit.py               # [Phase 2] Kiểm toán rò rỉ (dHash/aHash/Temporal Burst) & Phân chia Group-Disjoint Partition (0 group crossing)
├── legal_ir.py                    # [Phase 3] Động cơ Truy xuất Pháp lý Lai ghép (Okapi BM25 + Dense SVD + Visual-Conditioned RRF)
├── build_mqa_dataset.py           # [Phase 3] Xây dựng bộ dữ liệu VN-TrafficLaw-MQA (2.753 hội thoại đa phương thức, 4 tác vụ pháp lý)
├── train_peft_vlm.py              # [Phase 4] Fine-tuning Qwen2-VL-2B-Instruct bằng 4-bit NF4 QLoRA (PEFT) có xác thực SHA-256 split hash
├── run_paper_experiments.py       # [Phase 4] Kịch bản thực nghiệm toàn diện (Bảng 4–11 & Hình 2–9 của bài báo) trên GPU dưới CUDA AMP
├── server.py                      # [Phase 5] Máy chủ FastAPI kết nối Qwen2-VL PEFT + Hybrid Legal IR + Giao diện Web
├── index.html / app.js / styles.css # Giao diện Web Chatbot Pháp lý Giao thông Đa phương thức
├── traffic_vlm_peft_adapter/      # Trọng số LoRA Adapter đã huấn luyện cho Qwen2-VL-2B-Instruct
├── figures/                       # 8 biểu đồ khoa học độ phân giải cao (fig2_leakage_inflation.png -> fig9_tsne_silhouette.png)
└── data/                          # Toàn bộ dữ liệu chuẩn hóa, manifest phân tập và kết quả thực nghiệm
    ├── pdf_signs/                 # 346 ảnh biển báo chuẩn cắt từ PDF QCVN 41:2019/BGTVT (328 mã biển duy nhất, gồm cả Phụ lục F)
    ├── real_crops/                # 1.807 ảnh cắt biển báo thực tế từ 3.216 khung hình đường phố (52 lớp)
    ├── legal_articles.json        # 183 điều khoản pháp lý trích xuất 100% nguyên bản từ QCVN 41:2019/BGTVT (51-bgtvt-kem.pdf)
    ├── class_mapping_52.json      # Ánh xạ 100% 52 lớp thực tế sang mã chuẩn QCVN 41:2019/BGTVT và mẫu PDF
    ├── leakage_audit_report.json  # Báo cáo kiểm toán 543 nhóm sơ cấp độc lập & thống kê chuẩn hóa Per-Channel Train-Only
    ├── split_manifest.json        # Manifest phân chia Group-Disjoint vs Random Split kèm chữ ký SHA-256
    ├── mqa_dataset.json           # Bộ dữ liệu hỏi đáp đa phương thức VN-TrafficLaw-MQA (2.753 mẫu)
    └── experimental_results.json  # Kết quả định lượng đầy đủ (3 seeds: 42, 123, 2024; Mean ± SD; 95% Student-t CI; Holm p)
```

---

## 🚀 Hướng dẫn chạy nhanh (Quickstart)

### 1. Chạy toàn bộ quy trình dữ liệu, kiểm toán rò rỉ và thực nghiệm bài báo
```powershell
$env:PYTHONIOENCODING="utf-8"
python extract_pdf_and_crop_signs.py
python leakage_audit.py
python build_mqa_dataset.py
python run_paper_experiments.py
```

### 2. Huấn luyện mô hình VLM với PEFT (4-bit QLoRA trên Qwen2-VL-2B-Instruct)
```powershell
$env:PYTHONIOENCODING="utf-8"
python train_peft_vlm.py
```

### 3. Khởi động máy chủ Web Demo tương tác (FastAPI + Qwen2-VL PEFT + Hybrid Legal IR)
```powershell
$env:PYTHONIOENCODING="utf-8"
python server.py
```
Sau đó mở trình duyệt tại **`http://localhost:8000`** (hoặc xem tài liệu API tại `http://localhost:8000/docs`).
