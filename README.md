# VLSP 2025 MLQA-TSR: Vietnamese Multimodal Legal Question Answering on Traffic Sign Regulation

**Bài báo tham chiếu:** *VLSP 2025 MLQA-TSR Challenge: Vietnamese Multimodal Legal Question Answering on Traffic Sign Regulation* (Son T. Luu, Trung Vo, Hiep Nguyen, Khanh Quoc Tran, Kiet Van Nguyen, Vu Tran, Ngan Luu-Thuy Nguyen, Le-Minh Nguyen — VLSP 2025, [`archive/2025.vlsp-1.48.pdf`](file:///c:/ki8/DSP/datadsp/archive/2025.vlsp-1.48.pdf)).

Hệ thống Hỏi - Đáp Pháp lý Đa phương thức về Biển báo Giao thông Đường bộ Việt Nam tuân thủ đầy đủ 2 tác vụ chuẩn của **VLSP 2025 MLQA-TSR**:
1. **Subtask 1 — Multimodal Legal Retrieval (Mục 2.1):** Cho câu hỏi đa phương thức $q = (q_{\text{text}}, q_{\text{image}})$ và Cơ sở dữ liệu Luật $\mathcal{D} = \{d_i\}$, truy xuất danh sách các điều luật liên quan $\mathcal{R} = \{d_i\}$ (chứa định dạng đa phương thức `«IMAGE: ... /IMAGE»` và `«TABLE: ... /TABLE»`). Đánh giá bằng **Precision** (Công thức 1), **Recall** (Công thức 2) và **F2-score** (Công thức 3).
2. **Subtask 2 — Multimodal Legal Question Answering (Mục 2.2 & 4.1):** Trả lời câu hỏi **Trắc nghiệm 4 lựa chọn (`A, B, C, D`)** và **Đúng/Sai (`Đúng / Sai`)** dựa trên ảnh biển báo và điều luật truy xuất. Đánh giá bằng **Accuracy** (Công thức 4).

---

## 📁 Cấu trúc thư mục dự án (`c:\ki8\DSP\datadsp`)

```text
c:/ki8/DSP/datadsp/
├── archive/
│   ├── 2025.vlsp-1.48.pdf         # Bài báo gốc VLSP 2025 MLQA-TSR Challenge
│   └── 51-bgtvt-kem.pdf           # Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ QCVN 41
├── extract_pdf_and_crop_signs.py  # Cắt 1.807 biển thực tế (data/real_crops) để Train & lọc cắt đúng 51 ảnh biển chuẩn từ PDF QCVN 41 (data/pdf_signs) để Test
├── legal_ir.py                    # [Subtask 1] Truy xuất Pháp lý Đa phương thức (BM25 + Dense + Dynamic Top-k + F2-score Eq. 1, 2, 3)
├── build_mqa_dataset.py           # [Dataset] Xây dựng bộ dữ liệu chuẩn VLSP 2025 MLQA-TSR (Multiple-Choice A/B/C/D & Yes/No Đúng/Sai, Bảng 2)
├── train_peft_vlm.py              # [Subtask 2] Fine-tuning Qwen2-VL-2B-Instruct bằng 4-bit NF4 QLoRA theo Prompt Template Mục 4.1
├── run_paper_experiments.py       # Thực nghiệm toàn diện Subtask 1 (F2-score, Bảng 3) & Subtask 2 (Accuracy, Bảng 4) trên data/real_crops -> data/pdf_signs
├── test_eval_pdf_signs.py         # Script kiểm thử CLI trực tiếp trên 51 ảnh data/pdf_signs (báo cáo Subtask 1 F2 & Subtask 2 Accuracy)
├── server.py                      # Máy chủ FastAPI tích hợp bộ cắt mặt biển báo tự động (Sign Detector) + Subtask 1 Legal IR + Subtask 2 Legal QA
├── index.html / app.js / styles.css # Giao diện Web Chatbot Pháp lý Giao thông Đa phương thức
├── figures/                       # Bộ biểu đồ khoa học theo bài báo VLSP 2025 MLQA-TSR
└── data/
    ├── real_crops/                # [TẬP TRAIN] 1.807 ảnh cắt biển báo thực tế từ đường phố Việt Nam (52 lớp)
    ├── pdf_signs/                 # [TẬP TEST/ĐÁNH GIÁ] 51 ảnh biển báo chuẩn cắt từ PDF QCVN 41 (46 mã biển khớp với tập Train)
    ├── legal_articles.json        # Cơ sở dữ liệu luật (QCVN 41 + Luật TTATGTĐB 36/2024/QH15 kèm thẻ «IMAGE: ... /IMAGE»)
    ├── mqa_dataset.json           # Bộ dữ liệu câu hỏi VLSP 2025 MLQA-TSR (Multiple-Choice & Yes/No)
    └── experimental_results.json  # Kết quả định lượng đầy đủ Subtask 1 (F2-score) & Subtask 2 (Accuracy)
```

---

## 🚀 Hướng dẫn chạy nhanh (Quickstart)

### 1. Xây dựng dữ liệu và chạy thực nghiệm VLSP 2025 MLQA-TSR
```powershell
$env:PYTHONIOENCODING="utf-8"
python extract_pdf_and_crop_signs.py
python build_mqa_dataset.py
python run_paper_experiments.py
python test_eval_pdf_signs.py
```

### 2. Huấn luyện mô hình VLM với PEFT (4-bit QLoRA trên Qwen2-VL-2B-Instruct)
```powershell
$env:PYTHONIOENCODING="utf-8"
python train_peft_vlm.py
```

### 3. Khởi động máy chủ Web Demo tương tác
```powershell
$env:PYTHONIOENCODING="utf-8"
python server.py
```
Sau đó mở trình duyệt tại **`http://localhost:8000`**.
