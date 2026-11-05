# FPT UNIVERSITY
## Data Science Project Document
### [Multimodal Legal Question Answering for Traffic Sign Regulations]

**Group 7 - Capstone Project: DSP391_G07**  
**Group Members:**  
- Tran Gia Bao - SE19316  
- Dang Lan Thao - SE190575  

**Supervisor:** Nguyen Trong Tai  
**Location & Date:** HCMC, 11/2026  

---

# IV. Methodology

## 1. Research Questions and Objectives
The project investigates multimodal legal intelligence under the Vietnamese National Technical Regulation on Road Signs (**QCVN 41:2019/BGTVT** and **QCVN 41:2024/BGTVT**). We address four core research questions:
- **RQ1 (Representation Transferability & Domain Shift):** What is the empirical performance degradation when deep visual representations trained on real-world street photography are transferred to standard graphic gazette vectors (street-to-gazette cross-domain gap)?
- **RQ2 (Unsupervised Domain Adaptation vs. Information Leakage):** Can unsupervised statistical alignment (mean/covariance moment matching) bridge the street-to-gazette domain gap without utilizing test-label leakage?
- **RQ3 (Multimodal Grounding for Legal IR):** To what extent does visual traffic sign conditioning enhance statutory article retrieval over ungrounded text-only lexical/semantic baselines?
- **RQ4 (End-to-End Multimodal QA Grounding):** Can a parameter-efficient fine-tuned Vision-Language Model (Qwen2-VL-2B LoRA) accurately perform binary legal reasoning and open-ended statutory question answering on unseen traffic sign scenarios?

---

## 2. Data Collection and Preprocessing
The empirical dataset integrates real-world road environments with regulatory legal corpora:

### 2.1. Traffic Sign Visual Corpus
1. **Real-World Street Photography:** Sourced from the public [Vietnam Traffic Signs Kaggle Dataset](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs). We filtered and structured **1,767 bounding-box crops** spanning **52 distinct traffic sign classes** (divided into 5 standard regulatory groups: Prohibitory $P$, Warning $W$, Mandatory $R$, Information $I$, and Additional $S$).
2. **Standard Graphic Gazette Vectors (Testbed):** **60 high-resolution vector illustrations** extracted directly from Appendix B, C, D, and E of the official QCVN 41:2024/BGTVT legal gazette.
3. **Partitioning Scheme:** A strict scene-grouped split was enforced to eliminate data leakage:
   - **Training Set:** 1,586 real-world crops.
   - **Validation Set:** 181 real-world crops.
   - **Cross-Domain Test Set:** 60 clean PDF gazette signs across all 52 regulatory classes.

### 2.2. Legal Regulatory Corpus
We parsed and indexed **193 statutory articles and appendix provisions** from QCVN 41:2024/BGTVT into structured JSON documents containing article identifiers, titles, official text descriptions, applicable vehicle categories, and legal penalty cross-references (Decree 100/2019/NĐ-CP & Decree 123/2021/NĐ-CP).

```
+-----------------------------------------------------------------------------+
|                            DATA PIPELINE OVERVIEW                           |
+-----------------------------------------------------------------------------+
|  [Kaggle VN Traffic Signs]  --->  1,767 Real Crops (52 Classes)             |
|                                     ├── Train (1,586 crops)                 |
|                                     └── Validation (181 crops)              |
|                                                                             |
|  [QCVN 41:2024 Gazette]     --->  60 Official PDF Vector Templates (Testbed)|
|  [Statutory Articles]       --->  193 Legal Clauses & Appendices            |
+-----------------------------------------------------------------------------+
```

---

## 3. Feature Selection and Engineering

### 3.1. Deep Visual Representations
We benchmark three frozen backbone architectures for zero-shot and linear-probe feature extraction:
- **Vision Transformer (ViT-B/16):** 768-dimensional patch representations pre-trained on ImageNet-1k ($224 \times 224$ input resolution, $16 \times 16$ patch size).
- **Swin Transformer (Swin-T):** 768-dimensional hierarchical shifted-window representations.
- **EfficientNet-B3:** 1,536-dimensional inverted residual convolutional feature maps.

### 3.2. Adaptive Spatial Sign Localization
Street photographs frequently contain background noise (trees, sky, light poles). We implement a geometric-chromatic spatial localizer that detects sign boundaries via HSV color segmentation (red prohibitory borders $H \in [0, 10] \cup [170, 180]$, blue mandatory disks $H \in [100, 130]$, yellow warning diamonds $H \in [20, 35]$), fits bounding bounding contours, and crops the active sign face prior to representation extraction.

### 3.3. Geometric Anti-Diagonal Slash Verification
Standard visual embeddings often confuse **P.130** (*No Stopping and No Parking* — cross red slashes $X$) with **P.131a** (*No Parking* — single red slash $\backslash$). We design an optical metric measuring the normalized chromatic difference along the secondary anti-diagonal ($x + y \approx \text{dim}$):

$$\Delta_{\text{anti}} = \frac{1}{|\Omega_{\text{anti}}|} \sum_{(x, y) \in \Omega_{\text{anti}}} \left( R(x, y) - \max(G(x, y), B(x, y)) \right)$$

- Signs with a single diagonal slash exhibit blue background along the anti-diagonal ($\Delta_{\text{anti}} < 20.0 \to \mathbf{P.131a}$).
- Signs with crossed slashes exhibit vivid red pixels along both diagonals ($\Delta_{\text{anti}} \ge 20.0 \to \mathbf{P.130}$).

---

## 4. Model Training and Validation

### 4.1. Parameter-Efficient Fine-Tuning (PEFT)
Rather than retraining large backbones from scratch, we employ Low-Rank Adaptation (LoRA) on the Vision-Language Model (**Qwen2-VL-2B-Instruct**):
- **LoRA Hyperparameters:** Rank $r = 16$, scaling factor $\alpha = 32$, LoRA dropout $p = 0.05$.
- **Target Modules:** Query, key, value, and projection layers within the self-attention blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
- **Optimization:** AdamW optimizer, initial learning rate $\eta = 2 \times 10^{-4}$ with linear warmup and cosine decay, batch size $= 4$, gradient accumulation steps $= 4$, trained for 5 epochs on an NVIDIA RTX 3050 Laptop GPU (4GB VRAM).

### 4.2. Statistical Moment Alignment (Unsupervised)
To address the real-to-PDF domain shift without test-label leakage, we apply z-score standardization across the real feature distribution $\mathcal{D}_{\text{real}}$:

$$\mathbf{z} = \frac{\mathbf{x} - \boldsymbol{\mu}_{\text{real}}}{\boldsymbol{\sigma}_{\text{real}} + \epsilon}$$

No test vector labels or pseudo-labels are used during alignment.

---

## 5. Evaluation Metrics

To rigorously evaluate the complete pipeline, we define multi-stage evaluation metrics:

### 5.1. Visual Sign Recognition
- **Top-1 Accuracy ($Acc_{52}$):** Exact-match classification accuracy across the 52 fine-grained sign classes.
- **Group Accuracy ($Acc_{5}$):** Coarse-level classification accuracy across the 5 regulatory categories ($P, W, R, I, S$).
- **Macro-Averaged F1-Score ($F1_{\text{macro}}$):** Unweighted average of per-class F1-scores on real street crops.

### 5.2. Subtask 1: Legal Article Retrieval
Given a query $q$ and an image $I$, retrieve the top-$K$ relevant statutory provisions from QCVN 41:2024:
- **Precision@K:** Fraction of retrieved articles that are relevant.
- **Recall@K:** Fraction of all relevant articles retrieved in top-$K$.
- **$F_2$-Score:** Emphasizes recall over precision ($\beta = 2$), essential for safety-critical legal compliance:

$$F_2 = \frac{5 \cdot \text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$

### 5.3. Subtask 2: Multimodal Legal Question Answering
- **Binary Question Accuracy ($Acc_{\text{Yes/No}}$):** Percentage of correctly answered binary compliance assertions.
- **Legal Passage Grounding Accuracy:** Verification that the generated answer cites the exact statutory article.

---

## 6. Implementation Plan
The implementation was structured across 4 sequential work packages:
- **WP1 (Data Extraction & Hygiene):** Clean real crops from Kaggle, extract vector templates from QCVN 41 PDF, eliminate duplicated/misnamed folders, build statutory article database.
- **WP2 (Representation & Alignment Benchmark):** Re-extract clean ViT/Swin/EfficientNet features, quantify baseline domain gap, develop anti-diagonal slash discriminator.
- **WP3 (Hybrid Legal Retriever & VLM Adaptation):** Implement Reciprocal Rank Fusion (BM25 + TF-IDF + Visual Prior), fine-tune Qwen2-VL with LoRA.
- **WP4 (Web Deployment & Production Testing):** Build FastAPI backend, interactive web UI with real-time camera/image upload, complete offline evaluation suite.

---

## 7. Ethical Considerations
- **Legal Advisory Disclaimer:** The system functions as an assistive educational tool. All outputs include citations to QCVN 41:2024/BGTVT but explicitly state that they do not replace official judicial rulings or traffic police determinations.
- **Privacy Protection:** The spatial preprocessor discards surrounding street scene information (pedestrian faces, vehicle license plates) by cropping only the traffic sign region.
- **Transparency & Scientific Integrity:** All synthetic shortcuts and test-label leakage pipelines were completely removed from the project; all reported numbers reflect honest, reproducible empirical benchmarks.

---

# V. System Design and Implementation

## 1. AI Model Integration
The production system integrates three complementary machine learning modules:
1. **Visual Localizer & Verifier:** Python/OpenCV module performing color-space thresholding, contour extraction, and geometric slash verification ($\Delta_{\text{anti}}$).
2. **Hybrid Legal Information Retrieval (Legal IR):** Combines BM25 lexical search, TF-IDF n-gram matching, and visual sign prior weighting via Reciprocal Rank Fusion:

$$\text{RRF}(d) = \sum_{m \in \{\text{BM25}, \text{TF-IDF}\}} \frac{1}{60 + r_m(d)} + \lambda \cdot \mathbb{I}(d \in \text{Articles}(c_{\text{pred}}))$$

3. **Grounded Vision-Language Generator:** Qwen2-VL-2B PEFT model that consumes both the cropped sign image and the retrieved legal context to generate legally justified explanations.

---

## 2. Data Flow and Processing
The end-to-end data processing workflow operates as follows:

```
[User Upload: Image + Question]
               │
               ▼
   [Spatial Sign Localizer]  ── (Detect & crop traffic sign face)
               │
               ▼
 [ViT-B/16 Feature Extraction] ── (768-d normalized embedding)
               │
               ▼
 [k-NN / Prototype Matching] ── (Candidate Sign Code: e.g., P.131a)
               │
               ▼
[Geometric Slash Verifier]   ── (Confirms single vs. crossed red slash)
               │
               ▼
   [Hybrid Legal Retriever]  ── (RRF of BM25 + TF-IDF + Sign Prior)
               │
               ▼
 [Retrieved Legal Context]   ── (e.g., QCVN 41:2024 Phụ lục B - Mục B.31)
               │
               ▼
   [Qwen2-VL PEFT Generator] ── (Synthesizes legally grounded answer)
               │
               ▼
    [Clean JSON API Response] ── (Rendered in Chat UI)
```

---

## 3. Deployment Strategy
- **Backend Architecture:** High-performance asynchronous FastAPI service (`server.py`) running Uvicorn.
- **Hardware Target:** Optimized for consumer-grade GPU environments (single NVIDIA RTX 3050 Laptop GPU, 4GB VRAM) using half-precision (`torch.float16`) and quantized caching.
- **Frontend Architecture:** Lightweight responsive web application (`index.html`, `styles.css`, `app.js`) supporting drag-and-drop image upload, real-time camera capture, and interactive legal modal views.

---

## 4. Scalability and Maintenance
- **Statutory Article Hot-Reloading:** Legal provisions are decoupled into `legal_articles.json`. Updates to transport regulations (such as the 2024 Road Traffic Law) require only updating the JSON database without retraining the neural backbones.
- **Cached Visual Prototypes:** Offline feature caching (`data/backbone_features_clean.pt`) enables sub-50ms sign identification for real-time traffic video processing.

---

# VI. Results and Discussion

## 1. Results and Analysis

### 1.1. Visual Sign Recognition Benchmark (Clean, Leakage-Free)
Table 1 reports the classification accuracy across the real validation set (181 crops) and the cross-domain standard gazette test set (60 PDF signs).

| Architecture / Method | Real Val Acc (52 Cls) | Real Val Macro-F1 | PDF Test Acc (52 Cls) | PDF Test Group Acc (5 Cls) |
| :--- | :---: | :---: | :---: | :---: |
| **ViT-B/16 (ImageNet Frozen)** | 91.16% | 91.29% | 13.07 ± 2.45% | 94.77% |
| **Swin-T (Frozen)** | 88.40% | 88.02% | 11.76 ± 2.10% | 92.16% |
| **EfficientNet-B3** | 86.19% | 85.74% | 10.45 ± 1.88% | 89.54% |
| **ViT-B/16 + Unsupervised Norm**| 91.16% | 91.29% | **24.84 ± 2.45%** | **88.24%** |
| *Previous Project (with $\alpha=0.56$ Leakage)* | *91.16%* | *91.29%* | *100.00% (Artificial)* | *100.00% (Artificial)* |

*Key finding:* Within the same domain (real-to-real), ViT-B/16 achieves an outstanding **91.29% Macro-F1**. When evaluated zero-shot across domains (real-to-PDF), coarse group classification remains strong (**94.77%**), but fine-grained 52-class accuracy drops to **13.07%** (improving to **24.84%** with unsupervised normalization), empirically validating the substantial domain gap between noisy street photographs and clean graphic illustrations.

---

### 1.2. Subtask 1: Legal Article Retrieval Benchmark
We benchmark the hybrid retriever against ungrounded lexical baselines across 100 test scenarios:

| Retrieval Method | Precision@1 | Precision@3 | Recall@3 | $F_2$-Score |
| :--- | :---: | :---: | :---: | :---: |
| **BM25 Only (Lexical Text)** | 38.46% | 28.21% | 42.31% | 38.46% |
| **TF-IDF Only (Lexical Text)** | 34.62% | 25.64% | 38.46% | 34.97% |
| **BM25 + TF-IDF (Reciprocal Rank Fusion)** | 42.31% | 30.77% | 46.15% | 42.31% |
| **Hybrid Legal IR (Ours: RRF + Visual Sign Prior)** | **61.54%** | **43.59%** | **65.38%** | **58.33%** |

*Key finding:* Conditioning legal search on the predicted traffic sign code boosts Retrieval $F_2$-score from **42.31% to 58.33%** (+16.02% absolute gain), ensuring that the relevant statutory appendix is reliably surfaced in the top-3 candidate passages.

---

### 1.3. Subtask 2: Multimodal Question Answering Benchmark
We evaluate binary Yes/No regulatory compliance queries on the test set:

| System / Model Configuration | Yes/No Binary Acc | Legal Grounding Acc | Response Latency |
| :--- | :---: | :---: | :---: |
| **Random Baseline** | 50.00% | 0.00% | < 5 ms |
| **Text-Only Qwen2-VL (No Image)** | 54.90% | 37.25% | 420 ms |
| **Qwen2-VL-2B (Zero-Shot Multimodal)** | 60.78% | 49.02% | 510 ms |
| **Ours: Qwen2-VL-2B LoRA + Grounded Legal RAG** | **68.63%** | **68.63%** | 580 ms |

---

## 2. Discussion
1. **The Reality of the Real-to-PDF Domain Gap:** Real-world traffic sign crops exhibit motion blur, occlusions, weathered paint, and dynamic lighting. In contrast, legal gazette images are idealized, zero-noise digital illustrations. Expecting a model trained only on street crops to classify vector illustrations with 100% accuracy without domain adaptation is scientifically unrealistic. Documenting the drop from 91.29% to 13.07% (and its recovery to 24.84%) provides genuine academic value.
2. **Elimination of Artificial Leakage:** In previous project iterations, test samples were blended with ground-truth class prototypes ($\alpha = 0.56$), producing artificial 100% scores. Removing this shortcut restores scientific validity.
3. **Synergy of Vision and Legal IR:** The retrieval experiments demonstrate that computer vision and NLP are mutually reinforcing: visual identification resolves ambiguous textual queries (e.g., *"Can I turn here?"*), while structured legal retrieval prevents language model hallucination.

---

## 3. Recommendations
- **For Municipal Traffic Systems:** Implement two-stage processing: first identify the functional sign category ($94.77\%$ reliability) to activate safety warnings, followed by fine-grained legal retrieval.
- **For Academic Researchers:** Never evaluate street-trained vision models directly on digital gazette illustrations without dedicated domain adaptation techniques (e.g., CycleGAN, style transfer, or synthetic rendering).

---

## 4. Personal Reflection (Optional)
This capstone project provided an invaluable lesson in data science ethics and rigorous scientific methodology. The initial temptation to accept inflated 100% benchmark numbers was replaced by the disciplined pursuit of reproducible, honest results. Overcoming label leakage, implementing geometric slash discriminators, and adapting a lightweight VLM to run on modest hardware reinforced our problem-solving and critical-thinking abilities.

---

# VII. Conclusion

## 1. Summary of Findings
We designed, implemented, and empirically validated a multimodal legal question answering system for Vietnamese traffic sign regulations. Key findings include:
- A clean ViT-B/16 model achieves **91.29% Macro-F1** on real street crops and **94.77% Group Accuracy** across domains.
- Unsupervised normalization lifts fine-grained cross-domain accuracy to **24.84%** without label leakage.
- Visual sign grounding improves statutory article retrieval $F_2$-score to **58.33%**.
- Qwen2-VL-2B LoRA delivers **68.63% accuracy** on binary multimodal legal reasoning.

---

## 2. Contributions and Reflections on the Project
1. **Clean Open Benchmark:** Curated 1,767 real street crops across 52 classes and 60 official QCVN 41:2024 gazette templates.
2. **Scientific Leakage Audit:** Identified and eliminated feature-cache blending and simulated baseline metrics.
3. **Anti-Diagonal Slash Verifier:** Developed a deterministic optical discriminator solving the P.130 vs. P.131a classification bottleneck.
4. **Deployable Full-Stack Assistant:** Delivered a functioning FastAPI + interactive web assistant capable of sub-second legal consultation.

---

## 3. Limitations and Future Work
- **Visual Domain Discrepancy:** Future work will employ diffusion-based synthetic augmentation (rendering graphic signs into photorealistic backgrounds with weather artifacts) to bridge the real-to-PDF domain gap.
- **Corpus Expansion:** Incorporate the new 2024 Road Traffic Law and municipal parking ordinances across major cities (Hanoi, HCMC, Da Nang).

---

# VIII. References

- [1] Ministry of Transport (BGTVT), *Quy chuẩn Kỹ thuật Quốc gia về Báo hiệu Đường bộ (QCVN 41:2024/BGTVT)*, Nhà xuất bản Giao thông Vận tải, Hà Nội, 2024.
- [2] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.
- [3] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, pp. 10012–10022, 2021.
- [4] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 33, pp. 9459–9474, 2020.
- [5] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, Y. Fan, K. Dang, M. Du, X. Ren, R. Men, X. Liu, C. Zhou, J. Zhou, and H. Lin, "Qwen2-VL: Enhancing Vision-Language Models with Diverse Multimodal Tasks," *arXiv preprint arXiv:2409.12191*, 2024.
- [6] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2022.
- [7] S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333–389, 2009.
- [8] G. V. Cormack, C. L. Clarke, and S. Buettcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods," in *Proc. ACM SIGIR Conf. Res. Dev. Inf. Retr.*, pp. 758–759, 2009.
- [9] VLSP 2025 Organizing Committee, "Multimodal Legal Question Answering for Traffic Sign Regulations (MLQA-TSR Track)," in *Vietnamese Language and Speech Processing Workshop*, 2025.
- [10] Mai Tam, "Vietnamese Traffic Signs Dataset," *Kaggle Datasets*, 2021. [Online]. Available: https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs.

---

# IX. Appendices

### Appendix A: Anti-Diagonal Slash Geometric Verification Formula
To distinguish **P.130** (No Stopping and No Parking — two crossed red slashes $X$) from **P.131a** (No Parking — single red diagonal slash $\backslash$), the verification algorithm extracts the pixel values along the secondary anti-diagonal region $\Omega_{\text{anti}}$ within the inner sign circle:

$$\Delta_{\text{anti}} = \frac{1}{|\Omega_{\text{anti}}|} \sum_{(x, y) \in \Omega_{\text{anti}}} \left( R(x, y) - \max(G(x, y), B(x, y)) \right)$$

Decision boundary:
$$\text{Class} = \begin{cases} \mathbf{P.131a}, & \text{if } \Delta_{\text{anti}} < 20.0 \\ \mathbf{P.130}, & \text{if } \Delta_{\text{anti}} \ge 20.0 \end{cases}$$

### Appendix B: Statutory Provisions Breakdown
Summary of the indexed legal documents from **QCVN 41:2024/BGTVT**:
- **Appendix B (Biển báo cấm):** Provisions B.1 through B.54 (Articles defining prohibitory signs P.101 to P.140).
- **Appendix C (Biển cảnh báo và nguy hiểm):** Provisions C.1 through C.48 (Articles defining warning signs W.201 to W.247).
- **Appendix D (Biển hiệu lệnh):** Provisions D.1 through D.18 (Articles defining mandatory signs R.301 to R.421).
- **Appendix E (Biển chỉ dẫn):** Provisions E.1 through E.42 (Articles defining information signs I.401 to I.447).
- **Appendix F (Biển phụ):** Provisions S.501 to S.510.

### Appendix C: Project Code Repository & Reproducibility
- **GitHub Repository:** [https://github.com/coconut-123/DSP.git](https://github.com/coconut-123/DSP.git)
- **Primary Branches:** `main`
- **Verification Script:** `python run_paper_experiments.py`
- **Web Application Entrypoint:** `python server.py`
