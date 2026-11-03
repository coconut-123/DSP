"""
legal_ir.py
-----------
Hệ thống Truy xuất Thông tin Pháp lý Lai ghép (Hybrid Legal Information Retrieval - Legal IR)
cho Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ QCVN 41:2019/BGTVT (trích xuất từ 51-bgtvt-kem.pdf).

Kiến trúc gồm 3 thành phần:
1. Sparse Lexical Retriever (Okapi BM25): Chuẩn hóa văn bản tiếng Việt, tách từ kết hợp từ đồng nghĩa giao thông.
2. Dense / Subword Semantic Retriever (TF-IDF Word + Char N-gram + Truncated SVD / Cosine Similarity).
3. Visual-Conditioned Reciprocal Rank Fusion (Visual-RRF): Tích hợp tín hiệu thị giác (mã biển báo nhận diện từ VLM/ViT)
   với truy vấn ngôn ngữ tự nhiên của người dùng để truy xuất chính xác điều khoản pháp lý trong QCVN 41:2019/BGTVT.
"""

import os
import sys
import re
import math
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

VIETNAMESE_SYNONYMS = {
    "xe hơi": "ô tô con",
    "xế hộp": "ô tô con",
    "xe tải": "ô tô tải",
    "xe khách": "ô tô khách",
    "xe buýt": "buýt ô tô khách",
    "xe công": "sơ-mi rơ-moóc container",
    "xe container": "sơ-mi rơ-moóc",
    "đầu kéo": "sơ-mi rơ-moóc kéo",
    "xe máy": "mô tô xe gắn máy hai bánh",
    "mô tô": "xe máy hai bánh",
    "quay đầu": "quay xe u-turn",
    "đậu xe": "đỗ xe dừng xe",
    "dừng đỗ": "dừng xe và đỗ xe",
    "ngược chiều": "cấm đi ngược chiều một chiều",
    "tốc độ": "tốc độ tối đa giới hạn km/h",
    "phạt nguội": "camera giám sát xử phạt",
    "gồ giảm tốc": "sóng mấp mô nhân tạo không bằng phẳng",
    "người đi bộ": "đi bộ cắt ngang sang ngang",
}


def normalize_query_text(text: str) -> str:
    t = text.lower().strip()
    for k, v in VIETNAMESE_SYNONYMS.items():
        if k in t:
            t = t + " " + v
    return t


def tokenize_vn(text: str) -> List[str]:
    t = normalize_query_text(text)
    # Giữ các mã biển như p.102, w.224, r.302a, s.505a nguyên vẹn
    codes = re.findall(r"\b(?:[pwris]|dp|ie)\.[0-9]{1,3}[a-p]?\b", t)
    words = re.findall(r"[a-zA-Z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+", t)
    # Tạo thêm bigrams để bắt cụm từ tiếng Việt ("rẽ trái", "quay đầu", "cấm đỗ", "ngược chiều")
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
    return words + bigrams + [c.upper() for c in codes]


class HybridLegalRetriever:
    def __init__(
        self,
        legal_articles_path: Path = DATA_DIR / "legal_articles.json",
        class_mapping_path: Path = DATA_DIR / "class_mapping_52.json",
        pdf_catalog_path: Path = DATA_DIR / "pdf_signs_catalog.json",
    ):
        with open(legal_articles_path, "r", encoding="utf-8") as f:
            self.articles_dict: Dict[str, Dict[str, Any]] = json.load(f)

        self.class_mapping = []
        if class_mapping_path.exists():
            with open(class_mapping_path, "r", encoding="utf-8") as f:
                self.class_mapping = json.load(f)

        self.pdf_catalog = []
        if pdf_catalog_path.exists():
            with open(pdf_catalog_path, "r", encoding="utf-8") as f:
                cat_data = json.load(f)
                self.pdf_catalog = cat_data.get("dataset", [])

        # Xây dựng bảng ánh xạ từ sign_code -> article_id và metadata biển báo
        self.sign_to_article: Dict[str, str] = {}
        self.sign_metadata: Dict[str, Dict[str, Any]] = {}

        for item in self.pdf_catalog:
            sc = item["sign_code"]
            arts = item.get("relevant_articles", [])
            if arts:
                self.sign_to_article[sc.upper()] = arts[0]
            if sc.upper() not in self.sign_metadata:
                self.sign_metadata[sc.upper()] = {
                    "sign_code": sc,
                    "sign_name": item["sign_name"],
                    "category": item["category"],
                    "article_id": arts[0] if arts else "",
                    "pdf_image": item["image"],
                }

        for cm in self.class_mapping:
            raw_c = cm["dataset_code"].upper()
            can_c = cm["canonical_qcvn_code"].upper()
            art_id = cm["article_id"]
            self.sign_to_article[raw_c] = art_id
            self.sign_to_article[can_c] = art_id
            self.sign_metadata[raw_c] = {
                "sign_code": cm["canonical_qcvn_code"],
                "dataset_code": cm["dataset_code"],
                "sign_name": cm["name_vi"],
                "official_name": cm["official_qcvn_name"],
                "category": cm["category"],
                "article_id": art_id,
                "pdf_image": cm["pdf_template_images"][0] if cm["pdf_template_images"] else "",
            }
            if can_c not in self.sign_metadata:
                self.sign_metadata[can_c] = self.sign_metadata[raw_c]

        # Bổ sung thông tin tên biển báo vào văn bản điều luật để tăng độ phủ ngữ nghĩa
        self.doc_ids = sorted(list(self.articles_dict.keys()))
        self.doc_texts = []
        self.doc_tokens = []
        self.doc_freqs = Counter()

        art_extra_signs = defaultdict(list)
        for sc_key, meta in self.sign_metadata.items():
            aid = meta.get("article_id", "")
            if aid:
                art_extra_signs[aid].append(f"{meta.get('sign_code', '')} {meta.get('sign_name', '')}")

        for aid in self.doc_ids:
            art = self.articles_dict[aid]
            extra_txt = " ; ".join(art_extra_signs.get(aid, []))
            full_doc_str = (
                f"{art.get('title', '')} {art.get('appendix', '')} "
                f"{art.get('summary', '')} {art.get('full_text', '')} "
                f"{extra_txt}"
            )
            self.doc_texts.append(full_doc_str)
            toks = tokenize_vn(full_doc_str)
            self.doc_tokens.append(toks)
            for unique_t in set(toks):
                self.doc_freqs[unique_t] += 1

        self.n_docs = len(self.doc_ids)
        self.avg_dl = np.mean([len(t) for t in self.doc_tokens]) if self.n_docs > 0 else 1.0
        self.doc_term_counts = [Counter(t) for t in self.doc_tokens]

        # Xây dựng không gian vector ngữ nghĩa (TF-IDF + SVD Embeddings)
        self._build_dense_index()

    def _build_dense_index(self):
        vocab = [term for term, df in self.doc_freqs.items() if df >= 1]
        self.vocab_to_idx = {t: i for i, t in enumerate(vocab)}
        n_vocab = len(vocab)
        tfidf_mat = np.zeros((self.n_docs, n_vocab), dtype=np.float32)

        for d_idx, t_counts in enumerate(self.doc_term_counts):
            dl = max(1, len(self.doc_tokens[d_idx]))
            for term, cnt in t_counts.items():
                v_idx = self.vocab_to_idx.get(term)
                if v_idx is not None:
                    tf = (cnt / dl) ** 0.75
                    idf = math.log(1.0 + (self.n_docs - self.doc_freqs[term] + 0.5) / (self.doc_freqs[term] + 0.5))
                    tfidf_mat[d_idx, v_idx] = tf * idf

        norms = np.linalg.norm(tfidf_mat, axis=1, keepdims=True)
        self.tfidf_norm_mat = tfidf_mat / np.maximum(norms, 1e-8)

        # Nén chiều bằng SVD (Latent Semantic Indexing) tạo Dense Embeddings
        u, s, vt = np.linalg.svd(self.tfidf_norm_mat, full_matrices=False)
        k_dim = min(64, len(s))
        self.svd_vt = vt[:k_dim, :]  # (k_dim, n_vocab)
        doc_dense = u[:, :k_dim] * s[:k_dim]
        dense_norms = np.linalg.norm(doc_dense, axis=1, keepdims=True)
        self.doc_dense_vecs = doc_dense / np.maximum(dense_norms, 1e-8)

    def score_bm25(self, query_tokens: List[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for q_term in query_tokens:
            df = self.doc_freqs.get(q_term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            for d_idx in range(self.n_docs):
                tf = self.doc_term_counts[d_idx].get(q_term, 0)
                if tf > 0:
                    dl = len(self.doc_tokens[d_idx])
                    denom = tf + k1 * (1.0 - b + b * (dl / self.avg_dl))
                    scores[d_idx] += idf * ((tf * (k1 + 1.0)) / denom)
        return scores

    def score_dense(self, query_tokens: List[str]) -> np.ndarray:
        q_vec = np.zeros(len(self.vocab_to_idx), dtype=np.float32)
        q_counts = Counter(query_tokens)
        dl = max(1, len(query_tokens))
        for term, cnt in q_counts.items():
            v_idx = self.vocab_to_idx.get(term)
            if v_idx is not None:
                tf = (cnt / dl) ** 0.75
                idf = math.log(1.0 + (self.n_docs - self.doc_freqs[term] + 0.5) / (self.doc_freqs[term] + 0.5))
                q_vec[v_idx] = tf * idf

        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-8:
            return np.zeros(self.n_docs, dtype=np.float32)
        q_vec = q_vec / q_norm
        tfidf_sim = self.tfidf_norm_mat @ q_vec

        q_dense = self.svd_vt @ q_vec
        qd_norm = np.linalg.norm(q_dense)
        if qd_norm > 1e-8:
            q_dense = q_dense / qd_norm
            dense_sim = self.doc_dense_vecs @ q_dense
        else:
            dense_sim = np.zeros(self.n_docs, dtype=np.float32)

        return 0.65 * tfidf_sim + 0.35 * dense_sim

    def retrieve(
        self,
        query: str,
        visual_sign_codes: Optional[List[str]] = None,
        top_k: int = 5,
        mode: str = "visual_hybrid_rrf",
    ) -> List[Dict[str, Any]]:
        """
        Các chế độ truy xuất (mode) để phục vụ Ablation Study:
        - 'bm25_only': Chỉ dùng từ khóa BM25 trên câu hỏi văn bản.
        - 'dense_only': Chỉ dùng vector ngữ nghĩa Dense/TF-IDF trên câu hỏi văn bản.
        - 'text_hybrid': Kết hợp BM25 + Dense trên câu hỏi văn bản (không có tín hiệu thị giác).
        - 'visual_hybrid_rrf': Tích hợp mã biển báo thị giác (từ VLM/Detector) + câu hỏi văn bản qua RRF.
        """
        augmented_query = query
        matched_target_articles = set()

        if mode == "visual_hybrid_rrf" and visual_sign_codes:
            extra_parts = []
            for sc in visual_sign_codes:
                sc_up = sc.strip().upper()
                base_sc = sc_up.split("*")[0]
                meta = self.sign_metadata.get(sc_up) or self.sign_metadata.get(base_sc)
                if meta:
                    extra_parts.append(f"Biển số {meta['sign_code']} {meta['sign_name']} {meta.get('official_name', '')}")
                    if meta.get("article_id"):
                        matched_target_articles.add(meta["article_id"])
                else:
                    extra_parts.append(f"Biển số {base_sc}")
                if sc_up in self.sign_to_article:
                    matched_target_articles.add(self.sign_to_article[sc_up])
                elif base_sc in self.sign_to_article:
                    matched_target_articles.add(self.sign_to_article[base_sc])
            augmented_query = query + " " + " ".join(extra_parts)

        q_tokens = tokenize_vn(augmented_query)
        bm25_scores = self.score_bm25(q_tokens)
        dense_scores = self.score_dense(q_tokens)

        if mode == "bm25_only":
            final_scores = bm25_scores
        elif mode == "dense_only":
            final_scores = dense_scores
        else:
            # Reciprocal Rank Fusion (RRF, k=60)
            bm25_ranks = np.argsort(np.argsort(-bm25_scores)) + 1
            dense_ranks = np.argsort(np.argsort(-dense_scores)) + 1
            rrf_scores = 1.0 / (60.0 + bm25_ranks) + 1.0 / (60.0 + dense_ranks)

            if mode == "visual_hybrid_rrf" and matched_target_articles:
                for d_idx, aid in enumerate(self.doc_ids):
                    if aid in matched_target_articles:
                        rrf_scores[d_idx] += 0.50  # Điểm thưởng neo pháp lý từ nhận thức thị giác
            final_scores = rrf_scores

        top_indices = np.argsort(-final_scores)[:top_k]
        results = []
        for rank_idx, d_idx in enumerate(top_indices):
            aid = self.doc_ids[d_idx]
            art = self.articles_dict[aid]
            results.append({
                "rank": rank_idx + 1,
                "article_id": aid,
                "score": float(round(final_scores[d_idx], 4)),
                "title": art.get("title", ""),
                "appendix": art.get("appendix", ""),
                "summary": art.get("summary", ""),
                "full_text": art.get("full_text", ""),
                "source": art.get("source", "QCVN 41:2019/BGTVT"),
            })
        return results


if __name__ == "__main__":
    retriever = HybridLegalRetriever()
    print(f"[✓] Đã khởi tạo HybridLegalRetriever với {retriever.n_docs} điều luật và {len(retriever.sign_metadata)} mục biển báo.")
    sample_res = retriever.retrieve(
        query="Tôi đang lái xe ô tô tải gặp biển này thì có được đi vào không và quy định thế nào?",
        visual_sign_codes=["P.106a*Xe tải"],
        top_k=3,
        mode="visual_hybrid_rrf",
    )
    for r in sample_res:
        print(f"  Rank {r['rank']} [{r['article_id']}] (score={r['score']}): {r['title']}")
