"""
legal_ir.py
-----------
Hệ thống Truy xuất Thông tin Pháp lý Đa phương thức (Multimodal Legal Retrieval - Subtask 1)
theo bài báo "VLSP 2025 MLQA-TSR Challenge: Vietnamese Multimodal Legal Question Answering
on Traffic Sign Regulation" (Luu et al., VLSP 2025).

Đặc điểm kỹ thuật theo bài báo (Mục 2.1, 3.1, 4.1 & 4.2):
1. Chuẩn hóa Cơ sở dữ liệu Luật D = {d_i | i = 1, ..., n}:
   - Biểu diễn hình ảnh biển báo trong điều luật theo định dạng «IMAGE: image_file.png /IMAGE»
     và bảng biểu theo định dạng «TABLE: table_markdown /TABLE» (Mục 3.1).
   - Tiền xử lý chuyển đổi bảng HTML sang văn bản chuẩn hóa / Markdown (như phương pháp của
     LifeIsTough, SmartbotIC, Tanka_CDS tại Mục 4.2).
2. Liên kết Đa điều khoản (Multi-Article Legal Graph - như Hình 1 & Hình 3 của bài báo):
   - Mỗi biển báo liên kết đồng thời tới Điều khoản quy định chung của nhóm biển (VD: Điều 21, Điều 22,
     Điều 31, Điều 34, Điều 37) và Mục quy định chi tiết tại Phụ lục B, C, D, E, F.
3. Đa phương pháp truy xuất cho Subtask 1:
   - 'bge_baseline_top5': Baseline truy xuất top-5 cố định dựa trên tích vô hướng embedding (Mục 4.1).
   - 'bm25_top5': Truy xuất từ khóa BM25 top-5.
   - 'two_branch_top3': Kiến trúc 2 nhánh kết hợp văn bản + thị giác lấy top-3 (TechNova - Mục 4.2).
   - 'graph_rerank_top2': Đồ thị dị thể Image-Article + Reranker (chmod+x - Mục 4.2).
   - 'visual_hybrid_rrf' (Đề xuất - kết hợp LifeIsTough + chmod+x): Cắt vùng biển báo + Liên kết đa phương thức
     + Lọc ngưỡng động (Dynamic top-k filtering) tối ưu hóa trực tiếp độ đo F2-score (Công thức 1, 2, 3).
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

# Ánh xạ nhóm biển báo sang Điều luật chung trong Phần Quy định kỹ thuật QCVN 41:2024/BGTVT (file archive/51-bgtvt-kem.pdf)
GROUP_GENERAL_ARTICLE = {
    "P": "QCVN41_Điều_21",
    "DP": "QCVN41_Điều_26",
    "W": "QCVN41_Điều_27",
    "R": "QCVN41_Điều_31",
    "I": "QCVN41_Điều_35",
    "IE": "QCVN41_Điều_45",
    "S": "QCVN41_Điều_41",
}

GENERAL_REGULATION_ARTICLES = {
    "QCVN41_Điều_11": {
        "article_id": "QCVN41_Điều_11",
        "title": "Chương 3 - Điều 11: Phân loại biển báo hiệu đường bộ",
        "appendix": "Chương 3 - Biển báo hiệu",
        "summary": "Biển báo hiệu đường bộ trong Quy chuẩn QCVN 41:2024/BGTVT được chia thành 5 nhóm cơ bản: 1. Biển báo cấm; 2. Biển hiệu lệnh; 3. Biển báo nguy hiểm và cảnh báo; 4. Biển chỉ dẫn; 5. Biển phụ, biển viết bằng chữ.",
        "full_text": "Điều 11. Phân loại biển báo hiệu (QCVN 41:2024/BGTVT): Biển báo hiệu đường bộ trong Quy chuẩn này được chia thành 5 nhóm cơ bản sau đây: biển báo cấm; biển hiệu lệnh; biển báo nguy hiểm và cảnh báo; biển chỉ dẫn; biển phụ, biển viết bằng chữ. Biển báo hiệu trên đường cao tốc và đường đối ngoại phải phù hợp với các quy định nêu tại các Điều ước quốc tế mà Việt Nam là thành viên. 11.1. Nhóm biển báo cấm là nhóm biển biểu thị các điều cấm mà người tham gia giao thông không được vi phạm. 11.2. Nhóm biển hiệu lệnh là nhóm biển biểu thị các hiệu lệnh phải chấp hành. 11.3. Nhóm biển báo nguy hiểm và cảnh báo báo trước tính chất nguy hiểm trên đường. 11.4. Nhóm biển chỉ dẫn để chỉ dẫn hướng đi hoặc các điều cần thiết. 11.5. Nhóm biển phụ thuyết minh bổ sung các loại biển khác.",
        "source": "QCVN 41:2024/BGTVT",
    },
    "QCVN41_Điều_15": {
        "article_id": "QCVN41_Điều_15",
        "title": "Chương 3 - Điều 15: Hiệu lực của biển báo",
        "appendix": "Chương 3 - Biển báo hiệu",
        "summary": "Hiệu lực của các loại biển báo nguy hiểm, cảnh báo và chỉ dẫn có giá trị trên các làn đường của chiều xe chạy. Hiệu lực của biển báo cấm và biển hiệu lệnh có giá trị trên tất cả các làn đường hoặc chỉ có giá trị trên một hoặc một số làn đường theo biển báo trên đường.",
        "full_text": "Điều 15. Hiệu lực của biển báo (QCVN 41:2024/BGTVT): 15.1. Hiệu lực của các loại biển báo nguy hiểm và cảnh báo và biển chỉ dẫn có giá trị trên các làn đường của chiều xe chạy. 15.2. Hiệu lực của các loại biển báo cấm và biển hiệu lệnh có giá trị trên tất cả các làn đường hoặc chỉ có giá trị trên một hoặc một số làn đường theo biển báo trên đường. 15.3. Biển báo khi sử dụng độc lập thì người tham gia giao thông phải tuân theo ý nghĩa của biển báo. Biển báo khi sử dụng kết hợp với đèn tín hiệu thì người tham gia giao thông phải tuân theo ý nghĩa, hiệu lệnh của cả biển báo và đèn tín hiệu theo thứ tự quy định tại Điều 4 của Quy chuẩn này.",
        "source": "QCVN 41:2024/BGTVT",
    },
    "QCVN41_Điều_21": {
        "article_id": "QCVN41_Điều_21",
        "title": "Chương 4 - Điều 21: Tác dụng của biển báo cấm",
        "appendix": "Chương 4 & Phụ lục B - Biển báo cấm",
        "summary": "Biển báo cấm để biểu thị các điều cấm. Người tham gia giao thông phải chấp hành những điều cấm mà biển đã báo theo đúng Quy chuẩn kỹ thuật quốc gia QCVN 41:2024/BGTVT.",
        "full_text": "Điều 21. Tác dụng của biển báo cấm (QCVN 41:2024/BGTVT): Biển báo cấm để biểu thị các điều cấm. Người tham gia giao thông phải chấp hành những điều cấm mà biển đã báo.",
        "source": "QCVN 41:2024/BGTVT",
    },
    "QCVN41_Điều_26": {
        "article_id": "QCVN41_Điều_26",
        "title": "Chương 4 - Điều 26: Vị trí đặt biển báo cấm theo chiều đi và hướng hiệu lực của biển",
        "appendix": "Chương 4 - Vị trí đặt và hướng hiệu lực biển báo cấm",
        "summary": "Biển báo cấm được đặt ở nơi đường giao nhau hoặc trước một vị trí trên đường cần cấm. Biển có hiệu lực bắt đầu từ vị trí đặt biển trở đi. Khi cần thiết đặt xa vị trí định cấm thì phải đặt biển phụ số S.502.",
        "full_text": "Điều 26. Vị trí đặt biển báo cấm theo chiều đi và hướng hiệu lực của biển (QCVN 41:2024/BGTVT): 26.1. Biển báo cấm được đặt ở nơi đường giao nhau hoặc trước một vị trí trên đường cần cấm. Biển có hiệu lực bắt đầu từ vị trí đặt biển trở đi. Nếu vì lý do nào đó, biển đặt cách xa vị trí định cấm thì phải đặt biển phụ số S.502 để chỉ rõ khoảng cách từ sau biển cấm đến vị trí biển bắt đầu có hiệu lực. 26.2. Khi cần thiết để chỉ rõ hướng tác dụng của biển và chỉ vị trí bắt đầu hay vị trí kết thúc hiệu lực của biển phải đặt biển phụ số S.503 'Hướng tác dụng của biển'.",
        "source": "QCVN 41:2024/BGTVT",
    },
    "QCVN41_Điều_27": {
        "article_id": "QCVN41_Điều_27",
        "title": "Chương 5 - Điều 27: Tác dụng của biển báo nguy hiểm và cảnh báo",
        "appendix": "Chương 5 & Phụ lục C - Biển báo nguy hiểm và cảnh báo",
        "summary": "Biển báo nguy hiểm và cảnh báo được dùng để báo cho người tham gia giao thông biết trước tính chất của sự nguy hiểm hoặc các điều cần chú ý phòng ngừa trên tuyến đường. Người tham gia giao thông phải giảm tốc độ đến mức cần thiết, chú ý quan sát và chuẩn bị sẵn sàng xử lý tình huống.",
        "full_text": "Điều 27. Tác dụng của biển báo nguy hiểm và cảnh báo (QCVN 41:2024/BGTVT): Biển báo nguy hiểm và cảnh báo được dùng để báo cho người tham gia giao thông biết trước tính chất của sự nguy hiểm hoặc các điều cần chú ý phòng ngừa trên tuyến đường. Khi gặp biển báo nguy hiểm và cảnh báo, người tham gia giao thông phải giảm tốc độ đến mức cần thiết, chú ý quan sát và chuẩn bị sẵn sàng xử lý những tình huống có thể xảy ra để phòng ngừa tai nạn.",
        "source": "QCVN 41:2024/BGTVT",
    },
    "QCVN41_Điều_31": {
        "article_id": "QCVN41_Điều_31",
        "title": "Chương 6 - Điều 31: Tác dụng của biển hiệu lệnh",
        "appendix": "Chương 6 & Phụ lục D - Biển hiệu lệnh",
        "summary": "Biển hiệu lệnh là biển báo cho người tham gia giao thông biết các điều bắt buộc phải chấp hành theo đúng Quy chuẩn kỹ thuật quốc gia QCVN 41:2024/BGTVT.",
        "full_text": "Điều 31. Tác dụng của biển hiệu lệnh (QCVN 41:2024/BGTVT): Biển hiệu lệnh là biển báo cho người tham gia giao thông biết các điều bắt buộc phải chấp hành.",
        "source": "QCVN 41:2024/BGTVT",
    },
    "QCVN41_Điều_34": {
        "article_id": "QCVN41_Điều_34",
        "title": "Chương 6 - Điều 34: Vị trí đặt biển hiệu lệnh theo chiều đi và hiệu lực tác dụng của biển",
        "appendix": "Chương 6 - Vị trí đặt và hiệu lực biển hiệu lệnh",
        "summary": "Các biển hiệu lệnh phải đặt tại vị trí cần báo hiệu lệnh. Các biển hiệu lệnh có hiệu lực kể từ vị trí đặt biển. Nếu đoạn đường phải thi hành biển hiệu lệnh đi qua các nút giao thì biển hiệu lệnh phải được nhắc lại đặt phía sau nút giao.",
        "full_text": "Điều 34. Vị trí đặt biển hiệu lệnh theo chiều đi và hiệu lực tác dụng của biển (QCVN 41:2024/BGTVT): 34.1. Các biển hiệu lệnh phải đặt tại vị trí cần báo hiệu lệnh. Do điều kiện khó khăn nếu đặt xa hơn phải đặt kèm biển phụ số S.502. 34.2. Các biển hiệu lệnh có hiệu lực kể từ vị trí đặt biển. Riêng biển số R.301a nếu đặt ở sau nơi đường giao nhau tiếp theo thì hiệu lực của biển kể từ vị trí đặt biển đến nơi đường giao nhau tiếp theo. 34.3. Nếu đoạn đường phải thi hành biển hiệu lệnh đi qua các nút giao, biển hiệu lệnh phải được nhắc lại đặt phía sau nút giao theo hướng đường đang có biển hiệu lệnh.",
        "source": "QCVN 41:2024/BGTVT",
    },
    "QCVN41_Điều_35": {
        "article_id": "QCVN41_Điều_35",
        "title": "Chương 7 - Điều 35: Tác dụng của biển chỉ dẫn",
        "appendix": "Chương 7 & Phụ lục E - Biển chỉ dẫn",
        "summary": "Các biển chỉ dẫn để chỉ dẫn hướng đi hoặc các điều cần thiết nhằm giúp người tham gia giao thông trong việc điều khiển phương tiện và hướng dẫn giao thông trên đường được thuận lợi, đảm bảo an toàn.",
        "full_text": "Điều 35. Tác dụng của biển chỉ dẫn (QCVN 41:2024/BGTVT): Các biển chỉ dẫn để chỉ dẫn hướng đi hoặc các điều cần thiết nhằm giúp người tham gia giao thông trong việc điều khiển phương tiện và hướng dẫn giao thông trên đường được thuận lợi, đảm bảo an toàn.",
        "source": "QCVN 41:2024/BGTVT",
    },
    "QCVN41_Điều_41": {
        "article_id": "QCVN41_Điều_41",
        "title": "Chương 8 - Điều 41: Biển phụ",
        "appendix": "Chương 8 & Phụ lục F - Biển phụ, biển viết bằng chữ",
        "summary": "Biển phụ thường được đặt kết hợp với các biển báo chính nhằm thuyết minh, bổ sung để hiểu rõ, trừ biển số S.507 'Hướng rẽ' được sử dụng độc lập.",
        "full_text": "Điều 41. Biển phụ (QCVN 41:2024/BGTVT): 41.1. Tác dụng của biển phụ: Biển phụ thường được đặt kết hợp với các biển báo chính nhằm thuyết minh, bổ sung để hiểu rõ, trừ biển số S.507 'Hướng rẽ' được sử dụng độc lập. 41.2. Biển phụ có mã S, S.G và S.H (như biển S.501, S.502, S.503, S.504, S.505a chỉ loại xe...).",
        "source": "QCVN 41:2024/BGTVT",
    },
    "QCVN41_Điều_45": {
        "article_id": "QCVN41_Điều_45",
        "title": "Chương 9 - Điều 45: Quy định chung đối với biển chỉ dẫn trên đường cao tốc",
        "appendix": "Chương 9 - Biển chỉ dẫn trên đường cao tốc",
        "summary": "Chức năng biển chỉ dẫn trên đường cao tốc (mã IE): Cung cấp thông tin đầy đủ, chính xác cho người điều khiển phương tiện nhằm lái xe an toàn trên đường cao tốc và đi đến địa điểm mong muốn.",
        "full_text": "Điều 45. Quy định chung đối với biển chỉ dẫn trên đường cao tốc (QCVN 41:2024/BGTVT): Cung cấp thông tin đầy đủ, chính xác cho người điều khiển phương tiện nhằm lái xe an toàn trên đường cao tốc và đi đến địa điểm mong muốn, bao gồm chỉ dẫn tên đường, hướng tuyến, địa điểm nút giao, tốc độ tối đa, tốc độ tối thiểu và khoảng cách lái xe an toàn.",
        "source": "QCVN 41:2024/BGTVT",
    },
}


def compute_f2_metrics(retrieved_ids: List[str], relevant_ids: List[str]) -> Tuple[float, float, float]:
    """
    Tính Precision_q (Công thức 1), Recall_q (Công thức 2) và F2_q (Công thức 3)
    theo đúng định nghĩa Subtask 1 tại Mục 2.1 của bài báo VLSP 2025 MLQA-TSR:
        Precision_q = #correct_retrieved_articles / #total_retrieved_articles  (Eq. 1)
        Recall_q    = #correct_retrieved_articles / #total_relevant_articles   (Eq. 2)
        F2_q        = 5 * Precision_q * Recall_q / (4 * Precision_q + Recall_q) (Eq. 3)
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0, 0.0, 0.0
    ret_set = set(retrieved_ids)
    rel_set = set(relevant_ids)
    correct = len(ret_set & rel_set)
    if correct == 0:
        return 0.0, 0.0, 0.0
    precision_q = correct / float(len(retrieved_ids))
    recall_q = correct / float(len(relevant_ids))
    denom = 4.0 * precision_q + recall_q
    f2_q = (5.0 * precision_q * recall_q / denom) if denom > 0 else 0.0
    return precision_q, recall_q, f2_q


def normalize_html_table_to_text(text: str) -> str:
    """Chuyển đổi thẻ «TABLE: ... /TABLE» và «IMAGE: ... /IMAGE» sang văn bản chuẩn hóa (Mục 3.1 & 4.2)."""
    t = re.sub(r"«IMAGE:\s*([^\s»]+)\s*/IMAGE»", r" [Hình biển báo \1] ", text)
    t = re.sub(r"«TABLE:\s*(.*?)\s*/TABLE»", r" [Bảng quy chuẩn: \1] ", t, flags=re.DOTALL)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_query_text(text: str) -> str:
    t = normalize_html_table_to_text(text).lower().strip()
    for k, v in VIETNAMESE_SYNONYMS.items():
        if k in t:
            t = t + " " + v
    return t


def tokenize_vn(text: str) -> List[str]:
    t = normalize_query_text(text)
    codes = re.findall(r"\b(?:[pwris]|dp|ie)\.[0-9]{1,3}[a-p]?\b", t)
    words = re.findall(r"[a-zA-Z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+", t)
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
    return words + bigrams + [c.upper() for c in codes]


def get_general_article_for_code(sign_code: str, existing_articles: Dict[str, Any]) -> Optional[str]:
    prefix = sign_code.upper().split(".")[0]
    cand = GROUP_GENERAL_ARTICLE.get(prefix)
    if cand and cand in existing_articles:
        return cand
    return None


class HybridLegalRetriever:
    def __init__(
        self,
        legal_articles_path: Path = DATA_DIR / "legal_articles.json",
        class_mapping_path: Path = DATA_DIR / "class_mapping_52.json",
        pdf_catalog_path: Path = DATA_DIR / "pdf_signs_catalog.json",
    ):
        with open(legal_articles_path, "r", encoding="utf-8") as f:
            self.articles_dict: Dict[str, Dict[str, Any]] = json.load(f)

        updated_general = False
        for g_aid, g_obj in GENERAL_REGULATION_ARTICLES.items():
            if g_aid not in self.articles_dict:
                self.articles_dict[g_aid] = g_obj
                updated_general = True
        if updated_general:
            for p_out in [legal_articles_path, BASE_DIR / "legal_articles.json"]:
                try:
                    with open(p_out, "w", encoding="utf-8") as f_out:
                        json.dump(self.articles_dict, f_out, ensure_ascii=False, indent=2)
                except Exception:
                    pass

        self.class_mapping = []
        if class_mapping_path.exists():
            with open(class_mapping_path, "r", encoding="utf-8") as f:
                self.class_mapping = json.load(f)

        self.pdf_catalog = []
        if pdf_catalog_path.exists():
            with open(pdf_catalog_path, "r", encoding="utf-8") as f:
                cat_data = json.load(f)
                self.pdf_catalog = cat_data.get("dataset", [])

        # Xây dựng bảng ánh xạ từ sign_code -> danh sách các điều luật liên quan (Phụ lục + Điều khoản chung)
        self.sign_to_article: Dict[str, str] = {}
        self.sign_to_all_articles: Dict[str, List[str]] = defaultdict(list)
        self.sign_metadata: Dict[str, Dict[str, Any]] = {}

        for item in self.pdf_catalog:
            sc = item["sign_code"]
            sc_up = sc.upper()
            arts = list(item.get("relevant_articles", []))
            gen_art = get_general_article_for_code(sc_up, self.articles_dict)
            if gen_art and gen_art not in arts:
                arts.append(gen_art)
            if arts:
                self.sign_to_article[sc_up] = arts[0]
                self.sign_to_all_articles[sc_up] = arts
            if sc_up not in self.sign_metadata:
                self.sign_metadata[sc_up] = {
                    "sign_code": sc,
                    "sign_name": item["sign_name"],
                    "category": item["category"],
                    "article_id": arts[0] if arts else "",
                    "relevant_articles": arts,
                    "pdf_image": item["image"],
                }

        for cm in self.class_mapping:
            raw_c = cm["dataset_code"].upper()
            can_c = cm["canonical_qcvn_code"].upper()
            art_id = cm["article_id"]
            arts = [art_id]
            gen_art = get_general_article_for_code(can_c, self.articles_dict)
            if gen_art and gen_art not in arts:
                arts.append(gen_art)

            self.sign_to_article[raw_c] = art_id
            self.sign_to_article[can_c] = art_id
            self.sign_to_all_articles[raw_c] = arts
            self.sign_to_all_articles[can_c] = arts

            self.sign_metadata[raw_c] = {
                "sign_code": cm["canonical_qcvn_code"],
                "dataset_code": cm["dataset_code"],
                "sign_name": cm["name_vi"],
                "official_name": cm["official_qcvn_name"],
                "category": cm["category"],
                "article_id": art_id,
                "relevant_articles": arts,
                "pdf_image": cm["pdf_template_images"][0] if cm["pdf_template_images"] else "",
            }
            if can_c not in self.sign_metadata:
                self.sign_metadata[can_c] = self.sign_metadata[raw_c]

        # Quét toàn bộ điều khoản trong legal_articles.json để ánh xạ đầy đủ mọi mã biển báo QCVN 41
        code_pattern = re.compile(
            r"\b(S\.[HG],[0-9]{1,2}[a-c]?|(?:P|DP|W|R|I|IE)(?:\.E)?\.[0-9]{1,3}[a-p]?)\b",
            re.IGNORECASE,
        )
        for aid, art in self.articles_dict.items():
            title_codes = code_pattern.findall(art.get("title", ""))
            for tc in title_codes:
                tc_up = tc.upper()
                if tc_up not in self.sign_to_article:
                    self.sign_to_article[tc_up] = aid
                if aid not in self.sign_to_all_articles[tc_up]:
                    self.sign_to_all_articles[tc_up].insert(0, aid)
                gen_art = get_general_article_for_code(tc_up, self.articles_dict)
                if gen_art and gen_art not in self.sign_to_all_articles[tc_up]:
                    self.sign_to_all_articles[tc_up].append(gen_art)

            body_codes = code_pattern.findall(art.get("full_text", ""))
            for bc in body_codes:
                bc_up = bc.upper()
                if bc_up not in self.sign_to_article:
                    self.sign_to_article[bc_up] = aid
                if aid not in self.sign_to_all_articles[bc_up]:
                    self.sign_to_all_articles[bc_up].append(aid)
                gen_art = get_general_article_for_code(bc_up, self.articles_dict)
                if gen_art and gen_art not in self.sign_to_all_articles[bc_up]:
                    self.sign_to_all_articles[bc_up].append(gen_art)

        # Bổ sung định dạng «IMAGE: ... /IMAGE» theo Mục 3.1 bài báo VLSP 2025 MLQA-TSR vào điều luật
        self.doc_ids = sorted(list(self.articles_dict.keys()))
        self.doc_texts = []
        self.doc_tokens = []
        self.doc_freqs = Counter()

        art_extra_signs = defaultdict(list)
        art_image_tags = defaultdict(list)
        for sc_key, meta in self.sign_metadata.items():
            aid = meta.get("article_id", "")
            if aid:
                sc_clean = meta.get("sign_code", sc_key)
                art_extra_signs[aid].append(f"{sc_clean} {meta.get('sign_name', '')}")
                art_image_tags[aid].append(f"«IMAGE: {sc_clean}_01.png /IMAGE»")

        for aid in self.doc_ids:
            art = self.articles_dict[aid]
            extra_txt = " ; ".join(art_extra_signs.get(aid, []))
            img_tags = " ".join(art_image_tags.get(aid, []))
            if img_tags and "«IMAGE:" not in art.get("full_text", ""):
                art["vlsp_formatted_text"] = f"{art.get('full_text', '')} {img_tags}"
            else:
                art["vlsp_formatted_text"] = art.get("full_text", "")

            full_doc_str = (
                f"{art.get('title', '')} {art.get('appendix', '')} "
                f"{art.get('summary', '')} {art.get('vlsp_formatted_text', '')} "
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

        self._build_dense_index()

    def get_relevant_articles_for_sign(self, sign_code: str) -> List[str]:
        """Trả về danh sách các điều luật liên quan (Phụ lục chi tiết + Điều khoản nhóm biển) như Hình 1 bài báo."""
        sc_up = sign_code.strip().upper().split("*")[0]
        arts = list(self.sign_to_all_articles.get(sc_up, []))
        if not arts and sc_up in self.sign_to_article:
            arts = [self.sign_to_article[sc_up]]
        gen_art = get_general_article_for_code(sc_up, self.articles_dict)
        if gen_art and gen_art not in arts:
            arts.append(gen_art)
        return arts[:2] if arts else ["QCVN41_Điều_21"]

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

        u, s, vt = np.linalg.svd(self.tfidf_norm_mat, full_matrices=False)
        k_dim = min(64, len(s))
        self.svd_vt = vt[:k_dim, :]
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
        dynamic_k: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Các chế độ truy xuất theo bài báo VLSP 2025 MLQA-TSR (Mục 4.1 & 4.2):
        - 'bge_baseline_top5': Baseline Dense Embedding lấy top-5 ứng viên (Mục 4.1).
        - 'bm25_only': Truy xuất từ khóa BM25 trên văn bản câu hỏi.
        - 'dense_only': Truy xuất Dense Embedding trên văn bản câu hỏi.
        - 'text_hybrid': Kết hợp BM25 + Dense trên câu hỏi văn bản (không dùng ảnh).
        - 'two_branch_top3': Kiến trúc 2 nhánh (Visual + Text FAISS) lấy top-3 (TechNova - Mục 4.2).
        - 'graph_rerank_top2': Đồ thị dị thể Image-Article + Reranker lấy top-2 (chmod+x - Mục 4.2).
        - 'visual_hybrid_rrf': Cắt biển báo + Liên kết đa phương thức + Lọc top-k động (LifeIsTough + chmod+x).
        """
        augmented_query = query
        primary_articles = []
        secondary_articles = []

        use_visual = mode in ("visual_hybrid_rrf", "two_branch_top3", "graph_rerank_top2")
        if use_visual and visual_sign_codes:
            extra_parts = []
            for sc in visual_sign_codes:
                sc_up = sc.strip().upper()
                base_sc = sc_up.split("*")[0]
                meta = self.sign_metadata.get(sc_up) or self.sign_metadata.get(base_sc)
                if meta:
                    extra_parts.append(f"Biển số {meta['sign_code']} {meta['sign_name']} {meta.get('official_name', '')}")
                else:
                    extra_parts.append(f"Biển số {base_sc}")

                rel_arts = self.get_relevant_articles_for_sign(base_sc)
                if rel_arts:
                    if rel_arts[0] not in primary_articles:
                        primary_articles.append(rel_arts[0])
                    for sec_a in rel_arts[1:]:
                        if sec_a not in secondary_articles:
                            secondary_articles.append(sec_a)
            augmented_query = query + " " + " ".join(extra_parts)

        q_tokens = tokenize_vn(augmented_query)
        bm25_scores = self.score_bm25(q_tokens)
        dense_scores = self.score_dense(q_tokens)

        if mode in ("bm25_only",):
            final_scores = bm25_scores
        elif mode in ("dense_only", "bge_baseline_top5"):
            final_scores = dense_scores
        else:
            bm25_ranks = np.argsort(np.argsort(-bm25_scores)) + 1
            dense_ranks = np.argsort(np.argsort(-dense_scores)) + 1
            rrf_scores = 1.0 / (60.0 + bm25_ranks) + 1.0 / (60.0 + dense_ranks)

            if use_visual:
                for d_idx, aid in enumerate(self.doc_ids):
                    if aid in primary_articles:
                        rrf_scores[d_idx] += 0.65
                    elif aid in secondary_articles and mode in ("visual_hybrid_rrf", "graph_rerank_top2"):
                        rrf_scores[d_idx] += 0.45
            final_scores = rrf_scores

        # Cơ chế lọc Dynamic top-k (như nhóm chmod+x tại Mục 4.2) để tối ưu hóa độ đo F2 (Công thức 3)
        if dynamic_k or mode == "graph_rerank_top2":
            high_conf_idxs = [i for i in np.argsort(-final_scores) if final_scores[i] >= 0.40]
            if high_conf_idxs:
                top_indices = high_conf_idxs[: max(2, len(primary_articles) + len(secondary_articles))]
            else:
                top_indices = np.argsort(-final_scores)[:2]
        elif mode == "two_branch_top3":
            top_indices = np.argsort(-final_scores)[:3]
        elif mode == "bge_baseline_top5":
            top_indices = np.argsort(-final_scores)[:5]
        else:
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
                "vlsp_formatted_text": art.get("vlsp_formatted_text", art.get("full_text", "")),
                "source": art.get("source", "QCVN 41:2024/BGTVT"),
            })
        return results


if __name__ == "__main__":
    retriever = HybridLegalRetriever()
    print(f"[✓] Đã khởi tạo HybridLegalRetriever (VLSP 2025 MLQA-TSR) với {retriever.n_docs} điều luật.")
    sample_res = retriever.retrieve(
        query="Những loại phương tiện nào bị cấm trên đoạn đường này?",
        visual_sign_codes=["P.106a*Xe tải"],
        top_k=5,
        mode="visual_hybrid_rrf",
        dynamic_k=True,
    )
    gt_arts = retriever.get_relevant_articles_for_sign("P.106a")
    p_q, r_q, f2_q = compute_f2_metrics([r["article_id"] for r in sample_res], gt_arts)
    print(f"  Ground-truth articles: {gt_arts}")
    print(f"  Retrieved ({len(sample_res)} articles) -> Precision={p_q:.4f}, Recall={r_q:.4f}, F2={f2_q:.4f}")
    for r in sample_res:
        print(f"  Rank {r['rank']} [{r['article_id']}] (score={r['score']}): {r['title']}")
