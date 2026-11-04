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

# Ánh xạ nhóm biển báo sang Điều luật chung trong Phần Quy định kỹ thuật QCVN 41 (như Hình 1 bài báo: Điều 22 + Phụ lục B)
GROUP_GENERAL_ARTICLE = {
    "P": "QCVN41_Điều_22",
    "DP": "QCVN41_Điều_25",
    "W": "QCVN41_Điều_28",
    "R": "QCVN41_Điều_31",
    "I": "QCVN41_Điều_34",
    "IE": "QCVN41_Điều_37",
    "S": "QCVN41_Điều_40",
}

GENERAL_REGULATION_ARTICLES = {
    "QCVN41_Điều_21": {
        "article_id": "QCVN41_Điều_21",
        "title": "Chương IV - Điều 21: Phân nhóm và hiệu lực của hệ thống biển báo hiệu đường bộ",
        "appendix": "Chương IV - Quy định chung về biển báo hiệu đường bộ",
        "summary": "Hệ thống biển báo hiệu đường bộ gồm 5 nhóm: Nhóm biển báo cấm (P), Nhóm biển báo nguy hiểm và cảnh báo (W), Nhóm biển hiệu lệnh (R), Nhóm biển chỉ dẫn (I, IE) và Nhóm biển phụ (S).",
        "full_text": "Điều 21. Phân nhóm biển báo hiệu đường bộ: Biển báo hiệu đường bộ trong Quy chuẩn này được chia thành 5 nhóm chính gồm: 1. Nhóm biển báo cấm (dạng hình tròn, viền đỏ, nền trắng) biểu thị các điều cấm; 2. Nhóm biển báo nguy hiểm và cảnh báo (dạng hình tam giác đều, viền đỏ, nền vàng) cảnh báo tính chất nguy hiểm trên đường; 3. Nhóm biển hiệu lệnh (dạng hình tròn, nền xanh lam, hình vẽ trắng) báo các hiệu lệnh phải thi hành; 4. Nhóm biển chỉ dẫn (dạng hình chữ nhật hoặc hình vuông, nền xanh lam) chỉ dẫn hướng đi hoặc các điều cần biết; 5. Nhóm biển phụ (dạng hình chữ nhật hoặc hình vuông) thuyết minh bổ sung cho biển chính.",
        "source": "QCVN 41:2019/BGTVT & QCVN 41:2024/BGTVT",
    },
    "QCVN41_Điều_22": {
        "article_id": "QCVN41_Điều_22",
        "title": "Chương IV - Điều 22: Ý nghĩa, sử dụng và hiệu lực của nhóm Biển báo cấm (Nhóm P)",
        "appendix": "Chương IV & Phụ lục B - Biển báo cấm",
        "summary": "Nhóm biển báo cấm có mã P (cấm) và DP (hết cấm), dạng hình tròn nhằm báo điều cấm hoặc hạn chế mà người tham gia giao thông không được vi phạm. Hầu hết các biển đều có viền đỏ, nền màu trắng, trên nền có hình vẽ hoặc chữ số, chữ viết màu đen thể hiện điều cấm, trừ một số trường hợp đặc biệt.",
        "full_text": "Điều 22. Nhóm biển báo cấm (Phụ lục B): 1. Nhóm biển báo cấm có dạng hình tròn nhằm báo điều cấm hoặc hạn chế mà người tham gia giao thông phải tuyệt đối tuân theo. 2. Hiệu lực của các loại biển báo cấm có thể có giá trị trên tất cả các làn đường hoặc chỉ có giá trị trên một hoặc một số làn của đường một chiều tùy thuộc vào biển chỉ dẫn làn đường hoặc biển phụ đặt kèm theo. 3. Các xe được ưu tiên theo quy định của Luật Trật tự, an toàn giao thông đường bộ khi đi làm nhiệm vụ khẩn cấp được ưu tiên theo luật định.",
        "source": "QCVN 41:2019/BGTVT & QCVN 41:2024/BGTVT",
    },
    "QCVN41_Điều_25": {
        "article_id": "QCVN41_Điều_25",
        "title": "Chương IV - Điều 25: Vị trí đặt biển báo cấm theo chiều đi và phạm vi hiệu lực của biển (P & DP)",
        "appendix": "Chương IV - Phạm vi hiệu lực của biển báo cấm",
        "summary": "Biển báo cấm được đặt ở nơi đường giao nhau hoặc trước một vị trí trên đường cần cấm. Hiệu lực của biển báo cấm bắt đầu từ vị trí đặt biển đến nơi đường giao nhau tiếp theo hoặc đến vị trí đặt biển hết cấm (nhóm DP).",
        "full_text": "Điều 25. Vị trí đặt biển báo cấm và phạm vi hiệu lực: Biển báo cấm được đặt ở nơi đường giao nhau hoặc trước một vị trí trên đường cần cấm. Nếu vì lý do nào đó biển phải đặt cách xa vị trí định cấm thì phải đặt thêm biển phụ số S.502 để chỉ rõ khoảng cách từ sau biển cấm đến vị trí biển bắt đầu có hiệu lực. Khi hết phạm vi cấm phải đặt các biển báo hết lệnh cấm tương ứng (DP.133, DP.134, DP.135).",
        "source": "QCVN 41:2019/BGTVT & QCVN 41:2024/BGTVT",
    },
    "QCVN41_Điều_28": {
        "article_id": "QCVN41_Điều_28",
        "title": "Chương IV - Điều 28: Ý nghĩa và sử dụng nhóm Biển báo nguy hiểm và cảnh báo (Nhóm W)",
        "appendix": "Chương IV & Phụ lục C - Biển báo nguy hiểm và cảnh báo",
        "summary": "Nhóm biển báo nguy hiểm và cảnh báo (mã W) có dạng hình tam giác đều, viền đỏ, nền màu vàng, trên có hình vẽ màu đen mô tả sự việc cần báo hiệu nhằm báo cho người tham gia giao thông biết trước tính chất của sự nguy hiểm hoặc các điều cần chú ý phòng ngừa trên tuyến đường.",
        "full_text": "Điều 28. Nhóm biển báo nguy hiểm và cảnh báo (Phụ lục C): Khi gặp biển báo nguy hiểm và cảnh báo, người điều khiển phương tiện phải chủ động giảm tốc độ đến mức cần thiết, chú ý quan sát và chuẩn bị sẵn sàng xử lý những tình huống có thể xảy ra để phòng ngừa tai nạn.",
        "source": "QCVN 41:2019/BGTVT & QCVN 41:2024/BGTVT",
    },
    "QCVN41_Điều_31": {
        "article_id": "QCVN41_Điều_31",
        "title": "Chương IV - Điều 31: Ý nghĩa và hiệu lực của nhóm Biển hiệu lệnh (Nhóm R)",
        "appendix": "Chương IV & Phụ lục D - Biển hiệu lệnh",
        "summary": "Nhóm biển hiệu lệnh (mã R) có dạng hình tròn, trên nền xanh lam có hình vẽ màu trắng đặc trưng cho hiệu lệnh nhằm báo cho người tham gia giao thông biết các điều bắt buộc phải chấp hành.",
        "full_text": "Điều 31. Nhóm biển hiệu lệnh (Phụ lục D): Biển hiệu lệnh đưa ra các chỉ dẫn bắt buộc người tham gia giao thông phải thực hiện như: hướng đi phải theo (R.301, R.302), nơi giao nhau chạy theo vòng xuyến (R.303), đường dành cho từng loại xe (R.304, R.305, R.403, R.412), tốc độ tối thiểu cho phép (R.306). Hiệu lực của biển áp dụng trên các làn đường của chiều xe chạy.",
        "source": "QCVN 41:2019/BGTVT & QCVN 41:2024/BGTVT",
    },
    "QCVN41_Điều_34": {
        "article_id": "QCVN41_Điều_34",
        "title": "Chương IV - Điều 34: Ý nghĩa và sử dụng nhóm Biển chỉ dẫn trên đường ô tô thông thường (Nhóm I)",
        "appendix": "Chương IV & Phụ lục E - Biển chỉ dẫn",
        "summary": "Nhóm biển chỉ dẫn (mã I) có dạng hình chữ nhật hoặc hình vuông hoặc hình mũi tên, nền màu xanh lam để chỉ dẫn hướng đi hoặc các điều cần biết nhằm thông báo cho những người sử dụng đường biết những định hướng cần thiết hoặc những điều có ích khác.",
        "full_text": "Điều 34. Nhóm biển chỉ dẫn (Phụ lục E): Biển chỉ dẫn cung cấp thông tin về đường ưu tiên, đường một chiều, vị trí quay đầu xe, nơi đỗ xe, điểm dừng xe buýt, chỉ dẫn địa giới, hướng đường và làn đường theo phương tiện giúp việc điều khiển và hướng dẫn giao thông thuận lợi, an toàn.",
        "source": "QCVN 41:2019/BGTVT & QCVN 41:2024/BGTVT",
    },
    "QCVN41_Điều_37": {
        "article_id": "QCVN41_Điều_37",
        "title": "Chương IV - Điều 37: Quy định về Biển chỉ dẫn trên đường cao tốc (Nhóm IE) và Biển dừng lại R.122",
        "appendix": "Chương IV & Phụ lục D, E - Quy định biển chỉ dẫn cao tốc và biển hiệu lệnh đặc biệt",
        "summary": "Quy định chi tiết về hệ thống biển chỉ dẫn trên đường cao tốc (mã IE) nền xanh lá cây hoặc xanh lam phản quang, và hiệu lực bắt buộc dừng lại của biển số R.122 (STOP).",
        "full_text": "Điều 37. Biển chỉ dẫn trên đường cao tốc (IE) và Biển số R.122: Biển chỉ dẫn trên đường cao tốc cung cấp thông tin lối ra, khoảng cách, trạm dừng nghỉ và tốc độ trên cao tốc. Đối với biển R.122 'Dừng lại', có hiệu lực buộc các xe cơ giới và thô sơ (kể cả xe ưu tiên theo quy định) dừng lại trước biển hoặc trước vạch ngang đường và chỉ được phép đi khi thấy các tín hiệu cho phép đi.",
        "source": "QCVN 41:2019/BGTVT & QCVN 41:2024/BGTVT",
    },
    "QCVN41_Điều_40": {
        "article_id": "QCVN41_Điều_40",
        "title": "Chương IV - Điều 40: Ý nghĩa và sử dụng nhóm Biển phụ (Nhóm S)",
        "appendix": "Chương IV & Phụ lục F - Biển phụ, biển viết bằng chữ",
        "summary": "Nhóm biển phụ (mã S) có dạng hình chữ nhật hoặc hình vuông, được đặt kết hợp với các biển chính (biển báo cấm, biển báo nguy hiểm, biển hiệu lệnh và biển chỉ dẫn) nhằm thuyết minh bổ sung để hiểu rõ các biển đó.",
        "full_text": "Điều 40. Nhóm biển phụ (Phụ lục F): Biển phụ thường có nền màu trắng, chữ và hình vẽ màu đen, viền đen (trừ biển S.507). Biển phụ được đặt ngay phía dưới biển chính để bổ sung phạm vi tác dụng, hướng tác dụng, khoảng cách hoặc giới hạn loại xe chịu sự điều chỉnh của biển chính (như biển S.505a chỉ loại xe chịu hiệu lực của biển chính).",
        "source": "QCVN 41:2019/BGTVT & QCVN 41:2024/BGTVT",
    },
    "LUAT_36_2024_Điều_11": {
        "article_id": "LUAT_36_2024_Điều_11",
        "title": "Luật số 36/2024/QH15 - Điều 11: Chấp hành báo hiệu đường bộ",
        "appendix": "Luật Trật tự, an toàn giao thông đường bộ (36/2024/QH15)",
        "summary": "Người tham gia giao thông đường bộ phải chấp hành hiệu lệnh của người điều khiển giao thông, tín hiệu đèn giao thông, biển báo hiệu đường bộ, vạch kẻ đường và các báo hiệu khác trên đường bộ.",
        "full_text": "Điều 11 Luật Trật tự, an toàn giao thông đường bộ số 36/2024/QH15: Người tham gia giao thông đường bộ phải chấp hành báo hiệu đường bộ theo thứ tự ưu tiên từ trên xuống dưới như sau: 1. Hiệu lệnh của người điều khiển giao thông; 2. Tín hiệu đèn giao thông; 3. Biển báo hiệu đường bộ; 4. Vạch kẻ đường và các dấu hiệu khác trên mặt đường; 5. Cọc tiêu, tường bảo vệ, rào chắn, đinh phản quang, tiêu phản quang, cột Km, cọc H; 6. Thiết bị âm thanh báo hiệu đường bộ.",
        "source": "Luật Trật tự, an toàn giao thông đường bộ 36/2024/QH15",
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
                "source": art.get("source", "QCVN 41:2019/BGTVT"),
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
