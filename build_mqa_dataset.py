r"""
build_mqa_dataset.py
--------------------
Xây dựng Bộ dữ liệu Hỏi - Đáp Pháp lý Đa phương thức về Biển báo Giao thông Việt Nam
theo đúng cấu trúc bài báo "VLSP 2025 MLQA-TSR Challenge: Vietnamese Multimodal Legal
Question Answering on Traffic Sign Regulation" (Luu et al., VLSP 2025 - Mục 2, 3 & 4.1).

Đặc điểm bộ dữ liệu theo bài báo VLSP 2025 MLQA-TSR:
1. Phân chia dữ liệu theo yêu cầu:
   - Tập Huấn luyện (Train Set): Toàn bộ 1.807 ảnh cắt biển báo thực tế từ đường phố Việt Nam
     trong `C:\ki8\DSP\datadsp\data\real_crops` (52 lớp).
   - Tập Đánh giá / Kiểm thử (Evaluation / Test Set): 51 ảnh biển báo chuẩn trích xuất từ PDF
     QCVN 41 trong `C:\ki8\DSP\datadsp\data\pdf_signs` (khớp với các mã biển đã có trong tập Train).
2. Hai tác vụ chuẩn VLSP 2025 MLQA-TSR (Mục 2.1 & 2.2):
   - Subtask 1 (Multimodal Legal Retrieval): Truy xuất danh sách các điều luật liên quan R = {d_i}
     từ cơ sở dữ liệu luật D (mỗi câu hỏi có từ 1 đến 3 điều luật liên quan, trung bình ~2.0 điều luật
     như Hình 1, Bảng 2 & Hình 3 của bài báo), đánh giá bằng Precision, Recall và F2-score (Công thức 1, 2, 3).
   - Subtask 2 (Multimodal Legal Question Answering): CHỈ dùng câu hỏi Đúng/Sai
     (Yes/No: "Đúng" hoặc "Sai"). Dạng câu hỏi Trắc nghiệm 4 lựa chọn (Multiple-Choice A, B, C, D)
     đã được loại bỏ hoàn toàn khỏi bộ dữ liệu. Đánh giá bằng Accuracy (Công thức 4).

LƯU Ý TRUNG THỰC: Câu hỏi và nhãn `relevant_articles` được SINH TỰ ĐỘNG từ bảng ánh xạ
mã biển -> điều khoản (HybridLegalRetriever.get_relevant_articles_for_sign). Đây không phải nhãn do người gán,
nên kết quả Subtask 1 khi retriever dùng cùng bảng ánh xạ này chỉ phản ánh độ chính xác nhận diện mã biển.
"""

import os
import sys
import json
from pathlib import Path
from collections import Counter

from legal_ir import HybridLegalRetriever

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def segment_vietnamese_tokens(text: str):
    """Tách từ tiếng Việt tính độ dài câu hỏi (theo Mục 3.2 - Bảng 2 bài báo VLSP 2025 MLQA-TSR)."""
    try:
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize(text).split()
    except Exception:
        return text.strip().split()


def format_mc_prompt(question: str, choices: dict) -> str:
    """Mẫu câu lệnh (Prompt Template) chuẩn cho câu hỏi Multiple-Choice theo Mục 4.1 bài báo VLSP 2025."""
    return (
        f"<image>\n"
        f"{question}\n"
        f"A. {choices['A']}\n"
        f"B. {choices['B']}\n"
        f"C. {choices['C']}\n"
        f"D. {choices['D']}\n\n"
        f"Trả lời bằng một trong bốn đáp án: A, B, C hoặc D. Không giải thích thêm"
    )


def format_yn_prompt(question: str) -> str:
    """Mẫu câu lệnh (Prompt Template) chuẩn cho câu hỏi Yes/No theo Mục 4.1 bài báo VLSP 2025."""
    return (
        f"<image>\n"
        f"{question}\n\n"
        f"Trả lời bằng một trong hai đáp án: Đúng hoặc Sai. Không giải thích thêm."
    )


def build_vlsp_questions_for_sign(
    sign_code: str,
    dataset_code: str,
    sign_name: str,
    category: str,
    art_title: str,
    summary: str,
    idx_seed: int,
):
    """
    Tạo cặp câu hỏi chuẩn VLSP 2025 MLQA-TSR (Mục 3.1 & 4.1):
    1. Câu hỏi Multiple-Choice (4 lựa chọn A, B, C, D)
    2. Câu hỏi Yes/No ("Đúng" hoặc "Sai")
    3. Giải thích pháp lý đầy đủ phục vụ hội thoại đa lượt
    """
    base_c = sign_code.upper()
    c_up = dataset_code.upper()

    # Xác định phương án đúng và 3 phương án nhiễu hợp lý theo nhóm biển báo
    if "P.102" in base_c:
        mc_q = "Theo Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ, biển báo trong hình có ý nghĩa pháp lý là gì?"
        correct_text = "Cấm tất cả các loại xe (cơ giới và thô sơ) đi vào theo chiều đặt biển, trừ các xe được ưu tiên theo quy định."
        distractors = [
            "Cấm tất cả các loại phương tiện đi lại cả hai chiều trên đoạn đường đặt biển.",
            "Cấm các loại xe cơ giới dừng xe và đỗ xe ở phía đường có đặt biển.",
            "Chỉ cấm xe ô tô tải và xe ô tô khách đi vào, xe máy vẫn được phép lưu thông.",
        ]
        yn_true_q = "Khi gặp biển báo trong hình (Biển số P.102), người điều khiển xe ô tô con và xe máy bị cấm đi vào theo chiều đặt biển (trừ xe ưu tiên), đúng hay sai?"
        yn_false_q = "Khi gặp biển báo trong hình (Biển số P.102), người điều khiển xe mô tô hai bánh vẫn được phép đi vào bình thường, đúng hay sai?"
    elif "P.123A" in base_c or "P.103C" in base_c:
        mc_q = "Khi gặp biển báo trong hình trước nút giao thông, người điều khiển phương tiện phải chấp hành quy định nào sau đây?"
        correct_text = f"Cấm rẽ sang phía trái theo hiệu lực của biển {sign_code}, nhưng không có giá trị cấm quay đầu xe."
        distractors = [
            "Cấm đồng thời cả hành vi rẽ trái và hành vi quay đầu xe tại nút giao.",
            "Chỉ cấm quay đầu xe theo kiểu chữ U, người lái xe vẫn được phép rẽ trái.",
            "Bắt buộc tất cả các phương tiện phải rẽ trái, không được phép đi thẳng.",
        ]
        yn_true_q = f"Theo quy chuẩn về báo hiệu đường bộ, biển báo {sign_code} ({sign_name}) cấm phương tiện rẽ trái nhưng không có giá trị cấm quay đầu xe, đúng hay sai?"
        yn_false_q = f"Theo quy chuẩn về báo hiệu đường bộ, khi gặp biển báo {sign_code} ({sign_name}), phương tiện bị cấm cả rẽ trái lẫn quay đầu xe, đúng hay sai?"
    elif "P.123B" in base_c or "P.103B" in base_c:
        mc_q = "Phương tiện tham gia giao thông khi gặp biển báo trong hình tại ngã tư phía trước bị cấm thực hiện hành vi nào?"
        correct_text = f"Bị cấm rẽ phải tại nút giao theo hiệu lực của biển {sign_code} ({sign_name})."
        distractors = [
            "Bị cấm đi thẳng và bắt buộc phải rẽ phải vào đường nhánh.",
            "Bị cấm dừng xe và đỗ xe trong phạm vi 50 mét sau mặt biển.",
            "Bị cấm quay đầu xe nhưng vẫn được phép rẽ phải bình thường.",
        ]
        yn_true_q = f"Khi gặp biển báo {sign_code} ({sign_name}) trong hình, phương tiện vẫn được phép đi thẳng hoặc rẽ trái nếu không có biển cấm khác, đúng hay sai?"
        yn_false_q = f"Khi gặp biển báo {sign_code} ({sign_name}) trong hình, phương tiện được phép rẽ phải bình thường tại nút giao, đúng hay sai?"
    elif "P.124A" in base_c or "P.124B" in base_c:
        mc_q = "Biển báo cấm trong hình quy định như thế nào đối với hành vi chuyển hướng của phương tiện?"
        correct_text = f"Cấm quay đầu xe (theo kiểu chữ U) theo biển {sign_code} ({sign_name}), nhưng không có giá trị cấm rẽ trái để đi vào đường giao nhau."
        distractors = [
            "Cấm đồng thời cả hành vi quay đầu xe và hành vi rẽ trái tại nơi đường giao nhau.",
            "Cho phép các loại phương tiện quay đầu xe nhưng cấm rẽ phải.",
            "Cấm tất cả các loại xe cơ giới đi thẳng qua nút giao thông phía trước.",
        ]
        yn_true_q = f"Tại vị trí đặt biển báo {sign_code} ({sign_name}) trong hình, người điều khiển phương tiện bị cấm quay đầu xe nhưng vẫn được phép rẽ trái, đúng hay sai?"
        yn_false_q = f"Tại vị trí đặt biển báo {sign_code} ({sign_name}) trong hình, người điều khiển phương tiện bị cấm rẽ trái vào đường giao nhau, đúng hay sai?"
    elif "P.124C" in base_c or "P.124D" in base_c:
        direction = "rẽ trái và quay đầu xe" if "124C" in base_c else "rẽ phải và quay đầu xe"
        mc_q = "Khi gặp biển báo trong hình trước nút giao, những hành vi chuyển hướng nào bị nghiêm cấm?"
        correct_text = f"Cấm tất cả các loại xe thực hiện đồng thời cả hai hành vi: {direction} (Biển số {sign_code})."
        distractors = [
            "Chỉ cấm rẽ trái nhưng vẫn cho phép người điều khiển phương tiện quay đầu xe.",
            "Chỉ cấm xe ô tô tải trên 2,5 tấn, xe ô tô con được phép quay đầu bình thường.",
            "Bắt buộc phương tiện phải quay đầu xe trở lại, cấm đi thẳng qua nút giao.",
        ]
        yn_true_q = f"Biển báo {sign_code} trong hình có hiệu lực cấm các loại phương tiện đồng thời cả hai hành vi {direction}, đúng hay sai?"
        yn_false_q = f"Khi gặp biển báo {sign_code} trong hình, người điều khiển xe ô tô con vẫn được phép quay đầu xe tại nút giao, đúng hay sai?"
    elif "P.127" in base_c:
        spd = "50 km/h"
        for s_val in ["40", "60", "70", "80"]:
            if f"*{s_val}" in c_up:
                spd = f"{s_val} km/h"
        mc_q = "Biển báo trong ảnh quy định về giới hạn tốc độ lưu thông như thế nào đối với các phương tiện cơ giới?"
        correct_text = f"Cấm các loại xe cơ giới chạy với tốc độ tối đa vượt quá trị số ghi trên biển ({spd}), trừ các xe được ưu tiên theo quy định."
        distractors = [
            f"Bắt buộc các loại xe cơ giới phải chạy với tốc độ tối thiểu từ {spd} trở lên.",
            f"Chỉ dẫn khoảng cách an toàn tối thiểu giữa hai xe đang chạy trên đường là {spd.split()[0]} mét.",
            "Báo hiệu hết đoạn đường hạn chế tốc độ tối đa, cho phép xe chạy tốc độ bình thường.",
        ]
        yn_true_q = f"Khi gặp biển báo {sign_code} ({sign_name}) trong hình, người điều khiển xe cơ giới chạy vượt quá tốc độ ghi trên biển ({spd}) là vi phạm quy chuẩn giao thông, đúng hay sai?"
        yn_false_q = f"Biển báo {sign_code} trong hình quy định tốc độ tối thiểu bắt buộc các xe phải đạt được trên đường, đúng hay sai?"
    elif "P.130" in base_c:
        mc_q = "Trên đoạn đường có đặt biển báo trong hình, quy định về dừng xe và đỗ xe được áp dụng như thế nào?"
        correct_text = "Cấm các loại xe cơ giới dừng xe và đỗ xe ở phía đường có đặt biển (Biển số P.130), trừ các xe được ưu tiên theo quy định."
        distractors = [
            "Chỉ cấm đỗ xe lâu dài, tài xế vẫn được phép dừng xe tạm thời để đón trả khách hoặc xếp dỡ hàng hóa.",
            "Cho phép các phương tiện dừng xe và đỗ xe tự do trên vỉa hè và lòng đường.",
            "Chỉ cấm xe ô tô tải dừng đỗ, xe ô tô con cá nhân được phép đỗ xe trong 15 phút.",
        ]
        yn_true_q = "Tại đoạn đường đặt biển số P.130 (Cấm dừng xe và đỗ xe) như trong hình, mọi xe cơ giới (trừ xe ưu tiên) đều bị cấm cả dừng xe lẫn đỗ xe ở phía đặt biển, đúng hay sai?"
        yn_false_q = "Tại đoạn đường đặt biển số P.130 như trong hình, tài xế xe ô tô con được phép bật đèn cảnh báo để dừng xe chờ đón người trong 5 phút, đúng hay sai?"
    elif "P.131" in base_c:
        mc_q = "Khi gặp biển báo trong hình (Biển số P.131), người điều khiển xe cơ giới được phép và bị cấm thực hiện hành vi nào?"
        correct_text = "Bị cấm đỗ xe ở phía đường có đặt biển, nhưng vẫn được phép dừng xe tạm thời để đón trả khách hoặc xếp dỡ hàng hóa."
        distractors = [
            "Bị cấm tuyệt đối cả việc dừng xe tạm thời lẫn việc đỗ xe ở phía đường có đặt biển.",
            "Bị cấm đi vào đoạn đường phía trước đối với tất cả các loại phương tiện cơ giới.",
            "Được phép đỗ xe không giới hạn thời gian vào các ngày làm việc trong tuần.",
        ]
        yn_true_q = f"Gặp biển báo {sign_code} ({sign_name}) trong hình, tài xế bị cấm đỗ xe nhưng vẫn được phép dừng xe tạm thời để đón trả khách, đúng hay sai?"
        yn_false_q = f"Biển báo {sign_code} ({sign_name}) trong hình cấm tuyệt đối cả việc dừng xe tạm thời lẫn việc đỗ xe, đúng hay sai?"
    elif "P.106" in base_c or "P.107" in base_c or "P.108" in base_c:
        mc_q = "Những loại phương tiện nào chịu sự điều chỉnh của biển báo cấm trong hình và xe ô tô con có được phép đi vào không?"
        correct_text = f"Biển {sign_code} ({sign_name}) cấm các loại xe vận tải/xe khách/rơ-moóc theo quy định trên biển, nhưng xe ô tô con cá nhân vẫn được phép đi vào bình thường."
        distractors = [
            "Cấm tất cả các loại xe cơ giới bao gồm cả xe ô tô con 4–7 chỗ và xe mô tô hai bánh.",
            "Chỉ cấm xe đạp và người đi bộ, tất cả các loại xe ô tô tải nặng đều được phép đi vào.",
            "Báo hiệu đường dành riêng cho xe ô tô tải và xe kéo rơ-moóc lưu thông.",
        ]
        yn_true_q = f"Tại đoạn đường đặt biển số {sign_code} ({sign_name}) trong hình, người điều khiển xe ô tô con cá nhân vẫn được phép đi vào bình thường, đúng hay sai?"
        yn_false_q = f"Biển số {sign_code} ({sign_name}) trong hình có hiệu lực cấm cả xe ô tô con 5 chỗ và xe mô tô hai bánh đi vào, đúng hay sai?"
    elif base_c.startswith("W."):
        mc_q = "Biển báo hình tam giác nền vàng viền đỏ trong hình thuộc nhóm biển nào và người lái xe cần xử lý ra sao?"
        correct_text = f"Thuộc nhóm Biển báo nguy hiểm và cảnh báo — Biển số {sign_code} ({sign_name}); người lái xe phải chủ động giảm tốc độ, chú ý quan sát và đi đúng phần đường."
        distractors = [
            f"Thuộc nhóm Biển báo cấm — cấm tất cả các phương tiện cơ giới đi qua khu vực phía trước.",
            f"Thuộc nhóm Biển hiệu lệnh — bắt buộc các phương tiện phải tăng tốc độ tối thiểu lên 60 km/h.",
            f"Thuộc nhóm Biển phụ — chỉ có tác dụng bổ sung hiệu lực khi đặt kèm dưới biển báo cấm.",
        ]
        yn_true_q = f"Biển báo trong hình là Biển số {sign_code} ({sign_name}) thuộc nhóm Biển báo nguy hiểm và cảnh báo, yêu cầu người lái xe chú ý quan sát và giảm tốc độ an toàn, đúng hay sai?"
        yn_false_q = f"Biển báo tam giác nền vàng {sign_code} trong hình là biển báo cấm mọi phương tiện cơ giới đi tiếp qua đoạn đường này, đúng hay sai?"
    elif base_c.startswith("R."):
        mc_q = "Biển báo trong hình thuộc nhóm biển nào và có hiệu lực pháp lý như thế nào đối với người tham gia giao thông?"
        correct_text = f"Thuộc nhóm Biển hiệu lệnh — Biển số {sign_code} ({sign_name}), báo các hiệu lệnh bắt buộc người tham gia giao thông phải chấp hành."
        distractors = [
            "Thuộc nhóm Biển chỉ dẫn — chỉ mang tính chất khuyến nghị hướng đi, không bắt buộc thi hành.",
            "Thuộc nhóm Biển báo nguy hiểm — cảnh báo phía trước có công trường đang thi công.",
            "Thuộc nhóm Biển báo cấm — cấm mọi phương tiện cơ giới và thô sơ đi vào đường này.",
        ]
        yn_true_q = f"Biển số {sign_code} ({sign_name}) trong hình thuộc nhóm Biển hiệu lệnh và mang tính chất bắt buộc thi hành đối với người tham gia giao thông, đúng hay sai?"
        yn_false_q = f"Biển số {sign_code} ({sign_name}) trong hình chỉ mang tính chất tham khảo, người lái xe không bắt buộc phải chấp hành theo hướng chỉ định của biển, đúng hay sai?"
    elif base_c.startswith("S."):
        mc_q = "Biển hình chữ nhật trong ảnh thuộc nhóm biển nào và có vai trò pháp lý gì khi đặt kết hợp với biển chính?"
        correct_text = f"Thuộc nhóm Biển phụ — Biển số {sign_code} ({sign_name}), được đặt kết hợp dưới biển chính để thuyết minh, bổ sung hoặc giới hạn đối tượng/phạm vi hiệu lực của biển chính."
        distractors = [
            "Là biển báo cấm độc lập có hiệu lực cấm toàn bộ xe cơ giới trên cả hai chiều đường.",
            "Là biển báo nguy hiểm cảnh báo giao nhau với đường sắt có rào chắn.",
            "Là biển hiệu lệnh bắt buộc các phương tiện phải dừng lại trước vạch dừng.",
        ]
        yn_true_q = f"Biển số {sign_code} ({sign_name}) trong hình là biển phụ dùng để thuyết minh, bổ sung hoặc giới hạn đối tượng phương tiện áp dụng của biển báo chính đặt phía trên, đúng hay sai?"
        yn_false_q = f"Biển số {sign_code} trong hình là biển báo nguy hiểm độc lập cảnh báo đường trơn trượt, đúng hay sai?"
    else:
        mc_q = "Biển báo trong hình thuộc nhóm biển báo nào và cung cấp thông tin pháp lý gì theo Quy chuẩn báo hiệu đường bộ?"
        correct_text = f"Thuộc nhóm {category} — Biển số {sign_code} ({sign_name}): {summary[:180]}"
        distractors = [
            "Thuộc nhóm Biển báo cấm — cấm tất cả các phương tiện cơ giới rẽ trái và quay đầu xe.",
            "Thuộc nhóm Biển báo nguy hiểm — cảnh báo đoạn đường thường xảy ra tai nạn giao thông.",
            "Thuộc nhóm Biển hiệu lệnh — bắt buộc mọi phương tiện phải chạy tốc độ tối thiểu 80 km/h.",
        ]
        yn_true_q = f"Biển báo trong hình là Biển số {sign_code} ({sign_name}) thuộc nhóm {category} theo Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ, đúng hay sai?"
        yn_false_q = f"Biển báo {sign_code} ({sign_name}) trong hình thuộc nhóm Biển báo nguy hiểm và cấm mọi phương tiện lưu thông qua đoạn đường này, đúng hay sai?"

    # Xoay vòng vị trí đáp án đúng A / B / C / D theo idx_seed để phân bố đều 4 lựa chọn
    letters = ["A", "B", "C", "D"]
    correct_pos = idx_seed % 4
    correct_letter = letters[correct_pos]
    choices = {}
    d_idx = 0
    for pos, ltr in enumerate(letters):
        if pos == correct_pos:
            choices[ltr] = correct_text
        else:
            choices[ltr] = distractors[d_idx]
            d_idx += 1

    is_yn_true = (idx_seed % 2 == 0)
    yn_question = yn_true_q if is_yn_true else yn_false_q
    yn_answer = "Đúng" if is_yn_true else "Sai"

    full_explanation = (
        f"Biển báo trong hình là **Biển số {sign_code}: \"{sign_name}\"** thuộc **{category}**.\n\n"
        f"- **Căn cứ pháp lý:** {art_title}.\n"
        f"- **Quy định chi tiết:** {summary}\n"
        f"- **Đáp án trắc nghiệm đúng:** **{correct_letter}** ({correct_text})"
    )

    return {
        "mc_question": mc_q,
        "mc_choices": choices,
        "mc_answer": correct_letter,
        "mc_prompt": format_mc_prompt(mc_q, choices),
        "yn_question": yn_question,
        "yn_answer": yn_answer,
        "yn_prompt": format_yn_prompt(yn_question),
        "full_explanation": full_explanation,
    }


def compute_split_statistics(entries):
    """Tính bảng thống kê chuẩn Bảng 2 của bài báo VLSP 2025 MLQA-TSR (Mục 3.2)."""
    if not entries:
        return {}
    unique_imgs = set(x["image"] for x in entries)
    mc_count = sum(1 for x in entries if x["question_type"] == "multiple_choice")
    yn_count = sum(1 for x in entries if x["question_type"] == "yes_no")
    q_lengths = [x["question_token_length"] for x in entries]
    rel_counts = [len(x["relevant_articles"]) for x in entries]
    return {
        "total_questions": len(entries),
        "total_images": len(unique_imgs),
        "multiple_choice_questions": mc_count,
        "yes_no_questions": yn_count,
        "max_length_of_question": int(max(q_lengths)),
        "min_length_of_question": int(min(q_lengths)),
        "avg_length_of_question": round(float(sum(q_lengths)) / len(q_lengths), 2),
        "max_relevant_articles": int(max(rel_counts)),
        "min_relevant_articles": int(min(rel_counts)),
        "avg_relevant_articles": round(float(sum(rel_counts)) / len(rel_counts), 2),
    }


def build_benchmark():
    print("=" * 76)
    print(" XÂY DỰNG BỘ DỮ LIỆU VLSP 2025 MLQA-TSR (Train: real_crops -> Test: pdf_signs)")
    print("=" * 76)

    retriever = HybridLegalRetriever()

    with open(DATA_DIR / "legal_articles.json", "r", encoding="utf-8") as f:
        legal_articles = json.load(f)
    with open(DATA_DIR / "pdf_signs_catalog.json", "r", encoding="utf-8") as f:
        pdf_catalog = json.load(f)["dataset"]
    with open(DATA_DIR / "class_mapping_52.json", "r", encoding="utf-8") as f:
        class_mapping = json.load(f)
    with open(DATA_DIR / "scene_annotations.json", "r", encoding="utf-8") as f:
        scene_data = json.load(f)["scenes"]
    with open(DATA_DIR / "split_manifest.json", "r", encoding="utf-8") as f:
        split_manifest = json.load(f)

    img_to_splits = {
        r["image_id"]: (r["grouped_split"], r["random_split"], r["group_id"])
        for r in split_manifest["records"]
    }
    cls_map_by_id = {m["class_id"]: m for m in class_mapping}

    mqa_entries = []
    compat_labels = []

    # 1. TẬP ĐÁNH GIÁ / KIỂM THỬ (TEST SET): 51 ảnh chuẩn từ PDF QCVN 41 (data/pdf_signs)
    # Mỗi ảnh trong data/pdf_signs sinh ĐÚNG 1 câu hỏi Yes/No ("Đúng" hoặc "Sai")
    test_q_idx = 0
    for idx, item in enumerate(pdf_catalog):
        code = item["sign_code"]
        name = item["sign_name"]
        cat = item["category"]
        rel_arts = retriever.get_relevant_articles_for_sign(code)
        art_key = rel_arts[0]
        art_obj = legal_articles.get(art_key, {})
        art_title = art_obj.get("title", f"Quy định tại {art_key}")
        summary = art_obj.get("summary", item["description"])

        qa_pack = build_vlsp_questions_for_sign(code, code, name, cat, art_title, summary, idx)

        yn_tok_len = len(segment_vietnamese_tokens(qa_pack["yn_question"]))
        test_q_idx += 1
        mqa_entries.append({
            "id": f"vlsp_test_yn_{test_q_idx:04d}",
            "source_type": "pdf_canonical",
            "task_type": "subtask2_yes_no",
            "question_type": "yes_no",
            "image": item["image"],
            "compat_image": item["compat_image"],
            "sign_code": code,
            "dataset_code": code,
            "sign_name": name,
            "category": cat,
            "relevant_articles": rel_arts,
            "question": qa_pack["yn_question"],
            "choices": {"Đúng": "Đúng", "Sai": "Sai"},
            "correct_answer": qa_pack["yn_answer"],
            "vlsp_prompt": qa_pack["yn_prompt"],
            "question_token_length": yn_tok_len,
            "grouped_split": "test",
            "random_split": "test",
            "group_id": -1,
            "messages": [
                {"role": "user", "content": qa_pack["yn_prompt"]},
                {"role": "assistant", "content": f"**{qa_pack['yn_answer']}.**\n\n{qa_pack['full_explanation']}"},
            ],
        })

        compat_labels.append({
            "id": f"conv_{idx+1:03d}",
            "image": item["compat_image"],
            "sign_code": code,
            "sign_name": name,
            "relevant_articles": rel_arts,
            "messages": [
                {"role": "user", "content": qa_pack["yn_prompt"]},
                {"role": "assistant", "content": f"**{qa_pack['yn_answer']}.**\n\n{qa_pack['full_explanation']}"},
            ],
        })

    # 2. TẬP HUẤN LUYỆN (TRAIN SET): 1.807 ảnh cắt biển báo thực tế (data/real_crops)
    # Mỗi ảnh sinh ĐÚNG 1 câu hỏi Yes/No ("Đúng" hoặc "Sai")
    real_crop_qa_count = 0
    for sc in scene_data:
        img_id = sc["image_id"]
        orig_g_split, orig_r_split, gid = img_to_splits.get(img_id, ("train", "train", -1))
        for b in sc["boxes"]:
            if "crop_path" in b:
                cid = b["class_id"]
                cm = cls_map_by_id[cid]
                can_code = cm["canonical_qcvn_code"]
                ds_code = cm["dataset_code"]
                name_vi = cm["name_vi"]
                cat = cm["category"]
                rel_arts = retriever.get_relevant_articles_for_sign(can_code)
                art_key = rel_arts[0]
                art_obj = legal_articles.get(art_key, {})
                art_title = art_obj.get("title", f"Mục {art_key}")
                summary = art_obj.get("summary", cm["legal_summary"])

                real_crop_qa_count += 1
                qa_pack = build_vlsp_questions_for_sign(
                    can_code, ds_code, name_vi, cat, art_title, summary, real_crop_qa_count
                )

                tok_len = len(segment_vietnamese_tokens(qa_pack["yn_question"]))

                mqa_entries.append({
                    "id": f"vlsp_train_realcrop_{real_crop_qa_count:04d}",
                    "source_type": "real_street_crop",
                    "task_type": "subtask2_yes_no",
                    "question_type": "yes_no",
                    "scene_id": img_id,
                    "image": b["crop_path"],
                    "class_id": cid,
                    "sign_code": can_code,
                    "dataset_code": ds_code,
                    "sign_name": name_vi,
                    "category": cat,
                    "relevant_articles": rel_arts,
                    "question": qa_pack["yn_question"],
                    "choices": {"Đúng": "Đúng", "Sai": "Sai"},
                    "correct_answer": qa_pack["yn_answer"],
                    "vlsp_prompt": qa_pack["yn_prompt"],
                    "question_token_length": tok_len,
                    "grouped_split": "train",
                    "random_split": "train",
                    "orig_grouped_split": orig_g_split,
                    "orig_random_split": orig_r_split,
                    "group_id": gid,
                    "messages": [
                        {"role": "user", "content": qa_pack["yn_prompt"]},
                        {"role": "assistant", "content": f"**{qa_pack['yn_answer']}.**\n\n{qa_pack['full_explanation']}"},
                    ],
                })

    train_entries = [x for x in mqa_entries if x["grouped_split"] == "train"]
    test_entries = [x for x in mqa_entries if x["grouped_split"] == "test"]

    table2_stats = {
        "train_real_crops": compute_split_statistics(train_entries),
        "test_pdf_signs": compute_split_statistics(test_entries),
    }

    split_stats_grouped = Counter(x["grouped_split"] for x in mqa_entries)
    split_stats_source = Counter(x["source_type"] for x in mqa_entries)
    split_stats_qtype = Counter(x["question_type"] for x in mqa_entries)

    with open(DATA_DIR / "mqa_dataset.json", "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_name": "VLSP 2025 MLQA-TSR (Train: real_crops -> Test: pdf_signs)",
            "paper_reference": "VLSP 2025 MLQA-TSR Challenge: Vietnamese Multimodal Legal Question Answering on Traffic Sign Regulation",
            "total_qa_pairs": len(mqa_entries),
            "by_source_type": dict(split_stats_source),
            "by_grouped_split": dict(split_stats_grouped),
            "by_question_type": dict(split_stats_qtype),
            "vlsp_table2_statistics": table2_stats,
            "data": mqa_entries,
        }, f, ensure_ascii=False, indent=2)

    with open(BASE_DIR / "label.json", "w", encoding="utf-8") as f:
        json.dump(compat_labels, f, ensure_ascii=False, indent=2)
    with open(BASE_DIR / "label", "w", encoding="utf-8") as f:
        json.dump(compat_labels, f, ensure_ascii=False, indent=2)

    print(f"[✓] Đã xây dựng hoàn tất bộ dữ liệu theo chuẩn VLSP 2025 MLQA-TSR ({len(mqa_entries)} câu hỏi):")
    print(f"    - Tập Train (data/real_crops): {table2_stats['train_real_crops']}")
    print(f"    - Tập Test  (data/pdf_signs) : {table2_stats['test_pdf_signs']}")


if __name__ == "__main__":
    build_benchmark()
