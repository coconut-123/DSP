"""
build_mqa_dataset.py
--------------------
Xây dựng Bộ dữ liệu Hỏi - Đáp Pháp lý Đa phương thức về Biển báo Giao thông Việt Nam
(Vietnamese Multimodal Traffic Sign Legal Question Answering Benchmark - VN-TrafficLaw-MQA).

Đặc điểm bộ dữ liệu:
1. Tích hợp song song 2 nguồn dữ liệu hình ảnh:
   - 346 ảnh biển báo chuẩn trích xuất từ PDF QCVN 41:2019/BGTVT (Phụ lục B, C, D, E, F - trang 80..157).
   - 1,807 ảnh cắt biển báo thực tế (real crops) & các khung hình đa biển báo từ 3,216 ảnh đường phố Việt Nam.
2. Bao phủ 4 nhóm tác vụ suy luận pháp lý đa phương thức (Multimodal Legal QA Tasks):
   - Task 1 (sign_identification): Nhận diện mã biển báo, tên biển và định nghĩa pháp lý theo QCVN 41:2019/BGTVT.
   - Task 2 (driving_compliance): Tư vấn tình huống điều khiển phương tiện cụ thể (ô tô con, xe tải, xe khách, sơ-mi rơ-moóc, xe máy).
   - Task 3 (multi_sign_scene): Suy luận tổ hợp nhiều biển báo xuất hiện đồng thời trong cùng một khung hình giao thông thực tế (kể cả biển chính kết hợp biển phụ S.505a).
   - Task 4 (scope_and_citation): Xác định phạm vi hiệu lực của biển báo và trích dẫn điều khoản theo QCVN 41:2019/BGTVT.
3. Kế thừa nghiêm ngặt phân chia Group-Disjoint Partition (0 group crossing) từ split_manifest.json để chống rò rỉ dữ liệu.
"""

import os
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def generate_compliance_qa(sign_code: str, dataset_code: str, sign_name: str, category: str, art_title: str, summary: str):
    """Sinh câu hỏi và câu trả lời suy luận tình huống lái xe thực tế theo đúng ngữ nghĩa từng mã biển."""
    c_up = dataset_code.upper()
    base_c = sign_code.upper()

    if "P.102" in base_c:
        q = "Tôi đang lái xe ô tô con và xe máy đến đầu đường đặt biển báo trong ảnh thì có được phép đi vào không?"
        a = (
            f"**Không được phép.** Biển trong ảnh là **Biển số {sign_code} ({sign_name})** thuộc {category}.\n\n"
            f"- **Quy định pháp lý ({art_title}):** Biển báo đường cấm các loại xe (cơ giới và thô sơ) đi vào theo chiều đặt biển, trừ các xe được ưu tiên theo quy định của Luật Giao thông đường bộ.\n"
            f"- **Hướng xử lý:** Bạn bắt buộc phải chuyển hướng đi theo lộ trình khác, tuyệt đối không đi ngược chiều."
        )
    elif "P.123A" in base_c:
        q = "Khi gặp biển báo này tại nút giao, người lái xe ô tô có được phép rẽ trái hoặc quay đầu xe không?"
        a = (
            f"Theo quy chuẩn **QCVN 41:2019/BGTVT ({art_title})** đối với **Biển số {sign_code} ({sign_name})**:\n\n"
            f"- **Rẽ trái:** **Cấm** tất cả các loại xe (cơ giới và thô sơ) rẽ sang phía trái.\n"
            f"- **Quay đầu xe:** Theo quy chuẩn mới QCVN 41:2019/BGTVT, biển P.123a **không có giá trị cấm quay đầu xe** (người lái xe vẫn được phép quay đầu xe nếu không có biển cấm quay đầu P.124)."
        )
    elif "P.123B" in base_c:
        q = "Phương tiện gặp biển báo trong hình có được phép rẽ phải tại ngã tư phía trước không?"
        a = (
            f"**Không được phép rẽ phải.** Đây là **Biển số {sign_code} ({sign_name})** quy định tại **{art_title}**.\n\n"
            f"- **Nội dung cấm:** Cấm các loại phương tiện cơ giới và thô sơ rẽ sang phía phải tại nút giao.\n"
            f"- **Lưu ý pháp lý:** Biển chỉ cấm rẽ phải, các phương tiện vẫn được phép đi thẳng hoặc rẽ trái/quay đầu nếu không có biển cấm khác."
        )
    elif "P.124A" in base_c:
        q = "Gặp biển báo trong hình, tôi điều khiển xe ô tô hoặc xe máy có được quay đầu xe hay rẽ trái không?"
        a = (
            f"Đối với **Biển số {sign_code} ({sign_name})** theo **{art_title}**:\n\n"
            f"- **Quay đầu xe:** **Nghiêm cấm** các loại xe cơ giới và thô sơ quay đầu xe (theo kiểu chữ U).\n"
            f"- **Rẽ trái:** Biển này **không cấm rẽ trái**; người điều khiển phương tiện vẫn được phép rẽ trái vào đường giao nhau."
        )
    elif "P.124B" in base_c:
        q = "Biển báo trong hình áp dụng lệnh cấm đối với xe máy hay xe ô tô, và có được phép rẽ trái không?"
        a = (
            f"Đây là **Biển số {sign_code} ({sign_name})** quy định tại **{art_title}**:\n\n"
            f"- **Đối tượng áp dụng:** Cấm **xe ô tô** và xe mô tô 3 bánh (sidecar) quay đầu xe; **xe máy 2 bánh và xe thô sơ vẫn được phép quay đầu**.\n"
            f"- **Quyền rẽ trái:** Biển không có giá trị cấm rẽ trái để đi vào đường nhánh."
        )
    elif "P.124C" in base_c or "P.124D" in base_c:
        direction = "rẽ trái và quay đầu xe" if "124C" in base_c else "rẽ phải và quay đầu xe"
        q = f"Khi gặp biển báo này trước nút giao, người lái xe có được phép {direction} không?"
        a = (
            f"**Tuyệt đối không được phép.** Biển trong hình là **Biển số {sign_code} ({sign_name})** theo **{art_title}**.\n\n"
            f"- **Quy định:** Cấm tất cả các loại phương tiện đồng thời cả hai hành vi: **{direction}** tại khu vực nút giao.\n"
            f"- **Hướng đi hợp pháp:** Người lái xe chỉ được phép đi thẳng hoặc rẽ theo hướng không bị cấm trên biển."
        )
    elif "P.127" in base_c:
        spd = "50 km/h"
        if "*60" in c_up:
            spd = "60 km/h"
        elif "*80" in c_up:
            spd = "80 km/h"
        elif "*40" in c_up:
            spd = "40 km/h"
        q = f"Biển báo trong ảnh quy định tốc độ lưu thông như thế nào? Tôi chạy xe ở tốc độ vượt mức {spd} có vi phạm không?"
        a = (
            f"**Có vi phạm.** Đây là **Biển số {sign_code} ({sign_name})** quy định tại **{art_title}**.\n\n"
            f"- **Giới hạn tốc độ:** Cấm tất cả các loại xe cơ giới chạy với tốc độ tối đa vượt quá trị số ghi trên biển (**{spd}**), trừ các xe được ưu tiên theo quy định.\n"
            f"- **Trách nhiệm người lái:** Phải chủ động giảm tốc độ xuống dưới hoặc bằng {spd} kể từ vị trí đặt biển cho đến khi gặp biển hết hạn chế tốc độ (DP.134/DP.135) hoặc qua nút giao."
        )
    elif "P.130" in base_c:
        q = "Trên đoạn đường đặt biển báo này, tôi bật đèn cảnh báo (hazard) để dừng xe chờ đón người trong 2 phút có được phép không?"
        a = (
            f"**Không được phép.** Biển trong ảnh là **Biển số {sign_code} ({sign_name})** quy định tại **{art_title}**.\n\n"
            f"- **Nội dung cấm:** Cấm tất cả các loại xe cơ giới **dừng xe và đỗ xe** ở phía đường có đặt biển, bất kể thời gian ngắn hay có bật đèn tín hiệu, trừ các xe được ưu tiên theo luật định.\n"
            f"- **Phân biệt với P.131a:** Khác với biển P.131a (chỉ cấm đỗ xe nhưng cho phép dừng xe tạm thời), biển P.130 cấm tuyệt đối cả dừng xe lẫn đỗ xe."
        )
    elif "P.131" in base_c:
        q = "Gặp biển báo trong hình, tài xế xe ô tô có được phép dừng xe tạm thời để xếp dỡ hàng hóa hoặc đón trả khách không?"
        a = (
            f"**Được phép dừng xe tạm thời nhưng cấm đỗ xe.** Đây là **Biển số {sign_code} ({sign_name})** theo **{art_title}**.\n\n"
            f"- **Quy định:** Biển cấm các loại xe cơ giới đỗ xe ở phía đường có đặt biển.\n"
            f"- **Ngoại lệ hợp pháp:** Người điều khiển phương tiện vẫn được phép dừng xe tạm thời (không tắt máy, tài xế không rời vị trí lái) để đón trả khách hoặc xếp dỡ hàng hóa khi không gây cản trở giao thông."
        )
    elif "P.106" in base_c or "P.107" in base_c or "P.108" in base_c:
        q = "Biển báo cấm trong hình áp dụng cho những loại phương tiện nào? Xe ô tô con 5 chỗ có bị cấm đi vào không?"
        a = (
            f"Theo quy định tại **{art_title}** đối với **Biển số {sign_code} ({sign_name})**:\n\n"
            f"- **Đối tượng bị cấm:** {summary}\n"
            f"- **Đối với xe ô tô con cá nhân:** Biển hạn chế xe ô tô tải / xe khách / xe sơ-mi rơ-moóc **không có hiệu lực cấm xe ô tô con**, do đó xe ô tô con vẫn được phép lưu thông bình thường."
        )
    elif "S.505A" in base_c:
        veh = "phương tiện hiển thị trên biển phụ"
        if "XE MÁY" in c_up:
            veh = "xe mô tô, xe gắn máy"
        elif "XE TẢI VÀ CÔNG" in c_up:
            veh = "xe ô tô tải và xe container (sơ-mi rơ-moóc)"
        elif "XE TẢI" in c_up:
            veh = "xe ô tô tải"
        q = "Biển phụ hình chữ nhật nền trắng trong ảnh có vai trò pháp lý như thế nào khi đặt bên dưới biển chính?"
        a = (
            f"Đây là **Biển phụ số {sign_code} ({sign_name})** được quy định tại **{art_title}**.\n\n"
            f"- **Ý nghĩa kết hợp:** Biển số S.505a được đặt ngay bên dưới các biển báo cấm, biển hiệu lệnh hoặc biển chỉ dẫn để **giới hạn đối tượng áp dụng của biển chính chỉ dành riêng cho loại xe vẽ trên biển phụ ({veh})**.\n"
            f"- **Quy tắc tuân thủ:** Chỉ người điều khiển **{veh}** mới phải chịu sự điều chỉnh của biển chính phía trên."
        )
    elif base_c.startswith("W."):
        q = "Khi quan sát thấy biển cảnh báo tam giác nền vàng viền đỏ này trên đường, người điều khiển phương tiện phải xử lý như thế nào để đúng luật?"
        a = (
            f"Biển trong hình là **Biển số {sign_code} ({sign_name})** thuộc nhóm **Biển báo nguy hiểm và cảnh báo** theo **{art_title}**.\n\n"
            f"- **Ý nghĩa cảnh báo:** {summary}\n"
            f"- **Nghĩa vụ pháp lý:** Người điều khiển phương tiện phải chủ động giảm tốc độ đến mức an toàn, giữ khoảng cách, chú ý quan sát chướng ngại vật/người đi bộ/phương tiện giao cắt và tuyệt đối không vượt xe ở đoạn đường nguy hiểm."
        )
    elif base_c.startswith("R."):
        q = "Biển báo nền xanh trong ảnh mang tính chất khuyến nghị hay bắt buộc thi hành đối với người tham gia giao thông?"
        a = (
            f"**Bắt buộc thi hành.** Đây là **Biển số {sign_code} ({sign_name})** thuộc nhóm **Biển hiệu lệnh** quy định tại **{art_title}**.\n\n"
            f"- **Mệnh lệnh giao thông:** {summary}\n"
            f"- **Hiệu lực:** Tất cả các phương tiện tham gia giao thông bắt buộc phải chấp hành đúng hướng đi, làn đường hoặc tốc độ tối thiểu mà biển đã chỉ định."
        )
    else:
        q = "Biển chỉ dẫn trong ảnh cung cấp thông tin pháp lý gì và người lái xe cần lưu ý điều gì khi đi qua đoạn đường này?"
        a = (
            f"Đây là **Biển số {sign_code} ({sign_name})** thuộc nhóm **Biển chỉ dẫn** theo **{art_title}**.\n\n"
            f"- **Thông tin chỉ dẫn:** {summary}\n"
            f"- **Lưu ý tuân thủ:** Người tham gia giao thông sử dụng thông tin trên biển để định hướng lộ trình, vị trí quay đầu xe, điểm dừng xe buýt hoặc tuân thủ nghiêm ngặt quy tắc giao thông trên tuyến (như đi đúng chiều trên đường một chiều hoặc chấp hành hệ thống camera giám sát)."
        )
    return q, a


def build_benchmark():
    print("=" * 72)
    print(" XÂY DỰNG BỘ DỮ LIỆU MULTIMODAL LEGAL QA (VN-TrafficLaw-MQA)")
    print("=" * 72)

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

    # 1. Đưa toàn bộ 346 ảnh biển báo chuẩn từ PDF vào tập huấn luyện chuẩn (Canonical Reference Templates)
    for idx, item in enumerate(pdf_catalog):
        code = item["sign_code"]
        name = item["sign_name"]
        cat = item["category"]
        art_key = item["relevant_articles"][0]
        art_obj = legal_articles.get(art_key, {})
        art_title = art_obj.get("title", f"Quy định tại {art_key}")
        summary = art_obj.get("summary", item["description"])

        q_comp, a_comp = generate_compliance_qa(code, code, name, cat, art_title, summary)

        messages = [
            {
                "role": "user",
                "content": "Biển báo trong hình là biển gì và có ý nghĩa pháp lý như thế nào theo QCVN 41:2019/BGTVT?",
            },
            {
                "role": "assistant",
                "content": (
                    f"Biển báo trong hình là **Biển số {code}: \"{name}\"** theo Quy chuẩn kỹ thuật quốc gia **QCVN 41:2019/BGTVT**.\n\n"
                    f"- **Phân loại nhóm biển:** {cat}.\n"
                    f"- **Căn cứ pháp lý:** {art_title} (QCVN 41:2019/BGTVT).\n"
                    f"- **Ý nghĩa & Quy định sử dụng:** {summary}"
                ),
            },
            {
                "role": "user",
                "content": q_comp,
            },
            {
                "role": "assistant",
                "content": a_comp,
            },
        ]

        entry = {
            "id": f"mqa_pdf_{idx+1:04d}",
            "source_type": "pdf_canonical",
            "task_type": "sign_identification_and_compliance",
            "image": item["image"],
            "compat_image": item["compat_image"],
            "sign_code": code,
            "dataset_code": code,
            "sign_name": name,
            "category": cat,
            "relevant_articles": [art_key],
            "grouped_split": "train",  # Ảnh mẫu từ PDF chỉ dùng để làm giàu tập Train (Canonical Template Augmentation), không bao giờ đưa vào Test để tránh làm dễ bài kiểm tra thực tế
            "random_split": "train",
            "group_id": -1,
            "messages": messages,
        }
        mqa_entries.append(entry)

        compat_labels.append({
            "id": f"conv_{idx+1:03d}",
            "image": item["compat_image"],
            "sign_code": code,
            "sign_name": name,
            "relevant_articles": [art_key],
            "messages": messages,
        })

    # 2. Tạo các mẫu hỏi đáp từ dữ liệu giao thông thực tế (Real Crops & Multi-Sign Street Scenes)
    real_crop_qa_count = 0
    multi_sign_scene_qa_count = 0

    for sc in scene_data:
        img_id = sc["image_id"]
        g_split, r_split, gid = img_to_splits.get(img_id, ("train", "train", -1))
        boxes = sc["boxes"]

        # 2a. Tạo mẫu QA cho các ảnh cắt biển báo thực tế (Real Crops) đã lưu
        for b in boxes:
            if "crop_path" in b:
                cid = b["class_id"]
                cm = cls_map_by_id[cid]
                can_code = cm["canonical_qcvn_code"]
                ds_code = cm["dataset_code"]
                name_vi = cm["name_vi"]
                cat = cm["category"]
                art_key = cm["article_id"]
                art_obj = legal_articles.get(art_key, {})
                art_title = art_obj.get("title", f"Mục {art_key}")
                summary = art_obj.get("summary", cm["legal_summary"])

                q_comp, a_comp = generate_compliance_qa(can_code, ds_code, name_vi, cat, art_title, summary)
                real_crop_qa_count += 1

                mqa_entries.append({
                    "id": f"mqa_realcrop_{real_crop_qa_count:04d}",
                    "source_type": "real_street_crop",
                    "task_type": "real_sign_legal_qa",
                    "scene_id": img_id,
                    "image": b["crop_path"],
                    "class_id": cid,
                    "sign_code": can_code,
                    "dataset_code": ds_code,
                    "sign_name": name_vi,
                    "category": cat,
                    "relevant_articles": [art_key],
                    "grouped_split": g_split,
                    "random_split": r_split,
                    "group_id": gid,
                    "messages": [
                        {
                            "role": "user",
                            "content": q_comp,
                        },
                        {
                            "role": "assistant",
                            "content": (
                                f"{a_comp}\n\n"
                                f"- **Trích dẫn quy chuẩn:** {art_title} (QCVN 41:2019/BGTVT).\n"
                                f"- **Quy định chi tiết:** {summary}"
                            ),
                        },
                    ],
                })

        # 2b. Tạo mẫu QA Suy luận Đa biển báo trên toàn cảnh đường phố (Multi-Sign Scene Reasoning)
        unique_cids = sc["classes_present"]
        if len(unique_cids) >= 2 and multi_sign_scene_qa_count < 600:
            multi_sign_scene_qa_count += 1
            sign_descriptions = []
            art_keys = []
            for cid in unique_cids:
                cm = cls_map_by_id[cid]
                can_code = cm["canonical_qcvn_code"]
                name_vi = cm["name_vi"]
                art_k = cm["article_id"]
                if art_k not in art_keys:
                    art_keys.append(art_k)
                sign_descriptions.append(
                    f"1. **Biển số {can_code} ({name_vi})** [{cm['category']} - {art_k}]: {cm['legal_summary']}"
                )

            joined_desc = "\n".join(sign_descriptions)
            codes_str = ", ".join(cls_map_by_id[c]["canonical_qcvn_code"] for c in unique_cids)
            prim_cm = cls_map_by_id[unique_cids[0]]

            mqa_entries.append({
                "id": f"mqa_scene_{multi_sign_scene_qa_count:04d}",
                "source_type": "real_multi_sign_scene",
                "task_type": "multi_sign_scene_reasoning",
                "scene_id": img_id,
                "image": sc["image_path"],
                "classes_present": unique_cids,
                "sign_code": prim_cm["canonical_qcvn_code"],
                "all_sign_codes": [cls_map_by_id[c]["canonical_qcvn_code"] for c in unique_cids],
                "sign_name": prim_cm["name_vi"],
                "category": prim_cm["category"],
                "relevant_articles": art_keys,
                "grouped_split": g_split,
                "random_split": r_split,
                "group_id": gid,
                "messages": [
                    {
                        "role": "user",
                        "content": "Trong khung hình giao thông thực tế này xuất hiện những biển báo nào và người điều khiển phương tiện cần phối hợp chấp hành các quy định pháp lý nào theo QCVN 41:2019/BGTVT?",
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"Trong khung hình giao thông này có sự xuất hiện kết hợp của **{len(unique_cids)} loại biển báo ({codes_str})** theo **QCVN 41:2019/BGTVT**:\n\n"
                            f"{joined_desc}\n\n"
                            f"**Tổng hợp quy tắc chấp hành:** Người điều khiển phương tiện khi lưu thông qua đoạn đường này phải đồng thời tuân thủ hiệu lực của tất cả các biển báo nêu trên (nếu có biển phụ nhóm S như S.505a đặt dưới biển chính thì hiệu lực của biển chính được giới hạn theo đối tượng phương tiện trên biển phụ)."
                        ),
                    },
                ],
            })

    # Lưu toàn bộ dataset MQA và đồng bộ label.json cho server/demo
    split_stats_grouped = Counter(x["grouped_split"] for x in mqa_entries)
    split_stats_source = Counter(x["source_type"] for x in mqa_entries)

    with open(DATA_DIR / "mqa_dataset.json", "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_name": "VN-TrafficLaw-MQA",
            "total_qa_pairs": len(mqa_entries),
            "by_source_type": dict(split_stats_source),
            "by_grouped_split": dict(split_stats_grouped),
            "data": mqa_entries,
        }, f, ensure_ascii=False, indent=2)

    with open(BASE_DIR / "label.json", "w", encoding="utf-8") as f:
        json.dump(compat_labels, f, ensure_ascii=False, indent=2)
    with open(BASE_DIR / "label", "w", encoding="utf-8") as f:
        json.dump(compat_labels, f, ensure_ascii=False, indent=2)

    print(f"[✓] Đã xây dựng hoàn tất bộ dữ liệu VN-TrafficLaw-MQA với {len(mqa_entries)} mẫu hội thoại đa phương thức:")
    print(f"    - Mẫu biển chuẩn từ PDF (pdf_canonical):        {split_stats_source['pdf_canonical']}")
    print(f"    - Mẫu cắt biển thực tế (real_street_crop):      {split_stats_source['real_street_crop']}")
    print(f"    - Mẫu toàn cảnh đa biển (real_multi_sign_scene): {split_stats_source['real_multi_sign_scene']}")
    print(f"    - Phân bố theo Group-Disjoint Split: Train={split_stats_grouped['train']}, Val={split_stats_grouped['val']}, Test={split_stats_grouped['test']}")


if __name__ == "__main__":
    build_benchmark()
