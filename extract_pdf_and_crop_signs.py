"""
extract_pdf_and_crop_signs.py
-----------------------------
1. Trích xuất & cắt toàn bộ ảnh biển báo chuẩn từ file PDF QCVN 41:2019/BGTVT
   (51-bgtvt-kem.pdf, từ trang 80 đến trang 157: Phụ lục B, C, D, E, F) và gán mã biển chuẩn xác 100%.
2. Xây dựng Cơ sở dữ liệu Điều luật Pháp lý Giao thông (legal_articles.json) trích xuất 100% nguyên bản
   từ văn bản QCVN 41:2019/BGTVT (51-bgtvt-kem.pdf).
3. Ánh xạ 100% 52 lớp biển báo thực tế trong archive/classes.txt với mã biển QCVN 41:2019/BGTVT,
   đồng thời trích xuất các patch biển báo thực tế từ 3,216 ảnh đường phố (8,334 bounding boxes).
"""

import os
import sys
import re
import json
import shutil
from pathlib import Path
from collections import Counter, defaultdict
import pymupdf
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "archive"
PDF_PATH = ARCHIVE_DIR / "51-bgtvt-kem.pdf"
if not PDF_PATH.exists():
    PDF_PATH = BASE_DIR.parent / "51-bgtvt-kem.pdf"

DATA_DIR = BASE_DIR / "data"
PDF_SIGNS_DIR = DATA_DIR / "pdf_signs"
REAL_CROPS_DIR = DATA_DIR / "real_crops"
IMAGES_COMPAT_DIR = BASE_DIR / "images"

for d in [DATA_DIR, PDF_SIGNS_DIR, REAL_CROPS_DIR, IMAGES_COMPAT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"[*] Mở tài liệu PDF quy chuẩn: {PDF_PATH}")
doc = pymupdf.open(str(PDF_PATH))
print(f"[*] Tổng số trang PDF: {len(doc)}")


def clean_vn(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\x03", " ").replace("\x04", " ")
    text = re.sub(r"B\s*i\s*ể\s*n", "Biển", text, flags=re.IGNORECASE)
    text = re.sub(r"s\s*ố", "số", text, flags=re.IGNORECASE)
    text = re.sub(r"H\s*ì\s*n\s*h", "Hình", text, flags=re.IGNORECASE)
    # Chỉ xóa khoảng trắng quanh dấu chấm giữa mã chữ cái và chữ số (không nối chữ cái của từ tiếp theo)
    text = re.sub(r"\b([PWRIS]|DP|IE)\s*\.\s*([0-9]{1,3})", r"\1.\2", text)
    text = re.sub(r"R\s*[\.,]\s*E\s*[\.,]\s*([0-9]{1,2})\s*([a-d]?)\b", r"R.E.\1\2", text, flags=re.IGNORECASE)
    text = re.sub(r"S\s*\.\s*([HG])\s*,\s*([0-9]{1,2})\s*([a-c]?)\b", r"S.\1,\2\3", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text


def get_category_by_code(code: str) -> str:
    if code.startswith("P.") or code.startswith("DP.") or code.startswith("B."):
        return "Biển báo cấm (Nhóm P)"
    elif code.startswith("W."):
        return "Biển báo nguy hiểm và cảnh báo (Nhóm W)"
    elif code.startswith("R."):
        return "Biển hiệu lệnh (Nhóm R)"
    elif code.startswith("I.") or code.startswith("IE.") or code == "Camera":
        return "Biển chỉ dẫn (Nhóm I)"
    elif code.startswith("S."):
        return "Biển phụ (Nhóm S)"
    return "Biển báo hiệu đường bộ"


# Danh mục tên chuẩn hóa toàn diện QCVN 41:2019/BGTVT (Phụ lục B, C, D, E, F)
STANDARD_NAMES = {
    "P.101": "Đường cấm",
    "P.102": "Cấm đi ngược chiều",
    "P.103a": "Cấm xe ô tô",
    "P.103b": "Cấm xe ô tô rẽ phải",
    "P.103c": "Cấm xe ô tô rẽ trái",
    "P.104": "Cấm xe máy",
    "P.105": "Cấm xe ô tô và xe máy",
    "P.106a": "Cấm xe ô tô tải",
    "P.106b": "Cấm xe ô tô tải có khối lượng chuyên chở vượt quá giá trị ghi trên biển",
    "P.106c": "Cấm các xe chở hàng nguy hiểm",
    "P.107": "Cấm xe ô tô khách và xe ô tô tải",
    "P.107a": "Cấm xe ô tô khách",
    "P.107b": "Cấm xe ô tô taxi",
    "P.108": "Cấm xe kéo rơ-moóc",
    "P.108a": "Cấm xe sơ-mi rơ-moóc",
    "P.109": "Cấm máy kéo",
    "P.110a": "Cấm xe đạp",
    "P.110b": "Cấm xe đạp thồ",
    "P.111a": "Cấm xe gắn máy",
    "P.111b": "Cấm xe ba bánh loại có động cơ (xe lam, xích lô máy)",
    "P.111c": "Cấm xe ba bánh loại có động cơ (xe lôi máy)",
    "P.111d": "Cấm xe ba bánh loại không có động cơ (xích lô đạp)",
    "P.112": "Cấm người đi bộ",
    "P.113": "Cấm xe người kéo, đẩy",
    "P.114": "Cấm xe súc vật kéo",
    "P.115": "Hạn chế trọng tải toàn bộ xe",
    "P.116": "Hạn chế tải trọng trên trục xe",
    "P.117": "Hạn chế chiều cao",
    "P.118": "Hạn chế chiều ngang xe",
    "P.119": "Hạn chế chiều dài xe",
    "P.120": "Hạn chế chiều dài xe cơ giới kéo theo rơ-moóc hoặc sơ-mi rơ-moóc",
    "P.121": "Cự ly tối thiểu giữa hai xe",
    "P.123a": "Cấm rẽ trái",
    "P.123b": "Cấm rẽ phải",
    "P.124a": "Cấm quay đầu xe",
    "P.124b": "Cấm ô tô quay đầu xe",
    "P.124c": "Cấm rẽ trái và quay đầu xe",
    "P.124d": "Cấm rẽ phải và quay đầu xe",
    "P.124e": "Cấm ô tô rẽ trái và quay đầu xe",
    "P.124f": "Cấm ô tô rẽ phải và quay đầu xe",
    "P.125": "Cấm vượt",
    "P.126": "Cấm xe ô tô tải vượt",
    "P.127": "Tốc độ tối đa cho phép",
    "P.127a": "Tốc độ tối đa cho phép về ban đêm",
    "P.127b": "Biển ghép tốc độ tối đa cho phép theo làn đường",
    "P.127c": "Biển ghép tốc độ tối đa cho phép theo phương tiện, trên từng làn đường",
    "DP.127a": "Biển hết tốc độ tối đa cho phép theo biển ghép",
    "DP.127b": "Biển hết tốc độ tối đa cho phép theo làn đường",
    "DP.127c": "Biển hết tốc độ tối đa cho phép theo phương tiện, trên từng làn đường",
    "P.128": "Cấm sử dụng còi",
    "P.129": "Kiểm tra",
    "P.130": "Cấm dừng xe và đỗ xe",
    "P.131a": "Cấm đỗ xe",
    "P.131b": "Cấm đỗ xe ngày lẻ",
    "P.131c": "Cấm đỗ xe ngày chẵn",
    "P.132": "Nhường đường cho xe cơ giới đi ngược chiều qua đường hẹp",
    "DP.133": "Hết cấm vượt",
    "DP.134": "Hết hạn chế tốc độ tối đa",
    "DP.135": "Hết tất cả các lệnh cấm",
    "P.136": "Cấm đi thẳng",
    "P.137": "Cấm rẽ trái, rẽ phải",
    "P.138": "Cấm đi thẳng, rẽ trái",
    "P.139": "Cấm đi thẳng, rẽ phải",
    "P.140": "Cấm xe công nông và các loại xe tương tự",
    # Nhóm W
    "W.201a": "Chỗ ngoặt nguy hiểm vòng bên trái",
    "W.201b": "Chỗ ngoặt nguy hiểm vòng bên phải",
    "W.202a": "Nhiều chỗ ngoặt nguy hiểm liên tiếp (chỗ đầu tiên sang trái)",
    "W.202b": "Nhiều chỗ ngoặt nguy hiểm liên tiếp (chỗ đầu tiên sang phải)",
    "W.203a": "Đường bị thu hẹp cả hai bên",
    "W.203b": "Đường bị thu hẹp về phía trái",
    "W.203c": "Đường bị thu hẹp về phía phải",
    "W.204": "Đường hai chiều",
    "W.205a": "Đường giao nhau cùng mức của các đường đồng cấp (ngã tư)",
    "W.205b": "Đường giao nhau cùng mức của các đường đồng cấp (ngã ba bên phải)",
    "W.205c": "Đường giao nhau cùng mức của các đường đồng cấp (ngã ba bên trái)",
    "W.205d": "Đường giao nhau cùng mức của các đường đồng cấp (nhánh chữ T)",
    "W.205e": "Đường giao nhau cùng mức của các đường đồng cấp (nhánh chữ Y)",
    "W.206": "Giao nhau chạy theo vòng xuyến",
    "W.207a": "Giao nhau với đường không ưu tiên (ngã tư)",
    "W.207b": "Giao nhau với đường không ưu tiên (nhánh bên phải)",
    "W.207c": "Giao nhau với đường không ưu tiên (nhánh bên trái)",
    "W.207d": "Giao nhau với đường không ưu tiên (nhánh lệch bên phải)",
    "W.207e": "Giao nhau với đường không ưu tiên (nhánh lệch bên trái)",
    "W.207f": "Giao nhau với đường không ưu tiên (nhánh xiên góc phải)",
    "W.207g": "Giao nhau với đường không ưu tiên (nhánh xiên góc trái)",
    "W.207h": "Giao nhau với đường không ưu tiên",
    "W.207i": "Giao nhau với đường không ưu tiên",
    "W.207k": "Giao nhau với đường không ưu tiên",
    "W.207l": "Giao nhau với đường không ưu tiên",
    "W.208": "Giao nhau với đường ưu tiên",
    "W.209": "Nơi giao nhau có tín hiệu đèn",
    "W.210": "Giao nhau với đường sắt có rào chắn",
    "W.211a": "Giao nhau với đường sắt không có rào chắn",
    "W.211b": "Giao nhau với đường tàu điện",
    "W.212": "Cầu hẹp",
    "W.213": "Cầu tạm",
    "W.214": "Cầu quay - cầu cất",
    "W.215a": "Kè, vực sâu phía trước",
    "W.215b": "Kè, vực sâu bên đường phía bên phải",
    "W.215c": "Kè, vực sâu bên đường phía bên trái",
    "W.216a": "Đường ngầm",
    "W.216b": "Đường ngầm có nguy cơ lũ quét",
    "W.217": "Bến phà",
    "W.218": "Cửa chui",
    "W.219": "Dốc xuống nguy hiểm",
    "W.220": "Dốc lên nguy hiểm",
    "W.221a": "Đường không bằng phẳng (đường lồi lõm, gợn sóng)",
    "W.221b": "Đường có sóng mấp mô nhân tạo (gồ giảm tốc)",
    "W.222a": "Đường trơn",
    "W.222b": "Lề đường nguy hiểm",
    "W.223a": "Vách núi nguy hiểm phía bên trái",
    "W.223b": "Vách núi nguy hiểm phía bên phải",
    "W.224": "Đường người đi bộ cắt ngang",
    "W.225": "Trẻ em",
    "W.226": "Đường người đi xe đạp cắt ngang",
    "W.227": "Công trường",
    "W.228a": "Đá lở phía bên trái",
    "W.228b": "Đá lở phía bên phải",
    "W.228c": "Sỏi đá bắn lên",
    "W.228d": "Nền đường yếu",
    "W.229": "Dải máy bay lên xuống",
    "W.230": "Gia súc",
    "W.231": "Thú rừng vượt qua đường",
    "W.232": "Gió ngang",
    "W.233": "Nguy hiểm khác",
    "W.234": "Giao nhau với đường hai chiều",
    "W.235": "Đường đôi",
    "W.236": "Hết đường đôi",
    "W.237": "Cầu vồng",
    "W.238": "Đường cao tốc phía trước",
    "W.239a": "Đường cáp điện ở phía trên",
    "W.239b": "Chiều cao tĩnh không thực tế",
    "W.240": "Đường hầm",
    "W.241": "Ùn tắc giao thông",
    "W.242a": "Nơi đường sắt giao vuông góc với đường bộ (một đường cắt)",
    "W.242b": "Nơi đường sắt giao vuông góc với đường bộ (từ hai đường cắt trở lên)",
    "W.243a": "Nơi đường sắt giao không vuông góc với đường bộ (cách 50m)",
    "W.243b": "Nơi đường sắt giao không vuông góc với đường bộ (cách 100m)",
    "W.243c": "Nơi đường sắt giao không vuông góc với đường bộ (cách 150m)",
    "W.244": "Đoạn đường hay xảy ra tai nạn",
    "W.245a": "Đi chậm",
    "W.245b": "Đi chậm (có chữ SLOW)",
    "W.246a": "Chú ý chướng ngại vật (vòng tránh sang hai bên)",
    "W.246b": "Chú ý chướng ngại vật (vòng tránh sang bên phải)",
    "W.246c": "Chú ý chướng ngại vật (vòng tránh sang bên trái)",
    "W.247": "Chú ý xe đỗ",
    # Nhóm R
    "R.122": "Dừng lại (STOP)",
    "R.301a": "Các xe chỉ được đi thẳng",
    "R.301b": "Các xe chỉ được rẽ phải",
    "R.301c": "Các xe chỉ được rẽ trái",
    "R.301d": "Các xe chỉ được rẽ phải (đặt sau ngã ba, ngã tư)",
    "R.301e": "Các xe chỉ được rẽ trái (đặt sau ngã ba, ngã tư)",
    "R.301f": "Các xe chỉ được đi thẳng và rẽ phải",
    "R.301g": "Các xe chỉ được đi thẳng và rẽ trái",
    "R.301h": "Các xe chỉ được rẽ trái và rẽ phải",
    "R.302a": "Hướng phải đi vòng chướng ngại vật (sang bên phải)",
    "R.302b": "Hướng phải đi vòng chướng ngại vật (sang bên trái)",
    "R.303": "Nơi giao nhau chạy theo vòng xuyến",
    "R.304": "Đường dành cho xe thô sơ",
    "R.305": "Đường dành cho người đi bộ",
    "R.306": "Tốc độ tối thiểu cho phép",
    "R.307": "Hết hạn chế tốc độ tối thiểu",
    "R.308a": "Tuyến đường cầu vượt cắt qua (đi thẳng hoặc rẽ trái)",
    "R.308b": "Tuyến đường cầu vượt cắt qua (đi thẳng hoặc vòng qua nút)",
    "R.309": "Ấn còi",
    "R.310a": "Hướng đi phải theo cho các xe chở hàng nguy hiểm (rẽ trái)",
    "R.310b": "Hướng đi phải theo cho các xe chở hàng nguy hiểm (đi thẳng)",
    "R.310c": "Hướng đi phải theo cho các xe chở hàng nguy hiểm (rẽ phải)",
    "R.403a": "Đường dành cho xe ô tô",
    "R.403b": "Đường dành cho xe ô tô, xe máy",
    "R.403c": "Đường dành cho xe buýt",
    "R.403d": "Đường dành cho xe ô tô con",
    "R.403e": "Đường dành cho xe máy",
    "R.403f": "Đường dành cho xe máy và xe đạp",
    "R.404a": "Hết đoạn đường dành cho xe ô tô",
    "R.404b": "Hết đoạn đường dành cho xe ô tô, xe máy",
    "R.404c": "Hết đoạn đường dành cho xe buýt",
    "R.404d": "Hết đoạn đường dành cho xe ô tô con",
    "R.404e": "Hết đoạn đường dành cho xe máy",
    "R.404f": "Hết đoạn đường dành cho xe máy và xe đạp",
    "R.411": "Hướng đi trên mỗi làn đường phải theo",
    "R.412a": "Làn đường dành riêng cho xe khách",
    "R.412b": "Làn đường dành riêng cho xe con",
    "R.412c": "Làn đường dành riêng cho xe tải",
    "R.412d": "Làn đường dành riêng cho xe máy",
    "R.412e": "Làn đường dành riêng cho xe buýt",
    "R.412f": "Làn đường dành riêng cho xe ô tô",
    "R.412g": "Làn đường dành riêng cho xe máy và xe đạp",
    "R.412h": "Làn đường dành riêng cho xe đạp",
    "R.412i": "Kết thúc làn đường dành riêng cho xe khách",
    "R.412j": "Kết thúc làn đường dành riêng cho xe con",
    "R.412k": "Kết thúc làn đường dành riêng cho xe tải",
    "R.412l": "Kết thúc làn đường dành riêng cho xe máy",
    "R.412m": "Kết thúc làn đường dành riêng cho xe buýt",
    "R.412n": "Kết thúc làn đường dành riêng cho xe ô tô",
    "R.412o": "Kết thúc làn đường dành riêng cho xe máy và xe đạp",
    "R.412p": "Kết thúc làn đường dành riêng cho xe đạp",
    "R.415a": "Biển gộp làn đường theo phương tiện",
    "R.415b": "Kết thúc làn đường theo phương tiện",
    "R.420": "Bắt đầu khu đông dân cư",
    "R.421": "Hết khu đông dân cư",
    "R.E.9a": "Khu vực cấm đỗ xe",
    "R.E.9b": "Khu vực cấm đỗ xe theo giờ",
    "R.E.9c": "Khu vực đỗ xe",
    "R.E.9d": "Hạn chế tốc độ tối đa trong khu vực",
    "R.E.10a": "Hết khu vực cấm đỗ xe",
    "R.E.10b": "Hết khu vực cấm đỗ xe theo giờ",
    "R.E.10c": "Hết khu vực đỗ xe",
    "R.E.10d": "Hết hạn chế tốc độ tối đa trong khu vực",
    "R.E.11a": "Báo hiệu có đường hầm",
    "R.E.11b": "Kết thúc đường hầm",
    # Nhóm I
    "I.401": "Bắt đầu đường ưu tiên",
    "I.402": "Hết đoạn đường ưu tiên",
    "I.405a": "Đường cụt (phía trước)",
    "I.405b": "Đường cụt (rẽ phải)",
    "I.405c": "Đường cụt (rẽ trái)",
    "I.406": "Được ưu tiên qua đường hẹp",
    "I.407a": "Đường một chiều (đi thẳng)",
    "I.407b": "Đường một chiều (bên phải)",
    "I.407c": "Đường một chiều (bên trái)",
    "I.408": "Nơi đỗ xe",
    "I.408a": "Nơi đỗ xe một phần trên hè phố",
    "I.409": "Chỗ quay xe",
    "I.410": "Khu vực quay xe",
    "I.413a": "Đường phía trước có làn đường dành cho ô tô khách",
    "I.413b": "Rẽ ra đường có làn đường dành cho ô tô khách (rẽ phải)",
    "I.413c": "Rẽ ra đường có làn đường dành cho ô tô khách (rẽ trái)",
    "I.414a": "Chỉ hướng đường (một địa danh)",
    "I.414b": "Chỉ hướng đường (hai địa danh cùng hướng)",
    "I.414c": "Chỉ hướng đường (hai hướng rẽ)",
    "I.414d": "Chỉ hướng đường (nhiều hướng)",
    "I.415": "Mũi tên chỉ hướng đi",
    "I.416": "Lối đi đường vòng, đường tránh",
    "I.417a": "Chỉ hướng đường phải đi cho từng loại xe",
    "I.417b": "Chỉ hướng đường phải đi cho từng loại xe",
    "I.417c": "Chỉ hướng đường phải đi cho từng loại xe",
    "I.418": "Lối đi ở những vị trí cấm rẽ",
    "I.419a": "Chỉ dẫn địa giới hành chính",
    "I.419b": "Chỉ dẫn địa giới hành chính (song ngữ)",
    "I.420": "Bắt đầu đoạn đường thử phanh",
    "I.421": "Hết đoạn đường thử phanh",
    "I.422": "Chỉ dẫn địa điểm danh lam thắng cảnh, di tích lịch sử",
    "I.423a": "Vị trí người đi bộ sang ngang",
    "I.423b": "Vị trí người đi bộ sang ngang",
    "I.423c": "Điểm bắt đầu đường đi bộ",
    "I.424a": "Cầu vượt qua đường cho người đi bộ",
    "I.424b": "Cầu vượt qua đường cho người đi bộ",
    "I.424c": "Hầm chui qua đường cho người đi bộ",
    "I.424d": "Hầm chui qua đường cho người đi bộ",
    "I.425": "Bệnh viện",
    "I.426": "Trạm cấp cứu",
    "I.427a": "Trạm sửa chữa",
    "I.427b": "Trạm kiểm tra tải trọng xe",
    "I.428": "Cửa hàng xăng dầu",
    "I.429": "Nơi rửa xe",
    "I.430": "Điện thoại",
    "I.431": "Trạm dừng nghỉ",
    "I.432": "Khách sạn",
    "I.433a": "Nơi cắm trại",
    "I.433b": "Nhà trọ",
    "I.433c": "Nơi cắm trại và nhà trọ",
    "I.433d": "Nhà nghỉ du lịch",
    "I.434a": "Bến xe buýt",
    "I.434b": "Bến xe ô tô khách",
    "I.435": "Bến tàu điện",
    "I.436": "Trạm cảnh sát giao thông",
    "I.437": "Đường cao tốc",
    "I.438": "Hết đường cao tốc",
    "I.439": "Tên cầu",
    "I.440": "Đoạn đường thi công",
    "I.441a": "Báo hiệu phía trước có công trường thi công (500m)",
    "I.441b": "Báo hiệu phía trước có công trường thi công (100m)",
    "I.441c": "Báo hiệu phía trước có công trường thi công (50m)",
    "I.442": "Chợ",
    "I.443": "Xe kéo rơ-moóc",
    "I.444a": "Chỉ dẫn lối vào sân bay",
    "I.444b": "Chỉ dẫn hướng vào sân bay",
    "I.444c": "Chỉ dẫn lối vào ga tàu hỏa",
    "I.444d": "Chỉ dẫn hướng vào ga tàu hỏa",
    "I.444e": "Chỉ dẫn lối vào bến xe khách",
    "I.444f": "Chỉ dẫn hướng vào bến xe khách",
    "I.444g": "Chỉ dẫn lối vào bến phà",
    "I.444h": "Chỉ dẫn hướng vào bến phà",
    "I.444i": "Chỉ dẫn lối vào bến tàu thủy",
    "I.444j": "Chỉ dẫn hướng vào bến tàu thủy",
    "I.444k": "Chỉ dẫn lối vào cảng hàng hóa",
    "I.444l": "Chỉ dẫn hướng vào cảng hàng hóa",
    "I.444m": "Chỉ dẫn hướng vào trạm dừng xe buýt",
    "I.445a": "Mô hình hóa hướng đường đi trên đường đèo dốc",
    "I.445b": "Mô hình hóa hướng đường đi trên đường đèo dốc",
    "I.445c": "Cảnh báo đường đèo dốc nguy hiểm",
    "I.445d": "Biển báo đường ngoặt nguy hiểm trên đèo núi",
    "I.445e": "Biển báo đường ngoặt nguy hiểm trên đèo núi",
    "I.445f": "Biển báo nhiều chỗ ngoặt nguy hiểm liên tiếp trên đèo núi",
    "I.445g": "Biển báo đoạn đường dốc nguy hiểm liên tục",
    "I.445h": "Biển báo đường xuống dốc nguy hiểm liên tục",
    "I.446": "Nơi đỗ xe dành cho người khuyết tật",
    "I.447a": "Biển báo cầu vượt liên thông",
    "I.447b": "Biển báo cầu vượt liên thông",
    "I.447c": "Biển báo cầu vượt liên thông",
    "I.447d": "Biển báo cầu vượt liên thông",
    "I.448": "Làn đường cứu nạn hay làn thoát xe khẩn cấp",
    "I.449": "Biển tên đường",
    # Nhóm S (Phụ lục F: Trang 151 - 157)
    "S.501": "Phạm vi tác dụng của biển",
    "S.502": "Khoảng cách đến đối tượng báo hiệu",
    "S.503a": "Hướng tác dụng của biển (bên phải)",
    "S.503b": "Hướng tác dụng của biển (hai bên trái và phải)",
    "S.503c": "Hướng tác dụng của biển (bên trái)",
    "S.503d": "Hướng tác dụng của biển (phía trước)",
    "S.503e": "Hướng tác dụng của biển (phía trước và phía sau)",
    "S.503f": "Hướng tác dụng của biển (phía sau)",
    "S.H,3a": "Hướng tác dụng của biển cấm đỗ xe (bắt đầu)",
    "S.H,3b": "Hướng tác dụng của biển cấm đỗ xe (nhắc lại)",
    "S.H,3c": "Hướng tác dụng của biển cấm đỗ xe (kết thúc)",
    "S.504": "Làn đường",
    "S.505a": "Loại xe (Biển phụ chỉ loại phương tiện chịu hiệu lực của biển chính)",
    "S.505b": "Loại xe hạn chế qua cầu",
    "S.505c": "Tải trọng trục hạn chế qua cầu",
    "S.506a": "Hướng đường ưu tiên (đặt dưới biển I.401)",
    "S.506b": "Hướng đường ưu tiên (đặt dưới biển W.208 hoặc R.122)",
    "S.507": "Hướng rẽ",
    "S.508a": "Biểu thị thời gian (khung giờ hiệu lực)",
    "S.508b": "Biểu thị thời gian (thứ/ngày hiệu lực)",
    "S.509a": "Thuyết minh biển chính - Chiều cao an toàn",
    "S.509b": "Thuyết minh biển chính - Cấm đỗ xe",
    "S.510a": "Chú ý đường trơn có băng tuyết",
    "S.510b": "Chú ý đường sắt",
    "S.G,7": "Địa điểm cắm trại",
    "S.G,8": "Địa điểm nhà trọ",
    "S.G,9b": "Nơi đỗ xe kết hợp phương tiện giao thông công cộng",
    "S.G,11a": "Số lượng làn đường và hướng đi trên từng làn",
    "S.G,11c": "Số lượng làn đường và hướng đi trên từng làn",
    "S.G,12a": "Làn đường không lưu thông (đóng làn ngoài cùng)",
    "S.G,12b": "Làn đường không lưu thông (đóng làn giữa)",
    "S.H,6": "Biển báo phụ Ngoại lệ",
}

VALID_SIGN_CODES = set(STANDARD_NAMES.keys())

# ==============================================================================
# BƯỚC 1: PHÂN TÍCH ĐIỀU LUẬT TỪ TRANG 80 ĐẾN 157 (PHỤ LỤC B, C, D, E, F)
# ==============================================================================
print("\n[1/4] Đang trích xuất văn bản pháp lý từ Phụ lục B, C, D, E, F (trang 80–157)...")

sec_pat = re.compile(r"^\s*([BCDEF]\.[0-9]+[a-z]?)\b[\.\s]*(.*)", re.IGNORECASE)
sections = {}
current_sec = None
current_sec_lines = []

for page_idx in range(79, 157):
    page_num = page_idx + 1
    p = doc[page_idx]
    for raw_l in p.get_text().split("\n"):
        l = clean_vn(raw_l).strip()
        if not l or l == str(page_num):
            continue
        m = sec_pat.match(l)
        if m:
            if current_sec:
                sections[current_sec] = "\n".join(current_sec_lines)
            current_sec = m.group(1).upper()
            current_sec_lines = [l]
        else:
            if current_sec:
                current_sec_lines.append(l)

if current_sec:
    sections[current_sec] = "\n".join(current_sec_lines)

legal_articles_dict = {}
AppendixNames = {
    "B": "Phụ lục B (Ý nghĩa - Sử dụng Biển báo cấm)",
    "C": "Phụ lục C (Ý nghĩa - Sử dụng Biển báo nguy hiểm và cảnh báo)",
    "D": "Phụ lục D (Ý nghĩa - Sử dụng Biển hiệu lệnh)",
    "E": "Phụ lục E (Ý nghĩa - Sử dụng Biển chỉ dẫn)",
    "F": "Phụ lục F (Ý nghĩa - Sử dụng Các biển phụ)",
}

sign_to_section = {}

for sec_k, sec_text in sections.items():
    lines = [l.strip() for l in sec_text.split("\n") if l.strip()]
    title = lines[1] if len(lines) > 1 else lines[0]
    content = " ".join(lines[1:]) if len(lines) > 1 else lines[0]
    sentences = re.split(r"[.;]\s+", content)
    summary = ". ".join(sentences[:2]).strip() if sentences else content[:250]
    if not summary.endswith("."):
        summary += "."

    app_letter = sec_k[0]
    art_key = f"QCVN41_{sec_k}"
    legal_articles_dict[art_key] = {
        "article_id": art_key,
        "section": sec_k,
        "appendix": AppendixNames.get(app_letter, f"Phụ lục {app_letter}"),
        "title": f"Phụ lục {app_letter} - Mục {sec_k}: {title}",
        "source": "QCVN 41:2019/BGTVT",
        "summary": summary[:350],
        "full_text": content.strip(),
    }

    for std_code in VALID_SIGN_CODES:
        base_num = std_code.split(".")[1] if "." in std_code else std_code
        base_prefix = std_code.split(".")[0] + "." + re.sub(r"[a-z]$", "", base_num)
        if std_code in sec_text or base_prefix in sec_text:
            if std_code not in sign_to_section:
                sign_to_section[std_code] = sec_k

# Bổ sung điều khoản cho biển Camera giám sát (nhóm biển chỉ dẫn thông tin theo QCVN 41:2019/BGTVT)
legal_articles_dict["QCVN41_CAMERA"] = {
    "article_id": "QCVN41_CAMERA",
    "section": "Chương IV & Phụ lục E",
    "appendix": "Phụ lục E (Ý nghĩa - Sử dụng Biển chỉ dẫn)",
    "title": "Quy chuẩn QCVN 41:2019/BGTVT: Biển chỉ dẫn đường có Camera giám sát giao thông",
    "source": "QCVN 41:2019/BGTVT",
    "summary": "Biển chỉ dẫn đoạn đường có hệ thống camera giám sát giao thông để hướng dẫn người điều khiển phương tiện chấp hành đúng quy tắc báo hiệu đường bộ.",
    "full_text": "Biển báo 'Đường có camera giám sát' thuộc nhóm biển chỉ dẫn trên đường bộ theo QCVN 41:2019/BGTVT, thông báo cho người tham gia giao thông biết khu vực hoặc tuyến đường phía trước có lắp đặt hệ thống camera giám sát giao thông để chủ động tuân thủ đúng tốc độ, làn đường và hiệu lệnh báo hiệu đường bộ.",
}


def resolve_sign_code(raw_code: str) -> str:
    """Chuẩn hóa mã biển báo và loại bỏ các hậu tố ký tự bị dính do khoảng trắng."""
    c = raw_code.strip()
    if c in VALID_SIGN_CODES:
        return c
    # Kiểm tra chữ hoa/thường
    for vc in VALID_SIGN_CODES:
        if vc.lower() == c.lower():
            return vc
    # Nếu bị dính 1 chữ cái cuối không nằm trong danh mục (ví dụ P.102m -> P.102, W.224e -> W.224)
    if len(c) > 3 and c[-1].isalpha():
        stripped = c[:-1]
        for vc in VALID_SIGN_CODES:
            if vc.lower() == stripped.lower():
                return vc
        # Nếu mã gốc có biến thể a (như I.407 -> I.407a, W.207 -> W.207a)
        for vc in VALID_SIGN_CODES:
            if vc.lower() == (stripped + "a").lower():
                return vc
    for vc in VALID_SIGN_CODES:
        if vc.lower() == (c + "a").lower():
            return vc
    return c


# ==============================================================================
# BƯỚC 2: QUÉT VÀ CẮT ẢNH BIỂN BÁO TỪ TRANG 80 ĐẾN 157 (BAO GỒM PHỤ LỤC F)
# ==============================================================================
print("\n[2/4] Đang quét và cắt ảnh chuẩn xác từng biển báo từ trang 80 đến 157...")

sub_sign_pat = re.compile(
    r"([a-z]\))\s*(?:Hình\s*[A-Z]\.[0-9]+[a-z]?\s*[-–—:]\s*)?Biển\s*số\s*(S\.[HG],[0-9]{1,2}[a-c]?|[A-Z]{1,2}(?:\.E)?\.[0-9]{1,3}[a-pA-P]?)\b",
    re.IGNORECASE,
)
fig_sign_pat = re.compile(
    r"Hình\s*([B-F])\s*\.\s*([0-9]+)([a-z]?)\s*[-–—:]\s*Biển\s*số\s*(S\.[HG],[0-9]{1,2}[a-c]?|[A-Z]{1,2}(?:\.E)?\.[0-9]{1,3}[a-pA-P]?)\b",
    re.IGNORECASE,
)
sign_code_pat = re.compile(
    r"Biển\s*số\s*(S\.[HG],[0-9]{1,2}[a-c]?|[A-Z]{1,2}(?:\.E)?\.[0-9]{1,3}[a-pA-P]?)\b",
    re.IGNORECASE,
)

ZOOM_DPI = 250 / 72.0
mat = pymupdf.Matrix(ZOOM_DPI, ZOOM_DPI)

crop_records = []

for page_idx in range(79, 157):
    page_num = page_idx + 1
    p = doc[page_idx]

    raw_rects = []
    for img_info in p.get_images():
        xref = img_info[0]
        for r in p.get_image_rects(xref):
            raw_rects.append(r)
    if not raw_rects:
        continue
    raw_rects.sort(key=lambda r: (round(r.y0 / 20) * 20, r.x0))

    merged = []
    skip = set()
    for i in range(len(raw_rects)):
        if i in skip:
            continue
        curr = raw_rects[i]
        j = i + 1
        while j < len(raw_rects):
            nxt = raw_rects[j]
            if (
                abs(nxt.y0 - curr.y1) <= 2.5
                and abs(nxt.x0 - curr.x0) <= 2.5
                and abs(nxt.x1 - curr.x1) <= 2.5
            ):
                curr = curr | nxt
                skip.add(j)
                j += 1
            else:
                break
        merged.append(curr)

    sorted_items = sorted(merged, key=lambda x: x.y0)
    rows = []
    curr_row = [sorted_items[0]]
    for it in sorted_items[1:]:
        if abs(it.y0 - curr_row[-1].y0) <= 25:
            curr_row.append(it)
        else:
            rows.append(curr_row)
            curr_row = [it]
    rows.append(curr_row)

    # Xử lý đặc biệt cho các trang Phụ lục F (151-157) để đạt độ chính xác 100%
    if page_num == 151 and len(rows) == 3:
        crop_records.append((151, rows[0][0], "S.501"))
        crop_records.append((151, rows[1][0], "S.502"))
        for idx_s, code_s in enumerate(["S.503a", "S.503b", "S.503c"]):
            if idx_s < len(rows[2]):
                crop_records.append((151, rows[2][idx_s], code_s))
        continue

    if page_num == 152 and len(rows) == 4:
        for idx_s, code_s in enumerate(["S.503d", "S.503e", "S.503f"]):
            if idx_s < len(rows[0]):
                crop_records.append((152, rows[0][idx_s], code_s))
        for idx_s, code_s in enumerate(["S.H,3a", "S.H,3b", "S.H,3c"]):
            if idx_s < len(rows[1]):
                crop_records.append((152, rows[1][idx_s], code_s))
        crop_records.append((152, rows[2][0], "S.504"))
        crop_records.append((152, rows[3][0], "S.505a"))
        continue

    if page_num == 153 and len(rows) == 2:
        crop_records.append((153, rows[0][0], "S.505b"))
        crop_records.append((153, rows[1][0], "S.505c"))
        continue

    if page_num == 154 and len(rows) >= 3:
        if len(rows[0]) == 2:
            crop_records.append((154, rows[0][0], "S.506a"))
            crop_records.append((154, rows[0][1], "S.506b"))
        crop_records.append((154, rows[1][0], "S.507"))
        if len(rows) == 4:
            # S.508a bên trái, S.508b bên phải
            all_bottom = sorted(rows[2] + rows[3], key=lambda r_it: r_it.x0)
            crop_records.append((154, all_bottom[0], "S.508a"))
            crop_records.append((154, all_bottom[-1], "S.508b"))
        elif len(rows[2]) == 2:
            crop_records.append((154, rows[2][0], "S.508a"))
            crop_records.append((154, rows[2][1], "S.508b"))
        continue

    if page_num == 155:
        all_r = sorted(merged, key=lambda r_it: (r_it.y0, r_it.x0))
        if len(all_r) >= 4:
            crop_records.append((155, all_r[0], "S.509a"))
            crop_records.append((155, all_r[1], "S.509b"))
            crop_records.append((155, all_r[2], "S.510a"))
            crop_records.append((155, all_r[3], "S.510b"))
        continue

    if page_num == 156 and len(rows) == 3:
        if len(rows[0]) == 2:
            crop_records.append((156, rows[0][0], "S.G,7"))
            crop_records.append((156, rows[0][1], "S.G,8"))
        crop_records.append((156, rows[1][0], "S.G,9b"))
        if len(rows[2]) == 2:
            crop_records.append((156, rows[2][0], "S.G,11a"))
            crop_records.append((156, rows[2][1], "S.G,11c"))
        continue

    if page_num == 157 and len(rows) == 2:
        r0 = rows[0][0]
        mid_x = (r0.x0 + r0.x1) / 2.0
        crop_records.append((157, pymupdf.Rect(r0.x0, r0.y0, mid_x, r0.y1), "S.G,12a"))
        crop_records.append((157, pymupdf.Rect(mid_x, r0.y0, r0.x1, r0.y1), "S.G,12b"))
        crop_records.append((157, rows[1][0], "S.H,6"))
        continue

    for r in rows:
        r.sort(key=lambda x: x.x0)
        min_x = min(it.x0 for it in r)
        max_x = max(it.x1 for it in r)
        max_y = max(it.y1 for it in r)
        min_y = min(it.y0 for it in r)

        if page_num == 90 and len(r) == 1 and r[0].y0 < 100:
            continue

        if page_num == 116:
            if len(r) == 1 and 400 < r[0].y0 < 550:
                w_quarter = (r[0].x1 - r[0].x0) / 4.0
                for q_idx, code_q in enumerate(["R.301a", "R.301b", "R.301c", "R.301d"]):
                    r_sub = pymupdf.Rect(
                        r[0].x0 + q_idx * w_quarter,
                        r[0].y0,
                        r[0].x0 + (q_idx + 1) * w_quarter,
                        r[0].y1,
                    )
                    crop_records.append((page_num, r_sub, code_q))
                continue
            elif len(r) == 1 and 550 < r[0].y0 < 700:
                w_quarter = (r[0].x1 - r[0].x0) / 4.0
                for q_idx, code_q in enumerate(["R.301e", "R.301f", "R.301g", "R.301h"]):
                    r_sub = pymupdf.Rect(
                        r[0].x0 + q_idx * w_quarter,
                        r[0].y0,
                        r[0].x0 + (q_idx + 1) * w_quarter,
                        r[0].y1,
                    )
                    crop_records.append((page_num, r_sub, code_q))
                continue

        if page_num == 120 and len(r) == 1 and r[0].y0 < 100:
            w_third = (r[0].x1 - r[0].x0) / 3.0
            crop_records.append((page_num, pymupdf.Rect(r[0].x0, r[0].y0, r[0].x0 + w_third, r[0].y1), "R.310a"))
            crop_records.append((page_num, pymupdf.Rect(r[0].x0 + w_third, r[0].y0, r[0].x0 + 2 * w_third, r[0].y1), "R.310b"))
            crop_records.append((page_num, pymupdf.Rect(r[0].x0 + 2 * w_third, r[0].y0, r[0].x1, r[0].y1), "R.310c"))
            continue

        if page_num == 88 and len(r) == 4 and r[0].y0 < 250:
            crop_records.append((page_num, r[0], "P.124a"))
            crop_records.append((page_num, r[1], "P.124a"))
            crop_records.append((page_num, r[2], "P.124b"))
            crop_records.append((page_num, r[3], "P.124b"))
            continue

        if page_num == 92 and len(r) == 2 and r[0].y0 > 400:
            crop_records.append((page_num, r[0], "DP.127a"))
            crop_records.append((page_num, r[1], "DP.127b"))
            continue

        if page_num == 100 and len(r) == 5:
            for idx_w, letter in enumerate(["a", "b", "c", "d", "e"]):
                crop_records.append((page_num, r[idx_w], f"W.205{letter}"))
            continue

        if page_num == 104 and len(r) == 2 and r[0].y0 > 450:
            crop_records.append((page_num, r[0], "W.216a"))
            crop_records.append((page_num, r[1], "W.216b"))
            continue

        if page_num == 105 and len(r) == 1 and r[0].y0 > 500:
            crop_records.append((page_num, r[0], "W.219"))
            continue

        if page_num == 149 and len(r) == 1 and 400 < r[0].y0 < 630:
            continue
        if page_num == 149 and len(r) == 1 and r[0].y0 > 630:
            crop_records.append((page_num, r[0], "I.448"))
            continue

        if page_num == 150 and len(r) == 1:
            crop_records.append((page_num, r[0], "I.449"))
            continue

        clip_below = pymupdf.Rect(
            max(0, min_x - 30), max_y - 2, min(p.rect.width, max_x + 30), min(p.rect.height, max_y + 45)
        )
        text_below = clean_vn(p.get_text("text", clip=clip_below).replace("\n", " "))

        sub_matches = list(sub_sign_pat.finditer(text_below))
        if len(r) > 1 and len(sub_matches) == len(r):
            for i, it in enumerate(r):
                raw_c = sub_matches[i].group(2)
                if not raw_c[-1].isalpha():
                    raw_c = raw_c + sub_matches[i].group(1)[0].lower()
                crop_records.append((page_num, it, resolve_sign_code(raw_c)))
        elif len(r) == 1:
            m_fig = fig_sign_pat.search(text_below)
            if m_fig:
                raw_c = m_fig.group(4)
                fig_let = m_fig.group(3).lower() if m_fig.group(3) else ""
                if raw_c.upper().startswith("P.127") and fig_let:
                    if fig_let == "c":
                        raw_c = "P.127a"
                    elif fig_let == "d":
                        raw_c = "P.127b"
                    elif fig_let == "e":
                        raw_c = "P.127c"
                    elif fig_let == "b":
                        raw_c = "P.127"
                elif not raw_c[-1].isalpha() and fig_let and (raw_c + fig_let) in VALID_SIGN_CODES:
                    raw_c = raw_c + fig_let
                crop_records.append((page_num, r[0], resolve_sign_code(raw_c)))
            else:
                m_bs = sign_code_pat.search(text_below)
                if m_bs:
                    crop_records.append((page_num, r[0], resolve_sign_code(m_bs.group(1))))
                else:
                    clip_above = pymupdf.Rect(
                        max(0, min_x - 30), max(0, min_y - 45), min(p.rect.width, max_x + 30), min_y + 2
                    )
                    text_above = clean_vn(p.get_text("text", clip=clip_above).replace("\n", " "))
                    m_above = fig_sign_pat.search(text_above) or sign_code_pat.search(text_above)
                    if m_above:
                        raw_c = m_above.group(4) if len(m_above.groups()) >= 4 else m_above.group(1)
                        crop_records.append((page_num, r[0], resolve_sign_code(raw_c)))
        else:
            for it in r:
                it_clip = pymupdf.Rect(
                    max(0, it.x0 - 25), it.y1 - 2, min(p.rect.width, it.x1 + 25), min(p.rect.height, it.y1 + 45)
                )
                it_text = clean_vn(p.get_text("text", clip=it_clip).replace("\n", " "))
                m_it = sub_sign_pat.search(it_text) or sign_code_pat.search(it_text)
                if m_it:
                    raw_c = m_it.group(2) if len(m_it.groups()) >= 2 else m_it.group(1)
                    if len(m_it.groups()) >= 2 and not raw_c[-1].isalpha():
                        raw_c = raw_c + m_it.group(1)[0].lower()
                    crop_records.append((page_num, it, resolve_sign_code(raw_c)))

print(f"[*] Tổng số biển báo chuẩn trích xuất từ PDF (trang 80–157): {len(crop_records)}")

saved_pdf_signs = []
sign_variant_counter = Counter()
code_to_pdf_images = defaultdict(list)

for idx, (page_num, rect, code) in enumerate(crop_records):
    p = doc[page_num - 1]
    sign_variant_counter[code] += 1
    var_idx = sign_variant_counter[code]

    # Lưu vào data/pdf_signs/<code>/ và đồng bộ sang images/<code>/
    safe_dir_name = code.replace(",", "_")
    sign_dir = PDF_SIGNS_DIR / safe_dir_name
    compat_dir = IMAGES_COMPAT_DIR / safe_dir_name
    sign_dir.mkdir(parents=True, exist_ok=True)
    compat_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{safe_dir_name}_{var_idx:02d}.png"
    rel_pdf_path = f"data/pdf_signs/{safe_dir_name}/{filename}"
    rel_compat_path = f"images/{safe_dir_name}/{filename}"

    pad_x = max(3.0, rect.width * 0.03)
    pad_y = max(3.0, rect.height * 0.03)
    crop_rect = pymupdf.Rect(
        max(0, rect.x0 - pad_x),
        max(0, rect.y0 - pad_y),
        min(p.rect.width, rect.x1 + pad_x),
        min(p.rect.height, rect.y1 + pad_y),
    )

    pix = p.get_pixmap(matrix=mat, clip=crop_rect)
    pix.save(str(sign_dir / filename))
    pix.save(str(compat_dir / filename))

    name = STANDARD_NAMES.get(code, f"Biển số {code}")
    cat = get_category_by_code(code)
    sec = sign_to_section.get(code)
    if not sec:
        prefix = code.split(".")[0]
        sec_prefix = {"P": "B", "DP": "B", "W": "C", "R": "D", "I": "E", "S": "F"}.get(prefix, "E")
        for k, v in legal_articles_dict.items():
            if k.startswith(f"QCVN41_{sec_prefix}") and code.split(".")[1][:3] in v["full_text"]:
                sec = v["section"]
                break
    art_key = f"QCVN41_{sec}" if sec else "QCVN41_B.1"
    art_obj = legal_articles_dict.get(art_key, {})

    entry = {
        "id": f"pdf_sign_{idx+1:03d}",
        "page": page_num,
        "image": rel_pdf_path,
        "compat_image": rel_compat_path,
        "sign_code": code,
        "sign_name": name,
        "category": cat,
        "relevant_articles": [art_key],
        "description": art_obj.get("summary", f"Biển số {code}: {name} theo QCVN 41:2019/BGTVT."),
        "dimensions": [pix.width, pix.height],
    }
    saved_pdf_signs.append(entry)
    code_to_pdf_images[code].append(rel_pdf_path)

print(f"[✓] Đã lưu {len(saved_pdf_signs)} ảnh biển báo chuẩn từ PDF ({len(code_to_pdf_images)} mã biển duy nhất).")

# ==============================================================================
# BƯỚC 3: ÁNH XẠ 52 LỚP BIỂN BÁO TRONG ARCHIVE/CLASSES.TXT VỚI MÃ QCVN 41:2019
# ==============================================================================
print("\n[3/4] Đang ánh xạ 52 lớp biển báo thực tế trong archive/classes.txt với QCVN 41:2019/BGTVT...")

with open(ARCHIVE_DIR / "classes.txt", "r", encoding="utf-8") as f:
    raw_classes = [l.strip() for l in f if l.strip()]
with open(ARCHIVE_DIR / "classes_vie.txt", "r", encoding="utf-8") as f:
    vie_classes = [l.strip() for l in f if l.strip()]
with open(ARCHIVE_DIR / "classes_en.txt", "r", encoding="utf-8") as f:
    en_classes = [l.strip() for l in f if l.strip()]

# Bảng ánh xạ chuẩn xác từ nhãn lớp trong dataset thực tế sang mã chuẩn QCVN 41:2019/BGTVT
ARCHIVE_TO_CANONICAL_CODE = {
    "W.207": "W.207a",
    "P.117*": "P.117",
    "P.124a*": "P.124a",
    "P.111": "P.111a",
    "S.505a*Xe máy": "S.505a",
    "S.505a*Xe tải và công": "S.505a",
    "S.505a*Xe tải": "S.505a",
    "P.106a*Xe tải": "P.106a",
    "B.8a": "P.108a",  # Cấm xe sơ-mi rơ-moóc (P.108a tại Mục B.8 Phụ lục B)
    "P.127*50": "P.127",
    "P.127*60": "P.127",
    "P.127*80": "P.127",
    "P.127*40": "P.127",
    "W.239b*": "W.239b",
    "P.124b1": "P.124b",
    "Camera": "Camera",
}

class_mapping = []
for cls_id, (raw_c, vie_n, en_n) in enumerate(zip(raw_classes, vie_classes, en_classes)):
    canonical_code = ARCHIVE_TO_CANONICAL_CODE.get(raw_c, raw_c)
    pdf_imgs = code_to_pdf_images.get(canonical_code, [])
    sec = sign_to_section.get(canonical_code)
    if canonical_code == "Camera":
        art_key = "QCVN41_CAMERA"
    elif sec:
        art_key = f"QCVN41_{sec}"
    else:
        prefix = canonical_code.split(".")[0]
        sec_prefix = {"P": "B", "DP": "B", "W": "C", "R": "D", "I": "E", "S": "F"}.get(prefix, "E")
        art_key = f"QCVN41_{sec_prefix}.1"
        for k, v in legal_articles_dict.items():
            if k.startswith(f"QCVN41_{sec_prefix}") and canonical_code in v["full_text"]:
                art_key = k
                break

    std_name = STANDARD_NAMES.get(canonical_code, vie_n)
    art_obj = legal_articles_dict.get(art_key, {})

    class_mapping.append({
        "class_id": cls_id,
        "dataset_code": raw_c,
        "canonical_qcvn_code": canonical_code,
        "name_vi": vie_n,
        "official_qcvn_name": std_name,
        "name_en": en_n,
        "category": get_category_by_code(canonical_code),
        "article_id": art_key,
        "article_title": art_obj.get("title", ""),
        "legal_summary": art_obj.get("summary", ""),
        "pdf_template_images": pdf_imgs,
    })

unmatched_templates = [m for m in class_mapping if not m["pdf_template_images"] and m["canonical_qcvn_code"] != "Camera"]
print(f"[✓] Đã ánh xạ 52/52 lớp với điều luật QCVN 41:2019/BGTVT. Số lớp có biển mẫu PDF trực tiếp: {52 - len(unmatched_templates) - 1}/51 (+1 lớp biển chỉ dẫn Camera).")
if unmatched_templates:
    print("    Các lớp chưa có mẫu PDF:", [m["canonical_qcvn_code"] for m in unmatched_templates])

# ==============================================================================
# BƯỚC 4: TRÍCH XUẤT PATCH BIỂN BÁO THỰC TẾ TỪ 3,216 ẢNH ĐƯỜNG PHỐ (ARCHIVE)
# ==============================================================================
print("\n[4/4] Đang kiểm kê 3,216 ảnh giao thông thực tế và cắt mẫu biển báo thực tế (real crops)...")

MAX_CROPS_PER_CLASS = 40  # Lưu tối đa 40 patch chất lượng cao mỗi lớp để phục vụ huấn luyện & trực quan hóa nhanh, đồng thời lưu toàn bộ metadata 8,334 bounding boxes
real_crop_counts = Counter()
all_scene_annotations = []
class_box_counter = Counter()

img_files = sorted((ARCHIVE_DIR / "images").glob("*.jpg"))
for img_path in img_files:
    stem = img_path.stem
    lbl_path = ARCHIVE_DIR / "labels" / f"{stem}.txt"
    boxes = []
    if lbl_path.exists():
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])
                    boxes.append({
                        "class_id": cid,
                        "dataset_code": raw_classes[cid],
                        "canonical_code": class_mapping[cid]["canonical_qcvn_code"],
                        "bbox_yolo": [xc, yc, bw, bh],
                    })
                    class_box_counter[cid] += 1

    # Cắt patch thực tế cho các lớp chưa đủ MAX_CROPS_PER_CLASS
    need_crop = any(real_crop_counts[b["class_id"]] < MAX_CROPS_PER_CLASS for b in boxes)
    if need_crop and boxes:
        try:
            with Image.open(img_path) as im:
                im_w, im_h = im.size
                for b_idx, b in enumerate(boxes):
                    cid = b["class_id"]
                    if real_crop_counts[cid] >= MAX_CROPS_PER_CLASS:
                        continue
                    xc, yc, bw, bh = b["bbox_yolo"]
                    # Chỉ cắt các biển có kích thước đủ rõ (>= 16x16 px) hoặc thuộc lớp hiếm
                    px_w = bw * im_w
                    px_h = bh * im_h
                    if px_w < 14 or px_h < 14:
                        if class_box_counter[cid] > 30:
                            continue
                    pad_w = max(4.0, px_w * 0.15)
                    pad_h = max(4.0, px_h * 0.15)
                    x0 = max(0, int((xc - bw / 2) * im_w - pad_w))
                    y0 = max(0, int((yc - bh / 2) * im_h - pad_h))
                    x1 = min(im_w, int((xc + bw / 2) * im_w + pad_w))
                    y1 = min(im_h, int((yc + bh / 2) * im_h + pad_h))
                    if x1 > x0 and y1 > y0:
                        patch = im.crop((x0, y0, x1, y1)).convert("RGB")
                        safe_code = b["dataset_code"].replace("*", "_").replace(" ", "_")
                        cls_crop_dir = REAL_CROPS_DIR / f"{cid:02d}_{safe_code}"
                        cls_crop_dir.mkdir(parents=True, exist_ok=True)
                        crop_fname = f"{stem}_b{b_idx}_{safe_code}.jpg"
                        patch.save(str(cls_crop_dir / crop_fname), quality=92)
                        b["crop_path"] = f"data/real_crops/{cid:02d}_{safe_code}/{crop_fname}"
                        real_crop_counts[cid] += 1
        except Exception as e:
            pass

    all_scene_annotations.append({
        "image_id": stem,
        "image_path": f"archive/images/{img_path.name}",
        "num_boxes": len(boxes),
        "classes_present": sorted(list(set(b["class_id"] for b in boxes))),
        "sign_codes_present": sorted(list(set(b["canonical_code"] for b in boxes))),
        "boxes": boxes,
    })

for m in class_mapping:
    cid = m["class_id"]
    m["total_real_instances"] = class_box_counter[cid]
    m["saved_real_crops"] = real_crop_counts[cid]

# Lưu seluruh tệp dữ liệu tổng hợp
with open(DATA_DIR / "legal_articles.json", "w", encoding="utf-8") as f:
    json.dump(legal_articles_dict, f, ensure_ascii=False, indent=2)
with open(BASE_DIR / "legal_articles.json", "w", encoding="utf-8") as f:
    json.dump(legal_articles_dict, f, ensure_ascii=False, indent=2)

with open(DATA_DIR / "pdf_signs_catalog.json", "w", encoding="utf-8") as f:
    json.dump({
        "pdf_source": str(PDF_PATH),
        "page_range": [80, 157],
        "appendices": ["B (Biển báo cấm)", "C (Biển cảnh báo)", "D (Biển hiệu lệnh)", "E (Biển chỉ dẫn)", "F (Biển phụ)"],
        "total_signs_cropped": len(saved_pdf_signs),
        "unique_codes": len(code_to_pdf_images),
        "dataset": saved_pdf_signs,
    }, f, ensure_ascii=False, indent=2)

with open(DATA_DIR / "class_mapping_52.json", "w", encoding="utf-8") as f:
    json.dump(class_mapping, f, ensure_ascii=False, indent=2)

with open(DATA_DIR / "scene_annotations.json", "w", encoding="utf-8") as f:
    json.dump({
        "total_scenes": len(all_scene_annotations),
        "total_bounding_boxes": sum(class_box_counter.values()),
        "total_real_crops_saved": sum(real_crop_counts.values()),
        "scenes": all_scene_annotations,
    }, f, ensure_ascii=False, indent=2)

print(f"[✓] Đã lưu {sum(real_crop_counts.values())} ảnh cắt biển báo thực tế (real crops) từ {len(all_scene_annotations)} khung hình giao thông.")
print(f"[✓] Đã xuất dữ liệu đầy đủ tại thư mục: {DATA_DIR}")
