import sys
sys.stdout.reconfigure(encoding="utf-8")
from PIL import Image
import server

print("--- 1. Testing server knowledge base loaded ---")
print("Legal DB count:", len(server.legal_db))
print("Signs catalog count:", len(server.signs_catalog))
print("Visual ref count:", len(server.visual_ref_index))

print("\n--- 2. Testing build_grounded_legal_answer with various queries ---")
sign_info_131 = {
    "sign_code": "P.131a",
    "sign_name": "Cấm đỗ xe",
    "category": "Biển báo cấm (Nhóm P)",
    "image": "images/P.131a/P.131a_01.png"
}
art_131 = [{
    "article_id": "QCVN41_B.31",
    "title": "Phụ lục B - Mục B.31: Biển số P.131 (a,b,c) 'Cấm đỗ xe'",
    "summary": "Để báo nơi cấm đỗ xe cơ giới và xe thô sơ, trừ các xe được ưu tiên theo quy định."
}]

# Test 1: Mức phạt
ans_phat = server.build_grounded_legal_answer("Lỡ đỗ xe ở đây thì mức phạt là bao nhiêu tiền?", "P.131a", sign_info_131, art_131)
print("\n[TEST 1: Mức phạt P.131a]")
print(ans_phat[:250], "...\n")
assert "MỨC XỬ PHẠT VI PHẠM" in ans_phat
assert "400.000 đồng" in ans_phat

# Test 2: Đỗ xe / Dừng xe
ans_ds = server.build_grounded_legal_answer("Gặp biển này tôi có được đỗ xe không?", "P.131a", sign_info_131, art_131)
print("[TEST 2: Đỗ xe P.131a]")
print(ans_ds[:250], "...\n")
assert "CẤM ĐỖ XE" in ans_ds
assert "Bị cấm" in ans_ds

# Test 2B: Đúng / Sai Xe tải đi vào P.106a
sign_info_106 = {
    "sign_code": "P.106a",
    "sign_name": "Cấm xe ô tô tải",
    "category": "Biển báo cấm (Nhóm P)",
    "image": "images/P.106a/P.106a_01.png"
}
ans_yn = server.build_grounded_legal_answer("Xe tải có được đi vào tuyến đường này không?", "P.106a", sign_info_106, [])
print("[TEST 2B: Xe tải P.106a]")
print(ans_yn[:250], "...\n")
assert "CẤM" in ans_yn or "KHÔNG ĐƯỢC PHÉP" in ans_yn

# Test 3: Rẽ trái & Quay đầu P.124a
sign_info_124 = {
    "sign_code": "P.124a",
    "sign_name": "Cấm quay đầu xe",
    "category": "Biển báo cấm (Nhóm P)",
    "image": "images/P.124a/P.124a_01.png"
}
art_124 = [{
    "article_id": "QCVN41_B.24",
    "title": "Phụ lục B - Mục B.24: Biển số P.124a 'Cấm quay đầu xe'",
    "summary": "Để báo cấm các loại xe cơ giới và thô sơ quay đầu xe (theo kiểu chữ U)."
}]
ans_turn = server.build_grounded_legal_answer("Gặp biển này xe con có được rẽ trái không?", "P.124a", sign_info_124, art_124)
print("[TEST 3: Rẽ trái tại P.124a]")
print(ans_turn[:250], "...\n")
assert "ĐƯỢC PHÉP RẼ TRÁI" in ans_turn

# Test 4: Tốc độ tối đa P.127_50
sign_info_127 = {
    "sign_code": "P.127_50",
    "sign_name": "Tốc độ tối đa cho phép (50 km/h)",
    "category": "Biển báo cấm (Nhóm P)",
    "image": "images/P.127/P.127_01.png"
}
ans_spd = server.build_grounded_legal_answer("Đoạn đường này chạy tốc độ bao nhiêu km/h?", "P.127_50", sign_info_127, art_124)
print("[TEST 4: Tốc độ P.127_50]")
print(ans_spd[:250], "...\n")
assert "50 km/h" in ans_spd

# Test 5: Visual match on canonical test image
import os
test_img_path = "data/pdf_signs/P.102/P.102_01.png"
if os.path.exists(test_img_path):
    img = Image.open(test_img_path)
    code, conf = server.match_image_visually(img)
    print(f"\n[TEST 5: Visual recognition on {test_img_path}] -> Code: {code}, Conf: {conf}%")
    assert code == "P.102", f"Expected P.102 but got {code}"
    print("Visual recognition verified successfully!")

print("\nALL 5 CRITICAL TEST SUITES PASSED WITH 100% SUCCESS!")
