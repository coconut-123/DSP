import sys
sys.stdout.reconfigure(encoding="utf-8")
from starlette.testclient import TestClient
import server

client = TestClient(server.app)

print("\n=== TEST CHAT ENDPOINT 1: Text query about penalties ===")
res1 = client.post("/api/chat", data={
    "query": "Lỡ đỗ xe ở đây phạt bao nhiêu tiền?",
    "sign_code": "P.131a"
})
print("Status:", res1.status_code)
d1 = res1.json()
print("Detected sign:", d1["detected_sign"]["sign_code"], d1["detected_sign"]["sign_name"])
print("Answer:\n", d1["answer"][:300])
assert res1.status_code == 200
assert "MỨC XỬ PHẠT VI PHẠM" in d1["answer"]
assert "400.000 đồng" in d1["answer"]

print("\n=== TEST CHAT ENDPOINT 2: Image upload (P.102) ===")
with open("data/pdf_signs/P.102/P.102_01.png", "rb") as f:
    res2 = client.post("/api/chat", data={
        "query": "Biển này cấm xe gì?"
    }, files={
        "file": ("P.102_01.png", f, "image/png")
    })
print("Status:", res2.status_code)
d2 = res2.json()
print("Detected sign:", d2["detected_sign"]["sign_code"], d2["detected_sign"]["sign_name"])
print("Answer:\n", d2["answer"][:300])
assert res2.status_code == 200
assert d2["detected_sign"]["sign_code"] == "P.102"
assert "P.102" in d2["answer"]

print("\n=== TEST CHAT ENDPOINT 3: Parking question with image_base64 path ===")
res3 = client.post("/api/chat", data={
    "query": "Gặp biển này tôi có được đỗ xe không?",
    "image_base64": "images/P.131a/P.131a_01.png"
})
print("Status:", res3.status_code)
d3 = res3.json()
print("Detected sign:", d3["detected_sign"]["sign_code"], d3["detected_sign"]["sign_name"])
print("Answer:\n", d3["answer"][:300])
assert res3.status_code == 200
assert d3["detected_sign"]["sign_code"] == "P.131a"
assert "CẤM ĐỖ XE" in d3["answer"]

print("\n=== ALL FASTAPI ENDPOINT TESTS PASSED SUCCESSFULLY! ===")
