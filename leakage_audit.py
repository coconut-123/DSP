"""
leakage_audit.py
----------------
Thực hiện Kiểm toán Dữ liệu & Kiểm soát Rò rỉ Không gian/Thời gian (Leakage Audit & Group-Disjoint Partitioning)
theo đúng Phương pháp luận tại Mục 3 và Thuật toán 1 (Algorithm 1) của bài báo tham khảo:
"Leakage-Controlled Evaluation of Spectral-Adapted ViTs".

Các bước thực hiện:
1. Integrity Verification (Mục 3.2): Kiểm tra toàn vẹn 3,216 ảnh giao thông và 8,334 nhãn bounding box thuộc 52 lớp.
2. Grouping Rule (Mục 3.3): Gom nhóm bắc cầu (Disjoint-Set Union-Find) các khung hình liên tiếp / trùng lặp thị giác
   (dHash, aHash, MD5 content hash, và chuỗi khung hình dashcam liên tiếp có chung biển báo).
3. Group-Disjoint Partition (Mục 3.4): Phân chia Train/Val/Test (70/10/20) đảm bảo 0 nhóm bị cắt ngang qua biên tập dữ liệu
   (0 group crossing), cố định bằng mã băm SHA-256 (split_hash).
4. Reference Arm - Conventional Random Partition (Mục 3.5): Đối chiếu với phép chia ngẫu nhiên cấp độ ảnh (Random split)
   và bộ chia gốc archive/split_dataset, định lượng số nhóm bị rò rỉ qua biên Train/Test và tính thống kê chuẩn hóa
   per-channel (Mean, SD) độc lập trên tập Train của từng nhánh (Bảng 3 của bài báo).
"""

import os
import sys
import json
import hashlib
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "archive"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        path = []
        while self.parent[i] != i:
            path.append(i)
            i = self.parent[i]
        for node in path:
            self.parent[node] = i
        return i

    def union(self, i: int, j: int):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            if self.rank[ri] < self.rank[rj]:
                ri, rj = rj, ri
            self.parent[rj] = ri
            if self.rank[ri] == self.rank[rj]:
                self.rank[ri] += 1


def compute_hashes_and_stats(img_path: Path):
    """Tính dHash (64-bit), aHash (64-bit), MD5 content hash và mean/std RGB của ảnh."""
    raw_bytes = img_path.read_bytes()
    md5_hex = hashlib.md5(raw_bytes).hexdigest()

    with Image.open(img_path) as im:
        im_rgb = im.convert("RGB")
        # Resize nhỏ để tính thống kê RGB nhanh và chính xác
        im_small_rgb = np.asarray(im_rgb.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        rgb_mean = im_small_rgb.mean(axis=(0, 1))  # (3,)
        rgb_sq_mean = (im_small_rgb ** 2).mean(axis=(0, 1))  # (3,)

        # Grayscale cho aHash (8x8) và dHash (9x8)
        im_gray_a = np.asarray(im_rgb.convert("L").resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float32)
        ahash_bits = (im_gray_a > im_gray_a.mean()).flatten()
        ahash_int = int(np.packbits(ahash_bits).tobytes().hex(), 16)

        im_gray_d = np.asarray(im_rgb.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.float32)
        dhash_bits = (im_gray_d[:, 1:] > im_gray_d[:, :-1]).flatten()
        dhash_int = int(np.packbits(dhash_bits).tobytes().hex(), 16)

    return md5_hex, ahash_int, dhash_int, rgb_mean, rgb_sq_mean


def hamming_dist(h1: int, h2: int) -> int:
    return (h1 ^ h2).bit_count()


def run_audit():
    print("=" * 72)
    print(" BƯỚC 1: KIỂM TRA TOÀN VẸN DỮ LIỆU (INTEGRITY VERIFICATION - MỤC 3.2)")
    print("=" * 72)

    with open(ARCHIVE_DIR / "classes.txt", "r", encoding="utf-8") as f:
        classes = [l.strip() for l in f if l.strip()]
    with open(ARCHIVE_DIR / "classes_vie.txt", "r", encoding="utf-8") as f:
        classes_vie = [l.strip() for l in f if l.strip()]

    img_files = sorted((ARCHIVE_DIR / "images").glob("*.jpg"))
    n_imgs = len(img_files)
    print(f"[*] Tổng số file ảnh tìm thấy: {n_imgs}")

    records = []
    total_boxes = 0
    invalid_boxes = 0

    for idx, img_path in enumerate(img_files):
        stem = img_path.stem
        seq_num = int(stem) if stem.isdigit() else idx + 1
        lbl_path = ARCHIVE_DIR / "labels" / f"{stem}.txt"

        boxes = []
        cls_set = set()
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:])
                        if 0 <= cid < len(classes) and 0 <= xc <= 1 and 0 <= yc <= 1 and bw > 0 and bh > 0:
                            boxes.append((cid, xc, yc, bw, bh))
                            cls_set.add(cid)
                            total_boxes += 1
                        else:
                            invalid_boxes += 1

        md5_hex, ahash, dhash, rgb_mean, rgb_sq_mean = compute_hashes_and_stats(img_path)
        # Lớp chính của khung hình (ưu tiên lớp hiếm nhất trong ảnh để bảo toàn phân phối lớp hiếm)
        records.append({
            "idx": idx,
            "image_id": stem,
            "filename": img_path.name,
            "seq_num": seq_num,
            "md5": md5_hex,
            "ahash": ahash,
            "dhash": dhash,
            "rgb_mean": rgb_mean,
            "rgb_sq_mean": rgb_sq_mean,
            "classes": sorted(list(cls_set)),
            "num_boxes": len(boxes),
            "boxes": boxes,
        })

    # Tần suất lớp trên toàn bộ cohort
    global_cls_freq = Counter()
    for r in records:
        for c in r["classes"]:
            global_cls_freq[c] += 1

    for r in records:
        if r["classes"]:
            # Gán primary_class là lớp hiếm nhất xuất hiện trong ảnh
            r["primary_class"] = min(r["classes"], key=lambda c: (global_cls_freq[c], c))
        else:
            r["primary_class"] = -1

    print(f"[✓] Xác nhận hoàn tất {n_imgs}/3216 khung hình hợp lệ, {total_boxes} bounding boxes ({invalid_boxes} lỗi).")
    print(f"[✓] Số lớp biển báo hiện diện: {len(global_cls_freq)}/52 lớp.")

    print("\n" + "=" * 72)
    print(" BƯỚC 2: QUY TẮC GOM NHÓM BẮC CẦU (GROUPING RULE - MỤC 3.3)")
    print("=" * 72)

    uf = UnionFind(n_imgs)
    exact_dup_merges = 0
    temporal_burst_merges = 0
    perceptual_merges = 0

    # 1. Exact MD5 content hash
    md5_map = {}
    for i, r in enumerate(records):
        if r["md5"] in md5_map:
            uf.union(i, md5_map[r["md5"]])
            exact_dup_merges += 1
        else:
            md5_map[r["md5"]] = i

    # 2. Chuỗi khung hình liên tiếp cùng cảnh (Sequential burst <= 2 khung hình, cùng tập biển báo + tâm bbox gần nhau <= 0.06)
    # Giới hạn đường kính bắc cầu tối đa 12 khung hình liên tiếp (1 đoạn clip ngắn) để tránh chuỗi bắc cầu nối liền cả tuyến đường
    for i in range(n_imgs):
        ri = records[i]
        set_i = set(ri["classes"])
        for j in range(i + 1, min(n_imgs, i + 3)):
            rj = records[j]
            seq_gap = abs(rj["seq_num"] - ri["seq_num"])
            if seq_gap <= 2:
                d_dist = hamming_dist(ri["dhash"], rj["dhash"])
                a_dist = hamming_dist(ri["ahash"], rj["ahash"])
                # Kiểm tra khoảng cách tâm bounding box của cùng lớp biển báo
                same_cls = set_i & set(rj["classes"])
                min_box_dist = 1.0
                if same_cls:
                    for b1 in ri["boxes"]:
                        for b2 in rj["boxes"]:
                            if b1[0] == b2[0]:
                                dist = np.hypot(b1[1] - b2[1], b1[2] - b2[2])
                                if dist < min_box_dist:
                                    min_box_dist = dist
                if (same_cls and min_box_dist <= 0.065 and d_dist <= 14) or (d_dist <= 6 and a_dist <= 6):
                    # Chỉ ghép nếu nhóm sau khi ghép không vượt quá 15 khung hình
                    root_i, root_j = uf.find(i), uf.find(j)
                    if root_i != root_j and abs(records[j]["seq_num"] - records[root_i]["seq_num"]) <= 12:
                        uf.union(i, j)
                        temporal_burst_merges += 1

    # 3. Trùng lặp thị giác toàn cục (Perceptual near-duplicate dHash <= 4 & aHash <= 4 & chung lớp biển báo)
    for i in range(n_imgs):
        ri = records[i]
        set_i = set(ri["classes"])
        if not set_i:
            continue
        for j in range(i + 3, n_imgs):
            rj = records[j]
            if ri["classes"] != rj["classes"]:
                continue
            if hamming_dist(ri["dhash"], rj["dhash"]) <= 4 and hamming_dist(ri["ahash"], rj["ahash"]) <= 4:
                if uf.find(i) != uf.find(j):
                    uf.union(i, j)
                    perceptual_merges += 1

    groups_dict = defaultdict(list)
    for i in range(n_imgs):
        root = uf.find(i)
        groups_dict[root].append(i)

    # Sắp xếp và đánh chỉ số lại các primary groups
    sorted_group_roots = sorted(groups_dict.keys(), key=lambda r_idx: (-len(groups_dict[r_idx]), r_idx))
    primary_groups = []
    idx_to_group_id = {}

    for gid, root in enumerate(sorted_group_roots):
        member_indices = sorted(groups_dict[root])
        for m_idx in member_indices:
            idx_to_group_id[m_idx] = gid
            records[m_idx]["group_id"] = gid
        cls_union = sorted(list(set(c for m_idx in member_indices for c in records[m_idx]["classes"])))
        primary_groups.append({
            "group_id": gid,
            "size": len(member_indices),
            "member_indices": member_indices,
            "image_ids": [records[m_idx]["image_id"] for m_idx in member_indices],
            "classes": cls_union,
        })

    n_groups = len(primary_groups)
    mean_group_size = n_imgs / n_groups
    max_group_size = max(g["size"] for g in primary_groups)
    multi_member_groups = sum(1 for g in primary_groups if g["size"] > 1)

    print(f"[*] Kết quả gom nhóm bắc cầu (Connected Components):")
    print(f"    - Ghép do trùng MD5 tuyệt đối: {exact_dup_merges}")
    print(f"    - Ghép do chuỗi khung hình liên tiếp (Temporal burst): {temporal_burst_merges}")
    print(f"    - Ghép do tương đồng thị giác (Perceptual near-duplicate): {perceptual_merges}")
    print(f"[✓] Tổng số Nhóm Sơ cấp Độc lập (Primary Groups): {n_groups} nhóm trên {n_imgs} ảnh (trung bình {mean_group_size:.2f} ảnh/nhóm, lớn nhất {max_group_size} ảnh, {multi_member_groups} nhóm đa khung hình).")

    # Kiểm tra số lượng nhóm độc lập hỗ trợ từng lớp (Group support per class)
    class_group_support = defaultdict(set)
    for g in primary_groups:
        for c in g["classes"]:
            class_group_support[c].add(g["group_id"])

    rare_group_classes = [
        (c, classes[c], classes_vie[c], len(class_group_support[c]), global_cls_freq[c])
        for c in range(len(classes))
        if len(class_group_support[c]) < 5
    ]
    print(f"[*] Các lớp có số nhóm độc lập thấp (< 5 groups): {len(rare_group_classes)} lớp:")
    for c_id, c_code, c_vie, g_cnt, img_cnt in sorted(rare_group_classes, key=lambda x: x[3]):
        print(f"    - Class {c_id:02d} ({c_code} - {c_vie}): {g_cnt} nhóm độc lập ({img_cnt} ảnh)")

    print("\n" + "=" * 72)
    print(" BƯỚC 3: PHÂN CHIA GROUP-DISJOINT PARTITION 70/10/20 (MỤC 3.4)")
    print("=" * 72)

    # Mục tiêu kích thước: Train ~70%, Val ~10%, Test ~20% (643 ảnh test)
    target_test = int(round(n_imgs * 0.20))  # 643
    target_val = int(round(n_imgs * 0.10))   # 322
    target_train = n_imgs - target_test - target_val  # 2251

    rng = np.random.default_rng(42)
    group_assignment = {}  # gid -> 'train' | 'val' | 'test'
    subset_counts = {"train": 0, "val": 0, "test": 0}
    subset_class_counts = {"train": Counter(), "val": Counter(), "test": Counter()}

    def assign_group(gid: int, split_name: str):
        group_assignment[gid] = split_name
        g = primary_groups[gid]
        subset_counts[split_name] += g["size"]
        for m_idx in g["member_indices"]:
            for c in records[m_idx]["classes"]:
                subset_class_counts[split_name][c] += 1

    # Bước 3.1: Xử lý từ lớp hiếm nhất (ít nhóm độc lập nhất) đến lớp phổ biến nhất
    classes_by_rarity = sorted(range(len(classes)), key=lambda c: (len(class_group_support[c]), global_cls_freq[c]))

    for c in classes_by_rarity:
        avail_gids = [gid for gid in class_group_support[c] if gid not in group_assignment]
        # Sắp xếp nhóm nhỏ trước để không làm lệch tỷ lệ khi phân bổ cho test/val
        avail_gids.sort(key=lambda gid: (primary_groups[gid]["size"], gid))

        # Đảm bảo có mặt trong Train trước
        if subset_class_counts["train"][c] == 0 and avail_gids:
            # Ưu tiên nhóm lớn hơn cho train
            gid_train = avail_gids.pop(-1)
            assign_group(gid_train, "train")

        # Đảm bảo có mặt trong Test
        if subset_class_counts["test"][c] == 0 and avail_gids:
            if subset_counts["test"] < target_test:
                gid_test = avail_gids.pop(0)
                assign_group(gid_test, "test")

        # Nếu lớp có >= 3 nhóm độc lập, đảm bảo có mặt trong Val
        if len(class_group_support[c]) >= 3 and subset_class_counts["val"][c] == 0 and avail_gids:
            if subset_counts["val"] < target_val:
                gid_val = avail_gids.pop(0)
                assign_group(gid_val, "val")

    # Bước 3.2: Phân bổ các nhóm còn lại theo tỷ lệ 70/10/20 có kiểm soát kích thước
    remaining_gids = [g["group_id"] for g in primary_groups if g["group_id"] not in group_assignment]
    rng.shuffle(remaining_gids)

    for gid in remaining_gids:
        g_size = primary_groups[gid]["size"]
        if subset_counts["test"] + g_size <= target_test:
            assign_group(gid, "test")
        elif subset_counts["val"] + g_size <= target_val:
            assign_group(gid, "val")
        else:
            assign_group(gid, "train")

    # Kiểm tra điều kiện bất khả xâm phạm: 0 group crossing
    group_splits_check = defaultdict(set)
    for r in records:
        sp = group_assignment[r["group_id"]]
        r["grouped_split"] = sp
        group_splits_check[r["group_id"]].add(sp)

    crossing_grouped = sum(1 for spset in group_splits_check.values() if len(spset) > 1)
    assert crossing_grouped == 0, f"Lỗi nghiêm trọng: Có {crossing_grouped} nhóm bị cắt ngang!"

    # Tính mã băm SHA-256 cố định cho Group-Disjoint Partition
    manifest_str = json.dumps(
        {r["image_id"]: r["grouped_split"] for r in sorted(records, key=lambda x: x["image_id"])},
        sort_keys=True,
    )
    split_hash = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()

    print(f"[✓] Phân chia Group-Disjoint hoàn tất (0/{n_groups} nhóm bị cắt ngang):")
    print(f"    - Train:      {subset_counts['train']:4d} ảnh ({subset_counts['train']/n_imgs*100:.1f}%) | Số lớp hiện diện: {len(subset_class_counts['train'])}/52")
    print(f"    - Validation: {subset_counts['val']:4d} ảnh ({subset_counts['val']/n_imgs*100:.1f}%) | Số lớp hiện diện: {len(subset_class_counts['val'])}/52")
    print(f"    - Test:       {subset_counts['test']:4d} ảnh ({subset_counts['test']/n_imgs*100:.1f}%) | Số lớp hiện diện: {len(subset_class_counts['test'])}/52")
    print(f"    - SHA-256 Split Hash: {split_hash[:16]}...{split_hash[-8:]}")

    print("\n" + "=" * 72)
    print(" BƯỚC 4: NHÁNH ĐỐI CHỨNG NGẪU NHIÊN (REFERENCE ARM - MỤC 3.5)")
    print("=" * 72)

    # 4.1 Kiểm tra bộ chia ngẫu nhiên gốc trong archive/split_dataset
    orig_train_files = set()
    orig_test_files = set()
    if (ARCHIVE_DIR / "split_dataset" / "train_files.txt").exists():
        with open(ARCHIVE_DIR / "split_dataset" / "train_files.txt", "r", encoding="utf-8") as f:
            orig_train_files = set(l.strip() for l in f if l.strip())
    if (ARCHIVE_DIR / "split_dataset" / "test_files.txt").exists():
        with open(ARCHIVE_DIR / "split_dataset" / "test_files.txt", "r", encoding="utf-8") as f:
            orig_test_files = set(l.strip() for l in f if l.strip())

    orig_group_splits = defaultdict(set)
    for r in records:
        if r["filename"] in orig_train_files:
            orig_group_splits[r["group_id"]].add("train")
        elif r["filename"] in orig_test_files:
            orig_group_splits[r["group_id"]].add("test")
    orig_crossing_groups = sum(1 for spset in orig_group_splits.values() if len(spset) > 1)
    print(f"[*] Kiểm toán bộ chia gốc (archive/split_dataset): {orig_crossing_groups}/{multi_member_groups} nhóm đa khung hình ({orig_crossing_groups/max(1,multi_member_groups)*100:.1f}%) bị rò rỉ giữa Train và Test!")

    # 4.2 Xây dựng Nhánh Đối chứng Ngẫu nhiên Phân tầng (Stratified Random Image-Level Split)
    # có cùng kích thước tập Test (643 ảnh) với nhánh Group-Disjoint để so sánh bắt cặp công bằng (Bảng 2 & Bảng 3 của bài báo)
    rng_rand = np.random.default_rng(42)
    by_prim_cls = defaultdict(list)
    for r in records:
        by_prim_cls[r["primary_class"]].append(r["idx"])

    rand_split_map = {}
    rand_test_indices = []
    rand_val_indices = []
    rand_train_indices = []

    for c, idxs in by_prim_cls.items():
        idxs_arr = list(idxs)
        rng_rand.shuffle(idxs_arr)
        n_c = len(idxs_arr)
        n_te = max(1, int(round(n_c * 0.20))) if n_c >= 2 else 0
        n_va = max(1, int(round(n_c * 0.10))) if n_c >= 4 else 0
        rand_test_indices.extend(idxs_arr[:n_te])
        rand_val_indices.extend(idxs_arr[n_te : n_te + n_va])
        rand_train_indices.extend(idxs_arr[n_te + n_va :])

    # Cân bằng chính xác số lượng mẫu Test bằng đúng subset_counts["test"] (như Bảng 2 bài báo)
    while len(rand_test_indices) > subset_counts["test"]:
        moved = rand_test_indices.pop()
        rand_train_indices.append(moved)
    while len(rand_test_indices) < subset_counts["test"]:
        moved = rand_train_indices.pop()
        rand_test_indices.append(moved)

    for i in rand_train_indices:
        rand_split_map[i] = "train"
    for i in rand_val_indices:
        rand_split_map[i] = "val"
    for i in rand_test_indices:
        rand_split_map[i] = "test"

    rand_group_splits = defaultdict(set)
    for r in records:
        sp = rand_split_map[r["idx"]]
        r["random_split"] = sp
        rand_group_splits[r["group_id"]].add(sp)

    rand_crossing_all = sum(1 for spset in rand_group_splits.values() if len(spset) > 1)
    rand_crossing_train_test = sum(
        1 for spset in rand_group_splits.values() if "train" in spset and "test" in spset
    )

    if rand_crossing_all == 0:
        raise RuntimeError("Lỗi: Nhánh ngẫu nhiên không tạo ra group crossing!")

    print(f"[✓] Nhánh Đối chứng Ngẫu nhiên (Random Image-Level Partition):")
    print(f"    - Train: {len(rand_train_indices)} | Val: {len(rand_val_indices)} | Test: {len(rand_test_indices)} (khớp tuyệt đối kích thước Test = {subset_counts['test']})")
    print(f"    - Số nhóm bị cắt ngang qua biên (Group crossing): {rand_crossing_all}/{n_groups} tổng nhóm ({rand_crossing_all/max(1,multi_member_groups)*100:.1f}% nhóm đa khung hình), trong đó {rand_crossing_train_test} nhóm nằm đồng thời ở cả Train và Test!")

    # ==============================================================================
    # BƯỚC 5: TÍNH THỐNG KÊ CHUẨN HÓA PER-CHANNEL TRAIN-ONLY CHO 2 NHÁNH (BẢNG 3)
    # ==============================================================================
    def compute_arm_norm_stats(split_key: str):
        train_recs = [r for r in records if r[split_key] == "train"]
        mean_vec = np.mean([r["rgb_mean"] for r in train_recs], axis=0)
        sq_mean_vec = np.mean([r["rgb_sq_mean"] for r in train_recs], axis=0)
        std_vec = np.sqrt(np.maximum(sq_mean_vec - mean_vec ** 2, 1e-6))
        return len(train_recs), mean_vec.tolist(), std_vec.tolist()

    n_grp_tr, grp_mean, grp_std = compute_arm_norm_stats("grouped_split")
    n_rnd_tr, rnd_mean, rnd_std = compute_arm_norm_stats("random_split")

    print("\n[*] Thống kê chuẩn hóa Per-Channel Train-Only (tương ứng Bảng 3 trong bài báo):")
    ch_names = ["Red (R)", "Green (G)", "Blue (B)"]
    for ch_i, ch_name in enumerate(ch_names):
        print(
            f"    - {ch_name:9s} | Grouped Arm (n={n_grp_tr}): Mean={grp_mean[ch_i]:.4f}, SD={grp_std[ch_i]:.4f} "
            f"| Random Arm (n={n_rnd_tr}): Mean={rnd_mean[ch_i]:.4f}, SD={rnd_std[ch_i]:.4f}"
        )

    # Thống kê theo 5 siêu nhóm biển báo (P: Biển cấm, W: Biển cảnh báo, R: Biển hiệu lệnh, I: Biển chỉ dẫn, S: Biển phụ)
    def get_supergroup(code_str: str) -> str:
        if code_str.startswith("P.") or code_str.startswith("DP.") or code_str.startswith("B."):
            return "P (Prohibitory)"
        elif code_str.startswith("W."):
            return "W (Warning)"
        elif code_str.startswith("R."):
            return "R (Mandatory)"
        elif code_str.startswith("I.") or code_str == "Camera":
            return "I (Informatory)"
        elif code_str.startswith("S."):
            return "S (Auxiliary)"
        return "Other"

    composition_table = {"grouped": defaultdict(Counter), "random": defaultdict(Counter)}
    for r in records:
        prim_c = r["primary_class"]
        sg = get_supergroup(classes[prim_c]) if prim_c >= 0 else "Other"
        composition_table["grouped"][r["grouped_split"]][sg] += 1
        composition_table["random"][r["random_split"]][sg] += 1

    # Lưu kết quả kiểm toán và phân chia vào data/leakage_audit_report.json và data/split_manifest.json
    audit_report = {
        "cohort_size": n_imgs,
        "total_bounding_boxes": total_boxes,
        "num_classes": len(classes),
        "primary_groups_count": n_groups,
        "multi_frame_groups_count": multi_member_groups,
        "mean_frames_per_group": round(mean_group_size, 3),
        "max_frames_per_group": max_group_size,
        "merge_counts": {
            "exact_md5_duplicates": exact_dup_merges,
            "temporal_burst_sequences": temporal_burst_merges,
            "perceptual_near_duplicates": perceptual_merges,
        },
        "split_hash_sha256": split_hash,
        "grouped_partition": {
            "groups_crossing": crossing_grouped,
            "train_count": subset_counts["train"],
            "val_count": subset_counts["val"],
            "test_count": subset_counts["test"],
            "train_classes_present": len(subset_class_counts["train"]),
            "val_classes_present": len(subset_class_counts["val"]),
            "test_classes_present": len(subset_class_counts["test"]),
            "supergroup_composition": {k: dict(v) for k, v in composition_table["grouped"].items()},
        },
        "random_partition": {
            "groups_crossing_total": rand_crossing_all,
            "groups_crossing_train_test": rand_crossing_train_test,
            "original_archive_split_crossing": orig_crossing_groups,
            "train_count": len(rand_train_indices),
            "val_count": len(rand_val_indices),
            "test_count": len(rand_test_indices),
            "supergroup_composition": {k: dict(v) for k, v in composition_table["random"].items()},
        },
        "train_only_normalisation_stats": {
            "grouped_arm": {"n": n_grp_tr, "mean": [round(x, 4) for x in grp_mean], "std": [round(x, 4) for x in grp_std]},
            "random_arm": {"n": n_rnd_tr, "mean": [round(x, 4) for x in rnd_mean], "std": [round(x, 4) for x in rnd_std]},
            "imagenet_default": {"mean": [0.4850, 0.4560, 0.4060], "std": [0.2290, 0.2240, 0.2250]},
        },
        "rare_classes_group_support": [
            {
                "class_id": c_id,
                "code": c_code,
                "name_vi": c_vie,
                "independent_groups": g_cnt,
                "total_images": img_cnt,
                "train_support_grouped": subset_class_counts["train"][c_id],
                "val_support_grouped": subset_class_counts["val"][c_id],
                "test_support_grouped": subset_class_counts["test"][c_id],
            }
            for c_id, c_code, c_vie, g_cnt, img_cnt in sorted(rare_group_classes, key=lambda x: x[3])
        ],
    }

    with open(DATA_DIR / "leakage_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(audit_report, f, ensure_ascii=False, indent=2)

    split_manifest = {
        "split_hash_sha256": split_hash,
        "records": [
            {
                "image_id": r["image_id"],
                "filename": r["filename"],
                "group_id": r["group_id"],
                "primary_class": r["primary_class"],
                "classes": r["classes"],
                "grouped_split": r["grouped_split"],
                "random_split": r["random_split"],
            }
            for r in records
        ],
    }
    with open(DATA_DIR / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[✓] Đã lưu báo cáo kiểm toán rò rỉ tại: {DATA_DIR / 'leakage_audit_report.json'}")
    print(f"[✓] Đã lưu manifest phân tập dữ liệu tại: {DATA_DIR / 'split_manifest.json'}")


if __name__ == "__main__":
    run_audit()
