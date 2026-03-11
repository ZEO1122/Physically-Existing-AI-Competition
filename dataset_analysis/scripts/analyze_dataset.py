# analyze_dataset.py
import os
import json
import argparse
import random
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# tqdm은 있으면 쓰고, 없으면 그냥 진행
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

IMG_NAMES = ["front", "top"]
IMG_EXTS = ["png", "jpg", "jpeg", "bmp", "webp"]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def list_subdirs(path: str):
    if not os.path.isdir(path):
        return []
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

def infer_folder_id_width(split_dir: str):
    """
    split_dir 안의 폴더명을 보고 (숫자형이고 길이가 일정하면) zfill 길이를 추정.
    예: 000123 같은 폴더가 많으면 width=6
    """
    dirs = list_subdirs(split_dir)
    if not dirs:
        return None

    # 숫자 폴더만
    numeric_dirs = [d for d in dirs if d.isdigit()]
    if len(numeric_dirs) < max(10, len(dirs) // 2):
        return None  # 숫자 폴더가 충분히 많지 않으면 추정 안 함

    lengths = [len(d) for d in numeric_dirs]
    most_common_len, cnt = Counter(lengths).most_common(1)[0]
    # 충분히 지배적이면 채택
    if cnt / len(numeric_dirs) >= 0.8:
        return most_common_len
    return None

def resolve_sample_folder(root_split_dir: str, sample_id):
    """
    CSV의 id를 폴더명으로 매칭.
    - 기본: str(id)
    - 폴더가 숫자 zfill 형태면 infer된 width로 zfill도 시도
    """
    sid = str(sample_id)
    cand = os.path.join(root_split_dir, sid)
    if os.path.isdir(cand):
        return cand, sid

    width = infer_folder_id_width(root_split_dir)
    if width is not None and sid.isdigit():
        sid2 = sid.zfill(width)
        cand2 = os.path.join(root_split_dir, sid2)
        if os.path.isdir(cand2):
            return cand2, sid2

    # 그래도 못 찾으면 None
    return None, sid

def find_image_path(folder_path: str, name: str):
    """
    folder_path 내부에서 name.{ext} 찾기 (png 우선)
    """
    for ext in IMG_EXTS:
        p = os.path.join(folder_path, f"{name}.{ext}")
        if os.path.isfile(p):
            return p
    return None

def check_integrity(df: pd.DataFrame, split_dir: str, is_test: bool, max_check: int = 1000):
    """
    폴더 존재 여부 + front/top 존재 여부 검사
    """
    n = min(len(df), max_check)
    missing = []
    ok_count = 0

    for i in tqdm(range(n), desc=f"Integrity check ({os.path.basename(split_dir)})"):
        sid = df.iloc[i]["id"]
        folder, resolved_id = resolve_sample_folder(split_dir, sid)
        if folder is None:
            missing.append({
                "id": sid,
                "resolved_id": resolved_id,
                "folder_exists": False,
                "front_exists": False,
                "top_exists": False,
                "label": None if is_test else df.iloc[i].get("label", None)
            })
            continue

        front_path = find_image_path(folder, "front")
        top_path   = find_image_path(folder, "top")

        front_ok = front_path is not None
        top_ok = top_path is not None
        if front_ok and top_ok:
            ok_count += 1
        else:
            missing.append({
                "id": sid,
                "resolved_id": resolved_id,
                "folder_exists": True,
                "front_exists": front_ok,
                "top_exists": top_ok,
                "label": None if is_test else df.iloc[i].get("label", None)
            })

    return {
        "checked": n,
        "ok_count": ok_count,
        "missing_count": len(missing),
        "missing_examples": missing[:20],  # 예시만
        "missing_rows": missing            # 전체는 파일로 저장
    }

def image_stats_for_split(df: pd.DataFrame, split_dir: str, is_test: bool, n_stats: int = 300, img_size: int = 224):
    """
    랜덤 샘플 n_stats에 대해 front/top 각각:
    - 원본 크기 분포
    - (리사이즈 후) 채널별 평균/표준편차
    """
    if len(df) == 0:
        return {}

    idxs = list(range(len(df)))
    random.shuffle(idxs)
    idxs = idxs[:min(n_stats, len(idxs))]

    # 누적 통계
    size_counter = {name: Counter() for name in IMG_NAMES}
    ch_sum = {name: np.zeros(3, dtype=np.float64) for name in IMG_NAMES}
    ch_sq  = {name: np.zeros(3, dtype=np.float64) for name in IMG_NAMES}
    count_pixels = {name: 0 for name in IMG_NAMES}
    used = 0
    failed = 0

    for i in tqdm(idxs, desc=f"Image stats ({os.path.basename(split_dir)})"):
        sid = df.iloc[i]["id"]
        folder, _ = resolve_sample_folder(split_dir, sid)
        if folder is None:
            failed += 1
            continue

        paths = {}
        ok = True
        for name in IMG_NAMES:
            p = find_image_path(folder, name)
            if p is None:
                ok = False
                break
            paths[name] = p
        if not ok:
            failed += 1
            continue

        for name in IMG_NAMES:
            try:
                img = Image.open(paths[name]).convert("RGB")
                w, h = img.size
                size_counter[name][(w, h)] += 1

                # 통계는 일정 크기(img_size)로 리사이즈해서 계산(비교 가능하게)
                img_r = img.resize((img_size, img_size))
                arr = np.asarray(img_r, dtype=np.float32) / 255.0  # [H,W,3]
                arr = arr.reshape(-1, 3)
                ch_sum[name] += arr.sum(axis=0)
                ch_sq[name]  += (arr ** 2).sum(axis=0)
                count_pixels[name] += arr.shape[0]
            except Exception:
                failed += 1
                ok = False
                break

        if ok:
            used += 1

    out = {"used": used, "failed": failed, "n_stats_target": n_stats}
    for name in IMG_NAMES:
        n_pix = max(1, count_pixels[name])
        mean = ch_sum[name] / n_pix
        var = ch_sq[name] / n_pix - mean**2
        std = np.sqrt(np.maximum(var, 1e-12))
        common_sizes = size_counter[name].most_common(10)

        out[name] = {
            "channel_mean_rgb_resized": mean.tolist(),
            "channel_std_rgb_resized": std.tolist(),
            "most_common_original_sizes_top10": [
                {"size": [int(k[0]), int(k[1])], "count": int(v)} for k, v in common_sizes
            ],
            "unique_original_size_count": int(len(size_counter[name]))
        }
    return out

def save_sample_grid(df: pd.DataFrame, split_dir: str, out_path: str, is_test: bool,
                     grid_rows: int = 10, tile: int = 224, pad: int = 10):
    """
    matplotlib 없이도 볼 수 있게 PIL로 그리드 이미지 저장.
    각 row에 [front | top] 두 장을 배치하고, id/label을 텍스트로 적음.
    """
    if len(df) == 0:
        return False

    font = ImageFont.load_default()
    # row당 높이: tile + text area
    text_h = 18
    width = pad + (tile + pad) * 2
    height = pad + grid_rows * (tile + text_h + pad)

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # 랜덤 샘플
    idxs = list(range(len(df)))
    random.shuffle(idxs)
    idxs = idxs[:min(grid_rows, len(idxs))]

    y = pad
    used = 0
    for i in idxs:
        sid = df.iloc[i]["id"]
        label = None if is_test else df.iloc[i].get("label", None)

        folder, resolved = resolve_sample_folder(split_dir, sid)
        if folder is None:
            continue

        front_path = find_image_path(folder, "front")
        top_path   = find_image_path(folder, "top")
        if front_path is None or top_path is None:
            continue

        try:
            front = Image.open(front_path).convert("RGB").resize((tile, tile))
            top   = Image.open(top_path).convert("RGB").resize((tile, tile))
        except Exception:
            continue

        x1 = pad
        x2 = pad + tile + pad
        canvas.paste(front, (x1, y))
        canvas.paste(top,   (x2, y))

        text = f"id={sid} (resolved={resolved})"
        if label is not None:
            text += f" | label={label}"
        draw.text((pad, y + tile + 2), text, fill=(0, 0, 0), font=font)

        y += tile + text_h + pad
        used += 1

    canvas.save(out_path)
    return used > 0

def main():
    parser = argparse.ArgumentParser(description="Multi-view dataset analyzer (front/top).")
    parser.add_argument("--root", type=str, default="./open", help="dataset root directory (default: ./open)")
    parser.add_argument("--out", type=str, default="./dataset_report", help="output directory (default: ./dataset_report)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_check", type=int, default=2000, help="max rows to check integrity per split")
    parser.add_argument("--n_stats", type=int, default=400, help="number of samples to compute image stats per split")
    parser.add_argument("--grid_rows", type=int, default=12, help="rows in saved sample grid image")
    parser.add_argument("--tile", type=int, default=224, help="tile size for grid & stats resize")
    args = parser.parse_args()

    set_seed(args.seed)
    safe_mkdir(args.out)

    # ---- CSV load ----
    train_csv = os.path.join(args.root, "train.csv")
    dev_csv   = os.path.join(args.root, "dev.csv")
    sub_csv   = os.path.join(args.root, "sample_submission.csv")

    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f"train.csv not found: {train_csv}")
    if not os.path.isfile(dev_csv):
        raise FileNotFoundError(f"dev.csv not found: {dev_csv}")
    if not os.path.isfile(sub_csv):
        raise FileNotFoundError(f"sample_submission.csv not found: {sub_csv}")

    train_df = pd.read_csv(train_csv)
    dev_df   = pd.read_csv(dev_csv)
    test_df  = pd.read_csv(sub_csv)

    # ---- Split dirs ----
    train_dir = os.path.join(args.root, "train")
    dev_dir   = os.path.join(args.root, "dev")
    test_dir  = os.path.join(args.root, "test")

    report = {
        "root": args.root,
        "splits": {},
        "notes": []
    }

    def summarize_df(df: pd.DataFrame, name: str, is_test: bool):
        info = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
        }
        if (not is_test) and ("label" in df.columns):
            vc = df["label"].value_counts(dropna=False).to_dict()
            info["label_distribution"] = {str(k): int(v) for k, v in vc.items()}

            # pos_weight 힌트(unstable=1 이라고 가정)
            if "unstable" in vc and "stable" in vc and vc["unstable"] > 0:
                neg = vc.get("stable", 0)
                pos = vc.get("unstable", 0)
                info["pos_weight_hint_for_BCE"] = float(neg / pos)
        return info

    report["splits"]["train"] = summarize_df(train_df, "train", is_test=False)
    report["splits"]["dev"]   = summarize_df(dev_df, "dev", is_test=False)
    report["splits"]["test"]  = summarize_df(test_df, "test", is_test=True)

    print("\n==============================")
    print("1) CSV Summary")
    print("==============================")
    for k in ["train", "dev", "test"]:
        s = report["splits"][k]
        print(f"\n[{k}] rows={s['rows']}")
        print(" columns:", s["columns"])
        if "label_distribution" in s:
            print(" label_distribution:", s["label_distribution"])
            if "pos_weight_hint_for_BCE" in s:
                print(" pos_weight_hint_for_BCE (stable/unstable):", s["pos_weight_hint_for_BCE"])

    # ---- Integrity checks ----
    print("\n==============================")
    print("2) Integrity Check (folder/front/top 존재 여부)")
    print("==============================")
    integ_train = check_integrity(train_df, train_dir, is_test=False, max_check=args.max_check)
    integ_dev   = check_integrity(dev_df,   dev_dir,   is_test=False, max_check=args.max_check)
    integ_test  = check_integrity(test_df,  test_dir,  is_test=True,  max_check=args.max_check)

    report["splits"]["train"]["integrity"] = {
        "checked": integ_train["checked"],
        "ok_count": integ_train["ok_count"],
        "missing_count": integ_train["missing_count"],
        "missing_examples": integ_train["missing_examples"],
    }
    report["splits"]["dev"]["integrity"] = {
        "checked": integ_dev["checked"],
        "ok_count": integ_dev["ok_count"],
        "missing_count": integ_dev["missing_count"],
        "missing_examples": integ_dev["missing_examples"],
    }
    report["splits"]["test"]["integrity"] = {
        "checked": integ_test["checked"],
        "ok_count": integ_test["ok_count"],
        "missing_count": integ_test["missing_count"],
        "missing_examples": integ_test["missing_examples"],
    }

    def print_integ(name, integ):
        print(f"\n[{name}] checked={integ['checked']}, ok={integ['ok_count']}, missing={integ['missing_count']}")
        if integ["missing_examples"]:
            print(" missing_examples (up to 5):")
            for ex in integ["missing_examples"][:5]:
                print("  ", ex)

    print_integ("train", integ_train)
    print_integ("dev", integ_dev)
    print_integ("test", integ_test)

    # integrity missing rows 저장
    pd.DataFrame(integ_train["missing_rows"]).to_csv(os.path.join(args.out, "missing_train.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(integ_dev["missing_rows"]).to_csv(os.path.join(args.out, "missing_dev.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(integ_test["missing_rows"]).to_csv(os.path.join(args.out, "missing_test.csv"), index=False, encoding="utf-8-sig")

    # ---- Image stats (sampled) ----
    print("\n==============================")
    print("3) Image Stats (샘플링 기반, front/top 분리)")
    print("==============================")
    stats_train = image_stats_for_split(train_df, train_dir, is_test=False, n_stats=args.n_stats, img_size=args.tile)
    stats_dev   = image_stats_for_split(dev_df,   dev_dir,   is_test=False, n_stats=args.n_stats, img_size=args.tile)
    stats_test  = image_stats_for_split(test_df,  test_dir,  is_test=True,  n_stats=args.n_stats, img_size=args.tile)

    report["splits"]["train"]["image_stats_sampled"] = stats_train
    report["splits"]["dev"]["image_stats_sampled"]   = stats_dev
    report["splits"]["test"]["image_stats_sampled"]  = stats_test

    def print_stats(name, stats):
        if not stats:
            return
        print(f"\n[{name}] used={stats.get('used', 0)} failed={stats.get('failed', 0)}")
        for view in IMG_NAMES:
            if view in stats:
                m = stats[view]["channel_mean_rgb_resized"]
                s = stats[view]["channel_std_rgb_resized"]
                print(f"  - {view}: mean(RGB)={np.round(m, 4).tolist()} std(RGB)={np.round(s, 4).tolist()}")
                top_sizes = stats[view]["most_common_original_sizes_top10"][:3]
                if top_sizes:
                    print(f"    common sizes top3: {top_sizes}")

    print_stats("train", stats_train)
    print_stats("dev", stats_dev)
    print_stats("test", stats_test)

    # ---- Save sample grids ----
    print("\n==============================")
    print("4) Save Sample Grids (PIL로 저장, matplotlib 불필요)")
    print("==============================")
    grid_train_path = os.path.join(args.out, "grid_train.png")
    grid_dev_path   = os.path.join(args.out, "grid_dev.png")
    grid_test_path  = os.path.join(args.out, "grid_test.png")

    ok1 = save_sample_grid(train_df, train_dir, grid_train_path, is_test=False,
                           grid_rows=args.grid_rows, tile=args.tile)
    ok2 = save_sample_grid(dev_df,   dev_dir,   grid_dev_path,   is_test=False,
                           grid_rows=args.grid_rows, tile=args.tile)
    ok3 = save_sample_grid(test_df,  test_dir,  grid_test_path,  is_test=True,
                           grid_rows=args.grid_rows, tile=args.tile)

    print(" saved:", grid_train_path if ok1 else "(train grid failed)")
    print(" saved:", grid_dev_path if ok2 else "(dev grid failed)")
    print(" saved:", grid_test_path if ok3 else "(test grid failed)")

    # ---- Final report save ----
    report_path = os.path.join(args.out, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n==============================")
    print("DONE ✅")
    print("==============================")
    print("Output directory:", args.out)
    print(" - report.json: 데이터 요약/무결성/통계")
    print(" - missing_*.csv: 누락 샘플 목록")
    print(" - grid_*.png: 샘플 이미지 그리드")
    print("\nTip) grid_train.png를 열어 front/top이 실제로 어떤 뷰인지 먼저 눈으로 확인해봐.")

if __name__ == "__main__":
    main()