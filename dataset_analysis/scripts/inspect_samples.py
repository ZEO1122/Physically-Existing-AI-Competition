import os
import argparse
import random
import shutil
from collections import Counter

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

IMG_NAMES = ["front", "top"]
IMG_EXTS = ["png", "jpg", "jpeg", "bmp", "webp"]
VIDEO_EXTS = ["mp4", "mov", "avi", "mkv", "webm", "gif"]


def set_seed(seed: int):
    random.seed(seed)


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def list_subdirs(path: str):
    if not os.path.isdir(path):
        return []
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def infer_folder_id_width(split_dir: str):
    """
    split_dir 안 폴더명이 숫자 zfill(예: 000123) 형태이면 폭(width)을 추정한다.
    """
    dirs = list_subdirs(split_dir)
    if not dirs:
        return None

    numeric_dirs = [d for d in dirs if d.isdigit()]
    if len(numeric_dirs) < max(10, len(dirs) // 2):
        return None

    lengths = [len(d) for d in numeric_dirs]
    most_common_len, cnt = Counter(lengths).most_common(1)[0]
    if cnt / len(numeric_dirs) >= 0.8:
        return most_common_len
    return None


def resolve_sample_folder(split_dir: str, sample_id):
    """
    CSV의 id → 실제 폴더명 매칭.
    - 기본: str(id)
    - 실패하면, split_dir 안 폴더명을 보고 zfill 폭을 추정해서 다시 시도
    """
    sid = str(sample_id)
    cand = os.path.join(split_dir, sid)
    if os.path.isdir(cand):
        return cand, sid

    width = infer_folder_id_width(split_dir)
    if width is not None and sid.isdigit():
        sid2 = sid.zfill(width)
        cand2 = os.path.join(split_dir, sid2)
        if os.path.isdir(cand2):
            return cand2, sid2

    return None, sid


def find_image_path(folder_path: str, name: str):
    for ext in IMG_EXTS:
        p = os.path.join(folder_path, f"{name}.{ext}")
        if os.path.isfile(p):
            return p
    return None


def find_video_files(folder_path: str):
    vids = []
    if not os.path.isdir(folder_path):
        return vids

    for fn in os.listdir(folder_path):
        lower = fn.lower()
        for ext in VIDEO_EXTS:
            if lower.endswith("." + ext):
                vids.append(os.path.join(folder_path, fn))
                break
    vids.sort()
    return vids


def load_tile(path: str, tile: int):
    """
    원본 비율 유지(contain) + 흰 배경 패딩으로 tile x tile 맞춤
    """
    img = Image.open(path).convert("RGB")
    img = ImageOps.contain(img, (tile, tile))
    canvas = Image.new("RGB", (tile, tile), (255, 255, 255))
    canvas.paste(img, ((tile - img.width) // 2, (tile - img.height) // 2))
    return canvas


def make_pair_image(front_path: str, top_path: str, text: str, out_path: str, tile: int = 256, pad: int = 12):
    """
    front/top 2장을 나란히 붙이고 아래에 텍스트를 넣어 저장
    """
    font = ImageFont.load_default()
    text_h = 18  # default font 기준

    front = load_tile(front_path, tile)
    top = load_tile(top_path, tile)

    w = pad + tile + pad + tile + pad
    h = pad + tile + pad + text_h + pad
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x1 = pad
    x2 = pad + tile + pad
    y1 = pad

    canvas.paste(front, (x1, y1))
    canvas.paste(top, (x2, y1))

    # 라벨 텍스트
    draw.text((pad, pad + tile + 6), text, fill=(0, 0, 0), font=font)

    canvas.save(out_path)


def write_index_html(rows, out_dir: str, html_name: str = "index.html"):
    """
    rows: list of dict with keys: label, id, pair_path_rel, videos_rel(list)
    """
    html_path = os.path.join(out_dir, html_name)

    def esc(s: str):
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    stable_rows = [r for r in rows if r["label"] == "stable"]
    unstable_rows = [r for r in rows if r["label"] == "unstable"]

    def section(title, items):
        parts = []
        parts.append(f"<h2>{esc(title)}</h2>")
        parts.append("<div style='display:flex; flex-wrap:wrap; gap:16px;'>")
        for r in items:
            vids = ""
            if r["videos_rel"]:
                links = []
                for vp in r["videos_rel"]:
                    links.append(f"<a href='{esc(vp)}'>video</a>")
                vids = " | " + " ".join(links)
            parts.append(
                "<div style='border:1px solid #ddd; padding:10px; width:560px;'>"
                f"<div><b>id:</b> {esc(r['id'])} {vids}</div>"
                f"<div><img src='{esc(r['pair_path_rel'])}' style='max-width:540px; height:auto;'/></div>"
                "</div>"
            )
        parts.append("</div>")
        return "\n".join(parts)

    html = []
    html.append("<!doctype html>")
    html.append("<html><head><meta charset='utf-8'/>")
    html.append("<title>Sample Inspection</title></head><body>")
    html.append("<h1>Train Samples (front/top pairs)</h1>")
    html.append("<p>각 카드에서 이미지(pair)는 저장된 결과이고, video 링크가 있으면 원본 폴더의 영상 파일을 가리킵니다.</p>")
    html.append(section("Stable", stable_rows))
    html.append("<hr/>")
    html.append(section("Unstable", unstable_rows))
    html.append("</body></html>")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    return html_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./open", help="dataset root (default: ./open)")
    parser.add_argument("--out", type=str, default="./inspect_samples", help="output dir (default: ./inspect_samples)")
    parser.add_argument("--n", type=int, default=5, help="samples per class (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--tile", type=int, default=256, help="tile size for each view image")
    args = parser.parse_args()

    set_seed(args.seed)
    safe_mkdir(args.out)

    train_csv = os.path.join(args.root, "train.csv")
    train_dir = os.path.join(args.root, "train")

    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f"train.csv not found: {train_csv}")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"train directory not found: {train_dir}")

    df = pd.read_csv(train_csv)
    if "id" not in df.columns or "label" not in df.columns:
        raise ValueError(f"train.csv must have columns ['id','label'], got: {list(df.columns)}")

    # 라벨 분포 출력
    print("label distribution:")
    print(df["label"].value_counts(dropna=False))

    # stable/unstable 샘플링
    rows_out = []
    picked = []

    for label in ["stable", "unstable"]:
        sub = df[df["label"] == label].copy()
        if len(sub) == 0:
            print(f"[WARN] no rows for label={label}")
            continue

        k = min(args.n, len(sub))
        # random sample
        sampled = sub.sample(n=k, random_state=args.seed)

        for _, r in sampled.iterrows():
            sid = r["id"]
            folder, resolved = resolve_sample_folder(train_dir, sid)
            if folder is None:
                print(f"[SKIP] folder not found for id={sid}")
                continue

            front_path = find_image_path(folder, "front")
            top_path = find_image_path(folder, "top")
            if front_path is None or top_path is None:
                print(f"[SKIP] missing front/top for id={sid} (folder={folder})")
                continue

            videos = find_video_files(folder)
            # out paths
            label_dir = os.path.join(args.out, label)
            safe_mkdir(label_dir)

            pair_name = f"{str(sid)}_pair.png"
            pair_path = os.path.join(label_dir, pair_name)

            text = f"label={label} | id={sid} (resolved={resolved})"
            if videos:
                text += f" | videos={len(videos)}"

            try:
                make_pair_image(front_path, top_path, text, pair_path, tile=args.tile)
            except Exception as e:
                print(f"[SKIP] failed to make pair image for id={sid}: {e}")
                continue

            # 상대경로(HTML에서 클릭 가능하게)
            pair_rel = os.path.relpath(pair_path, args.out)
            videos_rel = [os.path.relpath(v, args.out) for v in videos]

            rows_out.append({
                "label": label,
                "id": sid,
                "resolved_id": resolved,
                "folder": folder,
                "front_path": front_path,
                "top_path": top_path,
                "pair_path": pair_path,
                "pair_path_rel": pair_rel,
                "videos": ";".join(videos),
            })
            picked.append({
                "label": label,
                "id": sid,
                "pair_path_rel": pair_rel,
                "videos_rel": videos_rel[:3],  # 너무 많으면 3개까지만 링크
            })

    # CSV로 저장
    out_csv = os.path.join(args.out, "picked_samples.csv")
    pd.DataFrame(rows_out).to_csv(out_csv, index=False, encoding="utf-8-sig")

    # HTML 인덱스 생성
    html_path = write_index_html(picked, args.out, html_name="index.html")

    print("\nDONE ✅")
    print(" - saved pairs & index at:", args.out)
    print(" - picked_samples.csv:", out_csv)
    print(" - index.html:", html_path)
    print("\n열어보기:")
    print(f"  xdg-open {html_path}")


if __name__ == "__main__":
    main()