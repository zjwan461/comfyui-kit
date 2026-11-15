#!/usr/bin/env python3
"""把 custom_nodes 里"没有公开仓库"的节点包打成 vendor 压缩包。

用法（在任意一台装好的机器上跑，用于更新仓库里的 vendor/）：

    python tools/pack_nodes.py --custom-nodes E:\\ComfyUI\\custom_nodes

产物：
    vendor/nodes/<包名>.zip              每个包一个 zip
    vendor/nodes/<包名>.zip.001 ...      超过 --part-mb 的包按字节切成多卷
    vendor/manifest.json                 每个包的文件名、字节数、sha256、文件数

install.py 解压时会先按顺序把多卷拼回 zip，再解到 custom_nodes/。
接口按 ComfyUI 的约定：包目录名 = custom_nodes 下的目录名，zip 内不带顶层目录。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile

# 运行期不需要的东西：版本库元数据、字节码、虚拟环境、前端依赖
SKIP_DIR = {".git", "__pycache__", "node_modules", ".venv", "venv", ".idea", ".vscode"}
SKIP_EXT = {".pyc", ".pyo", ".pyd.old"}
# 纯演示素材：README 截图、预览动画、宣传视频（--keep-media 可保留）
MEDIA_EXT = {".mp4", ".webm", ".mov", ".gif", ".pdf"}
MEDIA_DIR_HINT = ("assets", "asset", "docs", "doc", "preview", "previews", "example", "examples", "demo", "demos")


def is_media(path: str, root: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    if ext not in MEDIA_EXT:
        return False
    rel = os.path.relpath(path, root).lower().replace("\\", "/")
    parts = rel.split("/")[:-1]
    return any(h in p for p in parts for h in MEDIA_DIR_HINT)


def walk_pack(root: str, keep_media: bool):
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d not in SKIP_DIR)
        for f in sorted(fn):
            if os.path.splitext(f)[1].lower() in SKIP_EXT:
                continue
            p = os.path.join(dp, f)
            if not keep_media and is_media(p, root):
                continue
            yield p, os.path.relpath(p, root).replace("\\", "/")


def zip_pack(root: str, out_zip: str, keep_media: bool) -> tuple[int, int]:
    """写 zip（固定时间戳，保证同样输入产出同样字节）。返回 (文件数, 解压后字节数)。"""
    n = 0
    total = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path, rel in walk_pack(root, keep_media):
            zi = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            with open(path, "rb") as f:
                data = f.read()
            z.writestr(zi, data)
            n += 1
            total += len(data)
    return n, total


def split_file(path: str, part_bytes: int) -> list[str]:
    size = os.path.getsize(path)
    if size <= part_bytes:
        return [path]
    parts = []
    with open(path, "rb") as f:
        i = 1
        while True:
            chunk = f.read(part_bytes)
            if not chunk:
                break
            p = f"{path}.{i:03d}"
            with open(p, "wb") as g:
                g.write(chunk)
            parts.append(p)
            i += 1
    os.remove(path)
    return parts


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="打包 vendor 节点包")
    ap.add_argument("--custom-nodes", required=True, help="custom_nodes 目录")
    ap.add_argument("--config", default=os.path.join("config", "nodes.json"))
    ap.add_argument("--out", default=os.path.join("vendor", "nodes"))
    ap.add_argument("--part-mb", type=int, default=40, help="单卷上限（MB），超过就切卷")
    ap.add_argument("--keep-media", action="store_true", help="保留演示用视频/GIF/PDF")
    ap.add_argument("--only", default="", help="只打包这些包（逗号分隔）")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    os.makedirs(args.out, exist_ok=True)
    manifest = {}
    total_bytes = 0

    for pack in cfg["packs"]:
        name = pack["name"]
        if pack.get("kind") != "vendor" or (only and name not in only):
            continue
        src = os.path.join(args.custom_nodes, name)
        if not os.path.isdir(src):
            print(f"  [skip] {name}: 本地没有")
            continue
        tmp = os.path.join(args.out, name + ".zip")
        n, raw = zip_pack(src, tmp, args.keep_media)
        parts = split_file(tmp, args.part_mb * 1024 * 1024)
        files = []
        for p in parts:
            files.append({"file": os.path.basename(p), "bytes": os.path.getsize(p), "sha256": sha256(p)})
            total_bytes += os.path.getsize(p)
        manifest[name] = {"files": files, "raw_bytes": raw, "entries": n}
        print(f'  {name:45s} {n:5d} files  {raw/2**20:7.1f}MB raw -> {sum(x["bytes"] for x in files)/2**20:6.1f}MB '
              f'({len(files)} 卷)')

    with open(os.path.join("vendor", "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"\n{len(manifest)} 个包，压缩后合计 {total_bytes/2**20:.1f} MB -> vendor/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())