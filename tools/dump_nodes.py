#!/usr/bin/env python3
"""从一台装好的 ComfyUI 生成 config/nodes.json。

判定规则：
  · 有 .git 且 remote 是 github/gitcode → kind="git"（记下 remote url 与当前提交，装的时候钉住）
  · 否则 → kind="vendor"（必须用 tools/pack_nodes.py 打进 vendor/）

用法：
    python tools/dump_nodes.py --custom-nodes E:/ComfyUI/custom_nodes
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}
PUBLIC_HOSTS = ("github.com", "gitcode.com", "gitee.com")


def du(path: Path) -> int:
    total = 0
    for dp, dn, fn in os.walk(path):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            if f.endswith((".pyc", ".pyo")):
                continue
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def git_info(path: Path) -> tuple[str, str]:
    if not (path / ".git").is_dir():
        return "", ""

    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception:
            return ""

    url = run("remote", "get-url", "origin")
    sha = run("rev-parse", "HEAD")
    if not any(h in url for h in PUBLIC_HOSTS):
        return "", ""
    return url, sha


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 config/nodes.json")
    ap.add_argument("--custom-nodes", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("config/nodes.json"))
    args = ap.parse_args()

    rows = []
    for d in sorted(p for p in args.custom_nodes.iterdir() if p.is_dir()):
        if d.name in SKIP_DIRS:
            continue
        url, sha = git_info(d)
        rows.append({
            "name": d.name,
            "kind": "git" if url else "vendor",
            "url": url,
            "ref": sha,
            "requirements": (d / "requirements.txt").exists(),
            "size_mb": round(du(d) / 2 ** 20, 1),
            "managed": False,
        })

    doc = {
        "$comment": "kind=git 按 url 克隆并钉到 ref；kind=vendor 从 vendor/nodes/ 解压。"
                    "managed=true 表示由 ComfyUI-Manager 管，本工程不装。",
        "packs": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    git_n = sum(1 for r in rows if r["kind"] == "git")
    vendor = [r for r in rows if r["kind"] == "vendor"]
    print(f"{len(rows)} 个包：git {git_n}，vendor {len(vendor)}（{sum(r['size_mb'] for r in vendor):.1f}MB 未压缩）")
    if vendor:
        print("提示：改完清单记得跑 tools/pack_nodes.py 重新打包 vendor/")
    return 0


if __name__ == "__main__":
    sys.exit(main())