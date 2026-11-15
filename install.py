#!/usr/bin/env python3
"""ComfyUI 配置工程：一键装出和原机器一样的环境（Windows / Linux，只用标准库）

仓库里带着四样东西，缺一不可：
    config/nodes.json        节点包清单（git 仓库地址 / vendor 压缩包）
    config/models.json       47 个模型的下载地址、sha256、落位路径
    config/workflows/*.json  23 套调好的工作流
    vendor/nodes/*.zip       26 个"网上找不到仓库"的节点包（52MB）

按顺序做：
  1. 环境检查（python / git / 磁盘 / 暂存盘）
  2. clone ComfyUI（默认钉在 v0.39.0）+ 独立 venv + torch（默认 cu130）+ requirements
  3. 装节点包：kind=git 的 clone，kind=vendor 的从 vendor 解压，然后统一装 pip 依赖
  4. 下载模型：ModelScope 直连 → hf-mirror → HF（走代理）
     · 分段并发 + 断点续传（规格写进 .parts.json，规格变了就重下）
     · 每段出错只重传该段，偏移按磁盘真实大小重算（历史上 part 越滚越大就是这个 bug）
     · 下完拼装 → safetensors 头校验 → 有 sha256 就校验 → 从暂存盘搬到模型目录
  5. 装工作流（自动把模型路径改成当前系统的分隔符）
  6. 生成 extra_model_paths.yaml（模型放在别处时）+ 启动脚本
  7. 自检：import torch / cv2 / 数节点包

用法：
    python install.py                             # 装 ComfyUI + H3 主线 6 个模型（约 93GB）
    python install.py --skip-models               # 只装环境（节点包 + 工作流）
    python install.py --tier all --jobs 12        # 连模型一起下完（366GB）
    python install.py --skip-comfy --tier image   # 只往已有的 ComfyUI 里补模型
    python install.py --list                      # 看清单里有什么、本地有什么
    python install.py --verify-only --tier all    # 只体检已有模型
    python install.py --dry-run                   # 只打印将执行的命令

常用参数：
    --comfy-dir D:/ComfyUI          安装目录
    --models-dir D:/ComfyUI/models  模型目录（可指向大盘，会自动写 extra_model_paths.yaml）
    --staging-dir E:/staging        下载暂存盘（强烈建议 SSD，机械盘会和 ComfyUI 抢 IO）
    --proxy http://127.0.0.1:11081  只有 HF / GitHub 走代理；ModelScope 永远直连
    --tier h3-core,h3-extras,image,lora,video,video-wan,all
    --only 文件名1,文件名2          只下指定文件
    --jobs 12                       每个文件的分段并发数
    --source auto|ms|mirror|hf      源策略：auto=按上面顺序自动回落

设计约定：
    · 幂等：每步成功后写 .install-state.json，重跑会跳过；--force 全量重来
    · 任何一步失败都不静默：打最后 25 行日志并退出非 0
    · 日志：logs/install.log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOST = os.name == "nt"
VENV_PY = Path("Scripts/python.exe") if HOST else Path("bin/python")
HERE = Path(__file__).resolve().parent
STATE_PATH = HERE / ".install-state.json"
LOG_PATH = HERE / "logs" / "install.log"

# 模型键 → ComfyUI 的目录名（和 skill 里 scripts/models.py 的映射一致）
KEY_DIR = {
    "unet_name": "diffusion_models",
    "clip_name": "text_encoders",
    "vae_name": "vae",
    "lora_name": "loras",
    "ckpt_name": "checkpoints",
    "control_net_name": "control_net",
    "clip_vision_name": "clip_vision",
    "upscale_model_name": "upscale_models",
    "latent_upscale_model": "latent_upscale_models",
}

# 装机需要的节点包全部写在 config/nodes.json：
#   kind=git    → 按仓库地址 clone（ref 可选，钉住提交更可复现）
#   kind=vendor → 从 vendor/nodes/<包名>.zip 解压（找不到公开仓库的本地包，随仓库分发）
CONFIG = HERE / "config"
NODES_JSON = CONFIG / "nodes.json"
WORKFLOWS_DIR = CONFIG / "workflows"
VENDOR_DIR = HERE / "vendor"
VENDOR_MANIFEST = VENDOR_DIR / "manifest.json"

# 带这些扩展名的字符串是"模型路径"，跨平台要换分隔符
MODEL_EXT = (".safetensors", ".gguf", ".onnx", ".pth", ".bin", ".pt", ".ckpt")


def load_nodes(path: Path = NODES_JSON) -> list[dict]:
    if not path.exists():
        raise RuntimeError(f"找不到节点清单 {path}")
    return json.loads(path.read_text(encoding="utf-8"))["packs"]


def fix_model_paths(obj, sep: str):
    """把工作流里所有模型文件路径统一成当前系统的分隔符。

    ComfyUI 的 combo 校验是字符串全等：Windows 的 /object_info 返回反斜杠，
    工作流写正斜杠就会 400 value_not_in_list；Linux 反过来。
    """
    if isinstance(obj, dict):
        return {k: fix_model_paths(v, sep) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fix_model_paths(v, sep) for v in obj]
    if isinstance(obj, str) and obj.lower().endswith(MODEL_EXT):
        return obj.replace("/", sep).replace("\\", sep)
    return obj

# 这些包出现时一律不装（会改 torch / 编译不过 / ABI 冲突），装完再逐个按需处理
BLOCK = {
    "torch", "torchvision", "torchaudio", "numpy", "scipy", "transformers",
    "tokenizers", "huggingface-hub", "safetensors", "pillow",
    "flash-attn", "flash-attn-2", "xformers", "triton", "sageattention", "sageattn3",
    "bitsandbytes", "insightface", "onnxruntime", "onnxruntime-gpu",
    "opencv-python", "opencv-python-headless", "opencv-contrib-python",
    "opencv-contrib-python-headless", "llama-cpp-python", "nvvfx", "mediapipe",
}

# llama-cpp-python：默认源是 cu128，在 cu130 的机器上会 “Failed to load ggml / 找不到 cudart64_12”
LLAMA_WHEEL = (
    "https://github.com/JamePeng/llama-cpp-python/releases/download/"
    "v0.4.7-cu130-win-20261113/llama_cpp_python-0.4.7%2Bcu130-cp311-cp311-win_amd64.whl"
)

EXT = (".safetensors", ".gguf", ".onnx", ".pth", ".bin", ".pt", ".ckpt")


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
_LOCK = threading.Lock()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def opener(proxy: str = ""):
    """显式构造 opener：不传 proxy 时**不读环境变量**，避免公司代理/系统代理串味。"""
    if proxy:
        h = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    else:
        h = urllib.request.ProxyHandler({})
    return urllib.request.build_opener(h)


def remote_size(url: str, proxy: str = "", timeout: float = 40) -> int:
    """拿远端文件大小。先 HEAD，服务器不支持就退回 Range: bytes=0-0。"""
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "comfy-installer/1.0"})
        with opener(proxy).open(req, timeout=timeout) as r:
            n = int(r.headers.get("Content-Length") or 0)
            if n > 0:
                return n
    except Exception:
        pass
    req = urllib.request.Request(url, headers={
        "Range": "bytes=0-0", "User-Agent": "comfy-installer/1.0",
        "Accept-Encoding": "identity"})
    with opener(proxy).open(req, timeout=timeout) as r:
        cr = r.headers.get("Content-Range") or ""
        m = re.search(r"/(\d+)$", cr)
        if m:
            return int(m.group(1))
        n = int(r.headers.get("Content-Length") or 0)
        if n:
            return n
    raise RuntimeError(f"拿不到大小：{url}")


def safetensors_span(path: Path) -> tuple[bool, int, str]:
    """safetensors 完整性：8 字节头长 + JSON 头 + 最大 data_offsets[1] 必须等于文件大小。"""
    size = path.stat().st_size
    if size < 32:
        return False, size, "太小"
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            return False, size, "读不到头"
        n = struct.unpack("<Q", raw)[0]
        if n <= 2 or n > 200_000_000:
            return False, size, f"头长异常({n})"
        head = fh.read(n)
    try:
        meta = json.loads(head)
    except Exception as exc:
        return False, size, f"头不是 JSON：{exc}"
    mx = 0
    for k, v in meta.items():
        if k == "__metadata__":
            continue
        off = v.get("data_offsets") if isinstance(v, dict) else None
        if off:
            mx = max(mx, int(off[1]))
    need = 8 + n + mx
    if need != size:
        return False, need, f"应为 {need} 实为 {size}（差 {need - size:+d}）"
    return True, need, "OK"


def sha256_of(path: Path, bufsize: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def dur(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m}ms" if h else (f"{m}m{s}s" if m else f"{s}s")


def looks_like_model(name: str) -> bool:
    return name.lower().endswith(EXT)


# --------------------------------------------------------------------------- #
# 下载器
# --------------------------------------------------------------------------- #
class Segment:
    __slots__ = ("idx", "start", "end", "have")

    def __init__(self, idx: int, start: int, end: int, have: int = 0) -> None:
        self.idx, self.start, self.end, self.have = idx, start, end, have

    @property
    def want(self) -> int:
        return self.end - self.start + 1

    @property
    def done(self) -> bool:
        return self.have >= self.want


class Downloader:
    """分段并发下载 + 断点续传 + 整文件校验，先落暂存盘再搬走。"""

    def __init__(self, args, log) -> None:
        self.args = args
        self.log = log
        self.staging: Path = args.staging_dir
        self.staging.mkdir(parents=True, exist_ok=True)

    # -- 源 ---------------------------------------------------------------- #
    def sources(self, m: dict) -> list[tuple[str, str, str]]:
        """返回 [(标签, url, proxy)]；proxy 为空表示直连。"""
        out: list[tuple[str, str, str]] = []
        ms = (m.get("ms_url") or "").strip()
        url = (m.get("url") or "").strip()
        wmode = self.args.Source
        if ms and wmode in ("auto", "ms"):
            out.append(("ms", ms, ""))
        if url:
            if wmode in ("auto", "mirror", "hf"):
                mirror = re.sub(r"^https://huggingface\.co/", "https://hf-mirror.com/", url, flags=re.I)
                if mirror != url:
                    out.append(("mirror", mirror, ""))
            if wmode in ("auto", "hf"):
                out.append(("hf", url, self.args.proxy))
        return out

    # -- 单段 -------------------------------------------------------------- #
    def _grab(self, url: str, proxy: str, seg: Segment, path: Path, stop: threading.Event) -> None:
        pos = seg.start + seg.have
        retry = 0
        while not seg.done and not stop.is_set():
            try:
                req = urllib.request.Request(url, headers={
                    "Range": f"bytes={pos}-{seg.end}",
                    "User-Agent": "comfy-installer/1.0",
                    "Accept-Encoding": "identity",
                })
                with opener(proxy).open(req, timeout=60) as r:
                    if r.status not in (200, 206):
                        raise IOError(f"HTTP {r.status}")
                    with path.open("ab") as fh:
                        while True:
                            buf = r.read(1 << 20)
                            if not buf:
                                break
                            fh.write(buf)
                            seg.have += len(buf)
                            pos += len(buf)
                            if seg.done:
                                break
                    retry = 0
            except Exception as exc:
                retry += 1
                if retry > self.args.Retries:
                    raise RuntimeError(f"分段 {seg.idx} 重试 {retry} 次仍失败：{exc}") from exc
                # 关键：出错后必须按**磁盘真实大小**重算偏移，
                # 否则会把旧偏移再写一遍，part 越滚越大（历史事故）
                seg.have = path.stat().st_size if path.exists() else 0
                pos = seg.start + seg.have
                if seg.have > seg.want:  # 脏数据，整段重来
                    path.unlink(missing_ok=True)
                    seg.have = pos = 0
                time.sleep(min(2 ** retry, 20))

    # -- 单文件 ------------------------------------------------------------ #
    def fetch(self, m: dict, dest: Path) -> str:
        """返回 ok / skip / fail:<原因>"""
        label = m.get("name") or Path(m["dest"]).name
        total = int(m.get("remote") or m.get("size") or 0)
        cands = self.sources(m)
        if not cands:
            return "fail:清单里没有可用 URL"

        if dest.exists():
            ok, need, why = safetensors_span(dest) if looks_like_model(dest.name) and dest.suffix == ".safetensors" else (True, dest.stat().st_size, "OK")
            if ok and (not total or dest.stat().st_size == total):
                if self.args.Verify and m.get("sha256"):
                    if sha256_of(dest).lower() != m["sha256"].lower():
                        self.log(f"    sha256 不符，重下：{dest.name}")
                        dest.unlink()
                    else:
                        return "skip"
                else:
                    return "skip"
            else:
                self.log(f"    已有文件不完整（{why}），删掉重下：{dest.name}")
                dest.unlink()

        last = ""
        for tag, url, proxy in cands:
            try:
                size = total or remote_size(url, proxy)
                if total and size != total:
                    self.log(f"    [warn] {tag} 远端大小 {size} != 清单 {total}")
                    size = total
                st = self._fetch_one(m, url, proxy, size, dest, label)
                self.log(f"    [{tag}] {label}: {st}")
                return st
            except Exception as exc:
                last = f"{tag}: {exc}"
                self.log(f"    [warn] {tag} 失败：{exc}")
                self._cleanup(dest)
        return f"fail:{last}"

    def _cleanup(self, dest: Path) -> None:
        for p in [dest, dest.with_suffix(dest.suffix + ".parts.json")]:
            p.unlink(missing_ok=True)
        for p in self.staging.glob(dest.name + ".part*"):
            p.unlink(missing_ok=True)

    def _fetch_one(self, m: dict, url: str, proxy: str, size: int, dest: Path, label: str) -> str:
        st_dir: Path = self.staging
        st_dir.mkdir(parents=True, exist_ok=True)
        tmp = st_dir / (dest.name + ".download")
        meta_path = Path(str(tmp) + ".parts.json")

        n = 1 if size < 64 << 20 else max(1, min(self.args.Jobs, size // (16 << 20)))
        # 分段规格必须可复现：同一 (size, n) 才算同一套 part
        spec = {"size": size, "n": n, "v": 2}
        if meta_path.exists():
            try:
                if json.loads(meta_path.read_text(encoding="utf-8")) != spec:
                    for p in st_dir.glob(tmp.name + ".part*"):
                        p.unlink(missing_ok=True)
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass
        meta_path.write_text(json.dumps(spec), encoding="utf-8")

        step = size // n
        segs: list[Segment] = []
        for i in range(n):
            s = i * step
            e = size - 1 if i == n - 1 else (i + 1) * step - 1
            part = st_dir / f"{tmp.name}.part{i}"
            have = part.stat().st_size if part.exists() else 0
            if have > (e - s + 1):
                part.unlink()
                have = 0
            segs.append(Segment(i, s, e, have))

        stop = threading.Event()
        t0 = time.time()
        base = sum(x.have for x in segs)

        def worker(seg: Segment) -> None:
            part = st_dir / f"{tmp.name}.part{seg.idx}"
            self._grab(url, proxy, seg, part, stop)

        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = [pool.submit(worker, s) for s in segs]
            while any(not f.done() for f in futs):
                time.sleep(1.0)
                got = base + sum(x.have for x in segs)
                el = max(time.time() - t0, 0.001)
                speed = (got - base) / el
                pct = 100.0 * got / size if size else 0
                sys.stdout.write(f"\r      {label[:38]:38s} {pct:5.1f}% {human(got)}/{human(size)} "
                                 f"{human(speed)}/s ")
                sys.stdout.flush()
            for f in futs:
                f.result()
        sys.stdout.write("\r" + " " * 110 + "\r")

        # 拼装
        with tmp.open("wb") as out:
            for seg in segs:
                part = st_dir / f"{tmp.name}.part{seg.idx}"
                if part.stat().st_size != seg.want:
                    raise RuntimeError(f"分段 {seg.idx} 长度不对：{part.stat().st_size} != {seg.want}")
                with part.open("rb") as fh:
                    shutil.copyfileobj(fh, out, 8 << 20)
        for seg in segs:
            (st_dir / f"{tmp.name}.part{seg.idx}").unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

        if tmp.stat().st_size != size:
            tmp.unlink()
            raise RuntimeError(f"拼装后大小不对：{tmp.stat().st_size} != {size}")

        if dest.suffix == ".safetensors":
            ok, need, why = safetensors_span(tmp)
            if not ok:
                tmp.unlink()
                raise RuntimeError(f"safetensors 头校验失败：{why}")
        if m.get("sha256"):
            got = sha256_of(tmp)
            if got.lower() != m["sha256"].lower():
                tmp.unlink()
                raise RuntimeError(f"sha256 不符：{got[:16]}… != {m['sha256'][:16]}…")

        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        shutil.move(str(tmp), str(dest))  # 跨盘也没问题
        return "ok"


# --------------------------------------------------------------------------- #
# 安装器
# --------------------------------------------------------------------------- #
class Installer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.state = self._load_state()
        self.comfy: Path = args.comfy_dir
        self.venv: Path = self.comfy / ".venv"
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # -- 日志/状态 --------------------------------------------------------- #
    def log(self, msg: str = "") -> None:
        print(msg, flush=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    def _load_state(self) -> dict:
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state(self) -> None:
        STATE_PATH.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")

    def done(self, sid: str, fingerprint: str | None = None) -> bool:
        if self.args.force:
            return False
        cur = self.state.get(sid)
        if not cur:
            return False
        if fingerprint and cur.get("fp") != fingerprint:
            return False
        return True

    def mark(self, sid: str, fingerprint: str | None = None) -> None:
        if self.args.dry_run:
            return
        self.state[sid] = {"at": int(time.time()), "fp": fingerprint}
        self._save_state()

    def step(self, sid: str, title: str, fn, fingerprint: str | None = None) -> None:
        if self.done(sid, fingerprint):
            self.log(f"[ok]   {title}（已完成，跳过）")
            return
        self.log(f"[..]   {title}")
        fn()
        self.mark(sid, fingerprint)
        self.log(f"[done] {title}")

    # -- 命令 -------------------------------------------------------------- #
    def run(self, cmd: list, cwd: Path | None = None, quiet: bool = False,
            check: bool = True, env_extra: dict | None = None) -> int:
        pretty = " ".join(str(c) for c in cmd)
        self.log(f"  $ {pretty}")
        if self.args.dry_run:
            return 0
        env = os.environ.copy()
        if self.args.proxy:
            for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "PIP_PROXY"):
                env[k] = self.args.proxy
        if self.args.pip_mirror:
            env["PIP_INDEX_URL"] = self.args.pip_mirror
            if self.args.pip_mirror.startswith("http://"):
                env["PIP_TRUSTED_HOST"] = self.args.pip_mirror.split("//", 1)[1].split("/", 1)[0]
        env.setdefault("PIP_NO_INPUT", "1")
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        env.setdefault("PIP_TIMEOUT", "60")
        if HOST:  # git 深层路径
            env["GIT_CONFIG_COUNT"] = "1"
            env["GIT_CONFIG_KEY_0"] = "core.longpaths"
            env["GIT_CONFIG_VALUE_0"] = "true"
        if env_extra:
            env.update(env_extra)
        proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                                env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", bufsize=1)
        tail: list[str] = []
        assert proc.stdout is not None
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
                tail.append(line.rstrip())
                del tail[:-40]
                if not quiet:
                    print(line, end="", flush=True)
        code = proc.wait()
        if code != 0 and check:
            raise RuntimeError(f"命令失败（{code}）：{pretty}\n" + "\n".join(tail[-25:]))
        return code

    def py(self) -> str:
        return str(self.venv / VENV_PY)

    def pip(self, *args: str, check: bool = True) -> int:
        return self.run([self.py(), "-m", "pip", "install", *args], check=check)

    # -- 步骤 -------------------------------------------------------------- #
    def step_check(self) -> None:
        self.log(f"  python {sys.version.split()[0]} @ {sys.executable}")
        if sys.version_info < (3, 10):
            raise RuntimeError("需要 Python >= 3.10（推荐 3.11）")
        if not self.args.skip_comfy and not shutil.which("git"):
            raise RuntimeError("没找到 git，请先安装 git")
        free = shutil.disk_usage(HERE).free
        self.log(f"  脚本所在盘可用 {human(free)}；目标 {self.args.comfy_dir}")
        if self.args.comfy_dir.exists():
            try:
                free_c = shutil.disk_usage(self.args.comfy_dir).free
                self.log(f"  目标盘可用 {human(free_c)}")
            except OSError:
                pass
        if self.args.staging_dir:
            self.args.staging_dir.mkdir(parents=True, exist_ok=True)
            self.log(f"  暂存盘 {self.args.staging_dir}（可用 {human(shutil.disk_usage(self.args.staging_dir).free)}）")
        else:
            self.log("  [warn] 没给 --staging-dir：会直接写到模型目录；"
                     "模型目录若在机械盘，建议给一个 SSD 暂存盘")

    def step_clone(self) -> None:
        if (self.comfy / "main.py").exists():
            self.log("  已存在 main.py，跳过 clone")
            return
        self.comfy.parent.mkdir(parents=True, exist_ok=True)
        self.run(["git", "clone", "--depth", "1", "--branch", self.args.comfy_ref,
                  "https://github.com/comfyanonymous/ComfyUI.git", str(self.comfy)])

    def step_venv(self) -> None:
        if (self.venv / VENV_PY).exists():
            self.log("  venv 已存在")
            return
        self.run([sys.executable, "-m", "venv", str(self.venv)])
        self.pip("-U", "pip", "wheel", "setuptools")

    def step_torch(self) -> None:
        chan = self.args.torch_channel
        index = "https://download.pytorch.org/whl/cpu" if chan == "cpu" else f"https://download.pytorch.org/whl/{chan}"
        spec = ["torch", "torchvision", "torchaudio"]
        if self.args.torch_version:
            spec = [f"torch=={self.args.torch_version}"]
            if self.args.torchvision_version:
                spec.append(f"torchvision=={self.args.torchvision_version}")
            spec.append("torchaudio")
        self.pip(*spec, "--index-url", index)
        self.run([self.py(), "-c", "import torch;print('  torch', torch.__version__, 'cuda', torch.cuda.is_available())"])

    def step_deps(self) -> None:
        req = self.comfy / "requirements.txt"
        if req.exists():
            self.pip("-r", str(req))
        # 这几件事必须统一，否则 cv2 / onnxruntime 会 ABI 打架
        self.pip("opencv-contrib-python>=5.0,<5.1")
        self.pip("onnxruntime-gpu", "onnx")

    def _clone_pack(self, name: str, url: str, ref: str = "") -> None:
        dst = self.comfy / "custom_nodes" / name
        if dst.exists() and any(dst.iterdir()):
            self.log(f"  {name} 已存在，跳过")
            return
        cmd = ["git"]
        if self.args.proxy:
            cmd += ["-c", f"http.proxy={self.args.proxy}"]
        cmd += ["clone", "--depth", "1", url, str(dst)]
        self.run(cmd)
        if ref:
            # 钉住提交：先 shallow fetch 再 checkout（depth=1 的 clone 里没有别的历史）
            try:
                fetch = ["git"]
                if self.args.proxy:
                    fetch += ["-c", f"http.proxy={self.args.proxy}"]
                fetch += ["-C", str(dst), "fetch", "--depth", "1", "origin", ref]
                self.run(fetch, check=False)
                self.run(["git", "-C", str(dst), "checkout", "--detach", ref], check=False)
            except Exception as exc:
                self.log(f"  [warn] {name} 钉到 {ref[:8]} 失败：{exc}")
        gd = dst / ".git"
        if gd.exists() and self.args.drop_git:
            shutil.rmtree(gd, ignore_errors=True)

    def step_dirs(self) -> None:
        """把常见模型子目录建好（缺目录时有些节点会直接崩）。"""
        for d in ("diffusion_models", "text_encoders", "vae", "loras", "checkpoints", "clip_vision",
                  "control_net", "upscale_models", "latent_upscale_models", "detection", "LLM",
                  "seedvr2", "wav2vec2", "melbandroformer", "semantic_bridge", "audio_encoders",
                  "model_patches", "fishaudioS2", "diffusion_models/MelRoFormer"):
            (self.args.models_dir / d).mkdir(parents=True, exist_ok=True)

    def step_workflows(self) -> None:
        """把仓库里的工作流装进 ComfyUI 的用户工作流目录，并按当前系统改路径分隔符。"""
        if not WORKFLOWS_DIR.exists():
            return
        if self.args.dry_run:
            self.log(f"  将安装 {len(list(WORKFLOWS_DIR.glob('*.json')))} 个工作流（dry-run 不写盘）")
            return
        dst = self.comfy / "user" / "default" / "workflows"
        dst.mkdir(parents=True, exist_ok=True)
        sep = "\\" if HOST else "/"
        n = 0
        for f in sorted(WORKFLOWS_DIR.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception as exc:
                self.log(f"  [warn] {f.name} 读不了：{exc}")
                continue
            data = fix_model_paths(data, sep)
            (dst / f.name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            n += 1
        self.log(f"  {n} 个工作流 → {dst}")

    def step_extra_paths(self) -> None:
        """模型目录不在 <comfy-dir>/models 时，写 extra_model_paths.yaml 指过去。"""
        default = (self.comfy / "models").resolve()
        if self.args.models_dir.resolve() == default:
            return
        lines = [
            "# 由 install.py 生成：模型放在 ComfyUI 目录之外",
            "comfyui:",
            f"  base_path: {self.args.models_dir}",
            "  checkpoints: checkpoints",
            "  clip: text_encoders",
            "  clip_vision: clip_vision",
            "  configs: configs",
            "  controlnet: control_net",
            "  diffusion_models: |",
            "    diffusion_models",
            "    diffusion_models/minimax",
            "  loras: loras",
            "  style_models: style_models",
            "  text_encoders: text_encoders",
            "  upscale_models: upscale_models",
            "  vae: vae",
            "  latent_upscale_models: latent_upscale_models",
            "  detection: detection",
            "  LLM: LLM",
            "",
        ]
        p = self.comfy / "extra_model_paths.yaml"
        if self.args.dry_run:
            self.log(f"  将写 {p} → {self.args.models_dir}")
            return
        p.write_text("\n".join(lines), encoding="utf-8")
        self.log(f"  {p} → {self.args.models_dir}")

    def step_packs(self) -> None:
        """kind=git 的包：按 nodes.json 里的仓库地址 clone（可选钉到 ref）。"""
        packs = load_nodes(self.args.nodes)
        cn = self.comfy / "custom_nodes"
        cn.mkdir(parents=True, exist_ok=True)
        failed: list[str] = []
        for p in packs:
            if p.get("kind") != "git" or p.get("managed"):
                continue
            name, url = p["name"], (p.get("url") or "").strip()
            if not url:
                failed.append(f"{name}（清单里没有仓库地址）")
                continue
            try:
                self._clone_pack(name, url, p.get("ref") or "")
            except Exception as exc:
                self.log(f"  [warn] {name} 克隆失败：{exc}")
                failed.append(name)
        if failed:
            self.log(f"  [warn] {len(failed)} 个 git 包没装上：")
            for n in failed:
                self.log(f"         - {n}")

    def _vendor_zip(self, name: str, info: dict) -> Path:
        """把多卷 zip 拼回一个临时文件，顺手校验每卷 sha256。"""
        tmp = self.comfy / f".vendor-{name}.zip"
        with tmp.open("wb") as out:
            for part in info["files"]:
                src = VENDOR_DIR / "nodes" / part["file"]
                if not src.exists():
                    raise RuntimeError(f"缺 vendor 分卷 {src.name}")
                if part.get("sha256") and sha256_of(src) != part["sha256"]:
                    raise RuntimeError(f"vendor 分卷损坏 {src.name}")
                with src.open("rb") as fh:
                    shutil.copyfileobj(fh, out, 8 << 20)
        return tmp

    def step_vendor(self) -> None:
        """kind=vendor 的包：从仓库里的 vendor/nodes/*.zip 解压。"""
        packs = [p for p in load_nodes(self.args.nodes) if p.get("kind") == "vendor" and not p.get("managed")]
        if not packs:
            return
        if not VENDOR_MANIFEST.exists():
            raise RuntimeError(f"找不到 {VENDOR_MANIFEST}（用 tools/pack_nodes.py 生成）")
        man = json.loads(VENDOR_MANIFEST.read_text(encoding="utf-8"))
        cn = self.comfy / "custom_nodes"
        cn.mkdir(parents=True, exist_ok=True)
        missing = []
        for p in packs:
            name = p["name"]
            dst = cn / name
            if dst.exists() and any(dst.iterdir()):
                self.log(f"  {name} 已存在，跳过")
                continue
            info = man.get(name)
            if not info:
                missing.append(name)
                continue
            tmp = self._vendor_zip(name, info)
            try:
                with zipfile.ZipFile(tmp) as z:
                    z.extractall(dst)
            finally:
                tmp.unlink(missing_ok=True)
        if missing:
            self.log(f"  [warn] vendor/manifest.json 里没有这些包：{missing}")
        # 模型目录占位（ComfyUI 缺目录时某些节点会崩）
        for d in ("diffusion_models", "text_encoders", "vae", "loras", "checkpoints", "clip_vision",
                  "control_net", "upscale_models", "latent_upscale_models", "detection", "LLM",
                  "seedvr2", "wav2vec2", "melbandroformer", "semantic_bridge"):
            (self.args.models_dir / d).mkdir(parents=True, exist_ok=True)

    def _filtered_requirements(self, req: Path) -> Path | None:
        """把会被拦掉的包剔出去，写成临时文件。"""
        lines = []
        for raw in req.read_text(encoding="utf-8", errors="replace").splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            name = re.split(r"[=<>!~\[; ]", s, 1)[0].strip().lower()
            if name in BLOCK:
                continue
            if s.startswith(("-r", "--")):
                continue
            lines.append(s)
        if not lines:
            return None
        out = req.with_suffix(".filtered.txt")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out

    def step_node_deps(self) -> None:
        cn = self.comfy / "custom_nodes"
        if not cn.exists():
            return
        # 装 pip 依赖前先确认 ComfyUI 没在跑（Windows 上 cv2.pyd 会被锁）
        if HOST and self._port_open(self.args.comfy_port):
            self.log("  [warn] 8188 有服务在跑：装依赖可能因 cv2.pyd 被占用失败，建议先停")
        for d in sorted(p for p in cn.iterdir() if p.is_dir()):
            for req_name in ("requirements.txt",):
                req = d / req_name
                if not req.exists():
                    continue
                f = self._filtered_requirements(req)
                if not f:
                    continue
                self.log(f"  {d.name} ← {req_name}")
                self.pip("-r", str(f), "-q", check=False)
        # 公共补充：节点包里没写但实测必须要的
        self.pip("piexif", "segment-anything", "dill", "pyloudnorm", "tiktoken", "-q", check=False)
        self.pip("mediapipe", "-q", check=False)
        self.pip("dlib-bin", "-q", check=False)
        if HOST and self.args.with_nvidia_vfx:
            self.pip("nvidia-vfx", "-q", check=False)
        self._install_llama()

    def _install_llama(self) -> None:
        """llama-cpp-python：一定是「和 torch 同 CUDA 大版本」的轮子，否则 import 就崩。"""
        if not self.args.with_llama:
            return
        pyver = f"cp{sys.version_info.major}{sys.version_info.minor}"
        if HOST and pyver == "cp311" and self.args.torch_channel == "cu130":
            self.log("  装 llama-cpp-python cu130 轮子")
            if self.pip(LLAMA_WHEEL, "-q", check=False) != 0:
                self.log("  [warn] 轮子下载失败，退回源码安装（会很慢）")
                self.pip("llama-cpp-python", "-q", check=False)
        else:
            self.log(f"  [warn] 没有匹配 {sys.platform}/{pyver}/{self.args.torch_channel} 的预编译 llama 轮子，"
                     "跳过（需要时手工装，务必与 torch 的 CUDA 大版本一致）")

    @staticmethod
    def _port_open(port: int) -> bool:
        import socket
        s = socket.socket()
        s.settimeout(0.4)
        try:
            return s.connect_ex(("127.0.0.1", port)) == 0
        finally:
            s.close()

    # -- 模型 -------------------------------------------------------------- #
    def _load_manifest(self) -> list[dict]:
        p: Path = self.args.manifest
        if not p.exists():
            raise RuntimeError(f"找不到清单 {p}（--manifest 指定，或放在脚本同目录）")
        data = json.loads(p.read_text(encoding="utf-8"))
        models = data.get("models") if isinstance(data, dict) else data
        out = []
        for m in models:
            if not m.get("dest"):
                m["dest"] = f"{m.get('dir', 'loras')}/{m['rel']}"
            m["name"] = Path(m["dest"]).name
            out.append(m)
        return out

    def select(self, models: list[dict]) -> list[dict]:
        tiers = {t.strip() for t in (self.args.Tier or "").split(",") if t.strip()}
        only = {t.strip() for t in (self.args.Only or "").split(",") if t.strip()}
        if "all" in tiers or not tiers:
            tiers = set()
        picked = []
        for m in models:
            if only and m["name"] not in only:
                continue
            if tiers and m.get("tier") not in tiers:
                continue
            picked.append(m)
        return picked

    def step_models(self) -> None:
        models = self._load_manifest()
        picked = self.select(models)
        if self.args.list:
            self.list_models(models, picked)
            return
        if not picked:
            self.log("  没有匹配的文件（检查 --Tier / --Only）")
            return
        total = sum(int(m.get("remote") or 0) for m in picked)
        self.log(f"  清单 {len(models)} 个，选中 {len(picked)} 个，共 {human(total)}")
        dl = Downloader(self.args, self.log)
        ok = skip = 0
        fails: list[tuple[str, str]] = []
        t0 = time.time()
        for i, m in enumerate(picked, 1):
            dest = self.args.models_dir / m["dest"]
            size = int(m.get("remote") or 0)
            self.log(f"  [{i}/{len(picked)}] {m['dest']}  ({human(size)})")
            if self.args.verify_only:
                if not dest.exists():
                    fails.append((m["dest"], "缺失"))
                    continue
                okk, need, why = safetensors_span(dest) if dest.suffix == ".safetensors" else (True, dest.stat().st_size, "OK")
                if okk and (not size or dest.stat().st_size == size):
                    ok += 1
                else:
                    fails.append((m["dest"], why))
                    self.log(f"    [broken] {why}")
                continue
            st = dl.fetch(m, dest)
            if st == "ok":
                ok += 1
            elif st == "skip":
                skip += 1
                self.log("    已存在且校验通过，跳过")
            else:
                fails.append((m["dest"], st))
                self.log(f"    [fail] {st}")
        self.log(f"\n  完成：新增 {ok}，跳过 {skip}，失败 {len(fails)}，用时 {dur(time.time() - t0)}")
        if fails:
            self.log("  失败清单：")
            for name, why in fails:
                self.log(f"    - {name}: {why}")

    def list_models(self, all_models: list[dict], picked: list[dict]) -> None:
        by_tier: dict[str, list[dict]] = {}
        for m in all_models:
            by_tier.setdefault(m.get("tier", "?"), []).append(m)
        self.log("")
        for tier, items in by_tier.items():
            have = sum(1 for m in items if (self.args.models_dir / m["dest"]).exists())
            size = sum(int(m.get("remote") or 0) for m in items)
            self.log(f"  {tier:12s} {have}/{len(items)} 已存在  {human(size):>9s}")
            for m in items:
                p = self.args.models_dir / m["dest"]
                mark = "✓" if p.exists() else " "
                self.log(f"    [{mark}] {human(int(m.get('remote') or 0)):>9s}  {m['dest']}")
        self.log("")
        self.log(f"  选中 {len(picked)} 个")

    def step_scripts(self) -> None:
        py = self.py()
        if HOST:
            bat = self.args.comfy_dir / "start-comfyui.bat"
            lines = [
                "@echo off", "chcp 65001 >nul", f'cd /d "{self.args.comfy_dir}"',
                f'"{py}" main.py --listen 127.0.0.1 --port {self.args.comfy_port} '
                f'--preview-method none {self.args.launch_args}',
                "pause",
            ]
            bat.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
            self.log(f"  {bat}")
            return
        sh = self.args.comfy_dir / "start-comfyui.sh"
        body = ["#!/usr/bin/env bash", "set -euo pipefail",
                f'cd "{self.args.comfy_dir}"',
                f'exec "{py}" main.py --listen 127.0.0.1 --port {self.args.comfy_port} '
                f'--preview-method none {self.args.launch_args}']
        sh.write_text("\n".join(body) + "\n", encoding="utf-8")
        sh.chmod(0o755)
        self.log(f"  {sh}")

    def step_selfcheck(self) -> None:
        self.run([self.py(), "-c",
                  "import torch, cv2, numpy;"
                  "print('  torch', torch.__version__, 'cuda', torch.cuda.is_available(),"
                  "'| cv2', cv2.__version__, '| numpy', numpy.__version__)"])
        cn = self.comfy / "custom_nodes"
        packs = [p.name for p in cn.iterdir() if p.is_dir()] if cn.exists() else []
        self.log(f"  节点包 {len(packs)} 个")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ComfyUI 一键安装 + 模型下载（标准库实现）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--comfy-dir", type=Path, default=Path("D:/ComfyUI") if HOST else Path.home() / "ComfyUI")
    p.add_argument("--models-dir", type=Path, default=None, help="默认 <comfy-dir>/models")
    p.add_argument("--staging-dir", type=Path, default=None, help="下载暂存盘，建议 SSD")
    p.add_argument("--comfy-ref", default="v0.39.0")
    p.add_argument("--comfy-port", type=int, default=8188)
    p.add_argument("--torch-channel", default="cu130", choices=["cu130", "cu129", "cu128", "cu126", "cpu"])
    p.add_argument("--torch-version", default="2.11.0")
    p.add_argument("--torchvision-version", default="0.26.0")
    p.add_argument("--proxy", default="", help="只有 HF / GitHub 走它；ModelScope 永远直连")
    p.add_argument("--pip-mirror", default="", help="如 https://pypi.tuna.tsinghua.edu.cn/simple")
    p.add_argument("--manifest", type=Path, default=CONFIG / "models.json")
    p.add_argument("--nodes", type=Path, default=NODES_JSON, help="节点包清单")
    p.add_argument("--tier", "--Tier", dest="Tier", default="h3-core")
    p.add_argument("--only", "--Only", dest="Only", default="")
    p.add_argument("--source", "--Source", dest="Source", default="auto",
                   choices=["auto", "ms", "mirror", "hf"])
    p.add_argument("--jobs", "--Jobs", dest="Jobs", type=int, default=12)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--verify", action="store_true", help="额外校验整文件 sha256（慢）")
    p.add_argument("--verify-only", action="store_true", help="只体检，不下载")
    p.add_argument("--list", action="store_true", help="列清单与本地状态")
    p.add_argument("--launch-args", default="--lowvram --disable-pinned-memory --disable-async-offload",
                   help="写进启动脚本的额外参数")
    p.add_argument("--skip-comfy", action="store_true", help="只下模型")
    p.add_argument("--skip-models", action="store_true", help="只装环境")
    p.add_argument("--skip-packs", action="store_true", help="不装任何节点包")
    p.add_argument("--skip-vendor", action="store_true", help="不装 vendor 里那些包")
    p.add_argument("--skip-workflows", action="store_true", help="不装工作流")
    p.add_argument("--drop-git", action="store_true", help="克隆完删掉 .git（省空间）")
    p.add_argument("--with-llama", action="store_true", default=True, help="装 llama-cpp-python（cu130 轮子）")
    p.add_argument("--no-llama", dest="with_llama", action="store_false")
    p.add_argument("--with-nvidia-vfx", action="store_true", help="装 nvidia-vfx（RTX 超分节点用）")
    p.add_argument("--force", action="store_true", help="忽略状态文件，全部重跑")
    p.add_argument("--dry-run", action="store_true")
    return p


def detect_proxy() -> str:
    """没给 --proxy 时，看看本机常见端口上有没有本地代理（不猜系统代理）。"""
    import socket
    for port in (11081, 7890, 10809, 1080, 1081):
        s = socket.socket()
        s.settimeout(0.25)
        try:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return f"http://127.0.0.1:{port}"
        finally:
            s.close()
    return ""


def main() -> int:
    args = build_parser().parse_args()
    if not args.proxy:
        found = detect_proxy()
        if found:
            args.proxy = found
            print(f"  自动使用本机代理 {found}（只用于 HF / GitHub；ModelScope 直连）")
    args.Verify = args.verify   # Downloader 内部用大写，统一一下
    args.Retries = args.retries
    args.models_dir = (args.models_dir or (args.comfy_dir / "models")).resolve()
    args.comfy_dir = args.comfy_dir.resolve()
    if args.list and not args.staging_dir:
        args.staging_dir = args.models_dir  # list 不写盘，随便给一个
    inst = Installer(args)
    inst.log(f"=== ComfyUI 安装器 @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    inst.log(f"  目标 {args.comfy_dir}  模型 {args.models_dir}")

    if args.list or args.verify_only or args.skip_comfy:
        # 这些模式下不需要装环境
        if not args.skip_models:
            inst.step_models()
        return 0

    inst.step("check", "环境检查", inst.step_check)
    inst.step("clone", "clone ComfyUI", inst.step_clone)
    inst.step("venv", "创建 venv", inst.step_venv)
    inst.step("torch", f"安装 torch({args.torch_channel})", inst.step_torch)
    inst.step("deps", "安装 ComfyUI 依赖", inst.step_deps)
    if not args.skip_packs:
        inst.step("packs", "clone 节点包（git）", inst.step_packs)
    if not args.skip_packs and not args.skip_vendor:
        inst.step("vendor", "解压 vendor 节点包", inst.step_vendor)
    if not args.skip_packs:
        inst.step("node_deps", "安装节点包依赖", inst.step_node_deps)
    inst.step("dirs", "建模型目录", inst.step_dirs)
    if not args.skip_models:
        inst.step("models", "下载模型", inst.step_models)
    if not args.skip_workflows:
        inst.step("workflows", "安装工作流", inst.step_workflows)
    inst.step("extra_paths", "写 extra_model_paths.yaml", inst.step_extra_paths)
    inst.step("scripts", "生成启动脚本", inst.step_scripts)
    inst.step("selfcheck", "自检", inst.step_selfcheck)

    inst.log("")
    inst.log("装完了。启动：")
    inst.log(f"  {'start-comfyui.bat' if HOST else 'start-comfyui.sh'}")
    inst.log(f"  然后浏览器开 http://127.0.0.1:{args.comfy_port}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n中断", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\n[失败] {exc}", file=sys.stderr)
        sys.exit(1)
