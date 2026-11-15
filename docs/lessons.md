# 为什么这么写

这份文档讲的是「按这个仓库装环境」时必须避开的坑，以及每个默认值背后实测到的数字。
用法看根目录 README.md；这里只讲原因。

## 下载

### 源顺序 ModelScope → hf-mirror → HF 代理

实测同一时段：

| 源 | 单流 | 并发总量 |
|---|---|---|
| HF + 本地代理 | 0.23–0.6 MB/s | 8 路才 2.2 MB/s，16 路反而下降 |
| `huggingface_hub`（2.2.0 + hf_xet） | 40 秒 0 字节，连 `200` 响应都要 9.9 秒 | 2.x 没有 `max_workers`，换库没用 |
| ModelScope 直连 | 13.3 MB/s | 4 路 22.4 MB/s（打满 180Mbps） |
| hf-mirror 直连 | 大幅波动，0.4–30 MB/s | 平均可用 |

所以清单里 22 个主流文件全在 ModelScope 有同字节镜像——**优先 MS，且 MS 永远直连不走代理**（代理反而慢）。HF 只有在 MS 没有时才用，而 HF 的 LFS 会 302 跳到 `cas-bridge.xethub.hf.co`，直连只有 150–290 KB/s，**必须走代理**。

`--source` 可强制：`auto`（默认）/ `ms` / `mirror` / `hf`。

### 分段并发 + 断点续传 + 整文件校验

- 分段规格写进 `<文件名>.download.parts.json`（`{size, n}`）。**规格不匹配就清空所有 part**——历史上吃过亏：先把文件切成 8 段，中途改成 12 段，复用了旧的 `partN`，分段总和比文件还大，拼出来必坏。
- 每段出错后按**磁盘真实大小**重算偏移再续传。历史事故：只在函数入口算一次偏移，重试时把旧偏移又写一遍，part 越滚越大，最后长度校验失败。
- 拼装后校验：`safetensors` 读文件头（8 字节头长 + JSON + 最大 `data_offsets[1]` 必须等于文件大小）；清单里有 `sha256` 的再整文件校验（`--verify` 强制）。
- **长度对不代表内容对**：T4 那次就是 12.25GB 全下完、长度完全正确，但 256 个 `comfy_quant` 张量里有 1 个是随机字节，ComfyUI 加载时 `json.loads` 直接崩（`UnicodeDecodeError: 0xd2`）。所以能拿到 sha256 就一定要比。

### 先落暂存盘，再搬

模型目录在机械盘时，直接下会跟 ComfyUI 读模型抢 IO（实测掉到 <1MB/s）。`--staging-dir` 给一个 SSD，下完校验通过再 `move` 过去。

### llama-cpp-python 必须和 torch 同 CUDA 大版本

requirements 里给的 `v0.3.30+cu128` 轮子在本机（torch cu130）会 `Failed to load 'ggml'` / 找不到 `cudart64_12.dll`。要用 cu130 轮子：

```
llama_cpp_python-0.4.7+cu130-cp311-cp311-win_amd64.whl
```

脚本里 `--with-llama`（默认开）会自动装；`--no-llama` 关掉。

### 装 pip 包前先停 ComfyUI

Windows 上 `cv2.pyd` 被运行中的 ComfyUI 锁住 → `OSError: [WinError 5] 拒绝访问`。脚本会探测 8188 并警告。

另外别按 venv 路径杀进程（会连带杀掉同 venv 的下载任务），按命令行过滤 `main.py`。

### opencv / onnxruntime 必须统一装

节点包各自写 `opencv-python` / `opencv-contrib-python` / headless，混装必 ABI 崩。脚本把所有 `opencv*` 从节点 requirements 里剔掉，统一装一个 `opencv-contrib-python`。

### Windows 下模型路径要反斜杠

ComfyUI 在 Windows 上 `/object_info` 返回的是反斜杠路径，工作流里写正斜杠会 `HTTP 400 value_not_in_list`（validate 能过、submit 必挂）。同步到 Linux 部署时要换回正斜杠。

### 启动参数：`--lowvram` 而不是 `--disable-dynamic-vram`

- 机械盘时代为了防卡死加过 `--disable-dynamic-vram`；迁到 NVMe 后它反而挡住显存腾挪，H3 多段工作流会 OOM。
- 现在用 `--lowvram --disable-pinned-memory --disable-async-offload`，32GB 卡能跑 37GB 的模型组合（分阶段装卸）。
- `--preview-method none` 省显存；别把 `--preview-method` 写成 `--preview-mode`（服务直接起不来）。

### 拆掉 Zealman 加载器会带走它内置的 LoRA 槽

`ZealmanH3ModelLoader` 把 UNET + CLIP + 两个 VAE + 6 个 LoRA 槽捆成一个节点，所以它一加载就 37GB，32GB 卡必 OOM。拆成相邻的 `UNETLoader` / `CLIPLoader` / `VAELoader`×2 之后能跑，但 LoRA 槽会一起消失——作者默认开着的是 `minimax_h3_turbo_v4_step600_ema_pruned_comfyui`（1.0）和 `davham_cinema-h3-Euler-simple-0.8`（0.6），拆完速度与调色都会变。

`U06V9` / `U30` 已补一个 `905 LoraLoaderModelOnly`（turbo 1.0）接在 `900 UNETLoader` 与 `BlockSparseAttention` 之间；cinema 那个本地没有镜像，暂不补。默认关闭的四个（`H3_Combat_V2`、`Motion_Repair_V2`、`H3_Fight_Flow_V1`、`motion_speed_slider`）按需自己加。

反过来，`U22` 用的是 hybrid 底模，作者带的 `dmad` / `MysticXXX` 两个 LoRA 与它维度不符（日志一直刷 `shape '[96768, 8]' is invalid for input of size 260112384`），ComfyUI 会把整个 LoRA 静默跳过——那里是直接不要 `lora_stack`，不是忘了接。

turbo 挂回去之后实测（`U06V9`，1 秒 416×256，32GB 卡）：

| 配置 | 总耗时 | 采样 | 说明 |
|---|---|---|---|
| 不挂 turbo，8 步 + 1 精修 | 371s | 9 步 | 拆分后的状态 |
| 挂 turbo，8 步 + 1 精修 | 390s | 9 步 | 步骤没减，LoRA 只多 208 个 patch，反而慢一点 |
| 挂 turbo，4 步 + 1 精修 | **260s** | 5 步 | 快 33%，同 seed 同机位，画面结构一致 |

所以 turbo 不是"挂上就快"，是**允许把步数砍到 4**。日志里能看到 `MiniMaxH3 prepared for dynamic VRAM loading. 32427MB Staged. 208 patches attached.`，没有 LoRA 维度报错。出草稿用 4 步，交付再回 8 步。

## 常用开关

```powershell
--with-nvidia-vfx        # RTX 超分节点（nvidia-vfx）
--drop-git               # 克隆完删 .git，省空间
--skip-vendor            # 不装 vendor 里的 26 个包
--skip-workflows         # 不装工作流
--pip-mirror https://pypi.tuna.tsinghua.edu.cn/simple
--tier h3-core,h3-extras,image,lora,video,video-wan,all
--only 文件1,文件2
```

--tier 对应 50 个模型 / 387.3GB：h3-core 4 个（96.9GB）、h3-extras 2 个（3.2GB）、image 10 个（111.3GB）、lora 12 个（48.4GB）、video 7 个（37.0GB）、video-wan 15 个（90.5GB）。

## 装不上时

- **git 包**：会打印清单。多半是网络问题，重跑即可（脚本幂等，已克隆的会跳过）。
- **vendor 包**：直接从仓库里的 zip 解，理论上不会失败；若提示分卷损坏，用 `tools/pack_nodes.py` 重新生成。
- **节点类还是缺**：说明某套工作流依赖的包装上了但 import 失败，看 logs/install.log 里的 IMPORT FAILED。

## 状态与日志

- .install-state.json 记录已完成步骤，重跑跳过；--force 全量重来
- logs/install.log 是完整日志（pip / git 全部输出都在）
- 模型下载的暂存文件在 --staging-dir，中断后重跑自动续传

### ModelScope 大文件别开多线程

同一台机器实测（Z-Image White_Marble 19.3GiB）：

- 单流 curl -L --noproxy "*" -C - -o <file> <url>：**30–50MB/s**，19GiB 约 7 分钟
- 10 条 range 并发：全部卡在 0 字节（CDN 不拒绝、就是不给数据），白等 10 分钟
- python urllib 直接请求：302 跳到 cdn-lfs-cn-1.modelscope.cn 后读挂住

结论：ModelScope 的 resolve 地址对并发不友好，**单连接 + -C - 续传反而最快**；hf-mirror / Xet 那种才需要 12 线程分段（并且要走代理）。脚本里 --jobs 只对 HF 系有意义。


### 同一套「动漫转真人」，Krea2 上能用、Qwen-Edit 上不能用

RunningHub 上那版 `Qwen动漫转写实真人V2_V2` 作者没冻结发布，拿不到。ModelScope 上有同作者的
`Krea2_Anything2RealCharacters-V3`（228.6MB），但它是**给 Krea2 训的**，两者结构完全不同：

| | Qwen-Image-Edit-2509 | Krea2 LoRA |
|---|---|---|
| 键名 | `model.diffusion_model.transformer_blocks.*` | `diffusion_model.blocks.*` / `txtfusion.*` |
| 隐层 | 3072 | 6144 |
| 张量数 | 1933 | 512（rank 32） |

把 Krea2 的 LoRA 塞进 A08（Qwen-Edit）会被整份跳过，所以 A08 保持不挂。取而代之新增了 `A08b-Krea2二次元转真人`：
Krea2 底模 + 这个 LoRA 0.9，img2img（denoise 0.75，8 步 cfg 1）+ SeedVR2 3B 放大到 2048，
832×1216 → 2048×2992 全流程 20 秒，皮肤毛孔/碎发都是真的。触发词作者写在 README 里：`transform the image to realistic photograph`。
