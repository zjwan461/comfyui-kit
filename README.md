# comfyui-kit

一套「换台机器也能十分钟装回来」的 ComfyUI 环境包：**安装脚本 + 54 个模型清单 + 26 个找不到仓库的节点包 + 24 套调好的工作流**。

它是从一台实际跑通的机器上打包出来的（Windows + RTX 4080 SUPER 32GB + ComfyUI 0.39.0 + torch 2.11.0+cu130），不是理论配置。

**显存要求：32GB 起**（RTX 4080 SUPER 32GB、RTX 5090 32GB 这一档）。H3 多段工作流要 UNET 19.5GB + CLIP 14.6GB 同时在显存里，32GB 是本包的设计基线；24GB 卡不在支持范围内，不保证能跑，也不打算为它调参。

## 快速开始

```bash
git clone <这个仓库>
cd comfyui-kit
python install.py --comfy-dir E:/ComfyUI --staging-dir E:/staging
```

跑完会生成 `start-comfyui.bat`（Linux 下是 `.sh`），双击启动，浏览器开 http://127.0.0.1:8188。

默认只下 H3 主线要的 6 个模型（约 93GB）。要全量：

```bash
python install.py --tier all --jobs 12        # 54 个模型，387.8GB
```

只要环境（不下载模型）：

```bash
python install.py --skip-models
```

只想给已经装好的 ComfyUI 补模型：

```bash
python install.py --skip-comfy --skip-packs --tier image
```

先看看会做什么：

```bash
python install.py --dry-run
python install.py --list --tier all      # 清单 + 本地已有情况
python install.py --verify-only --tier all
```

## 仓库结构

```
install.py                  一键安装（只用标准库，Win/Linux 通用）
config/
  models.json               54 个模型：下载地址（ModelScope 优先）、sha256、落位路径、被哪些工作流用
  nodes.json                47 个节点包：21 个 git 仓库地址 + 26 个 vendor 包
  workflows/*.json          24 套工作流
vendor/
  nodes/*.zip               26 个本地节点包，压缩后 52MB
  manifest.json             每包的文件名、字节数、sha256
tools/
  pack_nodes.py             从一台装好的机器重新生成 vendor/（节点包更新时用）
  dump_nodes.py             重新生成 config/nodes.json
docs/lessons.md             为什么这么写：这一路踩的坑
```

## 装了什么

### 模型（54 个 / 387.8GB，按 tier 分组）

| tier | 数量 | 大小 | 内容 |
|---|---|---|---|
| `h3-core` | 4 | 96.9GB | MiniMax H3 四套底模（fl2va / ref2va / Singularity ×2） |
| `h3-extras` | 2 | 3.2GB | turbo 4step LoRA、clip_vision_h |
| `image` | 10 | 111.3GB | Qwen-Image 2512 / Edit 2509 / Edit 2511 + z_image_turbo + 配套 VAE、CLIP |
| `lora` | 16 | 48.9GB | 社区 LoRA 与蒸馏加速 LoRA |
| `video` | 7 | 37.0GB | LTX-2.3 全系 + Wan 2.1 i2v + umt5 |
| `video-wan` | 15 | 90.5GB | Wan 2.2 高/低噪、InfiniteTalk、Animate、PusaV1 等 |

模型清单里每条都带 `used_by`，能查到「删了这个文件哪套工作流会跑不起来」。原版 `Qwen动漫转写实真人V2_V2`（RunningHub 未冻结发布）拿不到，改用 Krea2 版 `Krea2_Anything2RealCharacters-V3` 走 `A08b` 这条线，原因见 `docs/lessons.md`。

### 节点包（47 个）

- **21 个有公开仓库**：按 `config/nodes.json` 里的地址 clone，并钉在当时那台机器上的提交（`ref`），保证版本一致。
- **26 个没有公开仓库**（本地改过的、群里发的、作者删库的）：打进 `vendor/nodes/*.zip` 随仓库分发，解压即用。

打包时剔除了纯演示素材（预览 GIF、宣传视频、PDF），所以 289.9MB 的目录压到 52MB。**代码和运行期资源一个不少**。

里面还带着两处必要的本地补丁，比如 `comfyUI-zealman/h3_model_loader.py` 把写死的 `device="default"` 改成读 `clip_device`，否则 32GB 卡上 H3 多段工作流必 OOM。

### 工作流（24 套）

| 组 | 文件 |
|---|---|
| H3 主线 | `h3_t2v_api`、`h3_i2v_api`、`h3_r2v_api` |
| H3 进阶 | `U00-提示词LLM扩写`、`U06V9-9图3音`、`U11-图音对口型`、`U17-9图1视频3音频`、`U22-9图3音`、`U24-文武双修T8版`、`U30-三段采样与AV分离`、`U32-Sigma精修与动作修复` |
| 生图 | `A01-Qwen2512文生图放大`、`A08-二次元转真人`、`A08b-Krea2二次元转真人`、`B13-角色多角度`、`C19-三视图zimage双采`、`E1-首帧重建-Qwen编辑`、`T4-Krea2三视图` |
| 视频 | `G02-Wan首尾帧`、`H41-LTX-MSR多参考`、`J13-InfiniteTalk长口播`、`N2-FishAudio声音克隆`、`P02-WanAnimate动作迁移`、`Y08-LTX四宫格` |

**首次跑会自己下东西的三套**（清单里没有，因为节点自己管）：

| 工作流 | 首跑自下 | 大小 |
|---|---|---|
| `N2-FishAudio声音克隆` | `s2-pro`（`models/fishaudioS2/`）、whisper 模型 | 11GB + 按档位 |
| `H41` / `Y08` | SeedVR2 的 fp8 权重 | 3.2GB |
| `J13` | 无（但三个加载器各认一个固定目录，见 `docs/lessons.md`） | — |

洗图（动漫→真人）有两条，按需要挑：

| | A08（Qwen-Edit） | A08b（Krea2） |
|---|---|---|
| 耗时 | ~160s | **~20s** |
| 保脸 | 保留原构图与五官 | img2img，denoise 0.75 会明显换脸 |
| 质感 | 偏渲染 | 皮肤毛孔/碎发更像实拍 |
| 用途 | 清 AI 感、沿用原分镜 | 换成真人演员、要实拍感 |

装的时候会自动把模型路径改成当前系统的分隔符（Windows 反斜杠 / Linux 正斜杠）——不改的话 ComfyUI 会报 `400 value_not_in_list`，而 `validate` 查不出来。

## 几个必须知道的

**模型目录可以放在大盘上。** `--models-dir` 指到别的盘，脚本会写 `extra_model_paths.yaml`。

**下载暂存盘用 SSD。** 模型目录在机械盘时，直接下会跟 ComfyUI 读模型抢 IO，速度掉到 1MB/s 以下。`--staging-dir` 指一个 SSD，下完校验通过再搬过去。

**装 pip 依赖前先停 ComfyUI。** Windows 上 `cv2.pyd` 会被占用，报 `WinError 5`。脚本检测到 8188 有服务会警告。

**显存不够用 `--lowvram`。** 生成的启动脚本默认带 `--lowvram --disable-pinned-memory --disable-async-offload`。32GB 卡跑 37GB 的 UNET+CLIP 组合就靠它分阶段装卸。

**没有代理也能跑。** 模型优先走 ModelScope（国内直连，实测 22MB/s）；HF 才需要代理，脚本会自动探测本机常见端口。

## 更新这个仓库

节点包升级后：

```bash
python tools/pack_nodes.py --custom-nodes E:/ComfyUI/custom_nodes
python tools/dump_nodes.py --custom-nodes E:/ComfyUI/custom_nodes
```

模型增减：改 `config/models.json`（`dest` 是相对 `models/` 的路径，`tier` 决定分组）。

## 没解决的部分

- **A08 的 `Qwen动漫转写实真人V2_V2` 没找到**：要的是 Qwen-Image-Edit-2509 版（`transformer_blocks.*`，1680 张量）。曾找到同名的 `Flux2 Klein` 版（`double_blocks.*`，224 张量），底模不同、ComfyUI 会直接跳过，没有采用。A08 目前不挂这个 LoRA，靠提示词里写死的"动漫转BJD娃娃，动漫转写实真人"顶着。
- **A08 的 ControlNet Union 是可选项**：`Qwen-Image-ControlNet-Union.safetensors` 只有 diffusers 格式，工作流里已把 ControlNet 三个节点旁路。
- **`comfy.py validate` 不能当验收**：它只比对输入名和节点类，不校验 combo 值和必填项，会把错的判成 OK。真实判据是提交跑一遍。