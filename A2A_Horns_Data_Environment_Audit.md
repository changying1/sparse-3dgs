# A2-A — Horns 真实数据与环境可行性审查

审计日期：2026-10-10（Asia/Shanghai）  
仓库：`D:\3DGS\Binocular3DGS + Dropout`  
范围：仅 A2-A；未运行 init/train/render/metrics，未初始化 CUDA，未安装、编译、下载、初始化 submodule、修改算法源码或提交代码。

## 0. 结论与阶段门

**A2-A 结论：有真实 Horns 数据，3-view 无泄漏 manifest 协议在这组真实文件名和 COLMAP image records 上通过；但该数据当前不能直接进入 A2-B。**

| 项目 | 状态 | 结论 |
|---|---|---|
| 真实 Horns 数据 | **CONFIRMED** | `D:\dataset\nerf_llff_data\horns` 存在，62 张原始 JPG 可全部解码，COLMAP 二进制模型齐全。 |
| 无泄漏 3-view manifest | **CONFIRMED** | 生成与再校验成功；3 个训练/PDC 视角严格一致，与 8 个测试视角零交集。 |
| 选中视角 K、尺寸一致 | **CONFIRMED（数值）** | 三张图共用 camera 1、同一 K、同为 4032×3024；`resolution=8` 后同为 504×378。 |
| 数据可直接用于当前 loader/PDC | **NO-GO / CONFIRMED** | 唯一相机模型为 `SIMPLE_RADIAL`；`scene/dataset_readers.py` 和 `triangulate.py` 只接受 `PINHOLE`/`SIMPLE_PINHOLE`。径向畸变参数不能静默忽略。 |
| RTX 5060 本机工具链基础 | **CONFIRMED** | GPU compute capability 12.0，8151 MiB；驱动 596.13；CUDA Toolkit/nvcc 12.9 支持 `compute_120`；VS 2022、CMake、Ninja 可见。 |
| 当前默认 Python 可运行项目 | **NO-GO / CONFIRMED** | 默认 `python` 是 Python 3.10.6，但没有 torch、rasterizer、simple_knn 或 CuPy。 |
| 历史 `fsgs` 可直接复用 | **NEEDS VERIFICATION** | 有 Torch 2.8.0+cu129 和旧仓库编译出的两个扩展；只验证了无 CUDA 初始化的 import/API 表面，未验证 sm_120 cubin/PTX、ABI 或 kernel 执行。原环境必须保持不变。 |
| PDCNet+ 权重 | **MISSING / CONFIRMED** | 当前仓库及 `D:\3DGS` 范围内均未找到预期的 `.pth` 或 `.pth.tar`。 |

因此，**在完成无畸变 pinhole 数据副本、隔离环境及扩展/权重验证前，不应启动 A2-B，也不应把 5k 降为 3k 来绕过这些前置问题。** 3k 只应在 5k 的真实显存诊断表明 8 GB 不足之后作为本地 fallback。

## 1. 数据位置与结构

### 1.1 实际位置

`D:\dataset\nerf_llff_data\horns`

关键结构：

```text
horns/
├─ images/                    62 × .jpg，4032×3024
├─ images_4/                  62 × .png
├─ images_8/                  62 × .png，504×378
├─ sparse/0/
│  ├─ cameras.bin             64 bytes
│  ├─ images.bin              15,924,616 bytes
│  ├─ points3D.bin            5,044,445 bytes
│  └─ project.ini
├─ 3_views/dense/fused.ply    历史 FSGS 产物；不是本次 3-view 协议输入
├─ database.db
└─ poses_bounds.npy
```

### 1.2 完整性与解码

- `images/`：62 个 JPG；通过 `System.Drawing.Image` 创建 bitmap 并读取首尾像素，**62/62 解码成功，0 错误**。
- 62 张解码尺寸均为 **4032×3024**。
- `images.bin`：62 个 image records；image ID 唯一，均能按 image name 映射到真实文件。
- `cameras.bin`：1 个 camera record；所有 62 个 image records 均引用 camera ID 1。
- 解码尺寸与 camera 1 元数据逐张相同，无缺图、无尺寸偏差。
- `images_8/` 也有 62 张且均为 504×378，但扩展名是 `.png`，而 COLMAP model 记录的是 `images/*.jpg`。当前 runner/PDC 从 `images/` 读原图后现场缩放，不会直接读取 `images_8/`。

## 2. COLMAP 相机、图像与 K

唯一相机：

| 字段 | 实际值 |
|---|---|
| camera ID | 1 |
| model ID / model | 2 / `SIMPLE_RADIAL` |
| width × height | 4032 × 3024 |
| params | `f=3368.8237176028883, cx=2016.0, cy=1512.0, k=0.012300143763124053` |

若仅把前三个参数写成针孔 K，其数值为：

```text
K = [[3368.8237176028883, 0,                  2016.0],
     [0,                  3368.8237176028883, 1512.0],
     [0,                  0,                  1.0]]
```

这只是数值审计，**不表示可以丢弃 `k=0.012300143763124053` 并把相机冒充为 pinhole**。当前 loader 在 `scene/dataset_readers.py:100-110` 对其他模型直接失败；PDC 在 `submodules/dense_matcher/triangulate.py:134-145` 同样失败。因此数据结构虽然是合法 COLMAP，但不符合本项目“已无畸变的 PINHOLE/SIMPLE_PINHOLE 数据”前提。

### 2.1 `resolution=8`

4032 和 3024 均可被 8 整除，所以代码中的整数缩放不会产生尺寸取整偏差：

```text
effective width × height = 504 × 378
K/8 = [[421.10296470036104, 0,                  252.0],
       [0,                  421.10296470036104, 189.0],
       [0,                  0,                  1.0]]
```

三张选中图像的 camera ID、原始尺寸、缩放后尺寸和上述 K 均完全相同，A1-Fix1 的 shared-K/shared-size 检查通过。`images_8/` 的实际尺寸也为 504×378。

**仍存在的语义不一致：** 缩放 K 与缩放尺寸彼此一致，但当前代码没有处理 `SIMPLE_RADIAL` 的畸变。因此在完成真实 undistortion 并生成相应 pinhole COLMAP model 前，不能把该数值检查称为几何输入完全合格。

## 3. 真实 3-view 划分

规则：按 `image_name` 字典序排序；`index % 8 == 0` 为 test；从剩余 54 张 train candidates 上用 `round(linspace(0, 53, 3))` 对应的均匀位置选择 3 张。

### 3.1 `selected_train_images`

1. `DJI_20200223_163017_967.jpg`（COLMAP image ID 2，camera ID 1）
2. `DJI_20200223_163053_863.jpg`（COLMAP image ID 31，camera ID 1）
3. `DJI_20200223_163225_243.jpg`（COLMAP image ID 62，camera ID 1）

### 3.2 `pdc_initialization_images`

与 `selected_train_images` **逐项、逐序严格相同**：

1. `DJI_20200223_163017_967.jpg`
2. `DJI_20200223_163053_863.jpg`
3. `DJI_20200223_163225_243.jpg`

### 3.3 `test_images`

1. `DJI_20200223_163016_842.jpg`
2. `DJI_20200223_163024_597.jpg`
3. `DJI_20200223_163033_432.jpg`
4. `DJI_20200223_163047_307.jpg`
5. `DJI_20200223_163055_437.jpg`
6. `DJI_20200223_163103_777.jpg`
7. `DJI_20200223_163201_633.jpg`
8. `DJI_20200223_163208_938.jpg`

集合检查：

- `selected_train_images == pdc_initialization_images`：**PASS**
- `test_images ∩ pdc_initialization_images == ∅`：**PASS**
- 当前 COLMAP image-name 集合/顺序与 manifest：**PASS**
- PDC 本地 record 顺序与 manifest：**PASS**
- 测试图像像素不会进入 PDC 的三图读入列表：**PASS（由 record/path 白名单静态及 CPU 检查确认）**

## 4. Manifest 验证结果

实际执行的唯一 runner 命令：

```powershell
python .\script\run_llff_single.py --profile 5k --scene horns --n-views 3 --resolution 8 `
  --dataset-root 'D:\dataset\nerf_llff_data' --output-root '.\output' `
  --experiment-name a2a_horns_manifest --stages manifest
```

结果：exit code 0。

- view manifest：`output/a2a_horns_manifest/view_manifest.json`
- run manifest：`output/a2a_horns_manifest/run_manifest.json`
- run status：`succeeded`
- selected stages：仅 `manifest`
- `init/train/render/metrics`：全部 `not_selected`，return code 为 null
- profile：5000 iterations；fallback 3000；resolution 8
- manifest source commit：`01677f59e9a23abcffc95a097ca59b3964fa03fd`

补充验证：`python -m unittest discover -s tests -v` 为 **17/17 PASS**。

审计注意：当前工作树含 A1/A1-Fix1 未提交改动，manifest 的 `source_commit` 只能记录 HEAD，不能唯一标识 dirty diff。A2-B 前应保存/冻结实际 diff 或提交 SHA，以保证可复现性；本次未提交。

## 5. 初始化点云采样逻辑专项判断

目标代码位于 `submodules/dense_matcher/triangulate.py:204-307,327-350,382-443`：

```python
points_3D = []
# 每个有效 ref/src pair append 一个 (Ni, 3) NumPy 块
...
points_3D = np.concatenate(points_3D, axis=0)
points_all = torch.tensor(points_3D).float().cuda()
...
sample_points = points_all[torch.randperm(len(points_3D))[:sample_points_num]]
```

### 5.1 实际数据结构与长度

- 在匹配循环中，`points_3D` 起初是 `list[np.ndarray]`，每个元素是一个点云块 `(Ni, 3)`。
- **在采样循环之前**，第 307 行已执行 `np.concatenate(..., axis=0)`；此后 `points_3D` 是单一 NumPy 数组 `(N0, 3)`。
- `points_all` 初始化时是 CUDA tensor `(N0, 3)`，所以刚进入采样循环时 `len(points_3D) == len(points_all) == N0`。
- 当通过 SSIM/可见性筛选追加新点后，`points_all` 变为 `(Nt, 3)` 且 `Nt >= N0`，但 `points_3D` 保持原来的 `(N0, 3)`。

### 5.2 是否错误使用“点云块数量”

**否。** 在该行实际执行时，`points_3D` 已不是块列表，`len(points_3D)` 是初始拼接后的点数 `N0`，不是 ref/src 块数量。不能把这里报告为“只从前 6 个块/点采样”。

### 5.3 实际问题与覆盖影响

**CONFIRMED：采样上界被永久固定为初始点数 `N0`。** `torch.randperm(N0)` 会在 `[0, N0)` 内均匀打乱，所以它不是偏向数组开头的“前几个点”；但迭代中新追加到 `points_all[N0:]` 的点永远不会成为后续随机扰动的 seed。

影响：

- 初始 triangulated cloud 内部仍是均匀无放回抽样（每轮最多 100 个）。
- 新接受的点不能递归成为新的局部探索中心；点云扩展始终围绕初始 seed cloud，可能限制空间覆盖和逐步外扩。
- 若 `N0 < 100`，实际 `sample_points` 少于 100 个，但后续 `rand_p` 第一维硬编码为 100，存在 shape/broadcast 失败风险。
- 该问题不影响本次 CPU manifest 结果，但会影响将来的 PDC 初始化行为。

### 5.4 最小修复建议（本次未修改）

若原算法目标是“从当前累计点云继续随机探索”，最小且稳健的修改应使用当前 tensor 长度并同步实际 batch 大小：

```python
current_sample_count = min(100, points_all.shape[0])
sample_indices = torch.randperm(
    points_all.shape[0], device=points_all.device
)[:current_sample_count]
sample_points = points_all[sample_indices]
rand_p = torch.randn(
    (current_sample_count, sample_num, 3), device=points_all.device
)
```

这保持“每轮最多随机选 100 个已有点、每点生成 200 个高斯扰动候选”的算法目标，同时允许新点成为 seed 并处理小点云。若作者明确只允许初始点作 seed，则应把 `N0` 命名为 `initial_points_count` 并补充注释/断言；现有 `points_all` 的增长语义更支持前一种解释。

## 6. 本机环境只读盘点

### 6.1 GPU、驱动和编译工具

| 项目 | 实际值 | 状态 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU | **CONFIRMED** |
| Compute Capability | 12.0 | **CONFIRMED** |
| 显存 | 8151 MiB | **CONFIRMED** |
| NVIDIA driver | 596.13 | **CONFIRMED** |
| `nvidia-smi` 显示的最高 driver CUDA | 13.2 | **CONFIRMED**；不等同于 Python runtime/toolkit |
| CUDA Toolkit / nvcc | 12.9 / V12.9.41 | **CONFIRMED** |
| nvcc target | 列表含 `compute_120` | **CONFIRMED** |
| MSVC | VS 2022 Build Tools；cl 19.44.35217 x64 | **CONFIRMED** |
| CMake | 4.1.1 | **CONFIRMED** |
| Ninja | 1.13.1 | **CONFIRMED** |

本次只调用了 `nvidia-smi`、版本/路径查询和编译器 target 列表；没有调用 `torch.cuda.*` 的设备探测、创建 CUDA context 或分配 GPU tensor。

### 6.2 默认 Python

PowerShell 实际解析到：

- `C:\Users\huashuo\AppData\Local\Programs\Python\Python310\python.exe`
- Python 3.10.6，MSC v.1932 x64
- torch、torchvision、torchaudio、`diff_gaussian_rasterization`、`simple_knn`、CuPy：均未发现

`where.exe` 还会列出 `D:\Users\huashuo\anaconda3\python.exe`，因此后续必须使用隔离环境的绝对 Python 路径或正确激活环境，不能只凭 `where` 第一行判断解释器。

### 6.3 历史 `fsgs` 环境（保持不变）

绝对路径：`D:\Users\huashuo\anaconda3\envs\fsgs\python.exe`

| 项目 | 实际值 | 判断 |
|---|---|---|
| Python | 3.13.5 | **CONFIRMED** |
| PyTorch | 2.8.0+cu129 | **CONFIRMED** |
| PyTorch CUDA runtime | 12.9 | **CONFIRMED** |
| torchvision | 0.23.0+cu129（metadata） | **CONFIRMED** |
| `diff_gaussian_rasterization` | 已安装，0.0.0，CPython 3.13 `.pyd` | **CONFIRMED** |
| rasterizer 来源 | `file:///D:/3DGS/FSGS/submodules/diff-gaussian-rasterization-confidence` | **CONFIRMED** |
| rasterizer Python API | settings 含 `confidence`；forward 返回 color/radii/depth/alpha | **CONFIRMED（静态/API import）** |
| `simple_knn._C` | 已安装，CPython 3.13 `.pyd`，导出 `distCUDA2` | **CONFIRMED（API import）** |
| simple_knn 来源 | `file:///D:/3DGS/FSGS/submodules/simple-knn` | **CONFIRMED** |
| CuPy | 未发现 | **MISSING** |

两个扩展的无 kernel import probe 成功，且 probe 前后 `torch.cuda.is_initialized()` 都是 false。此结果只证明 CPython 能装载 `.pyd` 并看到所需符号，**不能证明**：

- binary 含 sm_120 cubin 或可用于 sm_120 的 PTX；
- 在 RTX 5060 上能启动 kernel；
- forward/backward 与当前仓库在 tensor layout、ABI 和 autograd 上完全兼容；
- 该环境中的 PDC/CuPy 路径可运行。

为便于来源追踪，两个现有 binary 的 SHA-256 分别为：

- rasterizer `_C.cp313-win_amd64.pyd`：`0AF8A96B24D7F534E51E7013F1F0B282F55866BE447C2FBA3D6D4DA36249500B`
- simple_knn `_C.cp313-win_amd64.pyd`：`D08AFF11E75C3C64C165ECBC1E4FA1E6A4A4031F430DE70E5F3651B492F5FF87`

### 6.4 当前仓库扩展/权重状态

- `submodules/diff-gaussian-rasterization`：gitlink 期望 commit `8829d14f814fccdaf840b7b0f3021a616583c0a1`，`git submodule status` 前缀为 `-`，目录无源码，**未初始化**。
- `.gitmodules` 的 rasterizer URL 是 `ashawkey/diff-gaussian-rasterization.git`；当前渲染器却要求四输出及 confidence 字段。不能在验证其精确 API 前直接初始化/编译这个 URL。
- `submodules/simple-knn`：源码作为普通 git tree 存在，不是本仓库 `.gitmodules` 中的 submodule。
- PDCNet+ 预期权重：
  - `submodules/dense_matcher/pre_trained_models/PDCNet_plus_megadepth.pth`
  - fallback：同名 `.pth.tar`
- 上述两条均不存在；在 `D:\3DGS` 下也未找到同名文件。
- CuPy 不在 `requirements.txt`，但 local correlation 顶层直接依赖它；依赖清单不闭合。

## 7. 缺失项与兼容性边界

### CONFIRMED

- 缺少适配当前仓库的已就位 rasterizer 源码；仓库 rasterizer submodule 未初始化。
- 默认 Python 缺 torch、两个 CUDA 扩展和 CuPy。
- `fsgs` 环境缺 CuPy。
- PDCNet+ 权重缺失。
- 真实 Horns 相机模型不被当前 reader/PDC 支持。
- 本机 nvcc/MSVC/CMake/Ninja 具备后续构建的基本条件，且 nvcc 可生成 compute 12.0 目标。

### POTENTIAL

- Python 3.10 + 与 CUDA 12.9/Blackwell 兼容的 PyTorch binary 是较稳妥的新环境基线；本机已有 Torch 2.8.0+cu129 wheel 的旁证，但不在本次直接指定安装版本。
- 历史 FSGS confidence rasterizer 的 Python wrapper 与当前四输出调用表面吻合，可作为选择/固定源码版本的依据。
- 8 GB 在 resolution 8、3 views、5k 下可能可运行，但未做 GPU 初始化、显存曲线或 kernel smoke，不能在 A2-A 确认。

### NEEDS VERIFICATION

- 选定 PyTorch wheel 是否公开包含 sm_120 支持，以及 `torch.cuda.get_device_capability`/最小 tensor kernel 是否在新环境成功。
- 两个扩展必须在**同一新环境、同一 PyTorch/CUDA ABI** 下构建，并显式覆盖 sm_120；仅复用 `fsgs` 的 `.pyd` 不够。
- rasterizer forward/backward 的四输出、`confidence` 字段及当前调用约定需最小 API/kernel smoke。
- modern CuPy 与 CUDA 12.9、以及旧代码调用的 `cupy.cuda.compile_with_cache` 是否兼容。
- PDC checkpoint 的可信来源、SHA-256、模型键及实际 load compatibility。
- 经过 COLMAP undistortion 的图像、camera model、K、外参和文件名映射是否再次全部通过 manifest/解码检查。

## 8. 隔离环境建议（方案，不执行命令）

建议新建独立环境，例如 `binocular3dgs-rtx5060`，**绝不修改 `fsgs`**。

1. **Python 基线：POTENTIAL** — 优先 Python 3.10。它与项目现有 runner/default Python 及老 PDCNet 依赖的兼容风险通常低于 Python 3.13。
2. **PyTorch 基线：NEEDS VERIFICATION** — 选择官方明确支持 Windows、目标 Python、CUDA 12.x 且包含 Blackwell/sm_120 的 build；优先与本机 nvcc 12.9 对齐。不要按旧 README 直接降到 Torch 2.1.1+cu118，因为该组合不能作为 RTX 5060 架构支持的已证方案。
3. **源码固定：NEEDS VERIFICATION** — 先固定 confidence rasterizer 的准确来源/commit 和 simple-knn 来源；确认 rasterizer Python/C++ API确为四输出加 confidence，再构建。
4. **扩展构建：NEEDS VERIFICATION** — 仅在新环境中，针对 sm_120，并与该环境的 Torch、CUDA runtime、nvcc 和 MSVC 构建。记录编译日志、source commit、arch list、binary hash。
5. **CuPy：NEEDS VERIFICATION** — 选择与 CUDA 12.x 对应的发行包后，先检查旧 `compile_with_cache` API，再做最小 local-correlation kernel smoke；不能仅凭 `import cupy` 判定兼容。
6. **PDC 权重：MISSING** — 由用户提供/授权取得合法的 `PDCNet_plus_megadepth` checkpoint；记录来源与 hash。本阶段不下载。
7. **分层 gate** — CPU imports → 无 CUDA 初始化的 API probe → 最小设备/extension kernel probe → PDC 单对图 smoke → init-only；全部通过后才允许 5k training。
8. **显存策略** — 首选本地 5k、resolution 8；只有记录 OOM 发生阶段、allocated/reserved/max allocated、Gaussian 数量和 rasterizer/PDC 峰值之后，才转为 3k。正式长程训练仍放在 AutoDL 24 GB。

## 9. A2-B 的具体前置条件

以下条件全部满足后才进入 A2-B：

1. 在**新目录**制作 Horns 的 undistorted 数据副本，不覆盖原数据；COLMAP 输出必须是 `PINHOLE` 或 `SIMPLE_PINHOLE`，图像像素与新 K/尺寸严格配套。
2. 对新副本重新完成：全部图像真实解码、COLMAP image/camera 映射、尺寸一致性、resolution 8 K 缩放、3-view manifest、test/PDC 零交集检查。
3. 冻结 A1/A1-Fix1 的实际源码状态；不能只记录 dirty worktree 的 HEAD SHA。
4. 新隔离环境可明确定位解释器；`fsgs` 环境保持原样。
5. 新环境中的 PyTorch/torchvision/CUDA runtime 与 RTX 5060 sm_120 已经最小设备 probe 验证。
6. confidence rasterizer 与 simple_knn 从固定源码构建；API、ABI、sm_120 kernel smoke 全部通过。
7. CuPy local correlation 的 import/API/kernel smoke 通过。
8. PDCNet+ 权重存在、hash 已记录、CPU checkpoint structure/load probe 通过。
9. 明确处理本报告第 5 节的采样上界问题；若决定修复，应另开授权任务并补 CPU/逻辑测试。本次未修改。
10. A2-B 开始时仍应先做最小 init-only/smoke；不得直接启动 5k，更不得启动正式长程训练。

## 10. 本次产生的文件

- 本报告：`A2A_Horns_Data_Environment_Audit.md`
- CPU-only manifest：`output/a2a_horns_manifest/view_manifest.json`
- CPU-only run record：`output/a2a_horns_manifest/run_manifest.json`

除上述报告与获准生成的 manifest 记录外，本次未修改项目源码或真实数据。
