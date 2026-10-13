# A2-B1 — 现有 FSGS 环境复用与 RTX 5060 最小 CUDA 验证

审计日期：2026-10-11（Asia/Shanghai）  
仓库：`D:\3DGS\Binocular3DGS + Dropout`  
数据基线：`D:\dataset\nerf_llff_data_colmap\horns`（本次未读取图像像素）  
Python：`D:\Users\huashuo\anaconda3\envs\fsgs\python.exe`

## 0. 结论

**复用分类：B。现有 FSGS 环境经过最小源码适配即可复用，不需要重新安装或编译。**  
**阶段结论：CONDITIONAL GO。**

依据不是 import 成功，而是以下真实 kernel 结果：

- PyTorch CUDA 与 autograd 在 RTX 5060 / sm_120 上执行成功。
- 已安装 simple-knn 的 `distCUDA2` 在 sm_120 上执行成功，返回 shape/数值有限，且当前 `GaussianModel` 的 `[0]` 解包与其 API 一致。
- 已安装 confidence rasterizer 的 4-Gaussian、32×32 forward/backward 执行成功，返回 image/radii/depth/alpha 四项，photometric loss 的所有被测参数梯度均存在且有限。
- 但当前 Binocular3DGS renderer 没有向 `GaussianRasterizationSettings` 传必需的 `confidence`，不修改时会在 kernel 前发生 `TypeError`。因此不能判定为 A（直接复用）。
- 本地 PDC 初始化仍为 NO-GO：`CuPy`、`trimesh` 和 PDCNet+ 权重缺失；本次未安装、未下载、未加载模型。
- 若在 AutoDL 生成并冻结初始化 PLY，本机只做训练，则上述 PDC 缺失依赖不在训练导入路径内；当前训练模块已在 fsgs 环境中完成无 CUDA 初始化的 import probe。

本任务没有运行真实场景训练、渲染、评估或 PDC；没有编译扩展、修改 fsgs 环境、修改 renderer/GaussianModel/训练算法、实现 I1/I2、commit 或 push。到此停止，不进入 A2-B2 或 5k。

## 1. 环境信息

| 项目 | 实际值 | 状态 |
|---|---|---|
| Python | 3.13.5, Anaconda, MSC v.1929 x64 | CONFIRMED |
| Python executable | `D:\Users\huashuo\anaconda3\envs\fsgs\python.exe` | CONFIRMED |
| PyTorch | 2.8.0+cu129 | CONFIRMED |
| PyTorch git version | `a1cb3cc05d46d198467bebbb6e8fba50a325d4e7` | CONFIRMED |
| PyTorch CUDA runtime | 12.9 | CONFIRMED |
| cuDNN | 91002 | CONFIRMED |
| NVIDIA driver | 596.13 | CONFIRMED |
| Driver-advertised CUDA | 13.2 | CONFIRMED；不等同于 PyTorch runtime |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU | CONFIRMED |
| Compute capability | 12.0 | CONFIRMED |
| VRAM | 8,546,484,224 bytes / nvidia-smi 8151 MiB | CONFIRMED |
| SM count | 26 | CONFIRMED |

`torch.cuda.get_arch_list()`：

```text
sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120
```

PyTorch build 明确包含 `sm_120`，而不是依赖未知 PTX fallback。

## 2. B1-A：PyTorch/CUDA 最小测试

日志：`output/a2b1_fsgs_compat/b1a_cuda.json`

测试过程：

1. import torch 前未初始化 CUDA；`torch.cuda.is_initialized() == false`。
2. `torch.cuda.is_available() == true`。
3. 设备查询返回 capability `(12, 0)`。
4. 在 CUDA 上创建 16 元素 `float32` tensor。
5. 执行 `sum(sin(x) * x^2)` forward 和 backward。
6. 同步设备并验证 loss/gradient 有限。

结果：**PASS**。

- tensor shape：`[16]`
- gradient finite：true
- 未出现 `invalid device function`、`no kernel image`、device-side assert 或 CUDA launch error。
- 初始化前 PyTorch allocated/reserved：0/0。
- tensor 前 `cudaMemGetInfo`：7043 MiB free / 8151 MiB total。
- 操作后 `cudaMemGetInfo`：7021 MiB free / 8151 MiB total。
- PyTorch peak allocated：4096 bytes。
- PyTorch peak reserved：2 MiB。

`cudaMemGetInfo` 的上下文/驱动开销不等于 PyTorch allocator；报告分别记录，不能把两者混为 tensor 显存。

## 3. B1-B：simple-knn

日志：`output/a2b1_fsgs_compat/b1b_simple_knn.json`

### 3.1 来源与 API

- 安装来源：`file:///D:/3DGS/FSGS/submodules/simple-knn`
- 已加载 binary：`D:\Users\huashuo\anaconda3\envs\fsgs\Lib\site-packages\simple_knn\_C.cp313-win_amd64.pyd`
- binary SHA-256：`D08AFF11E75C3C64C165ECBC1E4FA1E6A4A4031F430DE70E5F3651B492F5FF87`
- `_C.distCUDA2`：存在。
- 实际返回：二元 tuple，第一项为距离 `(N,) float32 cuda`，第二项为 3 邻居索引 `(N,3) int32 cuda`。
- 当前仓库 `scene/gaussian_model.py` 使用 `distCUDA2(points)[0]`，与该 API 一致。

### 3.2 kernel smoke

对 8 个合成 3D 点执行 `distCUDA2` 并同步设备：

- distance shape：`[8]`
- neighbor-index shape：`[8,3]`
- 输出 device：`cuda:0`
- 距离全部有限且非负。
- PyTorch peak allocated：3584 bytes。
- PyTorch peak reserved：2 MiB。
- 未出现 device-side assert、invalid device function 或 no kernel image。

结果：**PASS，当前 `GaussianModel` API 兼容。**

## 4. B1-C：confidence rasterizer

日志：`output/a2b1_fsgs_compat/b1c_rasterizer.json`

### 4.1 来源与静态 API

- 安装来源：`file:///D:/3DGS/FSGS/submodules/diff-gaussian-rasterization-confidence`
- wrapper：`D:\Users\huashuo\anaconda3\envs\fsgs\Lib\site-packages\diff_gaussian_rasterization\__init__.py`
- wrapper SHA-256：`0B76DD99ED33B9B8AEE2FFFC88B605C01231F3B8336453192CB2B3917021DD0C`
- binary：`...\diff_gaussian_rasterization\_C.cp313-win_amd64.pyd`
- binary SHA-256：`0AF8A96B24D7F534E51E7013F1F0B282F55866BE447C2FBA3D6D4DA36249500B`

`GaussianRasterizationSettings` 实际字段：

```text
image_height, image_width, tanfovx, tanfovy, bg, scale_modifier,
viewmatrix, projmatrix, sh_degree, campos, prefiltered, debug, confidence
```

`GaussianRasterizer.forward` 实际签名：

```text
(self, means3D, means2D, opacities, shs=None, colors_precomp=None,
 scales=None, rotations=None, cov3D_precomp=None)
```

wrapper forward 返回 `color, radii, depth, alpha`；backward 用 `confidence` 乘各 Gaussian 的梯度。

### 4.2 当前仓库兼容性

当前 `gaussian_renderer/__init__.py` 构造 settings 时传入 12 个旧字段，没有 `confidence`。静态和运行时验证均确认：

```text
TypeError: GaussianRasterizationSettings.__new__()
missing 1 required positional argument: 'confidence'
```

除这一字段外，renderer 的 forward 参数和四返回值解包与安装的 wrapper 相符。

### 4.3 合成 CUDA forward/backward

为区分 API 调用缺口与 ABI/kernel 兼容性，探针显式提供 `confidence=ones(4,1)`，使用 4 个 Gaussian 和 32×32 输出：

| 输出 | shape | finite |
|---|---:|---|
| image | `[3,32,32]` | true |
| radii | `[4]` | true |
| depth | `[1,32,32]` | true |
| alpha | `[1,32,32]` | true |

- visible Gaussian：4/4。
- photometric L1 loss：`0.2413894236087799`。
- `means3D`、`means2D`、SH、opacity、scale、rotation 的梯度均存在且有限。
- PyTorch peak allocated：104,960 bytes（约 0.100 MiB）。
- PyTorch peak reserved：2 MiB。
- forward/backward 均未出现 CUDA kernel/ABI 错误。

结果：**KERNEL PASS / REPOSITORY API INCOMPATIBLE**。

### 4.4 最小兼容方案（仅建议，本次未修改）

当前任务不允许修改 renderer，因此只给出方案：在构造 settings 时增加中性、非学习的全 1 confidence，使行为与普通 rasterizer 的梯度一致。

必须考虑当前 renderer 的 dropout：settings 在过滤 Gaussian 之前创建，但 `dropout_mask` 已经生成。因此 confidence 的长度必须匹配实际传入 rasterizer 的 Gaussian 数：

```python
if dropout_factor > 0.0 and train:
    raster_confidence = torch.ones_like(pc.get_opacity[dropout_mask])
else:
    raster_confidence = torch.ones_like(pc.get_opacity)

raster_settings = GaussianRasterizationSettings(
    ...,
    confidence=raster_confidence,
)
```

不能无条件传完整 `ones_like(pc.get_opacity)`：启用 dropout 后，forward 使用过滤后的 M 个 Gaussian，而完整 confidence 有 N 行，backward 会发生 shape 不匹配。该最小适配不要求修改 GaussianModel、不要求重装或重新编译；正式实施应另开授权任务，并补 dropout=0、dropout>0、train=false 三种 synthetic renderer 测试。

## 5. B1-D：依赖盘点

日志：`output/a2b1_fsgs_compat/b1d_dependencies.json`

| 依赖 | 版本 | 状态 | 备注 |
|---|---:|---|---|
| numpy | 2.1.3 | PRESENT | import 成功 |
| scipy | 1.15.3 | PRESENT | `scipy.ndimage` 可用 |
| OpenCV | 4.12.0.88 | PRESENT | `cv2` import 成功 |
| imageio | 2.37.0 | PRESENT | `imageio.v3` 可用 |
| plyfile | 1.1.3 | PRESENT | import 成功 |
| torchvision | 0.23.0+cu129 | PRESENT | `transforms.Resize` 可用 |
| timm | 0.9.12 | PRESENT | `DropPath`、`trunc_normal_` 可用 |
| scikit-image | 0.25.0 | PRESENT | import 成功 |
| matplotlib | 3.10.0 | PRESENT | import 成功 |
| tqdm | 4.67.1 | PRESENT | import 成功 |
| Pillow | 11.1.0 | PRESENT | import 成功 |
| packaging | 26.0 | PRESENT | import 成功 |
| trimesh | — | **MISSING** | `triangulate.py` 直接依赖 |
| CuPy | — | **MISSING** | local correlation 直接依赖；PDC module graph 在此失败 |
| GPUtil | — | MISSING | requirements 中存在；当前 train import 不依赖 |
| PDCNet+ checkpoint | — | **MISSING** | `.pth` / `.pth.tar` 均不存在，模型加载 NOT TESTED |
| CuPy local-correlation API/kernel | — | NOT TESTED | CuPy 缺失，未安装、未运行 |

当前项目 `train` 模块在 fsgs 中 import 成功，且 import 前后 `torch.cuda.is_initialized()` 均为 false。这证明“使用已有初始化 PLY 的训练依赖图”不会因缺少 CuPy/trimesh 在 import 阶段失败；不代表真实训练已验证。

本地 PDC 初始化结论：**NO-GO**。原因是 CuPy、trimesh 和 checkpoint 同时缺失；按任务边界没有尝试安装或下载。

## 6. GPU 显存与错误汇总

| probe | peak allocated | peak reserved | 结果 |
|---|---:|---:|---|
| PyTorch CUDA/autograd | 4,096 B | 2 MiB | PASS |
| simple-knn | 3,584 B | 2 MiB | PASS |
| rasterizer fwd/bwd | 104,960 B | 2 MiB | KERNEL PASS |

本次 PyTorch allocator 总体最大峰值为 104,960 bytes；没有大张量分配。探针进程结束后，nvidia-smi 为 459–460 MiB used / 7440 MiB free / 8151 MiB total，系统同时存在桌面/第三方 GPU 进程。没有观察到 OOM、device-side assert、invalid device function、no kernel image 或非法访问。

## 7. B1-E：AutoDL 初始化与本地训练分离

### 7.1 当前训练是否强制执行 triangulate

**否。** `train.py` 不调用 `triangulate.py`。runner 把 `init` 和 `train` 定义为独立 stage，只执行 `--stages` 中明确选择的阶段。

- init-only 可用阶段计划：`--stages manifest init`。
- 跳过 init 的训练计划：`--stages manifest train`，会生成/校验 manifest 后直接训练。
- 仅 `--stages train` 时要求同一 experiment 目录中的 `view_manifest.json` 已存在并通过校验。

本次没有执行这些 GPU 阶段。

### 7.2 训练所需初始化 PLY

对于 scene=`horns`、dataset=`LLFF`、experiment name=`<name>`，当前 reader 固定查找：

```text
keypoints_to_3d/LLFF_<name>/horns_keypoints_to_3d.ply
```

文件存在时，reader 直接读取它；`Scene` 随后将相同字节复制到：

```text
output/<name>/input.ply
```

并调用 `GaussianModel.create_from_pcd()`。因此可以在 AutoDL 上生成一次 PLY，再复制回本地，不需要本地重新执行 triangulate。

### 7.3 Baseline/I1/I2 共用同一初始化

可行的无代码方案：

1. 在 AutoDL 使用完全相同的无畸变数据、3-view manifest、PDC 源码和 checkpoint。
2. 生成一次 `horns_keypoints_to_3d.ply`，立即计算 SHA-256，并将其设为只读/冻结 artifact。
3. Baseline、I1、I2 使用各自 experiment name；把同一 PLY 字节复制到各自预期的 `keypoints_to_3d/LLFF_<name>/` 路径。
4. 每次训练前重新计算 SHA-256，要求与冻结值严格相同；同时验证 manifest SHA-256。
5. 训练输出中的 `input.ply` 再次计算 hash，确认没有路径或复制错误。

这样三组实验的初始化点、颜色和顺序完全相同。不要只比较文件名或点数。

### 7.4 必须记录的复现信息

- 无畸变数据核心 tree SHA-256：`A368FB5ADD21CB92A68CBE0E800DA7D5D553A074EDCA203DDB65216768BFD4CF`。
- 当前 3-view manifest SHA-256：`A43D544C48F4767369AB37CF9FF82106F1D64365CB0E03D129559D297108586B`。
- PLY SHA-256、字节数、顶点数和属性 schema。
- PDC checkpoint SHA-256。
- triangulate/PDC 源码 commit + dirty diff/hash。
- Python、Torch、CUDA、GPU、关键依赖版本及完整 argv。
- Python `random`、NumPy、Torch CPU/CUDA 的随机种子和 deterministic 设置。

当前 `triangulate.py` 使用 `torch.randperm` 和 `torch.randn`，但没有显式设置种子。单次生成后用 PLY hash 冻结可以保证后续实验共用同一 artifact；若要求“重新生成也可复现”，后续必须增加显式 `--seed` 并记录 RNG 配置。训练入口的 `safe_state()` 当前把 Python/NumPy/Torch seed 设为 0。

### 7.5 是否需要修改 runner

- **功能上不必修改**：当前 stage 机制已经能 init-only / train-only。
- **严格安全复现建议修改**：初始化路径由 experiment name 隐式决定。Baseline/I1/I2 名称不同会要求三份路径，手工复制容易放错。

后续最小 runner 改进建议：增加显式 `--initialization-ply`、`--expected-initialization-sha256` 和 `--seed`；train stage 在启动前验证 PLY/manifest hash，并把 artifact metadata 写入 run manifest。这样无需改变训练算法，也无需让每个实验重复 PDC。

## 8. 分类与下一阶段最小建议

### 8.1 明确分类

| 选项 | 判定 |
|---|---|
| A. 直接复用 | 否；renderer 缺 `confidence` |
| **B. 最小源码适配后复用，无需安装/编译** | **是** |
| C. 扩展无法兼容，需独立环境 | 否；两个扩展的 sm_120 kernel 已真实通过 |
| D. 证据不足 | 否；基础 CUDA、两个扩展 kernel 和 rasterizer backward 已覆盖 |

总体：**CONDITIONAL GO**。

### 8.2 进入下一阶段前的最小工作

1. 另开授权任务，仅为当前 renderer 增加 dropout-aware 的全 1 `confidence`，不改变算法语义。
2. 补三种 synthetic renderer 回归：dropout=0、dropout>0、train=false，并验证四输出及有限梯度。
3. 若本机跳过 PDC，先从 AutoDL 取得同数据/manifest/代码/权重生成的冻结 PLY，记录 hash；不要在当前缺依赖环境强行运行 PDC。
4. 为 runner 增加显式 initialization artifact/hash/seed gate，或在修改前严格执行三路径同字节复制与 hash 校验。
5. 上述门槛通过后才考虑一次极短的 init-free 训练 smoke；仍不应直接启动 5k。是否降到 3k 必须由真实 5k 显存/OOM 证据决定。

## 9. 日志与哈希

所有日志/探针位于 `output/a2b1_fsgs_compat/`：

```text
ABD5CCAD37C159F8AB880E80E51E12DE7BA641F76AE625B9FB186CEBDC192ECF  b1a_cuda.json
C030B671B79091EC6B6E5E20EE56DB4F9B0DB1380A56626B2DFA004C065B61D1  b1b_simple_knn.json
11074237B599933F5660DC9C279D68BFEA9CA291B20A929CFE461B6109CBD3E2  b1c_rasterizer.json
9AF72B6AA298BA9B065C94B28AC6AB4A51C161D58B6A214C67B7BC39B0E25EC0  b1d_dependencies.json
```

探针源码也保存在同一目录。`git diff --check` 通过；本任务只新增报告和 `output/a2b1_fsgs_compat/` 审计文件，没有修改 fsgs 环境或项目算法源码。
