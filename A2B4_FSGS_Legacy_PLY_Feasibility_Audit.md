# A2-B4 — FSGS 既有点云复用可行性审查

审查日期：2026-10-11（Asia/Shanghai）  
仓库：`D:\3DGS\Binocular3DGS + Dropout`  
候选点云：`D:\dataset\nerf_llff_data\horns\3_views\dense\fused.ply`  
训练数据：`D:\dataset\nerf_llff_data_colmap\horns`  
Python：`D:\Users\huashuo\anaconda3\envs\fsgs\python.exe`

## 1. 结论

**工程 Smoke：CONDITIONAL GO。论文指标比较：NO-GO。**

候选 PLY 的格式、属性、数值、当前相机坐标系投影、生产 `fetchPly()` 读取和 A2-B3 冻结 artifact staging 均通过。19,397 个初始点对当前 manifest 的三个训练相机均为 100% 正深度、100% 投影在图内；旧/新 COLMAP 的 62 个相机外参逐元素相同，旧/新稀疏点的坐标集合也完全相同。因此没有发现坐标系明显不兼容或几何严重异常。

但是，`3_views` 目录名是唯一的“三视角”线索。PLY 不含生成视图、命令、代码版本或 manifest；同目录没有生成日志；legacy `database.db`/COLMAP model 包含全部 62 张图。无法证明生成它的三个视角就是当前 manifest 的三个训练视角，也无法排除测试图参与融合。因此来源状态必须标为 **PROVENANCE UNKNOWN**。

该点云只允许作为临时工程 Smoke 初始化，不得将所得 PSNR/SSIM/LPIPS 与 Baseline/I1/I2 或论文结果进行有效比较。未来真实 Smoke 仍需单独授权；本阶段没有启动训练。

## 2. PLY 格式、点数与 SHA-256

| 项目 | 结果 |
|---|---|
| 文件存在 | PASS |
| 文件大小 | 523,952 bytes |
| SHA-256 | `2ADDA809BC9AAA71B7D5DB51336527E22B4EC2B25DFB1B3090EBB62310253CDE` |
| PLY format | `binary_little_endian 1.0` |
| vertex count | 19,397 |
| vertex properties | `x y z nx ny nz red green blue` |
| A2-B3 完整 payload 校验 | PASS |
| `plyfile.PlyData.read()` | PASS |
| 当前 `dataset_readers.fetchPly()` | PASS，points/colors/normals 均为 `(19397, 3)` |

生产 reader 所需 `x/y/z` 与 `red/green/blue` 齐全，法线 `nx/ny/nz` 也存在。`fetchPly()` 会将 RGB 除以 255，并将法线原样放入 `BasicPointCloud`；该文件无需字段适配或重写。

在 fsgs 环境的组合审查进程中曾因第三方库加载顺序出现一次 Intel OpenMP duplicate-runtime 终止；没有使用 `KMP_DUPLICATE_LIB_OK` 绕过。独立 probe 随后确认：生产 `fetchPly()` 读取成功，且全新进程直接 `import train` 也成功、退出码为 0。因此它不是 PLY 格式失败或当前训练入口的已复现阻断项。

## 3. 点云数值与空间分布

### 3.1 坐标

| 轴 | min | 1% | median | 99% | max |
|---|---:|---:|---:|---:|---:|
| x | -63.4741 | -19.5412 | -1.38167 | 16.2183 | 39.4270 |
| y | -32.5176 | -16.1558 | -0.808878 | 13.0412 | 41.9689 |
| z | 13.7472 | 14.4677 | 28.7960 | 75.6030 | 153.3401 |

- 有限坐标：19,397 / 19,397；NaN/Inf：0。
- 完全重复的 XYZ：0。
- RGB 范围：R `[0,255]`、G `[0,255]`、B `[0,253]`。
- 法线有限：19,397 / 19,397；零法线：0。
- 法线长度 min/median/max：`0.99999987 / 1.00000000 / 1.00000014`，符合单位法线。

相对当前 37,399 个稀疏 COLMAP 点，legacy fused 点的最近邻距离 min/1%/median/99%/max 为：

```text
0.000965 / 0.014208 / 0.079436 / 1.945565 / 42.205178
```

距离大于 2、5、10 的点分别为 190、13、4 个。存在极少量长尾，但 99% 点在 1.946 以内，所有点仍在当前训练相机前方且落入图像；没有发现足以停止 staging 的严重离群或非有限几何。

## 4. 点云来源审查

来源结论：**PROVENANCE UNKNOWN**。

可确认的事实：

- 文件位于 `horns/3_views/dense/fused.ply`，目录标签声称 `3_views`。
- PLY header 没有 comment、生成命令、源图列表或软件版本。
- `3_views` 目录下只找到该 PLY，没有 fusion config、运行日志、manifest 或 sidecar hash。
- legacy `database.db` 包含 62 张图，名称集合与 legacy COLMAP 的 62 个 image records 相同。
- 当前 manifest 的三个训练视角是：
  - `DJI_20200223_163017_967.jpg`
  - `DJI_20200223_163053_863.jpg`
  - `DJI_20200223_163225_243.jpg`
- 当前 test split 有 8 张图。现有 artifact 不能证明这些测试图未参与旧点云生成。

因此不能确认：

1. PLY 确实只由三张图生成；
2. 那三张图与当前 manifest 的三张训练图一致；
3. 稠密融合没有使用任何当前测试图；
4. 当时使用的 FSGS commit、COLMAP 参数和输入选择。

目录名称只能作为弱线索，不能作为实验 provenance。该缺口对“能否跑通工程链路”不是硬阻断，但对任何质量指标比较是硬阻断。

## 5. 与当前无畸变 Horns 相机的坐标系兼容性

当前无畸变数据由 legacy `sparse/0` 通过 COLMAP `image_undistorter` 生成；既有 A2-A-Fix1 记录显示没有重新运行 feature extractor、matcher、mapper 或 bundle adjustment。

本次重新做了数值核对：

| 检查 | 结果 |
|---|---|
| legacy/current image records | 62 / 62，名称全集一致 |
| qvec 最大绝对差 | 0.0 |
| tvec 最大绝对差 | 0.0 |
| legacy/current sparse points | 37,399 / 37,399 |
| 稀疏 XYZ 按文件记录顺序比较 | 顺序不同，不可直接判定 |
| 稀疏 XYZ 排序后集合最大绝对差 | 0.0 |

相机内参变化符合无畸变过程：

- legacy：`SIMPLE_RADIAL`，4032×3024，`f=3368.8237, cx=2016, cy=1512, k=0.0123001`；
- current：`PINHOLE`，4003×3002，`fx=fy=3368.8237, cx=2001.5, cy=1501`。

外参和世界点坐标保持不变，仅相机模型、裁剪后的尺寸/主点、二维观测及序列化顺序发生变化。结合直接投影结果，判定 legacy fused PLY 与当前无畸变 Horns 位于兼容的 COLMAP 世界坐标系。

## 6. 可见性与图像覆盖

使用当前 `PINHOLE` K 和每张图的 COLMAP `qvec/tvec`，按 `X_cam = R(qvec) X_world + t` 投影全部 19,397 点。

| 当前训练视角 | 正深度 | 图内点 | 32×32 网格覆盖 | 深度 median | 深度 1%–99% |
|---|---:|---:|---:|---:|---:|
| `163017_967` | 100% | 100% | 39.75% | 29.463 | 16.591–79.605 |
| `163053_863` | 100% | 100% | 42.87% | 30.336 | 14.753–73.356 |
| `163225_243` | 100% | 100% | 41.11% | 29.856 | 14.270–73.311 |

- 至少在一个训练视角可见：19,397 / 19,397（100%）。
- 同时在三个训练视角可见：19,397 / 19,397（100%）。
- 62 个当前相机的正深度比例 min/median/max：100% / 100% / 100%。
- 62 个当前相机的图内比例 min/median/max：95.535% / 99.961% / 100%。

32×32 网格覆盖是离散占格率，不是表面面积或像素遮挡率。约 40% 的占格率及可见投影的宽广分位范围表明点没有塌缩在小区域；它不代表重建质量足够，也不能替代渲染结果。

## 7. 8 GB 显存风险

初步风险：**对 200 iterations、无 densification 的工程 Smoke 为低到中等；真实峰值尚未测量。**

有利因素：

- 初始点仅 19,397，远低于常见百万级稠密点云。
- `resolution=8` 时当前 4003×3002 图像进入 reader 的约为 500×375。
- `sh_degree=1` 时每点主要可训练字段约 23 个 float（XYZ 3、SH 12、scale 3、rotation 4、opacity 1）；纯参数下限约 1.70 MiB。即使粗略加入 gradient 与 Adam 两个 moment，也只有约 6.8 MiB，尚未计临时张量。
- 默认 `densify_from_iter=1000`，200-step Smoke 不会调用 `densify_and_prune()`，点数预期保持 19,397。

仍不能提前保证 8 GB：rasterizer 缓冲、图像/相机张量、loss、autograd、CUDA context、Binocular Consistency 和临时排序内存不包含在上述下限估算中。A2-B1/B2 的显存证据只来自极小合成 probe，不能外推真实场景。未来获批的 200-step 运行必须记录峰值显存和 OOM；不得把本次 CPU 审查写成“8 GB 已验证”。

200 iterations 低于 densification 的 `iteration > 1000` 门槛，只能验证数据读取、初始化、真实相机、renderer、loss、backward、optimizer 和保存链路的早期运行；**不能验证 densification，也不能得出模型质量结论。**

## 8. A2-B3 固定初始化接口兼容性

独立实验名：`a2b4_horns_legacy_ply_smoke_200`。

实际执行了 `--stages manifest train --dry-run`，结果：

- runner 退出码：0；
- run status：`dry_run`；
- PDC `init`：`not_selected`；
- train：`pending`，没有创建训练子进程；
- `train.log`：不存在，符合“未训练”；
- source hash 与 expected hash 一致；
- destination copy hash 与 source 完全一致；
- source/destination 都是 523,952 bytes、19,397 点、相同属性顺序；
- `source_unchanged_after_copy=true`、`byte_identical=true`；
- 原始 PLY 未修改；没有重排点、改颜色、抽样或改点数。
- dry-run 后对该真实 artifact 调用 A2-B3 `validate_frozen_initialization_for_training()`：PASS；当前 manifest hash、目标 PLY hash、可读性和 vertex count 均通过训练前 gate，且没有创建训练子进程。

staged 文件：

```text
D:\3DGS\Binocular3DGS + Dropout\keypoints_to_3d\LLFF_a2b4_horns_legacy_ply_smoke_200\horns_keypoints_to_3d.ply
```

Smoke `view_manifest.json` 与 A2-A-Fix1 已审计 manifest 的大小和 SHA-256 完全相同：

```text
A43D544C48F4767369AB37CF9FF82106F1D64365CB0E03D129559D297108586B
```

manifest 记录的 training views 与 `pdc_initialization_images` 完全相等，但这只证明当前 runner 的训练协议一致；不能反向证明 legacy PLY 当初由这些视角生成。

没有修改 A2-B3 runner、renderer、GaussianModel 或训练算法。

## 9. 200-iteration Smoke 命令预览

以下是未来获得单独授权后才可执行的命令；**本次未执行**。现有 manifest 和 staged PLY 已存在，因此使用 train-only stage，不调用 PDC：

```powershell
& 'D:\Users\huashuo\anaconda3\envs\fsgs\python.exe' `
  'D:\3DGS\Binocular3DGS + Dropout\script\run_llff_single.py' `
  --iterations 200 `
  --scene horns `
  --n-views 3 `
  --resolution 8 `
  --dropout-factor 0.2 `
  --sigma-noise 0 `
  --dataset-root 'D:\dataset\nerf_llff_data_colmap' `
  --output-root 'D:\3DGS\Binocular3DGS + Dropout\output' `
  --experiment-name a2b4_horns_legacy_ply_smoke_200 `
  --initialization-ply 'D:\dataset\nerf_llff_data\horns\3_views\dense\fused.ply' `
  --expected-initialization-sha256 '2ADDA809BC9AAA71B7D5DB51336527E22B4EC2B25DFB1B3090EBB62310253CDE' `
  --stages train
```

预览得到的训练子命令：

```text
D:\Users\huashuo\anaconda3\envs\fsgs\python.exe "D:\3DGS\Binocular3DGS + Dropout\train.py" -s D:\dataset\nerf_llff_data_colmap\horns -m "D:\3DGS\Binocular3DGS + Dropout\output\a2b4_horns_legacy_ply_smoke_200" --iterations 200 --dropout_factor 0.2 --sigma_noise 0.0 --suffix a2b4_horns_legacy_ply_smoke_200 --n_views 3 --resolution 8 --dataset_name LLFF --view_manifest "D:\3DGS\Binocular3DGS + Dropout\output\a2b4_horns_legacy_ply_smoke_200\view_manifest.json" --eval
```

runner 将完整命令、合并 stdout/stderr、开始/结束 UTC 和退出码写入：

```text
output/a2b4_horns_legacy_ply_smoke_200/train.log
```

## 10. 判定与使用边界

| 用途 | 判定 | 原因 |
|---|---|---|
| PLY 格式/reader | GO | schema、完整 payload、`PlyData`、生产 `fetchPly()` 均通过 |
| 当前相机坐标系 | GO | 外参完全一致、稀疏 XYZ 集合一致、直接投影通过 |
| A2-B3 staging | GO | source/destination/hash/manifest/byte identity 全部通过 |
| RTX 5060 8 GB、200-step 工程 Smoke | CONDITIONAL GO | 点数与分辨率有利，但真实峰值未测；需单独授权和日志/显存监控 |
| 有效 PSNR/SSIM/LPIPS 比较 | NO-GO | PROVENANCE UNKNOWN，无法排除视角不一致或测试图泄漏 |

最终结论：**允许在下一阶段经用户明确验收和授权后运行最多 200 iterations 的临时工程 Smoke；不允许把该点云或 Smoke 指标用于 Baseline/I1/I2、公平消融或论文质量结论。** 本阶段到此停止。

## 11. 可复核证据

- `output/a2b4_legacy_ply_audit/projection_audit.json`：PLY 数值、COLMAP 模型集合比较及投影统计。
- `output/a2b4_legacy_ply_audit/fetchply_probe.json`：生产 `fetchPly()` 返回 shape。
- `output/a2b4_legacy_ply_audit/train_import_probe.txt`：全新 fsgs 进程的 train import 结果。
- `output/a2b4_legacy_ply_audit/a2b3_gate_probe.txt`：本次 staged artifact 的训练前 gate 结果。
- `output/a2b4_horns_legacy_ply_smoke_200/view_manifest.json`：本次独立 Smoke manifest。
- `output/a2b4_horns_legacy_ply_smoke_200/run_manifest.json`：A2-B3 dry-run 与 artifact 记录。

本次没有安装/卸载依赖、修改 fsgs 环境、编译 CUDA、启动 PDC、启动 GPU 训练、运行 5k、实现 I1/I2、commit 或 push。
