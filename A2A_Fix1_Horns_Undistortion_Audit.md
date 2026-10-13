# A2-A-Fix1 — Horns 无畸变数据副本审计

审计日期：2026-10-11（Asia/Shanghai）  
仓库：`D:\3DGS\Binocular3DGS + Dropout`  
输入：`D:\dataset\nerf_llff_data\horns`  
输出：`D:\dataset\nerf_llff_data_colmap\horns`

## 0. 结论

**PASS。** 已使用现有 COLMAP `image_undistorter` 对 62 张 Horns 原始图像执行真实去畸变，并在全新目录中生成与像素对应的 `PINHOLE` COLMAP 模型。输出图像、图像名、image ID、camera ID 和相机位姿均通过审计；新的 3-view manifest、无泄漏检查、共享 K/尺寸检查、`read_and_validate_colmap_scene()` 和 CPU tests 均通过。原始 Horns 目录整树 SHA-256 在转换前后完全相同。

本任务未执行 PDC 初始化、GPU 训练/渲染、I1/I2、特征匹配、SfM 或 BA；未安装/卸载软件，未修改 `fsgs`，未 commit 或 push。到此停止，不进入 A2-B。

## 1. 现有数据与工具盘点

### 1.1 COLMAP

- 可执行文件：`D:\COLMAP\colmap-cuda\bin\colmap.exe`
- 实际版本：COLMAP 3.13.0
- commit：`0b31f98`（2025-11-07）
- build：CUDA
- `image_undistorter`：可用；支持 `COLMAP` 输出和 `copy` policy。

### 1.2 可复用数据检查

- `D:\dataset` 下只找到原始 `D:\dataset\nerf_llff_data\horns`，未找到第二个 Horns、`undistorted` 或对应 `colmap` 数据目录。
- 原始 `images_8/` 有 62 张 `.png`，62/62 可解码，尺寸均为 504×378。
- 原始 `sparse/0/images.bin` 记录的却是 62 个 `.jpg` 名称，并关联原始 `SIMPLE_RADIAL` 相机，而不是与 `images_8/` 配套的 pinhole 模型。
- 对 3 个代表性文件，将原始畸变 JPG 直接缩放到 504×378 后与 `images_8` 比较，ImageMagick Q16 归一化 MAE 分别为 `0.000100699`、`0.000117921`、`0.000110998`，与普通缩小图高度一致。

因此，`images_8` 只能认定为原图的缩小版本，不能认定为具有配套几何模型的无畸变数据，本次未复用它。

## 2. 实际转换

执行前已确认目标 `D:\dataset\nerf_llff_data_colmap\horns` 不存在；随后创建全新目录，没有覆盖任何已有有效文件。

实际命令：

```powershell
& 'D:\COLMAP\colmap-cuda\bin\colmap.exe' image_undistorter `
  --image_path 'D:\dataset\nerf_llff_data\horns\images' `
  --input_path 'D:\dataset\nerf_llff_data\horns\sparse\0' `
  --output_path 'D:\dataset\nerf_llff_data_colmap\horns' `
  --output_type COLMAP --copy_policy copy --max_image_size -1
```

COLMAP 日志报告读取 `62 images and 37399 points`，完成 `Undistorting image [62/62]` 并成功写出 reconstruction。没有调用 feature extractor、matcher、mapper 或 bundle adjuster。

COLMAP 默认将模型文件写在 `sparse/`；为符合当前 Binocular3DGS loader 的固定约定，仅在新副本内将 `cameras.bin`、`images.bin`、`points3D.bin`、`frames.bin`、`rigs.bin` 移入 `sparse/0/`。模型内容没有手工修改。

最终 loader 所需结构：

```text
D:\dataset\nerf_llff_data_colmap\horns\
├── images\                 62 × .jpg，4003×3002
└── sparse\0\
    ├── cameras.bin
    ├── images.bin
    ├── points3D.bin
    ├── frames.bin
    └── rigs.bin
```

COLMAP 还生成了 `stereo/` 及两个 MVS shell script；当前 loader 不读取它们。

## 3. 相机模型、K、尺寸与视野变化

### 3.1 转换前

- camera ID：1
- model：`SIMPLE_RADIAL`（model ID 2）
- 尺寸：4032×3024
- 参数：`f=3368.8237176028883, cx=2016, cy=1512, k=0.012300143763124053`

```text
K_original = [[3368.8237176028883, 0,                  2016.0],
              [0,                  3368.8237176028883, 1512.0],
              [0,                  0,                  1.0]]
```

### 3.2 转换后

- camera ID：1
- model：`PINHOLE`（model ID 1）
- 尺寸：4003×3002
- 参数：`fx=3368.8237176028883, fy=3368.8237176028883, cx=2001.5, cy=1501`

```text
K_undistorted = [[3368.8237176028883, 0,                  2001.5],
                 [0,                  3368.8237176028883, 1501.0],
                 [0,                  0,                  1.0]]
```

输出尺寸与新 K 完全配套；主点位于输出图像中心。使用 `image_undistorter` 默认 `blank_pixels=0` 后，输出相对原图总计裁去 29 列和 22 行，以避免无效空白区域。因此视野/边界有轻微裁剪，输出并非单纯修改 camera model ID，也不是删除 `k` 后冒充 pinhole。

### 3.3 `resolution=8`

现有训练 reader 使用 `round(width/8), round(height/8)`，PDC 路径使用整数除法；两者对 4003×3002 均得到 500×375。PDC 使用：

```text
K/8 = [[421.10296470036104, 0,                  250.1875],
       [0,                  421.10296470036104, 187.625],
       [0,                  0,                  1.0]]
```

由于新尺寸不能被 8 整除，实际有效除数为 `(8.006, 8.0053333333)`；相对理想 1/8 缩放的差异约为 x=0.075%、y=0.067%，K 主点与缩放图中心分别相差 0.1875 px 和 0.125 px。该误差来自现有整数输出尺寸规则，小于 1 像素，训练与 PDC 的输出尺寸一致，判定为 **合理 / PASS**；后续精度敏感实验应保留此记录。

## 4. 图像、ID、相机与位姿对应

- 转换前图像：62；转换后图像：62。
- 输出 62/62 均通过真实解码，并读取首尾像素；尺寸全部为 4003×3002。
- 原始与输出文件名集合逐项一致，集合差异数为 0；扩展名和大小写均未改变，故映射为同名一对一映射，不依赖目录排列顺序。
- 原始 62 个 image ID 为 1–62；输出仍为 1–62。
- 所有 image records 转换前后都引用 camera ID 1。
- 对按 image ID 对齐的 62 条记录逐字段比较：`image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, image_name` 差异数为 0。
- 上述 pose-record 规范化聚合 SHA-256 在转换前后均为 `55BD3935D8D97A4926CBFA138E4D3275C628EAE95496B426B33F4843B2B17664`。
- 62 个输出图像与原始同名文件的内容 SHA-256 均不同（相同对数 0），且尺寸/K 已改变，与真实像素重映射结果一致。
- `points3D.txt`、`frames.txt`、`rigs.txt` 经 COLMAP model converter 转成文本后的前后 SHA-256 完全相同。`points3D.bin` 的二进制 hash 不同是 COLMAP 3.13 重新序列化造成；文本模型内容相同。
- 所有 62 条输出 COLMAP image records 都能映射到 `images/` 的实际文件。

原始 reconstruction 含全部 62 个已注册视角，因此现有相机位姿是全视角 COLMAP 重建的产物。此事实只描述 pose 来源；本任务下方的初始化白名单仍严格只允许 3 张训练图像像素。两者没有混为一谈，也没有重新以 3 张图执行 SfM/BA。

## 5. 3-view manifest 与无泄漏协议

新 manifest：`output/a2a_fix1_horns_manifest/view_manifest.json`  
run record：`output/a2a_fix1_horns_manifest/run_manifest.json`

真实训练/PDC 三视角：

1. `DJI_20200223_163017_967.jpg` — image ID 2，camera ID 1
2. `DJI_20200223_163053_863.jpg` — image ID 31，camera ID 1
3. `DJI_20200223_163225_243.jpg` — image ID 62，camera ID 1

真实测试视角：

1. `DJI_20200223_163016_842.jpg`
2. `DJI_20200223_163024_597.jpg`
3. `DJI_20200223_163033_432.jpg`
4. `DJI_20200223_163047_307.jpg`
5. `DJI_20200223_163055_437.jpg`
6. `DJI_20200223_163103_777.jpg`
7. `DJI_20200223_163201_633.jpg`
8. `DJI_20200223_163208_938.jpg`

检查结果：

- `selected_train_images == pdc_initialization_images`：PASS（逐项、逐序相同）。
- `test_images ∩ pdc_initialization_images == ∅`：PASS。
- manifest 的 62 个规范化文件名与输出 COLMAP records/实际图像一致：PASS。
- 三个选中视角共享同一 camera ID、同一 4003×3002 尺寸和同一 `K/8`：A1-Fix1 `validate_shared_camera_intrinsics()` PASS。
- 现有 PDC 路径按 manifest 白名单构造精确的 3 文件读取集合；测试图像不进入初始化图像张量：PASS（CPU 静态/协议验证；未运行 PDC）。

## 6. 验收表

| # | 条件 | 结果 | 证据摘要 |
|---:|---|---|---|
| 1 | 前后均为 62 张图像 | PASS | 62 → 62 |
| 2 | 所有图像真实解码 | PASS | 输出 62/62，0 errors |
| 3 | 新模型为 PINHOLE/SIMPLE_PINHOLE | PASS | `PINHOLE`, model ID 1 |
| 4 | 新 K 与尺寸一致 | PASS | 4003×3002，主点 `(2001.5,1501)` |
| 5 | image_name 一一对应 | PASS | 同名集合差异 0，无扩展名/大小写变化 |
| 6 | image/camera/pose 可审计 | PASS | 62 条逐字段对比；文本模型留档 |
| 7 | 位姿无非预期改变 | PASS | pose-record 差异 0、聚合 hash 相同 |
| 8 | records 映射实际图像 | PASS | 62/62 |
| 9 | `read_and_validate_colmap_scene()` | PASS | 返回 62 records |
| 10 | 训练视角与 PDC 集合一致 | PASS | 三个文件逐项逐序一致 |
| 11 | 测试与初始化零交集 | PASS | `[]` |
| 12 | 三视角共享 K/尺寸 | PASS | A1-Fix1 validator 通过 |
| 13 | resolution=8 尺寸/K 合理 | PASS | 500×375；整数取整误差 <0.075%、<1 px |
| 14 | 原始目录未修改 | PASS | 转换前后整树 hash 完全一致 |

补充：`python -m unittest discover -s tests -v` 为 **17/17 PASS**。

## 7. 数据与审计哈希

原始目录在转换前后均为：

```text
file_count = 194
total_bytes = 471742318
tree_sha256 = 4DC9E72C33730CC00C2D321344ED6628858A82455CA683A5A997BA1C0F5E7E92
```

新副本核心树（`images/` + `sparse/0/`）：

```text
file_count = 67
total_bytes = 414894063
tree_sha256 = A368FB5ADD21CB92A68CBE0E800DA7D5D553A074EDCA203DDB65216768BFD4CF
```

关键输出：

```text
F328EFA43A68CD47A75E6C9BDB81DEBC5438D75735493F6EAD7ECEF03523C868  cameras.bin
9769D67BA9E5A152791C537B767E98317655A782CAC5F7557411975A5D9EDFDC  images.bin
B00085FFDDABB93733A2E1121D7F64F5EF17D36A6C151AB764A70210406DB8A0  points3D.bin
A43D544C48F4767369AB37CF9FF82106F1D64365CB0E03D129559D297108586B  view_manifest.json
3BCDFEF4DBE34F3627EE53A3C4B96EB4CFF638BD19C9541A1C29C75A7DDDC9A9  run_manifest.json
```

审计用 source/target COLMAP 文本模型保存在 `output/a2a_fix1_horns_audit/{source_model_txt,target_model_txt}/`；它们用于逐字段对比，不被 loader 使用。

## 8. 源码状态冻结

HEAD：`01677f59e9a23abcffc95a097ca59b3964fa03fd`。工作树包含 A1/A1-Fix1 的未提交修改，不能仅靠 HEAD 复现。

已导出当前 4 个 tracked 源码修改的 binary patch：

```text
output/a2a_fix1_horns_audit/tracked_worktree.patch
size = 16086 bytes
sha256 = FF15D11886A6AFAE55A221A061EFEE57E57B5983CEF684400300F0F28FE79C2C
```

staged patch 为空：

```text
output/a2a_fix1_horns_audit/staged_worktree.patch
size = 0 bytes
sha256 = E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855
```

实际代码文件及 SHA-256：

```text
FAC48363FF5595A82C578D677257527B44A2666EF8ABC57A91F5621001B87361  arguments/__init__.py
47E26E01F9E034DA357863035329E22238441B6E2BE824F9D30060517E1C12AB  scene/__init__.py
F83E6A9E8002AFF35D7A29DB981C96FE5CAA3CF12B8A13A89BDFA8C0CE8BA125  scene/dataset_readers.py
0397F359C47EF9DA3532B295532B5D0F40B48542D03AA1815DAA9A2FE609A598  submodules/dense_matcher/triangulate.py
581A8CE5996C2D52BC69682B2F3DDBBEB149902AC60A59FB6637FD88D8C49D8F  llff_view_protocol.py
32A4C282CA94735D8E87C5FACBA3225B994D581A972AB8B102903A59DCD3F975  script/run_llff_single.py
3436AD4DDD47E82501B4FCD44A3AEA0BA7919346842EC167866326530B53BD01  tests/test_llff_a1.py
```

以上 7 个代码文件按路径+文件 hash 聚合后的 SHA-256：

```text
451FE644769C7BE490AA9E5EBEBE57D6C80B20D640FAF8BEE88AF5BE086E0E21
```

`git diff --check` 通过；未 commit、未 push。审计过程未修改上述算法源码。

## 9. 最终边界

本报告只确认 A2-A-Fix1 的真实去畸变副本和 CPU 协议验收完成。它不代表 PDC 权重/环境、CUDA 扩展、GPU 初始化、训练或渲染已经可用。按任务边界，完成后停止并等待验收。
