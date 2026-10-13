# A1: LLFF 无泄漏协议与 Windows 单场景入口

本阶段只建立可审计基础设施，不改变 renderer、GaussianModel、Dropout、Opacity Decay、Densification、Binocular Consistency 或 I1/I2。

## 协议

LLFF 视角由仓库根目录 `llff_view_protocol.py` 唯一计算：

1. 使用 COLMAP `image_name`（不是 image ID）进行规范化和字典序排序。
2. 排序后满足 `index % 8 == 0` 的图像是测试集。
3. 从其余训练候选中用包含首尾的均匀位置选择 n views。
4. `selected_train_images` 必须逐项等于 `pdc_initialization_images`。
5. 测试集与 PDC 初始化集合必须不相交。
6. COLMAP image/camera 引用、重复标识和磁盘图像均在任何 GPU 阶段前校验。

PDC 初始化在 manifest 校验后建立只含 `selected_train_images` 的局部视角表；图像、外参、内参和局部索引使用同一顺序，仅这 3 张图像会被读取并进入 GPU tensor。当前 PDC 后续 patch projection/point sampling 复用一套 `K/focal/center`，因此入口显式验证选中视角的内参矩阵与图像尺寸一致；不一致时 fail closed，而不是静默使用第 0 张相机内参。

训练 reader 和 PDC triangulate 都读取同一个 `view_manifest.json`。不提供 manifest 的旧入口也会调用同一纯函数，但正式开发运行应始终使用新 runner 生成并传递 manifest。

## PowerShell 调用

安全默认只生成 view/run manifest，不启动 GPU 阶段：

```powershell
python .\script\run_llff_single.py --profile 5k --experiment-name horns_a1_5k
```

显示并记录完整命令但不执行 GPU 阶段：

```powershell
python .\script\run_llff_single.py --profile 5k --experiment-name horns_a1_5k --stages manifest init train render metrics --dry-run
```

显式执行阶段（A1 不执行此命令）：

```powershell
python .\script\run_llff_single.py --profile 5k --experiment-name horns_a1_5k --stages manifest init train render metrics
```

诊断 OOM 后，由用户单独选择 3k 配置重新运行；runner 不会自动 fallback：

```powershell
python .\script\run_llff_single.py --profile 3k --experiment-name horns_a1_3k --stages manifest init train render metrics
```

也可只执行某个阶段。省略 `manifest` 时，runner 要求同一实验目录中已有 manifest，并用当前 COLMAP model 重新校验：

```powershell
python .\script\run_llff_single.py --profile 5k --experiment-name horns_a1_5k --stages train
```

所有 subprocess 使用 argv list、绝对路径、固定仓库 cwd 和显式环境字典，不使用 shell 命令拼接或全局 `os.chdir()`。任何非零退出码都会停止后续阶段。

## 开发配置

| 参数 | 5k profile | 3k profile |
|---|---:|---:|
| scene | horns | horns |
| n_views | 3 | 3 |
| iterations | 5000 | 3000 |
| dropout_factor | 0.2 | 0.2 |
| sigma_noise | 0.0 | 0.0 |
| resolution | 8 | 8 |
| eval | true | true |
| GPU | 单卡 0 | 单卡 0 |

`arguments.OptimizationParams.iterations` 仍为官方默认 30000；开发预算只由 runner 的 `--iterations` 参数传入。

## 输出

默认位于 `output/<experiment-name>/`：

```text
view_manifest.json
run_manifest.json
init.log
train.log
render.log
metrics.log
cfg_args
point_cloud/...
test/...
```

初始化 PLY 使用隔离目录 `keypoints_to_3d/LLFF_<experiment-name>/`，训练通过已有 `--suffix` 读取同一目录。

## Manifest 示例

```json
{
  "dataset": "LLFF",
  "scene": "horns",
  "sorted_all_images": ["image_000.png", "image_001.png", "..."],
  "test_images": ["image_000.png", "image_008.png", "..."],
  "train_candidate_images": ["image_001.png", "image_002.png", "..."],
  "selected_train_images": ["image_001.png", "image_013.png", "image_024.png"],
  "pdc_initialization_images": ["image_001.png", "image_013.png", "image_024.png"],
  "view_count": 3,
  "split_rule": "sort(image_name); test=index%8==0; uniformly select n views from remaining candidates",
  "source_commit": "01677f59e9a23abcffc95a097ca59b3964fa03fd",
  "manifest_version": "llff-view-manifest-v1"
}
```

示例文件名仅展示 schema；真实内容由场景 COLMAP model 生成并校验。

## CPU 单元测试

```powershell
python -B -m unittest discover -s tests -p "test_llff_a1.py" -v
```

测试不导入 rasterizer、PDCNet+ 或 CUDA，不运行三角化/训练/渲染。
