# A2-B3 Frozen Initialization Artifact Audit

## 1. 结论

**固定初始化 PLY 复用接口：PASS。真实训练 Smoke：CONDITIONAL GO，当前尚不能启动。**

Windows 单场景 runner 现在可接收一份外部生成且已冻结的 PDC 初始化 PLY，在只使用 Python 标准库的情况下完成格式、顶点 schema、顶点数、SHA-256、逐字节复制和覆盖保护。复制后的 artifact 与当前实验 view manifest 绑定，train stage 在启动子进程前再次校验两者。

本阶段没有生成真实 PLY，没有调用 PDC 初始化，没有运行真实 Horns 训练、CUDA 或 GPU 测试，也没有修改 renderer、GaussianModel、Dropout、Binocular Consistency、densification 或 fsgs 环境。

真实 Smoke 的剩余硬条件是取得经过 AutoDL provenance 审计的真实 Horns PLY 及其 SHA-256。当前该文件不存在，因此本报告不将真实训练标为 PASS，也没有启动 5k/3k。

## 2. 修改文件

- 修改：`script/run_llff_single.py`
- 新增：`tests/test_a2b3_frozen_initialization.py`
- 新增：`output/a2b3_frozen_initialization/cpu_tests.txt`
- 新增：`output/a2b3_frozen_initialization/train_only_5k_preview.txt`
- 新增：`A2B3_Frozen_Initialization_Artifact_Audit.md`

`script/run_llff_single.py` 在本任务开始前已经是 A1 的未跟踪文件，不能由普通 `git diff` 得到相对 HEAD 的有效增量。A2-A-Fix1 冻结的任务前 SHA-256 是：

```text
32A4C282CA94735D8E87C5FACBA3225B994D581A972AB8B102903A59DCD3F975
```

A2-B3 当前 SHA-256 是：

```text
C27F2F20E6101B8E1B463AA75BA41C676BA2AD80CB5092E7DD2037E600F717DB
```

新增测试文件 SHA-256：

```text
19C86D1D035BA4A4D9A40F7097546017DC71806ECE87063BF615363873D0A2F3
```

Git HEAD 仍为 `01677f59e9a23abcffc95a097ca59b3964fa03fd`；没有 commit/push。既有 A1/A1-Fix1/A2 未提交内容未被覆盖或回退。

## 3. PLY 接口

### 3.1 新参数

```text
--initialization-ply <external.ply>
--expected-initialization-sha256 <64 hexadecimal characters>
```

两项必须同时提供。指定外部 PLY 时，`init` stage 被拒绝，防止 PDC 重新生成并覆盖冻结 artifact；应使用 `--stages train`。未指定外部 PLY 时保持 A1 原行为，仍可选择原 `manifest/init/train/render/metrics` 阶段。

本地 GPU stage 的默认解释器已改为经过 A2-B1/A2-B2 验证的绝对路径：

```text
D:\Users\huashuo\anaconda3\envs\fsgs\python.exe
```

不会再因省略 `--python` 而默认落到缺少 torch 的系统 Python 3.10。`--python` 仍保留为显式覆盖参数。

### 3.2 目标路径

目标文件遵循现有 loader 的实际约定：

```text
keypoints_to_3d/LLFF_<experiment-name>/<scene>_keypoints_to_3d.ply
```

例如：

```text
keypoints_to_3d/LLFF_horns_baseline_3view_5k/horns_keypoints_to_3d.ply
```

没有改动 dataset reader 或训练入口，也没有新增 Gaussian 数量、颜色或点顺序适配。

## 4. 格式和内容验证

runner 的标准库 PLY reader 会解析完整 header 与 payload，支持：

- `ascii 1.0`
- `binary_little_endian 1.0`
- `binary_big_endian 1.0`
- PLY 标准标量类型及 list properties

验证规则：

- 文件必须存在且非空；
- 必须有唯一 `vertex` element；
- vertex count 必须处于 `[1, 100,000,000]`；
- 必须有标量 `x/y/z/red/green/blue`；法线和 alpha 等附加字段允许存在；
- 顶点属性名不得重复；
- 全部 payload 必须完整可解析，拒绝截断或意外尾随数据；
- 坐标必须有限；RGB 必须有限且处于 `[0,255]`。

这些字段与当前 `scene/dataset_readers.py::fetchPly()` 的读取要求一致。测试包含带 RGBA 和空 face element 的 binary little-endian PLY，以覆盖 `trimesh` 常见导出结构。

## 5. 哈希与覆盖保护

处理顺序如下：

1. 源文件存在性检查；
2. 在解析或复制前计算源 SHA-256，并与显式 expected hash 比较；
3. 解析和验证完整 PLY；
4. 若目标已存在：仅当 SHA-256 完全一致时复用；不同则立即拒绝，绝不覆盖；
5. 若目标不存在：复制到目标目录内唯一临时文件，flush + `fsync`；
6. 校验临时文件 SHA-256；
7. 通过原子、no-clobber 的 hard-link 创建最终目标；并发出现的目标同样必须 hash 相同；
8. 再次解析目标并核对 hash、vertex count；
9. 再算源 hash，确认复制期间源文件未改变。

复制不进行 PLY 重写、字段转换、抽样、重排或参数调整。因此通过时源和目标字节完全一致。

## 6. run_manifest.json 记录与训练前 gate

使用外部 PLY 时，run manifest 使用 `llff-single-run-v2` 并新增 `initialization_artifact`；未指定外部 PLY 时继续使用原 A1 的 v1 schema。成功记录包含：

- mode/status/copy action；
- expected SHA-256；
- source 与 destination 的绝对路径、字节数、实际 SHA-256、PLY format；
- vertex count 与完整 vertex property 列表；
- `source_unchanged_after_copy` 和 `byte_identical`；
- 当前 `view_manifest.json` 的 SHA-256；
- dataset、scene、view count、selected training images、PDC initialization images。

若 staging 失败，run manifest 写入 failed 状态和错误，不启动训练。

train stage 在创建训练子进程前再次验证：

- view manifest 当前 SHA-256 等于 staging 时记录；
- destination PLY 存在且 SHA-256 等于 expected/source；
- PLY 仍可完整读取且 vertex count 未变。

任一不一致都会把 train stage 标为 failed，并在 `subprocess.run()` 前抛错。

## 7. Baseline/I1/I2 共享初始化验证

CPU 测试使用一份合成 binary PLY，分别 staging 到：

- `LLFF_baseline/...ply`
- `LLFF_i1/...ply`
- `LLFF_i2/...ply`

验证结果：

- 三个实验的 source SHA-256 完全相同；
- 三个 destination SHA-256 均与 source 相同；
- 三个目标内容均与 source 逐字节相同；
- source 在三次复制后字节未改变；
- 三份 view manifest 字节与 SHA-256 完全一致；
- 三份 `run_manifest.json` 均记录相同的 source SHA-256，且各自 destination SHA-256 与 source 相同；
- 没有调整点数、颜色、点顺序或训练参数。

另一个测试在 staging 后修改 view manifest 的 `source_commit`，train 命令的 marker 文件未生成，证明不一致在训练子进程启动前被拒绝。

需要注意：PLY 格式本身不包含“由哪份 manifest 生成”的标准字段。接口能够冻结用户提供的 PLY、将其绑定到当前实验 manifest 并防止后续漂移；它不能仅凭 PLY 内容反向证明源端 PDC 使用了该 manifest。因此 AutoDL 生成时仍须一并保存生成端 manifest、代码/权重 hash 和运行记录，之后用这里的 expected PLY hash 接续审计链。

## 8. CPU 单测

```text
D:\Users\huashuo\anaconda3\envs\fsgs\python.exe -B -m unittest discover -s tests -v
Ran 26 tests in 1.368s
OK
```

- A2-B3 新增测试：9/9 PASS
- 既有测试：17/17 PASS
- `git diff --check`：PASS

新增测试逐项覆盖：

1. 合法 binary PLY 验证并逐字节复制；artifact metadata 写入 JSON；
2. 错误 hash 立即终止；
3. 文件缺失立即终止；
4. 缺少关键字段的非法 PLY 立即终止；
5. 已存在不同内容的目标拒绝覆盖且原目标不变；
6. Baseline/I1/I2 共享同一源 PLY；
7. manifest 改变时拒绝启动训练进程；
8. 无 PDC init 的 train-only 5k 命令使用 fsgs Python；
9. 完整 ASCII PLY payload 可读。

测试只使用临时目录、标准库和合成 PLY；没有导入/运行 CUDA、PDCNet+ 或真实模型。

## 9. Train-only 5k 命令预览

以下只作为参数模板，未执行。`<...>` 必须替换成 AutoDL 返回的真实冻结 PLY 和其 64 位 SHA-256：

```powershell
& 'D:\Users\huashuo\anaconda3\envs\fsgs\python.exe' `
  'D:\3DGS\Binocular3DGS + Dropout\script\run_llff_single.py' `
  --profile 5k `
  --scene horns `
  --dataset-root 'D:\dataset\nerf_llff_data_colmap' `
  --output-root 'D:\3DGS\Binocular3DGS + Dropout\output' `
  --experiment-name horns_baseline_3view_5k `
  --initialization-ply '<EXTERNAL_FROZEN_HORNS_PLY>' `
  --expected-initialization-sha256 '<64_HEX_SHA256>' `
  --stages manifest train `
  --dry-run
```

这里显式包含 `manifest` 是为了让全新实验目录也能独立完成 view manifest 生成与冻结 PLY staging；唯一预览的 GPU 子命令仍是 `train`，没有 `init`/PDC 子命令。若该实验的 `view_manifest.json` 已在前一阶段生成并验证，则可改用 `--stages train --dry-run`。

生成的 5k train 子命令为：

```text
D:\Users\huashuo\anaconda3\envs\fsgs\python.exe "D:\3DGS\Binocular3DGS + Dropout\train.py" -s D:\dataset\nerf_llff_data_colmap\horns -m "D:\3DGS\Binocular3DGS + Dropout\output\horns_baseline_3view_5k" --iterations 5000 --dropout_factor 0.2 --sigma_noise 0.0 --suffix horns_baseline_3view_5k --n_views 3 --resolution 8 --dataset_name LLFF --view_manifest "D:\3DGS\Binocular3DGS + Dropout\output\horns_baseline_3view_5k\view_manifest.json" --eval
```

runner 会将训练的完整命令、开始/结束 UTC、合并后的完整 stdout/stderr 和退出码保存到：

```text
output/horns_baseline_3view_5k/train.log
```

5k 保持首选；接口没有自动回退 3k，也没有自动修改任何训练参数。只有未来实际 5k OOM 并完成分析后，才应由用户显式选择 3k。

## 10. 仍缺少的 PDC 初始化前置条件

根据 A2-B1 的实际环境审计，本地 PDC 初始化仍为 NO-GO：

- CuPy：MISSING；
- trimesh：MISSING；
- PDCNet+ checkpoint：MISSING；
- checkpoint 加载及 CuPy local-correlation API/kernel：NOT TESTED。

本接口的 train-only 路径不导入 `triangulate.py`，所以复用已冻结 PLY 时不要求本地存在上述依赖。没有进行任何安装、下载或模型加载。

## 11. 是否能够进入真实训练 Smoke

| 层级 | 状态 | 说明 |
|---|---|---|
| 冻结 PLY 接口 | PASS | 格式、hash、复制、覆盖、manifest gate 均经 CPU 测试 |
| fsgs CUDA/renderer | PASS | 来自 A2-B1/A2-B2 的最小 kernel 与项目 renderer 证据 |
| 真实初始化 PLY | MISSING | 本阶段明确不生成，尚无可供 runner 接受的真实文件/hash |
| 真实 Horns 训练 | NOT TESTED | 禁止运行 |

因此当前判定为 **CONDITIONAL GO**：代码接口已具备进入真实训练 Smoke 的条件，但在真实冻结 PLY、expected SHA-256 及 AutoDL provenance 到位并通过 staging 之前，不允许启动训练。到位后下一阶段应先显式 dry-run 审计命令和 run manifest，再按独立授权运行本地 5k smoke；本阶段到此停止。
