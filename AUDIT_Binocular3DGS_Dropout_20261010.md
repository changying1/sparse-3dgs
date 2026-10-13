# Binocular3DGS + Gaussian Dropout 完整宿主静态审计

审计日期：2026-10-10  
仓库：`D:\3DGS\Binocular3DGS + Dropout`  
审计对象：`01677f59e9a23abcffc95a097ca59b3964fa03fd`  
审计模式：**Audit Only（静态、只读）**

## 1. Executive Summary

### 结论

**总体判定：CONDITIONAL GO。** 该仓库仍值得作为第一优先候选宿主，因为它提供了强于 FSGS 的公开结果、稠密匹配初始化、双目一致性和 Gaussian Dropout；同时，现有方法没有实现 I1 的 Observation-Evidence Guided Structural-Prior Authorization，也没有实现 I2 的出生后效用追踪与闭环资源再分配，创新空间真实存在。但在任何正式比较前，必须先通过下列阻断 Gate：

1. **修正并冻结无泄漏 LLFF 初始化协议。** 发布的 `triangulate.py` 将训练集内部位置索引误作全图像索引，索引 0 会进入初始化，而索引 0 又被定义为测试视角，属于 **CONFIRMED split leakage**（`submodules/dense_matcher/triangulate.py:104-118`；`scene/dataset_readers.py:171-175`）。
2. **不要直接执行发布的 `script/run_llff.py`。** 它在三角化前全局 `chdir`，随后继续使用相对数据/日志路径；多线程共享 cwd；四个 `os.system()` 返回码全部被忽略；Windows 上 POSIX 环境变量前缀也不可用（`script/run_llff.py:31-67,72-109,162-165`）。
3. **建立隔离环境并验证冻结 rasterizer ABI/API。** rasterizer submodule 未初始化，主代码要求其 forward 返回 image、radii、depth、alpha 四项，但目标源码当前不可审计；现有 `fsgs` 环境为 Python 3.13/PyTorch 2.8/CUDA 12.9，与仓库声明的 Python 3.10、Torch 2.1.1+cu118 明显偏离（`.gitmodules:1-3`；`README.md:38-47`；`requirements.txt:11-12`；`gaussian_renderer/__init__.py:117-126`）。
4. **先在 AutoDL 24GB 做官方基线可行性验证，再移植 I1。** 8GB 本地只承担静态检查、协议单元检查、100–300 iteration smoke，以及在显存证据允许时的单场景 densification smoke；不能把本地 3k/5k 无收益作为 I1 的 NO-GO 证据。

### 核心事实与置信度

| 结论 | 类型 | 状态 |
|---|---|---|
| HEAD、origin 和冻结 commit 匹配 | FACT | CONFIRMED |
| rasterizer submodule 目标为 `8829d14...`，工作树未初始化 | FACT | CONFIRMED |
| `simple-knn`、`dense_matcher` 为主仓库内跟踪目录 | FACT | CONFIRMED |
| 发布 LLFF 脚本值是 3 views、resolution=2、dropout=0.3、noise=0 | FACT | CONFIRMED |
| README 论文表为 LLFF 8× 下采样，但脚本/训练实际为 2× | FACT | CONFIRMED；协议不等价 |
| 三角化阶段会使用测试视角 0 | FACT | CONFIRMED |
| 主渲染 Dropout，shifted-camera 渲染不 Dropout | FACT | CONFIRMED |
| 现有 `fsgs` 环境可安全复用 | UNKNOWN | NEEDS-RUNTIME-VERIFICATION；不建议原地试装 |
| 8GB 能承受完整 30k | UNKNOWN | NEEDS-RUNTIME-VERIFICATION；高 OOM 风险 |

## 2. Repository & Dependency Audit

### 2.1 Git 与目录结构

只读 Git 检查结果：

- `HEAD = 01677f59e9a23abcffc95a097ca59b3964fa03fd`；分支为 `main...origin/main`，工作树在审计开始时无未提交项。
- `origin = https://github.com/chenkangjie1123/Co-Adaptation-of-3DGS.git`。
- `.gitmodules` 只定义一个 submodule：`submodules/diff-gaussian-rasterization`，来源 `git@github.com:ashawkey/diff-gaussian-rasterization.git`（`.gitmodules:1-3`）。
- Git tree 将该 submodule 固定到 `8829d14f814fccdaf840b7b0f3021a616583c0a1`；`git submodule status` 前缀为 `-`，目录为空，即未初始化。
- `submodules/simple-knn` 与 `submodules/dense_matcher` 是普通 tracked tree，而非 `.gitmodules` 条目。前者含 CUDA 扩展源码，后者含 PDCNet/GOCor 实现。

压缩后的完整顶层树（共 220 个 tracked paths）：

```text
.
├─ arguments/                 参数定义
├─ gaussian_renderer/         rasterizer 包装、Dropout/opacity noise
├─ lpipsPyTorch/              内置 LPIPS 包装
├─ scene/                     Camera、dataset reader、GaussianModel
├─ script/                    LLFF/DTU/Blender 调度脚本
├─ submodules/
│  ├─ diff-gaussian-rasterization/  空；Git submodule
│  ├─ simple-knn/             vendored CUDA extension source
│  └─ dense_matcher/          PDCNet+、GOCor、triangulate
├─ utils/                     loss、warp、camera、图像工具
├─ train.py
├─ render.py
├─ metrics.py
├─ spiral.py
├─ read_eval_result.py
├─ requirements.txt
├─ README.md
└─ .gitmodules
```

### 2.2 必需依赖与缺口

**FACT。** `requirements.txt` 固定了 GPUtil、imageio、matplotlib、moviepy、numpy、OpenCV、plyfile、scikit-image、scipy、timm、torch、torchvision、tqdm、trimesh，其中 Torch/torchvision 为 `2.1.1+cu118` / `0.16.1+cu118`（`requirements.txt:1-14`）。README 推荐 Python 3.10，并要求安装 rasterizer 和 simple-knn（`README.md:38-47`）。

**FACT。** 两个训练期 CUDA 包均不可仅靠纯 Python 源码运行：

- `simple_knn._C` 由 `CUDAExtension` 构建（`submodules/simple-knn/setup.py:12-34`），并被 `scene/gaussian_model.py:20,134` 直接导入/调用。
- `diff_gaussian_rasterization` 被 `gaussian_renderer/__init__.py:14` 直接导入；当前 submodule 空，因此本仓库自身没有可用实现。

**FACT。** PDCNet+ 默认需要 `submodules/dense_matcher/pre_trained_models/PDCNet_plus_megadepth.pth`，若不存在则尝试同名 `.pth.tar`，再无则抛错（`submodules/dense_matcher/triangulate.py:20-53`；`submodules/dense_matcher/model_selection.py:219-235`）。本地该目录不存在，README 也要求手动放置模型（`README.md:52-53`）。

**FACT。** `requirements.txt` 未列 CuPy，但 PDCNet 模块顶层导入 local correlation，而其实现直接 `import cupy` 并运行 CuPy CUDA kernel；默认 PDCNet_plus 又选择 LocalGOCor（`submodules/dense_matcher/models/PDCNet/PDCNet.py:8`；`submodules/dense_matcher/models/PDCNet/base_pdcnet.py:14`；`submodules/dense_matcher/model_selection.py:144-171`；`submodules/dense_matcher/models/modules/local_correlation/correlation.py:1-8,243-280`）。因此依赖清单不闭合。

### 2.3 当前 fsgs 环境能否复用

**INFERENCE：理论上“同一算法栈”可复用，当前环境不能认定为安全可复用。** 原因：

- 当前 Python 3.13.5 / Torch 2.8.0+cu129 / CUDA 12.9 与项目声明栈不一致；旧版 Python 包、CuPy kernel 和 CUDA extensions 的兼容性均需运行验证。
- `simple_knn._C` 与 `diff_gaussian_rasterization` 使用通用 import 名称。若 fsgs 环境中已有同名扩展，Python 可能加载旧包；即使导入成功，rasterizer 返回签名也可能不符。本仓库期望四返回值（`gaussian_renderer/__init__.py:117-126`），而目标 submodule源码尚不可读。
- 原地重新编译/安装可能覆盖同名 wheel/egg、改变 Torch ABI 绑定或 CuPy 缓存，从而污染 FSGS。

结论：**NEEDS-RUNTIME-VERIFICATION；推荐创建隔离环境，不在 `fsgs` 中安装或编译。** 安全路径是：先记录现有环境清单和扩展来源；在独立 Python 3.10、与目标 GPU 驱动兼容的冻结环境中初始化依赖；只在 A1 做 import/API 探针。此处只提出路径，不执行命令。

### 2.4 Windows 兼容性

- `simple-knn` 对 MSVC 只增加 `/wd4624`，说明有 Windows 意图，但不证明当前 CUDA 12.9 + Torch 2.8 + Python 3.13 可编译（`submodules/simple-knn/setup.py:16-30`）。**NEEDS-RUNTIME-VERIFICATION**。
- CuPy JIT kernel、目标 rasterizer、MSVC/NVCC 版本组合均未由源码给出支持矩阵。**NEEDS-RUNTIME-VERIFICATION**。
- `CUDA_VISIBLE_DEVICES=... python ...` 是 POSIX shell 语法，Windows `os.system()` 不会以同样语义设置环境。**CONFIRMED**（`script/run_llff.py:36-67`）。
- 脚本路径未引用含空格参数；当前仓库路径包含空格，命令字符串存在拆词风险。**CONFIRMED POTENTIAL FAILURE**（同上）。

## 3. Official LLFF Protocol Audit

### 3.1 三套协议必须分开

| 协议 | 静态证据 | 结论 |
|---|---|---|
| Paper-reported | README：3 views，LLFF 原图 8× 下采样，Dropout 表格结果 22.12/0.777/0.154；脚本注释称 paper 用 p=0.2 | 不能由发布脚本等价复现（`README.md:109-131`; `script/run_llff.py:13-16`） |
| Released-code | 8 场景、3 views、resolution=2、dropout=0.3、noise=0、30k | 源码明确，但 runner 有路径/并发/错误传播缺陷（`script/run_llff.py:7-17`; `arguments/__init__.py:73-92`） |
| Local development | Windows、8GB、Python 3.13/Torch 2.8/cu129 | 尚未运行；不得与上两者视为等价 |

### 3.2 视角划分及真实 ID

**训练端的定义（FACT）：** 相机先按 `image_name` 排序；测试集为排序后全序列中 `i % 8 == 0`；训练候选为其余视角；3-view 再从训练候选序列的 `[0, round((M-1)/2), M-1]` 三个位置取样，其中 M 是训练候选数（`scene/dataset_readers.py:149-175`）。因此按全序列索引表示，训练 ID 应为：

```text
T = [i | 0 <= i < N, i mod 8 != 0]
train_full_ids = [T[0], T[round((len(T)-1)/2)], T[len(T)-1]]
test_full_ids  = [i | i mod 8 == 0]
```

**UNKNOWN：** 仓库没有 LLFF 数据目录，无法静态列出 fern 等八场景的具体文件名/相机 ID；且 COLMAP image ID 与文件名字典序未被证明相同。必须在 A2 以实际数据 manifest 验证。

**三角化端的严重偏差（FACT）：** 它先构造 `train_idx`，再构造训练候选内部位置 `idx_sub`，却直接令 `ref_indices = idx_sub`，没有映射回 `train_idx[idx_sub]`（`submodules/dense_matcher/triangulate.py:104-118`）。结果：

- `idx_sub[0] = 0`，而全序列 0 满足 `0 % 8 == 0`，是测试视角；初始化明确使用了测试图像。
- 三角化相机按 COLMAP dictionary key 排序（`triangulate.py:60-75`），训练相机按 image_name 排序（`scene/dataset_readers.py:149-151`），两套顺序也不保证一致。
- 因而“初始化所用 3 views”与“训练所用 3 views”不只是未知，而是实现上没有对齐保证。

状态：**CONFIRMED DATA LEAKAGE / PROTOCOL MISMATCH。** 这会污染初始化点云，正式实验必须先消除；不能将当前公开数值直接称为无泄漏 LLFF 3-view 结果。

### 3.3 resolution 的真实含义

`resolution=2` 在相机加载时把宽高各除以 2，即线性 2× 下采样、像素数约为 1/4（`utils/camera_utils.py:21-43`）；三角化也将内参和图像宽高除以 2（`triangulate.py:90-96,120-127`）。因此发布脚本是 **2× linear downsample**，不是 README 声称的 LLFF 8×。这是 paper-reported 与 released-code 的另一项 **CONFIRMED** 差异。

### 3.4 数据、位姿、初始化产物

发布 runner 期望：

```text
data/nerf_llff_data_colmap/<scene>/
├─ images/<COLMAP image names>
└─ sparse/0/{cameras.bin,images.bin,...}
```

三角化读取 COLMAP 内外参和原图，用 PDCNet+ 在选中视角对间匹配、OpenCV triangulate，并将点云写为 `keypoints_to_3d/LLFF/<scene>_keypoints_to_3d.ply`（`triangulate.py:55-75,135-217,243,379-384`）。训练在 eval 模式下断言该文件存在，并将其作为 Gaussian 初始化点云；相机位姿仍来自数据集 COLMAP sparse model（`scene/dataset_readers.py:137-164,182-198`; `scene/__init__.py:44-45,78-84`）。

### 3.5 训练、渲染、指标与输出

- 训练默认 30,000 iter、SH degree 1、从 1,000 后每 100 iter densify（`arguments/__init__.py:47-90`）。
- `run_llff.py` 把输出设为 `output/LLFF/<scene>_3views`（`script/run_llff.py:7-10,27-32,45-50`）。训练保存 `cfg_args`、最终 point cloud（`train.py:172-176,217-229,293-310`; `scene/__init__.py:86-88`）。
- render 通过 `cfg_args` 恢复 source/model 配置，默认载入最大 iteration，并同时渲染 train/test 到 `<model>/<split>/ours_<iter>/{renders,gt}`（`arguments/__init__.py:94-114`; `render.py:97-125,182-217`）。
- metrics 读取 `test/*/{renders,gt}`，写 `results.json` 和 `per_view.json`（`metrics.py:48-69,93-129`）。runner 传 `--model_path`，而 parser 声明 `--model_paths/-m`；argparse 的唯一前缀缩写通常会接受它，但依赖默认 abbreviation 行为，属于脆弱接口（`script/run_llff.py:63-67`; `metrics.py:137-142`）。
- metrics 用裸 `except` 吞掉所有异常并正常结束（`metrics.py:126-131`）；runner 又忽略所有 `os.system()` 返回码，故失败不会可靠传播（`script/run_llff.py:42,52,61,67,69`）。

### 3.6 runner 的 cwd 与并发故障

**FACT。** runner 先在仓库根创建 `./output/LLFF/...`，但执行 triangulate 前切到 `submodules/dense_matcher`；此时命令仍使用 `./data/...` 和 `./output/.../log.txt` 相对路径（`script/run_llff.py:23-43`）。在干净仓库中：

- data path 会解析到 `submodules/dense_matcher/data/...`，不是根目录 data；
- shell 重定向会尝试写 `submodules/dense_matcher/output/.../log.txt`，其父目录未创建，POSIX shell 可在启动 Python 前就失败；
- `os.chdir()` 是进程全局状态，而最多 8 个线程同时更改 cwd，构成确定存在的竞态设计（`script/run_llff.py:72-109,162-164`）。

状态：**CONFIRMED HIGH RISK。** 当前发布 runner 不能被视为可靠的官方复现入口。

## 4. Dropout Implementation Audit

### 4.1 精确行为

**FACT。** 训练且 `dropout_factor>0` 时，为完整 N 个 Gaussian 生成长度 N 的均匀随机数，阈值 `< 1-p`；各元素独立调用 `torch.rand`，等价于独立 Bernoulli keep sampling（`gaussian_renderer/__init__.py:39-45`）。

**FACT。** keep mask 同时索引 means3D、means2D、SH、opacity、scale、rotation，然后只把保留子集送入 rasterizer（`gaussian_renderer/__init__.py:98-126`）。因此：

- Dropout 不只是改变可见性，而是从该次 rasterization 图中移除 Gaussian。
- 未保留 Gaussian 不从该次 photometric render 获得参数梯度；原始 full-length `screenspace_points` 的对应项梯度应为 0（INFERENCE，来自张量索引计算图）。
- rasterizer 返回的 `radii` 与 `visibility_filter` 长度是 kept count，不是 N；`viewspace_points` 与 `dropout_mask` 长度仍是 N（`gaussian_renderer/__init__.py:117-137`）。

### 4.2 densification 索引映射

训练通过 `true_indices = nonzero(dropout_mask)`，再以子集长度的 `visibility_filter` 选择它，最后重建 N 长度 `combined_mask`；radii 也以子集 visibility 过滤后写回 full Gaussian 数组（`train.py:184-192`）。在默认 `convert_SHs_python=False`、`compute_cov3D_python=False` 下，该映射逻辑静态上对齐，未发现 off-by-one。

**POTENTIAL。** 如果启用 Python SH 或 Python covariance，`colors_precomp` / `cov3D_precomp` 没有被 dropout mask 过滤，且 `shs` 或 `scales` 可能为 None 却仍被索引（`gaussian_renderer/__init__.py:68-93,98-105`）。默认 pipeline 两开关为 False（`arguments/__init__.py:66-71`），所以发布配置未触发，但模块并非真正 plug-and-play 安全。

### 4.3 测试期 scaling 与状态不一致

**FACT。** `train=False` 时执行 `opacity *= 1-p`，即用训练 keep probability 缩放完整集合 opacity（`gaussian_renderer/__init__.py:106-109`）。最终 render 明确传 `train=False` 和脚本 dropout factor（`render.py:107-109`）。

**FACT。** 训练内 validation 调用 renderer 时既未传 `train=False`，也未传 dropout factor，因此使用默认 `train=True, dropout_factor=0`，得到未缩放完整集合结果（`train.py:239-256`; renderer signature `gaussian_renderer/__init__.py:18-20`）。最终 metrics 则评估 opacity 缩放后的结果。状态：**CONFIRMED evaluation mismatch**。

**FACT。** shifted-camera 调用没有传 dropout factor，因此使用完整 Gaussian 集合；其 disparity 却来自主视角 Dropout 子集渲染的 depth（`train.py:101-111,130-143`）。状态：**CONFIRMED mixed-state binocular supervision**。

### 4.4 I2 所需观测变量

| 概念 | 当前可直接取得 | 仍需新增诊断 |
|---|---|---|
| real-view sampling opportunity | 当前 iteration 的 `viewpoint_cam`；但没有 per-Gaussian 投影机会计数 | 对每 UID 的 frustum/projected opportunity，且必须在 dropout 前定义 |
| dropout inclusion | `dropout_mask` 已返回 | 按 UID 累计 keep/exposure 次数；记录 p 和 RNG context |
| screen-space visibility | kept 子集的 `radii>0`，可经 combined mask 映射 | dropout 前潜在可见性；当前未保留未抽中 Gaussian 的 visibility counterfactual |
| rendering contribution | 当前没有 per-Gaussian contribution | 需 per-Gaussian alpha/颜色贡献或消融近似；仅 image alpha 不足 |
| future utility | 无 | 出生后窗口、跨真实视角贡献/误差改善、置信区间、删前反事实审查 |

关键定义：**Dropout exclusion 不是 zero utility，更不是 death。** I2 必须用 opportunity → inclusion → visibility → contribution 的条件链归一化，低 exposure 只进入 “insufficient evidence”，不能进入 donor。

## 5. Opacity Decay / Pruning / Growth Audit

### 5.1 opacity decay

`get_opacity = sigmoid(_opacity)`；decay 执行：

```python
opacity = sigmoid(_opacity) * factor
_opacity.data = inverse_sigmoid(opacity)
```

证据：`scene/gaussian_model.py:38-39,113-115,307-309`。

结论：

- **FACT：** 衰减发生在激活后的 opacity 空间，再映回 logit 参数；默认 factor=0.995（`train.py:298-300`）。
- **FACT：** 通过 `.data` 原位替换可训练参数值，没有替换 Parameter，也没有衰减/清零 Adam `exp_avg`、`exp_avg_sq`。
- **FACT：** 无显式 opacity 下界；factor 为正且 sigmoid 理论落在开区间，静态上不会主动 clamp。
- **FACT：** backward 后、optimizer.step 前每 iteration 执行，包括最后一轮；最后一轮不再 optimizer.step（`train.py:152-182,208-211`）。因此该步使用在衰减前参数处计算的梯度更新衰减后的 logit。

### 5.2 pruning、clone、split 与 optimizer state

- prune 阈值固定为 activated opacity `<0.005`（`train.py:194-199`; `scene/gaussian_model.py:393-405`）。
- `size_threshold` 先可能置 20，opacity decay 时置 None，随后无条件再次置 None，因此 size pruning 永不启用（`train.py:194-199`）。
- opacity reset 条件块只有 `pass`，实际 reset 被注释（`train.py:201-206`）。
- clone 原样复制 xyz、features、raw opacity、raw scale、rotation，parent 保留（`scene/gaussian_model.py:378-391`）。
- split 从 parent scale 分布采样 N=2 children，复制 features/raw opacity/rotation，缩小 scale；添加 children 后删除 selected parents（`scene/gaussian_model.py:353-376`）。因此 split parent 的 death reason 必须是 `split_replaced`，不能记为训练低效 pruning。
- 扩展参数时 Adam moments 对 newborn 追加 0；prune 时同步过滤 Parameter 与 moments（`scene/gaussian_model.py:273-305,311-351`）。
- `densification_postfix` 每次 clone/split 后把 **所有** Gaussian 的 `xyz_gradient_accum`、`denom`、`max_radii2D` 清零（`scene/gaussian_model.py:333-351`）；clone 与 split 连续调用，因此 clone 后的统计清零会影响紧随其后的 split 输入解释，尽管 `grads` 局部变量在调用前已计算。

### 5.3 无限期 growth 风险

默认 `densify_until_iter=15000`，但任何 iteration>1000 都将它覆盖为总 iterations，故 densification 持续到 29,999（`arguments/__init__.py:86-90`; `train.py:178-199`）。同时 size pruning 禁用、opacity reset 禁用，只剩 opacity<0.005 prune。状态：**CONFIRMED unbounded-by-cap growth policy；OOM 是 POTENTIAL/NEEDS-RUNTIME-VERIFICATION。**

### 5.4 opacity retention 能否作为 I2 utility

**INFERENCE：不能作为主效用标签。** opacity 受三股机制共同作用：每轮人为乘 0.995、Adam momentum 未同步重置、Dropout 改变获得梯度的机会。因此 `opacity_after / opacity_at_birth` 混合了基线先验、优化动态和曝光随机性，不是 baseline-agnostic utility。

建议的 post-growth evidence（建议，不是已验证行为）：

1. 条件化真实视角机会后的 visible/contributing view count；
2. per-Gaussian alpha-weighted pixel contribution 或受控 leave-one-out 误差增量近似；
3. 跨独立训练视角的一致残差改善，而非单视角高梯度；
4. normalized screen footprint × contribution 的时间积分；
5. evidence sufficiency：opportunity、kept、visible 的有效样本量和置信下界；
6. opacity 只作为辅助特征，并以已知 decay schedule 去趋势，不能单独触发 release。

## 6. Binocular Consistency Audit

### 6.1 实际机制

- `binocular_consistency` 为 `store_true, default=True`；普通 CLI 无 `--no-binocular-consistency`，不能关闭（`train.py:302-304`）。
- `opacity_decay` 同样为 `store_true, default=True`，也不能用常规 CLI 关闭（`train.py:298-304`）。ParamGroup 中所有 bool 默认也使用 `store_true`（`arguments/__init__.py:19-38`）。
- 默认 `shift_cam_start=20000`、`cam_trans_dist=0.4`；仅 `iteration > 20000` 生效（`train.py:131-134,302-304`）。
- shifted camera 沿原相机局部 x 方向随机正/负平移 `[0,0.4)`，通过 inverse world-view 映射到 world translation，再构造同 R/T/FoV 的 Camera（`scene/__init__.py:96-115`; `train.py:131-135`）。
- disparity 为 `focal_x * (-trans_dist) / (depth + 1e-5)`，depth 来自主视角 rasterizer 的 Dropout render（`train.py:101-111,137-140`）。
- `inverse_warp_images` 只做水平双线性采样，越界清零（`utils/graphics_utils.py:79-125`）。
- binocular loss = masked L1(warped shifted image, real GT) + 0.05 × edge-aware disparity smoothness（`train.py:139-143`; `utils/loss_utils.py:68-91`）。
- alpha loss 只在数据有 alpha mask 或 DTU inferred background mask 时存在；LLFF RGB 通常没有此项（`train.py:117-150`）。
- 未发现 normal loss、外部 depth supervision 或 Where/How much/When 式结构授权；普通 photometric L1+SSIM 之外的几何正则就是上述 binocular disparity/warp/smoothness（`train.py:130-156`）。

### 6.2 已解决与未解决问题

当前机制解决的是：对训练相机附近的横向虚拟视角施加图像重投影一致性，以 rendered depth 形成 disparity，并以图像边缘引导其平滑。这提供局部双目几何约束。

它没有解决：

- 哪些空间/像素区域的结构先验可靠（Where）；
- 观测证据应赋予多大约束权重（How much）；
- 除固定 iteration 阈值外，何时证据成熟（When）；
- 错误 depth 是否被 shifted supervision 自强化；
- Dropout/full-set 两种渲染状态间监督是否一致。

因此 I1 不与 baseline 重复：baseline 是全局、固定启用时刻、非证据授权的 warp constraint；I1 是对结构先验的证据门控。

## 7. Innovation 1 Migration Compatibility

### 7.1 最小插入方案（不实施）

推荐作为独立开关模块，保持三个核心接口：

```text
evidence = I1.observe(real_view, primary_render_pkg, history)
authorization = I1.authorize(evidence, iteration)
  ├─ where_mask
  ├─ strength_map / scalar
  └─ active_when
structure_loss = I1.loss(primary/shifted states, authorization)
total_loss += structure_loss
```

最小代码插入点：

1. **观测输入：** `train.py:101-113`，主 render 取出 image/depth/alpha/dropout mask 之后。这里可构造真实训练视角证据，不改变 renderer。
2. **授权与结构项：** `train.py:130-155`，现有 disparity/alpha loss 完成后、`total_loss` 汇总前。Where/How much/When 在此只决定新增项是否及如何加权。
3. **状态更新：** `train.py:160-183` 的 `torch.no_grad()` 区，在 optimizer step 前后选择一致时点更新移动统计；不能用当前 iteration 的未来梯度信息反向泄漏到同一 iteration 授权。
4. **CLI 开关：** `train.py:283-308` 附近新增显式可开/关参数；不要复制 `store_true, default=True` 反模式。
5. **checkpoint：** 若 I1 有历史状态，必须与 `gaussians.capture()/restore()` 同步保存，而非只写 cfg（`scene/gaussian_model.py:61-93`; `train.py:213-215`）。

### 7.2 梯度与时序安全条件

I1 可保持独立，前提是：授权证据默认 `detach`，仅结构 loss 对允许的 image/depth/Gaussian 路径反传；不改原 L1/SSIM、dropout mask、densification stats、opacity decay 与 optimizer 的相对顺序。若 I1 需要 shifted state，应明确选择“同一 Dropout mask”“完整集合”或“期望一致性”并做消融，不能沿用当前混合状态而不声明。

短程 3k/5k 中，原固定 20k gate 永不触发。I1 的 When 不能机械线性缩放为 2k/3.3k，也不能为了看见收益而改变“证据成熟/late-stage”的科学定义。应以可观测的结构稳定度、增长趋缓、跨视角 evidence coverage 等准则决定；首次有效性应使用同一 baseline checkpoint 的 paired resume，在 24GB 上继续到真正 late-stage。短程无收益只表示“尚未进入适用阶段”，不足以 NO-GO。

## 8. Innovation 2 Lifecycle Compatibility

### 8.1 当前缺失状态

`GaussianModel` 当前只有参数、max radii、gradient accumulator、denominator、optimizer 与空间尺度；capture 也只保存这些（`scene/gaussian_model.py:44-75`）。以下全部 **NOT FOUND**：Gaussian UID、birth iteration、birth type、source UID、target UID、generation、newborn observation state、death reason、I2 checkpoint state。

### 8.2 迁移风险与对齐建议（设计审计，不实施）

- **稳定 UID：** 使用单调 int64 UID 张量，与每个 Gaussian 第一维严格对齐；clone 分配新 UID、parent 保留；split children 分配新 UID 并记录 source parent，parent 记 `split_replaced`；prune 前先按 mask 记录 death。UID 不可用数组下标代替，因为 cat/prune 会重建 Parameter（`gaussian_model.py:273-351`）。
- **生命周期同步：** 所有 metadata 必须在 `cat_tensors_to_optimizer` / `_prune_optimizer` 的同一 keep/append 语义下同步；checkpoint capture/restore 和 PLY-only reload 的语义需单独定义。仅 PLY 无法恢复出生历史。
- **真实视角观测：** iteration 选到真实训练相机只是 opportunity 起点；对某 Gaussian，建议定义为“投影在 frustum 且在 dropout 前具备有效屏幕 footprint”，然后单独记录 kept、raster-visible、contributing。shifted synthetic camera 不计真实视角机会。
- **Dropout 校正：** 以 observed opportunity 和 keep probability 做条件化；最低 opportunity/kept/visible 阈值未满足则 `insufficient_evidence`，禁止作为 donor。
- **decay 去污染：** 不以 raw opacity retention 判死；可将 decay schedule 作为协变量/去趋势项，utility 以 contribution 与 error reduction 为主。
- **donor eligibility：** 已过 newborn grace period；真实视角有效样本量充分；多视角贡献持续低且置信上界低；不是刚 split-replaced；删除不会破坏已验证结构覆盖；opacity 仅辅助。
- **Verified Representation Deficit：** 候选量可包括跨真实训练视角持续高残差、低 alpha coverage、稳定的未解释边缘/纹理、跨 dropout render 的高方差且有多视角一致性；必须排除单视角噪声和纯原生 gradient growth demand。
- **K=R：** 在原 `densify_and_prune` 前形成 donor UID 集 R 与 receiver deficit proposals；先生成恰好 K=R 个受 deficit 定位的新 Gaussian，再原子化更新参数、optimizer state、metadata，或在内存峰值要求下先释放再添加但保留 rollback。baseline 与 ours 的 budget cap、峰值 N 和统计口径必须相同。

### 8.3 D0/D1 基础设施风险

最大风险不是增加字段本身，而是：split 删除 parent、clone 保留 parent；每次 densification 清空统计；Dropout 使 exposure 缺失非随机地影响梯度；opacity decay 污染标签；checkpoint 不含 RNG/生命周期；最终 PLY 不含 I2 状态。这些均要求先做 D0（只记录不干预）与 D1（离线标签审查），再进入 Release/Reallocate。

## 9. RTX 5060 8GB Development Strategy

### 9.1 显存来源

静态可见的 Gaussian 常驻 FP32 payload（SH degree 1）包括 xyz 3、DC 3、rest SH 9、scale 3、rotation 4、opacity 1，共 23 floats/G；加梯度、Adam 两个 moments，以及 max_radii/gradient_accum/denom。由源码可推导约 **380 bytes/G 的下界 payload**，但这不是总显存估计，因为不含 allocator、临时 cat/prune、索引副本、rasterizer buffers、autograd saved tensors 与图像级 buffers（`arguments/__init__.py:49-50`; `gaussian_model.py:124-163,311-351`）。

主要不可静态定量项：

- rasterizer tile/binning/image buffers（submodule 未初始化）；
- H×W 的 RGB/depth/alpha/screenspace tensors；
- Dropout 索引产生的 kept-set tensors；
- 主 render 计算图尚存时再建立 shifted render 计算图（`train.py:101-156`）；
- clone/split 时 old+extended tensors 和 optimizer moments 的瞬时共存；
- PDCNet+ 初始化本身的网络与 dense matching buffers。

运行期必须记录：每 iteration `allocated/reserved/max_allocated`、N Gaussians、是否 binocular active、densify 前后峰值、分辨率、rasterizer buffer 峰值，并分开测 triangulate 与 training。

### 9.2 分级本地方案

**G1：100–300 iteration minimum smoke。** 单场景、单进程；验证 import/API、forward/backward、loss finite、optimizer step、保存/载入最小产物和显存曲线。由于 densify_from=1000、binocular start=20000，此 smoke 不验证增长或双目。

**G2：约 1200–1500 densification smoke。** 触发首次 interval；记录 clone/split/prune 前后 N、参数 shape、Adam moments shape、UID 设计探针（未来 D0）。注意现代码在 1100 附近首次满足 `iteration>1000` 且 `%100==0`。若 8GB 不足，先降分辨率并明确标为 local debug protocol。

**G3：3k 优先、必要时 5k pilot。** 只看稳定性、增长曲线、早期画质和成本，不宣称 paper quality，也不判断 late-stage I1。

### 9.3 降本措施分类

**A. 不改变核心算法语义的工程优化：** 单场景串行；不运行八场景 ThreadPool；分开执行一次 PDC 初始化与训练；禁用非必要 render 中 CA 五次 dropout/点云深拷贝诊断；及时关闭其他 GPU 进程；记录峰值；使用一致的已生成初始化；失败即停止。严格说，换硬件/调度不改训练目标。

**B. 改变训练协议、仅允许本地调试：** iterations=100–300/1500/3k；resolution 从 2 调大以进一步下采样；将 shift_cam_start 临时按“功能触发测试”提前；实验性 Gaussian cap；必要时关闭 binocular 以定位显存。所有结果标 `LOCAL-DEBUG-NOT-COMPARABLE`。

**C. 正式实验不可直接采用的非公平配置：** 只给 ours 或只给 baseline 设 cap；不同分辨率/初始化/iteration；只对一方关闭双目或 dropout；因 8GB OOM 而选择性裁剪场景；拿 3k/5k 指标替代 30k paper protocol。

若采用 Gaussian cap，baseline 与 I1/I2 必须使用同一场景、初始化、最大 N、增长调用次数和峰值预算；I2 的 K=R 应在 cap 内做资源守恒，报告 quality-vs-budget 曲线，而不是只限制某一方。

## 10. AutoDL 24GB Formal Experiment Strategy

AutoDL 承担：隔离冻结环境、PDCNet+ 初始化、无泄漏 protocol baseline、完整 30k、20k 后 binocular/late-stage 行为、paired checkpoint、八场景正式比较、峰值 N/VRAM/time 记录。

推荐顺序：

1. 在一个场景验证 rasterizer 的四输出 API、CuPy/PDCNet、simple-knn、数据 manifest。
2. 固定并导出每场景 train/test 文件名和初始化输入清单；确认 test 不进入 PDC matching/point sampling。
3. 只跑一个 baseline 30k，检查 released result 的数量级、N/VRAM 增长和 20k 后 binocular。
4. baseline 通过后跑八场景；不要先移植 I1/I2 掩盖宿主问题。
5. 从同一 pre-I1 checkpoint 分叉 baseline/ours，保持 RNG policy、budget、views、resolution、初始化一致；I1 first migration 只增加授权结构项。

**UNKNOWN / prerequisites：** 24GB 是否足够取决于场景分辨率、N 峰值和 rasterizer；目标 submodule API；PDC checkpoint hash；数据版本；CUDA/driver compatibility。均需 A1/A2/A3 证据，不应预设通过。

## 11. Risk Register

| 风险 | 状态 | 严重度 | 证据/说明 |
|---|---|---:|---|
| argparse `store_true, default=True` 不能常规关闭 opacity decay/binocular | CONFIRMED | High | `train.py:298-304`; `arguments/__init__.py:30-38` |
| `densify_until_iter` 动态覆盖为总 iterations | CONFIRMED | High | `train.py:178-185` |
| opacity reset 实际 pass | CONFIRMED | Medium | `train.py:201-206` |
| `size_threshold` 重复赋 None | CONFIRMED | Medium | `train.py:194-199` |
| 脚本 p=0.3、paper 注释 p=0.2 | CONFIRMED | High | `script/run_llff.py:13-16` |
| paper 8×、released script resolution=2 | CONFIRMED | Critical protocol | `README.md:109-111`; `script/run_llff.py:11-13` |
| Dropout mask 与默认 renderer 输出索引错位 | NOT FOUND（默认路径） | — | `train.py:187-192` 正确 remap |
| Python SH/covariance 模式 Dropout shape/None 风险 | POTENTIAL | High | `gaussian_renderer/__init__.py:68-105` |
| shifted camera 使用不同 Dropout 状态 | CONFIRMED | High | `train.py:101-143` |
| `os.chdir()` 与多线程竞态 | CONFIRMED | Critical | `script/run_llff.py:41-43,78-109,162-164` |
| chdir 后 data/log 相对路径错误 | CONFIRMED | Critical | `script/run_llff.py:23-43` |
| Linux env-prefix 命令在 Windows 失败 | CONFIRMED | Critical local | `script/run_llff.py:37-67` |
| 含空格路径未引用 | CONFIRMED | High local | 同上；当前 cwd 含空格 |
| train/render/metrics 路径不一致 | POTENTIAL | Medium | 主输出基本一致；metrics 参数名依赖 argparse abbreviation（`metrics.py:139`; runner `:64-65`） |
| `os.system()` 返回码忽略 | CONFIRMED | Critical | `script/run_llff.py:42,52,61,67` |
| metrics 裸 except 将失败变为成功退出 | CONFIRMED | High | `metrics.py:126-131` |
| LLFF preprocessing 使用测试图像 | CONFIRMED | Critical scientific | `triangulate.py:104-118`; `dataset_readers.py:171-175` |
| triangulate 与 train 相机排序准则不同 | CONFIRMED | High | `triangulate.py:66`; `dataset_readers.py:151` |
| 30k 持续增长导致 8GB OOM | POTENTIAL / NEEDS VERIFICATION | High | growth 至总 iter、无 cap；需测峰值 |
| checkpoint 恢复后 iteration/LR schedule 不一致 | NOT FOUND（默认 30k） | — | 恢复 first_iter，LR按 iteration 更新（`train.py:42-44,64-66,84`; `gaussian_model.py:169-175`） |
| checkpoint 不保存 RNG/I1/I2 生命周期状态 | CONFIRMED（对未来迁移） | High | `gaussian_model.py:61-93` |
| train-time validation 与 final render opacity scaling 不一致 | CONFIRMED | High | `train.py:255`; `render.py:108`; renderer `:106-109` |
| rasterizer 四返回值与冻结 submodule API 匹配 | NEEDS VERIFICATION | Critical build | submodule 空；caller `gaussian_renderer/__init__.py:117-126` |
| PDCNet checkpoint 缺失 | CONFIRMED local | Blocking A2 | `pre_trained_models` 目录不存在 |
| requirements 缺 CuPy | CONFIRMED | High | PDCNet import chain 与 requirements 对比 |
| opacity `.data` 改值但保留 Adam momentum | CONFIRMED | Medium/scientific | `gaussian_model.py:307-309` |
| summary 对失败场景仍计入分母 | CONFIRMED | Medium | `script/run_llff.py:128-152` |
| render CA 统计空 common mask 产生 NaN | POTENTIAL | Low/diagnostic | `render.py:139-175` |

## 12. GO / NO-GO Decision

### 五维评价

1. **绝对 baseline 强度：** README 报告强（LLFF 22.12/0.777/0.154），但 paper/released protocol 不一致且公开初始化存在泄漏。评分：有潜力，尚未可信复现。
2. **I1 补充空间：** 明确存在。现有 binocular 只有固定时点、非证据门控的 warp/smoothness；没有 Where/How much/When 授权。
3. **I2 补充空间：** 明确存在。没有 UID、出生/死亡、真实曝光、效用与 deficit；原生 growth 仅按 screen-space gradient 和 size 分类。
4. **创新边界：** Dropout/opacity noise 解决 co-adaptation；binocular 解决局部虚拟视角一致性；I1 解决结构先验授权，I2 解决观测证据驱动的预算闭环。边界可清晰陈述，但必须避免把原生 densification gradient 改名为 deficit。
5. **工程/硬件成本：** 高。依赖栈不闭合、runner 不可靠、rasterizer 未取回、8GB 风险高；24GB 隔离验证是必要条件。

### 最终判定：CONDITIONAL GO

值得继续作为第一优先宿主，但条件是：

- 先建立无泄漏且 train/init view 对齐的 frozen manifest；
- 在隔离环境验证 PDCNet、CuPy、simple-knn、四输出 rasterizer；
- AutoDL 单场景 baseline smoke/pilot/30k 依次过 Gate；
- 明确选择 paper 8×/p=0.2 或 released 2×/p=0.3，不能混称“官方协议”；最好两者都记录，并以论文主表协议为正式目标；
- baseline 通过后才移植 I1；I2 先 D0/D1 观测，不直接执行 reallocation。

当前 `fsgs` 环境：**不能声明安全复用；不建议原地安装/编译。**  
本地 8GB：最多承担静态审计、CPU/manifest 检查、单场景 100–300 smoke，以及显存允许时 1200–1500 growth smoke/3k debug。  
下一阶段最小验证：A1 隔离环境中的 import/API/ABI 探针 + A2 单场景无泄漏初始化 manifest；不是直接训练 30k。

## 13. Next Action Plan

本报告只定义 Gate，不执行。

### A0 — 静态审计（本报告）

**PASS with blockers recorded.** 输出依赖、协议、Dropout、growth、I1/I2、硬件与风险证据。Gate：报告接受；不把公开 runner 当可靠入口。

### A1 — 环境可行性

在独立环境验证：Python/Torch/CUDA/driver；`simple_knn._C`；冻结 rasterizer commit 与四输出 API；CuPy local correlation；PDC checkpoint load；包实际来源无 shadow。Gate：全部版本/hash/API 记录且 import smoke 通过。

### A2 — 官方初始化

准备单个 LLFF 场景；导出按 image_name 排序的 N、test IDs、train candidate IDs、3 train IDs、PDC 输入 IDs；断言交集为空；固定 PDC checkpoint hash和生成 PLY hash。Gate：无测试泄漏且训练/初始化 ID 完全一致。

### A3 — Baseline Smoke

100–300 iter 单场景，检查 finite loss、forward/backward、optimizer、显存；再以 1200–1500 检查 clone/split/prune/moments。Gate：无 NaN/Inf、shape 同步、显存曲线可解释、失败正确退出。

### A4 — 短程 Pilot

3k，必要时 5k；记录 N、VRAM、time、train/test render 配置和 early metrics。随后在 AutoDL 做一个 30k baseline 验证 20k 后 binocular 与最终 opacity scaling。Gate：协议冻结，单场景结果/资源可信，再扩八场景。

### A5 — I1 最小迁移

只增加独立开关的 observe → authorize(Where/How much/When) → structure loss；paired checkpoint 分叉；不改 Dropout、growth、decay 时序。Gate：开关关闭时数值/轨迹与 frozen baseline 等价；开启时证据覆盖、授权区域、梯度与成本均可审计。

---

## 审计边界声明

本次只执行了文件读取、文本搜索和只读 Git 查询；没有初始化 submodule、联网、下载权重、安装依赖、编译 CUDA、导入项目模块、运行 Python/pytest、执行训练/渲染/metrics、切换分支、修改现有源代码或 Git 配置。唯一创建的文件是本 Markdown 报告。
