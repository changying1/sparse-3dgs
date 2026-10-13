# A2-B7 — Gaussian 增密与剪枝事件定量诊断审计

日期：2026-10-11  
仓库：`D:\3DGS\Binocular3DGS + Dropout`  
Python：`D:\Users\huashuo\anaconda3\envs\fsgs\python.exe`

## 1. 结论

已实现默认关闭、按事件记录的 Gaussian densification 诊断。启用
`--densification_diagnostics` 后，每次 `densify_and_prune` 会在对应实验目录写入
`densification_events.csv`，记录 Clone、Split、Prune 的数量分解、事件前后总数、守恒检查，以及全部 optimizer parameter group 的 parameter、`exp_avg`、`exp_avg_sq` 首维检查。

默认关闭时：

- 不创建 CSV logger，不读写诊断文件；
- 不调用 `torch.count_nonzero(...).item()`，因此没有诊断引入的 GPU→CPU 标量同步；
- 不增加或改变任何随机数调用；
- Clone、Split 和 Prune 的选择表达式、阈值、调用顺序均保持不变；
- 训练参数原默认值保持不变，新开关默认值为 `False`。

本阶段只运行了 CPU 单元/回归测试和语法检查；没有运行 Horns、CUDA 合成训练、真实 GPU 训练或 5k，也没有安装依赖、修改 `fsgs` 环境、commit 或 push。

## 2. 代码修改

### `utils/densification_diagnostics.py`

新增与 CUDA 无关的诊断基础设施：

- `add_densification_diagnostics_argument`：注册默认 `False` 的 CLI 开关；
- `build_densification_event`：计算事件守恒期望值与差值；
- `optimizer_dimension_errors`：遍历每个 parameter group 及其所有 parameter，核对 parameter、`exp_avg`、`exp_avg_sq` 的首维；
- `DensificationEventLogger`：将事件追加到实验目录 CSV。若守恒或 optimizer 维度不成立，先写入失败记录，再抛出 `RuntimeError`，避免静默继续训练。

### `scene/gaussian_model.py`

给 `densify_and_clone`、`densify_and_split`、`densify_and_prune` 增加默认关闭的 `collect_diagnostics=False` 参数。

只有启用时才采集：

- Clone 新增张量的首维；
- Split 生成子点首维及默认 `N=2` 对应的父点选择数；
- Split 父点 prune 前后的实际首维差；
- opacity prune mask 的 `count_nonzero().item()`；
- 最终 prune 前后的实际首维差。

`size_prune_deleted` 是总 prune 删除数减去 opacity mask 已覆盖的删除数，因而表示 size 条件独有的额外删除数；当前训练代码仍强制 `size_threshold=None`，该列应为 0。该扩展字段保证即使以后调用端启用现有 size prune，守恒公式仍然完整。

没有修改以下实际选择条件：

```text
Clone: norm(grads) >= threshold AND max(scale) <= percent_dense * extent
Split: padded_grad >= threshold AND max(scale) > percent_dense * extent
Opacity prune: opacity < min_opacity
Size prune（若调用端启用）: max_radii2D > threshold OR max(scale) > 0.1 * extent
```

调用顺序仍为 Clone → Split（生成 `N` 个子点后删除父点）→ final prune → `torch.cuda.empty_cache()`。

### `train.py`

- 只在开关开启时构造 `DensificationEventLogger(scene.model_path)`；
- 关闭时保留原四参数 `densify_and_prune(...)` 调用；
- 开启时请求计数并在同一 densification event 后写 CSV、执行 optimizer 维度验证；
- CSV 路径为 `<model_path>\densification_events.csv`，即 `output/<实验名>/densification_events.csv`。

### `tests/test_a2b7_densification_diagnostics.py`

新增纯 CPU、无 Horns/renderer/CUDA 依赖测试，覆盖：

1. 开关默认关闭且显式参数可开启；
2. 19,397 → 18,812 的事件计数守恒示例；
3. 非守恒事件能被识别；
4. 6 个 optimizer group 的 parameter 与两个 Adam moment 首维通过检查；
5. parameter 或 `exp_avg` 维度错误能被定位到具体 group；
6. CSV 确实写入指定实验目录且带验证结果。

## 3. CSV 字段

| 字段 | 含义 |
|---|---|
| `iteration` | densification 发生的训练轮次 |
| `gaussians_before` | event 前 Gaussian 总数 |
| `clone_added` | Clone 新增数，Clone 父点保留 |
| `split_parents_selected` | Split 选中的父点数 |
| `split_children_created` | Split 产生的子点数；当前默认等于 `2 × split_parents_selected` |
| `split_parents_deleted` | Split 替换阶段实际删除的父点数 |
| `opacity_prune_deleted` | 满足 opacity prune mask 的删除数 |
| `size_prune_deleted` | size 条件在 opacity mask 之外额外删除的数量；当前训练应为 0 |
| `prune_deleted_total` | final prune 的实际总删除数 |
| `gaussians_after` | event 后 Gaussian 总数 |
| `expected_gaussians_after` | 按守恒公式计算的期望总数 |
| `conservation_delta` | `after - expected`，应为 0 |
| `conservation_ok` | 数量守恒布尔结果 |
| `optimizer_dimensions_ok` | 所有 parameter/Adam moment 首维检查结果 |
| `optimizer_dimension_errors` | 失败时的 group、state 与实际/期望维度 |

守恒公式：

```text
gaussians_after
= gaussians_before
 + clone_added
 + split_children_created
 - split_parents_deleted
 - prune_deleted_total
```

其中 `prune_deleted_total = opacity_prune_deleted + size_prune_deleted`；size mask 与 opacity mask 重叠的点只计入 opacity，避免重复扣减。

## 4. 测试结果

本任务测试：

```powershell
& 'D:\Users\huashuo\anaconda3\envs\fsgs\python.exe' `
  -m unittest discover -s tests -p 'test_a2b7*.py' -v
```

结果：`Ran 5 tests ... OK`。

完整 CPU 回归：

```powershell
& 'D:\Users\huashuo\anaconda3\envs\fsgs\python.exe' `
  -m unittest discover -s tests -p 'test_*.py' -v
```

结果：`Ran 31 tests ... OK`，包含既有 A1、A2-B3 与新增 A2-B7 测试。

语法与 diff whitespace 检查：

```text
py_compile: PASS
git diff --check: PASS
```

沙箱内首次调用指定 Python 时 Windows 以 `0xC0000022` 在解释器启动前退出；经允许在沙箱外调用同一个、未修改的 `fsgs` 解释器后，上述测试全部通过。这不是测试断言失败，也未触发依赖安装或环境修改。

## 5. 一次受控训练验证第 1100 轮

本阶段没有执行以下命令。验收时可沿用已通过的 1200-iteration Horns 协议，仅新增诊断开关，并使用新的实验目录，避免覆盖 A2-B6 结果：

```powershell
& 'D:\Users\huashuo\anaconda3\envs\fsgs\python.exe' `
  'D:\3DGS\Binocular3DGS + Dropout\train.py' `
  -s 'D:\dataset\nerf_llff_data_colmap\horns' `
  -m 'D:\3DGS\Binocular3DGS + Dropout\output\a2b7_horns_densification_diag_1200' `
  --iterations 1200 `
  --dropout_factor 0.2 `
  --sigma_noise 0.0 `
  --suffix a2b7_horns_densification_diag_1200 `
  --n_views 3 `
  --resolution 8 `
  --dataset_name LLFF `
  --view_manifest 'D:\3DGS\Binocular3DGS + Dropout\output\a2b6_horns_legacy_ply_smoke_1200\view_manifest.json' `
  --eval `
  --densification_diagnostics
```

该配置下 `densify_from_iter=1000`、`densification_interval=100`，而 densification 外层要求 `iteration < iterations`，因此 1200 轮运行应只记录 iteration 1100 的一条事件（1000 不满足严格大于，1200 不满足严格小于）。验证步骤：

1. 打开 `output\a2b7_horns_densification_diag_1200\densification_events.csv`；
2. 确认恰有一行且 `iteration=1100`；
3. 对照已知观测，检查 `gaussians_before=19397`、`gaussians_after=18812`；
4. 检查 `clone_added + split_children_created - split_parents_deleted - prune_deleted_total = -585`；
5. 检查 `split_children_created = 2 × split_parents_selected`、`split_parents_deleted = split_parents_selected`；
6. 检查 `conservation_delta=0`、`conservation_ok=True`、`optimizer_dimensions_ok=True` 且错误列为空；
7. 当前 `size_threshold=None`，检查 `size_prune_deleted=0`，因此 `prune_deleted_total=opacity_prune_deleted`。

若前后总数没有复现 19,397/18,812，应先核对 frozen initialization PLY、view manifest、分辨率、参数和随机种子协议；不要用修改阈值或重复运行来“校准”诊断结果。诊断只观测既有事件，不参与选择或优化。

## 6. 范围与停止点

交付文件为本报告、诊断模块、GaussianModel/训练入口的最小接线和 CPU 测试。未产生任何真实训练新结论；第 1100 轮的 Clone/Split/Prune 实数需由上述一次受控训练生成的 CSV 验收。到此停止，等待验收。
