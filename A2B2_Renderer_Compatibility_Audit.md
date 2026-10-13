# A2-B2 Renderer Compatibility Audit

## 1. 结论

**A2-B2 结论：API/KERNEL PASS；FSGS 环境可继续复用（GO）；真实训练 NOT TESTED。**

当前 Binocular3DGS `render()` 已能向现有 FSGS confidence rasterizer 提供必填的 `confidence`。三条规定路径均通过真实项目 renderer 的极小 CUDA forward/backward：

1. `train=True, dropout_factor=0.0`：PASS
2. `train=True, dropout_factor=0.2`：PASS
3. `train=False, dropout_factor=0.2`：PASS

本阶段没有安装、卸载或编译任何包/扩展，没有修改 fsgs 环境，没有运行 Horns、PDC、训练、渲染评估或 I1/I2，也没有 commit/push。

## 2. 环境与执行边界

| 项目 | 实际值 |
|---|---|
| 仓库 | `D:\3DGS\Binocular3DGS + Dropout` |
| Git HEAD | `01677f59e9a23abcffc95a097ca59b3964fa03fd` |
| Python | `D:\Users\huashuo\anaconda3\envs\fsgs\python.exe` |
| PyTorch | `2.8.0+cu129` |
| CUDA runtime | `12.9` |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| Compute Capability | 12.0 (`sm_120`) |
| 合成场景 | 12 Gaussians，32×32，float32，CUDA:0 |

CUDA 命令使用上述 fsgs Python 的绝对路径。测试只分配极小张量；未读取真实 Horns 图像或已有模型。

## 3. 修改内容

### 3.1 修改及新增文件

- 修改：`gaussian_renderer/__init__.py`
- 新增：`output/a2b2_renderer_compat/probe_project_renderer.py`
- 新增日志：`output/a2b2_renderer_compat/renderer_integration.json`
- 新增日志：`output/a2b2_renderer_compat/cpu_regression.txt`
- 新增本报告：`A2B2_Renderer_Compatibility_Audit.md`

项目算法源码中只修改了 `gaussian_renderer/__init__.py`。其相对 HEAD 的统计为 `14 insertions, 5 deletions`；其中重复的 `pc.get_opacity` 读取被合并为一次，Dropout 抽样仍是原来的一次 `torch.rand` 和原阈值。

### 3.2 核心 diff

```diff
@@
+    opacity = pc.get_opacity
@@
-        dropout_mask = torch.rand(pc.get_opacity.shape[0], device=pc.get_opacity.device).cuda()
+        dropout_mask = torch.rand(opacity.shape[0], device=opacity.device).cuda()
         dropout_mask = dropout_mask < (1 - dropout_factor)
@@
-        dropout_mask = torch.rand(pc.get_opacity.shape[0], device=pc.get_opacity.device).cuda()
+        dropout_mask = torch.rand(opacity.shape[0], device=opacity.device).cuda()
         dropout_mask = dropout_mask < 1.1
+
+    # Unit confidence preserves the existing gradients and must match the exact
+    # Gaussian subset passed to the rasterizer after training-time dropout.
+    if dropout_factor > 0.0 and train:
+        raster_confidence = torch.ones_like(opacity[dropout_mask], requires_grad=False)
+    else:
+        raster_confidence = torch.ones_like(opacity, requires_grad=False)
@@
-        debug=pipe.debug
+        debug=pipe.debug,
+        confidence=raster_confidence
@@
-    opacity = pc.get_opacity
@@
-        opacity *= 1 - dropout_factor
+        opacity = opacity * (1 - dropout_factor)
```

最后一项保持原有推理期 opacity 缩放公式和系数完全不变，只将原地乘法改成非原地乘法。规定的 `train=False` backward 首次运行时，原写法会修改 sigmoid 输出的 autograd version，报 `one of the variables needed for gradient computation has been modified by an inplace operation`；该等价改写消除了此冲突。没有更改 opacity scaling 的数值语义。

## 4. Confidence 形状、设备与语义

### 4.1 数量规则

| 路径 | 送入 rasterizer 的 Gaussian | confidence |
|---|---|---|
| `train=True, dropout_factor>0` | `tensor[dropout_mask]` | `ones_like(opacity[dropout_mask])` |
| `train=True, dropout_factor=0` | 完整集合 | `ones_like(opacity)` |
| `train=False`（任意 dropout_factor） | 完整集合 | `ones_like(opacity)` |

因此 confidence 形状为 `[实际 rasterized Gaussian 数量, 1]`，dtype/device 由真实 opacity 直接继承。本次均为 `torch.float32`、`cuda:0`。它显式 `requires_grad=False`，且测试确认无 `grad_fn`，不引入可学习参数或新置信度机制。

### 4.2 全 1 的中性检查

A2-B1 已确认该 confidence fork 不把 confidence 传入 C++ forward，而是在 Python backward 中对每个 Gaussian 的梯度乘以 confidence。由此：

- forward 渲染方程不受 confidence 影响；
- backward 的乘数全为 1，在已测参数和调用路径中保持原梯度权重；
- 三条路径的 loss 与全部被测梯度均有限，保留 Gaussian 均获得非零梯度。

这里的“中性”限于已安装扩展及本次调用路径；未与另一套无 confidence 扩展做逐 bit 梯度对照，也未覆盖长程训练中的数值积累。

## 5. 项目 Renderer CUDA 集成测试

测试不是绕过项目代码直接调用 `GaussianRasterizer`。脚本导入并调用真实 `gaussian_renderer.render()`；包装器只捕获 settings 中的 confidence 和 rasterizer 实际输入数量，随后调用真实已安装 CUDA rasterizer 的 `super().forward()` 和 backward kernel。

执行命令：

```powershell
& 'D:\Users\huashuo\anaconda3\envs\fsgs\python.exe' -B `
  'D:\3DGS\Binocular3DGS + Dropout\output\a2b2_renderer_compat\probe_project_renderer.py' `
  'D:\3DGS\Binocular3DGS + Dropout\output\a2b2_renderer_compat\renderer_integration.json'
```

使用 12 个可微 Gaussian、32×32 camera stub 和 L1 photometric loss。Gaussian stub 的属性形状及 opacity sigmoid 路径与真实 renderer 接口保持一致。

### 5.1 结果总表

| 状态 | 输入/保留/排除 | confidence | image / radii / depth / alpha | loss | 峰值 allocated | 峰值 reserved | 结论 |
|---|---:|---|---|---:|---:|---:|---|
| train, dropout 0 | 12/12/0 | `[12,1]` | `[3,32,32]` / `[12]` / `[1,32,32]` / `[1,32,32]` | 0.2430893183 | 112,128 B | 2,097,152 B | PASS |
| train, dropout 0.2 | 12/11/1 | `[11,1]` | `[3,32,32]` / `[11]` / `[1,32,32]` / `[1,32,32]` | 0.2426423728 | 114,176 B | 2,097,152 B | PASS |
| eval, dropout 0.2 | 12/12/0 | `[12,1]` | `[3,32,32]` / `[12]` / `[1,32,32]` / `[1,32,32]` | 0.2417391986 | 112,640 B | 2,097,152 B | PASS |

三种状态均确认：

- image、radii、depth、alpha 输出数量和 shape 正确且数值有限；
- 完成真实 CUDA forward、L1 loss backward 和 `torch.cuda.synchronize()`；
- xyz、opacity、features、scaling、rotation、viewspace points 的梯度均存在且有限；
- 未出现 `invalid device function`、`no kernel image`、device-side assert 或非法内存访问。

最大单 case 峰值 allocated 为 **114,176 B（约 111.5 KiB）**，峰值 reserved 为 **2 MiB**。这些只代表 12 点合成 smoke，不可外推真实训练显存。

## 6. Dropout 索引映射验证

训练 dropout case 使用固定 seed 1000，并在调用前预览、随后重置 RNG，以验证 renderer 没有改变原随机抽样过程：

- 原始索引：`[0..11]`
- 保留索引：`[0,1,2,3,4,5,6,7,8,9,11]`
- 排除索引：`[10]`
- rasterized/visible 原始索引：`[0,1,2,3,4,5,6,7,8,9,11]`
- 排除与 visible 交集：空
- `radii`、`visibility_filter` 长度：11，与保留集合一致
- `dropout_mask` 长度：12，保持原始集合索引域
- `kept_indices[visibility_filter]` 能正确还原原始可见索引
- 排除点在 xyz、opacity、features、scaling、rotation、viewspace points 上的梯度全部严格为零
- 11 个保留点在上述每类参数上均获得非零梯度

因此原有 `dropout_mask`、subset `radii/visibility_filter` 与训练端原始索引映射保持一致；confidence 只为实际保留的 11 点构造。

## 7. CPU 回归与源码状态

```text
python -B -m unittest discover -s tests -v
Ran 17 tests in 1.297s
OK
```

- CPU tests：17/17 PASS
- `git diff --check`：PASS（无输出）
- 没有 commit/push
- 工作树中 A1/A1-Fix1/A2 阶段的既有修改与未跟踪文件均被保留；本任务未清理、覆盖或回退它们

### 7.1 SHA-256 冻结记录

| 文件 | SHA-256 |
|---|---|
| `gaussian_renderer/__init__.py`（A2-B2 后） | `619158AE1BAF52A0D71B03477E45FC5598827F2B58EDD64B5846DD8BD17666F6` |
| `probe_project_renderer.py` | `1E9BC8899AABB11750725CE34AC82809D6B4B6359183EC1F27F90669ED6276F0` |
| `renderer_integration.json` | `1EC03EB6AD53FE46584014D3372DD6C1DB070F4EC917777798CB26D662D700DA` |

A2-B1 记录的修改前 renderer SHA-256 为 `112B7E48C2AB1B2918C7050475F8D1B7AAD9F0DC5D95AB88F2CCFE1523C7B063`。HEAD SHA、当前文件 SHA-256 与正文 diff 共同冻结本阶段实际代码状态，而非仅记录 upstream HEAD。

## 8. 风险与未验证项

以下不能由本次结果推导为 PASS：

- 真实 Horns 的训练、验证渲染和 3k/5k 稳定性；
- 真实点数下的 8 GB 显存峰值与 OOM 边界；
- densification/pruning、optimizer step、checkpoint/save/resume；
- `sigma_noise>0`、`render_pts=True`、Python covariance/SH 路径和其他未要求分支；
- 与无 confidence rasterizer 的逐 bit 等价性；
- PDC 初始化以及 I1/I2。

尤其是本次显存值仅验证 kernel 能在 sm_120 上执行，不代表真实训练可容纳于 8 GB。

## 9. 最终判定

- **接口兼容性：PASS**
- **confidence 数量/dtype/device/非学习属性：PASS**
- **三种项目 renderer CUDA forward/backward：PASS**
- **Dropout RNG、mask 与索引映射：PASS**
- **FSGS 环境复用：GO（无需重装或重编译）**
- **真实训练：NOT TESTED，不得表述为训练 PASS**

下一阶段若获单独授权，最小建议是先做受控的短程真实场景 smoke，并单独记录峰值显存、densification 前后索引行为和失败边界；本阶段到此停止。
