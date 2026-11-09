# Rosetta + 第三目标池（rosetta3）

在官方 Rosetta（`dominik-schnaus/unpaired-rosetta@fdfad84`）上加一个可选的第三目标池 Z。官方源码不改，只做子类。

## 方法

| 方法 | 第三目标池 Z | 作用 |
|---|---|---|
| rosetta | 无（λ=0） | 官方两空间 Rosetta，最强纯语言基线 |
| visual | bank 图像 × V_IE | 主方法 |
| bank_text | 同一批 bank 图像的英文描述 | 内容相同、模态不同；**H1 的关键对照** |
| extra_text | 另外 500 张图的英文描述 | 衡量“多一批新文本”本身的作用 |
| gaussian | 随机高斯点 | 只有几何、没有语义 |
| english_pool | 复用英文 Y 池作为第三目标 | 不增加新内容，只增加一次独立抽样的匹配与权重 |
| anchor_downweight | 无第三目标，仅降低平行句项 | 验证收益是否仅来自降低真实监督的相对权重 |

每步 refinement：W = polar(X_mᵀY_m + λ·(k/k_z)·X_mzᵀZ_mz + (k/p)·锚点项)。第一项和最后一项与官方完全相同。
v1 只在 refinement 加 Z；几何初始化与 readout 仍只用中文与英文。

## 关键设计

- λ=0 或不给 Z 时，与官方结果**逐位一致**（测试覆盖有/无平行句两种情况）。
- Z 用自己的均值中心化；Z 的 batch 来自独立随机流，不扰动官方的随机序列。
- 同一 seed 与平行句下，几何初始化对所有方法相同，`init_cache` 复用它并恢复随机流状态，结果仍逐位一致。
- L→I 评测对所有方法使用同一个图像均值，避免评测预处理造成差异。
- 平行句下标与 `run.py` 相同，可与 NumPy 原型逐行比较。
- 检测到 bank 描述也在 `e_pool` 中时直接报错（`--allow-overlap` 可强行复现旧设定）。

## 运行

使用已经修正的 `real_work/features.npz`；如果只有旧的重叠数据，先运行 `stage1_repool.py`，并把生成的 `features_stage1.npz` 路径传给运行脚本。

```bash
python -m pip install -r requirements_rosetta.txt
python -m pip install 'unpaired-rosetta @ git+https://github.com/dominik-schnaus/unpaired-rosetta@fdfad84ea488f29702a61a455be71622c4ff5d49'
python -m unittest -v test_core test_rosetta3
python run_rosetta3.py --data real_work/features.npz --budget 50 --seeds 0 --clusters 8 --restarts 1 --iterations 2 --out real_work/rosetta3_smoke
```

默认运行 **50、200、1000 对监督平行句**，不再自动运行 0 对（零配对需显式指定 `BUDGETS=0`）。
第三目标系数默认 **`lambda_z=0.25`**；官方 `rosetta` 仍始终使用 0。
为了与之前的官方 Rosetta 试验保持参数一致，默认采用 20 clusters、5 restarts、batch 256、20 iterations。

先快速试跑 50 对：

```bash
BUDGETS=50 SEEDS="0 1 2" LAMBDA_Z=0.25 bash run_rosetta3.sh
```

运行默认全部预算的脚本：

```bash
bash run_rosetta3.sh
```

## 实验局限

- 中文池约 1000 条，几何初始化零配对可能失败。
- 不改动原版 Rosetta 的初始化与 readout，只在 refinement 里加第三目标。
- `lambda_z` 默认从 1.0 降至 **0.25**（Visual、Bank Text、Extra Text、Gaussian 使用同一系数，以确保公平）；`rosetta` 仍为 0。它仍会改变伪配对与监督锚点的相对权重，因此仍需 0.1/0.25/0.5/1.0 消融。
- 本实验图像使用 bank 图像均值中心化，不可直接与原先使用 Rosetta 英文均值进行图像中心化的 baseline 指标作严格比较。
- 官方 Rosetta 依赖 `pylibmgm`；默认 `run_rosetta3.sh` 训练成本比小规模 smoke 测试高。

## 权重与重复英文池控制（独立实验分支）

新增两组方法用于检验已观察到的约 1.3 pp 文本收益是否来源于语义内容：

- `english_pool`：Z 直接设置为相同英文训练池 `e_pool`，Y 与 Z 内容完全相同但每个 refinement step **独立随机抽样并独立 Hungarian Matching**。因此同时改变总伪配对权重和匹配次数；并非只改变权重的完美控制。
- `anchor_downweight`：不添加 Z，仍仅执行一轮 Y 匹配；只在 **refinement 阶段**将真实平行句的协方差项乘以 `anchor_scale = 1/(1+lambda_z)`。初始 geometric initialization/readout 仍为原版。对于默认 `lambda_z=0.25`，对应 `anchor_scale=0.8`（不是减半）。
- `rosetta`：保持原始路径（`anchor_scale=1`, `lambda_z=0`）。

相同 budget、seed、参数，快速比较两个控制和视觉/英文文本目标：

```bash
python -m unittest -v test_rosetta3
python run_rosetta3.py --data real_work/features.npz --budget 50 \\
  --seeds 0 1 2 --dim 32 --clusters 20 --restarts 5 --batch 256 \\
  --iterations 20 --lambda-z 0.25 \\
  --methods rosetta visual bank_text extra_text gaussian english_pool anchor_downweight \\
  --out real_work/rosetta3_weight_controls_50
```

其他预算可改为 `--budget 200`、`--budget 1000`，并使用不同输出文件夹。所有实验使用相同 50/200/1000 监督锚点的选择方式。

当前测试集中逐样本命中结果的 paired bootstrap 比单纯把 3 个 seeds 平均更有解释力；`report.json` 中的 `paired_bootstrap` 包含每个 seed 的 R@1 对照及置信区间。注意不同 seed 共享评测问题，不能把三个 seed 当作三批完全独立测试集。
