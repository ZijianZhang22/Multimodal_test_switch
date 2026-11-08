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

运行默认全部预算的脚本：

```bash
bash run_rosetta3.sh
```

## 实验局限

- 中文池约 1000 条，几何初始化零配对可能失败。
- 不改动原版 Rosetta 的初始化与 readout，只在 refinement 里加第三目标。
- `lambda_z` 会改变伪配对与监督锚点的相对权重；需要额外的权重消融。
- 本实验图像使用 bank 图像均值中心化，不可直接与原先使用 Rosetta 英文均值进行图像中心化的 baseline 指标作严格比较。
- 官方 Rosetta 依赖 `pylibmgm`；默认 `run_rosetta3.sh` 训练成本比小规模 smoke 测试高。
