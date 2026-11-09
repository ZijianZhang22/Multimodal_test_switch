# CLIP 设置：把一种新语言接入现成的图文空间

英文（E）和图像（I）改用**同一个 CLIP 模型**的文本编码器和图像编码器。两者本来就对齐，CLIP 因此取代了原先用 1500 对图文自学的弱 bridge（在 1000 个候选中 R@1 约 11%），去掉了“bridge 质量”这个干扰因素。
中文（L）**仍然是单语编码器** text2vec-base-chinese。不要换成多语言 CLIP：它已经用大量平行句学过中文到 CLIP 的对齐，会让少配对实验失去意义。

研究问题：只用几十对平行句加无配对数据（中文文本、英文文本、未标注图像），能否让中文获得 CLIP 的图文能力？加入图像能带来多少提升？

## 运行

```bash
# 第一步：只提特征并看 bridge 诊断（建议先做）
BRIDGE_ONLY=1 bash run_clip.sh
# 第二步：低预算全流程（默认预算 10/25/50/100/200，5 个种子，8 种方法）
bash run_clip.sh
```

常用环境变量：`CLIP_MODEL`（默认 `openai/clip-vit-base-patch32`，可换 `openai/clip-vit-large-patch14`）、`WORK_DIR`（默认 `real_work_clip`）、`BUDGETS`、`SEEDS`、`DEVICE`。

特征写在**单独的工作目录**。`model_revisions.json` 现在记录模型名，若同一目录曾用其他编码器，脚本会报错而不是悄悄混用。默认设置的旧目录和旧缓存不受影响。

## bridge 诊断（`bridge_test.json` 的 `variants`）

| 键 | 含义 |
|---|---|
| `clip_native` | CLIP 原始空间直接做英文→图像检索，不降维、不学映射：CLIP 本身的上限 |
| `shared_pca_identity` | 英文和图像共用一个 PCA，不学映射 |
| `shared_pca_procrustes` | 共用 PCA 后再学 Procrustes 映射（实验实际使用的 bridge） |
| `shared_pca_map_distance_from_identity` | 学到的映射离单位矩阵多远；接近 0 说明 CLIP 已经对齐 |
| `separate_pca_procrustes` | 原来的做法（各自 PCA），用于和旧结果对照 |

**开工关卡**：`shared_pca_procrustes` 的 R@1 应该远高于原先的 11%。如果 32 维损失太多（明显低于 `clip_native`），用 `DIM=64` 或更高。

## 共用 PCA（`--shared-ei-pca`）

CLIP 的英文和图像共享坐标，各自做 PCA 会给两者换上不同的坐标轴。共用 PCA 时，英文和图像**各用自己的均值中心化**（去掉 CLIP 的模态差距），再在合并后的数据上拟合同一组主轴。`run.py`、`run_rosetta3.py`、`run_rosetta_baseline.py` 都支持这个参数，默认关闭，旧实验结果不变。

## 新增方法 `supervised`

只用平行句做正交 Procrustes，不用任何无配对数据；预处理与 Rosetta 相同。用它可以分出三层：

| 比较 | 回答的问题 |
|---|---|
| rosetta − supervised | 无配对文本带来多少 |
| visual − rosetta | 再加入图像带来多少 |
| visual − gaussian / english_pool / anchor_downweight | 图像的提升是否来自内容，而不是权重或“多一项” |

预算为 0 时自动跳过 `supervised`。预算低于维度（例如 10 对、32 维）时 Procrustes 欠定，这是该基线在极少配对下的真实弱点，不是错误。

## 注意

- CLIP 的文本编码器是用图像训练的，英文空间本身已含视觉信息，visual 与 bank_text 的差距可能比原设置更小；这不影响“加入图像比不加更好”的检验。
- CLIP 预训练数据可能包含 Flickr 图片，论文中需说明。
- 这里得到的是表示层面的图文能力（检索、零样本分类），不包括生成或问答。
