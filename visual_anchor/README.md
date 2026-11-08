# Visual anchors for low-resource language alignment — research prototype

这是提案核心算法的可运行实现，不是 SOTAlign / Rosetta 官方复现，不包含已验证的真实语言实验结果。
核心 run.py 从冻结特征开始运行；新增 run_real.sh 可自动下载并编码真实中文 pilot，见文末。编码器并非经过严格预训练审计的“纯单语”模型。

## 快速运行

```bash
bash run_demo.sh
```

脚本建立虚拟环境、安装 NumPy/SciPy、运行单元测试、生成合成数据并运行六种方法和三个随机种子。
合成数据有已知共享潜在结构，只用于验证程序管线，不能证明视觉有用。

## 真实数据运行

准备下面约定的 `features.npz` 后：

```bash
python run.py --data features.npz --dim 32 --budget 50 --seeds 0 1 2 --no-cycle --out results_50
python run.py --data features.npz --dim 32 --budget 200 --seeds 0 1 2 --no-cycle --out results_200
python run.py --data features.npz --dim 32 --budget 1000 --seeds 0 1 2 --no-cycle --out results_1000
# 伪配对消融，保持其他配置一致
python run.py --data features.npz --dim 32 --budget 50 --seeds 0 1 2 --out results_50_cycle
# 非平衡 OT 消融
python run.py --data features.npz --dim 32 --budget 50 --tau .5 --no-cycle --out results_50_unbalanced
```

先固定 d=32 比较各预算；不要让维度随预算变化后把差异全部归因于监督数量。
所有输出指标在 [0,1]；乘以 100 得百分比，差值乘以 100 得百分点。
参数目前是初始默认值，不是通过真实数据验证的推荐最优值。
小预算可在预算内划分/交叉验证，禁止用测试集挑参数。

## 特征文件格式

使用 `np.savez`；特征是有限数值的二维数组，每个空间允许不同原始宽度。
同一空间的所有数组必须来自相同冻结编码器、池化方式和原始特征宽度。
每个键都必须附带 `<key>_ids` 一维 Unicode 字符串数组，长度等于该数组行数。
IDs 应标记语义组：同图不同描述、平行句共享 ID；跨 split 的重复必须复用 ID。

| 键 | 含义 |
|---|---|
| l_fit / e_fit / i_fit | 仅训练集特征，用于各自 PCA；可与相应训练池重合 |
| l_pair / e_pair | 可用平行句候选，逐行配对；运行时按预算抽样 |
| i_bridge / e_bridge | 英文图文监督训练，逐行配对，拟合冻结的 V_IE |
| l_pool / e_pool | 无配对训练池，行顺序无关联 |
| i_bank / e_bank | 图像池及其英文描述，逐行配对，用于视觉 OT 与投票 |
| e_extra | 与其他训练数据不重叠的额外英文池，至少与 i_bank 等长 |
| l_test / e_test | 测试平行句，匹配由 IDs 定义 |
| l_img_test / i_test | 测试低资源描述和图像，多描述共享图像 ID |

同图多个描述时，训练桥可以重复图像行；投票实现假设一个 bank 行对应一条描述，建议 bank 每图选一条，避免重复图像造成任意 tie。
评测支持多描述，不把不同语言的 XM3600 描述误当逐句平行翻译。
检查只能发现 ID 层面的泄漏。你仍需先做文本规范化去重、图像 hash/近重复检测、编码器预训练来源审计。
特征模型、revision、池化方式、语料来源及具体性筛选规则应另行记录在数据清单中。
独立编码器并不自动意味着没有跨语言知识；多语预训练与平行监督也不是同一概念。

## 方法与主要修正

- supervised：监督 Procrustes。
- text：监督锚点 + 文本无配对 OT。
- visual：监督锚点 + 文本 OT + 图像映射到英文后的 OT。
- extra_text：第三项换为等量额外英文描述。
- gaussian：第三项换为单位归一化高斯点，检验无语义的几何正则化作用。
- broken_pairs：图像几何不变，仅打乱投票用的英文–图像对应关系。关闭 cycle 时应与 visual 一致；它不是视觉 OT 的负控制。

主分析先用 `--no-cycle` 比较 supervised/text/visual/extra_text/gaussian。
再开启 cycle 比较 visual 与其 no-cycle 版本、broken_pairs；额外文本控制也使用相同投票权重。
高斯控制不保留真实数据协方差/聚类结构；若要声称独立视觉信息，仍需更严格的领域和分布匹配控制。

公式（行向量）：L W -> E，I V_IE -> E。
目标是平均监督平方误差 + lambda_E OT(LW,E) + lambda_I OT(LW,I V_IE)。
固定 coupling 后，W=polar(平均锚点互协方差 + lambda_E L^T P_E E + lambda_I L^T P_I I V_IE)。
权重是平均损失尺度，避免预算变大仅仅改变各项相对权重。
默认 lambda_E=lambda_I=.25、lambda_p=.1；在 core.fit 接口中可配置。
每步重新抽样无配对 minibatch，交替求 Sinkhorn 和正交映射；不是全池目标单调下降保证。
平衡 OT 默认，--tau 启用 KL 边缘松弛；unbalanced coupling 保留原始总质量。

PCA 只在训练数据 fit，默认不白化；--whiten 0 到 1 控制部分/完全白化，之后 L2 归一化。
完全白化可能削弱协方差方向信息，OT 的分布匹配也不保证语义匹配。
50 对监督对 256 维映射存在秩不足；低维是简单起点，不是理论上彻底解决可辨识性。

## 输出和解释

- report.json：四个检索方向的 R@1/5/10、跨种子均值/样本标准差、伪配对数量和监督误差。
- 成对 bootstrap 按语义 ID 聚类，逐 seed 提供 R@1 差异的 95% 区间；没有跨种子联合推断或多重比较校正。
- 每种方法和 seed 的 NPZ：W、V_IE、训练 PCA 的均值与投影、监督抽样下标。
- bridge_train_cka 仅描述训练桥的配对几何，不是测试泛化证据，也不作为模型选择准则。

推理：先用保存的 mean/proj 做 (x-mean)@proj，再按行 L2 归一化，低资源乘 W、图像乘 V_IE。
CKA 需要语义匹配的行；不能随意排列两个无配对池再声称测得语义同构性。

## 当前边界

这是 CPU/NumPy 核心原型：minibatch OT 内存 O(batch²)，PCA 用 full SVD，CSLS 评测用稠密矩阵。
适合先验证几千条缓存特征。50 万样本高维 PCA、全池搜索须换 randomized/incremental PCA、分块/FAISS 检索和 GPU OT。
没有实现：零配对 Rosetta 初始化、SOTAlign KLOT、GW、残差 MLP、编码器训练、伪配对真值精度审计、VLM 接入。真实中文 pilot 已新增自动下载与冻结特征提取。
伪配对精度需要独立审计标签；不能把路线一致比例当作正确率。

视觉可能使文本检索下降却使图文检索上升，应分别解释。真正低资源语言、远距离语言未必受益更多，这些是待检验假设。
先用 held-out 英文图文桥检索验证 DINO->英文线性映射是否有用，再投入三语言大规模实验。
不要把嵌入检索提升写成 VLM 推理能力提升。

## 一键真实中文实验（新增）

```bash
bash run_real.sh
```

默认 CUDA。脚本下载真实 Flickr8K 图片及 Flickr8K-CN 标注，加载中文 text2vec、英文 MiniLM、DINOv2-small，缓存特征后运行 50/200/1000 对监督预算 × 6 方法 × 3 seeds。中文训练锚点来自数据集已有 Google 机器翻译；句子测试使用人工翻译，图文测试使用人工撰写中文，每图固定取第 0 条描述。没有用随机特征代替真实特征。

这是**中文少监督真实数据 pilot**，不是 Swahili 低资源实验，不是 FLORES/XM3600 全协议，也不证明预训练编码器从未见过测试数据。全部编码器冻结，使用三个不同 checkpoint；不宣称严格独立预训练。

训练图片固定划分为：1000 个平行句候选、1000 个纯中文池、1500 个英文图文桥、500 个英文图文 bank、500 个额外英文控制。纯中文池对应的图片和英文不进入训练；锚点图片也不进入图像训练。只向优化器暴露抽到预算内的平行对应；其余候选只参与训练 PCA 的单语分布估计。这里控制的是平行标注预算，不是预训练语言资源。held-out English/image 桥诊断输出单独 JSON，不参与调参。

自动检查官方 split、图片 SHA256、规范化文本与验证/测试集的完全重复；未做近重复/预训练语料审计。图片来自 Flickr8K 的第三方下载镜像，标注来自作者仓库的固定 commit，运行时记录模型 revision。请遵守原数据/模型许可证与引用要求。

默认评测官方测试集最多 1000 张图片的首条描述。`TEST_LIMIT=100` 可先试运行，但候选集规模改变后检索指标不可直接比较。

```bash
# 较小真实数据运行：仍下载图片压缩包并提取完整训练数据
BUDGETS=50 SEEDS=0 ITERATIONS=3 TEST_LIMIT=100 bash run_real.sh
# CPU 可运行，但特征提取会更慢
DEVICE=cpu bash run_real.sh
```

需要能访问 GitHub release、raw.githubusercontent.com、Hugging Face 与 PyPI。脚本自动安装依赖；使用 RunPod PyTorch CUDA 模板可复用其 PyTorch。图片压缩包是完整下载，预留足够磁盘空间。下载断点保存在 `.part`，特征按 batch 原子缓存，可重跑继续；训练阶段重跑会重算对应报告。

产物：`real_work/summary.csv`、每预算 `report.json`、`manifest.json`、`model_revisions.json`、`bridge_test.json`、映射权重。缓存文件夹按数据与模型配置散列隔离。

本环境验证了 9 项离线测试、语法和真实标注下载/解析（文本去重后保留 5161 个训练 image ID，1000 个测试 image ID 首条描述完整），尚未执行真实图片下载和 GPU 编码全流程；网络或依赖问题会明确报错，不回退到合成数据。

来源：https://github.com/li-xirong/flickr8kcn （Li et al., Adding Chinese Captions to Images, ICMR 2016）；https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k；https://huggingface.co/shibing624/text2vec-base-chinese；https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2；https://huggingface.co/facebook/dinov2-small。
