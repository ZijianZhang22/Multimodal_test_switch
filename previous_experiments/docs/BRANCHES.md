# 分支与归档来源

本次整理将以下分支的新增实验文件按原路径归入 main。源代码使用原 blob，未改动实验逻辑。原分支保留；旧 Pod 可以继续使用原分支，新的实验建议从 main 开始。

| 原分支 | 来源 commit | 归入路径 |
|---|---|---|
| experiment/long-reasoning-64k | 1f68fb1b6c4914c1f807a59504e107ebda7d4342 | mme_cot_conpress_pilot/run_long_reasoning.sh、select_long_cases.py |
| experiments/mantis-eval-caption-swap | 46b5c594bf8b7cf17b1890055ac11eb006640e1e | mantis_caption_swap/ |
| experiments/mmqa-four-image-50 | 0c8e7d9af2832b2c8882d0345a7985f25cea6e76 | mmqa_four_image/ |
| experiments/mmrb-caption-first4 | 95cf3753f9ba473afbfc0435419768e2d461b746 | mmrb_caption_first4/ |

整理基于 main 的 881bed442566e9b49d3fe50c1f220f77a4afc82c。各分支与该 main 的同路径 blob 一致；本次归入的是新增文件，随后更新两个 README 的新 Pod 指引。

mmqa-one-vision-caption 指向历史 commit 3842e13ba9c1ecbf922e0fbecd81defc000410ee，本次未发现其相对 main 独有的实验文件。不能仅凭这个分支名判断实际 benchmark；请按实验目录及 manifest 确认。

## 历史文档

- [旧版逐轮 README 原文](legacy/README_rounds.md)：仅作历史记录，里面的相对路径按原仓库根目录理解。
- [历史研究计划](../RESEARCH_PLAN.md)：保留原文并增加状态提示。
- [当前实验索引](EXPERIMENTS.md)：作为现阶段入口。

上述归入操作保留了分支文件路径。后续目录整理将根目录 Python / shell 脚本迁入 scripts/，见 [迁移表](SCRIPT_PATHS.md)；数据、结果目录与已有分支保留。没有把 Pod 数据或模型权重加入 git。
