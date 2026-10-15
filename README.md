# Multimodal Test Switch

研究图文证据的形式、位置与额外任务如何影响 VLM 的回答准确率和生成开销，并用内部表征分析区分可能的机制。

本仓库包含三条互相关联的实验线：**受控 toy → 真实 benchmark → 机制诊断**。额外任务 / compression 是另一类干预，应单独比较。

## 从哪里开始

| 想做什么 | 入口 | 实验含义 |
|---|---|---|
| 重跑 100 题、CoT、TTTT 对 1T3V | [toy_cot_100](toy_cot_100/README.md) | 同一条四步空间链，改变唯一文本事实所在的逻辑步骤 |
| 重跑 500 题、TTTT 对 3T1V | [run_3t1v_500.sh](run_3t1v_500.sh) | 唯一视觉事实所在步骤；与上一项是不同设计 |
| 分离逻辑角色、呈现顺序、单事实感知 | [实验索引](docs/EXPERIMENTS.md) | 根目录原有 factorial / role-position / perception 工具 |
| 在真实多图问题上换 caption | [Mantis](mantis_caption_swap/README.md)、[MMQA](mmqa_four_image/README.md)、[MMRB](mmrb_caption_first4/README.md) | 图片槽位替换；槽位不等于推理步骤 |
| 测额外任务与 token 开销 | [cross-task](vlm_cross_task_pilot/README.md)、[MME-CoT](mme_cot_conpress_pilot/README.md) | 联合任务、顺序与附加简单任务 |
| 分析逐层表征、probe、activation patching | [机制分析说明](docs/MECHANISM.md) | 当前工具主要观测语言 decoder；不能直接定位 encoder 丢失 |

## 建议阅读顺序

1. [实验索引与运行入口](docs/EXPERIMENTS.md)：条件、数据、命令和指标。
2. [机制分析：已经测到哪里](docs/MECHANISM.md)：观测对象与可支持的解释。
3. [结果状态与比较规范](docs/STATUS.md)：哪些已验证，哪些还需要原始结果。
4. [分支来源](docs/BRANCHES.md)：已归入 main 的实验及对应快照。

[RESEARCH_PLAN.md](RESEARCH_PLAN.md) 保留历史研究计划；旧版逐轮 README 已完整保存在 [历史记录](docs/legacy/README_rounds.md)。历史数值不代表此次重新验证的结论。

## 快速开始：100 题 toy CoT

```bash
git clone https://github.com/ZijianZhang22/Multimodal_test_switch.git
cd Multimodal_test_switch
pip install -r requirements.txt
bash toy_cot_100/run_runpod.sh
```

默认 Qwen3-VL-8B-Instruct，100 个 latent problems × 五个条件，共 500 次生成。条件为 **TTTT / TVVV / VTVV / VVTV / VVVT**，上限 4096 output tokens。安装与 GPU 要求以各实验 README 为准；根目录 requirements 不替代各 benchmark 的 setup。

改变模型、prompt、token budget 或条件时使用新的结果目录。不要让两个进程同时写同一 manifest / predictions 文件。

## 解释结果时

- **Toy 的 T** 是结构化事实；**benchmark 的 T** 是模型生成的 caption，可能丢失信息，二者不能直接合并统计。
- **输出 token**、**解析出的 thinking token**、**输入视觉 token** 是不同指标。Instruct 的可见 CoT 不是 Thinking 模型的隐藏推理段。
- token 减少只有结合准确率、截断率及配对样本才有意义；增加额外任务后的总 token 也不等于原任务独有开销。
- 可以将各实验归于“证据表示如何影响推理的可靠性与开销”；目前尚不能据此认定统一机制、纯 switching cost 或 encoder 信息丢失。

整理保留了原有脚本和目录路径；GPU 实验无需因为文档整理而重新运行。
