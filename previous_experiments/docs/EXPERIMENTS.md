# 实验索引

以下命令从仓库根目录执行，标明 cd 的除外。这里只列入口，完整参数、默认模型与输出文件见对应脚本或子目录 README。

## 1. 受控空间推理 toy

| 实验 | 数据 / 条件 | 入口 |
|---|---|---|
| 四步全因子 | 同一 latent chain 的 16 种 T/V 分配 | [generate_switching_dataset.py](../scripts/data/generate_switching_dataset.py)、[run_qwen3vl.py](../scripts/inference/run_qwen3vl.py)、[analyze_factorial_schedules.py](../scripts/analysis/analyze_factorial_schedules.py) |
| CoT 100 题，1T3V | TTTT、TVVV、VTVV、VVTV、VVVT | [toy_cot_100/README.md](../toy_cot_100/README.md) |
| Direct 500 题，3T1V | TTTT、VTTT、TVTT、TTVT、TTTV | [scripts/launch/run_3t1v_500.sh](../scripts/launch/run_3t1v_500.sh) |
| 逻辑角色 × 呈现位置 | 改变事实的逻辑步骤与呈现顺序；含 perception control | [generate_role_position_dataset.py](../scripts/data/generate_role_position_dataset.py)、[analyze_role_position.py](../scripts/analysis/analyze_role_position.py)、[analyze_perception.py](../scripts/analysis/analyze_perception.py) |
| 深度与显式轨迹 | hop 数、direct / short-CoT / coordinate trace 等设置 | [scripts/launch/run_depth_scaling_model.sh](../scripts/launch/run_depth_scaling_model.sh)、[scripts/launch/run_4hop_shortcot_pilot.sh](../scripts/launch/run_4hop_shortcot_pilot.sh)、[scripts/launch/run_4hop_trace_control.sh](../scripts/launch/run_4hop_trace_control.sh) |
| 跨模型复现 | InternVL 与 Qwen runner | [run_internvl35.py](../scripts/inference/run_internvl35.py)、[scripts/launch/run_internvl14b_factorial_suite.sh](../scripts/launch/run_internvl14b_factorial_suite.sh) |

100 题 CoT：

```bash
bash toy_cot_100/run_runpod.sh
```

全因子数据生成入口：

```bash
python -m scripts.data.generate_switching_dataset --out_dir data16 --n_problems 100 --seed 42 --schedule_set all16
python -m scripts.inference.run_qwen3vl --help
```

空间链可能重复。配对分析按 latent problem 对齐；probe / 泛化评估还需检查相同关系链是否跨 train/test。仅按 problem_id 分组不能自动排除重复链泄漏。

## 2. 真实 benchmark 的 caption 替换

| 实验 | 采样 | 替换条件 | 关键限制 |
|---|---|---|---|
| [Mantis-Eval](../mantis_caption_swap/README.md) | 默认随机 50；支持完整 217；每题 2–5 图 | All-V、All-T、逐张 T_at_i；每题 N+2 条件 | 不是 keep-one-V；物理图片位置不等于逻辑 hop |
| [MultiModalQA](../mmqa_four_image/README.md) | 目标 50 个 dev 问题，恰好四个候选图片文档，支持证据含图和另一模态 | VVVV、TTTT、VTTT、TVTT、TTVT、TTTV | 候选图可能是干扰项；不足 50 时 prepare 报错 |
| [MMRB](../mmrb_caption_first4/README.md) | 默认 50 题，4–10 图 | 前四槽位执行上述六条件；第 5 张起保留 V | TTTT 不一定是整个输入 all-text |

新 Pod 选择其中一个目录：

```bash
cd mantis_caption_swap
bash setup.sh
python experiment.py selftest
python experiment.py prepare --n 50 --seed 42
python experiment.py caption
python experiment.py evaluate --max-new-tokens 8192
python experiment.py summarize
python token_analysis.py
```

MMQA / MMRB 请按各自 README 准备数据，不复用 Mantis 的 prepare 命令。它们的现有 QA 默认预算为 1536；跨 benchmark 比较前需要明确预算差异。

三个实验的 caption 默认使用 LLaVA-1.5-7B、固定描述 prompt、最多 200 tokens，不传入下游问题。这个控制减少问题条件化差异，但不保证 caption 与图片信息等价。

## 3. 额外任务 / compression

| 实验 | 干预 | 入口 |
|---|---|---|
| Cross-task pilot | 视觉 Q1 与数学 Q2 联合、顺序、image-only 控制 | [vlm_cross_task_pilot/README.md](../vlm_cross_task_pilot/README.md) |
| Idis30 与 thinking 分析 | 扩大视觉分类与数学配对，解析推理段 token | [Idis30](../vlm_cross_task_pilot/README_IDIS30.md)、[thinking tokens](../vlm_cross_task_pilot/README_IDIS30_THINKING.md) |
| 文本 / 图片算术控制 | 附加任务的形式与图像保留控制 | [text_math_control](../vlm_cross_task_pilot/text_math_control/README.md)、[run_image_text_control.sh](../vlm_cross_task_pilot/run_image_text_control.sh)、[run_dual_image_q1_control.sh](../vlm_cross_task_pilot/run_dual_image_q1_control.sh) |
| MME-CoT | baseline 与追加简单 1+1 任务 | [mme_cot_conpress_pilot/README.md](../mme_cot_conpress_pilot/README.md) |
| MME-CoT 长推理筛选 | baseline 短预算筛选，再配对跑较大预算 | [run_long_reasoning.sh](../mme_cot_conpress_pilot/run_long_reasoning.sh)、[select_long_cases.py](../mme_cot_conpress_pilot/select_long_cases.py) |

```bash
cd mme_cot_conpress_pilot
bash run_runpod.sh
# 长推理版本为独立入口；默认最高 65536 tokens，运行成本明显更高
# bash run_long_reasoning.sh
```

目录名 conpress 是原有拼写，为兼容现有运行路径而保留。长推理子集按 baseline token 数筛选，结论只适用于该选择策略，不能直接视为整个 benchmark 的无偏估计。MME-CoT 自带的自由文本评分是代理评分，不能当作官方 judge 分数。

## 4. 机制分析

见 [机制说明](MECHANISM.md)。所有 probe / patching 都需要已有数据和模型结果，不是从空仓库一键开始。
