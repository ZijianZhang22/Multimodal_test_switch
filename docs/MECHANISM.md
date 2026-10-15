# 机制分析：观测位置与证据边界

核心问题：视觉条件表现差，是视觉信息已经损失，还是信息仍存在但没有被后续推理正确使用？当前仓库可以提供部分诊断，尚未直接检验完整视觉 encoder → projector → decoder 链路。

| 工具 | 实际观测 / 干预 | 可以回答 | 不能单独证明 |
|---|---|---|---|
| [extract_internal_probe_states.py](../extract_internal_probe_states.py) | 输入 prefill 阶段 decoder hidden_states；事实边界与最后 prompt token | 不同层能否解码局部方向、累计坐标等标签 | encoder 是否丢失；生成 CoT 每一步如何变化 |
| [train_internal_probes.py](../train_internal_probes.py) | 局部事实、精确状态、配对视觉减文本差分等 probe | 标签可读性与成功/失败的相关性 | 模型实际使用了 probe 可读信息；统一因果机制 |
| [analyze_matched_state_decoding.py](../analyze_matched_state_decoding.py) | 配对样本、多 split 的分类与回归 | 匹配样本后的状态可读性对比 | 数据外泛化；已排除重复关系链泄漏 |
| [run_layerwise_mechanism.py](../run_layerwise_mechanism.py) | decoder 逐层分析与最后 prompt 位置的 patching | 特定位置的替换是否改变回答 | 所有视觉信息的存储位置 |
| [run_fact_level_patching.py](../run_fact_level_patching.py) | 事实边界 patching；matched / other-fact / unrelated 控制 | 是否存在选择性的局部回答改善 | 未改善即代表信息已丢失 |

## 现有流程

1. 准备 role-position 数据及配对 T/V 模型输出。
2. 提取状态、训练 probe，检查采样子集和 train/test 划分。
3. 在相同失败样本上做 patching，并对比非匹配事实与无关样本控制。
4. 联合报告可读性、干预后准确率、样本数量和选择规则。

[run_internal_mechanism_suite.sh](../run_internal_mechanism_suite.sh) 是历史 Round-6 编排脚本：硬编码了 /workspace/Multimodal_test_switch，且标为 Optional 的 patching 步骤实际上仍会执行。使用前检查路径与预算，或直接逐条运行其中命令；本次整理未改变实验执行行为。

## 下一步如何检验 encoder 假设

需要新增视觉 encoder 输出、projector 输出、decoder 输入及后续层的同一标签 probe，并在相同样本上做受控干预。视觉 token 数量、空间对应关系和跨模态对齐都需要单独设计；直接把不同位置的向量塞进 decoder 并不能自动构成有效救援实验。

建议先固定一个确实需要视觉细节的 toy 子集，比较：
- 单事实感知能否答对；
- 各阶段局部事实是否可读；
- 各阶段累计状态是否可读；
- 匹配事实的干预能否选择性修复，且优于无关控制。

可读但答错支持“可访问性 / 使用问题”的候选解释；不可读也可能来自 probe、标签、采样或表征形式不合适。需要干预与对照进一步区分。

## 结果边界

历史 probe 与 patching 叙述见 [旧记录](legacy/README_rounds.md) 和 [研究计划](../RESEARCH_PLAN.md)。此次只整理代码与文档，未取得原始 GPU 结果重新计算这些数值。现有代码不支持直接将问题归因于 encoder，也尚未沿生成中的 CoT token 逐步追踪视觉信息。
