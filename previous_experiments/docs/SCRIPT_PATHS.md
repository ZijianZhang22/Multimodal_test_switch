# 脚本路径迁移表

根目录 Python / shell 文件已迁入以下目录。请先 cd 到仓库根目录；Python 使用 python -m，shell 使用 bash。数据与结果目录保留原位置。

| 旧路径 | 新路径 | 新命令 |
|---|---|---|
| `analyze_2v2t_200.py` | [scripts/analysis/analyze_2v2t_200.py](../scripts/analysis/analyze_2v2t_200.py) | `python -m scripts.analysis.analyze_2v2t_200` |
| `analyze_3t1v_500.py` | [scripts/analysis/analyze_3t1v_500.py](../scripts/analysis/analyze_3t1v_500.py) | `python -m scripts.analysis.analyze_3t1v_500` |
| `analyze_depth_scaling.py` | [scripts/analysis/analyze_depth_scaling.py](../scripts/analysis/analyze_depth_scaling.py) | `python -m scripts.analysis.analyze_depth_scaling` |
| `analyze_direct_trace_compare.py` | [scripts/analysis/analyze_direct_trace_compare.py](../scripts/analysis/analyze_direct_trace_compare.py) | `python -m scripts.analysis.analyze_direct_trace_compare` |
| `analyze_direct_vs_trace.py` | [scripts/analysis/analyze_direct_vs_trace.py](../scripts/analysis/analyze_direct_vs_trace.py) | `python -m scripts.analysis.analyze_direct_vs_trace` |
| `analyze_fact_level_patching.py` | [scripts/analysis/analyze_fact_level_patching.py](../scripts/analysis/analyze_fact_level_patching.py) | `python -m scripts.analysis.analyze_fact_level_patching` |
| `analyze_factorial_schedules.py` | [scripts/analysis/analyze_factorial_schedules.py](../scripts/analysis/analyze_factorial_schedules.py) | `python -m scripts.analysis.analyze_factorial_schedules` |
| `analyze_layerwise_mechanism.py` | [scripts/analysis/analyze_layerwise_mechanism.py](../scripts/analysis/analyze_layerwise_mechanism.py) | `python -m scripts.analysis.analyze_layerwise_mechanism` |
| `analyze_matched_state_decoding.py` | [scripts/analysis/analyze_matched_state_decoding.py](../scripts/analysis/analyze_matched_state_decoding.py) | `python -m scripts.analysis.analyze_matched_state_decoding` |
| `analyze_perception.py` | [scripts/analysis/analyze_perception.py](../scripts/analysis/analyze_perception.py) | `python -m scripts.analysis.analyze_perception` |
| `analyze_results.py` | [scripts/analysis/analyze_results.py](../scripts/analysis/analyze_results.py) | `python -m scripts.analysis.analyze_results` |
| `analyze_role_curve_exact.py` | [scripts/analysis/analyze_role_curve_exact.py](../scripts/analysis/analyze_role_curve_exact.py) | `python -m scripts.analysis.analyze_role_curve_exact` |
| `analyze_role_position.py` | [scripts/analysis/analyze_role_position.py](../scripts/analysis/analyze_role_position.py) | `python -m scripts.analysis.analyze_role_position` |
| `analyze_trace_states.py` | [scripts/analysis/analyze_trace_states.py](../scripts/analysis/analyze_trace_states.py) | `python -m scripts.analysis.analyze_trace_states` |
| `extract_internal_probe_states.py` | [scripts/mechanism/extract_internal_probe_states.py](../scripts/mechanism/extract_internal_probe_states.py) | `python -m scripts.mechanism.extract_internal_probe_states` |
| `generate_depth_scaling_dataset.py` | [scripts/data/generate_depth_scaling_dataset.py](../scripts/data/generate_depth_scaling_dataset.py) | `python -m scripts.data.generate_depth_scaling_dataset` |
| `generate_factorial_schedule_dataset.py` | [scripts/data/generate_factorial_schedule_dataset.py](../scripts/data/generate_factorial_schedule_dataset.py) | `python -m scripts.data.generate_factorial_schedule_dataset` |
| `generate_role_position_dataset.py` | [scripts/data/generate_role_position_dataset.py](../scripts/data/generate_role_position_dataset.py) | `python -m scripts.data.generate_role_position_dataset` |
| `generate_switching_dataset.py` | [scripts/data/generate_switching_dataset.py](../scripts/data/generate_switching_dataset.py) | `python -m scripts.data.generate_switching_dataset` |
| `mechanism_utils.py` | [scripts/mechanism/mechanism_utils.py](../scripts/mechanism/mechanism_utils.py) | `python -m scripts.mechanism.mechanism_utils` |
| `run_2v2t_200.sh` | [scripts/launch/run_2v2t_200.sh](../scripts/launch/run_2v2t_200.sh) | `bash scripts/launch/run_2v2t_200.sh` |
| `run_2v2t_400.sh` | [scripts/launch/run_2v2t_400.sh](../scripts/launch/run_2v2t_400.sh) | `bash scripts/launch/run_2v2t_400.sh` |
| `run_3t1v_500.sh` | [scripts/launch/run_3t1v_500.sh](../scripts/launch/run_3t1v_500.sh) | `bash scripts/launch/run_3t1v_500.sh` |
| `run_4hop_shortcot_pilot.sh` | [scripts/launch/run_4hop_shortcot_pilot.sh](../scripts/launch/run_4hop_shortcot_pilot.sh) | `bash scripts/launch/run_4hop_shortcot_pilot.sh` |
| `run_4hop_trace_control.sh` | [scripts/launch/run_4hop_trace_control.sh](../scripts/launch/run_4hop_trace_control.sh) | `bash scripts/launch/run_4hop_trace_control.sh` |
| `run_6hop_pilot.sh` | [scripts/launch/run_6hop_pilot.sh](../scripts/launch/run_6hop_pilot.sh) | `bash scripts/launch/run_6hop_pilot.sh` |
| `run_depth_scaling_model.sh` | [scripts/launch/run_depth_scaling_model.sh](../scripts/launch/run_depth_scaling_model.sh) | `bash scripts/launch/run_depth_scaling_model.sh` |
| `run_fact_level_patching.py` | [scripts/mechanism/run_fact_level_patching.py](../scripts/mechanism/run_fact_level_patching.py) | `python -m scripts.mechanism.run_fact_level_patching` |
| `run_internal_mechanism_suite.sh` | [scripts/launch/run_internal_mechanism_suite.sh](../scripts/launch/run_internal_mechanism_suite.sh) | `bash scripts/launch/run_internal_mechanism_suite.sh` |
| `run_internvl14b_4hop_replication.sh` | [scripts/launch/run_internvl14b_4hop_replication.sh](../scripts/launch/run_internvl14b_4hop_replication.sh) | `bash scripts/launch/run_internvl14b_4hop_replication.sh` |
| `run_internvl14b_factorial_suite.sh` | [scripts/launch/run_internvl14b_factorial_suite.sh](../scripts/launch/run_internvl14b_factorial_suite.sh) | `bash scripts/launch/run_internvl14b_factorial_suite.sh` |
| `run_internvl14b_full_role_curve.sh` | [scripts/launch/run_internvl14b_full_role_curve.sh](../scripts/launch/run_internvl14b_full_role_curve.sh) | `bash scripts/launch/run_internvl14b_full_role_curve.sh` |
| `run_internvl35.py` | [scripts/inference/run_internvl35.py](../scripts/inference/run_internvl35.py) | `python -m scripts.inference.run_internvl35` |
| `run_internvl_3t1v_500.sh` | [scripts/launch/run_internvl_3t1v_500.sh](../scripts/launch/run_internvl_3t1v_500.sh) | `bash scripts/launch/run_internvl_3t1v_500.sh` |
| `run_layerwise_mechanism.py` | [scripts/mechanism/run_layerwise_mechanism.py](../scripts/mechanism/run_layerwise_mechanism.py) | `python -m scripts.mechanism.run_layerwise_mechanism` |
| `run_qwen3vl.py` | [scripts/inference/run_qwen3vl.py](../scripts/inference/run_qwen3vl.py) | `python -m scripts.inference.run_qwen3vl` |
| `train_internal_probes.py` | [scripts/mechanism/train_internal_probes.py](../scripts/mechanism/train_internal_probes.py) | `python -m scripts.mechanism.train_internal_probes` |

mechanism_utils 是共享模块，供其他脚本导入；不作为独立实验入口。
旧分支不受此次路径变更影响。运行旧笔记中的命令前，按此表更新路径。
