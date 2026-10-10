# VLM Cross-Task Interference Pilot

An experimental extension of the [Idis benchmark](https://github.com/effl-lab/Idis).

Use the official Idis-perception visually distracted images for a Q1 object-classification task, then pair each input with an unrelated math Q2 from official GSM8K or MATH-500. The default model is Qwen/Qwen3-VL-8B-Thinking.

## Run on RunPod (NVIDIA GPU with CUDA PyTorch)

    git clone https://github.com/ZijianZhang22/Multimodal_test_switch.git
    cd Multimodal_test_switch/vlm_cross_task_pilot
    bash run_runpod.sh

Default: 6 image–text pairs from 2 classes, conflicting/irrelevant Idis distractor variants, 1 sampled output per condition. Only selected Hugging Face images are downloaded.

Use harder math questions:

    BENCHMARK=math500 PAIRS=12 N_CLASSES=3 bash run_runpod.sh

Add reversed ordering:

    CONDITIONS=q2_only,q1_only,image_q2,joint_vt,joint_tv bash run_runpod.sh

Repeat 5 times (matching the paper's sampling count):

    SAMPLES=5 PAIRS=24 N_CLASSES=4 bash run_runpod.sh

## Conditions

- q2_only: independent text-only baseline (no image).
- q1_only: original Idis visual category task prompt plus Idis distracted image.
- image_q2: distracted image and math Q2, no visual Q1.
- joint_vt: visually answer Q1 and mathematically answer Q2.
- joint_tv: reverse question order (optional control).
- original visual images (optional): ORIGINAL_ROOT=/path/to/ImageNet-9/original/val.

## Results

Generated in pilot_gsm8k/results or pilot_math500/results:

- predictions.jsonl: model outputs
- scored.csv: predictions and validity of each answer
- summary.csv: accuracy, generation lengths, and truncation summary
- candidate_interference.csv: Q2 correct with image only but incorrect in Q1+Q2

Primary contrast: paired Q2 accuracy(joint_vt) minus Q2 accuracy(image_q2). Interpret carefully: additional instructions and outputs can influence accuracy separately from actual visual leakage. Review truncated/ill-formatted model outputs, and use larger matched samples and matched text-only task-load controls before claiming causal interference.

Reproduction notes: original Idis Q1 base prompt and 512x512 preprocessing, T=0.7/top-p=0.95, but the default pilot uses 1 generation (paper used 5). Inference uses transformers rather than vLLM. MATH-500 verification may require manual review for symbolic answers.

Datasets: https://huggingface.co/datasets/Vail-2000/Idis ; https://huggingface.co/datasets/openai/gsm8k ; https://huggingface.co/datasets/HuggingFaceH4/MATH-500.
