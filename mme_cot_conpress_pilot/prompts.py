"""Keep the original target task prompt IDENTICAL in both experimental conditions."""

TARGET_SUFFIX = (
    "Think through the question carefully, using the image(s) when necessary. "
    "Show your reasoning and place your answer in "
    "<target_answer>YOUR_ANSWER</target_answer>."
)

ARITHMETIC_SUFFIX = (
    "\n\nAdditional independent task: What is 1 + 1? "
    "Answer this additional task as well. "
    "After the original task's <target_answer>...</target_answer>, "
    "provide <arithmetic_answer>YOUR_ANSWER</arithmetic_answer>. "
    "There is no requirement to shorten or change the original reasoning."
)


def build_prompt(row, condition):
    body = row["question"].strip()
    if row.get("options"):
        body += "\n\nOptions:\n" + "\n".join(row["options"])
    original = "Original multimodal question:\n" + body + "\n\n" + TARGET_SUFFIX
    if condition == "baseline":
        return original
    if condition == "plus_arithmetic":
        return original + ARITHMETIC_SUFFIX
    raise ValueError("Unknown condition " + condition)
