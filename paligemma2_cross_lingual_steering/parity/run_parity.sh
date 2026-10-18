#!/usr/bin/env bash
# One-click, upstream-aligned PaliGemma2-3B text-steering pipeline (EN parity + EN/ZH/SW).
#   bash run_parity.sh                     # full run
#   LIMIT=3 bash run_parity.sh             # smoke test (3 samples/task, separate run dir)
# All knobs are environment variables; see README "parity" section.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"; PKG="$(cd .. && pwd)"
REV="ce1c0f0cffed9b859901ce347f9d2b869a512178"

DATA="${DATA:-$PKG/outputs/parity_cvbench}"
N_TUNE="${N_TUNE:-150}"                     # per task; 0 = whole upstream train split (exact parity, ~4x cost)
LAYERS="${LAYERS:-5-20}"                    # upstream eval.sh
ALPHAS="${ALPHAS:-0.1,0.2,0.4,0.6,0.8,1.0}" # upstream eval.sh
CONCEPTS="${CONCEPTS:-spatial,counting,entity,attribute}"
CONCEPT_MODE="${CONCEPT_MODE:-best}"        # best (chosen on EN tune) | aligned (count->counting, else spatial)
VARIANTS="${VARIANTS:-en,zh,sw,zh_normen,sw_normen,random_s0}"
STYLES="${STYLES:-native}"                  # add ,upstream_en to also run the legacy mixed-language prompt
RANDOM_SEEDS="${RANDOM_SEEDS:-0}"
RETUNE_PER_LANG="${RETUNE_PER_LANG:-0}"     # 1 = stage C: tune layer/alpha per prompt x vector language
C_LAYERS="${C_LAYERS:-$LAYERS}"; C_ALPHAS="${C_ALPHAS:-$ALPHAS}"
LIMIT="${LIMIT:-0}"
INSTALL="${INSTALL:-1}"
STRICT_LEXICON="${STRICT_LEXICON:-0}"
RUN="${RUN:-$([[ "$LIMIT" == 0 ]] && echo main || echo "smoke_$LIMIT")}"
OUT="$DATA/runs/$RUN"

echo "== parity pipeline | DATA=$DATA RUN=$RUN N_TUNE=$N_TUNE LAYERS=$LAYERS ALPHAS=$ALPHAS =="

# 1. upstream code, pinned
if [[ ! -d "$PKG/.upstream/.git" ]]; then
  git clone --quiet https://github.com/Woodygan/TextSteersVision.git "$PKG/.upstream"
fi
git -C "$PKG/.upstream" fetch --quiet origin "$REV" 2>/dev/null || true
git -C "$PKG/.upstream" checkout --quiet "$REV"
if ! git -C "$PKG/.upstream" diff --quiet; then
  echo "ERROR: $PKG/.upstream has local modifications; parity requires the unmodified author code"; exit 1
fi

# 2. environment
PY="$PKG/.venv-parity/bin/python"
if [[ ! -x "$PY" ]]; then python -m venv --system-site-packages "$PKG/.venv-parity"; fi
if [[ "$INSTALL" == 1 ]]; then "$PY" -m pip install -q -r "$HERE/requirements.txt"; fi
"$PY" - <<'PY'
import torch, transformers
assert transformers.__version__ == "4.51.3", transformers.__version__
assert torch.cuda.is_available(), "CUDA GPU required"
print("torch", torch.__version__, "| transformers", transformers.__version__, "|", torch.cuda.get_device_name(0), flush=True)
PY
"$PY" - <<'PY' || { echo "ERROR: not logged in to Hugging Face. Run: hf auth login  (and accept the google/gemma-2-2b and google/paligemma2-3b-mix-448 licenses)"; exit 1; }
from huggingface_hub import whoami
print("HF user:", whoami()["name"])
PY

# 3. CPU protocol checks (scoring == upstream, split == upstream, translations == curated 840)
"$PY" -W ignore "$HERE/test_parity.py"

# 4. data + vectors (cached in DATA, shared by every RUN)
if [[ ! -f "$DATA/visual_eval.jsonl" ]]; then
  extra=(); [[ "$STRICT_LEXICON" == 1 ]] && extra+=(--strict-lexicon)
  "$PY" "$HERE/prepare.py" --data "$DATA" --n-tune "$N_TUNE" "${extra[@]}"
else
  have="$("$PY" -c "import json;print(json.load(open('$DATA/dataset_manifest.json'))['n_tune'])")"
  [[ "$have" == "$N_TUNE" ]] || { echo "ERROR: $DATA was prepared with N_TUNE=$have; use a new DATA path"; exit 1; }
  pv="$(cd "$HERE" && "$PY" -c "import json,common;m=json.load(open('$DATA/dataset_manifest.json'));print(m.get('protocol_version')==common.PROTOCOL_VERSION)")"
  if [[ "$pv" != True ]]; then
    echo "ERROR: $DATA prompts were built by an older protocol version. Rebuild them (images and vectors are kept):"
    echo "  rm $DATA/visual_eval.jsonl && bash run_parity.sh   (use a new RUN name)"; exit 1
  fi
fi
if [[ ! -f "$DATA/vectors/manifest.json" ]]; then
  "$PY" "$HERE/vectors.py" --data "$DATA" --random-seeds "$RANDOM_SEEDS"
fi
mkdir -p "$OUT"
EV=("$PY" "$HERE/evaluate.py" --data "$DATA" --limit "$LIMIT")

# 5. Stage A: English parity with the paper (EN prompt x EN vector, full upstream grid)
"$PY" "$HERE/plan.py" A-tune --out "$OUT/jobs_A_tune.json" --concepts "$CONCEPTS" --layers "$LAYERS" --alphas "$ALPHAS"
"${EV[@]}" --split tune --jobs "$OUT/jobs_A_tune.json" --out "$OUT/pred_A_tune.csv"
"$PY" "$HERE/select_best.py" --pred "$OUT/pred_A_tune.csv" --out "$OUT/best_A.json"
"$PY" "$HERE/plan.py" A-test --best "$OUT/best_A.json" --out "$OUT/jobs_A_test.json"
"${EV[@]}" --split test --jobs "$OUT/jobs_A_test.json" --out "$OUT/test_A_en_parity.csv"

# 6. Stage B: cross-lingual, hyper-parameters frozen from Stage A (no per-language selection)
"$PY" "$HERE/plan.py" B-test --best "$OUT/best_A.json" --out "$OUT/jobs_B_test.json" \
  --concept-mode "$CONCEPT_MODE" --variants "$VARIANTS" --styles "$STYLES"
"${EV[@]}" --split test --jobs "$OUT/jobs_B_test.json" --out "$OUT/test_B_crosslingual.csv"
TESTS=("$OUT/test_A_en_parity.csv" "$OUT/test_B_crosslingual.csv")

# 7. Stage C (optional): re-tune layer/alpha for every prompt x vector language
if [[ "$RETUNE_PER_LANG" == 1 ]]; then
  "$PY" "$HERE/plan.py" C-tune --best "$OUT/best_A.json" --out "$OUT/jobs_C_tune.json" \
    --concept-mode "$CONCEPT_MODE" --styles "$STYLES" --layers "$C_LAYERS" --alphas "$C_ALPHAS"
  "${EV[@]}" --split tune --jobs "$OUT/jobs_C_tune.json" --out "$OUT/pred_C_tune.csv"
  "$PY" "$HERE/select_best.py" --pred "$OUT/pred_C_tune.csv" --out "$OUT/best_C.json"
  "$PY" "$HERE/plan.py" C-test --best "$OUT/best_C.json" --out "$OUT/jobs_C_test.json"
  "${EV[@]}" --split test --jobs "$OUT/jobs_C_test.json" --out "$OUT/test_C_retuned.csv"
  TESTS+=("$OUT/test_C_retuned.csv")
fi

# 8. Analysis
"$PY" "$HERE/analyze.py" --pred "${TESTS[@]}" --out "$OUT/analysis"
cp "$DATA/vectors/vector_geometry.csv" "$DATA/vectors/vector_norms.csv" "$OUT/analysis/" 2>/dev/null || true
echo "FINISHED. Summary: $OUT/analysis/SUMMARY.md"
