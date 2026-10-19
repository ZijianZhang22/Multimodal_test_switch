#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-../.venv-parity/bin/python}"
DATA="${DATA:-../outputs/parity_cvbench}"
RUN="${RUN:-combo_main}"
LANGS="${LANGS:-zh,sw}"
STYLE="${STYLE:-upstream_en}"
LIMIT="${LIMIT:-0}"
LAYERS="${LAYERS:-4-22}"
ALPHAS="${ALPHAS:-0.25,0.5,1.0,1.5,2.0}"
CONCEPTS="${CONCEPTS:-combo_paper_en_best.json}"
OUT="$DATA/runs/$RUN"
[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 1; }
[[ -f "$CONCEPTS" ]] || { echo "Missing concepts config $CONCEPTS" >&2; exit 1; }
mkdir -p "$OUT"
if [[ -f "$OUT/concepts.json" ]]; then
 cmp -s "$CONCEPTS" "$OUT/concepts.json" || { echo "Concept config changed; choose a new RUN" >&2; exit 1; }
else
 cp "$CONCEPTS" "$OUT/concepts.json"
fi
"$PY" test_combo.py
"$PY" lang_vectors.py --data "$DATA" --langs "$LANGS" --styles "$STYLE" --limit "$LIMIT" --reuse-if-valid
"$PY" combo_eval.py --data "$DATA" --part select --out "$OUT/select.csv" --langs "$LANGS" --style "$STYLE" --layers "$LAYERS" --alphas "$ALPHAS" --limit "$LIMIT"
"$PY" combo_select.py --pred "$OUT/select.csv" --out "$OUT/lang_config.json"
"$PY" combo_eval.py --data "$DATA" --part test --out "$OUT/test.csv" --langs "$LANGS" --style "$STYLE" --lang-config "$OUT/lang_config.json" --concepts "$OUT/concepts.json" --limit "$LIMIT"
"$PY" combo_analyze.py --pred "$OUT/test.csv" --out "$OUT/analysis"
echo "DONE $OUT/analysis/SUMMARY.md"
