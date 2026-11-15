#!/usr/bin/env bash
set -euo pipefail
report_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$report_root"
mkdir -p build
if command -v latexmk >/dev/null 2>&1; then
  latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main_zh.tex
elif command -v xelatex >/dev/null 2>&1; then
  for pass in 1 2 3; do xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main_zh.tex; done
else
  compiler="${TECTONIC_BIN:-/home/yd/.local/share/vla-report-tools/tectonic/tectonic}"
  if command -v tectonic >/dev/null 2>&1; then compiler="$(command -v tectonic)"; fi
  if [[ ! -x "$compiler" ]]; then
    echo '需要XeLaTeX或Tectonic。可设置TECTONIC_BIN为已有编译器路径。' >&2
    exit 1
  fi
  "$compiler" --keep-logs --keep-intermediates --outdir build main_zh.tex
fi
