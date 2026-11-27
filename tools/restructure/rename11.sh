#!/usr/bin/env bash
# usage: tools/restructure/rename11.sh <dir>...   (libs/base and libs/core only)
set -euo pipefail
files=$(git ls-files -- "$@" | grep -E '\.(h|cpp|cu|cuh|inl)$')
perl -0pi -e '
  s/\bGfVec([234])([fdih])\b/UsdGenVec$1$2/g;
  s/\bGfMatrix([234])([df])\b/UsdGenMatrix$1$2/g;
  s/\bGfQuat([fdh])\b/UsdGenQuat$1/g;
  s/\bGfRange3([fd])\b/UsdGenRange3$1/g;
  s/\bGfHalf\b/UsdGenHalf/g; s/\bGfTimeCode\b/UsdGenTimeCode/g;
  s/\bGfDot\(/UsdGenDot(/g; s/\bGfCross\(/UsdGenCross(/g; s/\bGfSqrt\(/std::sqrt(/g;
  s/\bVtArray</UsdGenCowArray</g;
  s/\bVt(Float|Int|Bool|UInt64|Token)Array\b/UsdGen$1Array/g;
  s/\bVt(Vec[234][fdhi]|Matrix4d)Array\b/UsdGen$1Array/g;
  s/\bVtValue\b/UsdGenValue/g; s/\bTfToken\b/UsdGenToken/g; s/\bTfSpan\b/UsdGenSpan/g;
  s/\bTF_UNUSED\((\w+)\)/(void)$1/g;
  s/\bTRACE_FUNCTION\(\)/USDGEN_TRACE_FUNCTION()/g; s/\bTRACE_SCOPE\(/USDGEN_TRACE_SCOPE(/g;
  s/\bTRACE_COUNTER_DELTA\(/USDGEN_TRACE_COUNTER(/g;
  s/\bPXR_NS::|\bpxr::(?=Gf|Vt|Tf)//g;
  s{#include "pxr/base/gf/[^"]+"}{#include "usdGenMath/math.h"}g;
  s{#include "pxr/base/vt/(array|types)\.h"}{#include "usdGen/api/cowArray.h"}g;
  s{#include "pxr/base/vt/value\.h"}{#include "usdGen/api/value.h"}g;
  s{#include "pxr/base/tf/token\.h"}{#include "usdGen/api/token.h"}g;
  s{#include "pxr/base/tf/span\.h"}{#include "usdGen/api/span.h"}g;
  s{#include "pxr/base/trace/trace\.h"}{#include "usdGen/api/trace.h"}g;
' $files
# Deduplicate identical #include lines, then remove pxr.h and the using
# directive where no pxr name remains:
for f in $files; do
  perl -0pi -e '1 while s/^(#include "[^"]+"\r?\n)((?:.*\n)*?)\1/$1$2/m' "$f"
  if ! grep -qE '\b(Tf|Gf|Vt|Trace|Arch|Work|PxOsd|Hd)[A-Z]|PXR_NS|TF_|TRACE_' "$f"; then
    perl -ni -e 'print unless /^\s*(PXR_NAMESPACE_USING_DIRECTIVE|#include "pxr\/pxr\.h")\s*$/' "$f"
  fi
done
