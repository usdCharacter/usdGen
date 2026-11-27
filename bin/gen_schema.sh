#!/bin/bash
# bin/gen_schema.sh — regenerate the codeless usdGen schema plugin resources
# (plugInfo.json + generatedSchema.usda) from libs/hosts/usdGenUsd/schema/schema.usda.
#
# Mirrors usdRig's bin/gen_schema.sh. Run this ONLY after editing
# libs/hosts/usdGenUsd/schema/schema.usda, and review the diff before committing.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]:-$0}")/_env.sh"

usdgen_require_python
usdgen_require_usd "usdGenSchema"

cd "$GEN/libs/hosts/usdGenUsd/schema"
"$PY" "$USD/bin/usdGenSchema" schema.usda resources

# usdGenSchema intentionally omits arbitrary property customData.  Restore
# usdGen evaluation-granularity metadata from the source schema deterministically
# so generatedSchema.usda remains an equivalent authoring contract.
"$PY" restore_generated_metadata.py schema.usda resources/generatedSchema.usda

# Post-process into the checked-in "resource" (data-only) form:
#  - strip the @PLUG_INFO_LIBRARY_PATH@ placeholder (the build-tree copy in
#    CMake re-adds LibraryPath + implementsComputeExtent),
#  - root the resource dir at ".".
# Nothing is auto-applied: usdGen declares no auto-apply API schemas.
"$PY" - <<'PY'
import io
p = 'resources/plugInfo.json'
s = io.open(p, encoding='utf-8').read()
s = s.replace('"LibraryPath": "@PLUG_INFO_LIBRARY_PATH@", \n', '')
s = s.replace('"LibraryPath": "@PLUG_INFO_LIBRARY_PATH@"', '')
s = s.replace('"@PLUG_INFO_RESOURCE_PATH@"', '"."')
s = s.replace('"@PLUG_INFO_ROOT@"', '"."')
lines = s.splitlines()
io.open(p, 'w', encoding='utf-8').write('\n'.join(line.rstrip() for line in lines) + '\n')
print("post-processed", p)
PY

echo "regenerated libs/hosts/usdGenUsd/schema/resources — review the diff before committing"
