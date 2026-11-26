# bin/gen_schema.ps1 — Windows twin of bin/gen_schema.sh: regenerate the
# codeless usdGen schema plugin resources (plugInfo.json + generatedSchema.usda)
# from libs/hosts/usdGenUsd/schema/schema.usda. Run this ONLY after editing schema.usda
# and review the diff before committing. Keep the post-processing here in
# step with gen_schema.sh.

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Build = Join-Path $Root "build"

$UsdInstallDir = $env:USD
if (-not $UsdInstallDir) {
    $UsdInstallDir = Join-Path (Split-Path $Root) "usdRig\usd-install"
}
if (-not (Test-Path $UsdInstallDir)) {
    throw "OpenUSD prefix '$UsdInstallDir' does not exist. Set USD to the install prefix that usdRig builds against."
}
$UsdInstallDir = (Resolve-Path $UsdInstallDir).Path
$tool = Join-Path $UsdInstallDir "bin\usdGenSchema"
if (-not (Test-Path $tool)) { throw "usdGenSchema is not installed in '$UsdInstallDir'." }

# pxr must be importable and its DLLs loadable, as in launch_usdview.ps1.
$env:PATH = ((Join-Path $UsdInstallDir "bin") + ';' + (Join-Path $UsdInstallDir "lib") + ';' + $env:PATH)
# usdGenSchema resolves @usdGeom/schema.usda@ through the plugin registry,
# so the stock OpenUSD plugin roots must be discoverable (bin/_env.sh sets
# the same two roots on Linux; without them the tool exits 1).
$pluginDirs = @(
    (Join-Path $UsdInstallDir "plugin\usd"),
    (Join-Path $UsdInstallDir "lib\usd")
) | Where-Object { Test-Path $_ }
$env:PXR_PLUGINPATH_NAME = ($pluginDirs -join ';') + $(if ($env:PXR_PLUGINPATH_NAME) { ';' + $env:PXR_PLUGINPATH_NAME } else { '' })
$pythonDirs = @(
    (Join-Path $Build "python"),
    (Join-Path $UsdInstallDir "Lib\site-packages"),
    (Join-Path $UsdInstallDir "lib\python")
) | Where-Object { Test-Path $_ }
$env:PYTHONPATH = ($pythonDirs -join ';') + $(if ($env:PYTHONPATH) { ';' + $env:PYTHONPATH } else { '' })
$python = $env:PY
if (-not $python) {
    $python = Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
}
if (-not $python) { throw "Python is not on PATH. Set PY to the interpreter that imports this build's pxr." }

$schemaDir = Join-Path $Root "libs\hosts\usdGenUsd\schema"
$resources = Join-Path $schemaDir "resources"
# usdGenSchema banners on stderr; with $ErrorActionPreference = "Stop" that
# alone would throw, so tolerate stderr here — the LASTEXITCODE checks below
# are the real gates.
$oldEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
Push-Location $schemaDir
try {
    & $python $tool schema.usda $resources
    if ($LASTEXITCODE -ne 0) { throw "usdGenSchema failed with exit code $LASTEXITCODE" }
    # usdGenSchema omits arbitrary property customData; restore the usdGen
    # evaluation-granularity metadata deterministically (see gen_schema.sh).
    & $python restore_generated_metadata.py schema.usda (Join-Path $resources "generatedSchema.usda")
    if ($LASTEXITCODE -ne 0) { throw "restore_generated_metadata.py failed with exit code $LASTEXITCODE" }
} finally {
    $ErrorActionPreference = $oldEap
    Pop-Location
}

# Post-process plugInfo.json into the checked-in resource form: drop the
# LibraryPath placeholder and root the resource dir at ".".  Nothing is
# auto-applied: usdGen declares no auto-apply API schemas.
$post = @'
import io, sys
p = sys.argv[1]
# The Windows usdGenSchema emits trailing spaces after commas in plugInfo.json
# and, in generatedSchema.usda, writes the carriage returns of the doc strings
# it copies from the stock schemas as a literal "\r" escape at the line end;
# the checked-in resources are the Linux form, so normalise both here.
g = sys.argv[2]
t = io.open(g, encoding='utf-8', newline='').read()
t = t.replace('\r\n', '\n').replace('\\r\n', '\n')
io.open(g, 'w', encoding='utf-8', newline='\n').write(t)
s = io.open(p, encoding='utf-8').read()
s = s.replace('"LibraryPath": "@PLUG_INFO_LIBRARY_PATH@", \n', '')
s = s.replace('"LibraryPath": "@PLUG_INFO_LIBRARY_PATH@"', '')
s = s.replace('"@PLUG_INFO_RESOURCE_PATH@"', '"."')
s = s.replace('"@PLUG_INFO_ROOT@"', '"."')
lines = [line.rstrip() for line in s.splitlines()]
io.open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(line.rstrip() for line in lines) + '\n')
print("post-processed", p)
'@
# -I keeps the helper's own directory off sys.path (a stray module in a temp
# directory must not shadow the standard library) and needs no pxr.
$postFile = Join-Path ([IO.Path]::GetTempPath()) "usdgen_postprocess_pluginfo.py"
Set-Content -Path $postFile -Value $post -Encoding ascii
& $python -I $postFile (Join-Path $resources "plugInfo.json") (Join-Path $resources "generatedSchema.usda")
if ($LASTEXITCODE -ne 0) { throw "plugInfo post-processing failed with exit code $LASTEXITCODE" }
Remove-Item $postFile -ErrorAction SilentlyContinue

Write-Host "regenerated libs/hosts/usdGenUsd/schema/resources - review the diff before committing"
