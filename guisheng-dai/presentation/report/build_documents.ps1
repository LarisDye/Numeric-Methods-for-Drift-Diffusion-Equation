# Compatibility entry point. Compile existing source; never regenerate it.
$ErrorActionPreference = 'Stop'
& python (Join-Path $PSScriptRoot 'build_slides.py')
if ($LASTEXITCODE -ne 0) { throw 'Slides compilation failed' }
