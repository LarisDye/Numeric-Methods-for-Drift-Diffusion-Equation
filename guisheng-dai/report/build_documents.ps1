$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Push-Location $taskRoot
try {
    python scripts/04_build_documents.py
    if ($LASTEXITCODE -ne 0) { throw 'Document source generation failed' }
    Push-Location report
    try {
        xelatex -interaction=nonstopmode -halt-on-error report.tex
        if ($LASTEXITCODE -ne 0) { throw 'Report compilation failed' }
        xelatex -interaction=nonstopmode -halt-on-error report.tex
        if ($LASTEXITCODE -ne 0) { throw 'Report second pass failed' }
        xelatex -interaction=nonstopmode -halt-on-error slides.tex
        if ($LASTEXITCODE -ne 0) { throw 'Slides compilation failed' }
        xelatex -interaction=nonstopmode -halt-on-error slides.tex
        if ($LASTEXITCODE -ne 0) { throw 'Slides second pass failed' }
    } finally { Pop-Location }
    python scripts/05_build_pptx.py
    if ($LASTEXITCODE -ne 0) { throw 'PowerPoint generation failed' }
} finally { Pop-Location }
