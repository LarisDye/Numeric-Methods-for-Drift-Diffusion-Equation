# 从已有汇总 JSON 和图表构建文稿，不重跑实验，也不调用 03 汇总脚本。
$ErrorActionPreference = 'Stop'
# PSScriptRoot 是本脚本所在的 report/，上一级是 guisheng-dai/；切换后相对路径保持稳定。
$taskRoot = Split-Path -Parent $PSScriptRoot
Push-Location $taskRoot
try {
    # 先把实测汇总填入报告、Beamer 及可编辑演示稿的内容源文件。
    python scripts/04_build_documents.py
    # PowerShell 不会因外部程序非零退出码自动抛异常，逐步检查以免继续打包旧文件。
    if ($LASTEXITCODE -ne 0) { throw 'Document source generation failed' }
    Push-Location report
    try {
        # 两份文稿各编译两遍，以稳定交叉引用、页码及 Beamer 导航等辅助信息。
        xelatex -interaction=nonstopmode -halt-on-error report.tex
        if ($LASTEXITCODE -ne 0) { throw 'Report compilation failed' }
        xelatex -interaction=nonstopmode -halt-on-error report.tex
        if ($LASTEXITCODE -ne 0) { throw 'Report second pass failed' }
        xelatex -interaction=nonstopmode -halt-on-error slides.tex
        if ($LASTEXITCODE -ne 0) { throw 'Slides compilation failed' }
        xelatex -interaction=nonstopmode -halt-on-error slides.tex
        if ($LASTEXITCODE -ne 0) { throw 'Slides second pass failed' }
    } finally { Pop-Location }
    # 单独打包原生 PPTX，图表内嵌 XLSX，不将 PDF 页面的截图作为幻灯片内容。
    python scripts/05_build_pptx.py
    if ($LASTEXITCODE -ne 0) { throw 'PowerPoint generation failed' }
# 即使中途失败，finally 也恢复调用者工作目录，避免影响后续命令。
} finally { Pop-Location }
