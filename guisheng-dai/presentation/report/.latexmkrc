# Both documents need XeLaTeX (ctexbeamer + fontspec + local OTF fonts).
# Keeping this next to the sources makes plain `latexmk` / editor builds work
# without relying on a per-project editor setting.
$pdf_mode = 5;   # 5 = xelatex -> xdvipdfmx
