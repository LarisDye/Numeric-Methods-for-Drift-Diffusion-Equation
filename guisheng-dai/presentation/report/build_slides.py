"""Compile the existing, hand-edited slides.tex without regenerating it."""
from pathlib import Path
import hashlib
import shutil
import subprocess


def main():
    report_dir = Path(__file__).resolve().parent
    source = report_dir / "slides.tex"
    # Keep our auxiliary files separate from the editor's automatic build.
    build_dir = report_dir / ".build-slides"
    build_dir.mkdir(exist_ok=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    for _ in range(2):
        subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error",
             "-file-line-error", "-jobname=slides-build",
             f"-output-directory={build_dir}", "slides.tex"],
            cwd=report_dir, check=True,
        )
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError(
            "slides.tex changed during compilation. Save the intended version and build again."
        )
    # Some Windows PDF viewers permit writing but hold a handle that blocks rename.
    shutil.copyfile(build_dir / "slides-build.pdf", report_dir / "slides.pdf")
    print(f"Compiled existing source: {source}")
    print(f"PDF: {report_dir / 'slides.pdf'}")


if __name__ == "__main__":
    main()
