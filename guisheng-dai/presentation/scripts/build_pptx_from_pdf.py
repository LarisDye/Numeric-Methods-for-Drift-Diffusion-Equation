"""Preserve the current PDF pages in a local PPTX and embed the walk GIF.

Requires PyMuPDF, Pillow and Pandoc. Static content is a 400-dpi page image;
the animation is an independent embedded GIF on the detected placeholder page.
This does not compile or modify slides.tex or slides.pdf.
"""

from pathlib import Path
import argparse
import hashlib
import io
import json
import posixpath
import shutil
import subprocess
import zipfile
from datetime import datetime
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

import pymupdf
from PIL import Image


REPORT = Path(__file__).resolve().parents[1] / "report"
NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)
# xsi:type values contain this literal prefix; retain it during serialization.
ET.register_namespace("dcterms", "http://purl.org/dc/terms/")


def xml_bytes(node):
    return ET.tostring(node, encoding="utf-8", xml_declaration=True)


def element(xml):
    wrapper = ET.fromstring("<root " + " ".join(
        f'xmlns:{prefix}="{uri}"' for prefix, uri in NS.items()) + ">" + xml + "</root>")
    return wrapper[0]


def picture(sid, name, rid, x, y, w, h):
    return element(
        f'<p:pic><p:nvPicPr><p:cNvPr id="{sid}" name="{escape(name)}" '
        f'descr="{escape(name)}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/>'
        f'</p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="{rid}"/>'
        '<a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm>'
        f'<a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        '<a:ln><a:noFill/></a:ln></p:spPr></p:pic>'
    )


def white_mask(sid, x, y, w, h):
    return element(
        f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Animation placeholder cover"/>'
        '<p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm>'
        f'<a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        '<a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill>'
        '<a:ln><a:noFill/></a:ln></p:spPr></p:sp>'
    )


def find_placeholder(doc):
    candidates = []
    for index, page in enumerate(doc):
        labels = page.search_for("预留展示区域")
        if not labels:
            continue
        assert len(labels) == 1
        label = labels[0]
        center_x = (label.x0 + label.x1) / 2
        lines = [drawing["rect"] for drawing in page.get_drawings()
                 if drawing["type"] == "s" and drawing["rect"].height < 0.1
                 and 100 < drawing["rect"].width < page.rect.width * .6
                 and drawing["rect"].x0 < center_x < drawing["rect"].x1]
        top = max((r for r in lines if r.y0 < label.y0), key=lambda r: r.y0)
        bottom = min((r for r in lines if r.y0 > label.y1), key=lambda r: r.y0)
        assert abs(top.x0-bottom.x0) < 1 and abs(top.x1-bottom.x1) < 1
        candidates.append((index, pymupdf.Rect(top.x0, top.y0, bottom.x1, bottom.y0)))
    if len(candidates) != 1:
        raise ValueError(f"Expected one animation placeholder, found {len(candidates)}")
    return candidates[0]


def validate_package(path, expected_pages, animation_page, gif_bytes):
    with zipfile.ZipFile(path) as package:
        names = set(package.namelist())
        assert len([n for n in names if n.startswith("ppt/slides/slide") and n.endswith(".xml")]) == expected_pages
        for name in names:
            if name.endswith((".xml", ".rels")):
                tree = ET.fromstring(package.read(name))
                if name.endswith(".rels"):
                    base = posixpath.dirname(posixpath.dirname(name)) if "/_rels/" in name else ""
                    for rel in tree:
                        if rel.get("TargetMode") == "External":
                            continue
                        target = rel.get("Target")
                        absolute = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(base, target))
                        assert absolute in names, (name, absolute)
        assert package.read("ppt/media/backward_walk.gif") == gif_bytes
        presentation = ET.fromstring(package.read("ppt/presentation.xml"))
        assert presentation.find("p:notesMasterIdLst/p:notesMasterId", NS) is not None
        core_xml = package.read("docProps/core.xml")
        assert b'xmlns:dcterms="http://purl.org/dc/terms/"' in core_xml
        for number in range(1, expected_pages+1):
            root = ET.fromstring(package.read(f"ppt/slides/slide{number}.xml"))
            pictures = root.findall(".//p:pic", NS)
            assert len(pictures) == (2 if number == animation_page else 1)
        with Image.open(io.BytesIO(gif_bytes)) as gif:
            assert gif.n_frames > 1 and gif.info.get("loop") == 0
            return {"pages": expected_pages, "animation_page": animation_page,
                    "gif_frames": gif.n_frames, "gif_loop": "infinite",
                    "embedded_gif_unchanged": True, "all_internal_relationships_resolve": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=REPORT / "slides.pdf")
    parser.add_argument("--gif", type=Path, default=REPORT / "figures/backward_walk_animation.gif")
    parser.add_argument("--output", type=Path, default=REPORT / "slides.pptx")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    work = REPORT / ".build-pptx"
    work.mkdir(exist_ok=True)
    pdf_data = args.pdf.read_bytes()
    pdf_hash = hashlib.sha256(pdf_data).hexdigest()
    gif_bytes = args.gif.read_bytes()
    with Image.open(io.BytesIO(gif_bytes)) as gif:
        assert gif.n_frames > 1 and gif.info.get("loop") == 0
        gif_w, gif_h = gif.size
    doc = pymupdf.open(stream=pdf_data, filetype="pdf")
    animation_index, box = find_placeholder(doc)
    page_w, page_h = doc[0].rect.width, doc[0].rect.height
    assert all(abs(page.rect.width-page_w) < .01 and abs(page.rect.height-page_h) < .01 for page in doc)
    width = 12192000
    height = round(width * page_h/page_w)
    sx, sy = width/page_w, height/page_h
    sections = []
    for i, page in enumerate(doc, 1):
        # PDF math fonts may extract XML-forbidden control characters.
        notes = "".join(ch for ch in page.get_text()
                        if ord(ch) >= 32 or ch in "\n\r\t").replace(":::", "")
        sections.append(f"## Page {i}\n\nPage image\n\n::: notes\nSource: slides.pdf, page {i}.\n\n" + notes + "\n:::")
    source = work / "pdf-pages.md"
    source.write_text("\n\n".join(sections), encoding="utf-8")
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise RuntimeError("Pandoc is required to create the presentation package")
    base = work / "base.pptx"
    subprocess.run([pandoc, str(source), "-t", "pptx", "--slide-level=2", "-o", str(base)], check=True)
    with zipfile.ZipFile(base) as package:
        parts = {name: package.read(name) for name in package.namelist()}
    manifest = []
    for i, page in enumerate(doc, 1):
        pixmap = page.get_pixmap(dpi=args.dpi, alpha=False)
        png = pixmap.tobytes("png")
        (work / f"page-{i:02d}.png").write_bytes(png)
        media = f"pdf-page-{i:02d}.png"
        parts[f"ppt/media/{media}"] = png
        filename = f"ppt/slides/slide{i}.xml"
        root = ET.fromstring(parts[filename])
        root.set("showMasterSp", "0")
        tree = root.find("p:cSld/p:spTree", NS)
        for node in list(tree)[2:]:
            tree.remove(node)
        tree.append(picture(2, f"PDF page {i} at {args.dpi} dpi", "rIdPdfPage", 0, 0, width, height))
        relpath = f"ppt/slides/_rels/slide{i}.xml.rels"
        rels = ET.fromstring(parts[relpath])
        ET.SubElement(rels, f"{{{REL}}}Relationship", Id="rIdPdfPage", Type=NS["r"]+"/image", Target="../media/"+media)
        if i == animation_index+1:
            padding = .7
            cover = pymupdf.Rect(box.x0-padding, box.y0-padding, box.x1+padding, box.y1+padding)
            tree.append(white_mask(3, round(cover.x0*sx), round(cover.y0*sy), round(cover.width*sx), round(cover.height*sy)))
            fit = min(box.width/gif_w, box.height/gif_h)
            gw, gh = gif_w*fit, gif_h*fit
            gx, gy = box.x0+(box.width-gw)/2, box.y0+(box.height-gh)/2
            tree.append(picture(4, "反向随机游走动画（放映时循环播放）", "rIdWalkAnimation", round(gx*sx), round(gy*sy), round(gw*sx), round(gh*sy)))
            parts["ppt/media/backward_walk.gif"] = gif_bytes
            ET.SubElement(rels, f"{{{REL}}}Relationship", Id="rIdWalkAnimation", Type=NS["r"]+"/image", Target="../media/backward_walk.gif")
        parts[filename] = xml_bytes(root)
        parts[relpath] = xml_bytes(rels)
        manifest.append({"page": i, "pixel_width": pixmap.width, "pixel_height": pixmap.height,
                         "png_sha256": hashlib.sha256(png).hexdigest()})
    presentation = ET.fromstring(parts["ppt/presentation.xml"])
    # Some Pandoc versions emit NotesMasterId, which PowerPoint must repair.
    for node in presentation.findall("p:notesMasterIdLst/p:NotesMasterId", NS):
        node.tag = f"{{{NS['p']}}}notesMasterId"
    size = presentation.find("p:sldSz", NS)
    size.set("cx", str(width)); size.set("cy", str(height)); size.set("type", "custom")
    parts["ppt/presentation.xml"] = xml_bytes(presentation)
    content_types = ET.fromstring(parts["[Content_Types].xml"])
    for extension, mimetype in [("png", "image/png"), ("gif", "image/gif")]:
        if not any(item.get("Extension") == extension for item in content_types):
            ET.SubElement(content_types, f"{{{CT}}}Default", Extension=extension, ContentType=mimetype)
    parts["[Content_Types].xml"] = xml_bytes(content_types)
    if "docProps/core.xml" in parts:
        core = ET.fromstring(parts["docProps/core.xml"])
        for node in core:
            if node.tag.endswith(("creator", "lastModifiedBy")):
                node.text = ""
        parts["docProps/core.xml"] = xml_bytes(core)
    candidate = work / "slides-with-animation.pptx"
    with zipfile.ZipFile(candidate, "w", zipfile.ZIP_DEFLATED) as package:
        for name, data in parts.items():
            package.writestr(name, data)
    checks = validate_package(candidate, len(doc), animation_index+1, gif_bytes)
    assert hashlib.sha256(args.pdf.read_bytes()).hexdigest() == pdf_hash, "Source PDF changed during conversion"
    if args.output.exists():
        backup = REPORT / "backups" / ("slides-before-pptx-update-"+datetime.now().strftime("%Y%m%d-%H%M%S")+".pptx")
        backup.parent.mkdir(exist_ok=True)
        shutil.copyfile(args.output, backup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate, args.output)
    checks.update(source_pdf=str(args.pdf.resolve()), source_pdf_sha256=pdf_hash,
                  output=str(args.output.resolve()), dpi=args.dpi, slide_size_emu=[width, height],
                  animation_box_pdf_points=list(box), page_images=manifest,
                  static_content="Rasterized PDF; editable source remains slides.tex")
    (work / "pptx-validation.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in checks.items() if k != "page_images"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
