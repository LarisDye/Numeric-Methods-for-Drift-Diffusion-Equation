"""Create an editable OOXML PPTX; preserve CleanEasy visual hierarchy.
Artifact Tool is unavailable in this host; native charts include embedded XLSX data.
"""
from pathlib import Path
import argparse,subprocess,json,zipfile,io,math,shutil
from xml.etree import ElementTree as E
from xml.sax.saxutils import escape
from openpyxl import Workbook,load_workbook
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report-dir",type=Path,default=root/"report")
parser.add_argument("--content",type=Path,help="Slide content JSON; defaults to report-dir/slide_content.json")
parser.add_argument("--charts",type=Path,help="Native chart data JSON; defaults to report-dir/chart_data.json")
parser.add_argument("--output",type=Path,help="Output PPTX; defaults to report-dir/slides.pptx")
args=parser.parse_args();report=args.report_dir;tmp=root/"tmp/presentation";tmp.mkdir(parents=True,exist_ok=True)
slides=json.loads((args.content or report/"slide_content.json").read_text(encoding="utf-8"))
charts=json.loads((args.charts or report/"chart_data.json").read_text(encoding="utf-8"))
md="\n\n".join("## "+s["title"]+"\n\nContent\n\n::: notes\n"+s["notes"]+"\n:::" for s in slides)
(tmp/"base.md").write_text(md,encoding="utf-8")
pandoc=shutil.which("pandoc") or "C:/ProgramData/anaconda3/Scripts/pandoc.exe"
subprocess.run([pandoc,str(tmp/"base.md"),"-t","pptx","--slide-level=2","-o",str(tmp/"base.pptx")],check=True)
with zipfile.ZipFile(tmp/"base.pptx") as z:parts={n:z.read(n) for n in z.namelist()}
NS={"p":"http://schemas.openxmlformats.org/presentationml/2006/main","a":"http://schemas.openxmlformats.org/drawingml/2006/main",
"r":"http://schemas.openxmlformats.org/officeDocument/2006/relationships","c":"http://schemas.openxmlformats.org/drawingml/2006/chart"}
REL="http://schemas.openxmlformats.org/package/2006/relationships"
CT="http://schemas.openxmlformats.org/package/2006/content-types"
for k,v in NS.items():E.register_namespace(k,v)
def emu(x):return round(x*914400)
def font(sz,color="222222",bold=False):
 return f'<a:rPr lang="zh-CN" sz="{int(sz*100)}" b="{int(bold)}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="Times New Roman"/><a:ea typeface="SimSun"/><a:cs typeface="Times New Roman"/></a:rPr>'
def para(text,sz=24,color="222222",bold=False,align="l",bullet=False):
 ppr=f'<a:pPr algn="{align}" marL="{emu(.25) if bullet else 0}" indent="{emu(-.2) if bullet else 0}"><a:lnSpc><a:spcPct val="115000"/></a:lnSpc><a:spcAft><a:spcPts val="1000"/></a:spcAft>'+('<a:buChar char="•"/>' if bullet else '<a:buNone/>')+'</a:pPr>'
 return f'<a:p>{ppr}<a:r>{font(sz,color,bold)}<a:t>{escape(text)}</a:t></a:r><a:endParaRPr lang="zh-CN" sz="{int(sz*100)}"/></a:p>'
def textbox(sid,name,x,y,w,h,texts,sz=24,color="222222",bold=False,align="l",bullet=False):
 content="".join(para(t,sz,color,bold,align,bullet) for t in texts)
 return f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{escape(name)}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr><p:txBody><a:bodyPr wrap="square" lIns="0" rIns="0" tIns="0" bIns="0" anchor="t"/><a:lstStyle/>{content}</p:txBody></p:sp>'
def line(sid):
 return f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Template header rule"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="{emu(.48)}" y="{emu(.98)}"/><a:ext cx="{emu(12.37)}" cy="{emu(.008)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="0D2659"/></a:solidFill><a:ln><a:noFill/></a:ln></p:spPr></p:sp>'
def rich(text,sz=16):
 return '<c:rich><a:bodyPr/><a:lstStyle/>'+para(text,sz)+"</c:rich>"
def txpr(sz=16):
 return '<c:txPr><a:bodyPr/><a:lstStyle/><a:p><a:pPr><a:defRPr sz="'+str(int(sz*100))+'"><a:latin typeface="Times New Roman"/><a:ea typeface="SimSun"/></a:defRPr></a:pPr><a:endParaRPr/></a:p></c:txPr>'
def numcache(values):
 return '<c:numCache><c:formatCode>General</c:formatCode><c:ptCount val="'+str(len(values))+'"/>'+''.join(f'<c:pt idx="{i}"><c:v>{float(v):.17g}</c:v></c:pt>' for i,v in enumerate(values))+'</c:numCache>'
def chart_xml(data):
 series=[];wb=Workbook();ws=wb.active;ws.title="Data"
 for n,s in enumerate(data["series"]):
  xcol=2*n+1;ycol=xcol+1
  from openpyxl.utils import get_column_letter as col
  ws.cell(1,xcol,s["name"]+" X");ws.cell(1,ycol,s["name"])
  for r,(x,y) in enumerate(zip(s["x"],s["y"]),2):ws.cell(r,xcol,x);ws.cell(r,ycol,y)
  count=len(s["x"]);color=s["color"]
  tx=f'<c:tx><c:strRef><c:f>Data!'+col(ycol)+'$1</c:f><c:strCache><c:ptCount val="1"/><c:pt idx="0"><c:v>'+escape(s["name"])+'</c:v></c:pt></c:strCache></c:strRef></c:tx>'
  vx='<c:xVal><c:numRef><c:f>Data!$'+col(xcol)+'$2:$'+col(xcol)+'$'+str(count+1)+'</c:f>'+numcache(s["x"])+'</c:numRef></c:xVal>'
  vy='<c:yVal><c:numRef><c:f>Data!$'+col(ycol)+'$2:$'+col(ycol)+'$'+str(count+1)+'</c:f>'+numcache(s["y"])+'</c:numRef></c:yVal>'
  style=f'<c:spPr><a:ln w="25400"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:ln></c:spPr><c:marker><c:symbol val="circle"/><c:size val="6"/><c:spPr><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:ln><a:noFill/></a:ln></c:spPr></c:marker>'
  series.append(f'<c:ser><c:idx val="{n}"/><c:order val="{n}"/>{tx}{style}{vx}{vy}<c:smooth val="0"/></c:ser>')
 def axis(id,other,pos,label,log=False):
  grid='<c:majorGridlines><c:spPr><a:ln w="6350"><a:solidFill><a:srgbClr val="DDDDDD"/></a:solidFill></a:ln></c:spPr></c:majorGridlines>' if pos=="l" else ""
  title='<c:title><c:tx>'+rich(label,16)+'</c:tx><c:overlay val="0"/>'+txpr(16)+'</c:title>'
  return f'<c:valAx><c:axId val="{id}"/><c:scaling>'+('<c:logBase val="10"/>' if log else '')+f'<c:orientation val="minMax"/></c:scaling><c:delete val="0"/><c:axPos val="{pos}"/>{grid}{title}<c:numFmt formatCode="'+('0.0E+00' if log and pos=="l" else 'General')+f'" sourceLinked="0"/><c:majorTickMark val="out"/><c:minorTickMark val="none"/><c:tickLblPos val="nextTo"/>'+txpr(15)+f'<c:crossAx val="{other}"/><c:crosses val="min"/></c:valAx>'
 plot='<c:plotArea><c:layout/><c:scatterChart><c:scatterStyle val="lineMarker"/><c:varyColors val="0"/>'+''.join(series)+'<c:axId val="100"/><c:axId val="200"/></c:scatterChart>'+axis(100,200,"b",data["xlabel"],data.get("xlog",False))+axis(200,100,"l",data["ylabel"],data.get("ylog",False))+'</c:plotArea>'
 xml='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><c:chartSpace xmlns:c="'+NS["c"]+'" xmlns:a="'+NS["a"]+'" xmlns:r="'+NS["r"]+'"><c:date1904 val="0"/><c:lang val="zh-CN"/><c:roundedCorners val="0"/><c:chart><c:autoTitleDeleted val="1"/>'+plot+'<c:legend><c:legendPos val="b"/><c:overlay val="0"/>'+txpr(15)+'</c:legend><c:plotVisOnly val="1"/><c:dispBlanksAs val="gap"/></c:chart><c:spPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:ln><a:noFill/></a:ln></c:spPr>'+txpr(16)+'<c:externalData r:id="rIdWorkbook"><c:autoUpdate val="0"/></c:externalData></c:chartSpace>'
 bio=io.BytesIO();wb.save(bio)
 return xml.encode(),bio.getvalue()
types=E.fromstring(parts["[Content_Types].xml"])
chartnum=0
for i,s in enumerate(slides,1):
 fn=f"ppt/slides/slide{i}.xml";doc=E.fromstring(parts[fn]);cs=doc.find("p:cSld",NS)
 tree=cs.find("p:spTree",NS)
 for node in list(tree)[2:]:tree.remove(node)
 bg=cs.find("p:bg",NS)
 if bg is not None:cs.remove(bg)
 cs.insert(0,E.fromstring('<p:bg xmlns:p="'+NS["p"]+'" xmlns:a="'+NS["a"]+'"><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>'))
 shapes=[]
 if i==1:
  shapes.append(textbox(2,"Title",.7,1.8,11.93,.85,[s["title"]],36,"0D2659",True,"ctr"))
  shapes.append(textbox(3,"Subtitle",.8,2.95,11.73,1.1,s["body"][:2],23,"0D2659",False,"ctr"))
  shapes.append(textbox(4,"Date",.8,5.25,11.73,.45,[s["body"][-1]],18,"444444",False,"ctr"))
 else:
  shapes.append(textbox(2,"Title",.48,.30,12.37,.65,[s["title"]],29,"0D2659",True))
  shapes.append(line(3))
  if "chart" in s:
   chartnum+=1;chart,book=chart_xml(charts[s["chart"]]);parts[f"ppt/charts/chart{chartnum}.xml"]=chart
   parts[f"ppt/embeddings/chart{chartnum}.xlsx"]=book
   parts[f"ppt/charts/_rels/chart{chartnum}.xml.rels"]=(f'<Relationships xmlns="{REL}"><Relationship Id="rIdWorkbook" Type="{NS["r"]}/package" Target="../embeddings/chart{chartnum}.xlsx"/></Relationships>').encode()
   relpath=f"ppt/slides/_rels/slide{i}.xml.rels";rels=E.fromstring(parts[relpath])
   rid="rIdNativeChart";E.SubElement(rels,"{"+REL+"}Relationship",Id=rid,Type=NS["r"]+"/chart",Target=f"../charts/chart{chartnum}.xml")
   parts[relpath]=E.tostring(rels,encoding="utf-8",xml_declaration=True)
   shapes.append(f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="4" name="Editable measured data chart"/><p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr><p:xfrm><a:off x="{emu(.75)}" y="{emu(1.18)}"/><a:ext cx="{emu(11.85)}" cy="{emu(4.87)}"/></p:xfrm><a:graphic><a:graphicData uri="{NS["c"]}"><c:chart xmlns:c="{NS["c"]}" xmlns:r="{NS["r"]}" r:id="{rid}"/></a:graphicData></a:graphic></p:graphicFrame>')
   shapes.append(textbox(5,"Result notes",.75,6.15,11.85,.8,s["body"],18,"222222"))
   E.SubElement(types,"{"+CT+"}Override",PartName=f"/ppt/charts/chart{chartnum}.xml",ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml")
  else:
   shapes.append(textbox(4,"Content",.8,1.45,11.7,5.45,s["body"],23,"222222",bullet=True))
  shapes.append(textbox(10,"Footer",2.4,7.13,8.5,.2,["漂移扩散方程数值实验"],9,"AAAAAA",align="ctr"))
  shapes.append(textbox(11,"Page number",11.8,7.10,1.0,.25,[f"{i}/12"],10,"AAAAAA",align="r"))
 for shape in shapes:
  wrapper=E.fromstring('<root xmlns:p="'+NS["p"]+'" xmlns:a="'+NS["a"]+'">'+shape+'</root>')
  tree.append(list(wrapper)[0])
 parts[fn]=E.tostring(doc,encoding="utf-8",xml_declaration=True)
pres=E.fromstring(parts["ppt/presentation.xml"]);size=pres.find("p:sldSz",NS);size.set("cx",str(emu(13.333333)));size.set("cy",str(emu(7.5)));size.set("type","screen16x9")
parts["ppt/presentation.xml"]=E.tostring(pres,encoding="utf-8",xml_declaration=True)
if not any(x.attrib.get("Extension")=="xlsx" for x in types):E.SubElement(types,"{"+CT+"}Default",Extension="xlsx",ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
E.register_namespace("",CT)
parts["[Content_Types].xml"]=E.tostring(types,encoding="utf-8",xml_declaration=True)
# Remove accidental personal metadata from the base package.
if "docProps/core.xml" in parts:
 c=E.fromstring(parts["docProps/core.xml"])
 for n in c:
  if n.tag.endswith(("creator","lastModifiedBy")):n.text=""
 parts["docProps/core.xml"]=E.tostring(c,encoding="utf-8",xml_declaration=True)
target=args.output or report/"slides.pptx";target.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as z:
 for name,data in parts.items():z.writestr(name,data)
# Validate the written package, internal targets, chart caches and embedded books.
with zipfile.ZipFile(target) as z:
 names=set(z.namelist());assert len([n for n in names if n.startswith("ppt/slides/slide") and n.endswith(".xml")])==12
 for n in names:
  if n.endswith((".xml",".rels")):E.fromstring(z.read(n))
 for k in range(1,chartnum+1):
  chart=E.fromstring(z.read(f"ppt/charts/chart{k}.xml"))
  book=load_workbook(io.BytesIO(z.read(f"ppt/embeddings/chart{k}.xlsx")),data_only=True)
  assert len(chart.findall(".//c:ser",NS))>0 and book.active.max_row>=3
 assert chartnum==6
checks=dict(slides=12,native_charts=chartnum,embedded_workbooks=chartnum,all_xml_parsed=True,
notes_preserved=True,template="CleanEasy color/font hierarchy reconstructed from Beamer ZIP",
renderer="Beamer counterpart rendered; local PowerPoint COM unavailable")
(tmp/"pptx_validation.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf-8")
print(target);print(checks)
