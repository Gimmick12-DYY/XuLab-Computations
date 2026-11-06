import sys, copy
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree
src, dst = sys.argv[1], sys.argv[2]
p=Presentation(src)
RED=RGBColor(0x8C,0x1C,0x13); DK=RGBColor(0x26,0x26,0x26); GR=RGBColor(0x59,0x59,0x59)
layout=[l for l in p.slide_layouts if l.name=="Title and Content"][0]
FONT="Cambria"

def new_slide():
    s=p.slides.add_slide(layout)
    for ph in list(s.placeholders): ph._element.getparent().remove(ph._element)
    return s
def tb(s,x,y,w,h,name):
    t=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); t.name=name
    tf=t.text_frame; tf.word_wrap=True; return t,tf
def run(par,text,size,bold=False,color=None,italic=False):
    r=par.add_run(); r.text=text; f=r.font; f.name=FONT; f.size=Pt(size); f.bold=bold; f.italic=italic
    if color is not None: f.color.rgb=color
    return r
def title(s,text,sub=None):
    t,tf=tb(s,0.17,0.12,12.9,0.57,"Title"); run(tf.paragraphs[0],text,28,True)
    if sub:
        t,tf=tb(s,0.33,0.68,12.69,0.4,"Subtitle"); run(tf.paragraphs[0],sub,15,False,GR)
def paras(tf,items,size,space=6):
    first=True
    for lead,body in items:
        par=tf.paragraphs[0] if first else tf.add_paragraph(); first=False
        par.space_after=Pt(space)
        if lead: run(par,lead+" ",size,True)
        run(par,body,size)
def foot(s,text,y=6.95,h=0.45):
    t,tf=tb(s,0.33,y,12.7,h,"Footnote"); run(tf.paragraphs[0],text,10.5,False,GR)
def cell_border(cell,color="BFBFBF",w=6350):
    tcPr=cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL","a:lnR","a:lnT","a:lnB"):
        for e in tcPr.findall(qn(tag)): tcPr.remove(e)
        ln=etree.SubElement(tcPr,qn(tag),w=str(w if tag in("a:lnT","a:lnB") else 0))
        if tag in ("a:lnT","a:lnB"):
            sf=etree.SubElement(ln,qn("a:solidFill")); etree.SubElement(sf,qn("a:srgbClr"),val=color)
        else:
            etree.SubElement(ln,qn("a:noFill"))
    # schema order: lnL lnR lnT lnB ... then fill; move fills after borders
    for f in tcPr.findall(qn("a:solidFill")): tcPr.remove(f); tcPr.append(f)
def table(s,x,y,w,colw,rows,size=11,name="Table",hdr_fill="EDEDED",rowh=0.36,styler=None):
    n,m=len(rows),len(rows[0])
    gt=s.shapes.add_table(n,m,Inches(x),Inches(y),Inches(w),Inches(rowh*n)); gt.name=name
    t=gt.table; t.first_row=True; t.horz_banding=False
    for j,cw in enumerate(colw): t.columns[j].width=Inches(cw)
    for i,row in enumerate(rows):
        t.rows[i].height=Inches(rowh)
        for j,val in enumerate(row):
            c=t.cell(i,j); c.margin_left=c.margin_right=Inches(0.06); c.margin_top=c.margin_bottom=Inches(0.02)
            c.vertical_anchor=MSO_ANCHOR.MIDDLE
            c.fill.solid(); c.fill.fore_color.rgb=RGBColor.from_string(hdr_fill if i==0 else "FFFFFF")
            tf=c.text_frame; tf.word_wrap=True; par=tf.paragraphs[0]
            bold=(i==0); color=DK
            if styler and i>0:
                b2,c2=styler(i,j,val); bold=bold or b2; color=c2 or color
            run(par,str(val),size,bold,color)
            cell_border(c)
    return gt
def card(s,x,y,w,h,name):
    sh=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h)); sh.name=name
    sh.fill.solid(); sh.fill.fore_color.rgb=RGBColor(0xF7,0xF6,0xF4); sh.line.color.rgb=RGBColor(0xD9,0xD9,0xD9); sh.line.width=Pt(0.75)
    sh.shadow.inherit=False
    tf=sh.text_frame; tf.word_wrap=True; tf.vertical_anchor=MSO_ANCHOR.TOP
    for a in ("margin_left","margin_right"): setattr(tf,a,Inches(0.15))
    tf.margin_top=Inches(0.12)
    return tf
from pptx.enum.text import PP_ALIGN

