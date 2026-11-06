exec(open("helpers_part.py").read())
from pptx.oxml.ns import qn
GRN=RGBColor(0x1E,0x5E,0x3B)
def bullet(par,level):
    pPr=par._p.get_or_add_pPr()
    mar=[0,342900,685800][level]
    if level==0:
        pPr.set("marL","0"); pPr.set("indent","0")
        etree.SubElement(pPr,qn("a:buNone"))
    else:
        pPr.set("marL",str(mar)); pPr.set("indent","-228600")
        bf=etree.SubElement(pPr,qn("a:buFont")); bf.set("typeface","Arial")
        bc=etree.SubElement(pPr,qn("a:buChar")); bc.set("char","•" if level==1 else "–")
def blocks(tf,items,size=17,sub=16,gap=12):
    """items: list of (head, members, [subbullets]); subbullet may be (lead, text)"""
    first=True
    for head,members,subs in items:
        par=tf.paragraphs[0] if first else tf.add_paragraph(); first=False
        bullet(par,0); par.space_before=Pt(0 if par is tf.paragraphs[0] else gap); par.space_after=Pt(2)
        run(par,head,size,True,RED)
        if members: run(par,"  "+members,size,False,GR)
        for sb in subs:
            par=tf.add_paragraph(); bullet(par,1); par.space_after=Pt(1)
            if isinstance(sb,tuple):
                run(par,sb[0]+" ",sub,True); run(par,sb[1],sub)
            else: run(par,sb,sub)
