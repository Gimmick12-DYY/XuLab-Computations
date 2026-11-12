"""Fig 3a illustration (v3, journal style): hand-built SVG, Arial, Palette C + Fig 3 element colours.
Pure drawing, no data. Output: a_coordinated_binding_schematic.svg (183 x 71 mm); PNG/PDF rendered separately."""
import numpy as np
U=4.0                      # svg units per mm
W,H=183*U,71*U
TEAL="#2C7A6B"; PROM="#3B6FB6"; ENH="#E8821E"; GREY="#8C8C8C"; INK="#1A1A1A"; DARK="#4D4D4D"
LIGHT="#BDBDBD"; HIC="#B83A2E"; FONT="Arial, Helvetica, sans-serif"
pt=lambda p: p*0.3528*U    # font pt -> units
out=[]
def add(s): out.append(s)
def text(x,y,s,size=6,anchor="start",fill=INK,weight="normal",style="normal"):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{pt(size):.2f}" font-weight="{weight}" font-style="{style}" fill="{fill}" text-anchor="{anchor}">{s}</text>')
def sphere(x,y,r,grad,stroke="#FFFFFF",sw=0.8):
    add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="url(#{grad})" stroke="{stroke}" stroke-width="{sw}"/>')
def nucleosome(x,y,r=5.2):
    add(f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{r:.1f}" ry="{r*0.82:.1f}" fill="url(#gNuc)" stroke="#A9A3B3" stroke-width="0.6"/>')
    add(f'<path d="M{x-r*0.85:.1f},{y+r*0.25:.1f} Q{x:.1f},{y-r*0.55:.1f} {x+r*0.85:.1f},{y+r*0.25:.1f}" fill="none" stroke="#8E8799" stroke-width="0.7"/>')
def tf(x,y,s=1.0):
    # two-lobed protein glyph sitting on DNA at (x,y)
    add(f'<g transform="translate({x:.1f},{y:.1f}) scale({s})" filter="url(#shadow)">'
        f'<path d="M-7,0 C-8,-6 -5,-11 0,-11 C5,-11 8,-6 7,0 C4,-2.2 -4,-2.2 -7,0 Z" fill="url(#gTF)" stroke="#FFFFFF" stroke-width="0.7"/>'
        f'<ellipse cx="-2.4" cy="-7.4" rx="2.2" ry="1.3" fill="#FFFFFF" opacity="0.35"/></g>')
def element(x,y,kind,w=16,h=4.2,ah=None):
    ah=ah if ah is not None else (14 if w>=16 else 6)
    g="gProm" if kind=="P" else ("gEnh" if kind=="E" else "gOth")
    add(f'<rect x="{x-w/2:.1f}" y="{y-h/2:.1f}" width="{w}" height="{h}" rx="1.6" fill="url(#{g})"/>')
    if kind=="P":   # bent TSS arrow
        add(f'<path d="M{x+w/2-2:.1f},{y-h/2:.1f} v-{ah} h6" fill="none" stroke="{PROM}" stroke-width="0.9"/>'
            f'<path d="M{x+w/2+4:.1f},{y-h/2-ah-2.2:.1f} l3,2.2 l-3,2.2 z" fill="{PROM}"/>')
def dna(x0,x1,y):
    xs=np.linspace(x0,x1,int((x1-x0)/1.5)); a=1.3; k=2*np.pi/9
    for ph,c in [(0,"#9B9B9B"),(np.pi,"#C4C4C4")]:
        d="M"+" L".join(f"{x:.1f},{y+a*np.sin(k*(x-x0)+ph):.1f}" for x in xs)
        add(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="0.9"/>')
def arcpath(x0,x1,y,h,col,sw=1.1,dash=None,up=True):
    s=-1 if up else 1; dd=f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<path d="M{x0:.1f},{y:.1f} C{x0:.1f},{y+s*h*1.33:.1f} {x1:.1f},{y+s*h*1.33:.1f} {x1:.1f},{y:.1f}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linecap="round"{dd}/>')
def chevron(x,y):
    add(f'<path d="M{x:.1f},{y-7:.1f} l7,7 l-7,7" fill="none" stroke="{LIGHT}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>')
def header(x,y,s):
    text(x,y,s,size=6.5,weight="bold")
def crspline(P,n=20):
    P=np.array(P,float); P=np.vstack([P[0],P,P[-1]]); o=[]
    for k in range(1,len(P)-2):
        p0,p1,p2,p3=P[k-1],P[k],P[k+1],P[k+2]
        for t in np.linspace(0,1,n,endpoint=False):
            o.append(0.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t**2+(-p0+3*p1-3*p2+p3)*t**3))
    o.append(P[-2]); return np.array(o)

# ---------- defs ----------
def rg(id,c0,c1,c2,fx=.35,fy=.3):
    return (f'<radialGradient id="{id}" cx="0.45" cy="0.4" r="0.65" fx="{fx}" fy="{fy}">'
            f'<stop offset="0" stop-color="{c0}"/><stop offset="0.55" stop-color="{c1}"/><stop offset="1" stop-color="{c2}"/></radialGradient>')
def lg(id,c0,c1):
    return (f'<linearGradient id="{id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{c0}"/><stop offset="1" stop-color="{c1}"/></linearGradient>')
defs=[rg("gTF","#7FC1B3",TEAL,"#1D5A4E"), rg("sP","#9DBBE6",PROM,"#244C86"), rg("sE","#F6C08A",ENH,"#B35F0E"),
      rg("sG","#D8D8D8",GREY,"#5E5E5E"), rg("gNuc","#FFFFFF","#E7E3EC","#CBC4D4",fx=.4,fy=.35),
      lg("gProm","#6F96D0",PROM), lg("gEnh","#F2A55A",ENH), lg("gOth","#B5B5B5",GREY),
      '<radialGradient id="gCell" cx="0.5" cy="0.45" r="0.6"><stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#EAF2F0"/></radialGradient>',
      '<radialGradient id="gHalo" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stop-color="#FCE9A8" stop-opacity="0.95"/><stop offset="0.7" stop-color="#FCE9A8" stop-opacity="0.55"/><stop offset="1" stop-color="#FCE9A8" stop-opacity="0"/></radialGradient>',
      lg("gCellT","#3F8E7E",TEAL),
      '<filter id="shadow" x="-30%" y="-30%" width="160%" height="160%"><feDropShadow dx="0.4" dy="0.6" stdDeviation="0.6" flood-color="#000000" flood-opacity="0.22"/></filter>']

# ---------- panel 1: single cells ----------
header(8,16,"Single-cell binding of one TF")
B=[[1,1,1],[0,1,0],[1,1,1]]
cy=[48,104,160]
for i,y in enumerate(cy):
    add(f'<rect x="8" y="{y-22}" width="150" height="44" rx="22" fill="url(#gCell)" stroke="#BFD3CE" stroke-width="0.9"/>')
    text(20,y-11,f"Cell {i+1}" if i<2 else "Cell n",size=5,fill=DARK)
    dna(26,146,y+4)
    for xn in [34,66,106,140]: nucleosome(xn,y+4)
    for (xe,kind),b in zip([(50,"P"),(86,"E"),(122,"E")],B[i]):
        element(xe,y+4,kind)
        if b: tf(xe,y+1.8)
for k,yy in enumerate([188,193,198]): add(f'<circle cx="83" cy="{yy}" r="1.1" fill="{LIGHT}"/>')
for xe,lab,c in [(50,"P",PROM),(86,"E1",ENH),(122,"E2",ENH)]: text(xe,214,lab,size=6,anchor="middle",fill=c,weight="bold")
chevron(166,104)

# ---------- panel 2: coordinated binding matrix + pair scores ----------
x2=186; header(x2,16,"Coordinated binding across cells")
M=np.array([[1,1,1,0],[0,1,0,1],[1,1,1,0],[0,0,0,1],[1,1,1,0],[1,0,0,1],[0,0,0,0],[1,1,1,1]])
cs=13; mx,my=x2+22,40
for j,(lab,c) in enumerate([("P",PROM),("E1",ENH),("E2",ENH),("X",GREY)]):
    text(mx+j*cs+cs/2-0.5,my-4,lab,size=6,anchor="middle",fill=c,weight="bold")
for i in range(M.shape[0]):
    for j in range(4):
        f="url(#gCellT)" if M[i,j] else "#EFEFEF"
        add(f'<rect x="{mx+j*cs:.1f}" y="{my+i*cs:.1f}" width="{cs-2}" height="{cs-2}" rx="2.2" fill="{f}"/>')
add(f'<text x="{mx-8}" y="{my+M.shape[0]*cs/2}" font-family="{FONT}" font-size="{pt(5):.2f}" fill="{DARK}" text-anchor="middle" transform="rotate(-90 {mx-8} {my+M.shape[0]*cs/2})">cells</text>')
# bracket over P,E1,E2 columns -> correlated
bx0,bx1=mx,mx+3*cs-2; by=my+M.shape[0]*cs+5
add(f'<path d="M{bx0},{by} v4 h{bx1-bx0} v-4" fill="none" stroke="{DARK}" stroke-width="0.8"/>')
text((bx0+bx1)/2,by+13,"bound in the same cells",size=5,anchor="middle",fill=DARK)
# pair score triangle
tx,ty,t=mx+4,by+44,13
S={(0,1):.92,(0,2):.85,(1,2):.9,(0,3):.08,(1,3):.15,(2,3):.12}
def blue(v):
    a=np.array([0xF4,0xF6,0xFA]); b=np.array([0x1F,0x3F,0x8F]); c=(a+(b-a)*v).astype(int); return "#%02X%02X%02X"%tuple(c)
for (i,j),v in S.items():
    cx=tx+(i+j)*t/2; cyy=ty-(j-i)*t/2
    add(f'<path d="M{cx:.1f},{cyy-t/2+0.6:.1f} L{cx+t/2-0.6:.1f},{cyy:.1f} L{cx:.1f},{cyy+t/2-0.6:.1f} L{cx-t/2+0.6:.1f},{cyy:.1f} Z" fill="{blue(v)}"/>')
for k,c in enumerate(["gProm","gEnh","gEnh","gOth"]):
    add(f'<rect x="{tx+k*t-5:.1f}" y="{ty+6:.1f}" width="10" height="3.6" rx="1.4" fill="url(#{c})"/>')
text(tx+3*t+12,ty-14,"pair score",size=5,fill=DARK); text(tx+3*t+12,ty-7,"(Cicero)",size=5,fill=DARK)
for k,v in enumerate([.1,.5,.9]): add(f'<rect x="{tx+3*t+12+k*7:.1f}" y="{ty-3:.1f}" width="7" height="4" fill="{blue(v)}"/>')
chevron(320,104)

# ---------- panel 3: higher-order module ----------
x3=338; header(x3,16,"Higher-order coordinated binding")
gy=78; ex=[(352,"P"),(392,"E"),(432,"E"),(470,"X")]
dna(342,486,gy)
for xn in [372,412,452]: nucleosome(xn,gy)
for x,k in ex: element(x,gy,k if k!="X" else "O",ah=7)
for a,b,h in [(0,1,12),(1,2,12),(0,2,22)]: arcpath(ex[a][0],ex[b][0],gy-4,h,INK,1.2)
arcpath(ex[2][0],ex[3][0],gy-4,10,LIGHT,0.9,dash="2.5,2")
text(486,gy-30,"&gt;25 kb",size=5,anchor="end",fill=DARK)
# network glyph
nx,ny=414,150
add(f'<circle cx="{nx}" cy="{ny}" r="40" fill="url(#gHalo)"/>')
N={"P":(nx-22,ny-12,"sP"),"E1":(nx+22,ny-12,"sE"),"E2":(nx,ny+20,"sE")}
for a,b in [("P","E1"),("E1","E2"),("P","E2")]:
    add(f'<line x1="{N[a][0]}" y1="{N[a][1]}" x2="{N[b][0]}" y2="{N[b][1]}" stroke="{INK}" stroke-width="1.6"/>')
add(f'<line x1="{N["E1"][0]}" y1="{N["E1"][1]}" x2="{nx+52}" y2="{ny-28}" stroke="{LIGHT}" stroke-width="1.1" stroke-dasharray="2.5,2"/>')
sphere(nx+52,ny-28,5.5,"sG")
for k,(x,y,g) in N.items():
    sphere(x,y,8.5,g); text(x,y+2.6,k,size=5.5,anchor="middle",fill="#FFFFFF",weight="bold")
text(nx,ny+46,"module: ≥3 elements,",size=5.5,anchor="middle")
text(nx,ny+54,"all pairs coordinated",size=5.5,anchor="middle")
chevron(500,104)

# ---------- panel 4: 3D contact + Hi-C ----------
x4=518; header(x4,16,"Tested against 3D contacts")
hx,hy=624,84
add(f'<circle cx="{hx}" cy="{hy}" r="34" fill="url(#gHalo)"/>')
def pol(a,r): a=np.deg2rad(a); return np.array([hx+r*np.cos(a), hy-r*np.sin(a)])
def cub(p0,p1,p2,p3,n=40):
    t=np.linspace(0,1,n)[:,None]; return (1-t)**3*p0+3*(1-t)**2*t*p1+3*(1-t)*t**2*p2+t**3*p3
ra=12.0; R=44
angs=[225,135,45,-45]
anc={"P":pol(225,ra),"E1":pol(135,ra),"E2":pol(45,ra),"H":pol(-45,ra)}
def petal2(a0,a1,n=90):
    th=np.linspace(a0,a1,n); f=np.sin(np.pi*(th-a0)/(a1-a0))**0.75
    r=ra+R*f; return np.c_[hx+r*np.cos(np.deg2rad(th)), hy-r*np.sin(np.deg2rad(th))]
tail_in=cub(np.array([526.,124.]),np.array([570.,124.]),pol(240,34),anc["P"])
tail_out=cub(anc["H"],pol(300,34),np.array([680.,124.]),np.array([724.,124.]))
fib=np.vstack([tail_in,petal2(225,135),petal2(135,45),petal2(45,-45),tail_out])
d="M"+" L".join(f"{x:.1f},{y:.1f}" for x,y in fib)
add(f'<path d="{d}" fill="none" stroke="#B9B9B9" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>')
P3={k:tuple(v) for k,v in anc.items() if k!="H"}
L=np.r_[0,np.cumsum(np.hypot(*np.diff(fib,axis=0).T))]
for sx in np.arange(10,L[-1],14):
    p=fib[min(np.searchsorted(L,sx),len(fib)-1)]
    if np.hypot(p[0]-hx,p[1]-hy)>24: nucleosome(p[0],p[1],r=4.2)
for (k,(x,y)),g in zip(P3.items(),["sP","sE","sE"]): sphere(x,y,5.4,g)
sphere(*anc["H"],3.4,"sG")
tf(hx,hy+5,s=0.72); tf(hx-4.5,hy-1.5,s=0.6); tf(hx+4.5,hy-1.5,s=0.6)
for k,ang,c in [("P",225,PROM),("E1",135,ENH),("E2",45,ENH)]:
    p=pol(ang,ra+12); add(f'<text x="{p[0]:.1f}" y="{p[1]+2.2:.1f}" font-family="{FONT}" font-size="{pt(5.5):.2f}" font-weight="bold" fill="{c}" text-anchor="middle" stroke="#FFFFFF" stroke-width="2" paint-order="stroke">{k}</text>')
# mini Hi-C triangle with loop pixel
n=12; t=7.6; ox,oy=584,210
rng=np.random.default_rng(1)
for i in range(n):
    for j in range(i,n):
        dist=j-i; v=np.exp(-dist/2.0)*0.62+rng.random()*0.05
        if (i,j) in [(1,9)]: v=0.95
        if (i,j) in [(1,8),(2,9),(0,9),(1,10)]: v=max(v,0.42)
        a=np.array([0xFF,0xF5,0xF0]); b=np.array([0xA5,0x0F,0x15]); c="#%02X%02X%02X"%tuple((a+(b-a)*min(v,1)).astype(int))
        cx=ox+(i+j)*t/2; cyy=oy-(j-i)*t/2
        add(f'<path d="M{cx:.1f},{cyy-t/2:.1f} L{cx+t/2:.1f},{cyy:.1f} L{cx:.1f},{cyy+t/2:.1f} L{cx-t/2:.1f},{cyy:.1f} Z" fill="{c}"/>')
lx,ly=ox+(1+9)*t/2,oy-8*t/2
add(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="6.2" fill="none" stroke="{INK}" stroke-width="0.9"/>')
text(lx+9,ly-6,"loop",size=5,fill=INK)
# arcs below triangle: module edge matched to loop
gy2=oy+12
add(f'<line x1="{ox-4}" y1="{gy2}" x2="{ox+(n-1)*t+4}" y2="{gy2}" stroke="{LIGHT}" stroke-width="1.6" stroke-linecap="round"/>')
for i,k in [(1,"P"),(5,"E"),(9,"E")]: element(ox+i*t,gy2,k,w=10,h=3.6)
arcpath(ox+1*t,ox+9*t,gy2+3,9,HIC,1.3,up=False)
text(ox+5*t,gy2+23,"module edge on a Hi-C loop",size=5,anchor="middle",fill=HIC)
text(ox-6,oy-58,"Hi-C contacts",size=5.5,fill=DARK,weight="bold")

# ---------- legend ----------
ly=262; lx=8
tf(lx+6,ly+3,s=0.8); text(lx+14,ly+2,"TF (HITAG)",size=5.5)
element(lx+68,ly+2,"P",w=12,h=3.8); text(lx+86,ly+2,"promoter",size=5.5)
element(lx+132,ly,"E",w=12,h=3.8); text(lx+142,ly+2,"enhancer",size=5.5)
nucleosome(lx+190,ly,r=4.2); text(lx+198,ly+2,"nucleosome",size=5.5)
add(f'<path d="M{lx+250},{ly+2} q8,-10 16,0" fill="none" stroke="{INK}" stroke-width="1.2"/>'); text(lx+271,ly+2,"coordinated binding edge",size=5.5)
add(f'<path d="M{lx+372},{ly-2} q8,10 16,0" fill="none" stroke="{HIC}" stroke-width="1.3"/>'); text(lx+393,ly+2,"Hi-C loop",size=5.5)

svg=(f'<svg xmlns="http://www.w3.org/2000/svg" width="183mm" height="71mm" viewBox="0 0 {W:.0f} {H:.0f}">'
     f'<defs>{"".join(defs)}</defs><rect width="100%" height="100%" fill="#FFFFFF"/>'+"\n".join(out)+"</svg>")
open("a_coordinated_binding_schematic.svg","w").write(svg)
