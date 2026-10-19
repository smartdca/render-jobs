# Loop-safe animated billboard content. All behaviour periodic in LOOP seconds.
import math, numpy as np
from PIL import Image, ImageDraw, ImageFont
LOOP=120.0
FB='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'; FK='/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc'
FM='/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'
_fc={}
def F(s,k='b'):
    s=max(6,int(s)); key=(s,k)
    if key not in _fc: _fc[key]=ImageFont.truetype(FM,s) if k=='m' else ImageFont.truetype(FK if k=='k' else FB,s,index=0)
    return _fc[key]
def fit(d,text,maxw,maxh,k='b'):
    s=maxh
    while s>6 and d.textlength(text,font=F(s,k))>maxw: s*=0.93
    return F(s,k),s
UP=(40,205,115); DN=(240,72,72); WH=(250,250,248); DK=(16,16,20)
PAL={'yellow':((248,200,36),DK),'navy':((18,32,86),WH),'blue':((34,92,220),WH),'purple':((112,66,204),WH),
     'white':((242,242,238),DK),'orange':((246,124,32),WH),'cyan':((22,170,204),WH),'pink':((228,54,122),WH),
     'black':((14,14,18),WH),'teal':((14,120,120),WH),'magenta':((190,40,190),WH),'gold':((214,160,48),DK)}
BASE={'AAPL':238.4,'TSLA':412.7,'NVDA':186.2,'BTC':112480.0,'UBER':94.3,'MU':168.5,'META':742.1,'AMZN':226.8,
      'MCD':301.6,'KO':69.4,'MSFT':512.3,'V':348.9,'NKE':76.2,'AMD':214.6}
SYMS=list(BASE)
def h32(s): 
    v=2166136261
    for ch in s: v=((v^ord(ch))*16777619)&0xffffffff
    return v
def rng(key): return np.random.default_rng(h32(key))
DAY={s:float(rng('day'+s).choice([-1,1])*rng('dm'+s).uniform(0.25,2.6)) for s in SYMS}   # daily % change, sticky sign
class Ticker:
    """price events for (symbol, channel) — zero-sum so loop is seamless"""
    _c={}
    def __init__(self,sym,chan,rate):
        r=rng(sym+'|'+chan); n=max(0,int(round(rate)))
        self.t=np.sort(r.uniform(0,LOOP,n)) if n else np.zeros(0)
        d=r.normal(0,0.045,n) if n else np.zeros(0)
        if n>1: d-=d.mean()
        elif n==1: d[:]=0.0
        self.d=d; self.cum=np.concatenate([[0],np.cumsum(d)]); self.sym=sym
    @classmethod
    def get(cls,sym,chan,rate):
        k=(sym,chan,rate)
        if k not in cls._c: cls._c[k]=Ticker(sym,chan,rate)
        return cls._c[k]
    def at(self,t):
        t%=LOOP; i=int(np.searchsorted(self.t,t,side='right'))
        pct=DAY[self.sym]+self.cum[i]
        p=BASE[self.sym]*(1+pct/100)
        last=self.t[i-1] if i>0 else (self.t[-1]-LOOP if len(self.t) else -99)
        age=t-last; dirn=np.sign(self.d[i-1]) if i>0 else (np.sign(self.d[-1]) if len(self.d) else 0)
        return p,pct,age,dirn
def fmt(sym,p): return f"{p:,.0f}" if p>10000 else f"{p:,.2f}"
def flash_amt(age,dur=1.4): return max(0.0,1-age/dur)**1.6 if age>=0 else 0.0
def hist(sym,n=48):
    r=rng('hist'+sym); w=np.cumsum(r.normal(0,1,n)); w=(w-w.min())/(w.max()-w.min()+1e-9)
    tr=DAY[sym]/abs(DAY[sym]); w=w*0.7+np.linspace(0,1,n)*(0.3 if tr>0 else -0.3)
    return w-w[-1]
def badge(d,x,y,pct,size):
    s=f"{'▲' if pct>=0 else '▼'} {abs(pct):.2f}%"; f=F(size); tw=d.textlength(s,font=f); pad=size*0.35
    d.rounded_rectangle((x,y,x+tw+2*pad,y+size*1.45),radius=size*0.28,fill=UP if pct>=0 else DN)
    d.text((x+pad,y+size*0.1),s,font=f,fill=WH)
def mix(a,b,k): return tuple(int(x*(1-k)+y*k) for x,y in zip(a,b))
# ---------------- slide renderers: f(w,h,t,local_t,spec)->RGB Image
def s_card(w,h,t,spec):
    sym=spec['sym']; bg,fg=PAL[spec['col']]; tk=Ticker.get(sym,spec.get('chan','c'),spec.get('rate',6))
    p,pct,age,dirn=tk.at(t); fl=flash_amt(age,1.2)
    im=Image.new('RGB',(w,h),bg); d=ImageDraw.Draw(im)
    land=w>=h*1.2; flip=spec.get('flip',False)
    tmax=w*0.48 if land else w*0.84; x0=(w*0.50 if flip else w*0.06) if land else w*0.08; y=h*0.08
    fT,sT=fit(d,sym,tmax,h*(0.30 if land else 0.20),'k'); d.text((x0,y),sym,font=fT,fill=fg); y+=sT*1.28
    col=mix(fg,UP if dirn>0 else DN,fl*0.85)
    fP,sP=fit(d,fmt(sym,p),tmax,sT*0.6); d.text((x0,y),fmt(sym,p),font=fP,fill=col); y+=sP*1.35
    badge(d,x0,y,pct,sP*0.45); y+=sP*0.66
    v=hist(sym)
    if land: X0,X1,Y0,Y1=(w*0.05,w*0.44,h*0.18,h*0.82) if flip else (w*0.56,w*0.95,h*0.18,h*0.82)
    else: X0,X1,Y0,Y1=w*0.08,w*0.92,y+h*0.06,h*0.92
    if Y1-Y0>h*0.08:
        rng_=max(1e-9,v.max()-v.min()); pts=[(X0+(X1-X0)*i/(len(v)-1),Y1-(x-v.min())/rng_*(Y1-Y0)) for i,x in enumerate(v)]
        lc=UP if pct>=0 else DN
        if bg in ((248,200,36),(242,242,238),(214,160,48)): lc=mix(lc,(0,0,0),0.25)
        d.line(pts,fill=lc,width=max(2,int(min(w,h)*0.022)),joint='curve')
    return im
def s_multi(w,h,t,spec):
    bg,fg=PAL[spec['col']]; im=Image.new('RGB',(w,h),bg); d=ImageDraw.Draw(im); syms=spec['syms']; n=len(syms); cw=w/n
    for i,s in enumerate(syms):
        x=i*cw; p,pct,age,dirn=Ticker.get(s,'m'+spec.get('chan',''),spec.get('rate',5)).at(t); fl=flash_amt(age,1.2)
        if i: d.line((x,h*0.14,x,h*0.86),fill=mix(bg,fg,0.25),width=max(2,int(w*0.003)))
        fT,sT=fit(d,s,cw*0.8,h*0.24,'k'); d.text((x+cw*0.08,h*0.09),s,font=fT,fill=fg)
        fP,sP=fit(d,fmt(s,p),cw*0.8,h*0.15); d.text((x+cw*0.08,h*0.09+sT*1.3),fmt(s,p),font=fP,fill=mix(fg,UP if dirn>0 else DN,fl*0.85))
        badge(d,x+cw*0.08,h*0.09+sT*1.3+sP*1.4,pct,sP*0.42)
        v=hist(s,36); r_=max(1e-9,v.max()-v.min())
        pts=[(x+cw*0.08+cw*0.84*j/(len(v)-1),h*0.92-(y-v.min())/r_*h*0.2) for j,y in enumerate(v)]
        d.line(pts,fill=UP if pct>=0 else DN,width=max(2,int(h*0.016)))
    return im
def s_candle(w,h,t,spec):
    sym=spec['sym']; iv=spec.get('iv',2.0); N=int(round(LOOP/iv)); r=rng('cd'+sym+spec.get('chan',''))
    inc=r.normal(0,1,N); inc-=inc.mean(); path=np.cumsum(inc)*BASE[sym]*0.0016
    closes=BASE[sym]*(1+DAY[sym]/100)+path-path[-1]
    bg=(9,11,17); im=Image.new('RGB',(w,h),bg); d=ImageDraw.Draw(im)
    for gy in range(1,5): d.line((0,h*gy/5,w,h*gy/5),fill=(30,34,46))
    nshow=spec.get('n',30); cw=w*0.84/nshow; k=t/iv; i_now=int(k); fr=k-i_now
    seg=[]
    for j in range(i_now-nshow+1,i_now+1):
        o=closes[(j-1)%N]; c=closes[j%N]
        if j==i_now: c=o+(c-o)*min(1,fr*1.4)
        rr=rng(f'w{sym}{j%N}'); hi=max(o,c)+abs(rr.normal(0,1))*BASE[sym]*0.0012; lo=min(o,c)-abs(rr.normal(0,1))*BASE[sym]*0.0012
        seg.append((o,hi,lo,c))
    L=min(s[2] for s in seg); H_=max(s[1] for s in seg); Y=lambda v: h*0.86-(v-L)/(H_-L+1e-9)*h*0.6
    for j,(o,hi,lo,c) in enumerate(seg):
        x=w*0.03+j*cw+cw/2; col=UP if c>=o else DN
        d.line((x,Y(hi),x,Y(lo)),fill=col,width=max(1,int(cw*0.12))); d.rectangle((x-cw*0.33,min(Y(o),Y(c)),x+cw*0.33,max(Y(o),Y(c))+1),fill=col)
    last=seg[-1][3]; y=Y(last); col=UP if last>=seg[-1][0] else DN
    d.line((w*0.03,y,w*0.88,y),fill=mix(col,bg,0.4),width=1)
    fP,sP=fit(d,fmt(sym,last),w*0.11,h*0.07); d.rectangle((w*0.88,y-sP*0.7,w*0.995,y+sP*0.7),fill=col); d.text((w*0.885,y-sP*0.65),fmt(sym,last),font=fP,fill=WH)
    fT,sT=fit(d,sym+('/USD' if sym=='BTC' else ''),w*0.4,h*0.11,'k'); d.text((w*0.04,h*0.04),sym+('/USD' if sym=='BTC' else ''),font=fT,fill=WH)
    pct=(last/BASE[sym]-1)*100; badge(d,w*0.04+d.textlength(sym+('/USD' if sym=='BTC' else ''),font=fT)+sT*0.5,h*0.04+sT*0.2,pct,sT*0.42)
    return im
UPB=(0,232,120); DNB=(255,62,62)
def s_list(w,h,t,spec):
    syms=spec['syms']; bg=(5,7,12); im=Image.new('RGB',(w,h),bg); d=ImageDraw.Draw(im)
    rows=spec.get('rows') or max(3,int(round(h/(w*0.44)))); rh=h/rows
    sc=spec.get('scroll',0.0)
    off=(t/LOOP*sc)%len(syms) if sc else 0; i0=int(off); fr=off-i0
    fT,sT=fit(d,'MSFT',w*0.46,rh*0.40,'k'); fP,sP=fit(d,'000.00',w*0.40,rh*0.27,'b'); fC,sC=fit(d,'+0.00%',w*0.30,rh*0.22,'b')
    for r_ in range(-1,rows+1):
        idx=(i0+r_)%len(syms); s=syms[idx]; y=(r_-fr)*rh
        rate=spec['rates'][idx%len(spec['rates'])]
        p,pct,age,dirn=Ticker.get(s,'L'+spec.get('chan',''),rate).at(t); fl=flash_amt(age,1.4)
        cc=UPB if pct>=0 else DNB; fc=UPB if dirn>0 else DNB
        if fl>0: d.rectangle((0,y+2,w,y+rh-2),fill=mix(bg,fc,fl*0.45))
        if spec.get('compact'):
            fT2,sT2=fit(d,'MSFT',w*0.9,rh*0.34,'k'); tw=d.textlength(s,font=fT2); d.text(((w-tw)/2,y+rh*0.08),s,font=fT2,fill=WH)
            cs=f"{'▲' if pct>=0 else '▼'}{abs(pct):.2f}%"; fC2,sC2=fit(d,cs,w*0.8,rh*0.26,'b'); tw=d.textlength(cs,font=fC2)
            d.rounded_rectangle(((w-tw)/2-sC2*0.3,y+rh*0.52,(w+tw)/2+sC2*0.3,y+rh*0.52+sC2*1.35),radius=sC2*0.25,fill=mix(cc,(255,255,255),fl*0.25))
            d.text(((w-tw)/2,y+rh*0.52+sC2*0.06),cs,font=fC2,fill=WH); d.line((0,y+rh-1,w,y+rh-1),fill=(30,34,44),width=2); continue
        d.text((w*0.05,y+(rh-sT*1.2)/2),s,font=fT,fill=WH)
        ps=fmt(s,p); tw=d.textlength(ps,font=fP); d.text((w*0.95-tw,y+rh*0.10),ps,font=fP,fill=mix((225,225,230),fc,fl))
        cs=f"{'▲' if pct>=0 else '▼'}{abs(pct):.2f}%"; tw=d.textlength(cs,font=fC); pad=sC*0.3
        x1=w*0.95; x0=x1-tw-2*pad; y0=y+rh*0.52
        d.rounded_rectangle((x0,y0,x1,y0+sC*1.35),radius=sC*0.25,fill=mix(cc,(255,255,255),fl*0.25))
        d.text((x0+pad,y0+sC*0.06),cs,font=fC,fill=WH)
        d.line((0,y+rh-1,w,y+rh-1),fill=(30,34,44),width=max(1,int(rh*0.02)))
    return im
def s_tape(w,h,t,spec):
    syms=spec['syms']; bg=spec.get('bg',(8,8,10)); im=Image.new('RGB',(w,h),bg); d=ImageDraw.Draw(im)
    lanes=spec.get('lanes',1); lh=h/lanes
    for L_ in range(lanes):
        ss=syms if L_%2==0 else syms[::-1]; sz=min(lh*0.42,w*0.08); f=F(sz); fk=F(sz,'k')
        items=[]; widths=[]
        for s in ss:
            p,pct,age,dirn=Ticker.get(s,'T',3).at(t); items.append((s,fmt(s,p),pct))
            widths.append(d.textlength(s+'  ',font=fk)+d.textlength(fmt(s,p)+'  ',font=f)+d.textlength('▲0.00%      ',font=f))
        total=sum(widths); k=spec.get('laps',[3,2])[L_%2]; x=-((t/LOOP)*k*total)%total
        x-=total; yb=L_*lh+(lh-sz*1.3)/2
        while x<w:
            for (s,ps,pc),wd in zip(items,widths):
                if x+wd>0 and x<w:
                    d.text((x,yb),s,font=fk,fill=WH); xx=x+d.textlength(s+'  ',font=fk)
                    d.text((xx,yb),ps,font=f,fill=(205,205,210)); xx+=d.textlength(ps+'  ',font=f)
                    d.text((xx,yb),f"{'▲' if pc>=0 else '▼'}{abs(pc):.2f}%",font=f,fill=UP if pc>=0 else DN)
                x+=wd
        if L_: d.line((0,L_*lh,w,L_*lh),fill=(40,40,48),width=2)
    return im
def s_heat(w,h,t,spec):
    syms=spec['syms']; cols=spec['cols']; rows=math.ceil(len(syms)/cols); cw,ch=w/cols,h/rows
    im=Image.new('RGB',(w,h),(8,8,10)); d=ImageDraw.Draw(im)
    for i,s in enumerate(syms):
        p,pct,age,dirn=Ticker.get(s,'H',spec.get('rate',3)).at(t); fl=flash_amt(age,1.5)
        a=min(1,0.4+abs(pct)/3); col=mix((8,8,10),UP if pct>=0 else DN,a*0.85); col=mix(col,WH,fl*0.15)
        x=(i%cols)*cw; y=(i//cols)*ch; d.rectangle((x+2,y+2,x+cw-2,y+ch-2),fill=col)
        f,sz=fit(d,s,cw*0.8,min(cw,ch)*0.26,'k'); tw=d.textlength(s,font=f); d.text((x+(cw-tw)/2,y+ch*0.2),s,font=f,fill=WH)
        st=f"{pct:+.2f}%"; f2,_=fit(d,st,cw*0.8,min(cw,ch)*0.18); tw=d.textlength(st,font=f2); d.text((x+(cw-tw)/2,y+ch*0.56),st,font=f2,fill=WH)
    return im
WM=None
def s_score(w,h,t,spec):
    global WM
    if WM is None:
        WM=Image.open('dcascore-wordmark-white.png').convert('RGBA'); WM=WM.crop(WM.getbbox())
    im=Image.new('RGB',(w,h),(16,14,12)); d=ImageDraw.Draw(im); m=min(w,h)
    vals=[42,58,71,64,49]; seg=LOOP/len(vals); k=int((t%LOOP)//seg); ph=((t%LOOP)%seg)/seg
    a,b=vals[k],vals[(k+1)%len(vals)]; e=0 if ph<0.7 else 0.5-0.5*math.cos(math.pi*(ph-0.7)/0.3); v=a+(b-a)*e
    r=w*0.8/WM.width; wm=WM.resize((max(1,int(WM.width*r)),max(1,int(WM.height*r))),Image.LANCZOS)
    im.paste(wm,(int((w-wm.width)/2),int(h*0.10)),wm)
    cx,cy,R=w/2,h*0.12+wm.height+(h*0.88-wm.height)*0.5,min(w*0.36,(h*0.8-wm.height)*0.42)
    d.arc((cx-R,cy-R,cx+R,cy+R),140,400,fill=(58,52,46),width=max(3,int(R*0.2)))
    d.arc((cx-R,cy-R,cx+R,cy+R),140,140+260*v/100,fill=(232,150,60),width=max(3,int(R*0.2)))
    f,sz=fit(d,'00',R*1.1,R*0.8,'k'); s=str(int(round(v))); tw=d.textlength(s,font=f); d.text((cx-tw/2,cy-sz*0.62),s,font=f,fill=WH)
    return im
def s_glyph(w,h,t,spec):
    bg,fg=PAL[spec['col']]; im=Image.new('RGB',(w,h),bg); d=ImageDraw.Draw(im)
    s=spec['sym']; p,pct,age,dirn=Ticker.get(s,'g',2).at(t)
    f,sz=fit(d,'▲' if pct>=0 else '▼',w*0.8,h*0.7); g='▲' if pct>=0 else '▼'; tw=d.textlength(g,font=f)
    d.text(((w-tw)/2,h*0.08),g,font=f,fill=UP if pct>=0 else DN); return im
R={'card':s_card,'multi':s_multi,'candle':s_candle,'list':s_list,'tape':s_tape,'heat':s_heat,'score':s_score,'glyph':s_glyph}
def board_frame(spec,w,h,t):
    """spec: {'slides':[(dur,slidespec),...],'off':seconds}; durations sum divides LOOP"""
    sl=spec['slides']; tot=sum(s[0] for s in sl); tt=(t+spec.get('off',0))%tot
    acc=0
    for i,(du,ss) in enumerate(sl):
        if tt<acc+du: break
        acc+=du
    lt=tt-acc; im=R[ss['type']](w,h,t,ss)
    if len(sl)>1:
        TR=0.35   # LED-style dip at slide change
        k=1.0
        if lt<TR: k=lt/TR
        elif du-lt<TR: k=(du-lt)/TR
        if k<1: im=Image.blend(Image.new('RGB',(w,h),(0,0,0)),im,max(0.0,k))
    return im
