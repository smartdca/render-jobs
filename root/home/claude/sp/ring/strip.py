import sys; sys.path.insert(0,'/home/claude/dca-media/kit/scene')
import os; os.chdir('/home/claude/dca-media/kit/scene')
import content as C
from PIL import Image, ImageDraw
def c(sym,col,rate=6,chan='c',flip=False): return {'type':'card','sym':sym,'col':col,'rate':rate,'chan':chan,'flip':flip}
A=['AAPL','MSFT','NVDA','AMZN','META','TSLA','AMD','MU','V','KO','MCD','NKE','UBER','BTC']
# (width in units of strip height, spec)
SEG=[
 (3.2,{'type':'tape','syms':A,'lanes':2,'laps':[2,1]}),
 (1.25,{'type':'score'}),
 (1.7,c('AAPL','yellow',5,'hero',True)),
 (3.0,{'type':'multi','syms':['NVDA','MSFT','AMZN'],'col':'navy','rate':4}),
 (2.3,{'type':'candle','sym':'BTC','iv':1.5,'n':40}),
 (1.7,c('TSLA','pink',4,'hero',True)),
 (2.2,{'type':'heat','syms':['AAPL','MSFT','NVDA','AMZN','META','TSLA','KO','V'],'cols':4,'rate':2}),
 (1.0,{'type':'list','syms':['NVDA','AMD','MU','TSLA','UBER','META'],'rates':[1,4,0,7,2,1],'chan':'lg1','rows':3}),
 (1.7,c('META','blue',4,'hero',True)),
 (2.3,{'type':'candle','sym':'NVDA','iv':2.5,'n':30}),
 (3.0,{'type':'multi','syms':['V','KO','MCD'],'col':'black','rate':3,'chan':'b'}),
]
SEG2=[
 (2.3,{'type':'candle','sym':'TSLA','iv':2.0,'n':34}),
 (1.7,c('MSFT','navy',3,'hero',True)),
 (3.0,{'type':'multi','syms':['AMD','MU','UBER'],'col':'purple','rate':3,'chan':'m2'}),
 (3.2,{'type':'tape','syms':A[::-1],'lanes':2,'laps':[1,2]}),
 (1.7,c('KO','white',2,'hero',True)),
 (2.2,{'type':'heat','syms':['NVDA','AMD','MU','UBER','NKE','MCD','V','BTC'],'cols':4,'rate':2}),
 (1.25,{'type':'score'}),
 (1.7,c('AMZN','orange',4,'hero',True)),
 (2.3,{'type':'candle','sym':'META','iv':2.5,'n':30,'chan':'s2'}),
 (1.0,{'type':'list','syms':['AAPL','MSFT','AMZN','V','KO','MCD'],'rates':[2,0,5,1,3,0],'chan':'lg2','rows':3}),
 (1.7,c('NVDA','teal',4,'hero',True)),
]
SEG3=[
 (2.3,{'type':'candle','sym':'AMZN','iv':2.0,'n':34,'chan':'s3'}),
 (1.7,c('V','gold',2,'hero',True)),
 (3.0,{'type':'multi','syms':['NKE','MCD','UBER'],'col':'teal','rate':3,'chan':'m3'}),
 (1.7,c('AMD','orange',4,'hero',True)),
 (2.2,{'type':'heat','syms':['TSLA','AAPL','KO','MU','V','NKE','AMZN','MSFT'],'cols':4,'rate':2,'chan':'h3'}),
 (1.0,{'type':'list','syms':['BTC','NVDA','META','AMD','UBER','KO'],'rates':[3,1,2,0,1,2],'chan':'lg3','rows':3}),
]
def make(h,t=37.0,gap=0.03,seg=None):
  SEGX=seg or SEG
  W=int(sum(w for w,_ in SEGX)*h); im=Image.new('RGB',(W,h),(6,8,14)); x=0
  g=int(gap*h)
  for w,sp in SEGX:
    ww=int(w*h); tile=C.R[sp['type']](ww-g,h,t,sp); im.paste(tile,(x+g//2,0)); x+=ww
  return im
if __name__=='__main__':
  im=make(int(sys.argv[1]) if len(sys.argv)>1 else 400)
  im.save('/home/claude/sp/ring/strip.png'); print(im.size)
