"""TFT Adim Adim Gorselleri — 7 ayri PNG"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

DESKTOP = os.path.expanduser('~/OneDrive/Desktop')
outdir = os.path.join(DESKTOP, 'TFT_Adim_Adim')
os.makedirs(outdir, exist_ok=True)

BG='#0d1117'; CARD='#161b22'; BORDER='#30363d'; TEXT='#e6edf3'; MUTED='#8b949e'
BLUE='#58a6ff'; GREEN='#3fb950'; PURPLE='#bc8cff'; ORANGE='#d29922'
RED='#f85149'; CYAN='#39d2c0'

def setup(title, sub=None, fs=(16,9)):
    fig,ax=plt.subplots(1,1,figsize=fs); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(0,100); ax.set_ylim(0,60); ax.axis('off')
    ax.text(50,57.5,title,ha='center',fontsize=18,fontweight='bold',color=TEXT)
    if sub: ax.text(50,55.5,sub,ha='center',fontsize=10,color=MUTED)
    return fig,ax

def box(ax,x,y,w,h,c,a=0.15,lw=1.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3',facecolor=c,alpha=a,edgecolor=c,linewidth=lw))

def arr(ax,x1,y1,x2,y2,c=MUTED,lw=2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color=c,lw=lw))

def save(fig, name):
    fig.savefig(os.path.join(outdir, name), dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close(fig)

# ══════════════════════════════
# 1. PENCERE
# ══════════════════════════════
fig,ax = setup('Adim 1: Pencere Olustur','TFT gecmise 60 gun bakar, gelecek 30 gunu tahmin eder')
ax.plot([5,95],[30,30],color=BORDER,lw=2)
box(ax,8,24,42,12,PURPLE)
ax.text(29,34.5,'ENCODER (60 gun)',ha='center',fontsize=13,fontweight='bold',color=PURPLE)
ax.text(29,31.5,'Gecmis veri: pax, fiyat, doluluk...',ha='center',fontsize=9,color=MUTED)
ax.text(29,29,'HER SEYI biliyoruz',ha='center',fontsize=10,fontweight='bold',color=PURPLE)
box(ax,52,24,42,12,GREEN)
ax.text(73,34.5,'DECODER (30 gun)',ha='center',fontsize=13,fontweight='bold',color=GREEN)
ax.text(73,31.5,'Gelecek: ay, gun, tatil...',ha='center',fontsize=9,color=MUTED)
ax.text(73,29,'Sadece TAKVIM bilgisi',ha='center',fontsize=10,fontweight='bold',color=GREEN)
ax.text(8,22,'2026-02-01',fontsize=8,color=MUTED,fontfamily='monospace')
ax.text(48,22,'2026-03-31',fontsize=8,color=MUTED,fontfamily='monospace')
ax.text(52,22,'2026-04-01',fontsize=8,color=MUTED,fontfamily='monospace')
ax.text(90,22,'2026-04-30',fontsize=8,color=MUTED,fontfamily='monospace')
ax.plot([50.5,50.5],[22,38],color=RED,lw=2,ls='--')
ax.text(50.5,39,'BUGUN',ha='center',fontsize=9,fontweight='bold',color=RED)
box(ax,10,8,80,12,BLUE,a=0.08)
ax.text(50,18.5,'Ornek: IST-LHR Economy',ha='center',fontsize=11,fontweight='bold',color=BLUE)
ax.text(15,15,'Feb 01: pax=312, fare=$245',fontsize=8,color='#c9d1d9',fontfamily='monospace')
ax.text(15,12,'Mar 31: pax=475, fare=$310',fontsize=8,color='#c9d1d9',fontfamily='monospace')
ax.text(55,15,'Apr 01: ay=4, gun=Sali  -> ? yolcu',fontsize=8,color=GREEN,fontfamily='monospace')
ax.text(55,12,'Apr 30: ay=4, gun=Per   -> ? yolcu',fontsize=8,color=GREEN,fontfamily='monospace')
box(ax,20,1,60,5,ORANGE,a=0.1)
ax.text(50,4,'SORU: Nisan ayinda her gun kac yolcu olacak?',ha='center',fontsize=11,fontweight='bold',color=ORANGE)
save(fig,'01_Pencere_Olustur.png')
print('1/7')

# ══════════════════════════════
# 2. VSN
# ══════════════════════════════
fig,ax = setup('Adim 2: Variable Selection Network (VSN)','62 feature dan en onemlilerini otomatik secer')
box(ax,2,5,28,48,MUTED,a=0.05)
ax.text(16,51.5,'62 FEATURE',ha='center',fontsize=12,fontweight='bold',color=MUTED)
feats = ['Rota Kimligi','Ort. Bilet Fiyati','Yaz Tatili','Kalkis Ayi','Ucus Sayisi',
         'Baglanti Yolcu %','Is Seyahati','Hac/Umre','Gec Rezervasyon','Kalkis Gunu',
         'Doluluk','Kapasite','Cocuk %','Elite %','... +48 daha']
ws = [30,13.4,11,7.9,18.2,5.4,5.7,5.3,4.6,7.3,2.1,1.8,0.8,0.5,0]
for i,(f,w) in enumerate(zip(feats,ws)):
    y=48-i*2.8; c = TEXT if w>3 else MUTED
    ax.text(4,y,f,fontsize=7.5,color=c)

box(ax,34,15,24,30,CYAN,lw=2)
ax.text(46,43,'VARIABLE',ha='center',fontsize=15,fontweight='bold',color=CYAN)
ax.text(46,40,'SELECTION',ha='center',fontsize=15,fontweight='bold',color=CYAN)
ax.text(46,37,'NETWORK',ha='center',fontsize=15,fontweight='bold',color=CYAN)
ax.text(46,32,'Her feature icin',ha='center',fontsize=9,color=MUTED)
ax.text(46,30,'agirlik ogrenir',ha='center',fontsize=9,color=MUTED)
ax.text(46,26,'Onemli -> yuksek',ha='center',fontsize=9,color=GREEN)
ax.text(46,24,'Gereksiz -> ~0',ha='center',fontsize=9,color=RED)
ax.text(46,19,'Otomatik secim!',ha='center',fontsize=10,fontweight='bold',color=CYAN)
arr(ax,30,30,34,30,MUTED)

box(ax,62,8,36,42,GREEN,a=0.06)
ax.text(80,48,'SECILMIS ONEMLI OZELLIKLER',ha='center',fontsize=10,fontweight='bold',color=GREEN)
tops=[('Rota Kimligi',30.0,BLUE),('Ucus Sayisi',18.2,GREEN),('Ort. Fiyat',13.4,CYAN),
      ('Yaz Tatili',11.0,GREEN),('Kalkis Ayi',7.9,GREEN),('Kalkis Gunu',7.3,GREEN),
      ('Is Seyahati',5.7,PURPLE),('Baglanti %',5.4,PURPLE),('Hac/Umre',5.3,PURPLE)]
for i,(f,w,c) in enumerate(tops):
    y=44-i*3.8; bw=w/30*16
    ax.barh(y,bw,height=2.2,left=68,color=c,alpha=0.4,edgecolor=c)
    ax.text(68+bw+0.5,y,f'{w:.1f}%',fontsize=8,color=c,va='center',fontfamily='monospace')
    ax.text(67.5,y,f,fontsize=7.5,color=TEXT,va='center',ha='right')
arr(ax,58,30,62,30,CYAN)

box(ax,15,1,70,5,ORANGE,a=0.08)
ax.text(50,4.5,'"Bu rotada en cok bilet fiyatina ve yaz tatili etkisine bak"',ha='center',fontsize=10,fontweight='bold',color=ORANGE)
save(fig,'02_Variable_Selection_Network.png')
print('2/7')

# ══════════════════════════════
# 3. ENCODER
# ══════════════════════════════
fig,ax = setup('Adim 3: LSTM Encoder','Son 60 gunu sirayla okuyup tek bir ozet vektore sikistirir')
labels = [('Feb 01','h1'),('Feb 02','h2'),('...',''),('Mar 01','h30'),('...',''),('Mar 29','h58'),('Mar 30','h59'),('Mar 31','h60')]
for i,(d,h) in enumerate(labels):
    x=6+i*11.5; c=ORANGE if i==7 else PURPLE
    box(ax,x,28,9,12,c,a=0.15 if i!=7 else 0.25)
    if d=='...':
        ax.text(x+4.5,34,'...',ha='center',fontsize=16,color=MUTED); continue
    ax.text(x+4.5,38.5,'LSTM',ha='center',fontsize=8,fontweight='bold',color=c)
    ax.text(x+4.5,35.5,h,ha='center',fontsize=10 if i!=7 else 13,color=TEXT if i!=7 else ORANGE,fontweight='bold' if i==7 else 'normal',fontfamily='monospace')
    ax.text(x+4.5,30,d,ha='center',fontsize=7,color=MUTED)
    if i<7 and labels[i+1][0]!='...' and d!='...':
        arr(ax,x+9.7,34,x+11.3,34,c)
    elif i<7 and labels[i+1][0]=='...':
        arr(ax,x+9.7,34,x+11.3,34,MUTED)

box(ax,8,16,85,8,CYAN,a=0.06)
ax.text(50,22.5,'VSN den gelen agirlikli feature vektorleri (her gun icin)',ha='center',fontsize=9,color=CYAN)
ax.text(12,19,'pax=312, fare=$245',fontsize=7,color=MUTED,fontfamily='monospace')
ax.text(42,19,'pax=410, fare=$280',fontsize=7,color=MUTED,fontfamily='monospace')
ax.text(75,19,'pax=475, fare=$310',fontsize=7,color=ORANGE,fontfamily='monospace')

box(ax,25,44,50,8,ORANGE,a=0.12)
ax.text(50,50.5,'CIKTI: h60 = Son 60 gunun ozeti',ha='center',fontsize=12,fontweight='bold',color=ORANGE)
ax.text(50,47,'Trend + mevsimsellik + momentum bilgisi tek vektorde',ha='center',fontsize=9,color=TEXT)
arr(ax,88,40.5,75,44,ORANGE,lw=2.5)
save(fig,'03_LSTM_Encoder.png')
print('3/7')

# ══════════════════════════════
# 4. DECODER
# ══════════════════════════════
fig,ax = setup('Adim 4: LSTM Decoder','Bilinen gelecek bilgisiyle 30 gunu DOGRUDAN tahmin eder')
box(ax,2,40,22,10,ORANGE,a=0.15)
ax.text(13,48.5,'h60 (Encoder Ozeti)',ha='center',fontsize=10,fontweight='bold',color=ORANGE)
ax.text(13,43,'Son 60 gun bilgisi',ha='center',fontsize=8,color=MUTED)
arr(ax,24,45,28,38,ORANGE)

days=[('Apr 01','ay=4, Sal','456.6'),('Apr 05','ay=4, Cmt','483.2'),('Apr 15','ay=4, Sal','488.5'),('Apr 30','ay=4, Per','488.3')]
for i,(d,info,p) in enumerate(days):
    x=28+i*17
    box(ax,x,28,14,12,GREEN)
    ax.text(x+7,38.5,'DECODER',ha='center',fontsize=8,fontweight='bold',color=GREEN)
    ax.text(x+7,36,d,ha='center',fontsize=9,color=TEXT,fontfamily='monospace')
    ax.text(x+7,33.5,info,ha='center',fontsize=7,color=MUTED)
    ax.text(x+7,30.5,p+' yolcu',ha='center',fontsize=11,fontweight='bold',color=GREEN,fontfamily='monospace')
    if i<3:
        ax.text(x+15.5,34,'...' if i<3 else '',ha='center',fontsize=12,color=MUTED)

box(ax,10,5,80,14,BLUE,a=0.08)
ax.text(50,17,'ONEMLI: Iteratif DEGIL',ha='center',fontsize=14,fontweight='bold',color=BLUE)
ax.text(50,13.5,'ARIMA: t+1 tahminiyle t+2 yi tahmin eder -> hata birikir',ha='center',fontsize=10,color=RED)
ax.text(50,10.5,'TFT: Her gunu bagimsiz tahmin eder -> hata BIRIKMEZ',ha='center',fontsize=10,color=GREEN)
ax.text(50,7.5,'30. gundeki hata 1. gundeki hatayla ayni seviyede',ha='center',fontsize=10,fontweight='bold',color=TEXT)
save(fig,'04_LSTM_Decoder.png')
print('4/7')

# ══════════════════════════════
# 5. ATTENTION
# ══════════════════════════════
fig,ax = setup('Adim 5: Multi-Head Attention','Gecmisin hangi gunu gelecegi en cok etkiliyor?')
enc=['Feb 01','Feb 15','Mar 01','Mar 10','Mar 20','Mar 25','Mar 28','Mar 31']
atts=[0.02,0.03,0.08,0.10,0.15,0.22,0.28,0.37]
for i,(d,a) in enumerate(zip(enc,atts)):
    x=8+i*11.5
    box(ax,x,42,9,6,PURPLE,a=0.1)
    ax.text(x+4.5,46.5,d,ha='center',fontsize=7.5,color=MUTED,fontfamily='monospace')
    lw2=a*12+0.5; col=ORANGE if a>0.15 else MUTED
    ax.plot([x+4.5,50],[42,20],color=col,lw=lw2,alpha=min(a*2+0.1,0.9))
    ax.text(x+4.5,40,f'{a:.0%}',ha='center',fontsize=8,fontweight='bold',color=col,fontfamily='monospace')

box(ax,38,12,24,8,GREEN,lw=2)
ax.text(50,18.5,'Apr 01 Tahmini',ha='center',fontsize=12,fontweight='bold',color=GREEN)
ax.text(50,14,'y = 456.6 yolcu',ha='center',fontsize=10,color=TEXT,fontfamily='monospace')

box(ax,10,1,80,8,CYAN,a=0.08)
ax.text(50,7.5,'Attention Yorumu',ha='center',fontsize=11,fontweight='bold',color=CYAN)
ax.text(50,4.5,'"Apr 01 tahmini icin en cok son 1 haftaya baktim: %87 agirlik"',ha='center',fontsize=10,color=TEXT)
ax.text(50,2.5,'60 gun oncesinin etkisi sadece %2 — uzak gecmis unutuluyor',ha='center',fontsize=9,color=MUTED)
save(fig,'05_Multi_Head_Attention.png')
print('5/7')

# ══════════════════════════════
# 6. QUANTILE
# ══════════════════════════════
fig,ax = setup('Adim 6: Quantile Cikti','Tek sayi degil, 3 senaryo: iyimser / beklenen / kotumser',fs=(16,9))
np.random.seed(42)
days=np.arange(1,31)
p50=457+np.cumsum(np.random.randn(30)*8)
p10=p50-60-np.random.rand(30)*20
p90=p50+55+np.random.rand(30)*25
actual=p50+np.random.randn(30)*25
sx=lambda d: 10+(d-1)*2.7
sy=lambda v: 5+(v-350)/250*42

ax.fill_between([sx(d) for d in days],[sy(p) for p in p10],[sy(p) for p in p90],alpha=0.12,color=GREEN)
ax.plot([sx(d) for d in days],[sy(p) for p in p50],color=GREEN,lw=2.5,label='P50 (beklenen)')
ax.plot([sx(d) for d in days],[sy(p) for p in p10],color=BLUE,lw=1,ls='--',alpha=0.7,label='P10 (en kotu)')
ax.plot([sx(d) for d in days],[sy(p) for p in p90],color=ORANGE,lw=1,ls='--',alpha=0.7,label='P90 (en iyi)')
ax.scatter([sx(d) for d in days],[sy(a) for a in actual],color=RED,s=20,zorder=5,label='Gercek')

ax.text(sx(1)-2,sy(p90[0]),'P90',fontsize=9,color=ORANGE,fontweight='bold')
ax.text(sx(1)-2,sy(p50[0]),'P50',fontsize=9,color=GREEN,fontweight='bold')
ax.text(sx(1)-2,sy(p10[0]),'P10',fontsize=9,color=BLUE,fontweight='bold')
ax.legend(loc='upper right',fontsize=9,facecolor=CARD,edgecolor=BORDER,labelcolor=TEXT,bbox_to_anchor=(0.97,0.88))

box(ax,58,42,38,10,ORANGE,a=0.08)
ax.text(77,50.5,'Neden 3 senaryo?',ha='center',fontsize=11,fontweight='bold',color=ORANGE)
ax.text(77,48,'P90 cikarsa -> agresif fiyatla',ha='center',fontsize=9,color=GREEN)
ax.text(77,45.5,'P10 cikarsa -> ihtiyatli ol',ha='center',fontsize=9,color=RED)
ax.text(77,43.5,'Risk yonetimi icin kritik',ha='center',fontsize=9,color=TEXT)
save(fig,'06_Quantile_Cikti.png')
print('6/7')

# ══════════════════════════════
# 7. SIMULASYONA AKTARIM
# ══════════════════════════════
fig,ax = setup('Adim 7: Simulasyona Aktarim','TFT tahmini -> tavan/taban bandi -> fiyatlandirma karari')

box(ax,5,42,22,10,GREEN,lw=2)
ax.text(16,50.5,'TFT TAHMINI',ha='center',fontsize=11,fontweight='bold',color=GREEN)
ax.text(16,47.5,'IST-LHR Apr 01',ha='center',fontsize=9,color=MUTED)
ax.text(16,45,'P50 = 457 yolcu',ha='center',fontsize=12,fontweight='bold',color=TEXT,fontfamily='monospace')
arr(ax,27,47,35,47,GREEN,lw=2.5)

box(ax,35,38,24,16,CYAN,lw=2)
ax.text(47,52.5,'FORECAST BRIDGE',ha='center',fontsize=11,fontweight='bold',color=CYAN)
ax.text(47,49.5,'Tavan = 457 x 1.3 = 594',ha='center',fontsize=9,color=ORANGE,fontfamily='monospace')
ax.text(47,47,'Beklenen = 457',ha='center',fontsize=9,color=GREEN,fontfamily='monospace')
ax.text(47,44.5,'Taban = 457 x 0.5 = 228',ha='center',fontsize=9,color=BLUE,fontfamily='monospace')

arr(ax,59,50,68,50,ORANGE)
box(ax,68,46,28,8,RED,a=0.1)
ax.text(82,52.5,'Satilan > 594',ha='center',fontsize=10,fontweight='bold',color=RED)
ax.text(82,49,'Talep x 0.3 = FREN',ha='center',fontsize=9,color=RED)
ax.text(82,47,'"Cok hizli, yavasla"',ha='center',fontsize=8,color=MUTED)

arr(ax,59,44,68,37,GREEN)
box(ax,68,33,28,8,GREEN,a=0.1)
ax.text(82,39.5,'228 < Satilan < 594',ha='center',fontsize=10,fontweight='bold',color=GREEN)
ax.text(82,36,'Normal devam',ha='center',fontsize=9,color=GREEN)
ax.text(82,34,'"Her sey yolunda"',ha='center',fontsize=8,color=MUTED)

arr(ax,59,41,68,25,BLUE)
box(ax,68,21,28,8,BLUE,a=0.1)
ax.text(82,27.5,'Satilan < 228',ha='center',fontsize=10,fontweight='bold',color=BLUE)
ax.text(82,24,'Talep x 1.5 = GAZ',ha='center',fontsize=9,color=BLUE)
ax.text(82,22,'"Az satiyoruz, hizlan"',ha='center',fontsize=8,color=MUTED)

arr(ax,47,38,47,20,PURPLE,lw=2.5)
box(ax,12,4,70,14,PURPLE,a=0.08)
ax.text(47,16.5,'FIYATLANDIRMA ETKISI',ha='center',fontsize=12,fontweight='bold',color=PURPLE)
ax.text(30,13,'TFT "cok talep" derse',ha='center',fontsize=9,color=TEXT)
ax.text(30,10.5,'-> Ucuz siniflar kapanir',ha='center',fontsize=9,color=RED)
ax.text(30,8,'-> Fiyat yukselir',ha='center',fontsize=9,color=RED)
ax.text(65,13,'TFT "az talep" derse',ha='center',fontsize=9,color=TEXT)
ax.text(65,10.5,'-> Ucuz siniflar acik kalir',ha='center',fontsize=9,color=GREEN)
ax.text(65,8,'-> Koltuk doldurma oncelik',ha='center',fontsize=9,color=GREEN)
save(fig,'07_Simulasyona_Aktarim.png')
print('7/7')

print(f'\nKaydedildi: {outdir}')
