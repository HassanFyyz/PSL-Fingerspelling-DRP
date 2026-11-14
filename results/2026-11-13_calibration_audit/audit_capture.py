"""Audit a capture and decode its existing grayscale crops; never changes labels or training files."""
import argparse,base64,collections,hashlib,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ap=argparse.ArgumentParser();ap.add_argument('capture');ap.add_argument('out');a=ap.parse_args()
src=Path(a.capture);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
d=json.loads(src.read_text());size=d['img_size']; font_path='/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
font=ImageFont.truetype(font_path,20);small=ImageFont.truetype(font_path,14);titlefont=ImageFont.truetype(font_path,25)
by=collections.defaultdict(list);rows=[];bad=[];total=0
for t in d['takes']:
 l,tid=t['letter'],t['take'];prefix=f'U{ord(l):04X}_{l}';dst=out/'crops'/prefix/f'take_{tid:03d}';dst.mkdir(parents=True,exist_ok=True)
 p=np.asarray([f['pts'] for f in t['frames']],dtype=np.float32)
 decoded=[]
 for i,f in enumerate(t['frames']):
  if f.get('crop') is None:continue
  raw=base64.b64decode(f['crop'],validate=True)
  if len(raw)!=size*size:raise ValueError(f'bad crop {l}/{tid}/{i}')
  im=Image.fromarray(np.frombuffer(raw,dtype=np.uint8).reshape(size,size));im.save(dst/f'frame_{i:04d}.png');decoded.append((i,im));total+=1
 ci,im=decoded[len(decoded)//2] if decoded else (None,None)
 by[l].append((t,ci,im))
 rows.append({'letter':l,'take':tid,'n_frames':len(p),'handedness_recorded':t.get('handedness'),'hand_frames':t.get('hand_frames'), 'n_crops':len(decoded),'display_crop_frame':ci,'training_frame':len(p)//2,'landmarks_shape':list(p.shape),'finite':bool(np.isfinite(p).all()),'crop_dir':str(dst.relative_to(out))})
thumbs=[]
for l,ts in by.items():
 ncol=5;nrow=math.ceil(len(ts)/ncol);w=260*ncol;h=106+286*nrow
 sheet=Image.new('RGB',(w,h),'white');dr=ImageDraw.Draw(sheet)
 dr.text((16,10),f'Calibration {l} (U{ord(l):04X}) - {len(ts)} takes',font=titlefont,fill='black')
 dr.text((16,45),'Captured crops from early frames; training uses median-frame landmarks.',font=small,fill='black')
 dr.text((16,68),'Signs/labels unverified. MediaPipe hand tags are preserved; they are not physical-hand ground truth.',font=small,fill='black')
 for k,(t,ci,im) in enumerate(ts):
  x=(k%ncol)*260+10;y=106+(k//ncol)*286
  if im is not None:sheet.paste(im.resize((236,236),Image.Resampling.LANCZOS),(x,y))
  dr.text((x,y+238),f"take {t['take']} | {t.get('handedness')} | crop f{ci}",font=small,fill='black')
  dr.text((x,y+258),f"train: landmarks f{len(t['frames'])//2}",font=small,fill='black')
 dst=out/'contact_sheets';dst.mkdir(exist_ok=True);sheet.save(dst/f'U{ord(l):04X}_{l}.png')
 # compact overview, one crop per take, all letters grouped
 tile=Image.new('RGB',(1000,160),'white');tr=ImageDraw.Draw(tile);tr.text((5,10),f'{l} U{ord(l):04X}',font=font,fill='black')
 for j,(t,ci,im) in enumerate(ts):
  if im:tile.paste(im.resize((74,74)),(90+j*81,5));tr.text((90+j*81,83),str(t['take']),font=small,fill='black')
 thumbs.append(tile)
for page in range(math.ceil(len(thumbs)/9)):
 part=thumbs[page*9:(page+1)*9];ov=Image.new('RGB',(1000,160*len(part)),'white')
 for j,tile in enumerate(part):ov.paste(tile,(0,j*160))
 ov.save(out/f'overview_{page+1}.png')
summary={'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'n_takes':len(rows),'n_letters':len(by),'n_frames':sum(r['n_frames'] for r in rows),'n_crops':total,'takes':rows,'interpretation':'Images decoded from embedded crops; no new MediaPipe inference. Not necessarily the training frame.'}
(out/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='takes'},ensure_ascii=False,indent=2))
