"""Standalone mesh illustration, independent of the browser UI."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np, json
from PIL import Image,ImageDraw,ImageFont
import trimesh,grip,hand
cal=grip.calibration(json.loads((grip.ROOT/'data/calibration.json').read_text()))
b=grip.baseline(); m=trimesh.Trimesh(b['vertices'],b['faces'],process=False); h=hand.render_mesh(cal)
im=Image.new('RGB',(1600,1000),'#f6f7f1');draw=ImageDraw.Draw(im)
font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',24);small=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',17)
draw.text((40,28),'Grip Studio / Fixed impression + illustrative hand',fill='#254a37',font=font)
draw.text((40,67),'Existing grip surface preserved. The hand is a visual aid, not a recovered scan or contact simulation.',fill='#687363',font=small)
for index,view in enumerate([np.array([.8,1.2,1.8]),np.array([-.6,-1.4,1.6])]):
 view=view/np.linalg.norm(view);right=np.cross([0,1,0],view);right/=np.linalg.norm(right);up=np.cross(view,right);basis=np.column_stack([right,up,view])
 triangles=np.concatenate([m.triangles,h.triangles]);colors=np.concatenate([np.tile([76,127,97],(len(m.faces),1)),np.tile([204,164,131],(len(h.faces),1))]);q=triangles@basis
 mid=(q.min((0,1))+q.max((0,1)))/2;q-=mid
 scale=min(700/np.ptp(q[:,:,0]),740/np.ptp(q[:,:,1]));normal=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-12);intensity=.46+.54*np.clip(normal@np.array([-.3,.5,.81]),0,1)
 # Rasterize with a depth buffer and interpolated normals (no browser involved).
 canvas=np.array(im);depth=np.full((1000,1600),-np.inf)
 normals=np.concatenate([m.vertex_normals[m.faces],h.vertex_normals[h.faces]])@basis
 for j in range(len(q)):
  p=q[j].copy();p[:,0]=index*800+400+p[:,0]*scale;p[:,1]=530-p[:,1]*scale
  x0=max(index*800,int(np.floor(p[:,0].min())));x1=min(index*800+799,int(np.ceil(p[:,0].max())))
  y0=max(100,int(np.floor(p[:,1].min())));y1=min(920,int(np.ceil(p[:,1].max())))
  if x1<x0 or y1<y0:continue
  xx,yy=np.meshgrid(np.arange(x0,x1+1)+.5,np.arange(y0,y1+1)+.5)
  den=(p[1,1]-p[2,1])*(p[0,0]-p[2,0])+(p[2,0]-p[1,0])*(p[0,1]-p[2,1])
  if abs(den)<1e-8:continue
  u=((p[1,1]-p[2,1])*(xx-p[2,0])+(p[2,0]-p[1,0])*(yy-p[2,1]))/den
  v=((p[2,1]-p[0,1])*(xx-p[2,0])+(p[0,0]-p[2,0])*(yy-p[2,1]))/den;w=1-u-v
  zz=u*p[0,2]+v*p[1,2]+w*p[2,2];region=depth[y0:y1+1,x0:x1+1]
  mask=(u>=0)&(v>=0)&(w>=0)&(zz>region)
  if not mask.any():continue
  normal=u[...,None]*normals[j,0]+v[...,None]*normals[j,1]+w[...,None]*normals[j,2]
  normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-10)
  lighting=.55+.45*np.clip(normal@np.array([-.3,.5,.81]),0,1)
  pixels=np.clip(colors[j]*lighting[...,None],0,255).astype(np.uint8)
  canvas[y0:y1+1,x0:x1+1][mask]=pixels[mask];region[mask]=zz[mask]
 im=Image.fromarray(canvas);draw=ImageDraw.Draw(im)
 draw.text((index*800+40,935),'Palm side' if index==0 else 'Finger side',fill='#254a37',font=small)
im.save(grip.ROOT/'hand-grip-render.png')
