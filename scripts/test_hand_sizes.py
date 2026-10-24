"""Reproducible size stress test; ANSUR length/breadth, estimated palm length."""
import sys,json,io,zipfile,urllib.request,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,trimesh,grip
from PIL import Image,ImageDraw,ImageFont
out=grip.ROOT/'results/hand-size-tests';out.mkdir(exist_ok=True,parents=True)
cal=grip.calibration(json.loads((grip.ROOT/'data/calibration.json').read_text()))
b=grip.baseline();base=trimesh.Trimesh(b['vertices'],b['faces'],process=False)
data={};sources={}
for sex,file in [('Female','women'),('Male','men')]:
 url=f'https://mreed.umtri.umich.edu/mreed/downloads/anthro/ansur/ansur_{file}.txt.zip'
 blob=urllib.request.urlopen(url).read();(out/f'ansur_{file}.zip').write_bytes(blob)
 z=zipfile.ZipFile(io.BytesIO(blob));rows=list(csv.DictReader(io.StringIO(z.read(z.namelist()[0]).decode('latin1').replace('\r\n','\n').replace('\r','\n')),delimiter='\t'))
 data[sex]=rows;sources[sex]=url
im=Image.new('RGB',(1500,1370),'#f6f7f1');draw=ImageDraw.Draw(im)
font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',23);small=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',17)
draw.text((35,25),'Five hand-size stress tests · same scale in every view',font=font,fill='#234b39')
draw.text((35,65),'ANSUR 1988 length/breadth percentiles. Palm length estimated from saved baseline ratio.',font=small,fill='#54665b')
draw.text((35,92),'Reference: 190 / 90 / 105 mm. Grey outline = baseline. Green = deformed grip. No hand pose changes.',font=small,fill='#54665b')
results=[]
for idx,(sex,pct) in enumerate([('Female',1),('Female',25),('Female',50),('Male',50),('Male',99)]):
 vals=[]
 for key in ['HAND_LNTH','HAND_BRTH_AT_METACARPALE']:
  v=[float(r[key]) for r in data[sex] if r.get(key) and float(r[key])>0];vals.append(float(np.percentile(v,pct)))
 hl,hb=vals;pl=hl*cal['reference']['palm_length']/cal['reference']['hand_length']
 name=f'{sex} P{pct:02d}';target=dict(hand_length=hl,hand_breadth=hb,palm_length=pl)
 pro=grip.profile(dict(name=name,target=target));mesh,factors=grip.deform(cal,pro);valid=grip.validate(mesh)
 if valid['valid']:mesh.export(out/f'{sex.lower()}-p{pct:02d}.stl')
 (out/f'{sex.lower()}-p{pct:02d}.json').write_text(json.dumps(pro,indent=2))
 r=dict(name=name,target=target,dimensions_mm=mesh.extents.tolist(),percent_change=((mesh.extents/base.extents-1)*100).tolist(),validation=valid,factors=factors,volume_change_percent=(mesh.volume/base.volume-1)*100);results.append(r)
 y=155+idx*235
 draw.text((35,y),name,font=font,fill='#234b39');draw.text((35,y+36),f'Hand {hl:.1f} · breadth {hb:.1f}',font=small,fill='#34473c');draw.text((35,y+62),f'Palm {pl:.1f} mm (estimated)',font=small,fill='#34473c')
 draw.text((35,y+99),' × '.join(f'{v:.1f}' for v in mesh.extents)+' mm',font=small,fill='#34473c');draw.text((35,y+126),'Mesh PASS' if valid['valid'] else 'Mesh FAIL — export blocked',font=small,fill='#34473c')
 for j,view in enumerate([np.array([.15,-.65,.75]),np.array([.15,.75,.65])]):
  view/=np.linalg.norm(view);right=np.array([1.,0,0]);right-=view*np.dot(right,view);right/=np.linalg.norm(right);up=np.cross(view,right);basis=np.column_stack([right,up,view]);scale=3.6;cx=650+j*550;cy=y+95
  for m,isbase in [(base,True),(mesh,False)]:
   q=m.triangles@basis;order=np.argsort(q[:,:,2].mean(1));norm=m.face_normals@view
   for k in order:
    pts=[(cx+v[0]*scale,cy-v[1]*scale) for v in q[k]]
    if isbase:draw.polygon(pts,fill='#cbd0ca')
    else:
     light=.60+.40*abs(norm[k]);draw.polygon(pts,fill=tuple(int(c*light) for c in (76,140,108)))
  draw.text((cx-70,y+210),'Side A' if j==0 else 'Side B',font=small,fill='#54665b')
im.save(out/'comparison.png')
report=dict(reference=cal,baseline_dimensions_mm=base.extents.tolist(),sources=sources,method='Separate empirical percentiles for hand length and breadth; synthetic profiles, not individual people. Palm length = hand length × 105/190; not a measured ANSUR palm percentile. Existing app landmark definitions may differ from ANSUR. Physical fit unvalidated.',results=results)
(out/'results.json').write_text(json.dumps(report,indent=2))
print(json.dumps(results,indent=2))
