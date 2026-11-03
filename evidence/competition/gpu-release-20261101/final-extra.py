from pathlib import Path
import json,hashlib,time
from PIL import Image,ImageDraw,ImageFont
from competition.engine import RocmOCR,load_image
from competition.contract import normalize_for_score
from competition.text_crop import text_views
root=Path.cwd()
engine=RocmOCR(root/'models/qwen3-vl-2b')
rows=[]
expected=['ROAD WORK AHEAD','KEEP RIGHT','BRIDGE MAY BE ICY','DO NOT ENTER','YIELD','MAXIMUM HEIGHT 4.2M']
cases=[(root/f'runs/diagnostic-sign-{i}.png',x) for i,x in enumerate(expected)]
im=Image.new('RGB',(420,420),'white'); d=ImageDraw.Draw(im); f=ImageFont.load_default(size=42)
d.text((60,60),'UPPER',fill='black',font=f); d.text((60,300),'TEXT',fill=(100,100,100),font=f)
p=root/'runs/mixed-contrast.png'; im.save(p); cases.append((p,'UPPER TEXT'))
for p,want in cases:
 image=load_image(p,786432); views=text_views(image)
 got,metrics=engine.transcribe(image)
 row=dict(file=str(p.relative_to(root)),expected=want,actual=got,match=normalize_for_score(got)==normalize_for_score(want),views=[list(v.size) for v in views],sha256=hashlib.sha256(p.read_bytes()).hexdigest(),metrics=metrics)
 rows.append(row); print(json.dumps(row),flush=True)
result=dict(kind='additional_synthetic_regression',source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'competition').glob('*.py')},runtime=engine.describe(),cases=rows,passed=all(r['match'] for r in rows))
(root/'evidence/competition/additional-final.json').write_text(json.dumps(result,indent=2))
print('ADDITIONAL_PASSED',result['passed'],flush=True)
