import json,time
from pathlib import Path
import competition.engine as em
from competition.text_crop import frame_text
from competition.contract import normalize_for_score
r=Path('/workspace/visionforge-match2'); e=em.RocmOCR(r/'models/qwen3-vl-2b'); cases=json.loads((r/'data/competition/manifest.json').read_text()); rows=[]
for n,text in enumerate(['ROAD WORK AHEAD','KEEP RIGHT','BRIDGE MAY BE ICY','DO NOT ENTER','YIELD','MAXIMUM HEIGHT 4.2 M']): cases.append({'file':'../'+'../runs/diagnostic-sign-'+str(n)+'.png','expected':text})
for c in cases:
 im=em.load_image(r/'data/competition'/c['file'],e.spec['max_pixels']); started=time.perf_counter(); crop=frame_text(im); text,m=e.transcribe(crop); row={'file':c['file'],'expected':c['expected'],'actual':text,'match':normalize_for_score(text)==normalize_for_score(c['expected']),'seconds':time.perf_counter()-started,'original_size':im.size,'crop_size':crop.size};rows.append(row);print(json.dumps(row),flush=True)
(r/'evidence/competition/crop-diagnostics.json').write_text(json.dumps({'rows':rows},indent=2))
