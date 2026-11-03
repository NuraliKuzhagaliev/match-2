"""Contract tests, not substitutes for real ROCm inference acceptance."""
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import socket
import struct
import threading
import pytest
from PIL import Image
from competition import app
from competition.contract import (BASE_IMAGE, BASE_DIGEST, MAX_FRAME, MAX_VRAM_BYTES,
    atomic_json, normalize_for_score, observed_peak_bytes, sampler_coverage, receive, send, transcription, hash_file)
from competition.engine import RocmOCR, require_rocm, load_image, PROMPT
from competition.text_crop import frame_text, text_views
from competition.validate import image_gates, load_cases, validate

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('a,b',[
    ('7-abc.123','7ABC123'),('京A·12345','京A12345'),('ROAD\nWORK\tAHEAD','ROAD WORK AHEAD'),
    ('沪B_88888','沪B88888'),('a\u00a0b','AB')])
def test_only_official_normalization(a,b):
    assert normalize_for_score(a)==normalize_for_score(b)
    assert normalize_for_score('0') != normalize_for_score('O')
    assert normalize_for_score('1') != normalize_for_score('I')
    assert normalize_for_score('京A12345') != normalize_for_score('A12345')


def test_transcription_preserves_actual_characters():
    assert transcription('京A·12345\nROAD WORK')=='京A·12345 ROAD WORK'
    assert transcription('CALIFORNIA 7ABC123')=='CALIFORNIA 7ABC123'
    with pytest.raises(ValueError):transcription(['not a string'])
    with pytest.raises(ValueError):transcription('x'*16385)


def test_rocm_guard_rejects_cpu_cuda_and_missing_gpu():
    def runtime(hip,available=True):
        return SimpleNamespace(version=SimpleNamespace(hip=hip),cuda=SimpleNamespace(is_available=lambda:available,device_count=lambda:1))
    with pytest.raises(RuntimeError):require_rocm(runtime(None,False))
    with pytest.raises(RuntimeError):require_rocm(runtime(None,True))
    with pytest.raises(RuntimeError):require_rocm(runtime('10.0',False))
    require_rocm(runtime('10.0',True))  # Only guard logic; not a hardware assertion.


@pytest.mark.parametrize('fmt,suffix',[('PNG','.png'),('JPEG','.jpg'),('TIFF','.tiff')])
def test_decoder_real_content_and_dimensions(tmp_path,fmt,suffix):
    path=tmp_path/('actual'+suffix)
    Image.new('RGB',(2400,1200),'white').save(path,format=fmt)
    decoded=load_image(path,786432)
    assert decoded.mode=='RGB' and decoded.width*decoded.height<=786432
    assert abs(decoded.width/decoded.height-2)<.02


def test_decoder_matches_content_instead_of_filename(tmp_path):
    valid=tmp_path/'renamed.bin';Image.new('RGB',(60,60)).save(valid,format='PNG')
    assert load_image(valid,786432).size==(60,60)
    fake=tmp_path/'wrong.png';fake.write_text('not an image')
    with pytest.raises(OSError):load_image(fake,786432)
    webp=tmp_path/'wrong.webp';Image.new('RGB',(60,60)).save(webp,format='WEBP')
    with pytest.raises(ValueError):load_image(webp,786432)


def test_text_framing_keeps_all_rows_and_rejects_blank_or_wide_inputs():
    from PIL import ImageDraw, ImageFont
    blank=Image.new('RGB',(800,800),'#e9ae42')
    assert frame_text(blank).size==blank.size
    wide=Image.new('RGB',(800,250),'white')
    assert frame_text(wide) is wide
    image=blank.copy();draw=ImageDraw.Draw(image);font=ImageFont.load_default(size=70)
    for y,text in [(150,'FIRST'),(300,'SECOND'),(450,'THIRD')]:
        draw.text((150,y),text,font=font,fill='black')
    cropped=frame_text(image)
    assert cropped.width<image.width and cropped.height<image.height
    # Dark pixels from every row survive framing; no sign-border pixels exist here.
    assert sum(v<=64 for v in cropped.convert('L').tobytes())==sum(v<=64 for v in image.convert('L').tobytes())


def test_supplementary_view_never_erases_mixed_contrast_full_scene():
    from PIL import ImageDraw, ImageFont
    image=Image.new('RGB',(420,420),'white');draw=ImageDraw.Draw(image);font=ImageFont.load_default(size=42)
    draw.text((60,60),'UPPER',font=font,fill='black')
    draw.text((60,300),'TEXT',font=font,fill=(100,100,100))
    views=text_views(image)
    assert views[0] is image and views[0].size==(420,420)
    assert sum(v==100 for v in views[0].convert('L').tobytes())>500


def test_sampler_cannot_pass_with_one_sample_or_missing_inference_coverage():
    assert not sampler_coverage([{'monotonic_seconds':1}],2,3)
    assert not sampler_coverage([{'monotonic_seconds':1},{'monotonic_seconds':4}],5,10)
    assert not sampler_coverage([{'monotonic_seconds':1},{'monotonic_seconds':8}],2,8)
    assert sampler_coverage([{'monotonic_seconds':1},{'monotonic_seconds':4},{'monotonic_seconds':7}],3,9)


@pytest.mark.parametrize('size', [(3000,600),(600,3000)])
def test_area_cap_preserves_wide_and_tall_image_details(tmp_path,size):
    path=tmp_path/'wide.png';Image.new('RGB',size,'white').save(path)
    decoded=load_image(path,786432)
    assert 780000 <= decoded.width*decoded.height <= 786432
    assert abs(decoded.width/decoded.height-size[0]/size[1])<.02


def test_cold_warmup_uses_startup_budget_but_normal_images_remain_strict(monkeypatch):
    # Explicit runtime double exercises timeout policy, not model/GPU correctness.
    import competition.engine as module
    engine=RocmOCR.__new__(RocmOCR)
    engine.spec={'max_new_tokens':192}
    engine.torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None,
        reset_peak_memory_stats=lambda _:None),inference_mode=nullcontext)
    class Batch(dict):
        def to(self,_):return self
    batch=Batch(input_ids=SimpleNamespace(shape=(1,5)))
    engine.processor=SimpleNamespace(apply_chat_template=lambda *a,**kw:batch,
                                    batch_decode=lambda *a,**kw:['text'])
    observed=[]
    class Result:
        def __getitem__(self,_):return [2]
    def generate(**kw):observed.append(kw['max_time']);return Result()
    engine.model=SimpleNamespace(generate=generate,generation_config=SimpleNamespace(eos_token_id=2))
    engine.memory=lambda:{}
    monkeypatch.setattr(module.time,'perf_counter',iter([0.,1.,40.]).__next__)
    assert engine.transcribe(Image.new('RGB',(60,60)),budget_seconds=100)[0]=='text'
    assert observed[-1]==95
    monkeypatch.setattr(module.time,'perf_counter',iter([0.,1.,40.]).__next__)
    with pytest.raises(RuntimeError,match='per-image budget'):
        engine.transcribe(Image.new('RGB',(60,60)))
    assert observed[-1]==22


def test_missing_daemon_cannot_hide_unrun_acceptance_gates(tmp_path,monkeypatch):
    import competition.validate as module
    monkeypatch.setattr(module,'command',lambda *a,**kw:SimpleNamespace(returncode=1,stderr='No Docker daemon'))
    path=tmp_path/'report.json'
    report=validate('candidate:local',ROOT/'data/competition/manifest.json',path)
    gates={g['id']:g['status'] for g in report['gates']}
    assert report['decision']=='not_ready' and gates['docker_daemon']=='BLOCKED'
    assert gates['vram']==gates['startup']==gates['base_layer_identity']=='NOT_RUN'
    assert json.loads(path.read_text())['decision']=='not_ready'


def test_ipc_framing_transfers_unicode_and_detects_truncation():
    a,b=socket.socketpair()
    try:
        message={'request_id':'a','text':'粤C·60983'}
        send(a,message)
        assert receive(b)==message
    finally:a.close();b.close()
    a,b=socket.socketpair()
    a.sendall(struct.pack('!I',100)+b'{}');a.close()
    try:
        with pytest.raises(ConnectionError):receive(b)
    finally:b.close()


def test_ipc_refuses_oversized_frame():
    a,b=socket.socketpair()
    try:
        a.sendall(struct.pack('!I',MAX_FRAME+1))
        with pytest.raises(ValueError):receive(b)
    finally:a.close();b.close()


def test_json_atomicity_and_unicode(tmp_path):
    path=tmp_path/'answer.json';atomic_json(path,{'text':'粤C·60983'})
    assert json.loads(path.read_text(encoding='utf-8'))=={'text':'粤C·60983'}
    assert not list(tmp_path.glob('.visionforge-*'))
    assert len(hash_file(path))==64


def test_client_filename_and_schema_contract(tmp_path,monkeypatch):
    image=tmp_path/'plate_01.tiff';Image.new('RGB',(60,60)).save(image)
    # Deliberate protocol double: this does not test or claim GPU inference.
    monkeypatch.setattr(app,'rpc',lambda request:{'text':'粤C·60983\nAB', 'ok':True})
    assert app.main(['--input-image',str(image),'--output-dir',str(tmp_path/'out')])==0
    output=tmp_path/'out/plate_01_output.json'
    assert json.loads(output.read_text(encoding='utf-8'))=={'text':'粤C·60983 AB'}


def test_failed_repeat_does_not_reuse_an_old_answer(tmp_path,monkeypatch):
    image=tmp_path/'image.jpg';Image.new('RGB',(60,60)).save(image)
    output=tmp_path/'out/image_output.json';atomic_json(output,{'text':'old'})
    def unavailable(_):raise RuntimeError('ROCm worker unavailable')
    monkeypatch.setattr(app,'rpc',unavailable)
    assert app.main(['--input-image',str(image),'--output-dir',str(tmp_path/'out')])==1
    assert not output.exists()


def test_actual_layer_prefix_and_uncompressed_size_gates():
    base={'RootFS':{'Layers':['sha256:a','sha256:b']}}
    assert image_gates(base,{'RootFS':{'Layers':['sha256:a','sha256:b','sha256:c']},'Size':59*1024**3})==(True,True)
    assert image_gates(base,{'RootFS':{'Layers':['sha256:flattened']},'Size':1})==(False,True)
    assert image_gates(base,{'RootFS':{'Layers':['sha256:a','sha256:b']},'Size':61*1024**3})==(True,False)
    assert MAX_VRAM_BYTES==int(48*1.01*1024**3)


def test_peak_accounts_for_driver_observations_between_sampler_ticks():
    memories=[{'reserved_bytes':4400,'driver_used_bytes':5100,'device_total_bytes':48000},
              {'max_allocated_bytes':4600,'max_reserved_bytes':4700}]
    assert observed_peak_bytes(memories,[{'driver_used_bytes':5000}])==5100
    assert observed_peak_bytes([],[])==0


def test_dockerfile_pins_base_and_does_not_copy_studio_or_secrets():
    docker=(ROOT/'competition/Dockerfile').read_text(encoding='utf-8')
    assert 'FROM '+BASE_IMAGE+'@'+BASE_DIGEST in docker
    assert 'COPY competition/app.py /app/app.py' in docker
    assert 'COPY . ' not in docker and '--gpus all' not in docker
    assert 'python3 -m competition.download_model' in docker
    assert 'CMD ["python3", "-m", "competition.worker"]' in docker
    ignores=(ROOT/'.dockerignore').read_text(encoding='utf-8')
    assert '**\n' in ignores and '!competition/*.py' in ignores
    assert 'HF_TOKEN=' not in docker and 'trust_remote_code=True' not in (ROOT/'competition/engine.py').read_text()


def test_synthetic_validation_cases_are_separate_from_inference():
    cases=load_cases(ROOT/'data/competition/manifest.json')
    assert len(cases)==10
    assert {'us_plate','chinese_plate','stop_sign','speed_sign','warning_sign','advisory_sign'}<={c['kind'] for c in cases}
    spec=json.loads((ROOT/'competition/model-spec.json').read_text())
    assert len(spec['revision'])==40 and len(spec['weight_sha256'])==64
    assert spec['weight_bytes']>1024**3
    for case in cases:
        assert hash_file(ROOT/'data/competition'/case['file'])==case['sha256']
    assert 'manifest.json' not in (ROOT/'competition/engine.py').read_text().replace('visionforge-manifest.json','')


def test_gpu_engine_has_no_cpu_fallback_or_padding():
    source=(ROOT/'competition/engine.py').read_text(encoding='utf-8')
    assert 'require_rocm(torch)' in source
    assert 'torch.empty' not in source and 'torch.zeros' not in source
    assert 'confidence' not in source
    assert 'Chinese province' in PROMPT
