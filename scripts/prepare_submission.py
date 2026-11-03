"""Assemble a private user-upload kit without publishing or exporting huge layers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True, help='Docker Hub namespace/repository; kept only in ignored private output')
    parser.add_argument('--tag', default='amd-match2-v1')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9_.-]*', args.repository):
        raise ValueError('Expected Docker Hub namespace/repository')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_.-]{0,127}', args.tag):
        raise ValueError('Invalid tag')
    output = ROOT / 'output/amd-match2-upload'
    output.mkdir(parents=True, exist_ok=True)
    evidence = ROOT / 'evidence/competition'
    image = json.loads((evidence / 'image-verification.json').read_text(encoding='utf-8'))
    if not image['image_checks_passed']:
        raise RuntimeError('Local image checks have not passed')
    source = ROOT / 'output/visionforge-amd-source.zip'
    with zipfile.ZipFile(source) as z:
        if z.testzip() is not None:
            raise RuntimeError('Source ZIP failed CRC check')
        manifest = json.loads(z.read('SOURCE-MANIFEST.json'))
        for name, row in manifest['files'].items():
            data = z.read(name)
            if hashlib.sha256(data).hexdigest() != row['sha256']:
                raise RuntimeError('Source ZIP payload mismatch: ' + name)
            if args.repository.encode().lower() in data.lower():
                raise RuntimeError('Public source ZIP contains the real private submission reference: ' + name)
        for name, expected in image['container_checks']['source_sha256'].items():
            if hashlib.sha256(z.read('competition/' + name)).hexdigest() != expected:
                raise RuntimeError('Source archive differs from the reviewed Docker runtime: ' + name)
        if hashlib.sha256(z.read('competition/model-spec.json')).hexdigest() != image['container_checks']['model_spec_sha256']:
            raise RuntimeError('Source model spec differs from the reviewed Docker image')
    files = {}
    def put(name, content):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        files[name] = path
    def copy(name, origin):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, path)
        files[name] = path
    copy('publish.py', ROOT / 'scripts/submission_upload.py')
    copy('source/visionforge-amd-source.zip', source)
    copy('source/visionforge-amd-source.zip.sha256', source.with_suffix('.zip.sha256'))
    for name in ('image-verification.json', 'release-status-20261101.json', 'regression-tests-20261101.txt',
                 'packet-check-20261101.txt', 'brief-contract-20261101.json'):
        copy('evidence/' + name, evidence / name)
    for name in ('notebook-acceptance-release.json', 'additional-final.json'):
        copy('evidence/' + name, evidence / 'gpu-release-20261101/evidence/competition' / name)
    copy('evidence/amd-release-20261101.png', evidence / 'amd-release-20261101.png')
    artifact = {'created_at': datetime.now(timezone.utc).isoformat(), 'kit_status': 'READY_FOR_USER_UPLOAD',
                'publication': 'NOT_RUN', 'submission_ready': False,
                'local_image': image['image'], 'image_id': image['image_id'],
                'base_diff_ids': image['base_diff_ids'], 'size_limit_bytes': 60 * 1024**3,
                'docker_inspect_size_bytes': image['uncompressed_bytes'], 'size_method': image['size_method'],
                'repository': args.repository, 'tag': args.tag,
                'source_archive': 'source/visionforge-amd-source.zip', 'source_archive_sha256': sha(source),
                'docker_executable_hint': r'C:\Users\nural\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe',
                'amd_container_inference': 'NOT_RUN', 'organizer_execution': 'NOT_RUN',
                'documented_contract': 'Brief pages 5-6: calls in an already-running container; load model once'}
    put('artifact.json', json.dumps(artifact, ensure_ascii=False, indent=2) + '\n')
    python = ROOT / '.venv/Scripts/python.exe'
    for name, flag in [('01-CHECK.cmd', ''), ('02-UPLOAD.cmd', ' --upload')]:
        put(name, '@echo off\nsetlocal\nset "VF_PY=' + str(python) + '"\n'
            'if not exist "%VF_PY%" set "VF_PY=python"\n'
            '"%VF_PY%" "%~dp0publish.py"' + flag + '\n'
            'set "VF_EXIT=%ERRORLEVEL%"\necho.\npause\nexit /b %VF_EXIT%\n')
    put('00-START-HERE.txt', '''VISIONFORGE AI — MATCH 2 — ПРИВАТНЫЙ КОМПЛЕКТ ПОДАЧИ

На этом ПК проверенный Docker-образ уже существует. Повторная сборка, отдельная
AMD VM и большой docker save/tar не нужны. ZIP исходников не содержит веса/образ;
публикуется непосредственно существующий image из Docker Desktop.

1. При желании запустите 01-CHECK.cmd: только локальные проверки, без публикации.
2. Запустите 02-UPLOAD.cmd. ЭТО ПУБЛИКУЕТ ОБРАЗ в Docker Hub из artifact.json.
   Используется существующий вход Docker. Если он истёк, выполните docker login
   в терминале и войдите самостоятельно, затем повторите запуск. Токены не нужны
   в чате/файлах этого комплекта. Если Hub требует репозиторий, создайте его как
   Public с именем из artifact.json в своём namespace. Пароль в команды не вставляйте.
3. После успешной выгрузки откройте RESULTS-LOCATION.txt. В указанной папке:
   IMAGE_REFERENCE.txt — единственная обязательная ссылка для поля образа;
   SUBMIT-PRIVATE.txt — готовый текст, если форма просит описание;
   publication.json — фактический digest и результат анонимной проверки.
4. Вставьте ссылку в приватное поле Match 2 и отправьте форму самостоятельно.

Успех загрузки означает push + чтение manifest без credentials + настоящий
анонимный docker pull по digest. Pull использует уже имеющиеся локальные слои;
полный cold pull на чистом другом компьютере отдельно не проверяется.
При ошибке push или анонимных проверок новые IMAGE_REFERENCE/SUBMIT не создаются.
Журнал каждой попытки — results/. Большой образ может выгружаться долго; прогресс
слоёв показывается в окне. Не закрывайте окно до результата проверки.
Если тег уже занят другим image, загрузчик остановится и не перезапишет его.

ЧЕСТНЫЕ ГРАНИЦЫ
Прямой ROCm: 10/10 основных и 7/7 дополнительных СОБСТВЕННЫХ синтетических кейсов;
16.01 с startup, 3.57 с десять CLI, 4.790 GiB peak на последнем прогоне.
Docker build/base layers/size/provenance/offline processor/no-GPU rejection PASS.
GPU-инференс ВНУТРИ Docker на AMD ещё NOT_RUN. Это не обещание скрытой точности.
Brief p.5-6 прямо описывает уже работающий контейнер и однократную загрузку модели;
это соответствует постоянному worker. Реальный запуск grader организаторов не проверен.
Match 2 сейчас открыт для баллов; призовой этап закрыт по странице конкурса.

НЕ ПУБЛИКУЙТЕ ЭТОТ КОМПЛЕКТ ИЛИ results/ В ОТКРЫТОМ РЕПОЗИТОРИИ.
Brief p.8 требует публичный pull образа, но запрещает рекламировать/коммитить
его registry reference в публичные исходники. Ссылка нужна только для сабмита.
Если нужны исходники, source/visionforge-amd-source.zip не содержит реального
registry target; образ и модель туда не включены. Дополнительный GitHub/video
в приложенном brief не требуются; если ваша форма просит их, заполните её фактические поля.
''')
    put('submission-draft.txt', '''PRIVATE CHALLENGE IMAGE FIELD — paste only this line:
{{IMAGE_REFERENCE}}

Optional project title:
VisionForge AI — Offline Multilingual OCR on AMD ROCm

Optional short description:
Offline ROCm OCR for US/Chinese registration plates and road signs using a resident Qwen3-VL-2B model, packaged on the mandated AMD PyTorch base.

Optional technical description:
The image preserves the mandated ROCm base layers and includes pinned public Qwen3-VL-2B-Instruct weights (Apache-2.0). Its stock CMD loads and warms one FP16 model on the AMD GPU; each standard-library app.py invocation uses a local Unix socket and writes the required JSON text output. PNG, JPEG and TIFF are supported. Full-scene input is retained with an optional supplementary contrast view. US jurisdiction banners are excluded by instruction; Chinese province characters remain part of the registration. Runtime needs no model downloads or external inference APIs.

Validation disclosure:
The direct ROCm implementation passed 10 main and 7 additional self-generated synthetic cases on AMD Radeon Pro W7900D. Latest run: 16.01 s startup, 3.57 s for ten CLI calls, 4.790 GiB observed peak. Local Docker checks passed base-layer identity, size upper bound, code/model metadata/weight identity, offline processor and rejection without AMD GPU. GPU inference inside this exact Docker image has not been tested. These synthetic results do not establish hidden-set accuracy. Registry access results are recorded separately in publication.json after upload.

Keep the image reference out of public source repositories and promotional descriptions (brief p.8).
''')
    checksum = {'created_at': datetime.now(timezone.utc).isoformat(), 'private_kit': True,
                'files_sha256': {name: sha(path) for name, path in sorted(files.items())}}
    manifest_path = output / 'bundle-manifest.json'
    manifest_path.write_text(json.dumps(checksum, indent=2) + '\n', encoding='utf-8')
    files['bundle-manifest.json'] = manifest_path
    archive = ROOT / 'output/visionforge-amd-upload-kit.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, path in sorted(files.items()):
            z.write(path, 'amd-match2-upload/' + name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError('Upload kit ZIP failed CRC')
    archive.with_suffix('.zip.sha256').write_text(sha(archive) + '  ' + archive.name + '\n', encoding='utf-8')
    print(json.dumps({'folder': str(output), 'archive': str(archive), 'bytes': archive.stat().st_size,
                      'sha256': sha(archive), 'publication': 'NOT_RUN'}, indent=2))


if __name__ == '__main__':
    main()
