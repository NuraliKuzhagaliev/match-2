"""User-run Docker Hub publisher. Default execution checks locally and never pushes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid


class UploadError(RuntimeError):
    pass


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


class Docker:
    def __init__(self, executable):
        self.executable = executable

    @staticmethod
    def environment(args):
        env = os.environ.copy()
        if args and args[0] == '--config':
            # Anonymous commands must not inherit environment-based credentials
            # or a context that bypasses the isolated client configuration.
            for name in ('DOCKER_AUTH_CONFIG', 'DOCKER_CUSTOM_HEADERS', 'DOCKER_CONFIG',
                         'DOCKER_CONTEXT', 'DOCKER_HOST'):
                env.pop(name, None)
        return env

    def read(self, args):
        return subprocess.run([self.executable, *args], capture_output=True,
                              text=True, encoding='utf-8', errors='replace', timeout=60,
                              env=self.environment(args))

    def live(self, args, log):
        output = []
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen([self.executable, *args], stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True,
                                       encoding='utf-8', errors='replace', env=self.environment(args))
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    stream.write(line)
                    stream.flush()
                    output.append(line)
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        return subprocess.CompletedProcess(args, code, ''.join(output), '')


def require(result, operation):
    if result.returncode:
        raise UploadError(operation + ': ' + (result.stderr + result.stdout)[-2500:])
    return result.stdout.strip()


def prepare(bundle, docker):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / 'bundle-manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files_sha256'].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or not path.is_file() or digest_file(path) != expected:
            raise UploadError('Bundle file changed or missing: ' + name)
    artifact = json.loads((bundle / 'artifact.json').read_text(encoding='utf-8'))
    repository, tag = artifact['repository'], artifact['tag']
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9_.-]*', repository):
        raise UploadError('Expected a Docker Hub namespace/repository')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_.-]{0,127}', tag):
        raise UploadError('Invalid image tag')
    data = json.loads(require(docker.read(['image', 'inspect', artifact['local_image']]), 'Inspect local image'))[0]
    if data['Id'] != artifact['image_id']:
        raise UploadError('Local image ID differs from the reviewed image; do not upload')
    required = artifact['base_diff_ids']
    if not required or data['RootFS']['Layers'][:len(required)] != required:
        raise UploadError('Mandatory ROCm base layers do not match')
    if not 0 < data['Size'] <= artifact['size_limit_bytes']:
        raise UploadError('Image size exceeds the limit')
    source = bundle / artifact['source_archive']
    if digest_file(source) != artifact['source_archive_sha256']:
        raise UploadError('Source archive SHA-256 mismatch')
    return artifact


def publish(bundle, docker):
    bundle = Path(bundle).resolve()
    # Keep historical run folders, but invalidate the previous success pointer
    # before even the new local preflight: a failed retry must not look current.
    (bundle / 'RESULTS-LOCATION.txt').unlink(missing_ok=True)
    artifact = prepare(bundle, docker)
    target = 'docker.io/' + artifact['repository'] + ':' + artifact['tag']
    run = bundle / 'results' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6])
    run.mkdir(parents=True)
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'state': 'STARTED',
              'target': target, 'image_id': artifact['image_id'], 'publication': 'NOT_RUN',
              'anonymous_manifest': 'NOT_RUN', 'anonymous_pull': 'NOT_RUN', 'amd_container_inference': 'NOT_RUN',
              'submission_ready': False, 'own_synthetic_cases': '17/17 direct ROCm',
              'limitation': 'AMD execution inside this Docker image has not been tested'}
    try:
        remote = docker.read(['buildx', 'imagetools', 'inspect', target])
        already_uploaded = False
        if remote.returncode == 0:
            found = re.search(r'^Digest:\s*(sha256:[a-f0-9]{64})\s*$', remote.stdout, re.MULTILINE)
            if not found or found.group(1) != artifact['image_id']:
                raise UploadError('Remote tag already exists with a different or unverifiable image; refusing to overwrite')
            already_uploaded = True
        elif not any(x in (remote.stdout + remote.stderr).lower() for x in
                     ('not found', 'manifest unknown', 'no such manifest')):
            raise UploadError('Could not check the remote tag. Run docker login and create the Public repository, then retry. ' + remote.stderr[-1200:])
        if not already_uploaded:
            require(docker.read(['tag', artifact['image_id'], target]), 'Tag reviewed image')
            report['publication'] = 'FAIL'
            pushed = docker.live(['push', target], run / 'push.log')
            require(pushed, 'Push failed; run docker login or create the Public repository and retry')
            matches = re.findall(r'\bdigest:\s*(sha256:[a-f0-9]{64})', pushed.stdout)
            if not matches or matches[-1] != artifact['image_id']:
                raise UploadError('Push returned an unexpected digest; do not submit this result')
        digest = artifact['image_id']
        reference = 'docker.io/' + artifact['repository'] + '@' + digest
        report.update(publication='PASS', digest=digest)
        # A nonempty auth map containing an EMPTY Hub entry prevents Docker CLI
        # auto-discovering the platform's credential helper (ContainsAuth check).
        anonymous = run / 'anonymous-docker-config'
        anonymous.mkdir()
        (anonymous / 'config.json').write_text(json.dumps({
            'auths': {'https://index.docker.io/v1/': {}}, 'credsStore': ''}), encoding='utf-8')
        report['anonymous_manifest'] = 'FAIL'
        manifest = json.loads(require(docker.read(['--config', str(anonymous), 'manifest', 'inspect', reference]),
                                      'Anonymous registry manifest check failed; make the repository Public'))
        if not isinstance(manifest, dict) or manifest.get('schemaVersion') != 2:
            raise UploadError('Unexpected anonymous registry manifest')
        report['anonymous_manifest'] = 'PASS'
        endpoint = require(docker.read(['context', 'inspect', '--format', '{{(index .Endpoints "docker").Host}}']), 'Read Docker endpoint')
        if not endpoint.startswith(('npipe://', 'unix://')):
            raise UploadError('Anonymous check requires the local Docker daemon; unsupported endpoint')
        report['anonymous_pull'] = 'FAIL'
        pulled = docker.live(['--config', str(anonymous), '--host', endpoint, 'pull', reference], run / 'anonymous-pull.log')
        require(pulled, 'Anonymous pull failed; set the Docker Hub repository to Public and retry')
        received = json.loads(require(docker.read(['image', 'inspect', reference]), 'Inspect pulled digest'))[0]
        if received['Id'] != artifact['image_id']:
            raise UploadError('Pulled image differs from the reviewed image')
        report.update(state='UPLOADED_AND_ANONYMOUS_PULL_VERIFIED', anonymous_pull='PASS', reference=reference,
                      pull_uses_existing_local_layers=True, cold_clean_machine_pull='NOT_RUN')
        (run / 'IMAGE_REFERENCE.txt').write_text(reference + '\n', encoding='utf-8')
        (run / 'SUBMIT-PRIVATE.txt').write_text(
            (bundle / 'submission-draft.txt').read_text(encoding='utf-8').replace('{{IMAGE_REFERENCE}}', reference), encoding='utf-8')
        write_json(run / 'publication.json', report)
        (bundle / 'RESULTS-LOCATION.txt').write_text(str(run) + '\n', encoding='utf-8')
        print('\nUPLOAD VERIFIED. Paste IMAGE_REFERENCE.txt into the challenge image field.')
        print('AMD execution inside Docker remains NOT_RUN. Keep this reference out of public source repositories.')
        print(str(run))
        return report
    except Exception as exc:
        report.update(state='FAILED', error=type(exc).__name__ + ': ' + str(exc))
        write_json(run / 'publication.json', report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--upload', action='store_true', help='Actually publish the reviewed image to Docker Hub')
    args = parser.parse_args(argv)
    try:
        artifact = json.loads((args.bundle / 'artifact.json').read_text(encoding='utf-8'))
        executable = shutil.which('docker') or artifact['docker_executable_hint']
        docker = Docker(executable)
        if args.upload:
            publish(args.bundle, docker)
        else:
            prepare(args.bundle, docker)
            print('READY_FOR_USER_UPLOAD: exact local image, base layers and bundle hashes PASS. No image was tagged or uploaded.')
        return 0
    except Exception as exc:
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
