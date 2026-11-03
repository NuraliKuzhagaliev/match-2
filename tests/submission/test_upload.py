"""Publisher protocol tests with a fake Docker CLI: never perform a real push."""
import json
from pathlib import Path
import subprocess
import pytest
from scripts import submission_upload as m

ID = 'sha256:' + 'a' * 64


@pytest.fixture
def kit(tmp_path):
    (tmp_path / 'source.zip').write_bytes(b'unit-test source payload')
    (tmp_path / 'submission-draft.txt').write_text('{{IMAGE_REFERENCE}}\nGPU inside Docker: NOT_RUN', encoding='utf-8')
    artifact = {'repository': 'example/ocr-test', 'tag': 'test-v1', 'local_image': 'reviewed:local',
                'image_id': ID, 'base_diff_ids': ['base-a', 'base-b'], 'size_limit_bytes': 1000,
                'source_archive': 'source.zip', 'source_archive_sha256': m.digest_file(tmp_path / 'source.zip')}
    m.write_json(tmp_path / 'artifact.json', artifact)
    m.write_json(tmp_path / 'bundle-manifest.json', {'files_sha256': {
        p.name: m.digest_file(p) for p in tmp_path.iterdir() if p.is_file()}})
    return tmp_path


class FakeDocker:
    def __init__(self, *, image_id=ID, remote=None, push_code=0, push_digest=ID, pull_code=0, manifest_code=0):
        self.image_id, self.remote = image_id, remote
        self.push_code, self.push_digest = push_code, push_digest
        self.pull_code, self.manifest_code = pull_code, manifest_code
        self.calls = []

    def read(self, args):
        self.calls.append(args)
        if args[:2] == ['image', 'inspect']:
            return subprocess.CompletedProcess(args, 0, json.dumps([{'Id': self.image_id, 'Size': 900,
                'RootFS': {'Layers': ['base-a', 'base-b', 'app-layer']}}]), '')
        if args[:3] == ['buildx', 'imagetools', 'inspect']:
            if self.remote:
                return subprocess.CompletedProcess(args, 0, 'Digest: ' + self.remote + '\n', '')
            return subprocess.CompletedProcess(args, 1, '', 'not found')
        if 'manifest' in args:
            return subprocess.CompletedProcess(args, self.manifest_code, '{"schemaVersion":2}', 'denied' if self.manifest_code else '')
        if args[:2] == ['context', 'inspect']:
            return subprocess.CompletedProcess(args, 0, 'npipe:////./pipe/dockerDesktopLinuxEngine\n', '')
        if args[0] == 'tag':
            return subprocess.CompletedProcess(args, 0, '', '')
        raise AssertionError(args)

    def live(self, args, log):
        self.calls.append(args)
        if args[0] == 'push':
            result = subprocess.CompletedProcess(args, self.push_code, 'test-v1: digest: ' + self.push_digest + ' size: 856\n', '')
        else:
            assert '--config' in args and '--host' in args and 'pull' in args
            result = subprocess.CompletedProcess(args, self.pull_code, 'pull result\n', '')
        log.write_text(result.stdout, encoding='utf-8')
        return result


def test_prepare_only_never_tags_pushes_or_pulls(kit):
    cli = FakeDocker()
    assert m.prepare(kit, cli)['image_id'] == ID
    assert cli.calls == [['image', 'inspect', 'reviewed:local']]
    assert not (kit / 'results').exists()


def test_tampered_bundle_stops_before_docker(kit):
    (kit / 'source.zip').write_bytes(b'changed')
    cli = FakeDocker()
    with pytest.raises(m.UploadError, match='changed or missing'):
        m.prepare(kit, cli)
    assert cli.calls == []


def test_different_local_image_is_rejected(kit):
    cli = FakeDocker(image_id='sha256:' + 'b' * 64)
    with pytest.raises(m.UploadError, match='ID differs'):
        m.publish(kit, cli)
    assert len(cli.calls) == 1


def test_existing_different_remote_tag_cannot_be_overwritten(kit):
    cli = FakeDocker(remote='sha256:' + 'c' * 64)
    with pytest.raises(m.UploadError, match='refusing to overwrite'):
        m.publish(kit, cli)
    assert not any(c[0] in ('tag', 'push') for c in cli.calls)
    assert not list(kit.glob('results/*/IMAGE_REFERENCE.txt'))


@pytest.mark.parametrize('options', [dict(push_code=1), dict(push_digest='sha256:' + 'f' * 64),
                                    dict(manifest_code=1), dict(pull_code=1)])
def test_failed_publication_or_anonymous_check_never_emits_submit_reference(kit, options):
    cli = FakeDocker(**options)
    with pytest.raises(m.UploadError):
        m.publish(kit, cli)
    reports = list(kit.glob('results/*/publication.json'))
    assert len(reports) == 1 and json.loads(reports[0].read_text())['state'] == 'FAILED'
    assert 'reference' not in json.loads(reports[0].read_text())
    assert not list(kit.glob('results/*/IMAGE_REFERENCE.txt'))
    assert not (kit / 'RESULTS-LOCATION.txt').exists()


def test_verified_upload_emits_only_pinned_reference_and_keeps_gpu_limit_visible(kit):
    cli = FakeDocker()
    report = m.publish(kit, cli)
    assert report['anonymous_pull'] == report['anonymous_manifest'] == 'PASS'
    assert report['amd_container_inference'] == 'NOT_RUN' and report['submission_ready'] is False
    run = Path((kit / 'RESULTS-LOCATION.txt').read_text().strip())
    assert (run / 'IMAGE_REFERENCE.txt').read_text().strip() == 'docker.io/example/ocr-test@' + ID
    assert '{{IMAGE_REFERENCE}}' not in (run / 'SUBMIT-PRIVATE.txt').read_text()
    anonymous = [c for c in cli.calls if c[0] == '--config']
    assert len(anonymous) == 2 and anonymous[0][1] == anonymous[1][1]
    assert json.loads((Path(anonymous[0][1]) / 'config.json').read_text()) == {
        'auths': {'https://index.docker.io/v1/': {}}, 'credsStore': ''}


def test_same_remote_digest_rechecks_access_without_repush(kit):
    cli = FakeDocker(remote=ID)
    assert m.publish(kit, cli)['anonymous_pull'] == 'PASS'
    assert not any(c[0] in ('tag', 'push') for c in cli.calls)


def test_anonymous_commands_strip_env_credentials_without_changing_global_environment(monkeypatch):
    import os
    monkeypatch.setenv('DOCKER_AUTH_CONFIG', 'fake-unit-test-credential')
    monkeypatch.setenv('DOCKER_CONTEXT', 'some-other-context')
    env = m.Docker.environment(['--config', 'isolated', 'manifest', 'inspect', 'example/ocr-test'])
    assert 'DOCKER_AUTH_CONFIG' not in env and 'DOCKER_CONTEXT' not in env
    assert os.environ['DOCKER_AUTH_CONFIG'] == 'fake-unit-test-credential'
    assert m.Docker.environment(['image', 'inspect', 'reviewed:local'])['DOCKER_AUTH_CONFIG'] == 'fake-unit-test-credential'


def test_failed_retry_invalidates_previous_pointer_but_keeps_historical_proof(kit):
    m.publish(kit, FakeDocker())
    old = Path((kit / 'RESULTS-LOCATION.txt').read_text().strip())
    with pytest.raises(m.UploadError):
        m.publish(kit, FakeDocker(remote=ID, pull_code=1))
    assert not (kit / 'RESULTS-LOCATION.txt').exists()
    assert (old / 'IMAGE_REFERENCE.txt').is_file()
    failed = [json.loads(p.read_text()) for p in kit.glob('results/*/publication.json') if p.parent != old]
    assert len(failed) == 1 and failed[0]['state'] == 'FAILED' and 'reference' not in failed[0]


def test_failed_local_preflight_also_invalidates_previous_upload_pointer(kit):
    m.publish(kit, FakeDocker())
    (kit / 'source.zip').write_bytes(b'changed before retry')
    with pytest.raises(m.UploadError):
        m.publish(kit, FakeDocker())
    assert not (kit / 'RESULTS-LOCATION.txt').exists()
