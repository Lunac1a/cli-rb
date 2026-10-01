"""Host-side scoring stages; build time is separate from the test allowance."""
from pathlib import Path
from contextlib import contextmanager
import subprocess
from uuid import uuid4
from tempfile import TemporaryDirectory


@contextmanager
def local_httpbin(docker, image):
    """Supply Requests' existing HTTPBIN_URL hook on an internal Docker network."""
    if not (image.startswith('sha256:') or '@sha256:' in image):
        raise ValueError("Pin the httpbin image ID before evaluation")
    name = 'cli-rb-httpbin-' + uuid4().hex
    subprocess.run([docker, 'network', 'create', '--internal', name],
                   check=True, capture_output=True, timeout=30)
    try:
        subprocess.run([docker, 'run', '-d', '--name', name, '--network', name,
                        '--network-alias', 'httpbin', '--memory=256m', '--pids-limit=128',
                        image, 'sh', '-c',
                        "printf '[req]\\ndistinguished_name=dn\\nx509_extensions=ext\\n"
                        "[dn]\\n[ext]\\nsubjectAltName=DNS:httpbin\\n"
                        "basicConstraints=critical,CA:TRUE\\n' >/tmp/cert.cnf; "
                        'openssl req -x509 -nodes -newkey rsa:2048 -days 1 '
                        '-subj /CN=httpbin -config /tmp/cert.cnf '
                        '-keyout /tmp/key.pem -out /tmp/cert.pem && '
                        'touch /tmp/ready && '
                        'gunicorn -b 0.0.0.0:80 httpbin:app -k gevent & '
                        'while [ ! -f /tmp/ready ]; do sleep 0.2; done; '
                        'exec gunicorn -b 0.0.0.0:443 httpbin:app -k gevent '
                        '--certfile /tmp/cert.pem --keyfile /tmp/key.pem'],
                       check=True, capture_output=True, timeout=60)
        subprocess.run([docker, 'exec', name, 'sh', '-c',
                        'while [ ! -f /tmp/ready ]; do sleep 0.2; done'],
                       check=True, capture_output=True, timeout=30)
        with TemporaryDirectory() as directory:
            certificate = Path(directory) / 'httpbin-ca.pem'
            subprocess.run([docker, 'cp', name + ':/tmp/cert.pem', str(certificate)],
                           check=True, capture_output=True, timeout=30)
            yield {'network': name, 'environment': {'HTTPBIN_URL': 'http://httpbin/'},
                   'service_image': image, 'trusted_ca': str(certificate),
                   'evaluation_variant': 'local-httpbin-http-https'}
    finally:
        subprocess.run([docker, 'rm', '-f', name], capture_output=True, timeout=30)
        subprocess.run([docker, 'network', 'rm', name], capture_output=True, timeout=30)


def run_scoring_script(docker, container, script, output, *,
                       setup_seconds=1800, test_seconds=300,
                       use_preinstalled_build_dependencies=False):
    """Use an already created scorer container. Never retry a stage or a solver."""
    marker = "git apply -v - <<"
    setup, separator, tests = script.partition(marker)
    if not separator:
        raise ValueError("Official test patch boundary missing")
    if use_preinstalled_build_dependencies:
        # Prepared official images already contain build dependencies. PEP517's
        # fresh isolated environment would otherwise download them again offline.
        setup = setup.replace('python -m pip install ',
                              'python -m pip install --no-build-isolation --no-deps --no-index ')
    # Each exec starts a new shell: retain the official activation/cwd prologue.
    prologue = script[:script.index("git config")]
    stages = [
        ('setup', 'set -e\n' + setup, setup_seconds),
        ('tests', prologue + separator + tests, test_seconds),
    ]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, content, timeout in stages:
        path = output / (name + '.sh')
        path.write_text(content, encoding='utf-8', newline='\n')
        subprocess.run([docker, 'cp', str(path), container + ':/stage.sh'],
                       check=True, capture_output=True, timeout=30)
        with (output / (name + '.log')).open('wb') as log:
            try:
                process = subprocess.run([docker, 'exec', container, '/bin/bash', '/stage.sh'],
                                         stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired:
                return {**results, 'state': 'evaluation_error', 'phase': name, 'reason': 'timeout'}
        results[name + '_exit_code'] = process.returncode
        if name == 'setup' and process.returncode:
            return {**results, 'state': 'evaluation_error', 'phase': name, 'reason': 'setup_failed'}
    return {**results, 'state': 'ready_for_parser', 'test_log': str(output / 'tests.log')}


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', required=True)
    parser.add_argument('--script', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--setup-seconds', type=int, default=1800)
    parser.add_argument('--test-seconds', type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(run_scoring_script(
        'docker', args.container, args.script.read_text(encoding='utf-8'), args.output,
        setup_seconds=args.setup_seconds, test_seconds=args.test_seconds)))
