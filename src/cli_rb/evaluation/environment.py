"""Build an explicitly adapted SWE-bench environment for legacy nose-style tests.

Creates new scoring/solver image tags and a manifest; never retags official images.
Use both derived images together and report scores as environment-adapted results.
"""

import argparse
import json
from pathlib import Path
import subprocess

from cli_rb.workspace.docker_backend import prepare_image


def build_scoring_environment(base_ref, requirements, output, *, timeout=600):
    """Prepare explicitly pinned build tools before spending any solver budget.

    Network access is limited to image preparation. Runtime packages are not
    implicitly upgraded; callers specify every added/changed distribution and
    must preflight the official setup/tests before accepting the new image.
    """
    if '@sha256:' not in base_ref:
        raise ValueError('Use the immutable registry digest of the official image')
    if not requirements or any('==' not in item or '\n' in item for item in requirements):
        raise ValueError('Explicitly pinned build requirements are required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'requirements.txt').write_text('\n'.join(requirements) + '\n', encoding='utf-8')
    (output / 'Dockerfile').write_text(
        f'FROM {base_ref}\nCOPY requirements.txt /tmp/cli-rb-build-requirements.txt\n'
        'RUN /opt/miniconda3/envs/testbed/bin/python -m pip install --no-deps '
        '-r /tmp/cli-rb-build-requirements.txt\n'
        'RUN /opt/miniconda3/envs/testbed/bin/python -m pip freeze '
        '> /opt/cli-rb-scoring-packages.txt\n'
        'LABEL cli-rb.evaluation-variant="prepared-build-tools"\n', encoding='utf-8')
    with (output / 'build.log').open('wb') as log:
        subprocess.run(['docker', 'build', '--iidfile', str(output / 'image-id.txt'), str(output)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    image_id = (output / 'image-id.txt').read_text().strip()
    packages = subprocess.run(['docker', 'run', '--rm', '--network=none', image_id,
                               'cat', '/opt/cli-rb-scoring-packages.txt'],
                              capture_output=True, check=True, timeout=30).stdout
    (output / 'packages.txt').write_bytes(packages)
    manifest = {'evaluation_variant': 'prepared-build-tools', 'base_image_ref': base_ref,
                'scoring_image_id': image_id, 'build_requirements': requirements,
                'preflight': 'not_run', 'official_environment_unchanged': False}
    (output / 'environment.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def build_legacy_environment(base_image, tag_prefix, output):
    docker, base_id = prepare_image(base_image)
    digests = json.loads(subprocess.run(
        [docker, 'image', 'inspect', base_id, '--format', '{{json .RepoDigests}}'],
        check=True, capture_output=True, text=True, timeout=15,
    ).stdout)
    base_ref = digests[0]  # Official pulled image, pinned registry reference for BuildKit.
    output.mkdir(parents=True, exist_ok=False)
    scoring_tag, solver_tag = tag_prefix + '-scoring', tag_prefix + '-solver'
    # pytest 7.0.1 supports nose setup/teardown and Python 3.6+ used by old tasks.
    dockerfile = (
        f'FROM {base_ref} AS scoring\n'
        'RUN /opt/miniconda3/envs/testbed/bin/python -m pip install "pytest==7.0.1"\n'
        'LABEL cli-rb.evaluation-variant="legacy-pytest-7.0.1"\n'
        'FROM scoring AS solver\n'
        'RUN rm -rf /testbed && ln -s /workspace /testbed\n'
        'WORKDIR /workspace\n'
    )
    (output / 'Dockerfile').write_text(dockerfile, encoding='utf-8')
    ids = {}
    for target, tag in [('scoring', scoring_tag), ('solver', solver_tag)]:
        with (output / f'{target}-build.log').open('wb') as log:
            subprocess.run([docker, 'build', '--target', target, '-t', tag, str(output)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        ids[target] = prepare_image(tag)[1]
    manifest = {
        'evaluation_variant': 'legacy-pytest-7.0.1', 'official_environment_unchanged': False,
        'base_image_id': base_id, 'base_image_ref': base_ref, 'pytest_version': '7.0.1',
        'scoring_image': scoring_tag, 'scoring_image_id': ids['scoring'],
        'solver_image': solver_tag, 'solver_image_id': ids['solver'],
        'note': 'Adapted environment; preserve original scores separately. Solver remains offline.',
    }
    (output / 'environment.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-image', required=True)
    parser.add_argument('--tag-prefix', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_legacy_environment(args.base_image, args.tag_prefix, args.output), indent=2))


if __name__ == '__main__':
    main()
