"""Build an explicitly adapted SWE-bench environment for legacy nose-style tests.

Creates new scoring/solver image tags and a manifest; never retags official images.
Use both derived images together and report scores as environment-adapted results.
"""

import argparse
import json
from pathlib import Path
import subprocess

from cli_rb.workspace.docker_backend import prepare_image


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
