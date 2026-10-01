"""Noninteractive development benchmark runner using the production agent.

Inputs must already be public SolverTasks and clean checkouts. Images must be
prepared separately, with /testbed replaced by a symlink to /workspace.
"""

import argparse
import asyncio
from contextlib import aclosing
import hashlib
import json
from pathlib import Path
import subprocess
import time

from langchain_core.messages import HumanMessage, message_to_dict
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from cli_rb.runtime.agent import build_agent, astream_turn, TextDelta, ContextUpdate, reply_status
from cli_rb.evaluation.tasks import select_tasks, export_prediction
from cli_rb.interface.cli import disable_remote_tracing
from cli_rb.workspace.coding_backend import DockerCodingBackend
from cli_rb.config import ModelConfig
from cli_rb.workspace.docker_backend import prepare_image
from cli_rb.runtime.persistence import SessionStore
from cli_rb.workspace.snapshot import create_snapshot


class BenchmarkBackend(DockerCodingBackend):
    """Explicit dev-eval resources; interactive defaults remain unchanged."""

    def command(self, name):
        args = super().command(name)
        args[args.index('--memory=256m')] = '--memory=2g'
        args[args.index('--memory-swap=256m')] = '--memory-swap=2g'
        args[args.index('--pids-limit=32')] = '--pids-limit=128'
        index = args.index(self.image)
        args[index:index] = ['--env', 'CLI_RB_COMMAND_PATH=/opt/miniconda3/envs/testbed/bin:/usr/local/bin:/usr/bin:/bin']
        args[args.index(self.image) + 1] = '/opt/miniconda3/bin/python'
        return args


def verify_checkout(repo, task):
    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args], check=True,
                              capture_output=True, timeout=30).stdout.decode().strip()
    if git('rev-parse', 'HEAD') != task.base_commit:
        raise ValueError('仓库 HEAD 与题目 base_commit 不一致。')
    if git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('评测必须从无修改的基线开始。')


async def run_task(task, repo, image, output, config, *, seconds=600,
                   max_total_bytes=64 * 1024 * 1024):
    disable_remote_tracing()
    verify_checkout(repo, task)
    output.mkdir(parents=True, exist_ok=False)
    docker, image = prepare_image(image)
    # Benchmark fixtures are part of the task environment, including large DICOM files.
    # The manifest carries the total budget to patch export and verification.
    snapshot = create_snapshot(repo, output / 'sessions', max_file_bytes=64 * 1024 * 1024,
                               max_total_bytes=max_total_bytes, normalize_git_eol=True)
    if any(reason.startswith('larger than') for reason in snapshot.manifest['excluded'].values()):
        raise ValueError('评测资源超过副本大小限制；不能用缺少资源的副本求解。')
    backend = BenchmarkBackend(snapshot, docker, image)
    metadata = json.loads((snapshot.session / 'container.json').read_text())
    metadata.update(memory='2g', pids_limit=128, purpose='development benchmark')
    variant = subprocess.run(
        [docker, 'image', 'inspect', image, '--format', '{{index .Config.Labels "cli-rb.evaluation-variant"}}'],
        capture_output=True, text=True, check=True, timeout=15,
    ).stdout.strip()
    metadata['evaluation_variant'] = variant or 'original-prepared-environment'
    (snapshot.session / 'container.json').write_text(json.dumps(metadata, indent=2))
    # Preparation probe is independent of solver budget and contains no gold data.
    probe = json.loads(backend.run_command('python --version', 30))
    if probe['exit_code'] != 0 or probe['timed_out']:
        raise RuntimeError('评测 Python 环境探针失败。')
    prompt = (f'Fix the following issue in the repository. Inspect code, implement the fix, '
              f'and run relevant existing tests or your own reproduction. Dependencies are '
              f'already installed; use python / pytest offline.\n\n{task.problem_statement}')
    started = time.monotonic()
    status, error = 'ended', None
    last_message = None
    model_reply_count = 0
    with (output / 'events.jsonl').open('x', encoding='utf-8') as events:
        with SessionStore(snapshot.session, create=True, model_config={
            'base_url': config.base_url, 'model': config.model,
        }) as store:
            async with AsyncSqliteSaver.from_conn_string(str(snapshot.session / 'checkpoints.sqlite')) as saver:
                agent = build_agent(config.build_model(), backend, coding=True,
                                    task_timeout_seconds=seconds, checkpointer=saver, session=store)
                try:
                    async with aclosing(astream_turn(agent, [HumanMessage(content=prompt)])) as stream:
                        async for event in stream:
                            if isinstance(event, TextDelta):
                                continue
                            if not isinstance(event, ContextUpdate):
                                last_message = event
                                model_reply_count += event.type == 'ai'
                            data = ({'context_estimate': event.estimate} if isinstance(event, ContextUpdate)
                                    else message_to_dict(event))
                            events.write(json.dumps({'elapsed': time.monotonic() - started, 'event': data},
                                                    ensure_ascii=False) + '\n')
                            events.flush()
                except Exception as exc:
                    status, error = 'interrupted', {'type': type(exc).__name__, 'message': str(exc)}
                finally:
                    await asyncio.to_thread(backend.cancel_all)
    patch = output / 'model.patch'
    patch.write_bytes(backend.patches.diff().encode('utf-8'))
    export_prediction(task, config.model, patch, output / 'prediction.jsonl')
    result = dict(instance_id=task.instance_id, status=status, error=error,
                  final_reply_status=reply_status(last_message),
                  original_reply_status=(last_message.response_metadata.get('original_reply_status')
                                         if last_message else None),
                  model_reply_count_including_recovery=model_reply_count, task_success=None,
                  evaluation_variant=metadata['evaluation_variant'],
                  elapsed_seconds=time.monotonic() - started, session=str(snapshot.session),
                  image=image, base_commit=task.base_commit, command_evidence=backend.command_evidence,
                  patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
                  patch_bytes=patch.stat().st_size, official_score=None)
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--instance-id', required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    task, = select_tasks(args.tasks, [args.instance_id])
    result = asyncio.run(run_task(task, args.repo, args.image, args.output, ModelConfig.from_env()))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
