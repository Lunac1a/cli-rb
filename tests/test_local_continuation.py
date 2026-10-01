import json

import pytest
from test_local_backend import backend as local_backend
from cli_rb.runtime.persistence import SessionStore, SessionNotReady

backend = local_backend


def test_local_command_receipts_allow_next_turn_and_missing_receipt_still_blocks(backend):
    with SessionStore(backend.snapshot.session, create=True) as store:
        backend.run_command("Write-Output 'first turn'")
        store.require_idle()
        backend.approve = lambda _: False
        result = json.loads(backend.run_command('New-Item -ItemType Directory -Path expense-tracker'))
        assert result['exit_code'] == 0
        assert (backend.snapshot.workspace / 'expense-tracker').is_dir()
        store.require_idle()
        denied = json.loads(backend.run_command('Remove-Item app.py'))
        assert denied['denied']
        store.require_idle()
        journal = next(backend.logs.glob('*-action.json'))
        receipt = journal.with_name(journal.name.replace('-action.json', '.json'))
        receipt.unlink()
        with pytest.raises(SessionNotReady):
            store.require_idle()
