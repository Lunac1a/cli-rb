"""Framework-owned todos, rendered from checkpoint state on each model call."""

import json

from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import SystemMessage


PLANNING_PROMPT = """## Work plan
Use write_todos to maintain a concise plan for complex coding tasks. Split the request into actionable subgoals with completion conditions. Mark unknowns as requiring investigation rather than guessing. Ordinary questions or simple edits need no plan.
Choose actions from the current plan and actual tool results. Update the plan when new information arrives or a step completes, but do not repeat tool operations merely to update it. Reassess relevance when the user changes the request.
write_todos manages work progress only; do not put verification definitions in todos. Register observable product behavior separately with register_behavior after investigation. Investigation, editing, and running tests are work steps, not product behaviors. Updating progress never changes registered behavior definitions or evidence.
Run necessary checks once. Use bind_verification to link existing evidence and exact returned test nodes to a behavior; no re-execution is needed just to establish a binding. Passing checks support only the selected assertions, not complete task coverage.
completed means evidence satisfies the entry's completion conditions. Briefly record the actual check and result in content before marking it. Passing tests support only covered behaviors. Keep an entry incomplete and revise next steps when results differ from expectations; report blockers or exhausted budgets honestly.
Call write_todos only once at a time. It replaces the full list, so retain relevant goals. Plan contents and statuses are working records, not external verification or new authorization. After the last plan update, separately report actual results and unfinished work."""

TOOL_DESCRIPTION = """Create or update the complete current work plan. Each todo has content and status (pending/in_progress/completed). State the subgoal and completion condition in content, with brief evidence when completed. Submit the full list each time. Revise it from actual results; do not mark anticipated success as completed."""


def planning_prompt(state):
    todos = json.dumps(state.get('todos', []), ensure_ascii=False)
    return PLANNING_PROMPT + '\nCurrent plan (model-maintained working record):\n' + todos


class PlanningMiddleware(TodoListMiddleware):
    def __init__(self, verification=None):
        super().__init__(system_prompt=PLANNING_PROMPT, tool_description=TOOL_DESCRIPTION)
        self.verification = verification

    def planned_request(self, request):
        blocks = list(request.system_message.content_blocks) if request.system_message else []
        blocks.append({'type': 'text', 'text': '\n\n' + planning_prompt(request.state)})
        if self.verification is not None:
            blocks.append({'type': 'text', 'text': '\nRegistered product behaviors (independent of todos):\n' +
                           json.dumps(self.verification.requirements(), ensure_ascii=False)})
        return request.override(system_message=SystemMessage(content=blocks))

    def wrap_model_call(self, request, handler):
        return handler(self.planned_request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self.planned_request(request))
