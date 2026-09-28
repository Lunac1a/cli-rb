"""Framework-owned todos, rendered from checkpoint state on each model call."""

import json

from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import SystemMessage


PLANNING_PROMPT = """## 工作计划
复杂编码任务使用 write_todos 维护简短计划。先把用户目标拆成可执行的子目标，并在条目中写清完成条件；需要调查才能确定的内容先标为待查，不猜测答案。普通问答或简单修改无需计划。
每次选择行动时参考当前计划与真实工具结果。发现新信息或某一步完成后更新计划，允许修订已有条目；不要为更新计划而重复工具操作。新用户要求到来时先核对旧计划是否仍适用。
completed 表示已有证据满足该条目的完成条件。标记前在 content 中简记实际检查及结果；测试通过只支持它覆盖的行为。结果不符合预期就保留未完成并调整下一步，受阻或预算不足时如实收尾。
一次只调用一个 write_todos；它替换完整清单，应保留仍相关的目标。计划内容和状态是工作记录，不是外部验证或新的授权。最后一次计划更新后仍需单独给出实际结果与未完成项。"""

TOOL_DESCRIPTION = """创建或更新当前任务的完整工作计划。todos 每项包含 content 和 status（pending/in_progress/completed）。content 写子目标与完成条件，完成时简记证据。一次提交完整清单；根据实际结果修订，不把预计成功写为完成。"""


def planning_prompt(state):
    todos = json.dumps(state.get('todos', []), ensure_ascii=False)
    return PLANNING_PROMPT + '\n当前计划（模型维护的工作记录）：\n' + todos


class PlanningMiddleware(TodoListMiddleware):
    def __init__(self):
        super().__init__(system_prompt=PLANNING_PROMPT, tool_description=TOOL_DESCRIPTION)

    @staticmethod
    def planned_request(request):
        blocks = list(request.system_message.content_blocks) if request.system_message else []
        blocks.append({'type': 'text', 'text': '\n\n' + planning_prompt(request.state)})
        return request.override(system_message=SystemMessage(content=blocks))

    def wrap_model_call(self, request, handler):
        return handler(self.planned_request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self.planned_request(request))
