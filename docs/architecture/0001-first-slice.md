# ADR 0001：第一块只验证模型与工具往返

状态：采用，2026-09-23。

采用计划指定的 Python 3.12、uv、LangChain `create_agent`、Rich、prompt-toolkit、Typer。
依赖实际解析版本在 `uv.lock` 中；不另写模型/工具交替的核心循环。
暂不引入尚未使用的 SQLite checkpointer，阶段 2 再落实持久化。

`build_agent(model, backend)` 接收模型和读取后端，创建工具并交给框架。
CLI 只负责输入和显示，`stream_turn` 把框架消息事件交给 CLI。
演示及真实模型 smoke 共用 Agent 构造函数，仅模型来源不同。

公开依据：
- [LangChain create_agent quickstart](https://docs.langchain.com/oss/python/langchain/quickstart)
- [ChatOpenAI integration](https://docs.langchain.com/oss/python/integrations/chat/openai)

本次只采用这些公开接口依据；未声称复现 Codex 或 Claude Code 的内部实现。
后续涉及沙箱、权限、恢复时，再记录相应公开参考和偏离。

初次环境检查未找到 Docker，WSL 状态返回访问拒绝，LM Studio 本地服务不可达。
用户确认尚未下载模型。因此第一块采用无副作用的内存文件后端，验证结构化往返。
它不是 Docker 替代品；应用没有宿主机 shell 或真实仓库工具，也没有失败后转宿主机执行的路径。

计划分块实施要求：先观察一次往返，再逐步完成仓库问答、修改、测试、导出。
本次未提前建设通用插件、评测系统、多 Agent 或长期记忆。

模拟测试证明接线和错误反馈，不证明真实模型能力或沙箱隔离。
只读工具的路径校验是接口约束，不等同于未来 Docker 的安全边界。
