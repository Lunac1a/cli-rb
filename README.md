# CLI Rb

自然语言 CLI 编码 Agent，按 [实施计划](docs/PLAN.md) 逐块开发。

**当前状态：阶段 1 / 第一块——环境检查与只读工具往返。不是完整编码助手。**
已接入 LangChain `create_agent`、本地 OpenAI-compatible 模型适配、终端对话、
内存文件分段读取、模拟演示及自动化测试。现阶段没有真实仓库访问、代码修改、
命令执行、Docker 沙箱、补丁导出、SQLite 恢复或 SWE-bench 评分。

## 运行

需要 Python 3.12 和 uv，在项目目录执行：

```powershell
uv sync --locked
uv run cli-rb doctor
uv run cli-rb demo
uv run pytest -q
```

`doctor` 只检查环境；某项未就绪时退出码为 1。它不会安装 Docker、配置 WSL、
下载模型或启动服务。模型列表可达不等于模型能力验收通过。
未配置 `CLI_RB_MODEL` 时会列出模型并报告待选择，不把 embedding 模型自动当作聊天模型。

`demo` 是明确标注的脚本模型演示，无网络请求，也不能读写磁盘。它使用真实的
LangChain Agent 和工具调度，展示结构化调用及工具返回；不代表真实模型已验证。

## 连接真实本地模型

计划候选：**Qwen3.5-9B / GGUF / Q4_K_M**。在 LM Studio 下载并加载，
上下文长度先设为 **16384**，开启本地服务。
[下载来源](https://huggingface.co/lmstudio-community/Qwen3.5-9B-GGUF/tree/main)。

```powershell
uv run cli-rb doctor
# 将下面占位文本换成 doctor 输出的实际模型 ID。
$env:CLI_RB_MODEL = '实际模型 ID'
uv run cli-rb smoke
uv run cli-rb
```

默认地址 `http://localhost:1234/v1`，可用 `CLI_RB_BASE_URL` 指定其他本机地址。
如果服务开启鉴权，用 `CLI_RB_API_KEY` 提供密钥；不要把密钥写入项目文件。
第一块只接受 loopback 地址，不自动选择模型、不回退付费服务。云端接入后续实现。

默认启动进入对话；未设置模型 ID 时会询问。Enter 换行，Alt+Enter 提交，
Ctrl+D 或 Ctrl+C 结束。只有演示文件 `README.md` 可读；它是内存中的两行文字，
不是本项目的 README。事件会展示工具请求、工具结果和完整回答；逐 token 流式尚未实现。
当前历史只存在内存中，Ctrl+C 结束整个会话，没有检查点恢复承诺。

`smoke` 要求真实模型成功读取演示文件后再回答，失败返回非零退出码。
通过只说明结构化工具往返成立；16K 上下文稳定性、显存、流式输出和真实代码任务需另验收。
当前每次请求最多输出 2048 tokens、超时 120 秒、接口最多重试一次；
Agent 使用保守的图步数上限防止演示循环。计划中的 50/100 调用计数和 10 分钟总预算尚未实现。

## 阅读路线

- [第一块机制与学习记录](docs/learning/01-tool-roundtrip.md)
- [架构决策](docs/architecture/0001-first-slice.md)
- [验收证据与剩余工作](docs/STATUS.md)

源码入口 `src/cli_rb/cli.py`；框架循环接线在 `src/cli_rb/agent.py`，
工具和后端边界在 `src/cli_rb/workspace.py`。原有 PyCharm 示例 `main.py` 保持不变。
