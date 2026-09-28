# CLI Rb

基于 LangChain / LangGraph 的自然语言 Coding Agent，支持 OpenAI-compatible Chat Completions 和 Responses 服务。模型负责调查、规划和修改；程序负责隔离工具、持久化、预算与验证证据核对。

项目关注 Harness 如何维持工作质量。已有执行、恢复和验证流程，但测试覆盖仍由模型选择，不能把正常结束或局部测试通过当作任务解决。当前仍存在回归漏测、验证记录替代不完善、日志解析兼容和大仓库审阅开销问题，尚无完整 SWE-bench 成绩。

## 快速开始

需要 Python 3.12、uv；真实仓库操作另需 Docker Linux 引擎。

```powershell
uv sync --locked
uv run cli-rb demo
docker pull python:3.12-slim-bookworm
Copy-Item .env.example .env.local
```

编辑`.env.local`，填写模型ID、服务地址及必要的密钥，然后启动：

```powershell
$env:CLI_RB_ENV_FILE = '.env.local'
uv run cli-rb doctor
uv run cli-rb
```

只加载显式指定的配置文件，进程环境变量优先。真实配置与密钥由Git忽略，不写入会话模型配置。远程服务要求HTTPS和`CLI_RB_ALLOW_REMOTE=1`；不会自动回退其他服务。

| 配置 | 含义 / 默认值 |
| --- | --- |
| `CLI_RB_BASE_URL` | 服务根地址，默认`http://localhost:1234/v1` |
| `CLI_RB_MODEL` | 必填的服务端模型ID |
| `CLI_RB_API_KEY` | 服务要求的密钥 |
| `CLI_RB_USE_RESPONSES_API` | `1`启用Responses，默认`0` |
| `CLI_RB_REASONING_EFFORT` | 服务支持时设置，例如`medium` |
| `CLI_RB_CONTEXT_WINDOW` | 本地工作窗口，默认`200000`；需匹配模型能力 |
| `CLI_RB_MAX_OUTPUT_TOKENS` | 默认`32768`，同时用于上下文预留 |
| `CLI_RB_REQUEST_TIMEOUT_SECONDS` | 请求等待和流式静默超时，默认`120`秒 |

Responses模式使用`store=false`，保留工具往返需要的加密推理内容，发送当前工作历史，不依赖服务端`previous_response_id`。上下文使用启发式估算，另留输出预算和计数余量。服务商可能不执行客户端输出限制，调用次数与任务时限仍由Harness管理。

`doctor`只检查环境；`demo`使用脚本模型和内存文件，不调用真实服务。`smoke`会请求真实模型验证工具往返，可能产生费用；它不代表编码任务验收。

## 使用与执行边界

启动时选择模型、Git仓库并确认工作区。Enter换行，Alt+Enter提交（或先Esc再Enter），Ctrl+D / Ctrl+C退出。直接输入“解释入口”“修复问题并验证”等要求。

- 默认直接修改所选仓库，普通文件编辑立即生效。需要副本时使用`uv run cli-rb --isolated`。
- 工具支持浅目录、路径匹配、分页搜索、读取、精确编辑与差异查看。命令通过Docker执行，缺少Docker或镜像时失败，不回退宿主机执行。
- 直接模式的普通命令只读；可写命令和删除已有文件需要授权。授权不代表程序已理解命令的全部语义。
- 每条任务默认最多50次模型调用、100次工具调用、30分钟。使用`uv run cli-rb --help`查看入口选项；收尾反馈共享原预算。
- 编码模式最终正文经核对后显示，工具请求和结果仍显示。不展示隐藏推理或未经核对的结束草稿。

## 计划、验证与收尾

复杂任务通过`write_todos`维护计划，随检查点保存并提供给后续模型调用。计划完成状态由模型维护，不是独立验收结果。

编辑Python文件会提示语法问题和原测试定义消失或归属变化；允许合法重构，不自动撤销改动。

`run_verification`保存命令、模型声明的范围、完整日志引用、执行结果与前后工作区指纹。`test`模式识别pytest/unittest汇总；未知格式、零项、失败或超时不会标为通过。`check`模式只记录普通命令成功。调查命令仍可使用`run_command`。

`review_work`检查变更文件和结构风险。模型准备结束时，程序核对本轮验证是否与当前代码一致，必要时最多反馈一次；未解决的问题作为程序附注保留。纯只读问答不强制测试。

这些机制不证明测试充分或业务语义正确：范围由模型选择，输出来自被执行项目，弱测试或伪造输出不能自动排除。指纹覆盖过滤后的工作区，不覆盖依赖与外部资源。大仓库扫描成本、跨scope的证据替代、最终自然语言数字准确性仍待改进。

## 会话与恢复

会话保存在`.cli-rb/sessions/<ID>/`，包括SQLite检查点、基线、筛选清单、工具日志、验证记录和收尾审阅。长对话会整理工作上下文，保留用户要求和工具配对；原始记录不会因摘要删除。接近模型窗口上限的工作质量尚需验证。

```powershell
uv run cli-rb --resume '.cli-rb/sessions/<ID>'
uv run cli-rb inspect-session '.cli-rb/sessions/<ID>' --changes
uv run cli-rb explain-session '.cli-rb/sessions/<ID>' --request '原任务要求'
```

恢复沿用原工作区和基线，不重放旧工具。中断后核对不确定动作与差异，由用户确认继续，后续写入重新授权。拒绝会保留现场；基线损坏、检查点异常或遗留容器未处理时阻止恢复。

`inspect-session`读取本地证据；`explain-session`单独调用模型解释记录，不推进旧任务，也不能作为完成证明。Ctrl+C保留已发生修改，不自动撤销；客户端取消不证明服务端计算立即停止。

## 开发与评测

```powershell
uv run ruff check src tests
uv run pytest -q
# Docker引擎可用且基础镜像已准备时：
$env:CLI_RB_DOCKER_TESTS = '1'
uv run pytest tests/test_docker_integration.py -q
```

最近完整工程验证为137项离线测试与14项Docker集成检查通过；测试数量不是编码任务解决率。不在例行开发中自动重跑模型或SWE评测。

评测模块保留通用入口，历史轮次脚本和实验数据不属于产品发布：

| 模块（`python -m ... --help`） | 职责 |
| --- | --- |
| `cli_rb.evaluation.tasks` | 从JSONL导出公开任务字段，或将补丁封装为预测 |
| `cli_rb.evaluation.runner` | 在干净指定版本与已准备镜像上运行Agent，保存轨迹和补丁 |
| `cli_rb.evaluation.environment` | 显式构建旧项目所需的兼容环境变体 |
| `cli_rb.evaluation.scoring` | 在单独安装`swebench==3.0.17`的环境生成脚本与评分 |

原始数据、参考补丁和评分测试名单必须放在求解工作区之外。Agent只获得公开issue和固定版本仓库。评分与求解分离，使用指定目标及回归测试；节点解析缺失标为`evaluation_error`，不等同于模型失败。环境适配成绩需单列，日志解析和网络/fixture问题仍需审阅。`status=ended`仅代表执行结束。

## 源码结构

```text
src/cli_rb/
  runtime/      Agent循环、上下文、计划、持久化、提示加载
  workspace/    文件工具、Docker、快照、补丁、容器worker
  quality/      验证账本、风险审阅、收尾核对
  evaluation/   任务、环境、执行、评分
  interface/    CLI、授权、诊断、会话解释、演示
  prompts/      模型工作规范
  config.py     模型服务配置
tests/          工程回归与Docker集成检查
```

CLI入口为`cli_rb.interface.cli:app`。从旧版本升级后运行`uv sync --locked`刷新启动器；旧内部模块路径已移除。配置模板`.env.example`不含凭据，本地会话、实验数据与协作文档不进入发布包。
