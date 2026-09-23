# ADR 0002：只读副本与短生命周期 Docker 工具

状态：采用，2026-09-23。

沿用统一 Agent 循环，只替换工具后端。文件清单通过 Git 的已跟踪和未忽略文件取得，
内容取自工作树，因此包含未提交源码；不通过 git clone 丢失当前修改。
复制过程中不执行仓库脚本，排除路径和已知凭据内容，清单保存哈希与排除原因。
未知秘密无法仅靠通用规则识别，因此不把筛选描述为全面秘密扫描。
本块假定准备期间源目录不被其他进程恶意并发替换；源目录变化应重新准备会话。

当前只读工具无进程状态需求，因此每次调用创建一个容器，共用持久的只读副本。
这个选择减少常驻进程的清理负担，但每次有启动开销；后续命令执行若需要持久容器，
在后端层调整，不重写 Agent 循环。当前搜索用 stdlib 字面文本匹配，无新搜索依赖。

应用只使用固定工具 worker，不执行模型生成的代码；参数通过 JSON 而不是 shell 字符串传入。
不挂载原仓库或 Docker socket。容器采用 none 网络、非 root、read-only、cap-drop ALL、
no-new-privileges、内存/CPU/PID 限制。超时/中断尝试按唯一名称删除本次容器，不自动重放。

镜像在独立准备阶段拉取，工具调用时 --pull=never。会话记录并使用解析出的本地 SHA256 镜像 ID，
保证同一会话不随标签变化。长期跨机器复现所需的发布镜像固定仍待最终交付阶段整理。
只读问答不需要安装仓库依赖；下一块运行项目命令前再实现相应依赖准备。

依据：
- [Docker Windows 安装](https://docs.docker.com/desktop/setup/install/windows-install/)
- [docker run 权限与资源参数](https://docs.docker.com/reference/cli/docker/container/run/)
- [none 网络](https://docs.docker.com/engine/network/drivers/none/)

验收分别记录框架单元/集成、真实 Docker、真实模型；不以模拟后端通过替代隔离验收。
