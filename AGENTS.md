# 项目工作约定

- 当前目标见 [项目纲领](devDoc/project-charter.md)；用户的新指令优先。接续科学工作时读 [当前研究计划](devDoc/tasks/20261112-simulation-research/plan.md) 及其结果入口，按已支持范围与开放问题继续。
- 当前实现和入口见 [系统概览](devDoc/system/overview.md)，唯一研究路线为 [仿真路线](experiments/20261027-simulation/README.md)。退出当前实测数据路线的决定见纲领。
- 查旧实验事实时读 [实验结论](devDoc/archive/legacy-20261027/experimental-conclusions.md) 和所指原证；核人类原话时读 [聊天索引](devDoc/archive/legacy-20261027/chat-history-index.md)。历史目标、授权和运行状态按当时范围解释。
- 调整结构时使用 [file-structure](skills/file-structure/SKILL.md)，组织任务时使用 [agent-team](skills/agent-team/SKILL.md)。编写 agent 文档使用宿主的 `writing-for-agents`。开发文档只在根 `devDoc/` 维护，归属见 [文档维护](devDoc/documentation.md)。
- 临时测试与一次性脚本放根 `tmp/`，选留原证放 `artifacts/`；源码、依赖、配置和维护测试放所属实验路线。共享环境与模型通过配置定位。
- 保持实现简洁，先定位现有能力与实际问题，按改动执行必要检查。迁移保真清单用于核对原始算法；后续改算法时同步更新现行说明和相应验证。
- 完成任务时说明实际改动、检查与余项。Git 只纳入本次相关源码、文档、小型配置和测试；本机记忆库、凭据、大型原件与临时文件保持忽略。

<!-- basepack:start -->
## basePack

- 开始或接续任务时，按本机 Python/配置运行 `memory brief`；保留 mandatory，待核内容按业务需要回源：[启动](.basepack/instructions.md#startup)。
- 主任务直觉预计不超过十分钟时跳过整个自动维护；更长任务在宿主支持生命周期时异步运行全项目 `memory maintain --mode automatic`。范围明显扩大时重估，子任务不递归启动维护：[自动维护](.basepack/instructions.md#maintenance)。
- 仅持有维护且 `needs_semantic=true` 时，按脚本返回的 `task_name`、模型和推理强度实际派协调者，持续接批：[派工](.basepack/instructions.md#dispatch)。
- 业务结束时 stop 自有 control，取消自有协调者及后代，确认发布阶段结束后释放所有权；未明状态保留待处理：[收尾](.basepack/instructions.md#stop)。
- 用户明确要求维护时，将授权范围内的处置、实际阻塞和余项纳入交付：[显式维护](.basepack/instructions.md#explicit-maintenance)。
- 检索 worker 或调用 agent 使用中发现问题时，轻量报告疑点和观察版本，交维护处理：[使用反馈](.basepack/instructions.md#feedback)。
- 建立与普通维护在作者生成及程序提交后结束；仅用户明确要求记忆系统审计时按其范围审计：[手动审计](.basepack/instructions.md#audit)。

- 分配 AI、检索或调用资源工具时，按本项目配置解析档位与设施入口：[AI 与资源](.basepack/instructions.md#ai-resources)。
- 新检出缺少本机入口时，先完成 [安装与项目接入](https://github.com/YS-Pan/basePack/blob/main/experiments/20261106-basepack/docs/setup-project.md)；本机配置、凭据和记忆通过接入或备份恢复，不随源码分发。
<!-- basepack:end -->
