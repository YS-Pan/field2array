# 2026-11-10 项目迁移结果

本次迁移依据用户2026-11-10的迁移请求及追加要求，目标和范围摘要见[项目纲领](../../project-charter.md)。请求原文仅保留在本机 `artifacts/legacy-20261027/migration/user-request.original.md`，不随Git发布。新工作区为 `D:\project\20261109 array reconstruct`，Codex 项目为 `20261109 array reconstruct`；旧工作区与旧 Codex 聊天保留原位。

## 来源与去向

| 来源 | 新位置及处理 |
| --- | --- |
| 旧 `src/tests/configs/scripts` 与包声明 | `experiments/20261027-simulation/`；选留仿真/数值闭包，逐文件来源与哈希见路线 manifest |
| 必要参考阵元几何 | 路线 `metadata/` 保持兼容位置，`data/README.md` 说明来源；不改算法查找逻辑 |
| 旧实验报告及少量关键证据 | `artifacts/legacy-20261027/reports/`；按[结论](../legacy-20261027/experimental-conclusions.md)和[证据索引](../legacy-20261027/evidence-index.md)导航 |
| 旧项目主聊天及关联子线程 | `artifacts/legacy-20261027/chat-history/`；完整日志、可读对话和人类原话见[聊天索引](../legacy-20261027/chat-history-index.md) |
| 旧 `.memory` | 选留与聊天保全相关的原始证据；程序、数据库、模型、虚拟环境、发布状态和缓存不作为新运行系统迁入 |
| 新版 basepack | 普通安装 0.6.2 客户端；本项目重新 `init`，使用独立 UUID、端口和物理数据库 |
| 新版文件结构与协作技能 | `skills/file-structure/`、`skills/agent-team/` 的6份原样快照；根 `devDoc/` 统一维护现行文档 |
| 旧大型数据与全量实验输出 | 留旧目录，按源路径恢复；本次没有完整复制 |

## 代码与科学结论

从233份源代码/测试/脚本/配置候选中复制77份（原始1,108,138字节）；75份原字节保留，2份只适配命令入口与依赖声明，新增仿真命令分发入口。34个数值模块没有算法行改动。完整清单及排除原因见[源码来源清单](../../../experiments/20261027-simulation/source-manifest.json)，当前可用方式与实际验证见[仿真路线](../../system/simulation-route.md)。

历史科学结论已按适用条件整理，保留成功、失败和未完成项。最终 FFT/ASM 不能支持可靠1024逐元恢复；图中仍有粗纹理，不能改写为普遍物理不可能。当前退出实测路线来自用户本次决定。报告快照共53份、3,291,637字节，源/副本/复制后源SHA-256一致，其中30份JSON解析通过。

源码及技能在 Git 中以 `-text` 保留迁移字节，避免自动换行转换破坏来源哈希。旧源码目录没有 Git 历史，本次新建本地 `main` 仓库；没有配置远程或推送。

## 聊天保全与实际检查

- 21个主聊天和705个关联子/审查线程，共726份原始日志、988,026,968字节；726份源/目标SHA-256一致，源在复制期间稳定，全部解压逐行解析，JSON错误与ordinal间隙均为0。
- 主聊天97条真人原话逐字保留；子线程委托和自动上下文分别标识。5次交互提问的显示文本已从原始工具参数补入可读对话，5个真人回答均已在97条中，没有仅存在于tool/event的额外答案。可读导出共8417条文本修订，完整工具记录保留在raw原件中。
- 旧记忆聊天证据ZIP含6872项，逐项SHA-256复核通过。聊天保全目录含2940文件，约1.13GB；清单和覆盖限制见[聊天索引](../legacy-20261027/chat-history-index.md)。
- 943条源记录的 `retained_source.complete` 为false，现存正文照录；该标志本身不证明正文截断或历史丢失。97条真人记录的该标志均为true。工具超大turn读取失败、只读SQLite快照WAL可见性等限制在索引明确说明，以官方列表和全部rollout元数据交叉核对。
- 仿真路线441项CPU维护测试通过，71个Python文件编译、35个模块导入和命令入口检查通过。5份可选tensor/CUDA测试未执行。没有重跑科学实验。
- 77个迁入文件的Git blob哈希与manifest目标哈希一致；整个路线81文件和6份技能快照的Git blob与工作区字节一致。证据在 `artifacts/migration-20261110/git-blob-verification.json`。

根级入口、现行开发文档、技能和路线README的本地链接统一检查；原始历史报告内未选留的链接保持原样，按证据索引回旧项目恢复。独立记忆库的语义写入、检索与备份验收另见[basepack接入](../../system/basepack.md)及[实测记录](basepack-test-findings.md)。

## basepack 测试样本

[实测问题记录](basepack-test-findings.md)保存触发条件、原始错误、影响、处理结果与后续建议，区分新版产品/文档、宿主沙盒限制及旧项目资料问题。本次只在新项目记录问题，没有修改 memsys/basepack 源项目。

## 新记忆验收

首轮实际登记36个来源：15份入口/开发/路线文档，以及21个主聊天的真人原话文件；语义记录264个成员、21个节点和21条关系。264成员首次编码完成，用时18.62秒。当前纲领作为唯一mandatory入口，97条旧真人原话保留历史来源和日期。

主控在独立调用中实际取得brief：source_count=36，mandatory明确包含“仅迁移、纯仿真方向、长期第一版后GitHub并暂停、保全原话与结论、记录basepack问题”，mandatory_pending/pending/change/vector_pending均为0。历史来源范围说明曾被建模为21条coverage gap；此类范围应在摘要表达，修正结果见实测记录。真实历史可见性限制仍按档案保留。

首次统一apply触发HTTP413，已确认无部分写入；3个小批和后续关系批实际成功。原196,573字节spec序列化后的HTTP SQL体为1,108,205字节，服务限制值未独立确认。错误、分批体积与处理见实测记录。最终检索/回读/rank、文档小批维护、changes检查和原生备份的验收证据由[basepack接入](../../system/basepack.md)及[实测问题记录](basepack-test-findings.md)保留。

## 保全与使用

根 README 提供当前工作入口；历史原件从归档索引按需读取。旧运行时钟、预算、STOP、Goal、grant、seal 和旧完成声明保留其历史身份，后续研究按新任务组织。

`artifacts/` 和 `.basepack/` 被 Git 忽略，包含本地历史原件、完整聊天、独立数据库及凭据。**仅复制 Git 仓库不能带走全部记忆**；备份新项目时应一并保留这两个目录。数据库使用原生导出，恢复时保留本项目配置/UUID和原件；不要在线复制store。`tmp/` 为可清除的迁移中间材料。

后续工作是制定纯仿真第一版算法的任务与验收方案；本次没有启动算法调整、新实验或 GitHub 发布。
