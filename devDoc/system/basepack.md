# basepack 接入与项目记忆

本项目使用 basepack 0.11.0.dev1 开发版本的项目独立客户端。机器设施只作为共享资源连接，本项目配置、UUID、记忆数据库及日志由本项目独立持有。业务范围与执行授权以 [项目章程](../project-charter.md) 为准。

## 本机绑定

- Python：`D:\project\20261109 array reconstruct\.basepack\client-0.10\Scripts\python.exe`，安装模式 `wheel`，包位于该环境的 `Lib/site-packages/basepack`；环境目录沿用原名，实际包版本为0.11.0.dev1。机器共用的0.9.1环境保持原样。
- 本机配置：`.basepack/config.toml`，schema 2；机器配置：`D:\AgentTools\basepack.toml`，schema 1。机器配置路径仅从 memsys 的 `runtime.machine_config` 读取，未复制其配置或状态。
- 新项目 UUID：`3b76dcd5-2b05-46fa-b73d-d6e9e4ba9e7f`，由新根 `basepack init` 生成。
- SurrealDB 4.0.0：`D:\AgentTools\bin\surrealdb-4.0.0\surreal.exe`。本项目 endpoint 为 `http://127.0.0.1:18089`；namespace/database 为 `basepack/memory`，物理 store 为本项目 `.basepack/memory/store`。
- 已有共享 embedding：`http://127.0.0.1:8765`，health 200；服务报告 basepack 0.5.0、LitServe 0.2.19、Sentence Transformers 6.2.0。本轮只连接，未重启或变更机器服务。
- `.basepack/` 全部忽略；密码由本项目新建，保存在 `.basepack/memory/password.txt`，由客户端按项目配置自动读取；`memory.password_file` 可覆盖默认路径。终端和文档不输出密码。

两个技能 `skills/file-structure/` 与 `skills/agent-team/` 是 memsys 中同名目录的原样快照，包括 references 和 agents；6 个文件逐个 SHA256 相等。其它 memsys 业务、数据库和作业状态未迁入。

## 启动与接续

本地入口及安装包 memory skill 的位置见 `.basepack/instructions.md`。调用时显式指定当前项目配置，客户端自动读取密码文件：

```powershell
$python = 'D:\project\20261109 array reconstruct\.basepack\client-0.10\Scripts\python.exe'
$config = 'D:\project\20261109 array reconstruct\.basepack\config.toml'
& $python -X utf8 -m basepack --config $config memory brief --task '<当前任务>'
if ($LASTEXITCODE -ne 0) { throw 'memory brief failed' }
```

数据库已作为隐藏的普通用户进程启动。本项目 launcher 是 `.basepack/memory/serve.ps1`（UTF-8 BOM），从同一配置读取 endpoint 和 data_dir，以记忆目录为 cwd；stdout/stderr 在 `.basepack/memory/logs/`。没有注册 SCM 服务，也没有改变共享设施。维护者重启前先确认本项目进程和监听已退出；不要对同一 store 重复启动。

需要启动时，用 `Start-Process powershell.exe -WindowStyle Hidden` 运行上述 launcher，cwd 设为 `.basepack/memory` 并分别重定向日志。在下一次独立工具调用检查 `/ready`，再用本项目配置执行 `memory init` 和 `brief`；PID 不能代替验收。受限沙盒曾使隐藏 launcher 随调用回收，已保留错误并改用获准的沙盒外隐藏启动方式。

## 记忆维护

任务开始或接续先运行 `brief`；它读取已发布知识，不扫描原件，pending 为零也不能证明原件已同步。知识原件在正常项目路径保留；摘要与语义章节保留其来源、行号、决策依据及 coverage gaps。原始聊天由历史归档保存，记忆维护区分发现、采集、语义接受和兼容向量覆盖。历史运行授权保持历史身份。

按 AGENTS 的关键动作与 `.basepack/instructions.md` 接续维护。主任务先直觉估计业务时长：约十分钟以内只运行brief并跳过整个自动maintain，更长任务并行业务启动一次，范围明显扩大时重新判断；子任务不递归维护。自动调用明确使用 `memory maintain --mode automatic`，显式维护使用 `--mode manual`（命令默认）。启动时分配全新control、刷新宿主注册清单，程序在预检前取得单项目owner；已有owner时继续业务。仅按返回的worker与固定layout派工，业务结束stop、取消自有后代、确认发布结束后release，未明结果保留owner。完整命令与分支以生成入口为准。

宿主容量快照在本项目 `[ai]`：2026-11-12从用户级Codex配置采得 `max_concurrent_threads_per_session=12`，项目无覆盖，归一化为不含主agent、包含全部后代的C=12。安装/换宿主时通过 `config set-concurrency`定点更新容量及来源；本轮可用额度A另扣已占席位后交脚本，自动维护的业务预留由脚本计算。自动只启用第0队，固定名称由脚本返回，零reader时协调者亲读；只有第0队协调者发布。当前Git默认发现已跟踪及未忽略新文件，`source_scope`为额外纳入项，已有登记源与历史待核继续参与。

批次作者先prepare并阅读必要原件、依据与关系，生成表示后由同一owner携control执行apply；程序准备候选向量，经WebSocket事务原子发布。普通维护在作者生成和程序提交后结束，按需通过feedback-report保存使用疑点，由后续owner处理；手动语义审计只在用户明确要求时启动。检索通过独立 `retrieval-dispatch`/`retrieval-prompt`按脚本给定名称、档位和参数派工，无需维护owner；零席位时由主agent执行同一检索合同。可重建阅读包、batch和验证放根tmp，长期选留证据放artifacts/runs，配置、数据库、受管原件、control及恢复回执保持持久位置。当前mandatory应跟随现行项目纲领，已发布版本是否同步须核实际状态。

0.10旧维护回执没有新版固定layout，保留作历史原证；升级先完成旧轮次收尾，再用新control和新maintain取得0.11回执，不直接交新版prompt续派。升级客户端/入口不会自动接受积压知识，也不等于实现“缺乏遗忘机制”的产品修复；BP-T28仍按用户要求搁置。

## 接入验收与限制

2026-11-10 已完成：`init`、`config show`、`doctor --facility core`；独立调用 `/ready` 200、`memory init`、`guide`、`brief`，随后又一次独立 `brief` 成功。初始空库 `source_count=0`，证明尚未将其它项目知识导入。首轮随后实际登记36源、264成员、21节点、21关系：15份入口/开发/路线文档与21个主聊天真人文件，97条真人消息按原日期、身份、thread与原日志定位分块；旧人类语句均为历史意图，唯一mandatory为现行纲领。

沙盒内 `netstat` 曾只返回表头；获准在沙盒外复核，18080/18081 已占用，18089 空闲。首次沙盒隐藏启动 PID 15996 消失，独立 ready 返回“由于目标计算机积极拒绝，无法连接”，其日志为空，下游 init/brief 当即停止。获准沙盒外重新启动隐藏 launcher 后，ready/init/brief 成功，监听为 `127.0.0.1:18089`。设施验收不代表算法、仿真或 Condor 作业验收；本轮没有提交计算。

新检出不会自带被忽略的配置、凭据或数据库。重建应重新 init 并选择独立端口/UUID；恢复同一项目时使用它自己的配置、原件、assets 与原生数据库备份，保留 UUID。正在运行的数据库使用 SurrealDB 原生 export，物理 store 复制只在停写停服后进行。


### 本次语义闭环与备份

首轮264成员通过既有共享embedding服务编码，18.62秒，pending=0。独立brief取得36源，mandatory_pending/pending/change/vector_pending均为0；现行纲领、历史选名原话可原文read，neighbors可展开依据，显式登记源的deep changes为空。文件摘要的“真人请求不能证明助手结论/执行完成”是source scope说明，置于摘要正文；真实历史可见性限制保留在聊天索引，不作为21个伪coverage gap重复启动噪声。

检索验收按本机 `config ai --purpose retrieval` 返回的tier1 `gpt-6-luna` / `max` 单次委托，实际使用hybrid、text literal、原文read、neighbors与评分/rank，覆盖当前范围、历史名称与地址、FFT/ASM条件、历史预算/停止与prepare错误的分类。首轮与最后维护的无凭据结果在 `artifacts/basepack-test-20261110/`，最终计数和freshness以其中 `memory-final-verification.json` 为准。

原生初始备份路径为 `.basepack/memory/backups/migration-initial.surql`，使用SurrealDB 4的namespace/database export创建；本轮最后小批维护后刷新同一路径。备份hash与大小只写验收JSON，避免已登记文档为备份元数据反复失效。本轮没有做异地restore演练；恢复集合仍必须包含本项目配置/UUID、原件及必要assets，不能只保留Git仓库或导出文件。

## 独立技能的本机发现入口

2026-11-10 按 basePack 接入计划补齐两项现有技能的宿主发现入口：`.agents/skills/agent-team` 与 `.agents/skills/file-structure` 是指向本项目 `skills/` 原件的 junction。完整技能及同目录参考仍只在 `skills/` 维护；本机 junction 按名称被 Git 忽略，新检出后需重建。此次没有安装其它技能或修改算法；Codex App Server 的 skills/list（本项目 cwd、forceReload=true）已实际返回 agent-team 与 file-structure，scope 均为 repo、errors=[]；未运行技能行为或效果测试。

## 2026-11-11 认证凭据自动加载升级

共用客户端已普通升级为0.8.1；保留原密码、账户、UUID、store和认证数据库进程。新NoProfile PowerShell从其它cwd直接执行生成入口brief、原文read与默认vector检索均通过；七张知识表逐记录逐字段与升级前备份一致，原密码和launcher字节、进程PID/创建时间保持。配置已移除password_env并补宿主Codex项目绑定；此轮没有为升级采集或编码全部聊天。

## 2026-11-11 原地升级至 0.9.1 接入

共享客户端在本次开始前已是 0.9.1 普通 wheel；与 memsys 当前 `src/basepack` 的42个分发文件逐个一致，两项独立技能也一致，本次复用已有安装。已先完成原生 export、原件及凭据恢复集合备份，再运行 `memory init` 刷新本项目入口。刷新后七表逐记录（含向量）与升级前一致，原配置、密码和 launcher 字节保持，入口标记外的本地说明保留。

从其它 cwd 直接执行 brief、默认 vector、hybrid literal、read 和 neighbors 均通过。已在独立目录和端口用升级前 export 实际恢复；七表含向量与备份逐记录一致，全部21条语义关系的46条依据按保存版本与行范围回读匹配，关系方向与展开核验通过。此次恢复验证覆盖数据库和36份登记原件；完整选留 artifacts 已另行备份并核哈希，未将这次演练称为整个历史聊天项目的异地搬迁。

本次默认全项目维护已完成机械变化，并采集6个可证明归属本项目的聊天，继续派工接受语义候选。受限沙箱内的 Codex App Server 曾因无法初始化用户目录中的 SQLite 运行时失败；本次改用获准的普通用户上下文后采集成功。宿主返回的归属未明条目保留边界，不直接计作本项目缺失聊天。实际接受范围、任务结束时的待维护项与证据见[本次升级记录](../archive/20261111-basepack-upgrade/result.md)。

## 2026-11-12 独立客户端升级至 0.10.0.dev1

使用memsys最终wheel，SHA256为 `f9036c12ea38fdaf024b66252354174139834accaf7ecdab2b4c9cc53a769040`，50个包文件与当前源码一致。只在本项目 `.basepack/client-0.10/` 安装，配置绑定该解释器；已有独立技能保持不变。新版memory init刷新AGENTS关键动作及memory托管块，块外仅显式替换旧Python路径，其余原字节保留。

升级前数据库和共享embedding服务未运行；已使用原launcher恢复本项目数据库，并经Windows正常管理员确认启动现有BasepackEmbedding服务，未更改服务配置或模型安装。升级前原生导出、七表记录、原配置/凭据/入口和原件恢复集合已保留。新版schema继续使用原UUID、18089和store，新增反馈表及维护owner状态。实际兼容检查、发布与所有权收尾见[本轮升级记录](../archive/20261112-basepack-upgrade/result.md)。

## 2026-11-12 原位升级至 0.11.0.dev1

本机上游最终wheel SHA256为 `5c97f8641de6154fa881387539d42764fa27a1a3c3a782a78d5f0a80d4325a91`，52个包文件与上游active源码及本项目实际安装逐字节相同。保留现有项目独立环境目录和解释器路径，普通wheel原位升级；依赖仍为psutil7.2.2、websocket-client1.9.2，pip check通过。两项独立技能与上游一致，无需更换或新增。

升级前确认owner为空并保存完整恢复集合：`.basepack/memory/backups/upgrade-0.11.0.dev1-20261112_124037/`，含八表全字段快照、原生export、当前原件/配置/原凭据、受管聊天与维护回执、全部选留artifacts及新旧wheel。原生导出19,726,021字节；原件和约1.13GB选留归档已核哈希。没有迁库、换UUID、轮换凭据或变更共享服务。

配置仅通过新版set-concurrency新增C=12及来源两键，再运行memory init刷新AGENTS和instructions托管块。独立兼容验收确认八表全记录逐字段不变（含向量与反馈），90项保存依据按原source_version回读相符，396/396活跃向量兼容；配置其它字节、两个入口块外字节、凭据和launcher原字节保持。其它cwd的新无profile PowerShell实际brief、vector、hybrid literal、read、neighbors及dispatch渲染通过。该验收验证客户端升级兼容，不声称语义积压或遗忘机制已修复。

长任务按新版automatic模式实际取得固定布局：C=12、本轮可用A=11、业务预留1、仅第0队1协调者+4reader；独立检索按脚本指定的bp_mem_search_0派工，不取得维护owner。实际闭环、发布范围、备份及收尾见[0.11升级结果](../archive/20261112-basepack-011-upgrade/result.md)，其余候选按新版生命周期接续。
