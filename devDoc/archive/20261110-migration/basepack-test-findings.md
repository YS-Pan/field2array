# basepack 迁移测试发现

初次记录：2026-11-10；后续观察更新：2026-11-12。用户要求将本次真实迁移作为 basepack 测试样本；本轮记录观察与可复现条件，修复留给后续 basepack 维护，不修改 `memsys` 源项目。无凭据证据位于 [`artifacts/basepack-test-20261110/`](../../../artifacts/basepack-test-20261110/)，后续使用反馈的证据位置在相应条目列出。客户端与库隔离细节见[本机接入](../../system/basepack.md)。

## 后续处理

优先级只表示后续维护顺序，条目包括已绕过的环境限制和旧资料问题，并非均为新版产品缺陷。

| 优先级 | ID | 归属 | 下一步 | 状态 |
| --- | --- | --- | --- | --- |
| P2 | BP-T03 | basepack 生成入口/文档 | 在首个 brief 前提示 password_env 与凭据加载入口 | 本项目加提醒，产品待评估 |
| P2 | BP-T26 | basepack/Surreal请求容量与CLI引导 | 显示实际序列化体积与清晰分批指引，保留原子语义 | 本次3小批+关系批绕过，服务limit未知 |
| P2 | BP-T22 | basepack CLI 输入校验/帮助 | prepare 明示文件列表，目录输入给出清晰错误或显式展开 | 本轮显式枚举文件后成功，待改进 |
| 待定 | BP-T28 | basePack 记忆生命周期与检索分层 | 后续由 basePack 项目研究“缺乏遗忘机制”问题 | 用户要求记录；当前搁置，未启动修复 |
| P3 | BP-T04 | basepack config 可观测性 | 考虑无秘密地显示 endpoint/UUID；继续分开静态与真实闭环验收 | 建议，未认定缺陷 |
| P2 | BP-T01、T02、T05、T21、T25 | sandbox/宿主运行器 | 提供端口视图、后台寿命及可写临时目录的明确能力说明 | 已绕过，环境未改 |
| P2 | BP-T16–T19、T23、T24 | Codex 历史工具/导入 adapter | 单 turn 分页或 raw fallback；识别 zst、只读快照与路径身份 | 本轮历史保全已绕过 |
| P2 | BP-T06、T13–T15、T20 | 旧资料与归档语义 | 保留日期/身份/缺口/恢复定位，当前意图由新纲领定义 | 本轮已标明，原始缺口保留 |
| P3 | BP-T07–T12 | Git 与旧代码迁移 | 保真 hash、命令闭包、输入恢复与可选依赖检查可作为迁移样例 | 本轮已处理，不倒推为技能缺陷 |

## 测试条件与状态

- Windows PowerShell，Codex workspace-write 沙盒；普通安装 basepack 0.6.2 客户端，SurrealDB 4.0.0。
- 新根 init 生成独立 UUID；数据库位于本项目 `.basepack/memory/store`，endpoint 为 `http://127.0.0.1:18089`。未迁入 memsys 数据库或业务状态。
- 共享 embedding health 200，服务报告 basepack 0.5.0；本轮只连接，未重启服务、修改机器配置或注册 SCM。
- 已通过 `config show`、`doctor core`、跨调用 ready/init/brief，以及再一次独立 brief。首轮语义入库36源264成员、21节点21关系，264成员共享编码完成；跨调用brief与登记源deep changes通过。最终小批、原文检索/rank和原生备份证据见末节及验收JSON。

## BP-T01：沙盒 netstat 空结果不能证明端口空闲

**分类：sandbox/宿主可见性限制；未证实为 basepack 产品缺陷。**

复现：在普通沙盒工具调用运行 `netstat -ano -p tcp` 或 `netstat -an -p tcp`。原始输出只有 `Active Connections` 和表头，退出码 0。相同命令获准在沙盒外执行后显示实际连接；缩小到 `18080–18089` 得到 18080/PID 6232 与 18081/PID 1236 已监听，18089 空闲。

影响：只看成功退出和空表会误判默认端口 18080 可用，并有误连其它项目数据库的风险。本次在启动前复核并选 18089，没有误连或改动已有库。

状态：本次已绕过；系统层可见性差异仍存在。未来接入说明可明确“空输出与成功退出不等于具有系统连接视图”，并使用真实 endpoint、独立 UUID 检验保护接入。证据：`setup-observations.json` 的 `sandbox_netstat`、`external_port_check`。

## BP-T02：沙盒隐藏进程返回 PID 后不能持续运行

**分类：sandbox/宿主进程寿命限制；现有管理员文档已明确要求独立 ready 检验。**

复现：在沙盒工具调用中用 `Start-Process powershell.exe ... -WindowStyle Hidden -PassThru` 启动本项目 `serve.ps1`，cwd 为记忆目录，分别重定向 stdout/stderr。调用退出码 0，返回 launcher PID 15996。下一次独立调用 `/ready` 失败，原始错误为：`Invoke-WebRequest: 由于目标计算机积极拒绝，无法连接。` 后续 `Get-Process -Id 15996 -ErrorAction SilentlyContinue` 无结果，两份日志为空，尚未创建 store。

影响：PID 不能作为数据库可用证据。该调用链在 ready 失败处停止，没有继续 init 或 brief，也没有重复占用同一 store。

状态：本次经合法 escalation 在沙盒外启动同一个隐藏 launcher（PID 33180），随后独立 `/ready` HTTP 200、memory init/guide/brief 成功，再一次工具调用 brief 成功；Surreal 只监听 `127.0.0.1:18089`，PID 33428。没有注册 SCM，也没有改共享机器设施。尚未证明所有宿主/会话寿命均可持续；未来重启需沿用逐次 ready/init/brief 验收。证据：`setup-observations.json` 的两个 launcher 记录。

## BP-T03：生成入口的凭据引导可更直接

**分类：basepack 入口文档/UX 观察；未触发运行错误。**

0.6.2 的 `memory init` 将 `memory brief --task ...` 命令与 memory skill 指针加入 `.basepack/instructions.md`；该生成段没有直接提醒每个新 shell 需要按 `memory.password_env` 加载凭据。随包 memory skill 与管理员文档已经写明此要求。本次提前按文档加载 `BASEPACK_MEMORY_PASSWORD`，因此没有产生认证失败。

影响：只照生成段运行的新会话容易先遇到认证/缺环境错误，再去找 skill；“先读 skill，再执行首个 brief”的启动路径更清楚。

状态：本项目忽略目录中的 instructions 已加一条实际凭据来源提醒；本机接入文档给出不回显凭据的完整命令。后续可考虑生成入口加入密码环境变量说明，或直接指向本机受控凭据加载流程。未修改 basepack 包或 memsys。

## BP-T04：静态设施检查与真实记忆闭环分别验收

**分类：验收范围记录；不是已发现缺陷。**

`config show` 与 `doctor --facility core` 在数据库未启动时已成功；doctor 输出明确指出只读设施检查，实际推理和作业另行验收。`config show` 有 memory_dir，但 endpoint/UUID 仍需读取项目配置。共享 embedding `status` health 200 也仅说明共享服务就绪。

因此本样本分别保留配置与安装核验、端口/物理 store/UUID 隔离、跨调用 ready/init/brief，以及待完成的 prepare/apply/embed/search/read/changes 验收。前几项不能代替语义入库和来源回读。未来可考虑在 config show 增加无秘密的 endpoint/UUID 观测字段，降低手工查配置成本。

## 后续连续记录

后续 memory 批次、向量、查询、来源回读和原生 export 的真实错误与限制追加在本文件，并将无凭据结果保存到同一证据目录。不会将没有实际触发的问题写成已证实缺陷。其他迁移分支提供的观察需保留条件和原始错误，再归类为产品、文档、数据兼容或宿主限制。


## BP-T05：CIM 查询被受限 shell 拒绝

**分类：sandbox/宿主限制；主控分支实际观察。** `Get-NetTCPConnection` 和 `Get-CimInstance` 均返回原始错误“无法从客户端中访问 CIM 资源”。本轮使用获准的沙盒外 `netstat` 与实际 endpoint 检验，不把 CIM 查询失败当作无进程或无监听。该限制未修复，也未改变 Windows 策略。

## BP-T06、T13–T15：旧记忆时态、原证链接和写入保真

**分类：旧资料质量与归档合同；结论分支未调用新版 basepack CLI。**

- BP-T06：旧 README/AGENTS 的最新说明在 11/03，而科学报告追加至 11/04。旧 `docs/memory/README.md:5`、`repair-upgrade:29` 仍将 07:05 的 `r-d640…` 称当前，`task-card:105` 已指向更晚 `r-923…`。本轮全文核报告并将旧 current/运行状态保留为历史，现行权威由新纲领定义。未来记忆可强化来源日期、冲突和“当前”定义归属展示；该观察不证明新版 stale 检测失败。
- BP-T13：按用户要求只选留结论和依据后，原报告指向许多未选留 outputs 或旧绝对路径的链接自然无法本地解析。结论分支保留 53 份原件、3,291,637 字节，并逐文件登记 source/retained 路径和三方 hash；索引显式声明未选留链接需回旧项目恢复。未来归档工具可支持“原文不改、披露外链缺失、保留恢复定位”。
- BP-T14：旧科学报告 249,028 字节、1,283 行，按时间追加并含 Goal/grant/STOP。并行分段全文审读、条件化总结与原行定位用于防止历史授权进入当前任务。旧 AGENTS 改名 `AGENTS.original.md`。未来检索应继续保留来源身份、时段与有效边界，大文件按语义章节处理。
- BP-T15：旧报告第 1264 行记载首轮 summary 遭并行覆盖后确定性重跑，首轮完整字节已缺失。本轮只保留现有原证，不补造初始字节。唯一 writer 和不可变快照是后续产物管理建议，未作为本轮可修复缺陷。

证据和复核入口：`branch-observations.json` 的 root/conclusions 记录，及结论归档 manifest、实验结论和来源索引。该小节是分支观察汇总，原证逐项以结论归档为准。

## BP-T07–T12：目录技能、Git 与旧仿真耦合

**分类：迁移兼容与旧项目边界；仿真分支未发现新版 basepack 实现缺陷。**

- BP-T07：Git 自动换行可能改变要求保真的字节。主控在 `.gitattributes` 对 `experiments/20261027-simulation/**` 和 `skills/**` 设置 `-text`，以 Git blob 与源 hash 核实；属于本次保真检查项。
- BP-T08：旧 CLI 同时声明四个 synthetic 命令和懒导入的 `analyze-real`；仅迁仿真闭包时会宣传缺失命令。保留旧 CLI 字节，新增纯分发 `simulation_cli` 并适配包入口。结构参考可提示检查命令集和懒导入闭包。
- BP-T09：`joint_virtual.py:71` 固定引用 `metadata/reference_element_geometry.csv`，用于 32×32 虚拟阵列。强行搬入 data 会改数值源码；本次小 fixture 留在 metadata，data/README 导航并保留来源 hash。现结构技能已经允许按项目适配，不认定技能有误。
- BP-T10：旧 `joint_amplitude_gpu` 顶层 import torch，但 pyproject 未声明。新增可选 `gpu=[torch]`；原测试的 torch/importorskip/CUDA 查询方式混杂，本轮只做 CPU 维护检查并明确 GPU 模块未验证。
- BP-T11：thesis coded/calibration 完整 runner 依赖 operators NPZ 大缓存；缓存留旧路径，文档通过 `--operators` 指向恢复输入，API/维护测试独立；没有复制全量 outputs。
- BP-T12：结构技能未规定完整源 hash manifest 或数值字节判据。本次按用户保真要求新增 manifest，作为样例，不将新增要求倒推成技能缺陷。

复核以仿真路线的 migration manifest、README 和本次结果为准。`branch-observations.json` 保留分支报告及分类，不把 CPU 维护通过扩大为算法科学验收。

## BP-T16–T21：历史导入工具与压缩日志覆盖

**分类：Codex 历史工具、受限环境与旧归档语义；不是已证实的新版 basepack 缺陷。**

- BP-T16：官方 `read_thread(turnLimit=1, includeOutputs=false, maxOutputCharsPerItem=1000)` 在两个大聊天仍失败，原始错误分别为 `Thread history exceeds the read size limit. Try fewer turns. A single oversized turn cannot be read by this tool.` 与 `Tool response exceeds the size limit. Request fewer turns or omit outputs. A single oversized turn cannot be returned by this tool.` 单 turn 过大时没有可用分页。本次转为本机原始 rollout JSONL/.zst 保全及流式消息导出。证据在历史归档 `audit/official-read-1.json`、`official-read-2.json`；未来 adapter 可提供 message/item 分页或明确 raw fallback。
- BP-T17：相关 726 份历史日志中 705 份已透明压为 `.jsonl.zst`；只扫描 `*.jsonl` 会漏大部分子线程。本次加入 zstd 流式读取并对压缩原字节校验 SHA256。未来 importer 的 inventory 应同时识别两种扩展名。
- BP-T18：受限环境以普通 `mode=ro` 打开 `state_5.sqlite` 因锁/SHM 权限返回 `unable to open database file`。本次使用 `mode=ro&immutable=1` 与 `query_only`，只读 checkpoint 主库，并披露 WAL 元数据限制，再以官方 list 与 rollout metadata 交叉核对；该路径不是 live SQLite 全状态读取。
- BP-T19：历史 state5 的 cwd 带 `\?\` 路径前缀且 project_id 均 null；当前官方 projectId 可用。本次规范化 cwd 并按 parent/fork 闭包补齐关系，不能仅按历史 projectId 筛选。
- BP-T20：旧 `.memory/input-history` 的 25 批是冻结项目文档，不是用户原话。旧 thread-captures 有 328 批、5,174 条消息且重叠，“完整 cursor”不证明 raw 完整。本次将主聊天 97 条真人消息与子线程委托输入分开，保留其身份与来源，避免把委托指令记作用户决定。
- BP-T21：初次安装解压依赖 zstandard 使用沙盒默认 TEMP 触发 `PermissionError`。本次仅将当前进程 TEMP/TMP 指向项目 tmp，并 `--target` 到 vendor，未修改全局 Python。未来运行器可提供明确可写的 task tmp。

复核入口：历史索引、聊天 audit 与 manifest；该小节保留分支的原始错误和当时计数，最终完整性计数以历史分支最终报告为准。


## BP-T22：prepare 的目录输入报 PermissionError traceback

**分类：basepack CLI 输入校验/帮助的可操作性问题；已实际触发。**

复现：在新库执行 `memory prepare --paths README.md AGENTS.md devDoc experiments/20261027-simulation/README.md --output <reading.json>`，其中 `devDoc` 为目录。`prepare --help` 仅列 `--paths PATHS [PATHS ...]`，没有明示只支持文件；`changes --scope` 与 source_scope 又支持目录，容易混用。

退出码 1，原始调用链为 `cli.py:51 → store.py:136 _load_source → content.py:85 read_source → pathlib.read_bytes`，错误：`PermissionError: [Errno 13] Permission denied: 'D:\\project\\20261109 array reconstruct\\devDoc'`。这是把目录当文件读取，不是实际文件读取权限不足；完整证据保存为 `prepare-directory-error.json`。prepare 没有写入知识库，依赖链停止，未执行 apply/embed。

处理：由本任务明确枚举 README、AGENTS、所有 devDoc Markdown 和路线 README，使用21个主聊天 human原话文件时也逐个传入。随后 prepare 成功生成带完整编号原文与 revision 的 reading JSON。

建议：帮助明确文件参数合同，在读取前检测目录并给出清晰 ValueError/CLI提示；若以后支持目录，则显式定义递归、扩展名、排除、顺序和大小限制。当前只记录，不修改产品实现。


## BP-T23、T24：可见提问工具与 retained flag 需要保留语义

**分类：Codex 历史导入 adapter 的消息类型与元数据解释；历史分支实际复核。**

BP-T23：只提取 `response_item.message` 的 user/assistant 会漏掉展示给用户的 `request_user_input_async` 问题。旧 capture 有5条 assistant_final 对应的正文在 raw 中实际是 function_call。历史分支从原工具参数的 questions/title/options 精确重建5/5，导出为 `assistant_tool_question_display`，保留 call_id 关联。真人答案是后续 role=user 的 `<send_user_message_question_reply>` JSON 包装，已包含在97条真人消息中。未来 importer 应区分工具调用记录与确实展示给人的问题，不能将所有工具内容统一排除在可读聊天之外。

BP-T24：`metadata.retained_source.complete=false` 不等同于 `capture.truncated=true`，不能仅凭 flag 推断正文缺失或不是最终回复。本次943条带 false 的修订，现有正文已完整保留并照录源 flag；97条真人原话全部 true。历史导出 header/索引已按此修正。未来应将原 flag 与实际截断证据分开记录，不擅自补全，也不擅自断言丢失。

## BP-T25：Git 索引写入受沙盒保护

**分类：sandbox/宿主权限边界；主控实际观察，不是 basepack 产品缺陷。**

新本地仓库 `git init` 成功，但沙盒 `git add` 失败，原始错误：`fatal: Unable to create 'D:/project/20261109 array reconstruct/.git/index.lock': Permission denied`。权限 profile 将 `.git` 作为只读保护路径。合法 `require_escalated` 后 `git add` 成功；没有配置远程或推送。主控随后独立核对77份迁入文件的 Git blob SHA256，与 manifest 目标 hash 全部一致。未来宿主可提前显示 Git 索引权限能力，接入指南不应把这类错误归因于项目文件权限或 basepack。


## BP-T26：apply 序列化后的 HTTP SQL 请求超过服务容量

**分类：basepack/Surreal接口容量与CLI错误引导；真实触发，不是凭spec大小推断。**

首轮包含36源、264成员、21节点和21关系的 `memory apply --batch migration-batch-01.json` 失败：`urllib.error.HTTPError: HTTP Error 413: Payload Too Large`，服务原文附注 `length limit exceeded`。依赖链停止，没有继续embed；随后 `memory sources` 仍为 `[]`，确认该次没有部分来源写入。

该spec文件196,573字节；在已确认空库前态下，调用本机未改动的0.6.2 `Memory.apply` 序列化逻辑，以只读拦截统计 `_execute` 的UTF-8 SQL体（含正常project身份前缀），得到1,108,205字节。此测量没有再发HTTP、没有打印或保存Authorization。服务实际body限制未独立确认，不将“约1MiB”当作已核参数。

恢复采用按源组的小批：core 13源/114成员，SQL体471,896字节；science 2源/32成员，124,544字节；human 21源/118成员，399,796字节。三次apply均真实成功，来源累计13→15→36。之后重新prepare所有关系两端和依据的264知识版本，再成功写入21节点/21关系；未改Surreal启动设置、机器配置或包源码。随后共享embed对264成员完成，pending=0。

证据：`memory-apply-01.stderr.txt`、`apply-payload-measurement.json`、`memory-apply-02-*.stdout.txt`、`sources-after-*.json`、`memory-apply-03-graph.stdout.txt`、`memory-embed-01.json`。建议为apply提供无秘密的实际请求体积预检及可操作的413分批提示；分批必须保留来源/证据expected版本和每批原子性，不能静默改变一大批的原子合同。

## 入库建模与验收调用的校正（非产品缺陷）

21份human来源都已全文覆盖，但最初将“本文件仅真人原话，助手结论/工具结果/执行完成需查全角色日志”的范围说明放入 `coverage_gaps`，导致每次brief出现21条相同噪声。该说明应在各文件摘要的scope正文表达；最后小批据此调整并保留真实历史限制。用户请求不证明执行完成这一证据边界不被抹掉，也不把它伪装成原件缺失。

一次验收将整个 `chat-history/human` 目录传入 `changes --scope`，正确发现705个有意未登记的子线程header文件；任务要求只语义入库21主聊天真人原件，其它由完整索引/raw按需读取。改用已登记21文件及文档发现scope后deep changes为空。另一次本任务的辅助断言误以为prepare选中来源的成员放在neighborhood；实际合同为 `sources[].members`，邻域只含额外相关成员。改正断言后，264成员范围/摘要与264 expected版本核对通过，才提交关系批。两项均为调用方纠正，没有改产品或额外导入705份空真人来源。

## 语义闭环与备份验收

已实际完成首轮prepare→apply小批→关系共读/提交→共享embed；独立brief的36源及所有pending/change计数为0，现行纲领、历史选名等来源可read，neighbors可展开。检索使用实际配置的tier1 `gpt-6-luna` / `max`，原始hybrid/literal/read/neighbors、已读候选评分与rank结果分别保留，未把数据库词法分数冒充agent评分。

本轮最后将本文件、接入说明、迁移结果和21份human摘要范围一并维护，再执行embed、检索、brief、显式scope deep changes和原生export。最终无凭据计数、freshness、备份hash/大小保存在 `memory-final-verification.json`；原始检索验收在 `retrieval-acceptance.json`。稳定备份为本项目 `.basepack/memory/backups/migration-initial.surql`；已实际创建，最后维护后刷新，不在线复制store。本轮未执行restore演练，未把静态设施检查或备份文件存在等同为恢复验收。

## BP-T27：跨历史问题未使用记忆检索 worker，旧默认盖过后续科学要求

**观察日期：2026-11-12。分类：调用方检索选择与历史时序整合不足；不是已证实的搜索、向量或数据库缺陷。** 本次观察客户端为 `0.10.0.dev1`，聊天为 `01a1cdb6-cd05-7310-bc84-f139afd74ce4`。用户明确要求记录该现象。

本聊天先询问 GitHub 发布剩余工作、当前科学目标，以及“小误差、稀疏异常”的数值。调用方运行了 `memory brief`，随后直接阅读项目文档、配置、源码和报告，并委派普通代码核查子代理；这几轮没有派记忆检索 worker，也没有执行 `memory search/read/neighbors/rank`。普通代码核查子代理不等同于承担完整记忆检索流程的 worker。

回答将 `validation.py` 和普通 CLI 的正常幅度高斯标准差 `0.1 dB`、相位高斯标准差 `1°`、5 个注入故障作为重点，虽提到了大误差失败，却未追溯并整合用户曾要求按真实数据风格扩大正常离散的较晚原话与实际实验。用户随后提醒曾提出这一要求，并追问检索 worker 是否实际使用。此时已能确认代码另有 `large_error_validation.py` 及独立策略，包含全阵幅度均匀分布 `[-2.5,+2.5] dB`、相位标准差 `3°/5°/10°` 和幅度 `[-5,+5] dB` 档；普通 CLI 默认没有随之替换。CLI 默认值、历史成功条件和用户后续科学要求需要分别说明。

本机记忆合同允许在已知少量文件时直接阅读；本条不把“所有问题均必须调用 worker”追认为旧要求。实际问题是：涉及用户历史意图和跨材料前提时，仅靠 brief 线索、当前默认和定向文件阅读，没有充分发现并整合较晚约束。此前已经看过包含大离散结论的文件，因此也不能把影响全部归因为检索工具未调用。

本轮已按 `config ai --purpose retrieval` 的实际映射派出干净上下文检索 worker（`tier1`、`gpt-6-luna`、`max`），负责多轮搜索、原文回读、评分与 rank；不改数值算法、不重启实测路线。

### 回源核查与回答纠正

已通过 worker 的记忆检索和原文回读定位用户要求，主控也重新阅读了原话和执行报告：2026-10-28 10:47:14（Asia/Shanghai；原日志 UTC 02:47:14.694），用户在[真人原话](../../../artifacts/legacy-20261027/chat-history/human/01a18064-d738-78d0-a46d-fd24c41704da.md)第14行明确指出旧假设“幅度标准差0.1 dB，相位标准差1°”过小，实际幅度“可能有最大5dB的差距”，相位经校准但“应该也到不了1度这么小”，并要求“把误差调大，再做一次分析”。消息 ID 为 `msg_01a18067-9606-79d0-bc60-9319a8eb5d56`；原压缩会话与行号由该文件第11行保存。用户没有指定精确分布、相位标准差或故障数。

后续[大离散验证报告](../../../artifacts/legacy-20261027/reports/docs/large_error_validation_report.md)第24–36行明确将幅度均匀 `[-2.5,+2.5] dB`（峰峰5 dB）、相位高斯标准差 `3°/5°/10°` 作为敏感性设定，主情形为相位 `5°`；另测幅度 `[-5,+5] dB`。这些精确分布是助手将原话操作化后的实验假设，不是已测得或用户确认的芯片统计。正常离散遍布全阵，额外5个注入故障是另一个维度。

报告第7–20行记录实际完成54个32×32案例、216次拟合；18个小误差对照、36个大误差案例，36个大误差案例均没有任何方法同时通过原三项恢复门。`large_error_validation_policy.json` 与 `large_error_validation.py:55–87` 已迁入，专用脚本也保留；普通 `simulate` 仍调用小误差 `validation.excitation()`。因此应回答“较大正常离散已经要求、实现并做过验证，但普通 CLI 仍保留旧默认”，不能说只是假设调整计划，也不能把旧小误差成功条件代替后续科学要求。该核查只说明历史任务及现有代码事实，不恢复历史运行授权。

已实际提交 `retrieval` 使用反馈 `feedback:241dfdf4990943f6b9e482162c64ca3e`，回执保存在 [`retrieval-feedback.json`](../../../artifacts/runs/assumption-history-20261112/retrieval-feedback.json)。上报时未运行检索，故没有被遗漏命中的成员 ID；反馈保留查询、来源线索、观察版本和调用方身份，不伪造搜索失败或召回率。反馈后续处置由单一维护 owner 按正常维护合同进行。

本轮实际检索的评分候选、rank 结果和用户原话版本回读分别选留为 [`retrieval-scored.json`](../../../artifacts/runs/assumption-history-20261112/retrieval-scored.json)、[`retrieval-ranked.json`](../../../artifacts/runs/assumption-history-20261112/retrieval-ranked.json) 和 [`read-human-error-request.json`](../../../artifacts/runs/assumption-history-20261112/read-human-error-request.json)。rank 预算12000字符，选中6项、正文3985字符、无预算遗漏；这是本次问题的证据选择，不代表全项目记忆覆盖。记录及回答已纠正，不将调用方行为问题标为产品修复；收尾时反馈保持 pending。自动维护已 stop、取消自有协调者及后代，确认 publications 为空后 release，最终 owner 为 null；未完成语义候选留后续维护。

## BP-T28：缺乏遗忘机制（初步命名；待 basePack 项目修改）

**2026-11-12 用户判断：这是记忆系统的一项缺陷，初步称为“缺乏遗忘机制”。状态：已记录、当前搁置；未来由用户在 basePack 项目另行启动修改。** 本项目不因这条记录开始产品修复、批量退休知识或删除历史原件，研究计划也不依赖该问题先解决。授权依据为本聊天 `01a1cdb6-cd05-7310-bc84-f139afd74ce4` 的当日原话：“我觉得这是记忆系统的一个缺陷，初步称之为缺乏遗忘机制，把这个问题记录下来，我未来让basepack项目做修改。现在先搁置这个问题。”

观察客户端为 `0.10.0.dev1`。过时的科学假设、用户否定旧假设的原话、有效的旧实验事实和现行要求，可以同时作为活跃记忆进入默认检索。`devDoc/archive/` 与 `artifacts/` 是文件归档位置，已经入库的条目不会因路径含 archive 或被 Git 忽略而自动退出检索。来源未变的历史条目仍可为 `stale=false`，这一状态并不判断其内容是否仍是现行科学假设。

现有实现已经有 `retired`：普通搜索和 brief 排除退休条目，已知 ID 或来源版本仍可定向回读。此次缺陷的初步含义是：缺少面向语义有效期、后续替代和日常召回的完整遗忘/降级流程，以及默认检索与按需历史追溯的明确分层；不等于底层完全没有退休能力。如何实现、自动化到什么程度、如何验证，留给后续 basePack 项目决定。

具体样本是 `0.1 dB / 1°`：当前 CLI 仍保留旧默认，历史报告保存旧基线；记忆检索命中的相关原话和摘要已明确写作“原仿真过于乐观，用户要求增大”，本轮没有发现将旧参数无条件称为现行科学目标的条目。BP-T27 记录的调用方没有使用检索 worker、对已有历史结论整合不足，与本条的产品生命周期问题分别保留，不能将二者互相替代。

后续 basePack 项目可据此明确：过时的现行内容如何退出日常召回、历史原证和替代依据如何保留、何时显式启用历史检索。当前只记录问题与样本，不预先确定新架构，不声称已经修复。选留检索状态与反馈回执保存在 `artifacts/runs/forgetting-issue-20261112/`；已实际提交反馈 `feedback:6f1d98e878a94b93942a000b590f4b54`，见[回执](../../../artifacts/runs/forgetting-issue-20261112/feedback.json)，保持 pending 等待未来处置。
