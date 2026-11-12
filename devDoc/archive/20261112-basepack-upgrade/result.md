# basepack 0.10.0.dev1 升级结果

2026-11-12，用户要求调研新版，适合则升级当前项目。已确认兼容并完成项目独立客户端升级，现行入口见[接入说明](../../system/basepack.md)。本次没有调整算法或迁移其它项目；普通维护按新版合同完成，不开展默认语义审计。

## 选择与实际改动

新版是开发版本0.10.0.dev1，上游已实际验证维护所有权、反馈、分消息WebSocket事务和默认无审计流程。使用active路线最终dist wheel，SHA256 `f9036c12ea38fdaf024b66252354174139834accaf7ecdab2b4c9cc53a769040`，50个分发文件与active源码和本机新安装逐字节一致。较早的同版本tmp构建不作为本次依据。

在 `.basepack/client-0.10/` 安装独立客户端，依赖psutil7.2.2、websocket-client1.9.2；共用 `D:/AgentTools/envs/basepack-client/` 仍为0.9.1。两项现有项目技能与上游一致，无需替换。配置只更新runtime.python；执行新版memory init同时刷新AGENTS关键动作和本机memory块，显式替换块外旧Python路径，其他块外原字节保留。原UUID、18089、store、凭据及launcher保持。

当前系统说明已改为holder/control单owner、WS事务发布、使用反馈和stop/cancel/确认发布结束/release流程。AGENTS业务约定没有改写。

## 设施恢复与升级前备份

任务开始时18089数据库和8765共享embedding均未运行。用原launcher隐藏启动数据库；首次ready检查早于服务完成启动，后续独立调用ready=200。共享BasepackEmbedding服务确认为Stopped，普通权限Start-Service被拒绝后，经Windows正常管理员确认仅启动既有服务。最终数据库ready及embedding health均200；没有修改服务配置、模型安装或SCM注册。

恢复集合：`.basepack/memory/backups/upgrade-0.10.0.dev1-20261112_085218/`。升级前原生export为13,742,625字节，SHA256 `dfea6377a63699de7d2abf30f7a0948e5d32ee7ad79ace373f8f5afae0afa57d`；保留七表全字段、AGENTS/config/instructions/launcher、原凭据、原件、受管采集和完整选留artifacts，以及旧/新wheel。密码及数据库导出保持本机忽略。

升级后另存包含feedback的新原生 `memory.after.surql`、`originals.after/` 当前覆盖层、受管assets/附件及 `after-recovery.json`。恢复取齐基础原件、覆盖层、原凭据、目标wheel和依赖版本，按清单处理已归档的旧路径。旧0.9.1环境保留，但切解释器不等于已验证的新数据库降级；严格回退使用升级前恢复集合，且保留升级后写入。

## 升级兼容性验收

独立只读验收在维护写入前完成：

- 七表含全部向量逐字段相等：36源、42来源快照、278成员、21节点、21语义关系、453变更，meta仍仅项目身份；新增feedback为空。
- 配置仅runtime.python变更，AGENTS托管块外字节不变，instructions块外仅Python替换，原凭据和launcher字节一致。
- 原46条关系依据均按保存版本和行范围回读匹配，代表neighbors的边、方向及内容正确。
- 其它cwd使用新CLI实际brief、默认vector、hybrid literal、read、neighbors通过，278/278向量兼容，查询未改变数据。

一项验收脚本最初错误要求literal组内按lexical_rank单调排序；核实实际合同是精确命中组优先、组内按RRF后修正脚本并通过，原观察保留。这不是产品修复。未重做前一天的1GB隔离恢复，未验证共享环境重装/回退、长期负载、自定义WS限额或全面磁盘负测。

## 新维护生命周期与实际发布

本次control为 `.basepack/memory/maintenance/upgrade-010-20261112-085218.control`，固定回执 `.basepack/memory/maintenance/d24f3a6b22b14cd086f7731ed032f736.json`；owner run_id为 `7bc94e89ae2f43f080c35fd68b79d115`。第二个独立维护请求实际返回acquired=false且原owner保持，未派另一套维护。

默认预检发现142个来源，完成2个机械批次、复用6块；采集8个归属本项目的聊天、保留35个来源。4个采集边界为referenced_subagent_parent_mismatch，336个全宿主归属未明条目保留原边界，不计为本项目336个缺失聊天。

回执129个语义候选（127 initial、2 existing-change），反馈和pending relations均0。作者完成后冻结批次并经新版apply成功发布三批：

| 批次 | 实际来源 | 提交成员 | 结果 |
| --- | ---: | ---: | --- |
| 技能与Git配置 | 8 initial | 42 | 成功 |
| 旧升级记录与后续计划 | 3 initial | 22 | 成功 |
| AGENTS、当前接入说明、本次计划 | 1 initial + 2 existing-change | 17 | 成功 |

累计14个来源、81个提交成员，均无节点/关系改写或失效。原图保留；不能据此声称已经建立更全面的关系图。作者生成及程序提交结束这些普通批次，没有追加例行语义审计。

任务结束已stop，取消自有协调者与全部后代，确认publications=[]后release，最终owner=null。5条committed publication标记数量与2个机械批次和3个作者apply一致；schema没有记录batch hash，逐批归因仍以冻结快照和成功CLI收据为边界。停止后无未明发布。

## 实际阻塞与后续范围

仿真配置批的8个initial来源在候选向量编码阶段收到HTTP500，exit1，没有发布。原trace、冻结批次及每个提交chunk范围已保留，未盲目重试或修改共享服务；原因尚未确认。它不计入14个成功来源。

固定回执还剩115个initial来源未接受：上述失败8个和其它107个；后续任务按新维护合同接续。任务计划归档、本结果及变更记录在停止后产生，属于后续发现范围；源新鲜度和最终向量计数以收尾JSON为准。反馈列表实际可读且为空，没有为测试创建人工反馈。

最终正式库为48个登记源、56份来源快照、348个成员、21节点、21关系、540条变更与0反馈；348/348活跃向量兼容，旧42份来源快照和21条关系ID均保留，AGENTS与接入说明已登记版本匹配当前文件。deep changes的118个变化候选还包含归档路径删除/新增和新变更记录，与固定回执余下115个是不同截点。

## 证据与范围

无凭据选留证据在根 `artifacts/runs/basepack-upgrade-20261112/`：`installation-and-backup.json`、`doctor.json`、`init-verification.json`、`maintenance-summary.json`、`owner-contention.json`、`owner-stopped.json`、`owner-released.json`、`owner-final.json`、`final-owner-publication-verification.json` 和 `final-verification.json`。三批冻结JSON/hash、配置批失败JSON/trace与验收脚本随结果选留；完整原件、控制及恢复必要回执保留在配置的memory目录。

主agent统一写入与集成，独立agent负责上游调研和兼容性/所有权验收，维护协调者管理受限reader并独占知识发布。团队token用量未知。前轮未提交的0.9.1文档改动保留，本次未修改仿真源码、数值配置或机器模型环境。
