# basePack 0.11.0.dev1 原位升级结果

2026-11-12，Asia/Shanghai。用户要求核对新版并在适用时升级本项目。软件安装、容量配置、独立兼容验收和固定名称检索闭环已完成；自动维护的实际完成范围在收尾记录。本次不修改科学算法，不运行科学实验，不修改basePack源项目或共享设施。前轮研究计划及未提交文档保留。

## 升级依据与实际改动

本机basePack上游HEAD为 `3012a91`。选用active路线最终 `dist/dispatch-0.11/basepack-0.11.0.dev1-py3-none-any.whl`，SHA256 `5c97f8641de6154fa881387539d42764fa27a1a3c3a782a78d5f0a80d4325a91`；独立检查52个包文件与源码一致，安装后52个文件也与保存wheel一致。原0.10的schema、store、向量、维护所有权/反馈及共享资源核心与新版相同，新改动集中于配置、CLI、派工和随包提示词。

本项目 `.basepack/client-0.10/` 目录沿用，实际包由0.10.0.dev1原位升级为0.11.0.dev1；pip使用本地保存wheel、no-index/no-deps，原psutil7.2.2与websocket-client1.9.2保持，pip check通过。未升级其它项目的解释器或本机共享embedding服务。两项已装独立技能完整目录与上游一致，没有增加非必要技能。

新版主要联动：业务预计不超过十分钟时跳过整个自动maintain；长任务明确automatic，命令默认manual；脚本按容量固定队伍、名称和模型，第0队唯一发布，零reader时协调者亲读。检索使用独立retrieval-dispatch/retrieval-prompt，零可用席位由主agent执行同一合同，不依赖维护owner。配置维护责任见新版生成入口的条件链接。

## 容量与入口

实际读取用户级 `C:/Users/ysPan/.codex/config.toml` 的 `agents.max_concurrent_threads_per_session=12`，本项目没有 `.codex/config.toml` 覆盖；宿主亦提供含主agent共13席。以不含主agent、包含全部后代的口径记录C=12，执行新版config set-concurrency只新增 `ai.max_concurrent_subagents` / `ai.concurrency_source`。

随后memory init刷新AGENTS的basepack块和instructions的memory块，解释器路径保留。独立验收确认去除新增配置行后其余配置原字节相同、入口块外业务正文原字节相同。更新[本机接入](../../system/basepack.md)中的实际版本、时长判断、固定派工及旧回执边界。

0.10旧回执不含新版layout，未交给新版prompt续派；保留其原证并用新control启动0.11轮次。没有靠更换客户端覆盖旧身份、当前研究计划或历史授权边界。

## 升级前恢复集合

`.basepack/memory/backups/upgrade-0.11.0.dev1-20261112_124037/` 保存：原生 `memory.surql`、八表 `records.before.json`、137份当前原件/入口/配置、原凭据和machine配置、受管codex/assets/维护回执、完整选留artifacts及新旧wheel。原生export为19,726,021字节，SHA256 `fa05236c43ac18459404209b2b0abc6440764ee3c993859a59c1351ef12b1425`。

3,121份artifacts共1,133,362,725字节逐文件核源/副本哈希，保留研究计划v2、旧报告和补迁小结果。升级前后记录快照在复制/export期间一致，owner为空，无并行维护写入。凭据与原始数据只在被Git忽略的位置保存。恢复需取齐本项目UUID、原件、受管材料及原凭据，不能只取Git或单一数据库文件。

## 独立兼容验收

在任何新版维护写入前完成：

- 八表全部ID和字段（含向量）相等：meta16、source56、source_version71、member399、node23、semantic21、change655、feedback3；初始化没有增加或改写这些记录。
- 节点/语义关系的90项保存basis按原source_version与行范围回读一致，旧来源快照和关系证据保留。
- 396/396活跃向量兼容，凭据与launcher原字节不变；配置只增加两键，AGENTS/instructions托管块外原字节不变。
- 其它cwd的新无profile PowerShell实际brief、默认vector、hybrid literal、read、neighbors、AI映射和检索派工/渲染通过；查询前后八表全字段不变、owner仍空。
- 本项目配置下manual/automatic及0/12容量布局、pending派工缩减纯函数检查通过。它们不冒充满队真实维护。

验收脚本最初误把brief.source_count当作活跃源数，实际合同包含1个deleted源、合计56；修正后完整验收通过，未发现产品或数据差异。此次未重做全量隔离恢复、图片语义重建、共享环境回滚或长期性能试验；没有默认语义审计。

## 新派工实际运行与收尾

独立检索按脚本指定 `bp_mem_search_0` / tier1 / gpt-6-luna / max、干净上下文真实派发；完成search、原文read、neighbors、评分和rank。两条证据入选，1693/1800字符、无预算遗漏，向量覆盖396/396。首次将描述性scope当路径前缀未命中，worker在本项目内去掉该前缀重搜并回读成功，不作为产品错误。worker评分文件已选留；rank原输出仅在stdout，主控只用同一scored重跑纯rank保存证据，没有重做检索或评分。

自动维护在备份/升级及只读兼容验证后才开始，避免改写升级比较基线。control为 `.basepack/memory/maintenance/upgrade-011-automatic-20261112-01a1cdb6.control`，回执为 `.basepack/memory/maintenance/d610b34a88e945ea88427e0646d1a8a4.json`。固定布局C=12、实际A=11、业务预留1，仅第0队1协调者+4reader；根bp_mem_coord_0与其固定子名称均已实际派发。

预检发现171个来源，完成7个机械批次、复用23块；固定回执146个语义候选、0待处理关系、2反馈。采集10个归属本项目的聊天，保留58个来源、40个采集边界；336个全宿主归属未明条目不计作本项目缺失聊天。

实际成功发布6个语义批次：14来源（11 initial、3 existing-change）、76次成员写入、2节点、1关系。包括研究计划变更记录、两份旧结果承接表、AGENTS、历史证据索引、现行研究计划与仿真说明及6份受条件约束的历史聊天。固定名称团队的派发与原子发布已在本项目实际工作，不将这些有界批次当成全项目完成。

纲领批次被程序以 `ValueError: Evidence source revision mismatch for devDoc/archive/20261110-migration/basepack-test-findings.md` 拒绝，无部分发布；作者的原草稿、原错与刷新读取保留，后续必须按当前版本重新prepare，不能直接提交旧草稿。旧mandatory修正尚未发布，README/overview、basepack.md及部分问题记录/聊天仍待接受。回执余下126 initial、6 existing-change及2反馈保持待处理；BP-T28“缺乏遗忘机制”继续按用户要求搁置，升级本身没有修复该产品问题。

业务结束已stop，结束自有协调者与全部reader，确认publications为空后release，最终owner=null，无未明发布。收尾原生导出与当前原件/受管材料覆盖层保存在同一恢复集合，取齐方式见 `after-recovery.json`；最终表计数、向量覆盖和当前入口接受状态见选留 `final-verification.json`。升级前逐字段兼容验收与后续正常语义更新属于不同阶段，后者改变知识不算升级数据丢失。

最终正式库67个登记源、89份来源快照、464个成员、25节点、21关系、757变更及3反馈；461/461活跃向量兼容，原71份来源快照和3反馈记录均保留。收尾export为22,955,477字节，SHA256与当前原件覆盖层清单保存在恢复集合。source版本登记与成员语义接受是不同状态；即使纲领的source登记版本匹配当前文件，旧mandatory成员修正仍未发布，不能据此宣布同步完成。

## 证据

无秘密证据在根 `artifacts/runs/basepack-upgrade-011-20261112/`：`installation.json`、独立 `compatibility-acceptance.json`、`business-preservation.json`、检索scored/ranked、维护启动/收尾回执、`document-check.json` 和 `final-verification.json`等。与升级前快照相比，132份已存在的业务文件原字节保持，含当前研究计划v2；只更新本次授权的配置/入口和basePack接入说明。完整数据快照、凭据、备份清单、恢复必要回执留在 `.basepack/`；运行中请求、派工和检查脚本在根tmp，必要原证已选留。

五份入口/升级文档的32个本地链接检查通过，git diff --check通过。首次用内联命令执行文档链接检查遇到shell引号解析错误，改为一次性Python脚本后通过，未改动业务内容。最终报告及检查结果在本机恢复覆盖层中同步保留；四个外部链接沿用产品入口，未作网络可达性验收。
