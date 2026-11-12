# 当前系统

更新：2026-11-12。当前唯一研究方向是虚拟环境中的阵列复幅度反演。[当前研究计划](../tasks/20261112-simulation-research/plan.md)关注较大且遍布全阵的正常幅相离散；本轮16次机制开发完成，稳定轻档K4受限首版20例新独立确认全部通过原系数基准。[首版结果](../tasks/20261112-simulation-research/research/k4-first-version-results.md)限定11.7 GHz和固定跨平面的状态误差，保留会话漂移与频率扩展开放。迁入[仿真路线](../../experiments/20261027-simulation/README.md)的原数值算法保持不变，新增当前策略化执行入口与证据；未将旧运行重计为新确认。

## 维护位置

| 路径 | 职责 |
| --- | --- |
| `experiments/20261027-simulation/` | 迁入的仿真代码、依赖、配置与维护测试 |
| `devDoc/project-charter.md` | 用户明确表达的目标与本次范围 |
| `devDoc/system/` | 当前实现与本机环境使用说明 |
| `devDoc/archive/legacy-20261027/` | 旧实验结论、聊天索引与原证导航 |
| `devDoc/archive/20261110-migration/` | 本次迁移依据、映射和实际验收 |
| `devDoc/change_log/major/` | 重要变更记录 |
| `artifacts/legacy-20261027/` | 选留报告、完整聊天原件与来源清单，本机归档 |
| `.basepack/` | 本项目独立配置、知识库和运行材料 |
| `skills/` | 从新版 basepack 取得的通用目录与协作技能 |
| `tmp/` | 一次性脚本和校验结果，可在任务后清理 |

项目使用一套根 `devDoc/`。未来独立开发任务放在 `devDoc/tasks/<日期-主题>/`，完成后将结论对齐到系统说明，再将任务材料移入 `archive/`。路线内 README 只维护代码使用入口。

## 记忆与历史

新版 basepack 使用本项目独立知识库；入口、配置和已执行验证见[basepack 接入](basepack.md)。原始证据与当前指令分开保存，历史指令按当时上下文解释。旧项目 `.memory` 程序、SQLite 库、模型、缓存和调度服务没有作为新项目运行依赖迁入。

本次迁移同时作为 basepack 的真实项目测试样本，实际故障、环境限制及后续建议集中在[实测问题记录](../archive/20261110-migration/basepack-test-findings.md)。本次负责记录与验证处理结果；basepack 源项目的修复留待后续任务。

旧大型实测数据和全量输出留在原项目路径。结论是否成立，应通过[结论与依据](../archive/legacy-20261027/experimental-conclusions.md)中的适用条件和源索引核实。

## 当前研究与后续边界

2026-11-12按用户请求建立、以旧实验结果修订并执行当前active研究计划。旧报告复核和小结果补迁已完成；本轮K4有界机制开发及轻档稳定条件的独立确认已完成，受限首版通过原三门，最坏0.166144 dB／1.969274°／3.759260%。故障诊断存在正常通道误报，单列报告；会话漂移、精确容限和其他频率未确认。新增入口及102项相关检查见仿真说明，本轮证据保留在 `artifacts/runs/k4-research-20261112/`；未复制大型旧缓存，入口支持重建算子。历史上的停止标记、时钟、预算、grant、seal 和旧 Goal 状态只用于理解原实验，不成为新项目的执行流程。随后用户要求调查并提交GitHub；普通wheel几何、安装回执和公开证据已整理，实际检查与远端状态见[发布记录](../archive/20261112-github-publication/result.md)。

记忆系统“缺乏遗忘机制”记录于[BP-T28](../archive/20261110-migration/basepack-test-findings.md)，已按用户要求搁置，由未来basePack项目任务处理，不阻塞科学主线。
