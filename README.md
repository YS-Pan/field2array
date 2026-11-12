# field2array · 阵列复幅度反演

用虚拟测量环境研究近场阵列复幅度反演。[GitHub 仓库](https://github.com/YS-Pan/field2array)。现有实测数据路线已退出。

稳定轻档 K4 受限首版在20例独立确认中全部通过原三项系数基准，最坏幅度 RMSE **0.166 dB**、相位 RMSE **1.969°**、复系数 NRMSE **3.759%**。支持11.7 GHz、已知32×32错列几何、正常幅度±2.5 dB／相位σ5°，以及跨平面固定的非参考码未知误差σ0.2 dB／3°。恢复的是相对逐元标量等效辐射系数。

会话漂移、其他频率及未知几何尚未确认；故障诊断存在误报，单状态反演不继承K4的通过结论。完整条件、评分定义和逐案结果见[首版报告](devDoc/tasks/20261112-simulation-research/research/k4-first-version-results.md)。

## 安装与复现

Python 3.10及以上，从项目根执行：

```powershell
git clone https://github.com/YS-Pan/field2array.git
cd field2array
python -m pip install -e "./experiments/20261027-simulation[test]"
python -m array_reconstruct --help

$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
python -B experiments/20261027-simulation/scripts/run_k4_research.py `
  --policy experiments/20261027-simulation/configs/k4_confirmation_20261112.json `
  --generate-operators --output tmp/k4-confirmation-reproduction
```

输出目录须尚不存在。完整确认会重建算子并执行20案，不需要旧项目、实测数据、GPU或basePack。其他shell可设置同名环境变量，并将最后一条命令合并为一行。普通wheel安装也包含所需参考几何；可选GPU依赖和CPU测试命令见[路线说明](experiments/20261027-simulation/README.md)。

## 结果与证据

![独立确认的逐元真值、恢复及误差](artifacts/runs/k4-visuals-20261112/element_maps.png)

[36案指标 CSV](artifacts/runs/k4-research-20261112/metrics.csv)、[确认结果 JSON](artifacts/runs/k4-research-20261112/confirmation/results.json)、[开发结果 JSON](artifacts/runs/k4-research-20261112/development/results.json)和[47秒中文演示视频](artifacts/runs/k4-visuals-20261112/k4-explained.mp4)随仓库提供。公开小证据的大小与SHA-256见[发布清单](devDoc/archive/20261112-github-publication/published-evidence.json)；完整NPZ和历史档案仅保存在本机，清单中的全量归档路径不表示文件均随Git分发。

## 项目入口

| 入口 | 内容 |
| --- | --- |
| [项目纲领](devDoc/project-charter.md) | 用户目标、本次范围及意图原文 |
| [当前系统](devDoc/system/overview.md) | 新结构、可用能力与边界 |
| [当前研究计划（active）](devDoc/tasks/20261112-simulation-research/plan.md) | 有界开发16案、稳定轻档K4确认20/20；会话及频率扩展仍开放 |
| [仿真路线](experiments/20261027-simulation/README.md) | 迁入代码、依赖、验证及运行入口 |
| [发布检查](devDoc/archive/20261112-github-publication/result.md) | 打包修正、检查范围与首版发布状态 |
| [旧实验结论](devDoc/archive/legacy-20261027/experimental-conclusions.md) | 仿真与实测结果、FFT/角谱结论及局限 |
| [聊天历史索引](devDoc/archive/legacy-20261027/chat-history-index.md) | 人类原话、完整日志、覆盖范围与缺口 |
| [迁移记录](devDoc/archive/20261110-migration/result.md) | 来源到新结构的映射、校验及保留范围 |
| [迁移后续整改计划](devDoc/tasks/20261110-migration-followup/plan.md) | 提供者原话补迁、几何打包与恢复核验；待执行 |
| [basepack 接入](devDoc/system/basepack.md) | 独立记忆库与本机工具使用 |
| [basepack 实测问题](devDoc/archive/20261110-migration/basepack-test-findings.md) | 本次作为测试样本发现的问题、处理和待修复项 |

根目录只保留一套开发文档 `devDoc/`；路线代码和依赖在 `experiments/`。完整聊天日志、用户请求原文、历史原件和本机basePack配置/凭据/数据库不进入公开Git；`artifacts/` 仅精确放行本次已审查的小型科学证据与媒体，`tmp/` 保持忽略。聊天索引是本机档案导航，不包含聊天正文。

旧项目仍留在原位，以便查阅未迁入的大型数据与运行产物。历史运行计划和记录属于档案，后续工作从项目纲领和当前系统入口开始。
