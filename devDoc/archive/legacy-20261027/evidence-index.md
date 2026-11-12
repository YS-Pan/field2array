# 旧实验报告与证据索引

初次核查：2026-11-10；补充核查与小结果补迁：2026-11-12。先读[条件化实验结论](experimental-conclusions.md)，再按问题选择原证。唯一旧来源根为 `D:\project\trifles\20261027 array reconstruct`；保留根为 `artifacts/legacy-20261027/reports/`。当前研究如何承接这些结果，见[active研究计划](../../tasks/20261112-simulation-research/plan.md)及其两份研究表。

## 原文身份与使用边界

- 选留53份原件、**3,291,637字节**：11份顶层报告、旧README、旧AGENTS，以及40份定向证据；其中30份JSON、3份CSV、6份PNG、1份软件独审Markdown。
- [完整快照清单](../../../artifacts/legacy-20261027/reports/snapshot-manifest.json)逐文件记录旧相对路径、新相对路径、字节数、源/副本/复制后源SHA-256；[迁移检查](../../../artifacts/legacy-20261027/reports/migration-checks.json)记录53份全部字节相同、30份JSON解析通过。哈希证明复制一致，不是科学独立复验。
- 报告和证据均按原字节保留。原文中的相对链接、绝对旧路径及“当前/本轮”没有重写；未选留链接需恢复旧项目的对应路径才能使用。这是选留证据集，不是完整可重现实验包。
- [README快照](../../../artifacts/legacy-20261027/reports/README.md)和[AGENTS原文快照](../../../artifacts/legacy-20261027/reports/AGENTS.original.md)仅为历史导航/规则证据。后者改名以避免作为现行AGENTS加载；旧Goal/STOP/grant/lease及记忆状态不具有现行授权效力。
- 未复制完整outputs、NPZ、ZIP、GIF/MP4、软件二进制、旧服务或进程状态。原件仍留旧项目，本次未修改旧目录或执行科学计算。

## 报告快照

| ID | 旧相对路径及原文快照 | 选择阅读的条件 |
| --- | --- | --- |
| R01 | [docs/validation_report.md](../../../artifacts/legacy-20261027/reports/docs/validation_report.md) | 核小误差稀疏恢复、原失败、弱模态状态对及92项历史测试 |
| R02 | [docs/large_error_validation_report.md](../../../artifacts/legacy-20261027/reports/docs/large_error_validation_report.md) | 核54例/216拟合、大误差36例失败、噪声流修订及报警误报 |
| R03 | [docs/thesis_methods_experiments.md](../../../artifacts/legacy-20261027/reports/docs/thesis_methods_experiments.md) | 核区域代理、K2/K4编码及三步闭环的条件和功率代价 |
| R04 | [docs/real_measurement_report.md](../../../artifacts/legacy-20261027/reports/docs/real_measurement_report.md) | 核旧单通道0/180数据、共享延迟和已更正的V3扫描假设 |
| R05 | [docs/new_nearfield_scan_analysis.md](../../../artifacts/legacy-20261027/reports/docs/new_nearfield_scan_analysis.md) | 核14会话/28场、扫描大小、角标签及Main/Cross变化 |
| R06 | [docs/scan_inversion_report.md](../../../artifacts/legacy-20261027/reports/docs/scan_inversion_report.md) | 核联合位置姿态、梯度、实测歧义、最后FFT/ASM及11-04边界；最新科学报告 |
| R07 | [docs/pnf_analyzer_assessment_20261030.md](../../../artifacts/legacy-20261027/reports/docs/pnf_analyzer_assessment_20261030.md) | 核软件计算对象、距离单位缺陷与硬件解释限制 |
| R08 | [docs/pnf_analyzer_usage_20261030.md](../../../artifacts/legacy-20261027/reports/docs/pnf_analyzer_usage_20261030.md) | 核实际合成/Ku试用设置、截图解释和未验证页面 |
| R09 | [docs/clean_stop_20261102.md](../../../artifacts/legacy-20261027/reports/docs/clean_stop_20261102.md) | 核11-02停止截点和未完成风险覆盖；仅历史状态 |
| R10 | [docs/github_repository.md](../../../artifacts/legacy-20261027/reports/docs/github_repository.md) | 核用户历史选定的YS-Pan/field2array；不表示已建remote/提交/推送 |
| R11 | [docs/project_maintenance_20261028.md](../../../artifacts/legacy-20261027/reports/docs/project_maintenance_20261028.md) | 还原旧sources/metadata/outputs与旧记忆的保存关系 |

## 选留的小原证

以下路径均相对于快照根；完整文件列表及精确SHA-256以清单为准。每个文件在保留根保持旧相对路径，因此可由清单直接回溯旧位置。

| 主题 | 原始小证据及用途 |
| --- | --- |
| 初始虚拟恢复与弱模态 | [validation_v2/results.json](../../../artifacts/legacy-20261027/reports/outputs/validation_v2/results.json)、[sparse_holdout/results.json](../../../artifacts/legacy-20261027/reports/outputs/sparse_holdout/results.json)、[sparse_random_holdout/results.json](../../../artifacts/legacy-20261027/reports/outputs/sparse_random_holdout/results.json)保留逐案指标；大误差与论文三组的原execution/run/archive receipt保留原审计与输入哈希 |
| 完整C12负结果 | [C0_aggregate.json](../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/resume_20261030/baseline_aggregate_actual_v1/C0_aggregate.json)保留12场景、eligible11/12、几何/held聚合及原gate；538尝试、11402 profiles与逐频解释在R06的669–677行 |
| 理想定位局部阳性 | [public_information.json](../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/ideal_unknown_response_preparation_20261029/public_info_lock_actual_v1_20261029/public_information.json)、[geometry_score.json](../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/ideal_unknown_response_preparation_20261029/geometry_score_actual_v1_20261029/geometry_score.json)分别保留nuisance消去后的局部信息与锁后几何真值评分 |
| 先验/窗口/正则失配 | [development_interpretation.json](../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/resume_20261030/pose_development_v1/pose_prior_analysis_v1/three_method_v1/development_interpretation.json)、[window geometry_summary.json](../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/window_staged_development_preparation/run_20261029_v1_01/post_lock_known_geometry_description_v1/geometry_summary.json)、[tau03 result.json](../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/tau03_strong_failure_readonly_analysis_20261029/actual_fixed_cross_v1_20261029/result.json)、[case1 sensitivity report](../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/development/sensitivity_gpu_queue_20261028T145500_881543Z/case1_sensitivity_report.json)保留改善及失败两面 |
| 梯度与后端数值资格 | [V0 result](../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/v0_linear/result.json)、[finite FD37 receipt](../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/finite_gpu_FD37_qualification_20261029/actual_FD37_v1_qualification_receipt.json)、[pose numerical closure](../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/resume_20261030/pose_development_v1/pose_numerical_path_closure_v1.json)、[pose52 qualification](../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/resume_20261030/pose_development_v1/native_closures/qualification_v1.json)保留通过的实际范围 |
| 缺块伪影 | [report_aggregates.json](../../../artifacts/legacy-20261027/reports/outputs/bounded_geometry_diagnosis_20261028/report_aggregates.json)保留移动缺块、补全、散删及合成机制对照 |
| 11-03初步真实反演 | [fit_summary.json](../../../artifacts/legacy-20261027/reports/outputs/preliminary_real_20261103_v1/fit_summary.json)、[field_metrics.csv](../../../artifacts/legacy-20261027/reports/outputs/preliminary_real_20261103_v1/field_metrics.csv)、[audit.json](../../../artifacts/legacy-20261027/reports/outputs/preliminary_real_20261103_v1/audit.json)保留两C0数值未过门、同相模板较差与27行场指标 |
| 最后扩距和FFT对照 | [distance_qualification.json](../../../artifacts/legacy-20261027/reports/outputs/fft_distance_real_20261103_v1/distance_qualification.json)、[fit_summary.json](../../../artifacts/legacy-20261027/reports/outputs/fft_distance_real_20261103_v1/fit_summary.json)、[comparison_analysis.json](../../../artifacts/legacy-20261027/reports/outputs/fft_distance_real_20261103_v1/comparison_analysis.json)、[comparison_metrics.csv](../../../artifacts/legacy-20261027/reports/outputs/fft_distance_real_20261103_v1/comparison_metrics.csv)、[fft_field_metrics.csv](../../../artifacts/legacy-20261027/reports/outputs/fft_distance_real_20261103_v1/fft_field_metrics.csv)、[audit_comparison_v2.json](../../../artifacts/legacy-20261027/reports/outputs/fft_distance_real_20261103_v1/audit_comparison_v2.json)保留资格、两端点、距离/方法差异和120行FFT指标；三张原幅度/相位/系数PNG辅助审读 |
| 11-04边界解释 | [statistics.json](../../../artifacts/legacy-20261027/reports/outputs/fft_boundary_explanation_20261104_v1/statistics.json)、[audit.json](../../../artifacts/legacy-20261027/reports/outputs/fft_boundary_explanation_20261104_v1/audit.json)、[amplitude_distance_difference.png](../../../artifacts/legacy-20261027/reports/outputs/fft_boundary_explanation_20261104_v1/amplitude_distance_difference.png)保留保存场的0→100 mm幅度变化、边缘场强与72项原审计 |
| 最终宽幅FFT显示 | [config.json](../../../artifacts/legacy-20261027/reports/outputs/fft_fullscan_animation_20261104_v4_wide/config.json)、[summary.json](../../../artifacts/legacy-20261027/reports/outputs/fft_fullscan_animation_20261104_v4_wide/summary.json)、[audit.json](../../../artifacts/legacy-20261027/reports/outputs/fft_fullscan_animation_20261104_v4_wide/audit.json)、[depth_100mm.png](../../../artifacts/legacy-20261027/reports/outputs/fft_fullscan_animation_20261104_v4_wide/depth_100mm.png)、[input_zero_padding.png](../../../artifacts/legacy-20261027/reports/outputs/fft_fullscan_animation_20261104_v4_wide/input_zero_padding.png)保留798 mm显示、399 mm红框、零填充输入、传播与参考合同；不复制121帧媒体/复数组 |
| 软件距离单位 | [unit_blackbox_independent_review.md](../../../artifacts/legacy-20261027/reports/outputs/pnf_analyzer_review_20261030/comparison/unit_blackbox_independent_review.md)保留该版本/口径页/GHz-mm合同的独立静态与黑箱解释；原软件和截图仍在旧项目 |

## 2026-11-12复核及补迁的小结果

本次重新读取原53件清单，逐件核现存旧源、迁入副本与原manifest SHA-256：53/53一致；旧 `docs/` 顶层11份报告/记录均在，合计357,519字节，无遗漏。详见[本次迁移复核](../../../artifacts/runs/plan-history-reconciliation-20261112/report-migration-check.json)。这说明报告已迁入，不能把当前计划承接不足解释成报告丢失；哈希检查仍不等于重跑科学实验。

为直接读取历史逐案指标与已知编码/联合模型边界，补迁下列8份小原件，共1,229,658字节，源/副本/复制后源哈希一致，见[独立补迁清单](../../../artifacts/runs/plan-history-reconciliation-20261112/supplement-manifest.json)。保留旧相对路径，原53件清单及原报告字节不改写；两批共61件原件、4,521,295字节，清单/校验元数据和本次脚本不计入原件数。大型NPZ、完整outputs及媒体仍不迁移。

| 补迁原件 | 用途 |
| --- | --- |
| [编码group_summary.json](../../../artifacts/legacy-20261027/reports/outputs/thesis_methods_20261028/coded/group_summary.json) | K1/K2/K4、重复扫描与两档未知状态误差的聚合比较 |
| [编码results.json](../../../artifacts/legacy-20261027/reports/outputs/thesis_methods_20261028/coded/results.json) | 6确认案例、54拟合的逐案系数和检测结果 |
| [闭环group_summary.json](../../../artifacts/legacy-20261027/reports/outputs/thesis_methods_20261028/calibration/group_summary.json) | 一致性、恢复精度、绝对功率代理和控制代价 |
| [区域确认results.json](../../../artifacts/legacy-20261027/reports/outputs/thesis_methods_20261028/aperture/confirmatory/results.json) | 五方法、30次确认拟合及原评分 |
| [修正噪声流后的大离散metrics.csv](../../../artifacts/legacy-20261027/reports/outputs/large_error_validation_20261028_independent_noise/metrics.csv) | 54案例、216方法拟合的逐案指标 |
| [大离散group_summary.json](../../../artifacts/legacy-20261027/reports/outputs/large_error_validation_20261028_independent_noise/group_summary.json) | 正常散布、故障状态、频率和方法的聚合比较 |
| [T2_finite_qsplit_risk_closeout.ROOT.v1.json](../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/resume_20261030/pose_development_v1/qsplit_risk_v1/T2_finite_qsplit_risk_closeout.ROOT.v1.json) | fixed6条件性阳性、C0系数失败及相关开发视图限制 |
| [B2_information_scalars.json](../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/resume_20261030/pose_development_v1/next_validation_preparation_v1/B2_information_preparation_v1/metadata_scalar_extraction_r5_v1/pair_INFO_PAIR_B1_INTERIOR_v1/B2_information_scalars.json) | data-only与prior-assisted局部信息对比，防止把先验改善误记成新增观测信息 |

## 阅读覆盖与未验证事项（原迁移记录）

已全文审读旧README、11份顶层docs报告与 `docs/memory/` 全部27份Markdown（合计38份docs Markdown）。大扫描报告按行段并行审读，覆盖1–1283行；memory及其上游快照没有发现超出业务报告的新阵列科学结论。它们把科学判断回链到业务报告，索引/发布成功不是科学验证。上游blueprint/history/research-plan是记忆系统背景，不是阵列实验。

旧记忆导航存在时态漂移，例如memory/README仍称较早release为“当前”，task-card已记更晚代；本次未验证旧实际CURRENT。人类原话和聊天的保存、覆盖与缺口另见[聊天历史索引](chat-history-index.md)，不从旧导航推断现行用户授权。

本次复制前、复制后均对所选旧原件做SHA-256，目标逐项一致；30份所选原始JSON均可解析。人工查看了旧ASM幅度比较图和最终100 mm宽幅图，确认有粗尺度纹理且图中已标连续场/实测红框；这只是图义核查。没有重跑算法、重算科学指标、校准供应商软件或验证完整旧输出链。R06里的旧独审结论仍是当时报告，不能写成本次独立复验。
