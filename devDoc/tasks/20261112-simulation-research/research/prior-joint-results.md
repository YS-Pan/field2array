# 旧联合反演证据对当前仿真研究的约束

核查：2026-11-12，Asia/Shanghai。用途：修订[当前计划](../plan.md)，不启动科学实验。已分段通读迁入的 [R06 扫描反演报告][R06] **1–1283行**，并核[条件化结论](../../../archive/legacy-20261027/experimental-conclusions.md)、[证据索引](../../../archive/legacy-20261027/evidence-index.md)及下列小JSON。报告中的旧目标、授权、锁、时钟和停机状态仅按当时范围解释。

**接续判断：数值实现、局部信息量、先验偏差和低残差歧义已有大量实质答案，当前研究不应重新从通用诊断起步。** 但旧联合研究主要回答未知位置/姿态下的条件几何恢复，并非当前“已知几何、遍布全阵较大正常幅相离散”的完整系数验收。应复用旧结论，只补当前观测合同、候选或迁入实现实际改变后留下的缺口。

## 已做的仿真、数值与信息诊断

| 已执行的问题与条件 | 实际结果及保留边界 | 对当前计划的约束 | 直接证据 |
| --- | --- | --- | --- |
| 线性变量投影与稳定求解；完整1024列 | V0增广QR与受控normal解：系数相对差`1.41e-13`、J差`2.19e-14`、驻点`1.09e-16`；只覆盖所测端点和设置。小型已知b阳性不等于未知b可恢复 | 未改相应数值路径时，不重做整套“能否正确求解”研究；新合同可做一次有针对性的迁入接合核对，发现差异再追查 | R06 924–944行；[V0原JSON][V0] |
| 独立有限孔径CPU/GPU与完整FD；4 mm源/4 mm探头、GL3/81平均核 | FD37为37点、9轴、两步长，最坏J列约`1.15e-8`、H差约`6.80e-9`。其后新warm/cold轨迹均过原数值门，相对点模型最大坐标变化`0.04538093 mm`，仅固定λ条件稳定，不是真值误差；不覆盖6/4 mm或所有λ | 独立真值和有限孔径不是未做过的设想。只有当前核/参数域改变时补相应资格；不以旧有限核一致性替代当前稠密系数恢复 | R06 747–758、816–818、853行；[FD37回执][FD37] |
| 姿态12外层参数的导数与后端完整路径 | 原52固定点资格覆盖零/非零姿态r/J/H和12列双步长FD，最坏H列约`2.15e-7<1e-5`；同START CPU/GPU各8 profiles/4H，重放全部CPU实际点/H通过。后续q=1/1、0/2的C0及正μ静态/条件路径也完成T1；这些只是数值资格 | 当前先固定几何，可搁置姿态/qsplit工程大矩阵。未来确需解冻几何时复用原资格范围，检查新增路径；不把CPU/GPU一致当科学恢复成功 | R06 291–332、368–495、638–650行；[姿态52][POSE52]、[代表路径闭合][POSEPATH] |
| 匹配模型、无噪声、未知三频3×1024复响应及会话复增益的定位 | 24起点、333 profiles，公开cost锁主解后最大坐标真值误差`4.49793831e-5 mm`；消去6150实nuisance后data-only局部rank3、condition=`1.88495`。只评分geometry，**未读取权重真值、未做held评分**；W不是噪声精度，无置信区间 | “未知响应是否一概不能定位”已有局部阳性；不能据此声称1024系数恢复、噪声鲁棒或全局唯一。当前无需为这个旧问题再铺24起点 | R06 794–808行；[局部信息][IDEALINFO]、[锁后几何评分][IDEALSCORE] |
| 同一3 mm观测的15/9/6 mm训练掩码对照；全部1024列 | 两开发例固定λ时，最坏未见角clean场误差由`8.875%/5.225%`降至`3.389%/1.557%`。更密训练及内部选λ后，12.7 GHz主频逐元可过门，单法向副频仍失败；各掩码W随训练范数重算，不是只改间距的纯信息实验；开发例并非全独立 | 不把曾用729<1024点的粗采样失败归为原始3 mm数据必然不足，也不把主频多状态成绩迁作每频单状态成功。当前8.35 mm、单状态合同要明确与这些条件的差别 | R06 930–934、948–970行；该节原密度摘要未在原53件快照中，结论据完整报告 |
| 正则、会话gain和训练判据的混淆 | case1原λ×10仍有内部+15°形状误差`3.852282%<5%`，却相对原解漂移`1.9346318 mm`；主频b范数缩至`.445`、gain约×2.23，而固定参考会话训练误差`55.87%`。后来固定保存g逐会话训练门排除了此端点；旧失败仍保留 | 不再只用可重新拟合公共复标量的场形状筛选。记录固定增益训练残差、公共g和系数；新增训练门也不是精度证明，窗口反例仍成立 | R06 974–980、988–1009行；[原敏感性JSON][SENS] |
| 窗口失配是否能被多起点/训练门发现；101窗x错位1.5 mm | 76尝试、1681 profiles。公开主最大坐标误差`.729608 mm`，主24跨度仅`5.93e-6 mm`；λ/10的24个admissible替代都为`1.015368–1.015370 mm`误差。λ×10真值更近却被训练门排除，不能按真值改选 | 重复收敛和低残差不能排除共同偏差；“admissible”只表示观测规则合格。几何/窗口压力可留作边界反例，不必先完整重跑为主线前置 | R06 832–849行；[76端点几何摘要][WINDOW] |
| 非线性幅度先验内层是否可靠 | τRMS=.03在部分设置通过完整FD，却未消除约`1.932397 mm`正则漂移；τRMS=.3的完整重解FD相对差`679.263`失败。固定交叉表在原中心找到成本低`2.59383e-6`的已知可行b，原三初值漏解。局部驻点和正定Hessian不能保证选支完整 | 不把非凸幅度先验列为已合格默认候选，也不为完成旧路线而全面修复它。只有明确选择这条候选且预期收益足够时，才先解决分支漏解/适用域；不能靠加起点或小步FD的局部通过宣布全域正确 | R06 764–778、920、940、982–1001行；[固定交叉原JSON][TAUCROSS] |

## 恢复结果、先验收益与未完成范围

| 历史范围 | 成功、失败与unknown的实际分母 | 当前处理及条件 | 直接证据 |
| --- | --- | --- | --- |
| C12连续共享几何确认 | 6独立parent×40/30 dB，共12相关视图；538尝试、11402 profiles、12公开锁。主几何轴最大`.144728/.039550/.079626 mm`，未见角场最坏`3.46318%`；eligible仅11/12，原完整gate=false。三频系数联合通过**0/12**，12.7 GHz为11/12，两个副频24视图均未过幅度/复误差门 | 这是已完成的负基线，不重开同群体确认；几何/场好不替代系数恢复。原882群体已因开发反馈退役；新确认必须另用未见parent | R06 667–679行；[原聚合][C12]（逐频系数分母据R06，不在该聚合字段内） |
| 固定旧端点的复模板μ诊断 | 1 parent×两SNR×3μ共6行。40 dB公开选μ=.0001，副频仍`.6786/.7038 dB`、`.1075/.1104`失败；固定μ=.001则两SNR全部六频视图通过。只有固定旧几何/g的线性开发 | 场预测选择规则与系数目标不总一致。模板收益不是当前稠密离散已解决；不依据看过的系数真值补选μ，再把该群体当确认 | R06 675–677行 |
| 两相关非零姿态A/B：C0、联合模板、fixed6 staged | C0四端点三频系数全失败；baseline四端点和staged两端点过预定描述门。B的point P90由`.463247`降至`.113809 mm`、最坏相位由`4.932666°`降至`2.148805°`；A的point P90由`.063490`升至`.098951 mm`。staged物理6固定于公开C0锚，其位置与C0完全相同 | 保留“模板—姿态补偿”的机制和条件阳性；staged不是第二轮姿态恢复、full12联合KKT或独立确认。A匹配模板、B仅8°RMS错模板，两view共享退役parent与噪声方向，不能外推至任意较大稠密幅相离散 | R06 499–569、606–634行；[完整解释与原分母][STAGED] |
| T2有限qsplit风险开发 | 两相关qtruth视图、4 cell，8个C0三频系数全失败；4个fixed6终点（2 matched、2 wrong_main）系数/位置/位移/held原门均过，公开拒绝0，有限risk=true；仍无独立parent、原C12=false | 只保留有限错核稳健性线索；未选择或识别真实q，未建立3×3 selector。当前已知几何主线无需恢复旧qsplit任务；若未来引入pattern误差，另按新候选做最小压力 | R06 209–276行；[T2正式小摘要][T2] |
| B1与B2局部pose6信息 | B1为73/75公开端点，完整barrier和私有恢复评分未执行。B2 INTERIOR只有C0/fixed6一对，data-only均rank6（rtol=`1e-10`），per-mm/per-degree condition=`153.0853/153.0606`；prior-assisted=`102.4761/32.1670`。为已拟合b/g处局部切线投影，非完整profile Hessian，W非noise precision | 先验改善条件数没有新增数据；rank6不证明系数1024唯一或统计精度。无需再泛泛“检查局部秩”；若当前候选声称新增信息，应在当前系数任务下比较data-only观测谱与噪声，而非拿先验增广谱作证 | R06 183–207行；[B2原标量][B2] |
| RISK6停止截点 | 原150槽：49真实优化终点+1实际withheld/non-producer槽+100未启动unknown；2/6公共锁。A/SNR30的24 C0及fixed6通过公开观测门；A/SNR20的24 C0训练/内部约`.09735/.10176`超`.05`，fixed6 withheld。完整屏障与本批私有系数/held/错先验评分均未执行 | 未执行不改为成功或科学失败；不为补旧槽位而重启大矩阵。只把噪声、错误先验、弱模态与state-transfer这些科学风险挑成当前合同真正需要的有限新实验 | R06 155–175行；本分区未补迁其大量运行回执，截点据完整R06 |

## 实测与FFT只作约束

| 已有事实 | 对纯仿真计划的约束 | 证据 |
| --- | --- | --- |
| 连续缺块会产生可移动伪影；固定惩罚下原四孔区域误差`88.25%`，补完整9 mm格`3.70%`，移孔后新孔`104.26%`，同数散删`4.00%`。同核无噪声合成也重现，α=0时A/B数值rank1024且权重真值误差约`7.9e-11–2.3e-10` | 观测布局与正则会共同塑造系数图；满秩仍不保证有噪声稳定恢复。独立空间测试不能只选易插值散点，缺块压力另标其合同；不把补回训练数据后的低残差当独立成绩或故障证据 | R06 1039–1083行；[实测/合成聚合][MASK] |
| 最后扩距C0 test三频约`3.597%/4.708%/5.471%`，FFT同面插值约`4.563%/5.318%/5.994%`，特定ASM点代理约`12.480%/14.636%/15.801%`；旧→新C0系数形状差`86.21%–91.06%`。均同扫描相关空间留出，无端口/位姿真值。FFT A回传再正传因子抵消，本身无d信息 | 保留“场预测不认证系数”的实测反例，不把退出实测解释为普遍不可能。不将FFT回传场直接当馈电系数，也不将此特定代理较差推广为所有FFT方法失败；当前不恢复实测路线 | R06 79–123、1171–1248行；[FFT比较小原证][FFT]、[85 mm局部数值资格][DIST] |
| 全点ASM用扫描外零输入、402²周期域、传播圆、倏逝置零、无探头补偿；0–120 mm动画全用已见观测。0→100 mm幅度图相对差`20.86%/28.35%/30.35%`，仍有非零边缘场；只扩大显示不增加测量 | 看起来平滑或相似不能判距离/端口正确。当前若用FFT作机制基线，要明确输出对象、有限扫描截断和支持条件；无需重做旧动画、软件或单位研究作为科学前置 | R06 1–61行；[边界统计][BOUNDARY] |

## 直接用于修订当前计划的接续

1. **把“从头失败诊断”改为“按已知结论补缺”。** 将大离散负基线、弱模态难辨、数值资格、先验偏差及条件编码收益分别入账。旧结果可作为立项依据，不能计为本轮独立确认完成；也不因“当前完成0”把历史证据变成待从头重跑。
2. **保持第一轮几何已知，先回答可实现的系数合同。** 当前没有证据要求先重开位置/姿态、qsplit、非凸幅度先验或旧RISK6。只在明确候选依赖这些未知量时进入相应分支；不用机械参数阳性替代稠密系数恢复。
3. **若仍保留数值/谱诊断，必须点名尚缺的问题。** 例如：迁入实现与当前策略是否接合一致；当前8.35 mm单状态合同相对旧条件改变了哪些弱方向；候选是否在噪声以下的方向强行注入模板；新增编码/平面在相同成本下改善了多少data-only信息。未改算子不重复全套FD/CPU-GPU资格，未新增问题不铺完整无噪声/SNR/起点网格。
4. **候选和选参都要与真实离散目标匹配。** 均匀同相、稀疏或接近已知模板成功只能覆盖各自先验；场误差选参可能漏掉逐元失败。保留同一公共复规范下逐元幅/相/复误差，以及固定gain训练与独立场预测；只在冻结后的新parent判主门，不以平滑图、局部rank或多起点spread代替。
5. **保留有限风险，取消旧任务债务。** 弱μ=.0003、完整Icompact/F4/R12、未启动RISK6槽和全新独立确认从未因准备或工程闭合变成通过；它们也不是必须逐项补完的现行任务。新方案只为当前候选选择能改变决策的风险，先有限开发，再独立确认；实测原点、探头和硬件加载未知保持归档约束。

本分区只新增本证据表，未改主计划、源码或旧原件；未运行科学计算。小JSON仅解析/读取已有值，未重算恢复指标；本次不声称独立复验旧结果。

[R06]: ../../../../artifacts/legacy-20261027/reports/docs/scan_inversion_report.md
[V0]: ../../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/v0_linear/result.json
[FD37]: ../../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/finite_gpu_FD37_qualification_20261029/actual_FD37_v1_qualification_receipt.json
[POSE52]: ../../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/resume_20261030/pose_development_v1/native_closures/qualification_v1.json
[POSEPATH]: ../../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/resume_20261030/pose_development_v1/pose_numerical_path_closure_v1.json
[IDEALINFO]: ../../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/ideal_unknown_response_preparation_20261029/public_info_lock_actual_v1_20261029/public_information.json
[IDEALSCORE]: ../../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/ideal_unknown_response_preparation_20261029/geometry_score_actual_v1_20261029/geometry_score.json
[SENS]: ../../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/development/sensitivity_gpu_queue_20261028T145500_881543Z/case1_sensitivity_report.json
[WINDOW]: ../../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/window_staged_development_preparation/run_20261029_v1_01/post_lock_known_geometry_description_v1/geometry_summary.json
[TAUCROSS]: ../../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/tau03_strong_failure_readonly_analysis_20261029/actual_fixed_cross_v1_20261029/result.json
[C12]: ../../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/resume_20261030/baseline_aggregate_actual_v1/C0_aggregate.json
[STAGED]: ../../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/resume_20261030/pose_development_v1/pose_prior_analysis_v1/three_method_v1/development_interpretation.json
[T2]: ../../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/resume_20261030/pose_development_v1/qsplit_risk_v1/T2_finite_qsplit_risk_closeout.ROOT.v1.json
[B2]: ../../../../artifacts/legacy-20261027/reports/outputs/joint_geometry_execution_20261028/resume_20261030/pose_development_v1/next_validation_preparation_v1/B2_information_preparation_v1/metadata_scalar_extraction_r5_v1/pair_INFO_PAIR_B1_INTERIOR_v1/B2_information_scalars.json
[MASK]: ../../../../artifacts/legacy-20261027/reports/outputs/bounded_geometry_diagnosis_20261028/report_aggregates.json
[FFT]: ../../../../artifacts/legacy-20261027/reports/outputs/fft_distance_real_20261103_v1/comparison_analysis.json
[DIST]: ../../../../artifacts/legacy-20261027/reports/outputs/fft_distance_real_20261103_v1/distance_qualification.json
[BOUNDARY]: ../../../../artifacts/legacy-20261027/reports/outputs/fft_boundary_explanation_20261104_v1/statistics.json
